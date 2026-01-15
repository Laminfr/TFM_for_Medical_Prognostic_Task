"""
TabICL Classifier Wrapper for Survival Stacking

This module provides a sklearn-compatible wrapper around TabICL for direct
binary classification in the survival stacking framework.

Key insight: Instead of using TabICL for embedding extraction and then 
training XGBoost on those embeddings, we use TabICL directly as the 
binary classifier. This leverages TabICL's in-context learning capabilities
for the actual prediction task.

Usage:
    from survivalStacking.stacking_model import SurvivalStackingModel
    
    model = SurvivalStackingModel(
        classifier='tabicl',
        classifier_params={'n_estimators': 4, 'device': 'cuda'}
    )
    model.fit(X, T, E)
"""

import numpy as np
import pandas as pd
from typing import Optional, Dict, Any
from sklearn.base import BaseEstimator, ClassifierMixin
import warnings

try:
    from tabicl import TabICLClassifier
    TABICL_AVAILABLE = True
except ImportError:
    TABICL_AVAILABLE = False
    warnings.warn("tabicl package not found. Install with: pip install tabicl")


class TabICLBinaryClassifier(BaseEstimator, ClassifierMixin):
    """
    Sklearn-compatible wrapper for TabICL binary classification.
    
    This wrapper adapts TabICL for use in the survival stacking framework
    where we need a binary classifier that predicts P(event in interval | at risk).
    
    Parameters
    ----------
    n_estimators : int, default=4
        Number of TabICL ensemble members (default is 4 per TabICL paper)
    device : str, default='cuda'
        Device to run inference on ('cuda' or 'cpu')
    max_context_samples : int, default=1000
        Maximum number of context samples to use (TabICL memory constraint)
    random_state : int, default=42
        Random seed for reproducibility
    verbose : bool, default=False
        Print progress information
        
    Attributes
    ----------
    clf_ : TabICLClassifier
        Fitted TabICL classifier
    classes_ : np.ndarray
        Class labels [0, 1]
    X_context_ : pd.DataFrame
        Context samples for in-context learning
    y_context_ : np.ndarray
        Context labels
    """
    
    def __init__(
        self,
        n_estimators: int = 4,
        device: str = 'cuda',
        max_context_samples: int = 1000,
        random_state: int = 42,
        verbose: bool = False
    ):
        self.n_estimators = n_estimators
        self.device = device
        self.max_context_samples = max_context_samples
        self.random_state = random_state
        self.verbose = verbose
        
        # Fitted attributes
        self.clf_ = None
        self.classes_ = np.array([0, 1])
        self.X_context_ = None
        self.y_context_ = None
        self.feature_names_ = None
        
    def _check_tabicl_available(self):
        """Check if TabICL is available."""
        if not TABICL_AVAILABLE:
            raise ImportError(
                "TabICL not installed. Install with: pip install tabicl"
            )
    
    def _subsample_context(
        self, 
        X: np.ndarray, 
        y: np.ndarray
    ) -> tuple:
        """
        Subsample context to fit within TabICL's memory constraints.
        
        Uses stratified sampling to maintain class balance.
        """
        n_samples = len(y)
        
        if n_samples <= self.max_context_samples:
            return X, y
        
        # Stratified subsampling
        rng = np.random.RandomState(self.random_state)
        
        pos_idx = np.where(y == 1)[0]
        neg_idx = np.where(y == 0)[0]
        
        # Calculate how many samples from each class
        pos_ratio = len(pos_idx) / n_samples
        n_pos = max(1, int(self.max_context_samples * pos_ratio))
        n_neg = self.max_context_samples - n_pos
        
        # Sample from each class
        if len(pos_idx) > n_pos:
            pos_sampled = rng.choice(pos_idx, n_pos, replace=False)
        else:
            pos_sampled = pos_idx
            
        if len(neg_idx) > n_neg:
            neg_sampled = rng.choice(neg_idx, n_neg, replace=False)
        else:
            neg_sampled = neg_idx
        
        selected_idx = np.concatenate([pos_sampled, neg_sampled])
        rng.shuffle(selected_idx)
        
        return X[selected_idx], y[selected_idx]
    
    def fit(
        self, 
        X: np.ndarray, 
        y: np.ndarray,
        **fit_params
    ) -> 'TabICLBinaryClassifier':
        """
        Fit the TabICL classifier.
        
        Note: TabICL is an in-context learner, so "fitting" means storing
        the context data that will be used during prediction.
        
        Parameters
        ----------
        X : np.ndarray
            Training features (n_samples, n_features)
        y : np.ndarray
            Binary labels (0 or 1)
            
        Returns
        -------
        self
        """
        self._check_tabicl_available()
        
        X = np.asarray(X)
        y = np.asarray(y).flatten().astype(int)
        
        if self.verbose:
            print(f"TabICL: Fitting on {len(y)} samples, "
                  f"{y.sum()} positives ({100*y.mean():.2f}%)")
        
        # Subsample if needed for memory constraints
        X_context, y_context = self._subsample_context(X, y)
        
        if self.verbose and len(y_context) < len(y):
            print(f"TabICL: Subsampled context to {len(y_context)} samples")
        
        # Create feature names
        n_features = X.shape[1]
        self.feature_names_ = [f'f{i}' for i in range(n_features)]
        
        # Store context as DataFrame (TabICL expects this format)
        self.X_context_ = pd.DataFrame(X_context, columns=self.feature_names_)
        self.y_context_ = y_context
        
        # Initialize TabICL classifier
        self.clf_ = TabICLClassifier(
            n_estimators=self.n_estimators,
            device=self.device,
            random_state=self.random_state
        )
        
        # Fit TabICL with context data
        self.clf_.fit(self.X_context_, self.y_context_)
        
        if self.verbose:
            print(f"TabICL: Fitted with context shape {self.X_context_.shape}")
        
        return self
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities.
        
        Parameters
        ----------
        X : np.ndarray
            Features (n_samples, n_features)
            
        Returns
        -------
        proba : np.ndarray
            Class probabilities (n_samples, 2)
            Column 0: P(y=0), Column 1: P(y=1)
        """
        if self.clf_ is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        X = np.asarray(X)
        X_df = pd.DataFrame(X, columns=self.feature_names_)
        
        # Get predictions from TabICL
        proba = self.clf_.predict_proba(X_df)
        
        # Ensure correct shape (n_samples, 2)
        if proba.ndim == 1:
            # Single class probability, expand to 2 columns
            proba = np.column_stack([1 - proba, proba])
        elif proba.shape[1] == 1:
            proba = np.column_stack([1 - proba[:, 0], proba[:, 0]])
            
        return proba
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels.
        
        Parameters
        ----------
        X : np.ndarray
            Features (n_samples, n_features)
            
        Returns
        -------
        y_pred : np.ndarray
            Predicted labels (0 or 1)
        """
        proba = self.predict_proba(X)
        return (proba[:, 1] >= 0.5).astype(int)


class TabICLBatchClassifier(TabICLBinaryClassifier):
    """
    TabICL classifier with batched prediction for large datasets.
    
    The survival stacking framework expands data to person-period format,
    which can result in very large prediction sets. This class handles
    batched prediction to avoid memory issues.
    
    Parameters
    ----------
    batch_size : int, default=5000
        Number of samples to predict at once
    **kwargs
        Additional arguments passed to TabICLBinaryClassifier
    """
    
    def __init__(
        self,
        batch_size: int = 5000,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.batch_size = batch_size
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities in batches.
        
        Parameters
        ----------
        X : np.ndarray
            Features (n_samples, n_features)
            
        Returns
        -------
        proba : np.ndarray
            Class probabilities (n_samples, 2)
        """
        if self.clf_ is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        X = np.asarray(X)
        n_samples = X.shape[0]
        
        # If small enough, predict all at once
        if n_samples <= self.batch_size:
            return super().predict_proba(X)
        
        # Batched prediction
        if self.verbose:
            print(f"TabICL: Batched prediction on {n_samples} samples "
                  f"(batch_size={self.batch_size})")
        
        probas = []
        for start_idx in range(0, n_samples, self.batch_size):
            end_idx = min(start_idx + self.batch_size, n_samples)
            X_batch = X[start_idx:end_idx]
            proba_batch = super().predict_proba(X_batch)
            probas.append(proba_batch)
        
        return np.vstack(probas)
