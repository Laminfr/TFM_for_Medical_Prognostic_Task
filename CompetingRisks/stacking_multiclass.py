"""
Phase 1: Multi-Class Survival Stacking for Competing Risks

Extends Survival Stacking to handle competing risks through multi-class
classification. Uses XGBClassifier with objective='multi:softprob'.

Key idea:
- Transform data to person-period format (like standard survival stacking)
- Instead of binary y (event/no-event), use multi-class y:
  * 0 = survived this interval
  * 1 = Event type 1 occurred
  * 2 = Event type 2 occurred
  * etc.
"""

import numpy as np
from typing import Dict, Optional, Tuple, Any
from sklearn.base import BaseEstimator
import warnings

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    warnings.warn("XGBoost not available. Install with: pip install xgboost")

from .discrete_time_multiclass import DiscreteTimeCompetingRisksTransformer


class MultiClassSurvivalStacking(BaseEstimator):
    """
    Multi-Class Survival Stacking for Competing Risks.
    
    Transforms competing risks data into person-period format and trains
    a multi-class classifier to predict interval-specific event probabilities.
    CIF curves are reconstructed from the predicted probabilities.
    
    Parameters
    ----------
    n_intervals : int, default=20
        Number of time intervals for discretization
    interval_strategy : str, default='quantile'
        How to create intervals: 'quantile' or 'uniform'
    classifier_params : dict, optional
        Parameters for XGBClassifier
    random_state : int, default=42
        Random seed for reproducibility
        
    Attributes
    ----------
    transformer_ : DiscreteTimeCompetingRisksTransformer
        Fitted time discretizer
    classifier_ : XGBClassifier
        Trained multi-class classifier
    n_risks_ : int
        Number of competing risks
    """
    
    def __init__(
        self,
        n_intervals: int = 20,
        interval_strategy: str = 'quantile',
        classifier_params: Optional[Dict[str, Any]] = None,
        random_state: int = 42
    ):
        self.n_intervals = n_intervals
        self.interval_strategy = interval_strategy
        self.classifier_params = classifier_params or {}
        self.random_state = random_state
        
        # Fitted attributes
        self.transformer_ = None
        self.classifier_ = None
        self.n_risks_ = None
        self.times_ = None
        
    def _create_classifier(self, n_classes: int) -> XGBClassifier:
        """Initialize the multi-class XGBoost classifier."""
        if not XGBOOST_AVAILABLE:
            raise ImportError("XGBoost not installed. pip install xgboost")
        
        default_params = {
            'objective': 'multi:softprob',
            'num_class': n_classes,
            'eval_metric': 'mlogloss',
            'n_estimators': 300,
            'max_depth': 5,
            'learning_rate': 0.05,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'min_child_weight': 10,
            'gamma': 0.1,
            'reg_alpha': 0.1,
            'reg_lambda': 1.0,
            'random_state': self.random_state,
            'n_jobs': -1,
            'verbosity': 0
        }
        default_params.update(self.classifier_params)
        return XGBClassifier(**default_params)
    
    def fit(
        self,
        X: np.ndarray,
        T: np.ndarray,
        E: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        T_val: Optional[np.ndarray] = None,
        E_val: Optional[np.ndarray] = None,
        verbose: bool = True
    ) -> 'MultiClassSurvivalStacking':
        """
        Fit the multi-class survival stacking model.
        
        Parameters
        ----------
        X : np.ndarray
            Training feature matrix (n_samples, n_features)
        T : np.ndarray
            Training times to event/censoring
        E : np.ndarray
            Training event indicators (0=censored, 1,2,...=event types)
        X_val, T_val, E_val : optional
            Validation data for early stopping
        verbose : bool
            Whether to print progress
            
        Returns
        -------
        self
        """
        X = np.asarray(X)
        T = np.asarray(T).flatten()
        E = np.asarray(E).flatten()
        
        self.n_risks_ = int(E.max())
        
        if verbose:
            print(f"Fitting MultiClassSurvivalStacking...")
            print(f"  Samples: {len(X)}, Features: {X.shape[1]}, Risks: {self.n_risks_}")
        
        # Fit transformer
        self.transformer_ = DiscreteTimeCompetingRisksTransformer(
            n_intervals=self.n_intervals,
            strategy=self.interval_strategy,
            include_time_features=True
        )
        self.transformer_.fit(T, E)
        self.times_ = self.transformer_.interval_midpoints_
        
        # Transform training data
        X_expanded, y_expanded = self.transformer_.transform(X, T, E)
        
        if verbose:
            print(f"  Expanded: {X_expanded.shape[0]} person-periods")
            print(f"  Class distribution: {dict(zip(*np.unique(y_expanded, return_counts=True)))}")
        
        # Create and train classifier
        n_classes = self.n_risks_ + 1  # +1 for "survived interval"
        self.classifier_ = self._create_classifier(n_classes)
        
        # Prepare validation data if provided
        fit_params = {}
        if X_val is not None and T_val is not None and E_val is not None:
            X_val_exp, y_val_exp = self.transformer_.transform(X_val, T_val, E_val)
            fit_params['eval_set'] = [(X_val_exp, y_val_exp)]
            fit_params['verbose'] = False
        
        self.classifier_.fit(X_expanded, y_expanded, **fit_params)
        
        if verbose:
            print("  Training complete.")
        
        return self
    
    def predict_cif(
        self,
        X: np.ndarray,
        times: Optional[np.ndarray] = None
    ) -> Dict[int, np.ndarray]:
        """
        Predict Cumulative Incidence Functions for each risk.
        
        Parameters
        ----------
        X : np.ndarray
            Feature matrix (n_samples, n_features)
        times : np.ndarray, optional
            Time points for prediction (default: training interval midpoints)
            
        Returns
        -------
        cif : dict
            CIF for each risk: {risk: array (n_samples, n_times)}
        """
        if self.transformer_ is None or self.classifier_ is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        X = np.asarray(X)
        n_samples = len(X)
        
        # Create prediction matrix (all patients x all intervals)
        X_pred = self.transformer_.transform_for_prediction(X)
        
        # Get multi-class probabilities
        probs = self.classifier_.predict_proba(X_pred)
        
        # Compute CIF from probabilities
        cif, survival = self.transformer_.compute_cif_from_probs(probs, n_samples)
        
        # If specific times requested, interpolate
        if times is not None:
            times = np.asarray(times)
            cif_interp = {}
            for risk, cif_matrix in cif.items():
                cif_interp[risk] = np.zeros((n_samples, len(times)))
                for i in range(n_samples):
                    cif_interp[risk][i] = np.interp(
                        times, self.times_, cif_matrix[i],
                        left=0, right=cif_matrix[i, -1]
                    )
            return cif_interp
        
        return cif
    
    def predict_survival(self, X: np.ndarray) -> np.ndarray:
        """
        Predict overall survival function (1 - sum of CIFs).
        
        Parameters
        ----------
        X : np.ndarray
            Feature matrix
            
        Returns
        -------
        survival : np.ndarray
            Survival probabilities (n_samples, n_intervals)
        """
        X = np.asarray(X)
        n_samples = len(X)
        
        X_pred = self.transformer_.transform_for_prediction(X)
        probs = self.classifier_.predict_proba(X_pred)
        
        _, survival = self.transformer_.compute_cif_from_probs(probs, n_samples)
        return survival
    
    def get_risk_scores(
        self,
        X: np.ndarray,
        risk: int = 1,
        time_quantile: float = 0.5
    ) -> np.ndarray:
        """
        Get risk scores for a specific cause at a specific time quantile.
        
        Parameters
        ----------
        X : np.ndarray
            Feature matrix
        risk : int
            Which competing risk (1, 2, ...)
        time_quantile : float
            Which time quantile (0.5 = median)
            
        Returns
        -------
        scores : np.ndarray
            Risk scores (higher = higher risk)
        """
        cif = self.predict_cif(X)
        
        # Get time index for the quantile
        t_idx = int(time_quantile * (len(self.times_) - 1))
        
        return cif[risk][:, t_idx]
    
    def get_times(self) -> np.ndarray:
        """Return the time points used for predictions."""
        if self.times_ is None:
            raise ValueError("Model not fitted.")
        return self.times_


if __name__ == '__main__':
    # Test the model
    from datasets.datasets import load_dataset
    from .utils import split_data
    
    print("Testing MultiClassSurvivalStacking...")
    
    # Load data using existing datasets module
    X, T, E, features = load_dataset('SYNTHETIC_COMPETING')
    n_risks = int(E.max())
    splits = split_data(X, T, E)
    
    X_train, T_train, E_train = splits['train']
    X_val, T_val, E_val = splits['val']
    X_test, T_test, E_test = splits['test']
    
    # Fit model
    model = MultiClassSurvivalStacking(n_intervals=15)
    model.fit(X_train, T_train, E_train, X_val, T_val, E_val)
    
    # Predict
    cif = model.predict_cif(X_test)
    print(f"\nCIF shapes: {[f'Risk {k}: {v.shape}' for k, v in cif.items()]}")
    
    # Get risk scores
    scores = model.get_risk_scores(X_test, risk=1)
    print(f"Risk scores shape: {scores.shape}")
    print(f"Risk scores range: [{scores.min():.3f}, {scores.max():.3f}]")
