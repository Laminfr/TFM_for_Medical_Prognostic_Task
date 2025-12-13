"""
Survival Stacking Model

Implements discrete-time survival prediction using gradient boosting
classifiers. The key innovation is treating survival analysis as a
sequence of binary classification problems.

Key Benefits over Cox-based approaches:
1. Better probability calibration (directly optimizes log-loss)
2. Handles non-proportional hazards naturally
3. Can capture complex time-varying effects
4. Standard ML toolkit (XGBoost, LightGBM, etc.)
"""

import numpy as np
import pandas as pd
from typing import Optional, Dict, Any, Tuple, Union
from sklearn.base import BaseEstimator
from sklearn.model_selection import RandomizedSearchCV
import warnings

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    warnings.warn("XGBoost not available. Install with: pip install xgboost")

from .discrete_time import DiscreteTimeTransformer


class SurvivalStackingModel(BaseEstimator):
    """
    Survival Stacking Model using discrete-time hazard estimation.
    
    This model transforms survival data into person-period format and
    trains a binary classifier to predict interval-specific hazard rates.
    Survival curves are reconstructed via the product-limit formula.
    
    Parameters
    ----------
    n_intervals : int, default=20
        Number of time intervals for discretization
    interval_strategy : str, default='quantile'
        How to create intervals: 'quantile' or 'uniform'
    classifier : str, default='xgboost'
        Base classifier to use: 'xgboost', 'lightgbm', or 'logistic'
    classifier_params : dict, optional
        Parameters for the base classifier
    random_state : int, default=42
        Random seed for reproducibility
        
    Attributes
    ----------
    transformer_ : DiscreteTimeTransformer
        Fitted time discretizer
    classifier_ : BaseEstimator
        Trained binary classifier
    """
    
    def __init__(
        self,
        n_intervals: int = 20,
        interval_strategy: str = 'quantile',
        classifier: str = 'xgboost',
        classifier_params: Optional[Dict[str, Any]] = None,
        random_state: int = 42
    ):
        self.n_intervals = n_intervals
        self.interval_strategy = interval_strategy
        self.classifier = classifier
        self.classifier_params = classifier_params or {}
        self.random_state = random_state
        
        # Fitted attributes
        self.transformer_ = None
        self.classifier_ = None
        self.classes_ = np.array([0, 1])
        
    def _create_classifier(self, scale_pos_weight: float = 1.0) -> BaseEstimator:
        """Initialize the base classifier with configured parameters."""
        if self.classifier == 'xgboost':
            if not XGBOOST_AVAILABLE:
                raise ImportError("XGBoost not installed. pip install xgboost")
            
            default_params = {
                'objective': 'binary:logistic',
                'eval_metric': 'auc',  # Better for imbalanced data
                'n_estimators': 300,
                'max_depth': 5,
                'learning_rate': 0.05,  # Lower LR with more trees
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'min_child_weight': 10,  # More regularization
                'gamma': 0.1,  # Minimum loss reduction
                'reg_alpha': 0.1,  # L1 regularization
                'reg_lambda': 1.0,  # L2 regularization
                'scale_pos_weight': scale_pos_weight,  # Handle class imbalance
                'random_state': self.random_state,
                'n_jobs': -1,
                'verbosity': 0
            }
            default_params.update(self.classifier_params)
            return XGBClassifier(**default_params)
            
        elif self.classifier == 'lightgbm':
            try:
                from lightgbm import LGBMClassifier
                default_params = {
                    'objective': 'binary',
                    'metric': 'auc',
                    'n_estimators': 300,
                    'max_depth': 5,
                    'learning_rate': 0.05,
                    'subsample': 0.8,
                    'colsample_bytree': 0.8,
                    'min_child_samples': 20,
                    'scale_pos_weight': scale_pos_weight,
                    'reg_alpha': 0.1,
                    'reg_lambda': 1.0,
                    'random_state': self.random_state,
                    'n_jobs': -1,
                    'verbose': -1
                }
                default_params.update(self.classifier_params)
                return LGBMClassifier(**default_params)
            except ImportError:
                raise ImportError("LightGBM not installed. pip install lightgbm")
                
        elif self.classifier == 'logistic':
            from sklearn.linear_model import LogisticRegression
            default_params = {
                'penalty': 'l2',
                'C': 1.0,
                'max_iter': 1000,
                'class_weight': 'balanced',  # Handle imbalance for logistic
                'random_state': self.random_state,
                'n_jobs': -1
            }
            default_params.update(self.classifier_params)
            return LogisticRegression(**default_params)
            
        else:
            raise ValueError(f"Unknown classifier: {self.classifier}")
    
    def fit(
        self, 
        X: np.ndarray, 
        T: np.ndarray, 
        E: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        T_val: Optional[np.ndarray] = None,
        E_val: Optional[np.ndarray] = None,
        verbose: bool = True
    ) -> 'SurvivalStackingModel':
        """
        Fit the survival stacking model.
        
        Parameters
        ----------
        X : np.ndarray
            Training features (n_samples, n_features)
        T : np.ndarray
            Time to event/censoring
        E : np.ndarray
            Event indicator (1=event, 0=censored)
        X_val, T_val, E_val : optional
            Validation data for early stopping
        verbose : bool
            Print progress information
            
        Returns
        -------
        self
        """
        X = np.asarray(X)
        T = np.asarray(T).flatten()
        E = np.asarray(E).flatten()
        
        if verbose:
            print(f"Fitting SurvivalStackingModel...")
            print(f"  - Samples: {len(T)}, Events: {E.sum()} ({100*E.mean():.1f}%)")
            print(f"  - Features: {X.shape[1]}")
            print(f"  - Intervals: {self.n_intervals}")
        
        # Step 1: Fit the time discretizer
        self.transformer_ = DiscreteTimeTransformer(
            n_intervals=self.n_intervals,
            strategy=self.interval_strategy,
            include_time_features=True
        )
        self.transformer_.fit(T, E)
        
        if verbose:
            actual_intervals = len(self.transformer_.cut_points_) - 1
            print(f"  - Actual intervals: {actual_intervals}")
        
        # Step 2: Transform to person-period format
        X_expanded, y_expanded = self.transformer_.transform(X, T, E)
        
        if verbose:
            print(f"  - Expanded rows: {len(y_expanded)} (avg {len(y_expanded)/len(T):.1f} per patient)")
            print(f"  - Event rate in expanded data: {100*y_expanded.mean():.2f}%")
        
        # Adaptive class weighting based on ORIGINAL event rate
        # Only apply class weighting if original data is imbalanced (event rate < 45%)
        original_event_rate = E.mean()
        n_neg = (y_expanded == 0).sum()
        n_pos = (y_expanded == 1).sum()
        raw_imbalance_ratio = n_neg / max(n_pos, 1)
        
        if original_event_rate < 0.45:
            # Imbalanced dataset (like PBC with 37% events)
            # Use sqrt of ratio for milder correction
            scale_pos_weight = np.sqrt(raw_imbalance_ratio)
            weighting_strategy = "sqrt (imbalanced dataset)"
        else:
            # Balanced dataset (like METABRIC with 58% events)
            # No class weighting needed
            scale_pos_weight = 1.0
            weighting_strategy = "none (balanced dataset)"
        
        if verbose:
            print(f"  - Original event rate: {100*original_event_rate:.1f}%")
            print(f"  - Person-period imbalance: {raw_imbalance_ratio:.1f}:1")
            print(f"  - Class weighting: {weighting_strategy}, scale_pos_weight={scale_pos_weight:.2f}")
        
        # Step 3: Create and fit classifier with adaptive class weighting
        self.classifier_ = self._create_classifier(scale_pos_weight=scale_pos_weight)
        
        # Handle validation data if provided
        fit_params = {}
        if X_val is not None and T_val is not None and E_val is not None:
            X_val_expanded, y_val_expanded = self.transformer_.transform(X_val, T_val, E_val)
            
            if self.classifier in ['xgboost', 'lightgbm']:
                fit_params['eval_set'] = [(X_val_expanded, y_val_expanded)]
                if self.classifier == 'xgboost':
                    fit_params['verbose'] = False
        
        if verbose:
            print(f"  - Training {self.classifier} classifier...")
        
        self.classifier_.fit(X_expanded, y_expanded, **fit_params)
        
        if verbose:
            print("  - Done!")
        
        return self
    
    def predict_hazard(self, X: np.ndarray) -> np.ndarray:
        """
        Predict hazard probabilities for all time intervals.
        
        Parameters
        ----------
        X : np.ndarray
            Feature matrix (n_samples, n_features)
            
        Returns
        -------
        hazard : np.ndarray
            Hazard probabilities (n_samples, n_intervals)
            h[i,j] = P(event in interval j | survived to interval j) for patient i
        """
        if self.classifier_ is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        X = np.asarray(X)
        n_samples = X.shape[0]
        n_intervals = len(self.transformer_.cut_points_) - 1
        
        # Create prediction matrix (all patients x all intervals)
        X_pred = self.transformer_.transform_for_prediction(X)
        
        # Predict hazard probabilities
        hazard_flat = self.classifier_.predict_proba(X_pred)[:, 1]
        
        # Reshape to (n_samples, n_intervals)
        hazard = hazard_flat.reshape(n_samples, n_intervals)
        
        return hazard
    
    def predict_survival(
        self, 
        X: np.ndarray, 
        times: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Predict survival probabilities.
        
        Uses the product-limit formula: S(t) = prod_{j: t_j <= t} (1 - h_j)
        
        Parameters
        ----------
        X : np.ndarray
            Feature matrix (n_samples, n_features)
        times : np.ndarray, optional
            Specific times at which to evaluate survival
            If None, returns survival at interval midpoints
            
        Returns
        -------
        survival : np.ndarray
            Survival probabilities
            If times is None: (n_samples, n_intervals)
            If times provided: (n_samples, len(times))
        """
        # Get hazard probabilities
        hazard = self.predict_hazard(X)
        
        # Compute survival via product-limit: S(t_j) = prod_{k<=j} (1 - h_k)
        # Using cumulative product of (1 - hazard)
        survival_at_intervals = np.cumprod(1 - hazard, axis=1)
        
        if times is None:
            return survival_at_intervals
        
        # Interpolate to requested times
        interval_times = self.transformer_.get_interval_times()
        n_samples = X.shape[0]
        n_times = len(times)
        
        survival = np.zeros((n_samples, n_times))
        for i in range(n_samples):
            # Interpolate survival curve
            survival[i, :] = np.interp(
                times, 
                interval_times, 
                survival_at_intervals[i, :],
                left=1.0,  # S(0) = 1
                right=survival_at_intervals[i, -1]  # Extrapolate last value
            )
        
        return survival
    
    def predict_risk(self, X: np.ndarray, time: Optional[float] = None) -> np.ndarray:
        """
        Predict risk scores (1 - survival).
        
        Parameters
        ----------
        X : np.ndarray
            Feature matrix
        time : float, optional
            Time at which to evaluate risk. If None, uses median time.
            
        Returns
        -------
        risk : np.ndarray
            Risk scores (n_samples,)
        """
        if time is None:
            time = np.median(self.transformer_.get_interval_times())
        
        survival = self.predict_survival(X, times=np.array([time]))
        return 1 - survival[:, 0]
    
    def get_feature_importance(self) -> Optional[np.ndarray]:
        """Get feature importance from the classifier if available."""
        if self.classifier_ is None:
            return None
        
        if hasattr(self.classifier_, 'feature_importances_'):
            return self.classifier_.feature_importances_
        elif hasattr(self.classifier_, 'coef_'):
            return np.abs(self.classifier_.coef_).flatten()
        return None


class SurvivalStackingCV:
    """
    Cross-validated Survival Stacking with hyperparameter tuning.
    """
    
    def __init__(
        self,
        n_intervals: int = 20,
        classifier: str = 'xgboost',
        param_grid: Optional[Dict] = None,
        n_iter: int = 20,
        cv: int = 5,
        random_state: int = 42
    ):
        self.n_intervals = n_intervals
        self.classifier = classifier
        self.param_grid = param_grid
        self.n_iter = n_iter
        self.cv = cv
        self.random_state = random_state
        
        self.best_model_ = None
        self.best_params_ = None
        
    def fit(self, X: np.ndarray, T: np.ndarray, E: np.ndarray, verbose: bool = True):
        """Fit with hyperparameter search."""
        
        if self.param_grid is None:
            # Default parameter grid
            if self.classifier == 'xgboost':
                self.param_grid = {
                    'n_estimators': [100, 200, 300],
                    'max_depth': [4, 6, 8],
                    'learning_rate': [0.05, 0.1, 0.2],
                    'min_child_weight': [1, 5, 10],
                    'subsample': [0.7, 0.8, 0.9],
                    'colsample_bytree': [0.7, 0.8, 0.9]
                }
        
        # Create transformer and expand data
        transformer = DiscreteTimeTransformer(
            n_intervals=self.n_intervals,
            strategy='quantile',
            include_time_features=True
        )
        transformer.fit(T, E)
        X_expanded, y_expanded = transformer.transform(X, T, E)
        
        if verbose:
            print(f"Running {self.n_iter} random search iterations...")
        
        # Create base classifier
        if self.classifier == 'xgboost':
            base_clf = XGBClassifier(
                objective='binary:logistic',
                random_state=self.random_state,
                n_jobs=-1,
                verbosity=0
            )
        else:
            raise ValueError(f"CV not implemented for {self.classifier}")
        
        # Random search
        search = RandomizedSearchCV(
            base_clf,
            self.param_grid,
            n_iter=self.n_iter,
            cv=self.cv,
            scoring='neg_log_loss',
            random_state=self.random_state,
            n_jobs=-1,
            verbose=1 if verbose else 0
        )
        search.fit(X_expanded, y_expanded)
        
        self.best_params_ = search.best_params_
        
        if verbose:
            print(f"Best params: {self.best_params_}")
            print(f"Best CV score: {-search.best_score_:.4f} (log-loss)")
        
        # Refit with best params
        self.best_model_ = SurvivalStackingModel(
            n_intervals=self.n_intervals,
            classifier=self.classifier,
            classifier_params=self.best_params_,
            random_state=self.random_state
        )
        self.best_model_.fit(X, T, E, verbose=verbose)
        
        return self
    
    def predict_survival(self, X: np.ndarray, times: Optional[np.ndarray] = None):
        return self.best_model_.predict_survival(X, times)
    
    def predict_risk(self, X: np.ndarray, time: Optional[float] = None):
        return self.best_model_.predict_risk(X, time)
