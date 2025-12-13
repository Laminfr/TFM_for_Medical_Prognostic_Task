"""
Discrete Time Transformation for Survival Analysis

Transforms continuous survival data into discrete person-period format
suitable for binary classification.

The key insight is that survival analysis can be reformulated as:
- For each patient, create multiple rows (one per time interval they were at risk)
- Target y=1 only in the interval where the event occurred (if any)
- Censored patients contribute y=0 rows up to their censoring time

This allows using standard classifiers (XGBoost, LogisticRegression, etc.)
while maintaining proper survival semantics.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional, List, Union
from sklearn.base import BaseEstimator, TransformerMixin


class DiscreteTimeTransformer(BaseEstimator, TransformerMixin):
    """
    Transforms survival data to discrete person-period format.
    
    Parameters
    ----------
    n_intervals : int, default=20
        Number of time intervals to create
    strategy : str, default='quantile'
        How to create intervals: 'quantile' (balanced events) or 'uniform' (equal width)
    include_time_features : bool, default=True
        Whether to add time-related features (interval index, elapsed time, etc.)
    
    Attributes
    ----------
    cut_points_ : np.ndarray
        The boundaries of time intervals (fitted from training data)
    interval_midpoints_ : np.ndarray
        Midpoint of each interval (useful for evaluation)
    """
    
    def __init__(
        self, 
        n_intervals: int = 20,
        strategy: str = 'quantile',
        include_time_features: bool = True
    ):
        self.n_intervals = n_intervals
        self.strategy = strategy
        self.include_time_features = include_time_features
        
        # Fitted attributes
        self.cut_points_ = None
        self.interval_midpoints_ = None
        self.max_time_ = None
        
    def fit(self, T: np.ndarray, E: np.ndarray) -> 'DiscreteTimeTransformer':
        """
        Fit the transformer by computing time interval boundaries.
        
        Parameters
        ----------
        T : np.ndarray
            Time to event or censoring
        E : np.ndarray  
            Event indicator (1=event, 0=censored)
            
        Returns
        -------
        self
        """
        T = np.asarray(T).flatten()
        E = np.asarray(E).flatten()
        
        self.max_time_ = T.max()
        
        if self.strategy == 'quantile':
            # Use event times for quantile computation (more balanced)
            event_times = T[E > 0]
            if len(event_times) < self.n_intervals:
                # Fallback to all times if not enough events
                event_times = T
            
            # Compute quantile-based cut points
            percentiles = np.linspace(0, 100, self.n_intervals + 1)
            self.cut_points_ = np.percentile(event_times, percentiles)
            
            # Ensure unique and sorted cut points
            self.cut_points_ = np.unique(self.cut_points_)
            
            # Adjust to include 0 and max_time
            if self.cut_points_[0] > 0:
                self.cut_points_ = np.concatenate([[0], self.cut_points_])
            if self.cut_points_[-1] < self.max_time_:
                self.cut_points_ = np.concatenate([self.cut_points_, [self.max_time_ + 1e-6]])
                
        elif self.strategy == 'uniform':
            self.cut_points_ = np.linspace(0, self.max_time_ + 1e-6, self.n_intervals + 1)
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")
        
        # Compute interval midpoints for evaluation
        self.interval_midpoints_ = (self.cut_points_[:-1] + self.cut_points_[1:]) / 2
        
        return self
    
    def transform(
        self, 
        X: np.ndarray, 
        T: np.ndarray, 
        E: np.ndarray,
        return_indices: bool = False
    ) -> Union[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """
        Transform to person-period format.
        
        Parameters
        ----------
        X : np.ndarray
            Feature matrix (n_samples, n_features)
        T : np.ndarray
            Time to event or censoring
        E : np.ndarray
            Event indicator
        return_indices : bool
            If True, also return original patient indices
            
        Returns
        -------
        X_expanded : np.ndarray
            Expanded features with time index (n_rows, n_features + time_features)
        y_expanded : np.ndarray
            Binary targets for each person-period
        patient_indices : np.ndarray (optional)
            Original patient index for each row
        """
        if self.cut_points_ is None:
            raise ValueError("Transformer not fitted. Call fit() first.")
            
        X = np.asarray(X)
        T = np.asarray(T).flatten()
        E = np.asarray(E).flatten()
        
        n_samples = len(T)
        n_intervals = len(self.cut_points_) - 1
        
        # For each patient, determine which intervals they were at risk
        expanded_rows = []
        expanded_y = []
        expanded_patient_idx = []
        
        for i in range(n_samples):
            t_i = T[i]
            e_i = E[i]
            
            # Find which interval the event/censoring falls into
            interval_idx = np.searchsorted(self.cut_points_[1:], t_i, side='left')
            interval_idx = min(interval_idx, n_intervals - 1)
            
            # Patient contributes rows for all intervals up to and including their event/censoring interval
            for j in range(interval_idx + 1):
                # Create row with features + time information
                if self.include_time_features:
                    time_features = self._create_time_features(j, n_intervals)
                    row = np.concatenate([X[i], time_features])
                else:
                    row = X[i]
                
                expanded_rows.append(row)
                expanded_patient_idx.append(i)
                
                # y=1 only if event occurred in this interval
                if j == interval_idx and e_i > 0:
                    expanded_y.append(1)
                else:
                    expanded_y.append(0)
        
        X_expanded = np.array(expanded_rows)
        y_expanded = np.array(expanded_y)
        patient_indices = np.array(expanded_patient_idx)
        
        if return_indices:
            return X_expanded, y_expanded, patient_indices
        return X_expanded, y_expanded
    
    def _create_time_features(self, interval_idx: int, n_intervals: int) -> np.ndarray:
        """Create time-related features for a given interval."""
        features = [
            interval_idx,  # Integer time index
            interval_idx / n_intervals,  # Normalized time (0-1)
            self.interval_midpoints_[interval_idx],  # Actual time value
        ]
        return np.array(features)
    
    def transform_for_prediction(self, X: np.ndarray) -> np.ndarray:
        """
        Create prediction matrix for new patients.
        
        Each patient needs predictions for ALL intervals to construct
        their survival curve.
        
        Parameters
        ----------
        X : np.ndarray
            Feature matrix (n_samples, n_features)
            
        Returns
        -------
        X_pred : np.ndarray
            Expanded matrix with all intervals (n_samples * n_intervals, n_features + time_features)
        """
        if self.cut_points_ is None:
            raise ValueError("Transformer not fitted. Call fit() first.")
            
        X = np.asarray(X)
        n_samples = X.shape[0]
        n_intervals = len(self.cut_points_) - 1
        
        rows = []
        for i in range(n_samples):
            for j in range(n_intervals):
                if self.include_time_features:
                    time_features = self._create_time_features(j, n_intervals)
                    row = np.concatenate([X[i], time_features])
                else:
                    row = X[i]
                rows.append(row)
        
        return np.array(rows)
    
    def get_interval_times(self) -> np.ndarray:
        """Return the midpoint times of each interval."""
        return self.interval_midpoints_.copy()
    
    def get_cut_points(self) -> np.ndarray:
        """Return the interval boundaries."""
        return self.cut_points_.copy()
    
    def time_to_interval(self, t: Union[float, np.ndarray]) -> Union[int, np.ndarray]:
        """Convert continuous time to interval index."""
        return np.searchsorted(self.cut_points_[1:], t, side='left')


def create_person_period_dataset(
    X: np.ndarray,
    T: np.ndarray, 
    E: np.ndarray,
    n_intervals: int = 20,
    strategy: str = 'quantile'
) -> Tuple[np.ndarray, np.ndarray, DiscreteTimeTransformer]:
    """
    Convenience function to create person-period dataset.
    
    Parameters
    ----------
    X : np.ndarray
        Features
    T : np.ndarray
        Event times
    E : np.ndarray
        Event indicators
    n_intervals : int
        Number of time intervals
    strategy : str
        'quantile' or 'uniform'
        
    Returns
    -------
    X_expanded : np.ndarray
        Person-period features
    y_expanded : np.ndarray
        Binary targets
    transformer : DiscreteTimeTransformer
        Fitted transformer for inference
    """
    transformer = DiscreteTimeTransformer(
        n_intervals=n_intervals,
        strategy=strategy,
        include_time_features=True
    )
    transformer.fit(T, E)
    X_expanded, y_expanded = transformer.transform(X, T, E)
    
    return X_expanded, y_expanded, transformer
