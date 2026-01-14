"""
Discrete Time Transformation for Competing Risks (Multi-Class)

Transforms continuous competing risks data into discrete person-period format
suitable for multi-class classification.

Key difference from binary survival:
- Target y has multiple values: 0 (survived interval), 1 (Event A), 2 (Event B), etc.
- Each patient contributes rows for each interval they were at risk
- Final row has y = event_type if event occurred in that interval
"""

import numpy as np
from typing import Tuple, Optional, List, Union
from sklearn.base import BaseEstimator, TransformerMixin


class DiscreteTimeCompetingRisksTransformer(BaseEstimator, TransformerMixin):
    """
    Transforms competing risks survival data to discrete person-period format
    for multi-class classification.
    
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
    n_classes_ : int
        Number of classes (n_risks + 1 for survival)
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
        self.n_classes_ = None
        self.n_risks_ = None
        
    def fit(self, T: np.ndarray, E: np.ndarray) -> 'DiscreteTimeCompetingRisksTransformer':
        """
        Fit the transformer by computing time interval boundaries.
        
        Parameters
        ----------
        T : np.ndarray
            Time to event or censoring
        E : np.ndarray  
            Event indicator (0=censored, 1=cause1, 2=cause2, ...)
            
        Returns
        -------
        self
        """
        T = np.asarray(T).flatten()
        E = np.asarray(E).flatten()
        
        self.max_time_ = T.max()
        self.n_risks_ = int(E.max())  # Number of competing events
        self.n_classes_ = self.n_risks_ + 1  # +1 for "survived this interval"
        
        if self.strategy == 'quantile':
            # Use event times (any event) for quantile computation
            event_times = T[E > 0]
            if len(event_times) < self.n_intervals:
                event_times = T
            
            percentiles = np.linspace(0, 100, self.n_intervals + 1)
            self.cut_points_ = np.percentile(event_times, percentiles)
            self.cut_points_ = np.unique(self.cut_points_)
            
            if self.cut_points_[0] > 0:
                self.cut_points_ = np.concatenate([[0], self.cut_points_])
            if self.cut_points_[-1] < self.max_time_:
                self.cut_points_ = np.concatenate([self.cut_points_, [self.max_time_ + 1e-6]])
                
        elif self.strategy == 'uniform':
            self.cut_points_ = np.linspace(0, self.max_time_ + 1e-6, self.n_intervals + 1)
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")
        
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
        Transform to person-period format with multi-class targets.
        
        Parameters
        ----------
        X : np.ndarray
            Feature matrix (n_samples, n_features)
        T : np.ndarray
            Time to event or censoring
        E : np.ndarray
            Event indicator (0=censored, 1,2,...=event types)
        return_indices : bool
            If True, also return original patient indices
            
        Returns
        -------
        X_expanded : np.ndarray
            Expanded features with time features (n_rows, n_features + time_features)
        y_expanded : np.ndarray
            Multi-class targets for each person-period:
            - 0 = survived this interval
            - 1 = Event type 1 occurred
            - 2 = Event type 2 occurred
            - etc.
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
        
        expanded_rows = []
        expanded_y = []
        expanded_patient_idx = []
        
        for i in range(n_samples):
            t_i = T[i]
            e_i = E[i]
            
            # Find which interval the event/censoring falls into
            interval_idx = np.searchsorted(self.cut_points_[1:], t_i, side='left')
            interval_idx = min(interval_idx, n_intervals - 1)
            
            # Patient contributes rows for all intervals up to event/censoring
            for j in range(interval_idx + 1):
                if self.include_time_features:
                    time_features = self._create_time_features(j, n_intervals)
                    row = np.concatenate([X[i], time_features])
                else:
                    row = X[i].copy()
                
                expanded_rows.append(row)
                expanded_patient_idx.append(i)
                
                # Multi-class target:
                # - y = 0: survived this interval (or censored, or event in later interval)
                # - y = e_i: event of type e_i occurred in this interval
                if j == interval_idx and e_i > 0:
                    expanded_y.append(int(e_i))  # Event occurred
                else:
                    expanded_y.append(0)  # Survived this interval
        
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
        
        Each patient needs predictions for ALL intervals to compute CIF.
        
        Parameters
        ----------
        X : np.ndarray
            Feature matrix (n_samples, n_features)
            
        Returns
        -------
        X_pred : np.ndarray
            Expanded features (n_samples * n_intervals, n_features + time_features)
        """
        if self.cut_points_ is None:
            raise ValueError("Transformer not fitted. Call fit() first.")
            
        X = np.asarray(X)
        n_samples = len(X)
        n_intervals = len(self.cut_points_) - 1
        
        rows = []
        for i in range(n_samples):
            for j in range(n_intervals):
                if self.include_time_features:
                    time_features = self._create_time_features(j, n_intervals)
                    row = np.concatenate([X[i], time_features])
                else:
                    row = X[i].copy()
                rows.append(row)
        
        return np.array(rows)
    
    def get_interval_for_time(self, t: float) -> int:
        """Get the interval index for a specific time point."""
        if self.cut_points_ is None:
            raise ValueError("Transformer not fitted. Call fit() first.")
        idx = np.searchsorted(self.cut_points_[1:], t, side='left')
        return min(idx, len(self.interval_midpoints_) - 1)
    
    def compute_cif_from_probs(
        self, 
        probs: np.ndarray,
        n_samples: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute Cumulative Incidence Functions from interval probabilities.
        
        The CIF for cause k at time t is:
        CIF_k(t) = sum_{j: t_j <= t} P(event=k | interval j, survived to j) * S(t_{j-1})
        
        Where S(t) is the overall survival function.
        
        Parameters
        ----------
        probs : np.ndarray
            Predicted probabilities (n_samples * n_intervals, n_classes)
            From classifier.predict_proba()
        n_samples : int
            Number of original patients
            
        Returns
        -------
        cif : dict
            CIF for each risk: {risk: array (n_samples, n_intervals)}
        survival : np.ndarray
            Overall survival function (n_samples, n_intervals)
        """
        n_intervals = len(self.interval_midpoints_)
        n_classes = probs.shape[1]
        n_risks = n_classes - 1  # Class 0 is "survived interval"
        
        # Reshape probs: (n_samples, n_intervals, n_classes)
        probs_reshaped = probs.reshape(n_samples, n_intervals, n_classes)
        
        # Initialize arrays
        survival = np.ones((n_samples, n_intervals))
        cif = {k: np.zeros((n_samples, n_intervals)) for k in range(1, n_risks + 1)}
        
        for j in range(n_intervals):
            # Probability of surviving this interval (class 0)
            p_survive = probs_reshaped[:, j, 0]
            
            if j == 0:
                survival[:, j] = p_survive
            else:
                # S(t_j) = S(t_{j-1}) * P(survive interval j)
                survival[:, j] = survival[:, j-1] * p_survive
            
            # CIF for each cause
            for k in range(1, n_risks + 1):
                # P(event=k in interval j)
                p_event_k = probs_reshaped[:, j, k]
                
                # CIF_k(t_j) = CIF_k(t_{j-1}) + S(t_{j-1}) * P(event=k | interval j)
                if j == 0:
                    cif[k][:, j] = p_event_k
                else:
                    cif[k][:, j] = cif[k][:, j-1] + survival[:, j-1] * p_event_k
        
        return cif, survival


if __name__ == '__main__':
    # Test transformer
    np.random.seed(42)
    
    n = 100
    n_features = 5
    X = np.random.randn(n, n_features)
    T = np.random.exponential(10, n)
    E = np.random.choice([0, 1, 2], n, p=[0.3, 0.4, 0.3])
    
    transformer = DiscreteTimeCompetingRisksTransformer(n_intervals=10)
    transformer.fit(T, E)
    
    print(f"Cut points: {transformer.cut_points_}")
    print(f"N classes: {transformer.n_classes_}")
    
    X_exp, y_exp, idx = transformer.transform(X, T, E, return_indices=True)
    print(f"\nOriginal: {X.shape}")
    print(f"Expanded: {X_exp.shape}")
    print(f"Y distribution: {dict(zip(*np.unique(y_exp, return_counts=True)))}")
