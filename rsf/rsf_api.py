import numpy as np
from pandas_patch import pd
from sksurv.ensemble import RandomSurvivalForest
from sksurv.metrics import concordance_index_censored, integrated_brier_score

from metrics.calibration import integrated_brier_score as nfg_integrated_brier
from metrics.discrimination import truncated_concordance_td as nfg_cindex_td


class RSFFG():

    def __init__(self, n_estimators=200, max_depth=10, min_samples_split=20, 
                 min_samples_leaf=10, random_state=42):
        self.model = None
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state
        self.fitted = False
        self._train_times = None
        self._train_events = None

    def fit(self, X_train, y_train):
        """
        Fit Random Survival Forest
        
        Args:
            X_train: Features (numpy array)
            y_train: Structured array with dtype=[('event', bool), ('time', float)]
        """
        rsf_model = RandomSurvivalForest(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            random_state=self.random_state,
            n_jobs=-1,
            verbose=0
        )
        
        rsf_model.fit(X_train, y_train)
        
        self.model = rsf_model
        self.fitted = True
        
        # Store training data for metrics
        self._train_times = y_train['time']
        self._train_events = y_train['event']
        
        return self

    def predict_survival(self, X, times):
        """
        Predict survival probabilities at given times
        
        Args:
            X: Features (n_samples, n_features)
            times: Array of time points
            
        Returns:
            Survival probabilities (n_samples, n_times)
        """
        if not self.fitted:
            raise RuntimeError("Call fit() first.")
        
        # Get survival functions
        surv_funcs = self.model.predict_survival_function(X, return_array=True)
        model_times = self.model.unique_times_
        
        # Interpolate to requested times
        surv_probs = np.zeros((X.shape[0], len(times)))
        
        for i, t in enumerate(times):
            idx = np.searchsorted(model_times, t, side='left')
            if idx >= len(model_times):
                idx = len(model_times) - 1
            surv_probs[:, i] = surv_funcs[:, idx]
        
        return surv_probs

    def predict_risk_matrix(self, X, times):
        """Predict risk (1 - survival) at given times"""
        return 1.0 - self.predict_survival(X, times)

    def predict_risk(self, X):
        """Predict cumulative hazard (risk scores)"""
        if not self.fitted:
            raise RuntimeError("Call fit() first.")
        return self.model.predict(X)

    def ibs(self, 
            X_train, t_train, e_train,
            X_test, t_test, e_test,
            times, t_eval=None):
        """
        Integrated Brier Score
        
        Args:
            X_train, t_train, e_train: Training data (for censoring distribution)
            X_test, t_test, e_test: Test data
            times: Evaluation times
            t_eval: Optional time horizon (unused for compatibility)
        """
        if not self.fitted:
            raise RuntimeError("Call fit() first.")
        
        # Convert to pandas Series if needed
        if isinstance(t_test, pd.Series):
            t_test = t_test.values
        if isinstance(e_test, pd.Series):
            e_test = e_test.values
        
        # Get risk predictions
        risk_pred = self.predict_risk_matrix(X_test, times)
        
        # Use NFG's integrated Brier score
        ibs, _ = nfg_integrated_brier(
            e_test.astype(int),
            t_test.astype(float),
            risk_pred,
            times,
            t_eval=None if t_eval is None else np.asarray(t_eval, dtype=float),
            km=(e_train.astype(int), t_train.astype(float)),
            competing_risk=1  # single risk
        )
        return float(ibs)

    def c_index_td(self, X_test, t_test, e_test, times, t=None):
        """
        Time-dependent concordance index
        
        Args:
            X_test, t_test, e_test: Test data
            times: Evaluation times
            t: Specific time point for truncation
        """
        if not self.fitted:
            raise RuntimeError("Call fit() first.")
        
        # Convert to pandas Series if needed
        if isinstance(t_test, pd.Series):
            t_test = t_test.values
        if isinstance(e_test, pd.Series):
            e_test = e_test.values
        
        # Get risk predictions
        risk_pred = self.predict_risk_matrix(X_test, times)
        
        # Use NFG's C-index TD
        ctd, _ = nfg_cindex_td(
            e_test.astype(int),
            t_test.astype(float),
            risk_pred,
            times,
            t=None if t is None else float(t),
            km=(self._train_events.astype(int), self._train_times.astype(float)),
            competing_risk=1  # single risk
        )
        return float(ctd)