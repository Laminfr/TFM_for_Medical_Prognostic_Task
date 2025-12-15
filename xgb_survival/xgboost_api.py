import numpy as np
from pandas_patch import pd
import xgboost as xgb
from lifelines import KaplanMeierFitter

np.seterr(over='ignore', invalid='ignore')

from metrics.calibration import integrated_brier_score as nfg_integrated_brier
from metrics.discrimination import truncated_concordance_td as nfg_cindex_td

class XGBoostFG():

    def __init__(self, n_estimators=200, max_depth=6, learning_rate=0.1, 
                 min_child_weight=10, subsample=0.8, colsample_bytree=0.8,
                 random_state=42):
        self.model = None
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.min_child_weight = min_child_weight
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.random_state = random_state
        self.fitted = False
        self._train_times = None
        self._train_events = None
        self._km_model = None

    def fit(self, X_train, t_train, e_train):
        """
        Fit XGBoost for survival (Cox objective)
        
        Args:
            X_train: Features (numpy array or DataFrame)
            t_train: Time to event (array or Series)
            e_train: Event indicator (array or Series)
        """
        # Convert to numpy if needed
        if isinstance(X_train, pd.DataFrame):
            X_train = X_train.values
        if isinstance(t_train, pd.Series):
            t_train = t_train.values
        if isinstance(e_train, pd.Series):
            e_train = e_train.values
        
        # Store for metrics
        self._train_times = t_train
        self._train_events = e_train
        
        # Fit Kaplan-Meier for baseline
        self._km_model = KaplanMeierFitter()
        self._km_model.fit(t_train, e_train)
        
        # Create DMatrix with survival labels
        # For XGBoost Cox, label is: sign(event) * time
        # Negative time = censored, positive = event
        y_xgb = np.where(e_train > 0, t_train, -t_train)
        
        dtrain = xgb.DMatrix(X_train, label=y_xgb)
        
        # Parameters
        params = {
            'objective': 'survival:cox',
            'eval_metric': 'cox-nloglik',
            'max_depth': self.max_depth,
            'learning_rate': self.learning_rate,
            'min_child_weight': self.min_child_weight,
            'subsample': self.subsample,
            'colsample_bytree': self.colsample_bytree,
            'seed': self.random_state,
            'verbosity': 0
        }
        
        # Train
        self.model = xgb.train(
            params,
            dtrain,
            num_boost_round=self.n_estimators,
            verbose_eval=False
        )
        
        self.fitted = True
        return self

    def predict_risk(self, X):
        """Predict log hazard ratio (risk scores)"""
        if not self.fitted:
            raise RuntimeError("Call fit() first.")
        
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        dtest = xgb.DMatrix(X)
        return self.model.predict(dtest)

    def predict_survival(self, X, times):
        """
        Predict survival probabilities using Breslow estimator
        
        S(t|x) = S0(t)^exp(f(x))
        where S0 is baseline (KM), f(x) is XGBoost risk score
        """
        if not self.fitted:
            raise RuntimeError("Call fit() first.")
        
        # Get risk scores
        risk_scores = self.predict_risk(X)
        
        # Get baseline survival at times
        S0 = np.array([self._km_model.predict(t) for t in times])
        
        # Apply proportional hazards
        # S(t|x) = S0(t)^exp(risk_score)
        surv_probs = np.zeros((len(X), len(times)))
        for i, risk in enumerate(risk_scores):
            surv_probs[i, :] = S0 ** np.exp(risk)
        
        return surv_probs

    def predict_risk_matrix(self, X, times):
        """Predict risk (1 - survival) at times"""
        return 1.0 - self.predict_survival(X, times)

    def ibs(self, 
            X_train, t_train, e_train,
            X_test, t_test, e_test,
            times, t_eval=None):
        """Integrated Brier Score"""
        if not self.fitted:
            raise RuntimeError("Call fit() first.")
        
        # Convert to numpy
        if isinstance(t_test, pd.Series):
            t_test = t_test.values
        if isinstance(e_test, pd.Series):
            e_test = e_test.values
        if isinstance(t_train, pd.Series):
            t_train = t_train.values
        if isinstance(e_train, pd.Series):
            e_train = e_train.values
        
        risk_pred = self.predict_risk_matrix(X_test, times)
        
        ibs, _ = nfg_integrated_brier(
            e_test.astype(int),
            t_test.astype(float),
            risk_pred,
            times,
            t_eval=None if t_eval is None else np.asarray(t_eval, dtype=float),
            km=(e_train.astype(int), t_train.astype(float)),
            competing_risk=1
        )
        return float(ibs)

    def c_index_td(self, X_test, t_test, e_test, times, t=None):
        """Time-dependent concordance index"""
        if not self.fitted:
            raise RuntimeError("Call fit() first.")
        
        if isinstance(t_test, pd.Series):
            t_test = t_test.values
        if isinstance(e_test, pd.Series):
            e_test = e_test.values
        
        risk_pred = self.predict_risk_matrix(X_test, times)
        
        ctd, _ = nfg_cindex_td(
            e_test.astype(int),
            t_test.astype(float),
            risk_pred,
            times,
            t=None if t is None else float(t),
            km=(self._train_events.astype(int), self._train_times.astype(float)),
            competing_risk=1
        )
        return float(ctd)