import numpy as np
from pandas_patch import pd
import xgboost as xgb
from lifelines import KaplanMeierFitter

from nfg.nfg_api import NeuralFineGray
from .utilities import train_xgboost_model, evaluate_xgboost_model, summary_output
from metrics.calibration import integrated_brier_score as nfg_integrated_brier
from metrics.discrimination import truncated_concordance_td as nfg_cindex_td

class XGBoostFG(NeuralFineGray):

    def __init__(
        self,
        hyper_grid=None,
        n_iter=100,
        fold=None,
        k=5,
        random_seed=0,
        path="results",
        save=True,
        delete_log=False,
        times=100,
        n_estimators=200,
        learning_rate=0.05,
        min_child_weight=10,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
    ):

        super().__init__(
            hyper_grid=hyper_grid,
            n_iter=n_iter,
            fold=fold,
            k=k,
            random_seed=random_seed,
            path=path,
            save=save,
            delete_log=delete_log,
            times=times,)
        
        self.model = None
        self.objective = 'survival:cox'
        self.eval_metric = 'cox-nloglik'
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.random_state = random_state
        self.eval_params = None
        self._train_times = None
        self._train_events = None
        self._km_model = None
        self.min_child_weight = min_child_weight
        self.fitted = False

    # ---------------- fitting ----------------

    def fit(self, x, t, e, vsize=0.15, val_data=None, random_state=100):
        processed_data = self._preprocess_training_data(
            x, t, e, vsize, val_data, random_state
        )
        x_train, t_train, e_train, x_val, t_val, e_val = processed_data

        t_train = super()._normalise(t_train, save=True)
        t_val = super()._normalise(t_val)

        x_train = convert_cpu_numpy(x_train)
        t_train = convert_cpu_numpy(t_train)
        x_val = convert_cpu_numpy(x_val)
        t_val = convert_cpu_numpy(t_val)
        e_train = convert_cpu_numpy(e_train)
        e_val = convert_cpu_numpy(e_val)

        # Convert to numpy if needed
        if isinstance(x_train, pd.DataFrame):
            x_train = x_train.values
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

        y_xgb = np.where(e_train > 0, t_train, -t_train)
        
        dtrain = xgb.DMatrix(x_train, label=y_xgb)
        
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
        model = xgb.train(
            params,
            dtrain,
            num_boost_round=self.n_estimators,
            verbose_eval=False
        )
        
        self.eval_params = evaluate_xgboost_model(
            model,
            x_train,
            t_train,
            e_train,
            x_val,
            t_val,
            e_val,
        )

        summary_output(x_train, t_train, e_train, x_val, t_val, e_val, self.eval_params)

        self.model = model
        self.fitted = True
        return self

    # ------------- predictions -------------

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


def convert_cpu_numpy(tensor):
    if hasattr(tensor, "detach"):
        tensor = tensor.detach().cpu().numpy()
    else:
        tensor = np.asarray(tensor)
    return tensor

def wrap_np_to_pandas(X, index=None, prefix="x"):
    if isinstance(X, (pd.DataFrame, pd.Series)):
        return X

    X = np.asarray(X)

    if X.ndim == 1:
        return pd.Series(X, index=index)

    if X.ndim == 2:
        n_cols = X.shape[1]
        cols = [f"{prefix}{i}" for i in range(n_cols)]
        return pd.DataFrame(X, columns=cols, index=index)

    raise ValueError("Input must be 1D or 2D numpy array or pandas object.")