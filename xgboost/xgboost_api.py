import numpy as np
import xgboost as xgb
from lifelines import KaplanMeierFitter

from metrics.calibration import integrated_brier_score as nfg_integrated_brier
from metrics.discrimination import truncated_concordance_td as nfg_cindex_td


class XGBoostFG():

    def __init__(self, n_estimators = 200, max_depth = 10, min_samples_split = 20, min_samples_leaf = 10, random_state = 42):
        self.model = None,
        self.n_estimators = n_estimators,
        self.max_depth = max_depth,
        self.min_samples_split = min_samples_split,
        self.min_samples_leaf = min_samples_leaf,
        self.random_state = random_state,
        self.fitted = False

    # ---------------- fitting ----------------

    def fit(self, X_train, y_train):
        pass

    # ------------- predictions -------------

    def predict_survival(self, ):
        pass

    # ------------- NFG metrics wrappers (single risk) -------------

    def ibs(self,):
        if not self.fitted:
            raise RuntimeError("Call fit() first.")
        pass

    def c_index_td(self):
        if not self.fitted:
            raise RuntimeError("Call fit() first.")
        pass