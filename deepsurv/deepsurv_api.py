import numpy as np

from metrics.calibration import integrated_brier_score as nfg_integrated_brier
from metrics.discrimination import truncated_concordance_td as nfg_cindex_td


class DeepSurvFG():
    def __init__(self, ):
        pass

    # ---------------- fitting ----------------

    def fit(self, X_train, y_train):
        pass

    # ------------- predictions -------------

    def predict_survival(self, X, times):
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