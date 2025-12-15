from pandas_patch import pd
from nfg.nfg_api import NeuralFineGray
import numpy as np
from .utilities import *

from metrics.calibration import integrated_brier_score as nfg_integrated_brier
from metrics.discrimination import truncated_concordance_td as nfg_cindex_td


class CoxPHFG(NeuralFineGray):

    def __init__(self, penalizer = 0.01):
        super().__init__()
        self.cuda = 0
        self.penalizer = penalizer
        self.model = None
        self.eval_params = None

    # ---------------- fitting ----------------

    def fit(self, x, t, e, vsize = 0.15, val_data = None, random_state = 100):

        processed_data = self._preprocess_training_data(x, t, e, vsize, val_data, random_state)
        x_train, t_train, e_train, x_val, t_val, e_val = processed_data

        t_train = super()._normalise(t_train, save = True)
        t_val = super()._normalise(t_val)

        x_train = np.asarray(x_train)
        t_train = np.asarray(t_train)
        x_val = np.asarray(x_val)
        t_val = np.asarray(t_val)

        # if E_train is a tensor: to CPU → numpy
        if hasattr(e_train, "detach"):
            e_train = e_train.detach().cpu().numpy()
        else:
            e_train = np.asarray(e_train)

        # if e_val is a tensor: to CPU → numpy
        if hasattr(e_val, "detach"):
            e_val = e_val.detach().cpu().numpy()
        else:
            e_val = np.asarray(e_val)

        x_train = pd.DataFrame(x_train)
        t_train = pd.Series(t_train, name="duration")
        e_train = pd.Series((e_train > 0).astype(int), name="event")

        x_val = pd.DataFrame(x_val)
        t_val = pd.Series(t_val, name="duration")
        e_val = pd.Series((e_val > 0).astype(int), name="event")

        self.model = train_cox_model(x_train, t_train, e_train, self.penalizer)
        self.eval_params = evaluate_model(self.model, x_train, x_val, t_train, t_val, e_train, e_val)
        # summary_output(self.model, x_train, t_train, e_train, x_val, t_val, e_val, self.eval_params)
        self.fitted = True
        return self

    # ------------- predictions -------------

    def predict_survival(self):
        if not self.fitted or self.model is None:
            raise RuntimeError("Call fit() first.")
        return self.eval_params['surv_probs']


    def predict_risk_matrix(self):
        if not self.fitted or self.model is None:
            raise RuntimeError("Call fit() first.")
        return 1.0 - self.eval_params['surv_probs']

