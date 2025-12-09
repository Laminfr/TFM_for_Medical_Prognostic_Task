import numpy as np

from nfg.nfg_api import NeuralFineGray
from .utilities import train_rsf_model, evaluate_rsf_model, summary_output


class RSFFG(NeuralFineGray):

    def __init__(
        self,
        n_estimators=200,
        max_depth=10,
        min_samples_split=20,
        min_samples_leaf=10,
        random_state=42,
    ):
        super().__init__()
        self.model = None
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state
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

        model = train_rsf_model(
            x_train,
            t_train,
            e_train,
            t_val,
            e_val,
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            random_state=self.random_state,
        )
        self.eval_params = evaluate_rsf_model(
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
    def predict_survival(self, x , times):
        if not self.fitted or self.model is None:
            raise RuntimeError("Call fit() first.")
        # return self.eval_params["surv_probs_val"]
        surv_funcs = self.model.predict_survival_function(x, return_array=True)
        model_times = self.model.unique_times_
        
        # Interpolate to requested times
        surv_probs = np.zeros((x.shape[0], len(times)))
        
        for i, t in enumerate(times):
            idx = np.searchsorted(model_times, t, side='left')
            if idx >= len(model_times):
                idx = len(model_times) - 1
            surv_probs[:, i] = surv_funcs[:, idx]
        
        return surv_probs

    def predict_risk_matrix(self):
        if not self.fitted or self.model is None:
            raise RuntimeError("Call fit() first.")
        return 1.0 - self.eval_params["surv_probs_val"]


def convert_cpu_numpy(tensor):
    if hasattr(tensor, "detach"):
        tensor = tensor.detach().cpu().numpy()
    else:
        tensor = np.asarray(tensor)
    return tensor
