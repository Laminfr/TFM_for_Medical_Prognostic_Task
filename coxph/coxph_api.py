"""
CoxPHFG — single-risk Cox model with NFG-like API and NFG metrics.

- Assumes binary event: e ∈ {0,1}. Any e>0 is treated as 1.
- Provides: fit, predict_survival, ibs (Integrated Brier Score),
  c_index_td (time-dependent truncated concordance).
"""

from __future__ import annotations
from typing import List, Optional
import numpy as np
import pandas as pd
from lifelines import CoxPHFitter

from metrics.calibration import integrated_brier_score as nfg_integrated_brier
from metrics.discrimination import truncated_concordance_td as nfg_cindex_td


class CoxPHFG:
    """
    Single-risk Cox Proportional Hazards with an NFG-like API.

    Parameters
    ----------
    penalizer : float
        L2 penalty for lifelines' CoxPHFitter.
    """

    def __init__(self, penalizer: float = 0.01):
        self.penalizer = penalizer
        self.model: Optional[CoxPHFitter] = None
        self.fitted = False
        self._train_t: Optional[pd.Series] = None
        self._train_e: Optional[pd.Series] = None

    # ---------------- fitting ----------------

    def fit(self, X: pd.DataFrame, t: pd.Series, e: pd.Series) -> "CoxPHFG":
        """
        Fit a single-risk Cox model: event = (e > 0).
        """
        self._train_t = t.copy()
        self._train_e = (e > 0).astype(int).copy()

        df = pd.concat(
            [X,
             t.rename("duration"),
             (e > 0).astype(int).rename("event")],
            axis=1
        )
        cph = CoxPHFitter(penalizer=self.penalizer)
        cph.fit(df, duration_col="duration", event_col="event")
        self.model = cph
        self.fitted = True
        return self

    # ------------- predictions -------------

    def predict_survival(self, X: pd.DataFrame, times: List[float] | np.ndarray) -> np.ndarray:
        """
        Survival probabilities S(t|x) at given times.
        """
        if not self.fitted or self.model is None:
            raise RuntimeError("Call fit() first.")
        times = np.asarray(times, dtype=float)
        if times.ndim == 0:
            times = np.array([float(times)])
        sf = self.model.predict_survival_function(X, times=times)
        return sf.T.values  # [N, len(times)]

    def predict_risk_matrix(self, X: pd.DataFrame, times: np.ndarray) -> np.ndarray:
        """
        Return risk at times as NxK matrix (what NFG metrics expect):
        risk(t|x) = 1 - S(t|x).
        """
        S = self.predict_survival(X, times)
        return 1.0 - S

    # ------------- NFG metrics wrappers (single risk) -------------

    def ibs(self,
            X_train: pd.DataFrame, t_train: pd.Series, e_train: pd.Series,
            X_test: pd.DataFrame,  t_test: pd.Series,  e_test: pd.Series,
            times: np.ndarray, t_eval: Optional[np.ndarray] = None) -> float:
        """
        Integrated Brier Score using your NFG calibration metric.
        """
        if not self.fitted:
            raise RuntimeError("Call fit() first.")

        times = np.asarray(times, dtype=float)
        risk_pred = self.predict_risk_matrix(X_test, times)  # (N, K)

        ibs, _km = nfg_integrated_brier(
            e_test.values.astype(int),
            t_test.values.astype(float),
            risk_pred,
            times,
            t_eval=None if t_eval is None else np.asarray(t_eval, dtype=float),
            km=((e_train > 0).astype(int).values, t_train.values.astype(float)),
            competing_risk=1  # single risk
        )
        return float(ibs)

    def c_index_td(self,
                   X_test: pd.DataFrame, t_test: pd.Series, e_test: pd.Series,
                   times: np.ndarray, t: Optional[float] = None) -> float:
        """
        Time-dependent truncated concordance using your NFG discrimination metric.
        """
        if not self.fitted:
            raise RuntimeError("Call fit() first.")

        times = np.asarray(times, dtype=float)
        risk_pred = self.predict_risk_matrix(X_test, times)

        ctd, _km = nfg_cindex_td(
            e_test.values.astype(int),
            t_test.values.astype(float),
            risk_pred,
            times,
            t=None if t is None else float(t),
            km=((self._train_e > 0).astype(int).values, self._train_t.values.astype(float)),
            competing_risk=1  # single risk
        )
        return float(ctd)
