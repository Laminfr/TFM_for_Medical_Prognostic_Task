from typing import Tuple, List, Optional
import numpy as np
import pandas as pd
from lifelines import CoxPHFitter

from metrics.calibration import integrated_brier_score as nfg_integrated_brier
from metrics.discrimination import truncated_concordance_td as nfg_cindex_td


def apply_tabpfn_embeddings(
    X_train: np.ndarray,
    X_val: np.ndarray,
    X_test: np.ndarray,
    E_train: np.ndarray,
    mode: str = "deep+raw",
    verbose: bool = True,
    n_estimators: int = 1,
    n_fold: int = 0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if mode == "raw":
        return X_train, X_val, X_test

    try:
        from tabpfn_extensions import TabPFNClassifier
        from tabpfn_extensions.embedding import TabPFNEmbedding

        use_deep = "deep" in mode
        concat_raw = "+raw" in mode

        if not (
            np.isfinite(X_train).all()
            and np.isfinite(X_val).all()
            and np.isfinite(X_test).all()
        ):
            raise ValueError(
                "Non-finite values found in X. Please impute/clean before TabPFN embeddings."
            )

        y_train = E_train.astype(int)

        clf = TabPFNClassifier(n_estimators=n_estimators)

        embedding_extractor = TabPFNEmbedding(tabpfn_clf=clf, n_fold=n_fold)

        embedding_extractor.fit(X_train, y_train)

        train_emb = embedding_extractor.get_embeddings(
            X_train, y_train, X_train, data_source="train"
        )
        val_emb = embedding_extractor.get_embeddings(
            X_train, y_train, X_val, data_source="test"
        )
        test_emb = embedding_extractor.get_embeddings(
            X_train, y_train, X_test, data_source="test"
        )

        train_emb = _tabpfn_emb_to_2d(train_emb, expected_n=X_train.shape[0])
        val_emb   = _tabpfn_emb_to_2d(val_emb,   expected_n=X_val.shape[0])
        test_emb  = _tabpfn_emb_to_2d(test_emb,  expected_n=X_test.shape[0])


        if not use_deep:
            return X_train, X_val, X_test

        if concat_raw:
            X_train_out = np.concatenate([X_train, train_emb], axis=1)
            X_val_out = np.concatenate([X_val, val_emb], axis=1)
            X_test_out = np.concatenate([X_test, test_emb], axis=1)
        else:
            X_train_out, X_val_out, X_test_out = train_emb, val_emb, test_emb

        if verbose:
            print(
                f"TabPFN embeddings shapes: train={train_emb.shape}, val={val_emb.shape}, test={test_emb.shape}"
            )

        return X_train_out, X_val_out, X_test_out

    except ImportError as e:
        if verbose:
            print(
                f"WARNING: TabPFN embeddings not available ({e}). Using raw features."
            )
        return X_train, X_val, X_test
    

def _tabpfn_emb_to_2d(emb: np.ndarray, expected_n: int) -> np.ndarray:
    """
    Convert TabPFN embeddings to (n_samples, emb_dim).

    Handles common shapes:
      - (n_samples, d)
      - (k, n_samples, d)  -> average over k
      - (n_samples, k, d)  -> average over k
      - (1, n_samples, d)  -> squeeze/average over axis 0 (your case)
    """
    emb = np.asarray(emb)

    # Already 2D
    if emb.ndim == 2:
        if emb.shape[0] != expected_n:
            raise ValueError(f"2D emb has wrong n: {emb.shape}, expected first dim {expected_n}")
        return emb

    if emb.ndim == 3:
        a, b, d = emb.shape

        if b == expected_n:
            return emb.mean(axis=0)  # (n_samples, d)

        if a == expected_n:
            return emb.mean(axis=1)  # (n_samples, d)

        if a == 1:
            return _tabpfn_emb_to_2d(emb[0], expected_n)  # becomes (b, d)
        if b == 1:
            return _tabpfn_emb_to_2d(emb[:, 0, :], expected_n)  # becomes (a, d)

        raise ValueError(f"Unrecognized 3D embedding shape {emb.shape} for expected_n={expected_n}")

    raise ValueError(f"Unrecognized embedding ndim={emb.ndim}, shape={emb.shape}")




class CoxPHFG():

    def __init__(self, penalizer: float = 0.01):
        self.penalizer = penalizer
        self.model: Optional[CoxPHFitter] = None
        self.fitted = False
        self._train_t: Optional[pd.Series] = None
        self._train_e: Optional[pd.Series] = None

    # ---------------- fitting ----------------

    def fit(self, X: pd.DataFrame, t: pd.Series, e: pd.Series) -> "CoxPHFG":
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
        if not self.fitted or self.model is None:
            raise RuntimeError("Call fit() first.")
        times = np.asarray(times, dtype=float)
        if times.ndim == 0:
            times = np.array([float(times)])
        sf = self.model.predict_survival_function(X, times=times)
        return sf.T.values  # [N, len(times)]

    def predict_risk_matrix(self, X: pd.DataFrame, times: np.ndarray) -> np.ndarray:
        S = self.predict_survival(X, times)
        return 1.0 - S

    # ------------- NFG metrics wrappers (single risk) -------------

    def ibs(self,
            X_train: pd.DataFrame, 
            t_train: pd.Series, 
            e_train: pd.Series,
            X_test: pd.DataFrame,  
            t_test: pd.Series,  
            e_test: pd.Series,
            times: np.ndarray, 
            t_eval: Optional[np.ndarray] = None) -> float:
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