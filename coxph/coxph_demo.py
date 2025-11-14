"""
Example: train single-risk Cox on METABRIC, compute IBS and time-dependent c-index
with the exact same NFG metrics.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from coxph_api import CoxPHFG
from datasets import load_dataset

def main():
    X, t, e, cols = load_dataset("METABRIC", normalize=True)
    X = pd.DataFrame(X, columns=cols)
    t = pd.Series(t, name="duration")
    e = pd.Series(e, name="event")

    X_tr, X_te, t_tr, t_te, e_tr, e_te = train_test_split(
        X, t, e, test_size=0.2, random_state=42, stratify=(e > 0)
    )

    model = CoxPHFG(penalizer=0.01).fit(X_tr, t_tr, e_tr)

    lo, hi = np.quantile(t_tr, [0.05, 0.95])
    times = np.linspace(max(1.0, lo), hi, 50)

    ibs_val = model.ibs(X_tr, t_tr, e_tr, X_te, t_te, e_te, times)
    ctd_val = model.c_index_td(X_te, t_te, e_te, times)

    print(f"IBS (val): {ibs_val:.4f}")
    print(f"Time-dependent C-index (val): {ctd_val:.4f}")

if __name__ == "__main__":
    main()
