"""
Experiment with TARTE embedding extraction.
Compare all baseline models by running them with crossvalidation and collecting results.
"""
import warnings
warnings.filterwarnings("ignore", message="TypedStorage is deprecated")
import os
import sys
sys.path.append("/")

from datasets.data_loader import load_and_preprocess_data
from experiments.experiment import CoxExperiment
from tfm.TARTE.legacy.legacy_embedding_strategies import get_embeddings_tarte_cross, get_embeddings_dummy_tarte_cross, get_embeddings_combination_tarte_cross
from images.utilities import plot_results_relative, plot_results_absolute
import numpy as np
import pandas as pd
import time
from metrics.calibration import integrated_brier_score


def concordance_index_from_risk_scores(e, t, risk_scores, tied_tol=1e-8):
    """
    Compute C-index directly from risk scores (for Cox-like models).
    Higher risk score should correspond to higher risk (shorter survival).
    """
    event = e.values.astype(bool) if hasattr(e, 'values') else e.astype(bool)
    t = t.values if hasattr(t, 'values') else t
    n_events = event.sum()

    if n_events == 0:
        return np.nan

    concordant = 0
    permissible = 0

    for i in range(len(t)):
        if not event[i]:
            continue

        # Compare with all samples at risk at time t[i]
        at_risk = t > t[i]

        # Higher risk score means higher risk (shorter time to event)
        concordant += (risk_scores[at_risk] < risk_scores[i]).sum()
        concordant += 0.5 * (np.abs(risk_scores[at_risk] - risk_scores[i]) <= tied_tol).sum()
        permissible += at_risk.sum()

    if permissible == 0:
        return np.nan

    return concordant / permissible

def evaluate_cox_cv(e_train, t_train, e_val, t_val, risk_train=None, risk_val=None, surv_probs_val=None, times=None):
    """
    Evaluate a fitted CoxExperiment on one fold.

    Returns:
        results: 'c_index_train', 'c_index_val', 'ibs_val'
    """
    # --- C-index ---
    c_index_train = concordance_index_from_risk_scores(e_train, t_train, risk_train)
    c_index_val = concordance_index_from_risk_scores(e_val, t_val, risk_val)

    # --- IBS ---
    if surv_probs_val is None or times is None:
        ibs_val = np.nan
    else:
        # Filter validation times to be within training max
        max_time_train = t_train.max()
        valid_mask = t_val.values < max_time_train
        t_val_filtered = t_val.values[valid_mask]
        e_val_filtered = e_val.values[valid_mask]
        surv_probs_filtered = surv_probs_val[valid_mask, :]  # subset rows

        if len(t_val_filtered) > 0:
            risk_pred_val = 1 - surv_probs_filtered
            km = (e_train.values.astype(int), t_train.values.astype(float))
            ibs_val, _ = integrated_brier_score(
                e_val_filtered.astype(int),
                t_val_filtered,
                risk_pred_val,
                times,
                t_eval=times,
                km=km,
                competing_risk=1
            )
        else:
            ibs_val = np.nan

    return c_index_train, c_index_val, ibs_val

def baselines_crossevaluate_embeddings(dataset='METABRIC', normalize=True, test_size=0.2, random_state=42, embeddings_flag=None):
    """Run all baselines on raw data and embeddings and collect results."""
    print("= " *70)
    print(f"CROSS EVALUATION BASELINE COMPARISON ON {dataset} DATASET")
    print("= " *70)

    # Load data once
    X, t, e = load_and_preprocess_data(
        dataset=dataset,
        normalize=normalize,
        test_size=test_size,
        random_state=random_state,
        as_sksurv_y=False,
        cross_val=True
    )
    X_sk, y_sk = load_and_preprocess_data(
        dataset=dataset,
        normalize=normalize,
        test_size=test_size,
        random_state=random_state,
        as_sksurv_y=True,
        cross_val=True
    )

    print(f"Samples: {len(X)}")
    print(f"Features: {X.shape[1]}")
    print(f"Event rate: {e.mean():.2%}")

    if embeddings_flag=="emb":
        print(f"\n>>> TARTE Embeddings with no target\n")
        print("Extract TARTE embeddings ...")
        X = get_embeddings_tarte_cross(X)
        X_sk = X
        print("Run Baselines on embeddings")
    elif embeddings_flag=="dummy":
        print(f"\n>>> TARTE Embeddings with dummy target\n")
        print("Extract TARTE embeddings with dummy y ...")
        X = get_embeddings_dummy_tarte_cross(X)
        X_sk = X
        print("Run Baselines on embeddings")
    elif embeddings_flag == "combi":
        print(f"\n>>> TARTE Embeddings for time and event combined\n")
        print("Extract TARTE embeddings with time and event combined...")
        X = get_embeddings_combination_tarte_cross(X, t, e)
        X_sk = X
        print("Run Baselines on embeddings")
    else:
        print(f"\n>>> Use raw data for baseline predictions\n")

    results = []

    # --------------------------
    # 1. Cox Proportional Hazards with cross-validation
    # --------------------------
    print("Cox PH (cross-validation)")
    start_time = time.time()
    try:
        # convert X to plain DataFrame with integer columns
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(np.asarray(X))
        else:
            X.columns = range(X.shape[1])
        if not isinstance(X_sk, pd.DataFrame):
            X_sk = pd.DataFrame(np.asarray(X_sk))
        else:
            X_sk.columns = range(X_sk.shape[1])

        # Define hyperparameter grid
        hyper_grid = [
            {"penalizer": [0.01, 0.1, 1.0]}
        ]

        # Create CoxExperiment instance
        cox_exp = CoxExperiment(hyper_grid=hyper_grid, k=5, random_seed=42, path="results/cox_exp", save=False)

        # Train: this runs k-fold CV and stores best model per fold in cox_exp.best_model
        predictions = cox_exp.train(X, t, e, cause_specific=False)

        # Evaluate metrics fold-wise
        times = np.linspace(t.min(), t.max(), 100)
        c_train_indices = []
        c_test_indices = []
        ibs_scores = []

        for fold, model in cox_exp.best_model.items():
            # Get fold indices
            fold_idx = cox_exp.fold_assignment == fold
            train_idx = cox_exp.fold_assignment != fold

            X_train_fold =  X.loc[train_idx].copy()
            t_train_fold = t.loc[train_idx].copy()
            e_train_fold = e.loc[train_idx].copy()

            X_test_fold = X.loc[fold_idx].copy()
            t_test_fold = t.loc[fold_idx].copy()
            e_test_fold = e.loc[fold_idx].copy()

            # --- Compute predictions ---
            risk_train = model.model.predict_partial_hazard(X_train_fold).values.flatten()
            risk_val = model.model.predict_partial_hazard(X_test_fold).values.flatten()
            surv = model.model.predict_survival_function(X_test_fold, times=times).T.values

            # --- Compute metrics ---
            c_idx_train, c_idx_val, ibs_val = evaluate_cox_cv(
                e_train=e_train_fold, t_train=t_train_fold,
                e_val=e_test_fold, t_val=t_test_fold,
                risk_train=risk_train, risk_val=risk_val,
                surv_probs_val=surv, times=times
            )
            c_train_indices.append(c_idx_train)
            c_test_indices.append(c_idx_val)
            ibs_scores.append(ibs_val)

        train_time = time.time() - start_time

        results.append({
            "Model": "Cox PH (CV)",
            "C-index (train)": np.mean(c_train_indices),
            "C-index (val)": np.mean(c_test_indices),
            "IBS (val)": np.mean(ibs_scores),
            "Time (s)": train_time
        })

    except Exception as e:
        print(f"✗ Failed: {e}")
        import traceback
        traceback.print_exc()
        results.append({
            "Model": "Cox PH (CV)",
            "C-index (train)": "Error",
            "C-index (val)": "Error",
            "IBS (val)": "Error",
            "Time (s)": "-"
        })
    return results


if __name__ == "__main__":
    raw_results = baselines_crossevaluate_embeddings(embeddings_flag=None, dataset='PBC')
    emb_results = baselines_crossevaluate_embeddings(embeddings_flag="emb", dataset='PBC')
    dummy_results = baselines_crossevaluate_embeddings(embeddings_flag="dummy", dataset='PBC')
    combi_results = baselines_crossevaluate_embeddings(embeddings_flag="combi", dataset='PBC')

    # Combine results with a method column
    def combine_results(results_dicts, method_name):
        df = pd.DataFrame(results_dicts)
        df["Method"] = method_name
        return df

    all_results = pd.concat([
        combine_results(raw_results, "Raw"),
        combine_results(emb_results, "TARTE Embeddings"),
        combine_results(dummy_results, "TARTE Embeddings with dummy y"),
        combine_results(combi_results, "TARTE Embeddings with time and event combined")
    ])

    # Define metrics
    metrics = ["C-index (train)", "C-index (val)", "IBS (val)", "Time (s)"]

    # extract raw rows
    df_raw = all_results[all_results["Method"] == "Raw"].set_index("Model")

    # overwrite column in-place
    for metric in metrics:
        all_results[f"{metric} to Baseline"] = all_results.apply(
            lambda row: row[metric] - df_raw.loc[row["Model"], metric]
            if row["Method"] != "Raw" else 0.0,
            axis=1
        )

    plot_results_absolute(all_results)
    plot_results_relative(all_results)