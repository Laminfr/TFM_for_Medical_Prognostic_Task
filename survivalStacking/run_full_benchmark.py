#!/usr/bin/env python
"""
Full Benchmark Runner for Survival Stacking

Runs comprehensive experiments comparing:
1. SurvStack-Raw: Survival Stacking with raw features + XGBoost
2. SurvStack-Emb: Survival Stacking with raw + TabICL embeddings + XGBoost
2.b SurvStack-TabPFN-Emb: Survival Stacking with raw + TabPFN embeddings + XGBoost
3. SurvStack-TabICL: Survival Stacking with raw features + TabICL classifier
3.b SurvStack-TabPFN: Survival Stacking with raw features + TabPFN classifier

Plus baselines (on non-stacked data):
4. CoxPH: Standard Cox Proportional Hazards
5. XGBoost: XGBoost survival (via deephit/discrete-time)
6. DeepSurv: Neural network survival model

Usage:
    python -m survivalStacking.run_full_benchmark --dataset METABRIC
    python -m survivalStacking.run_full_benchmark --dataset all --cv 5
"""

import os
import sys
import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings
import traceback

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Local imports
from survivalStacking.stacking_model import SurvivalStackingModel
from survivalStacking.evaluation import (
    compute_survival_metrics,
    concordance_index,
    integrated_brier_score,
)

# Output directory
RESULTS_DIR = PROJECT_ROOT / "results" / "survival_stacking"


def load_dataset_for_survival(
    dataset: str, normalize: bool = True
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """Load a survival dataset."""
    from datasets.datasets import load_dataset

    X, T, E, feature_names = load_dataset(dataset, normalize=normalize)
    return X, T, E, list(feature_names)


def apply_tabicl_embeddings(
    X_train: np.ndarray,
    X_val: np.ndarray,
    X_test: np.ndarray,
    E_train: np.ndarray,
    feature_names: List[str],
    mode: str = "deep+raw",
    verbose: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply TabICL embedding extraction."""
    if mode == "raw":
        return X_train, X_val, X_test

    try:
        from datasets.tabicl_embeddings import apply_tabicl_embedding

        use_deep = "deep" in mode
        concat_raw = "+raw" in mode

        X_train_emb, X_val_emb, X_test_emb, _ = apply_tabicl_embedding(
            X_train,
            X_val,
            X_test,
            E_train,
            feature_names=feature_names,
            use_deep_embeddings=use_deep,
            concat_with_raw=concat_raw,
            verbose=verbose,
        )
        return X_train_emb, X_val_emb, X_test_emb

    except ImportError as e:
        if verbose:
            print(f"WARNING: TabICL not available ({e}). Using raw features.")
        return X_train, X_val, X_test


def run_survstack_fold(
    X_train: np.ndarray,
    T_train: np.ndarray,
    E_train: np.ndarray,
    X_val: np.ndarray,
    T_val: np.ndarray,
    E_val: np.ndarray,
    X_test: np.ndarray,
    T_test: np.ndarray,
    E_test: np.ndarray,
    classifier: str = "xgboost",
    n_intervals: int = 20,
    calibrate: bool = True,
    weighting_strategy: str = "adaptive",
    verbose: bool = True,
) -> Tuple[Dict[str, float], "SurvivalStackingModel"]:
    """
    Run survival stacking on a single fold.

    Returns
    -------
    metrics : dict
        Dictionary with c_index_q25, c_index_q50, c_index_q75, ibs
    model : SurvivalStackingModel
        The fitted model
    """

    classifier_params = (
        {
            "n_estimators": 200,
            "max_depth": 6,
            "learning_rate": 0.1,
            "min_child_weight": 5,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
        }
        if classifier == "xgboost"
        else {}
    )

    model = SurvivalStackingModel(
        n_intervals=n_intervals,
        interval_strategy="quantile",
        classifier=classifier,
        classifier_params=classifier_params,
        calibrate=calibrate,
        weighting_strategy=weighting_strategy,
        random_state=42,
    )

    model.fit(
        X_train,
        T_train,
        E_train,
        X_val=X_val,
        T_val=T_val,
        E_val=E_val,
        verbose=verbose,
    )

    # Predict survival curves
    eval_times = model.transformer_.get_interval_times()
    survival_matrix = model.predict_survival(X_test)

    # Compute metrics
    metrics = compute_survival_metrics(
        T_test, E_test, survival_matrix, eval_times, T_train=T_train, E_train=E_train
    )

    return metrics, model


def run_coxph_fold(
    X_train: np.ndarray,
    T_train: np.ndarray,
    E_train: np.ndarray,
    X_test: np.ndarray,
    T_test: np.ndarray,
    E_test: np.ndarray,
    verbose: bool = True,
) -> Dict[str, float]:
    """Run CoxPH baseline."""
    try:
        # from coxph.coxph_api import CoxPHFG
        from survivalStacking.tools import CoxPHFG
        import pandas as pd

        # Convert to DataFrames as expected by CoxPHFG
        feature_cols = [f"x{i}" for i in range(X_train.shape[1])]
        X_train_df = pd.DataFrame(X_train, columns=feature_cols)
        X_test_df = pd.DataFrame(X_test, columns=feature_cols)
        T_train_s = pd.Series(T_train)
        E_train_s = pd.Series(E_train)
        T_test_s = pd.Series(T_test)
        E_test_s = pd.Series(E_test)

        model = CoxPHFG(penalizer=0.01)
        model.fit(X_train_df, T_train_s, E_train_s)

        # Get evaluation times
        eval_times = np.percentile(T_train[E_train > 0], [25, 50, 75])
        all_times = np.linspace(T_train.min() + 1e-6, np.percentile(T_train, 90), 20)

        # Compute C-Index at different quantiles
        metrics = {}
        for i, q in enumerate(["q25", "q50", "q75"]):
            t = eval_times[i]
            # Use the c_index_td method
            c_idx = model.c_index_td(X_test_df, T_test_s, E_test_s, all_times, t=t)
            metrics[f"c_index_{q}"] = c_idx

        # IBS
        ibs = model.ibs(
            X_train_df, T_train_s, E_train_s, X_test_df, T_test_s, E_test_s, all_times
        )
        metrics["ibs"] = ibs

        return metrics

    except Exception as e:
        if verbose:
            print(f"CoxPH failed: {e}")
            traceback.print_exc()
        return {"c_index_q25": 0, "c_index_q50": 0, "c_index_q75": 0, "ibs": 0}


def run_xgboost_survival_fold(
    X_train: np.ndarray,
    T_train: np.ndarray,
    E_train: np.ndarray,
    X_test: np.ndarray,
    T_test: np.ndarray,
    E_test: np.ndarray,
    verbose: bool = True,
) -> Dict[str, float]:
    """Run XGBoost survival baseline (non-stacked, Cox objective)."""
    try:
        from xgb_survival.xgboost_api import XGBoostFG
        import pandas as pd

        model = XGBoostFG(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            min_child_weight=10,
            subsample=0.8,
            colsample_bytree=0.8,
        )
        model.fit(X_train, T_train, E_train)

        # Get evaluation times
        eval_times = np.percentile(T_train[E_train > 0], [25, 50, 75])
        all_times = np.linspace(T_train.min() + 1e-6, np.percentile(T_train, 90), 20)

        metrics = {}
        for i, q in enumerate(["q25", "q50", "q75"]):
            t = eval_times[i]
            c_idx = model.c_index_td(X_test, T_test, E_test, all_times, t=t)
            metrics[f"c_index_{q}"] = c_idx

        # IBS
        ibs = model.ibs(X_train, T_train, E_train, X_test, T_test, E_test, all_times)
        metrics["ibs"] = ibs

        return metrics

    except Exception as e:
        if verbose:
            print(f"XGBoost survival failed: {e}")
            traceback.print_exc()
        return {"c_index_q25": 0, "c_index_q50": 0, "c_index_q75": 0, "ibs": 0}


def run_deepsurv_fold(
    X_train: np.ndarray,
    T_train: np.ndarray,
    E_train: np.ndarray,
    X_val: np.ndarray,
    T_val: np.ndarray,
    E_val: np.ndarray,
    X_test: np.ndarray,
    T_test: np.ndarray,
    E_test: np.ndarray,
    verbose: bool = True,
) -> Dict[str, float]:
    """Run DeepSurv baseline."""
    try:
        from deepsurv.deepsurv_api import DeepSurv

        model = DeepSurv(
            layers=[100, 100], dropout=0.3, lr=1e-3, weight_decay=1e-4, cuda=True
        )
        model.fit(
            X_train,
            T_train,
            E_train,
            val_data=(X_val, T_val, E_val),
            n_iter=500,
            bs=256,
            patience_max=20,
        )

        # Get evaluation times
        eval_times = np.percentile(T_train[E_train > 0], [25, 50, 75])
        all_times = np.linspace(T_train.min() + 1e-6, np.percentile(T_train, 90), 20)

        # Predict survival matrix
        survival_matrix = model.predict_survival(X_test, all_times)

        metrics = {}
        for i, q in enumerate(["q25", "q50", "q75"]):
            t = eval_times[i]
            # Find closest time index
            closest_idx = np.argmin(np.abs(all_times - t))
            # Risk = 1 - survival
            risk = 1.0 - survival_matrix[:, closest_idx]
            metrics[f"c_index_{q}"] = concordance_index(T_test, E_test, risk)

        # IBS
        metrics["ibs"] = integrated_brier_score(
            T_test, E_test, survival_matrix, all_times, T_train=T_train, E_train=E_train
        )

        return metrics

    except Exception as e:
        if verbose:
            print(f"DeepSurv failed: {e}")
            traceback.print_exc()
        return {"c_index_q25": 0, "c_index_q50": 0, "c_index_q75": 0, "ibs": 0}


def run_full_cv(
    X: np.ndarray,
    T: np.ndarray,
    E: np.ndarray,
    feature_names: List[str],
    dataset_name: str = "unknown",
    n_folds: int = 5,
    n_intervals: int = 20,
    calibrate: bool = True,
    weighting_strategy: str = "adaptive",
    run_survstack_raw: bool = True,
    run_survstack_tabicl_emb: bool = True,
    run_survstack_tabpfn_emb: bool = True,
    run_survstack_tabicl: bool = True,
    run_survstack_tabpfn: bool = True,
    run_baseline_coxph: bool = True,
    run_baseline_xgboost: bool = True,
    run_baseline_deepsurv: bool = True,
    save_models: bool = True,
    models_dir: Optional[Path] = None,
    verbose: bool = True,
) -> Dict[str, Dict]:
    """
    Run full cross-validation experiment with all methods.

    Parameters
    ----------
    calibrate : bool, default=True
        Whether to apply isotonic calibration (fixes IBS while keeping C-Index)
    weighting_strategy : str, default='adaptive'
        'adaptive', 'sqrt', 'full', or 'none'
    save_models : bool, default=True
        Whether to save fitted models
    models_dir : Path, optional
        Directory to save models

    Returns
    -------
    results : dict
        Dictionary with results for each method
    """
    print(f"\n{'='*70}")
    print("Running Full CV Benchmark")
    print(f"{'='*70}")
    print(f"  Samples: {len(T)}, Events: {E.sum()} ({100*E.mean():.1f}%)")
    print(f"  Folds: {n_folds}")
    print(f"  Calibration: {calibrate}")
    print(f"  Weighting: {weighting_strategy}")

    if save_models and models_dir is None:
        models_dir = RESULTS_DIR / "models" / dataset_name
    if models_dir:
        models_dir.mkdir(parents=True, exist_ok=True)

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

    # Initialize results storage
    methods = []
    survstack_methods = []  # Track which methods are survstack (for model saving)
    if run_survstack_raw:
        methods.append("survstack_raw")
        survstack_methods.append("survstack_raw")
    if run_survstack_tabicl_emb:
        methods.append("survstack_tabicl_emb")
        survstack_methods.append("survstack_tabicl_emb")
    if run_survstack_tabpfn_emb:
        methods.append("survstack_tabpfn_emb")
        survstack_methods.append("survstack_tabpfn_emb")
    if run_survstack_tabicl:
        methods.append("survstack_tabicl")
        survstack_methods.append("survstack_tabicl")
    if run_survstack_tabpfn:
        methods.append("survstack_tabpfn")
        survstack_methods.append("survstack_tabpfn")
    if run_baseline_coxph:
        methods.append("coxph")
    if run_baseline_xgboost:
        methods.append("xgboost")
    if run_baseline_deepsurv:
        methods.append("deepsurv")

    fold_results = {m: [] for m in methods}

    # Track best models (by C-Index on test set)
    best_models = {m: None for m in survstack_methods}
    best_scores = {m: -1.0 for m in survstack_methods}

    for fold_idx, (train_val_idx, test_idx) in enumerate(skf.split(X, E)):
        print(f"\n{'─'*50}")
        print(f"Fold {fold_idx + 1}/{n_folds}")
        print(f"{'─'*50}")

        # Split data
        X_train_val = X[train_val_idx]
        T_train_val = T[train_val_idx]
        E_train_val = E[train_val_idx]

        X_test = X[test_idx]
        T_test = T[test_idx]
        E_test = E[test_idx]

        # Further split train_val into train and val (80/20)
        n_train_val = len(train_val_idx)
        n_val = int(0.2 * n_train_val)

        np.random.seed(42 + fold_idx)
        val_indices = np.random.choice(n_train_val, n_val, replace=False)
        train_indices = np.array(
            [i for i in range(n_train_val) if i not in val_indices]
        )

        X_train = X_train_val[train_indices]
        T_train = T_train_val[train_indices]
        E_train = E_train_val[train_indices]

        X_val = X_train_val[val_indices]
        T_val = T_train_val[val_indices]
        E_val = E_train_val[val_indices]

        print(f"  Train: {len(T_train)}, Val: {len(T_val)}, Test: {len(T_test)}")

        # 1. SurvStack with Raw features + XGBoost
        if run_survstack_raw:
            print("\n  [1/6] SurvStack-Raw (XGBoost + calibration)...")
            try:
                metrics, model = run_survstack_fold(
                    X_train,
                    T_train,
                    E_train,
                    X_val,
                    T_val,
                    E_val,
                    X_test,
                    T_test,
                    E_test,
                    classifier="xgboost",
                    n_intervals=n_intervals,
                    calibrate=calibrate,
                    weighting_strategy=weighting_strategy,
                    verbose=(fold_idx == 0 and verbose),
                )
                fold_results["survstack_raw"].append(metrics)
                print(
                    f"      C-Index: {metrics['c_index_q50']:.4f}, IBS: {metrics['ibs']:.4f}"
                )

                # Track best model
                if (
                    save_models
                    and metrics["c_index_q50"] > best_scores["survstack_raw"]
                ):
                    best_scores["survstack_raw"] = metrics["c_index_q50"]
                    best_models["survstack_raw"] = model
            except Exception as e:
                print(f"      FAILED: {e}")
                traceback.print_exc()
                fold_results["survstack_raw"].append(
                    {"c_index_q25": 0, "c_index_q50": 0, "c_index_q75": 0, "ibs": 0}
                )

        # 2. SurvStack with Raw + Embeddings + XGBoost
        if run_survstack_tabicl_emb:
            print(
                "\n  [2/6] SurvStack-TabICLEmb (XGBoost + TabICL embeddings + calibration)..."
            )
            try:
                # Apply embeddings
                X_train_emb, X_val_emb, X_test_emb = apply_tabicl_embeddings(
                    X_train,
                    X_val,
                    X_test,
                    E_train,
                    feature_names,
                    mode="deep+raw",
                    verbose=(fold_idx == 0),
                )
                metrics, model = run_survstack_fold(
                    X_train_emb,
                    T_train,
                    E_train,
                    X_val_emb,
                    T_val,
                    E_val,
                    X_test_emb,
                    T_test,
                    E_test,
                    classifier="xgboost",
                    n_intervals=n_intervals,
                    calibrate=calibrate,
                    weighting_strategy=weighting_strategy,
                    verbose=(fold_idx == 0 and verbose),
                )
                fold_results["survstack_tabicl_emb"].append(metrics)
                print(
                    f"      C-Index: {metrics['c_index_q50']:.4f}, IBS: {metrics['ibs']:.4f}"
                )

                # Track best model
                if (
                    save_models
                    and metrics["c_index_q50"] > best_scores["survstack_tabicl_emb"]
                ):
                    best_scores["survstack_tabicl_emb"] = metrics["c_index_q50"]
                    best_models["survstack_tabicl_emb"] = model
            except Exception as e:
                print(f"      FAILED: {e}")
                fold_results["survstack_tabicl_emb"].append(
                    {"c_index_q25": 0, "c_index_q50": 0, "c_index_q75": 0, "ibs": 0}
                )

        # 2b. SurvStack with Raw + TabPFN Embeddings + XGBoost
        if run_survstack_tabpfn_emb:
            print(
                "\n  [2b/6] SurvStack-TabPFN-Emb (XGBoost + TabPFN embeddings + calibration)..."
            )
            try:
                from survivalStacking.tools import apply_tabpfn_embeddings

                X_train_pfn, X_val_pfn, X_test_pfn = apply_tabpfn_embeddings(
                    X_train,
                    X_val,
                    X_test,
                    E_train,
                    mode="deep+raw",
                    verbose=(fold_idx == 0),
                    n_estimators=1,
                    n_fold=n_folds,
                )
                metrics, model = run_survstack_fold(
                    X_train_pfn,
                    T_train,
                    E_train,
                    X_val_pfn,
                    T_val,
                    E_val,
                    X_test_pfn,
                    T_test,
                    E_test,
                    classifier="xgboost",
                    n_intervals=n_intervals,
                    calibrate=calibrate,
                    weighting_strategy=weighting_strategy,
                    verbose=(fold_idx == 0 and verbose),
                )
                fold_results["survstack_tabpfn_emb"].append(metrics)
                print(
                    f"      C-Index: {metrics['c_index_q50']:.4f}, IBS: {metrics['ibs']:.4f}"
                )

                # Track best model
                if (
                    save_models
                    and metrics["c_index_q50"] > best_scores["survstack_tabpfn_emb"]
                ):
                    best_scores["survstack_tabpfn_emb"] = metrics["c_index_q50"]
                    best_models["survstack_tabpfn_emb"] = model
            except Exception as e:
                print(f"      FAILED: {e}")
                traceback.print_exc()
                fold_results["survstack_tabpfn_emb"].append(
                    {"c_index_q25": 0, "c_index_q50": 0, "c_index_q75": 0, "ibs": 0}
                )

        # 3. SurvStack with Raw + TabICL classifier
        if run_survstack_tabicl:
            print("\n  [3/6] SurvStack-TabICL (TabICL classifier + calibration)...")
            try:
                metrics, model = run_survstack_fold(
                    X_train,
                    T_train,
                    E_train,
                    X_val,
                    T_val,
                    E_val,
                    X_test,
                    T_test,
                    E_test,
                    classifier="tabicl",
                    n_intervals=n_intervals,
                    calibrate=calibrate,
                    weighting_strategy="none",  # TabICL handles imbalance internally
                    verbose=(fold_idx == 0 and verbose),
                )
                fold_results["survstack_tabicl"].append(metrics)
                print(
                    f"      C-Index: {metrics['c_index_q50']:.4f}, IBS: {metrics['ibs']:.4f}"
                )

                # Track best model
                if (
                    save_models
                    and metrics["c_index_q50"] > best_scores["survstack_tabicl"]
                ):
                    best_scores["survstack_tabicl"] = metrics["c_index_q50"]
                    best_models["survstack_tabicl"] = model
            except Exception as e:
                print(f"      FAILED: {e}")
                fold_results["survstack_tabicl"].append(
                    {"c_index_q25": 0, "c_index_q50": 0, "c_index_q75": 0, "ibs": 0}
                )

        # 3.b SurvStack with Raw + TabPFN classifier
        if run_survstack_tabpfn:
            print("\n  [3b/6] SurvStack-TabPFN (TabPFN classifier + calibration)...")
            try:
                # You can pass TabPFN params via classifier_params_override if needed.
                # Example: {'device': 'cuda'} or any params supported by TabPFNClassifier.
                metrics, model = run_survstack_fold(
                    X_train,
                    T_train,
                    E_train,
                    X_val,
                    T_val,
                    E_val,
                    X_test,
                    T_test,
                    E_test,
                    classifier="tabpfn",
                    n_intervals=n_intervals,
                    calibrate=calibrate,
                    weighting_strategy="none",
                    verbose=(fold_idx == 0 and verbose),
                )
                fold_results["survstack_tabpfn"].append(metrics)
                print(
                    f"      C-Index: {metrics['c_index_q50']:.4f}, IBS: {metrics['ibs']:.4f}"
                )

                if (
                    save_models
                    and metrics["c_index_q50"] > best_scores["survstack_tabpfn"]
                ):
                    best_scores["survstack_tabpfn"] = metrics["c_index_q50"]
                    best_models["survstack_tabpfn"] = model
            except Exception as e:
                print(f"      FAILED: {e}")
                fold_results["survstack_tabpfn"].append(
                    {"c_index_q25": 0, "c_index_q50": 0, "c_index_q75": 0, "ibs": 0}
                )

        # 4. CoxPH baseline
        if run_baseline_coxph:
            print("\n  [4/6] CoxPH baseline...")
            try:
                metrics = run_coxph_fold(
                    X_train,
                    T_train,
                    E_train,
                    X_test,
                    T_test,
                    E_test,
                    verbose=(fold_idx == 0 and verbose),
                )
                fold_results["coxph"].append(metrics)
                print(
                    f"      C-Index: {metrics['c_index_q50']:.4f}, IBS: {metrics['ibs']:.4f}"
                )
            except Exception as e:
                print(f"      FAILED: {e}")
                fold_results["coxph"].append(
                    {"c_index_q25": 0, "c_index_q50": 0, "c_index_q75": 0, "ibs": 0}
                )

        # 5. XGBoost survival baseline
        if run_baseline_xgboost:
            print("\n  [5/6] XGBoost survival baseline...")
            try:
                metrics = run_xgboost_survival_fold(
                    X_train,
                    T_train,
                    E_train,
                    X_test,
                    T_test,
                    E_test,
                    verbose=(fold_idx == 0 and verbose),
                )
                fold_results["xgboost"].append(metrics)
                print(
                    f"      C-Index: {metrics['c_index_q50']:.4f}, IBS: {metrics['ibs']:.4f}"
                )
            except Exception as e:
                print(f"      FAILED: {e}")
                fold_results["xgboost"].append(
                    {"c_index_q25": 0, "c_index_q50": 0, "c_index_q75": 0, "ibs": 0}
                )

        # 6. DeepSurv baseline
        if run_baseline_deepsurv:
            print("\n  [6/6] DeepSurv baseline...")
            try:
                metrics = run_deepsurv_fold(
                    X_train,
                    T_train,
                    E_train,
                    X_val,
                    T_val,
                    E_val,
                    X_test,
                    T_test,
                    E_test,
                    verbose=(fold_idx == 0 and verbose),
                )
                fold_results["deepsurv"].append(metrics)
                print(
                    f"      C-Index: {metrics['c_index_q50']:.4f}, IBS: {metrics['ibs']:.4f}"
                )
            except Exception as e:
                print(f"      FAILED: {e}")
                fold_results["deepsurv"].append(
                    {"c_index_q25": 0, "c_index_q50": 0, "c_index_q75": 0, "ibs": 0}
                )

    # Aggregate results
    results = {}
    for method in methods:
        if fold_results[method]:
            metric_keys = fold_results[method][0].keys()
            results[method] = {
                "mean": {
                    k: np.mean([f[k] for f in fold_results[method]])
                    for k in metric_keys
                },
                "std": {
                    k: np.std([f[k] for f in fold_results[method]]) for k in metric_keys
                },
                "fold_results": fold_results[method],
            }

    # Save only the best models (one per method)
    if save_models and models_dir:
        print(f"\n{'─'*50}")
        print("Saving best models...")
        for method in survstack_methods:
            if best_models[method] is not None:
                model_path = models_dir / f"{method}_best.pkl"
                best_models[method].save(model_path)
                print(f"  {method}: C-Index={best_scores[method]:.4f} -> {model_path}")

    return results


def print_results_table(results: Dict, dataset: str):
    """Print formatted results table."""
    print(f"\n{'='*80}")
    print(f"RESULTS SUMMARY: {dataset}")
    print(f"{'='*80}")

    print(f"\n{'Method':<25} {'C-Index (q50)':<18} {'IBS':<18}")
    print(f"{'-'*60}")

    method_names = {
        "survstack_raw": "SurvStack-Raw",
        "survstack_tabicl_emb": "SurvStack-TabICL-Emb",
        "survstack_tabpfn_emb": "SurvStack-TabPFN-Emb",
        "survstack_tabicl": "SurvStack-TabICL",
        "survstack_tabpfn": "SurvStack-TabPFN",
        "coxph": "CoxPH",
        "xgboost": "XGBoost",
        "deepsurv": "DeepSurv",
    }

    for method, data in results.items():
        name = method_names.get(method, method)
        c_idx = data["mean"]["c_index_q50"]
        c_std = data["std"]["c_index_q50"]
        ibs = data["mean"]["ibs"]
        ibs_std = data["std"]["ibs"]
        print(f"{name:<25} {c_idx:.4f} ± {c_std:.4f}     {ibs:.4f} ± {ibs_std:.4f}")

    print(f"{'='*80}")


def main():
    parser = argparse.ArgumentParser(description="Full Survival Stacking Benchmark")

    parser.add_argument(
        "--dataset",
        type=str,
        default="METABRIC",
        choices=["METABRIC", "PBC", "SUPPORT", "all"],
        help="Dataset to use",
    )

    parser.add_argument("--cv", type=int, default=5, help="Number of CV folds")

    parser.add_argument(
        "--n_intervals", type=int, default=20, help="Number of time intervals"
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/survival_stacking",
        help="Output directory",
    )

    parser.add_argument(
        "--skip-tabicl", action="store_true", help="Skip TabICL-based methods (faster)"
    )
    parser.add_argument(
        "--skip-tabpfn",
        action="store_true",
        help="Skip TabPFN embedding method (if not installed / faster)",
    )
    parser.add_argument(
        "--no-calibrate", action="store_true", help="Disable isotonic calibration"
    )
    parser.add_argument(
        "--weighting",
        type=str,
        default="adaptive",
        choices=["adaptive", "sqrt", "full", "none"],
        help="Class weighting strategy",
    )
    parser.add_argument(
        "--no-save-models", action="store_true", help="Do not save fitted models"
    )
    parser.add_argument("--verbose", action="store_true", default=True)

    args = parser.parse_args()

    # Create output directory
    output_dir = PROJECT_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Determine datasets
    if args.dataset == "all":
        datasets = ["METABRIC", "PBC", "SUPPORT"]
    else:
        datasets = [args.dataset]

    all_results = {}

    for dataset in datasets:
        print(f"\n{'#'*80}")
        print(f"# Dataset: {dataset}")
        print(f"{'#'*80}")

        start_time = time.time()

        # Load data
        print(f"\nLoading {dataset}...")
        X, T, E, feature_names = load_dataset_for_survival(dataset)
        print(f"  Samples: {len(T)}, Features: {X.shape[1]}")
        print(f"  Events: {E.sum()} ({100*E.mean():.1f}%)")

        # Run experiments
        results = run_full_cv(
            X,
            T,
            E,
            feature_names,
            dataset_name=dataset,
            n_folds=args.cv,
            n_intervals=args.n_intervals,
            run_survstack_raw=True,
            run_survstack_tabicl_emb=not args.skip_tabicl,
            run_survstack_tabpfn_emb=not args.skip_tabpfn,
            run_survstack_tabicl=not args.skip_tabicl,
            run_survstack_tabpfn=not args.skip_tabpfn,
            run_baseline_coxph=True,
            run_baseline_xgboost=True,
            run_baseline_deepsurv=True,
            calibrate=not args.no_calibrate,
            weighting_strategy=args.weighting,
            save_models=not args.no_save_models,
            models_dir=output_dir / "models" / dataset,
            verbose=args.verbose,
        )

        elapsed = time.time() - start_time

        # Print results
        print_results_table(results, dataset)
        print(f"\nTime: {elapsed/60:.1f} minutes")

        # Store results
        all_results[dataset] = results

        # Save to JSON
        save_data = {
            "dataset": dataset,
            "n_folds": args.cv,
            "n_intervals": args.n_intervals,
            "results": {
                method: {
                    "mean": data["mean"],
                    "std": data["std"],
                    "fold_results": data["fold_results"],
                }
                for method, data in results.items()
            },
            "elapsed_time": elapsed,
            "timestamp": datetime.now().isoformat(),
        }

        result_file = output_dir / f"{dataset}_full_benchmark_{args.cv}fold.json"
        with open(result_file, "w") as f:
            json.dump(save_data, f, indent=2, default=str)
        print(f"Saved: {result_file}")

    # Print final summary
    if len(datasets) > 1:
        print(f"\n{'#'*80}")
        print("# FINAL SUMMARY")
        print(f"{'#'*80}")
        for dataset, results in all_results.items():
            print_results_table(results, dataset)

    return all_results


if __name__ == "__main__":
    main()
