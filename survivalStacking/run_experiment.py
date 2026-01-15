#!/usr/bin/env python
"""
Survival Stacking Experiment Runner

Benchmarks Survival Stacking with TFM embeddings (TabICl, TARTE) against DeepSurv and CoxPH
baselines on METABRIC and PBC datasets.

Target: C-Index > 0.8676 (PBC SOTA), IBS < 0.12

Usage:
    python -m survivalStacking.run_experiment --dataset METABRIC --model TARTE --mode deep+raw
    python -m survivalStacking.run_experiment --dataset PBC --model TARTE --mode deep --cv 5
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

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Local imports
from survivalStacking.stacking_model import SurvivalStackingModel, SurvivalStackingCV
from survivalStacking.evaluation import (
    compute_survival_metrics, 
    concordance_index, 
    integrated_brier_score,
    print_metrics
)


def load_dataset_for_survival(
    dataset: str,
    normalize: bool = True
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """
    Load a survival dataset.
    
    Parameters
    ----------
    dataset : str
        Dataset name: 'METABRIC', 'PBC', 'SUPPORT', 'GBSG'
    normalize : bool
        Whether to standardize features
        
    Returns
    -------
    X, T, E, feature_names
    """
    # Try using our datasets module
    try:
        from datasets.datasets import load_dataset
        X, T, E, feature_names = load_dataset(dataset, normalize=normalize)
        return X, T, E, list(feature_names)
    except ImportError:
        pass
    
    # Fallback: direct loading
    if dataset == 'METABRIC':
        from pycox import datasets
        from sklearn.preprocessing import StandardScaler
        
        df = datasets.metabric.read_df()
        df = df.rename(columns={
            'x0': 'MKI67', 'x1': 'EGFR', 'x2': 'PGR', 'x3': 'ERBB2',
            'x4': 'Hormone', 'x5': 'Radiotherapy', 'x6': 'Chemotherapy',
            'x7': 'ER-positive', 'x8': 'Age at diagnosis'
        })
        df['duration'] += 1e-8
        
        covariates = df.drop(['duration', 'event'], axis='columns')
        feature_names = list(covariates.columns)
        
        if normalize:
            X = StandardScaler().fit_transform(covariates.values)
        else:
            X = covariates.values
            
        T = df['duration'].values
        E = df['event'].values.astype(int)
        
        return X, T, E, feature_names
        
    elif dataset == 'PBC':
        from auton_survival.datasets import load_dataset as load_dsm
            
        from sklearn.preprocessing import StandardScaler
        
        X_raw, T, E = load_dsm('PBC')
        feature_names = [f'feat_{i}' for i in range(X_raw.shape[1])]
        
        if normalize:
            X = StandardScaler().fit_transform(X_raw)
        else:
            X = X_raw
            
        return X, T.astype(float), E.astype(int), feature_names
    
    elif dataset == 'SUPPORT':
        from auton_survival.datasets import load_dataset as load_dsm
            
        from sklearn.preprocessing import StandardScaler, OrdinalEncoder
        from sklearn.impute import SimpleImputer
        
        outcomes, features = load_dsm('SUPPORT')
        T = outcomes['time'].values.astype(float)
        E = outcomes['event'].values.astype(int)
        feature_names = features.columns.tolist()
        
        # Encode categorical features and impute missing values
        features_encoded = features.copy()
        for col in features_encoded.columns:
            if features_encoded[col].dtype == 'object':
                features_encoded[col] = features_encoded[col].fillna('missing')
                features_encoded[col] = OrdinalEncoder().fit_transform(
                    features_encoded[[col]]
                ).flatten()
        
        # Impute remaining numerical NaNs with median
        imputer = SimpleImputer(strategy='median')
        features_imputed = imputer.fit_transform(features_encoded.values)
        
        if normalize:
            X = StandardScaler().fit_transform(features_imputed)
        else:
            X = features_imputed
            
        return X, T, E, feature_names
        
    else:
        raise ValueError(f"Unknown dataset: {dataset}")


def apply_tabicl_embeddings(
    X_train: np.ndarray,
    X_val: np.ndarray,
    X_test: np.ndarray,
    E_train: np.ndarray,
    feature_names: List[str],
    mode: str = 'deep+raw',
    verbose: bool = True
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Apply TabICL embedding extraction.
    
    Parameters
    ----------
    mode : str
        'raw' - original features only
        'deep' - 512D TabICL embeddings only
        'deep+raw' - embeddings concatenated with original features
    """
    if mode == 'raw':
        return X_train, X_val, X_test
    
    try:
        from datasets.tabicl_embeddings import apply_tabicl_embedding
        
        use_deep = 'deep' in mode
        concat_raw = '+raw' in mode
        
        X_train_emb, X_val_emb, X_test_emb, _ = apply_tabicl_embedding(
            X_train, X_val, X_test,
            E_train,
            feature_names=feature_names,
            use_deep_embeddings=use_deep,
            concat_with_raw=concat_raw,
            verbose=verbose
        )
        
        return X_train_emb, X_val_emb, X_test_emb
        
    except ImportError as e:
        if verbose:
            print(f"WARNING: TabICL not available ({e}). Using raw features.")
        return X_train, X_val, X_test

def apply_tabpfn_embeddings():
    pass

def apply_tarte_embeddings(
    X_train: np.ndarray,
    X_val: np.ndarray,
    X_test: np.ndarray,
    feature_names: List[str],
    mode: str = 'deep+raw',
    verbose: bool = True
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
        Apply TARTE embedding extraction.

        Parameters
        ----------
        mode : str
            'raw' - original features only
            'deep' - TARTE embeddings only
            'deep+raw' - embeddings concatenated with original features
        """
    if mode == 'raw':
        return X_train, X_val, X_test

    try:
        from datasets.tarte_embeddings import apply_tarte_embedding

        use_deep = 'deep' in mode
        concat_raw = '+raw' in mode

        X_train_emb, X_val_emb, X_test_emb, _ = apply_tarte_embedding(
            X_train=X_train,
            X_val=X_val,
            X_test=X_test,
            feature_names=feature_names,
            use_deep_embeddings=use_deep,
            concat_with_raw=concat_raw,
            verbose=verbose
        )

        return X_train_emb, X_val_emb, X_test_emb

    except ImportError as e:
        if verbose:
            print(f"WARNING: TARTE not available ({e}). Using raw features.")
        return X_train, X_val, X_test

def run_single_fold(
    X_train: np.ndarray,
    T_train: np.ndarray,
    E_train: np.ndarray,
    X_val: np.ndarray,
    T_val: np.ndarray,
    E_val: np.ndarray,
    X_test: np.ndarray,
    T_test: np.ndarray,
    E_test: np.ndarray,
    config: Dict,
    fold_idx: int,
    verbose: bool = True
) -> Dict[str, float]:
    """
    Run survival stacking on a single fold.
    
    Returns
    -------
    metrics : dict
        Dictionary with c_index_q25, c_index_q50, c_index_q75, ibs
    """
    if verbose:
        print(f"\n{'='*50}")
        print(f"Fold {fold_idx + 1}")
        print(f"  Model: {config.get('classifier', 'xgboost')}")
        print(f"  Train: {len(T_train)}, Val: {len(T_val)}, Test: {len(T_test)}")
        print(f"  Event rate: {100*E_train.mean():.1f}%")
    
    # Configure model
    model_params = {
        'n_intervals': config.get('n_intervals', 20),
        'interval_strategy': config.get('interval_strategy', 'quantile'),
        'classifier': config.get('classifier', 'xgboost'),
        'random_state': config.get('random_state', 42)
    }

    if config.get('classifier', 'xgboost') == 'xgboost':
        # XGBoost hyperparameters
        classifier_params = {
            'n_estimators': config.get('n_estimators', 200),
            'max_depth': config.get('max_depth', 6),
            'learning_rate': config.get('learning_rate', 0.1),
            'min_child_weight': config.get('min_child_weight', 5),
            'subsample': config.get('subsample', 0.8),
            'colsample_bytree': config.get('colsample_bytree', 0.8),
        }
    elif config.get('classifier', 'xgboost') == 'tabicl':
        # tabicl classifier parameters
        classifier_params = {
            'n_estimators': config.get('tabicl_n_estimators', 4),
            'device': config.get('device', 'cuda'),
            'max_context_samples': config.get('max_context_samples', 2000),
            'verbose': False
        }
    else:
        raise ValueError(f"Unknown model: {config.get('classifier', 'xgboost')}")

    model_params['classifier_params'] = classifier_params
    
    # Train model
    model = SurvivalStackingModel(**model_params)
    model.fit(
        X_train, T_train, E_train,
        X_val=X_val, T_val=T_val, E_val=E_val,
        verbose=verbose
    )
    
    # Predict survival curves for test set
    eval_times = model.transformer_.get_interval_times()
    survival_matrix = model.predict_survival(X_test)
    
    # Compute metrics
    metrics = compute_survival_metrics(
        T_test, E_test,
        survival_matrix,
        eval_times,
        T_train=T_train,
        E_train=E_train
    )
    
    if verbose:
        print(f"\nFold {fold_idx + 1} Results:")
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}")
    
    return metrics


def run_cv_experiment(
    X: np.ndarray,
    T: np.ndarray,
    E: np.ndarray,
    feature_names: List[str],
    config: Dict,
    n_folds: int = 5,
    model: str = 'TabICL',
    tfm_mode: str = 'deep+raw',
    verbose: bool = True
) -> Tuple[Dict[str, float], Dict[str, float], List[Dict]]:
    """
    Run full cross-validation experiment.
    
    Returns
    -------
    mean_metrics : dict
        Mean metrics across folds
    std_metrics : dict
        Std metrics across folds
    fold_results : list
        Per-fold metrics
    """
    if verbose:
        print("\n" + "="*60)
        print(f"Cross-Validation Experiment")
        print(f"  Model: {model}")
        print(f"  Folds: {n_folds}")
        print(f"  TFM mode: {tfm_mode}")
        print(f"  Samples: {len(T)}, Events: {E.sum()}")
        print("="*60)
    
    # Stratified K-Fold
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=config.get('random_state', 42))
    
    fold_results = []
    
    for fold_idx, (train_val_idx, test_idx) in enumerate(skf.split(X, E)):
        # Split into train+val and test
        X_train_val = X[train_val_idx]
        T_train_val = T[train_val_idx]
        E_train_val = E[train_val_idx]
        
        X_test_fold = X[test_idx]
        T_test_fold = T[test_idx]
        E_test_fold = E[test_idx]
        
        # Further split train_val into train and val (80/20)
        n_train_val = len(train_val_idx)
        n_val = int(0.2 * n_train_val)
        
        # Simple random split (stratified)
        np.random.seed(config.get('random_state', 42) + fold_idx)
        val_indices = np.random.choice(n_train_val, n_val, replace=False)
        train_indices = np.array([i for i in range(n_train_val) if i not in val_indices])
        
        X_train = X_train_val[train_indices]
        T_train = T_train_val[train_indices]
        E_train = E_train_val[train_indices]
        
        X_val = X_train_val[val_indices]
        T_val = T_train_val[val_indices]
        E_val = E_train_val[val_indices]

        # get tfm embeddings
        if model == "TabICL":
            # Apply TabICL embeddings
            X_train_emb, X_val_emb, X_test_emb = apply_tabicl_embeddings(
                X_train, X_val, X_test_fold,
                E_train, feature_names,
                mode=tfm_mode,
                verbose=(verbose and fold_idx == 0)  # Only verbose on first fold
            )
        elif model == "TabPFN":
            # Apply TabPFN embeddings
            X_train_emb, X_val_emb, X_test_emb = apply_tabpfn_embeddings()
        elif model == "TARTE":
            # Apply TARTE embeddings
            X_train_emb, X_val_emb, X_test_emb = apply_tarte_embeddings(
                X_train, X_val, X_test_fold,
                feature_names,
                mode=tfm_mode,
                verbose=(verbose and fold_idx == 0)
            )
        else:
            raise ValueError(f"Unknown model: {model}")

        
        if verbose and fold_idx == 0:
            print(f"\nFeature dimensions after TFM: {X_train_emb.shape[1]}")
        
        # Run single fold
        fold_metrics = run_single_fold(
            X_train_emb, T_train, E_train,
            X_val_emb, T_val, E_val,
            X_test_emb, T_test_fold, E_test_fold,
            config, fold_idx, verbose
        )
        
        fold_results.append(fold_metrics)
    
    # Aggregate results
    metric_keys = fold_results[0].keys()
    mean_metrics = {k: np.mean([f[k] for f in fold_results]) for k in metric_keys}
    std_metrics = {k: np.std([f[k] for f in fold_results]) for k in metric_keys}
    
    return mean_metrics, std_metrics, fold_results


def run_hyperparameter_tuning(
    X_train: np.ndarray,
    T_train: np.ndarray,
    E_train: np.ndarray,
    X_val: np.ndarray,
    T_val: np.ndarray,
    E_val: np.ndarray,
    verbose: bool = True
) -> Dict:
    """
    Run hyperparameter tuning using validation set.
    """
    if verbose:
        print("\nHyperparameter Tuning...")
    
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [4, 6, 8],
        'learning_rate': [0.05, 0.1, 0.2],
        'min_child_weight': [1, 5, 10],
        'subsample': [0.7, 0.8, 0.9],
        'colsample_bytree': [0.7, 0.8, 0.9]
    }
    
    best_score = -np.inf
    best_params = {}
    
    # Simple grid search (can be replaced with RandomizedSearchCV)
    from itertools import product
    
    # Random sample of configurations for efficiency
    np.random.seed(42)
    n_configs = 30
    
    configs = []
    for _ in range(n_configs):
        config = {
            'n_estimators': np.random.choice(param_grid['n_estimators']),
            'max_depth': np.random.choice(param_grid['max_depth']),
            'learning_rate': np.random.choice(param_grid['learning_rate']),
            'min_child_weight': np.random.choice(param_grid['min_child_weight']),
            'subsample': np.random.choice(param_grid['subsample']),
            'colsample_bytree': np.random.choice(param_grid['colsample_bytree']),
        }
        configs.append(config)
    
    for i, cfg in enumerate(configs):
        model = SurvivalStackingModel(
            n_intervals=20,
            classifier='xgboost',
            classifier_params=cfg,
            random_state=42
        )
        model.fit(X_train, T_train, E_train, verbose=False)
        
        # Evaluate on validation set
        eval_times = model.transformer_.get_interval_times()
        survival_matrix = model.predict_survival(X_val)
        
        metrics = compute_survival_metrics(
            T_val, E_val, survival_matrix, eval_times,
            T_train=T_train, E_train=E_train
        )
        
        # Score: average C-Index (higher is better) - IBS (lower is better)
        score = metrics['c_index_q50'] - metrics['ibs']
        
        if score > best_score:
            best_score = score
            best_params = cfg.copy()
            if verbose:
                print(f"  Config {i+1}/{n_configs}: C-Index={metrics['c_index_q50']:.4f}, IBS={metrics['ibs']:.4f} (new best)")
    
    if verbose:
        print(f"\nBest params: {best_params}")
    
    return best_params


def main():
    parser = argparse.ArgumentParser(description='Survival Stacking Benchmark')
    parser.add_argument('--dataset', type=str, default='PBC',
                        choices=['METABRIC', 'PBC', 'SUPPORT', 'GBSG'],
                        help='Dataset to use')
    parser.add_argument('--model', default='TARTE',
                        choices=['TARTE', 'TabPFN', 'TabICL'], help='Model to use')
    parser.add_argument('--mode', type=str, default='deep+raw',
                        choices=['raw', 'deep', 'deep+raw'],
                        help='TFM embedding mode')
    parser.add_argument('--classifier', type=str, default='xgboost',
                        choices=['xgboost', 'tabicl'],
                        help='Classifier model to use')
    parser.add_argument('--cv', type=int, default=5,
                        help='Number of CV folds')
    parser.add_argument('--n_intervals', type=int, default=20,
                        help='Number of time intervals')
    parser.add_argument('--tune', action='store_true',
                        help='Run hyperparameter tuning')
    parser.add_argument('--output_dir', type=str, default='results/survival_stacking',
                        help='Output directory for results')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--verbose', action='store_true', default=True,
                        help='Verbose output')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = PROJECT_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*70)
    print("SURVIVAL STACKING BENCHMARK")
    print("="*70)
    print(f"Dataset: {args.dataset}")
    print(f"Mode: {args.mode}")
    print(f"CV Folds: {args.cv}")
    print(f"Time Intervals: {args.n_intervals}")
    print(f"Seed: {args.seed}")
    
    # Load dataset
    print(f"\nLoading {args.dataset}...")
    start_time = time.time()
    
    X, T, E, feature_names = load_dataset_for_survival(args.dataset)
    
    print(f"  Samples: {len(T)}")
    print(f"  Features: {X.shape[1]}")
    print(f"  Events: {E.sum()} ({100*E.mean():.1f}%)")
    print(f"  Time range: [{T.min():.2f}, {T.max():.2f}]")
    
    # Configuration
    config = {
        'n_intervals': args.n_intervals,
        'interval_strategy': 'quantile',
        'classifier': args.classifier,
        'random_state': args.seed,
        # Default XGBoost params
        'n_estimators': 200,
        'max_depth': 6,
        'learning_rate': 0.1,
        'min_child_weight': 5,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
    }
    
    # Run CV experiment
    mean_metrics, std_metrics, fold_results = run_cv_experiment(
        X, T, E, feature_names,
        config,
        n_folds=args.cv,
        model=args.model,
        tfm_mode=args.mode,
        verbose=args.verbose
    )
    
    elapsed = time.time() - start_time
    
    # Print summary
    print("\n" + "="*70)
    print("FINAL RESULTS")
    print("="*70)
    print(f"Dataset: {args.dataset}")
    print(f"Model: {args.model}")
    print(f"Mode: {args.mode}")
    print(f"CV Folds: {args.cv}")
    print("-"*70)
    
    for metric in ['c_index_q25', 'c_index_q50', 'c_index_q75', 'ibs']:
        mean_val = mean_metrics.get(metric, 0)
        std_val = std_metrics.get(metric, 0)
        print(f"  {metric}: {mean_val:.4f} ± {std_val:.4f}")
    
    print("-"*70)
    print(f"Total time: {elapsed/60:.1f} minutes")
    
    # Check against targets
    c_index_mean = mean_metrics.get('c_index_q50', 0)
    ibs_mean = mean_metrics.get('ibs', 1)
    
    if args.dataset == 'PBC':
        target_c = 0.8676
        target_ibs = 0.12
        print(f"\nTarget: C-Index > {target_c}, IBS < {target_ibs}")
        print(f"Achieved: C-Index = {c_index_mean:.4f} ({'✓' if c_index_mean > target_c else '✗'})")
        print(f"          IBS = {ibs_mean:.4f} ({'✓' if ibs_mean < target_ibs else '✗'})")
    
    # Save results
    results = {
        'dataset': args.dataset,
        'mode': args.mode,
        'n_folds': args.cv,
        'n_intervals': args.n_intervals,
        'config': config,
        'mean_metrics': mean_metrics,
        'std_metrics': std_metrics,
        'fold_results': fold_results,
        'elapsed_time': elapsed,
        'timestamp': datetime.now().isoformat()
    }
    
    result_file = output_dir / f"{args.dataset}_{args.model.lower()}_{args.mode}_{args.cv}-fold_results.json"
    with open(result_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\nResults saved to: {result_file}")
    
    return results


if __name__ == '__main__':
    main()
