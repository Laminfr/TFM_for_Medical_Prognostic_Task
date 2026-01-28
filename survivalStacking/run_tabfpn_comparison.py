#!/usr/bin/env python
"""
Experiment: TabPFN Direct Classification vs Embedding Extraction

This script compares two approaches for using TabPFN in survival stacking:
1. TabPFN as direct binary classifier (no embeddings)
2. TabPFN for embedding extraction + XGBoost classifier

Usage:
    python -m survivalStacking.run_tabpfn_comparison --dataset METABRIC
    python -m survivalStacking.run_tabpfn_comparison --dataset PBC --cv 5
"""

import os
import sys
import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple
import warnings

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from survivalStacking.stacking_model import SurvivalStackingModel
from survivalStacking.evaluation import compute_survival_metrics, pairwise_t_test_vs_baseline
from survivalStacking.run_experiment import load_dataset_for_survival

ALL_BASELINES = ['xgboost', 'lightgbm', 'logistic']

def run_single_fold_comparison(
    X_train: np.ndarray,
    T_train: np.ndarray,
    E_train: np.ndarray,
    X_test: np.ndarray,
    T_test: np.ndarray,
    E_test: np.ndarray,
    feature_names: List[str],
    config: Dict,
    fold_idx: int,
    verbose: bool = True
) -> Dict[str, Dict[str, float]]:
    """
    Run comparison between different approaches on a single fold.
    
    Returns metrics for:
    - 'xgboost_raw': XGBoost on raw features
    - 'xgboost_embeddings': XGBoost on TabPFN embeddings  
    - 'tabpfn_direct': TabPFN as direct classifier
    """
    results = {}
    
    # Common model config
    model_config = {
        'n_intervals': config.get('n_intervals', 20),
        'interval_strategy': config.get('interval_strategy', 'quantile'),
        'random_state': config.get('random_state', 42)
    }

    for baseline in ALL_BASELINES:
        results_baseline = {}

        if baseline == 'xgboost':
            classifier_params = {
                'n_estimators': config.get('n_estimators', 200),
                'max_depth': config.get('max_depth', 5),
                'learning_rate': config.get('learning_rate', 0.05),
            }
        elif baseline == 'logistic':
            classifier_params = {
                'C': config.get('C', 0.1),
                'tol': 1e-3,
                'penalty': config.get('penalty', 'l2'),
                'max_iter': config.get('max_iter', 5000),
                'class_weight': config.get('class_weight', 'balanced'),
            }
        elif baseline == 'lightgbm':
            classifier_params = {
                'n_estimators': config.get('n_estimators', 200),
                'max_depth': config.get('max_depth', 5),
                'learning_rate': config.get('learning_rate', 0.05)
            }
        else:
            raise ValueError(f"Unknown classifier: {baseline}")
        # === Approach 1: Baseline on raw features ===
        if verbose:
            print(f"\n  [1/2] {baseline} on raw features...")

        model_baseline = SurvivalStackingModel(
            classifier=baseline,
            classifier_params=classifier_params,
            **model_config
        )
        model_baseline.fit(X_train, T_train, E_train, verbose=False)

        eval_times = model_baseline.transformer_.get_interval_times()
        survival_baseline = model_baseline.predict_survival(X_test)

        results_baseline[f'{baseline}_raw'] = compute_survival_metrics(
            T_test, E_test, survival_baseline, eval_times, T_train, E_train
        )

        if verbose:
            print(f"      C-Index: {results_baseline[f'{baseline}_raw'].get('c_index_q50', 0):.4f}, "
                  f"IBS: {results_baseline[f'{baseline}_raw'].get('ibs', 0):.4f}")

        # === Approach 2: XGBoost on TabPFN embeddings ===
        if verbose:
            print(f"  [2/2] {baseline} on TabPFN embeddings...")

        try:
            from survivalStacking.tools import apply_tabpfn_embeddings
            X_train_emb, _, X_test_emb = apply_tabpfn_embeddings(
                X_train, X_train[:10], X_test,  # Dummy validation
                E_train,
                mode = "deep+raw",
                verbose=True,
                n_estimators = 1,
                n_fold = 0)

            model_baseline_emb = SurvivalStackingModel(
                classifier=baseline,
                classifier_params=classifier_params,
                **model_config
            )
            model_baseline_emb.fit(X_train_emb, T_train, E_train, verbose=False)

            eval_times = model_baseline_emb.transformer_.get_interval_times()
            survival_baseline_emb = model_baseline_emb.predict_survival(X_test_emb)

            results_baseline[f'{baseline}_embeddings'] = compute_survival_metrics(
                T_test, E_test, survival_baseline_emb, eval_times, T_train, E_train
            )

            if verbose:
                print(f"      C-Index: {results_baseline[f'{baseline}_embeddings'].get('c_index_q50', 0):.4f}, "
                      f"IBS: {results_baseline[f'{baseline}_embeddings'].get('ibs', 0):.4f}")
        except Exception as e:
            if verbose:
                print(f"      FAILED: {e}")
            results_baseline[f'{baseline}_embeddings'] = {'error': str(e)}

        results[baseline] = results_baseline
    

    # === Approach 3: TabPFN direct classification ===
    tabpfn_results = {}
    if verbose:
        print(f"  [1/1] TaPFN direct classifier...")
    
    try:
        model_tabpfn = SurvivalStackingModel(
            classifier='tabpfn',
            classifier_params={
                'n_estimators': config.get('tabpfn_n_estimators', 4),
                'device': config.get('device', 'cuda'),
                'max_context_samples': config.get('max_context_samples', 2000),
                "random_state":config.get('random_state', 42),
                'verbose': False
            },
            **model_config
        )
        model_tabpfn.fit(X_train, T_train, E_train, verbose=False)

        eval_times = model_tabpfn.transformer_.get_interval_times()
        survival_tabpfn = model_tabpfn.predict_survival(X_test)
        
        tabpfn_results['tabpfn_direct'] = compute_survival_metrics(
            T_test, E_test, survival_tabpfn, eval_times, T_train, E_train
        )
        
        if verbose:
            print(f"      C-Index: {tabpfn_results['tabpfn_direct'].get('c_index_q50', 0):.4f}, "
                  f"IBS: {tabpfn_results['tabpfn_direct'].get('ibs', 0):.4f}")
    except Exception as e:
        if verbose:
            print(f"      FAILED: {e}")
        tabpfn_results['tabpfn_direct'] = {'error': str(e)}

    results['TabPFN'] = tabpfn_results
    
    return results


def run_cv_comparison(
    dataset: str,
    n_folds: int = 5,
    config: Dict = None,
    verbose: bool = True
) -> Dict:
    """
    Run cross-validated comparison experiment.
    """
    config = config or {}
    
    # Load dataset
    if verbose:
        print(f"\n{'='*60}")
        print(f"Dataset: {dataset}")
        print(f"{'='*60}")
    
    X, T, E, feature_names = load_dataset_for_survival(dataset, normalize=True)
    
    if verbose:
        print(f"Samples: {len(T)}, Features: {X.shape[1]}")
        print(f"Events: {E.sum()} ({100*E.mean():.1f}%)")
    
    # Setup CV
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    # Collect results per fold
    fold_results = []
    
    for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X, E)):
        if verbose:
            print(f"\n{'='*40}")
            print(f"Fold {fold_idx + 1}/{n_folds}")
            print(f"{'='*40}")
        
        X_train, X_test = X[train_idx], X[test_idx]
        T_train, T_test = T[train_idx], T[test_idx]
        E_train, E_test = E[train_idx], E[test_idx]
        
        fold_result = run_single_fold_comparison(
            X_train, T_train, E_train,
            X_test, T_test, E_test,
            feature_names, config, fold_idx, verbose
        )
        fold_results.append(fold_result)
    
    # Aggregate results
    summary = {}
    for baseline in (ALL_BASELINES+['TabPFN']):
        for approach in ['xgboost_raw', 'xgboost_embeddings',
                         'logistic_raw', 'logistic_embeddings',
                         'lightgbm_raw', 'lightgbm_embeddings',
                         'tabpfn_direct']:

            approach_results = []
            for fold in fold_results:
                # fold[baseline] may contain nested dict for each approach
                baseline_data = fold.get(baseline, {})
                approach_data = baseline_data.get(approach)

                if approach_data and 'error' not in approach_data:
                    approach_results.append(approach_data)

            if approach_results:
                summary[approach] = {
                    'c_index_q50_mean': np.mean([r['c_index_q50'] for r in approach_results]),
                    'c_index_q50_std': np.std([r['c_index_q50'] for r in approach_results]),
                    'ibs_mean': np.mean([r.get('ibs', 0) for r in approach_results]),
                    'ibs_std': np.std([r.get('ibs', 0) for r in approach_results]),
                    'n_folds': len(approach_results)
                }

    t_test_results = pairwise_t_test_vs_baseline(fold_results, baseline='TabPFN', baseline_method="tabpfn_direct", metric='c_index_q50')
    return {
        'dataset': dataset,
        'n_folds': n_folds,
        'config': config,
        'fold_results': fold_results,
        't_test_vs_baseline': t_test_results,
        'summary': summary
    }


def print_summary(results: Dict):
    """Print comparison summary."""
    print(f"\n{'='*60}")
    print(f"SUMMARY: {results['dataset']} ({results['n_folds']}-fold CV)")
    print(f"{'='*60}")
    
    summary = results['summary']
    
    print(f"\n{'Approach':<25} {'C-Index (mean±std)':<22} {'IBS (mean±std)':<18}")
    print("-" * 65)
    
    for approach, metrics in summary.items():
        if 'error' in metrics:
            print(f"{approach:<25} {'FAILED':<22} {'FAILED':<18}")
        else:
            c_str = f"{metrics['c_index_q50_mean']:.4f} ± {metrics['c_index_q50_std']:.4f}"
            ibs_str = f"{metrics['ibs_mean']:.4f} ± {metrics['ibs_std']:.4f}"
            print(f"{approach:<25} {c_str:<22} {ibs_str:<18}")
    
    print("-" * 65)
    
    # Highlight winner
    valid_approaches = {k: v for k, v in summary.items() if 'error' not in v}
    if valid_approaches:
        best_approach = max(valid_approaches.keys(), 
                          key=lambda k: valid_approaches[k]['c_index_q50_mean'])
        print(f"\nBest approach: {best_approach} "
              f"(C-Index: {valid_approaches[best_approach]['c_index_q50_mean']:.4f})")

        # -------------------------------
        # Statistical significance section
        # -------------------------------
        t_tests = results.get('t_test_vs_baseline', {})

        def significance_stars(p):
            if p < 0.001:
                return '***'
            elif p < 0.01:
                return '**'
            elif p < 0.05:
                return '*'
            else:
                return 'ns'

        if t_tests:
            print(f"\nPaired t-tests vs baseline (TabPFN):")
            print(f"{'Method':<30} {'Mean diff':<12} {'p-value':<10} {'Sig'}")
            print("-" * 65)

            for method, stats in sorted(t_tests.items()):
                p = stats['p_value']
                sig = significance_stars(p) if not np.isnan(p) else 'NA'
                diff = stats.get('mean_diff', float('nan'))

                print(f"{method:<30} {diff:+.4f}     {p:.4f}    {sig}")

            print("\nSignificance codes: *** p<0.001, ** p<0.01, * p<0.05, ns not significant")


def main():
    parser = argparse.ArgumentParser(
        description='Compare TabPFN approaches in Survival Stacking'
    )
    parser.add_argument('--dataset', type=str, default='PBC',
                        choices=['METABRIC', 'PBC', 'SUPPORT'],
                        help='Dataset to use')
    parser.add_argument('--cv', type=int, default=5,
                        help='Number of CV folds')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device for TabPFN (cuda/cpu)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output JSON file for results')
    
    args = parser.parse_args()
    
    config = {
        'n_intervals': 20,
        'n_estimators': 200,
        'max_depth': 5,
        'learning_rate': 0.05,
        'device': args.device,
        'tabpfn_n_estimators': 4,
        'max_context_samples': 2000
    }
    
    results = run_cv_comparison(
        dataset=args.dataset,
        n_folds=args.cv,
        config=config,
        verbose=True
    )
    
    print_summary(results)
    
    # Save results
    if args.output:
        output_path = Path(args.output)
    else:
        output_dir = PROJECT_ROOT / 'results' / 'tabpfn_comparison'
        output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_path = output_dir / f'{args.dataset}_comparison_{timestamp}.json'
    
    # Convert numpy types for JSON serialization
    def convert_numpy(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.float64, np.float32)):
            return float(obj)
        return obj
    
    results_json = json.loads(
        json.dumps(results, default=convert_numpy)
    )
    
    with open(output_path, 'w') as f:
        json.dump(results_json, f, indent=2)
    
    print(f"\nResults saved to: {output_path}")


if __name__ == '__main__':
    main()
