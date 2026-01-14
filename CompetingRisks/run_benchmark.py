"""
Competing Risks Benchmark Runner

Runs the complete 3-phase benchmark:
- Phase 1: Multi-Class Survival Stacking (XGBoost)
- Phase 2: Pure Neural Fine-Gray
- Phase 3: Hybrid (NFG Embeddings + XGBoost)

Evaluates on SEER and SYNTHETIC_COMPETING datasets using:
- Cause-Specific C-Index
- CIF-based Integrated Brier Score
"""

import numpy as np
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from sklearn.model_selection import StratifiedKFold
import warnings

# Use existing modules
from datasets.datasets import load_dataset
from .utils import (
    get_competing_risks_datasets,
    get_evaluation_times,
    evaluate_competing_risks_model,
    aggregate_metrics
)
from .stacking_multiclass import MultiClassSurvivalStacking
from .nfg_wrapper import NFGCompetingRisks
from .hybrid_model import HybridNFGStacking


# Default configurations
DEFAULT_STACKING_CONFIG = {
    'n_intervals': 20,
    'interval_strategy': 'quantile',
}

DEFAULT_NFG_CONFIG = {
    'layers': [100, 100, 100],
    'layers_surv': [100],
    'dropout': 0.0,
    'n_iter': 1000,
}

DEFAULT_HYBRID_CONFIG = {
    'nfg_layers': [100, 100, 100],
    'nfg_layers_surv': [100],
    'nfg_n_iter': 1000,
    'n_intervals': 20,
    'use_raw_features': True,
}


def run_single_phase(
    phase: str,
    X_train: np.ndarray,
    T_train: np.ndarray,
    E_train: np.ndarray,
    X_val: np.ndarray,
    T_val: np.ndarray,
    E_val: np.ndarray,
    X_test: np.ndarray,
    T_test: np.ndarray,
    E_test: np.ndarray,
    config: Optional[Dict] = None,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Run a single phase of the benchmark.
    
    Parameters
    ----------
    phase : str
        Which phase: 'stacking', 'nfg', or 'hybrid'
    X_train, T_train, E_train : training data
    X_val, T_val, E_val : validation data
    X_test, T_test, E_test : test data
    config : dict, optional
        Model configuration
    verbose : bool
        Whether to print progress
        
    Returns
    -------
    results : dict
        Metrics and model info
    """
    phase = phase.lower()
    n_risks = int(E_train.max())
    
    results = {
        'phase': phase,
        'n_train': len(X_train),
        'n_val': len(X_val),
        'n_test': len(X_test),
        'n_features': X_train.shape[1],
        'n_risks': n_risks,
    }
    
    start_time = time.time()
    
    try:
        if phase == 'stacking':
            config = {**DEFAULT_STACKING_CONFIG, **(config or {})}
            model = MultiClassSurvivalStacking(**config)
            model.fit(X_train, T_train, E_train, X_val, T_val, E_val, verbose=verbose)
            times = model.get_times()
            cif = model.predict_cif(X_test)
            
        elif phase == 'nfg':
            config = {**DEFAULT_NFG_CONFIG, **(config or {})}
            n_iter = config.pop('n_iter', 1000)
            model = NFGCompetingRisks(**config)
            model.fit(X_train, T_train, E_train, X_val, T_val, E_val, 
                     n_iter=n_iter, verbose=verbose)
            times = model.get_times()
            cif = model.predict_cif(X_test)
            
        elif phase == 'hybrid':
            config = {**DEFAULT_HYBRID_CONFIG, **(config or {})}
            model = HybridNFGStacking(**config)
            model.fit(X_train, T_train, E_train, X_val, T_val, E_val, verbose=verbose)
            times = model.get_times()
            cif = model.predict_cif(X_test)
            
        else:
            raise ValueError(f"Unknown phase: {phase}")
        
        results['training_time'] = time.time() - start_time
        results['status'] = 'success'
        
        # Evaluate metrics using existing metrics module
        metrics = evaluate_competing_risks_model(
            T_test, E_test, cif, times,
            T_train=T_train, E_train=E_train
        )
        
        # Add per-risk metrics
        for metric_name in ['c_index', 'ibs']:
            for risk, value in metrics[metric_name].items():
                results[f'{metric_name}_risk{risk}'] = value
        
        # Add aggregated metrics
        aggregated = aggregate_metrics(metrics, n_risks)
        results.update(aggregated)
        
        if verbose:
            print(f"\n{phase.upper()} Results:")
            for risk in range(1, n_risks + 1):
                print(f"  Risk {risk}: C-Index={metrics['c_index'][risk]:.4f}, "
                      f"IBS={metrics['ibs'][risk]:.4f}")
        
    except Exception as e:
        results['status'] = 'failed'
        results['error'] = str(e)
        if verbose:
            print(f"  Phase {phase} failed: {e}")
    
    return results


def run_cv_benchmark(
    dataset: str,
    n_folds: int = 5,
    phases: List[str] = ['stacking', 'nfg', 'hybrid'],
    configs: Optional[Dict[str, Dict]] = None,
    random_state: int = 42,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Run cross-validated benchmark on a dataset.
    
    Parameters
    ----------
    dataset : str
        Dataset name ('SEER' or 'SYNTHETIC_COMPETING')
    n_folds : int
        Number of CV folds
    phases : list
        Which phases to run
    configs : dict, optional
        Per-phase configurations
    random_state : int
        Random seed
    verbose : bool
        Whether to print progress
        
    Returns
    -------
    results : dict
        Complete benchmark results
    """
    configs = configs or {}
    
    if verbose:
        print(f"\n{'='*70}")
        print(f"COMPETING RISKS BENCHMARK: {dataset}")
        print(f"{'='*70}")
    
    # Load data using existing datasets module
    X, T, E, feature_names = load_dataset(dataset)
    n_risks = int(E.max())  # Number of event types (excluding 0 = censored)
    
    if verbose:
        print(f"Data: {X.shape[0]} samples, {X.shape[1]} features, {n_risks} risks")
        print(f"Event distribution: {dict(zip(*np.unique(E, return_counts=True)))}")
    
    # Setup CV
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    
    # Results storage
    all_results = {
        'dataset': dataset,
        'n_samples': len(X),
        'n_features': X.shape[1],
        'n_risks': n_risks,
        'n_folds': n_folds,
        'phases': {},
        'timestamp': datetime.now().isoformat()
    }
    
    for phase in phases:
        all_results['phases'][phase] = {
            'folds': [],
            'mean_metrics': {},
            'std_metrics': {}
        }
    
    # Run CV
    for fold, (train_idx, test_idx) in enumerate(cv.split(X, E)):
        if verbose:
            print(f"\n{'='*50}")
            print(f"FOLD {fold + 1}/{n_folds}")
            print(f"{'='*50}")
        
        # Split data
        X_train_full, X_test = X[train_idx], X[test_idx]
        T_train_full, T_test = T[train_idx], T[test_idx]
        E_train_full, E_test = E[train_idx], E[test_idx]
        
        # Create validation split from training data
        val_size = int(0.15 * len(X_train_full))
        val_idx = np.random.RandomState(random_state + fold).permutation(len(X_train_full))[:val_size]
        train_mask = np.ones(len(X_train_full), dtype=bool)
        train_mask[val_idx] = False
        
        X_train = X_train_full[train_mask]
        T_train = T_train_full[train_mask]
        E_train = E_train_full[train_mask]
        
        X_val = X_train_full[~train_mask]
        T_val = T_train_full[~train_mask]
        E_val = E_train_full[~train_mask]
        
        # Run each phase
        for phase in phases:
            if verbose:
                print(f"\n--- Phase: {phase.upper()} ---")
            
            phase_config = configs.get(phase, {})
            
            fold_results = run_single_phase(
                phase,
                X_train, T_train, E_train,
                X_val, T_val, E_val,
                X_test, T_test, E_test,
                config=phase_config,
                verbose=verbose
            )
            fold_results['fold'] = fold
            
            all_results['phases'][phase]['folds'].append(fold_results)
    
    # Compute mean and std across folds
    for phase in phases:
        fold_results = all_results['phases'][phase]['folds']
        
        # Collect metrics
        metric_keys = [k for k in fold_results[0].keys() 
                      if k not in ['phase', 'fold', 'status', 'error', 'n_train', 'n_val', 'n_test']]
        
        for key in metric_keys:
            values = [r.get(key) for r in fold_results 
                     if r.get('status') == 'success' and r.get(key) is not None]
            if values:
                all_results['phases'][phase]['mean_metrics'][key] = float(np.mean(values))
                all_results['phases'][phase]['std_metrics'][key] = float(np.std(values))
    
    if verbose:
        print(f"\n{'='*70}")
        print("SUMMARY")
        print(f"{'='*70}")
        for phase in phases:
            phase_data = all_results['phases'][phase]
            mean = phase_data['mean_metrics']
            std = phase_data['std_metrics']
            
            print(f"\n{phase.upper()}:")
            if 'c_index_mean' in mean:
                print(f"  C-Index (mean): {mean['c_index_mean']:.4f} ± {std.get('c_index_mean', 0):.4f}")
            if 'ibs_mean' in mean:
                print(f"  IBS (mean): {mean['ibs_mean']:.4f} ± {std.get('ibs_mean', 0):.4f}")
            for risk in range(1, n_risks + 1):
                c_key = f'c_index_risk{risk}'
                ibs_key = f'ibs_risk{risk}'
                if c_key in mean:
                    print(f"  Risk {risk}: C-Index={mean[c_key]:.4f}, IBS={mean.get(ibs_key, float('nan')):.4f}")
    
    return all_results


def run_full_benchmark(
    datasets: List[str] = ['SYNTHETIC_COMPETING', 'SEER'],
    n_folds: int = 5,
    phases: List[str] = ['stacking', 'nfg', 'hybrid'],
    output_dir: Optional[str] = None,
    verbose: bool = True
) -> Dict[str, Dict]:
    """
    Run the complete benchmark across all datasets.
    
    Parameters
    ----------
    datasets : list
        Dataset names to evaluate
    n_folds : int
        Number of CV folds
    phases : list
        Which phases to run
    output_dir : str, optional
        Where to save results
    verbose : bool
        Whether to print progress
        
    Returns
    -------
    all_results : dict
        Results for all datasets
    """
    if output_dir is None:
        output_dir = Path(__file__).parent.parent / 'results' / 'competing_risks'
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    all_results = {}
    
    for dataset in datasets:
        try:
            results = run_cv_benchmark(
                dataset=dataset,
                n_folds=n_folds,
                phases=phases,
                verbose=verbose
            )
            all_results[dataset] = results
            
            # Save individual dataset results
            filename = output_dir / f'{dataset.lower()}_benchmark_{n_folds}fold.json'
            with open(filename, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            if verbose:
                print(f"\nSaved results to: {filename}")
                
        except Exception as e:
            warnings.warn(f"Failed to run benchmark on {dataset}: {e}")
            all_results[dataset] = {'status': 'failed', 'error': str(e)}
    
    # Save combined results
    combined_file = output_dir / f'full_benchmark_{n_folds}fold.json'
    with open(combined_file, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    
    if verbose:
        print(f"\nSaved combined results to: {combined_file}")
    
    return all_results


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Run Competing Risks Benchmark')
    parser.add_argument('--datasets', nargs='+', default=['SYNTHETIC_COMPETING', 'SEER'],
                       help='Datasets to evaluate (SEER requires local file at datasets/seer/seernfg.csv)')
    parser.add_argument('--n-folds', type=int, default=5, help='Number of CV folds')
    parser.add_argument('--phases', nargs='+', default=['stacking', 'nfg', 'hybrid'],
                       help='Phases to run')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Output directory for results')
    parser.add_argument('--quiet', action='store_true', help='Suppress output')
    
    args = parser.parse_args()
    
    run_full_benchmark(
        datasets=args.datasets,
        n_folds=args.n_folds,
        phases=args.phases,
        output_dir=args.output_dir,
        verbose=not args.quiet
    )
