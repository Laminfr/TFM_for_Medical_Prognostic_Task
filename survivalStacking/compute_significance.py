#!/usr/bin/env python
"""
Compute Statistical Significance (Paired T-Tests) for Benchmark Results
========================================================================

Run after training to determine if differences between methods are statistically significant.

Usage:
    python -m survivalStacking.compute_significance
    python -m survivalStacking.compute_significance --dataset METABRIC
    python -m survivalStacking.compute_significance --baseline deepsurv --metric ibs
"""

import argparse
import json
import numpy as np
from pathlib import Path
from scipy.stats import ttest_rel
from typing import Dict, List, Tuple

PROJECT_ROOT = Path(__file__).parent.parent
RESULTS_DIR = PROJECT_ROOT / 'results' / 'survival_stacking'

# Method display names
METHOD_NAMES = {
    'survstack_raw': 'SurvStack-Raw',
    'survstack_tabicl_emb': 'SurvStack-TabICL-Emb',
    'survstack_tabpfn_emb': 'SurvStack-TabPFN-Emb',
    'survstack_tabicl': 'SurvStack-TabICL',
    'survstack_tabpfn': 'SurvStack-TabPFN',
    'coxph': 'CoxPH',
    'xgboost': 'XGBoost',
    'deepsurv': 'DeepSurv',
}


def load_results(dataset: str) -> Dict:
    """Load benchmark results for a dataset."""
    patterns = [
        f'{dataset.upper()}_full_benchmark_5fold.json',
        f'{dataset.upper()}_full_benchmark_3fold.json',
    ]
    
    for pattern in patterns:
        filepath = RESULTS_DIR / pattern
        if filepath.exists():
            with open(filepath, 'r') as f:
                return json.load(f)
    
    return None


def paired_ttest(fold_results_a: List[Dict], fold_results_b: List[Dict], 
                 metric: str) -> Tuple[float, float, float, float]:
    """
    Perform paired t-test between two methods.
    
    Returns: (mean_a, mean_b, t_statistic, p_value)
    """
    values_a = [f[metric] for f in fold_results_a if metric in f and f[metric] > 0]
    values_b = [f[metric] for f in fold_results_b if metric in f and f[metric] > 0]
    
    # Need same number of paired observations
    n = min(len(values_a), len(values_b))
    if n < 2:
        return np.nan, np.nan, np.nan, np.nan
    
    values_a = values_a[:n]
    values_b = values_b[:n]
    
    mean_a = np.mean(values_a)
    mean_b = np.mean(values_b)
    
    t_stat, p_val = ttest_rel(values_a, values_b)
    
    return mean_a, mean_b, t_stat, p_val


def compute_all_pairwise_tests(results: Dict, baseline: str, metric: str) -> Dict:
    """
    Compute paired t-tests for all methods against baseline.
    
    Parameters
    ----------
    results : dict
        Results dictionary with 'results' key containing method results
    baseline : str
        Baseline method key (e.g., 'deepsurv', 'coxph')
    metric : str
        Metric to compare (e.g., 'c_index_q50', 'ibs')
    
    Returns
    -------
    dict : T-test results for each method
    """
    method_results = results.get('results', {})
    
    if baseline not in method_results:
        raise ValueError(f"Baseline '{baseline}' not found in results. "
                        f"Available: {list(method_results.keys())}")
    
    baseline_folds = method_results[baseline].get('fold_results', [])
    
    output = {}
    for method, data in method_results.items():
        if method == baseline:
            continue
        
        method_folds = data.get('fold_results', [])
        
        if not method_folds or all(f.get(metric, 0) == 0 for f in method_folds):
            # Method failed (all zeros)
            output[method] = {
                'status': 'FAILED',
                'reason': 'Method returned all zeros (likely failed)'
            }
            continue
        
        mean_baseline, mean_method, t_stat, p_val = paired_ttest(
            baseline_folds, method_folds, metric
        )
        
        if np.isnan(p_val):
            output[method] = {
                'status': 'INSUFFICIENT_DATA',
                'reason': 'Not enough paired observations'
            }
            continue
        
        # Determine if method is better (depends on metric direction)
        # C-Index: higher is better; IBS: lower is better
        if 'ibs' in metric.lower():
            diff = mean_baseline - mean_method  # Positive = method better
            better = mean_method < mean_baseline
        else:
            diff = mean_method - mean_baseline  # Positive = method better
            better = mean_method > mean_baseline
        
        output[method] = {
            'status': 'OK',
            'baseline_mean': mean_baseline,
            'method_mean': mean_method,
            'difference': diff,
            'method_better': better,
            't_statistic': t_stat,
            'p_value': p_val,
            'significant_005': p_val < 0.05,
            'significant_001': p_val < 0.01,
        }
    
    return output


def print_significance_table(dataset: str, baseline: str, metric: str, 
                             test_results: Dict):
    """Print formatted significance table."""
    print(f"\n{'='*80}")
    print(f"STATISTICAL SIGNIFICANCE: {dataset}")
    print(f"{'='*80}")
    print(f"Baseline: {METHOD_NAMES.get(baseline, baseline)}")
    print(f"Metric: {metric}")
    print(f"Test: Paired t-test (two-tailed)")
    print(f"{'─'*80}")
    
    print(f"\n{'Method':<25} {'Mean':<10} {'Diff':<10} {'p-value':<12} {'Significant?'}")
    print(f"{'-'*70}")
    
    # Print baseline first
    baseline_mean = None
    for method, res in test_results.items():
        if res.get('status') == 'OK':
            baseline_mean = res['baseline_mean']
            break
    
    if baseline_mean is not None:
        print(f"{METHOD_NAMES.get(baseline, baseline):<25} {baseline_mean:.4f}     {'(baseline)':<10} {'-':<12} -")
    
    # Print other methods
    for method, res in sorted(test_results.items()):
        name = METHOD_NAMES.get(method, method)
        
        if res.get('status') != 'OK':
            print(f"{name:<25} {'N/A':<10} {'N/A':<10} {res.get('reason', 'Unknown')}")
            continue
        
        mean_val = res['method_mean']
        diff = res['difference']
        p_val = res['p_value']
        
        # Significance stars
        if p_val < 0.001:
            sig = "*** (p<0.001)"
        elif p_val < 0.01:
            sig = "**  (p<0.01)"
        elif p_val < 0.05:
            sig = "*   (p<0.05)"
        else:
            sig = "    (n.s.)"
        
        # Direction indicator
        direction = "↑" if res['method_better'] else "↓"
        
        print(f"{name:<25} {mean_val:.4f}     {diff:+.4f} {direction}  {p_val:.6f}   {sig}")
    
    print(f"\n{'─'*80}")
    print("Legend: ↑ = method better, ↓ = baseline better")
    print("        *** p<0.001, ** p<0.01, * p<0.05, n.s. = not significant")
    print(f"{'='*80}\n")


def main():
    parser = argparse.ArgumentParser(
        description='Compute statistical significance for benchmark results'
    )
    parser.add_argument(
        '--dataset', '-d', type=str, 
        choices=['METABRIC', 'PBC', 'SUPPORT', 'all'],
        default='all',
        help='Dataset to analyze'
    )
    parser.add_argument(
        '--baseline', '-b', type=str,
        default='deepsurv',
        choices=['survstack_raw', 'survstack_tabicl_emb', 'survstack_tabpfn_emb',
                 'survstack_tabicl', 'survstack_tabpfn', 'coxph', 'xgboost', 'deepsurv'],
        help='Baseline method for comparison'
    )
    parser.add_argument(
        '--metric', '-m', type=str,
        default='c_index_q50',
        choices=['c_index_q25', 'c_index_q50', 'c_index_q75', 'ibs'],
        help='Metric to compare'
    )
    parser.add_argument(
        '--output', '-o', type=str,
        default=None,
        help='Output JSON file for results'
    )
    
    args = parser.parse_args()
    
    # Determine datasets
    if args.dataset == 'all':
        datasets = ['METABRIC', 'PBC', 'SUPPORT']
    else:
        datasets = [args.dataset]
    
    all_results = {}
    
    print("\n" + "="*80)
    print("PAIRED T-TEST ANALYSIS FOR SURVIVAL STACKING BENCHMARK")
    print("="*80)
    
    for dataset in datasets:
        results = load_results(dataset)
        
        if results is None:
            print(f"\nNo results found for {dataset}")
            continue
        
        try:
            test_results = compute_all_pairwise_tests(
                results, 
                baseline=args.baseline,
                metric=args.metric
            )
            
            print_significance_table(dataset, args.baseline, args.metric, test_results)
            all_results[dataset] = test_results
            
        except ValueError as e:
            print(f"\nError for {dataset}: {e}")
    
    # Save to JSON if requested
    if args.output:
        output_path = Path(args.output)
        with open(output_path, 'w') as f:
            json.dump({
                'baseline': args.baseline,
                'metric': args.metric,
                'results': all_results
            }, f, indent=2, default=str)
        print(f"Results saved to: {output_path}")
    
    return all_results


if __name__ == "__main__":
    main()
