#!/usr/bin/env python
"""
Survival Stacking Results Visualization
========================================

This script generates 3 key insight plots comparing Survival Stacking against
TFM embedding extraction baselines (CoxPH, RSF, XGBoost, DeepSurv).

1. PLOT 1: "Method Comparison by Dataset" (Discrimination)
   - Compares Survival Stacking vs all baseline models
   - Shows C-Index at median time (q50) as the primary metric
   - Groups by dataset (METABRIC, PBC, SUPPORT)
   
2. PLOT 2: "Performance Across Time Horizons" (Temporal Robustness)
   - Shows C-Index at early (q25), median (q50), and late (q75) times
   - Compares Survival Stacking vs best baseline for each dataset
   
3. PLOT 3: "Discrimination vs Calibration Trade-off"
   - Plots C-Index (discrimination) against IBS (calibration)
   - Shows where Survival Stacking falls in the trade-off space
   - Ideal: high C-Index AND low IBS (upper-left corner)

Usage:
    python visualize_results.py                        # All datasets
    python visualize_results.py --dataset pbc          # Single dataset
    python visualize_results.py --compare-tfm       # Include TFM baselines

Author: Auto-generated for Survival Stacking benchmark
"""

import argparse
import matplotlib.pyplot as plt
import numpy as np
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ==========================================
# CONFIGURATION
# ==========================================
PROJECT_ROOT = Path(__file__).parent.parent
SURV_STACK_RESULTS_DIR = PROJECT_ROOT / 'results' / 'survival_stacking'
TFM_RESULTS_DIR = PROJECT_ROOT / 'results'

# Dataset configurations
DATASET_CONFIG = {
    'metabric': {
        'name': 'METABRIC',
        'tfm_cv_dir': 'cv_metabric',
        'n_features': 9,
        'description': 'Breast Cancer Dataset'
    },
    'pbc': {
        'name': 'PBC',
        'tfm_cv_dir': 'cv_pbc',
        'n_features': 25,
        'description': 'Primary Biliary Cirrhosis'
    },
    'support': {
        'name': 'SUPPORT',
        'tfm_cv_dir': 'cv_support',
        'n_features': 24,
        'description': 'SUPPORT Study Dataset'
    }
}

# Models to compare
BASELINE_MODELS = ['CoxPH', 'RSF', 'XGBoost', 'DeepSurv']
TFM_MODES = ['raw', 'deep', 'deep+raw']

# Colors
MODEL_COLORS = {
    'SurvivalStacking': '#e74c3c',  # Red - our method
    'CoxPH': '#3498db',             # Blue
    'RSF': '#2ecc71',               # Green
    'XGBoost': '#9b59b6',           # Purple
    'DeepSurv': '#f39c12',          # Orange
    'NFG': '#1abc9c',               # Teal
}

MODE_COLORS = {
    'Raw': '#4a90e2',       # Blue
    'Deep': '#d9534f',      # Red  
    'Deep+Raw': '#5cb85c',  # Green
    'SurvStack': '#e74c3c'  # Bright Red
}

QUANTILE_COLORS = {
    'q25': '#ff7f0e',  # Orange - Early
    'q50': '#2ca02c',  # Green - Median
    'q75': '#9467bd'   # Purple - Late
}


def load_survival_stacking_results(dataset: str, model: str) -> Optional[Dict]:
    """Load Survival Stacking results for a dataset."""
    # Try different file patterns
    patterns = [
        f'{dataset.upper()}_{model.lower()}_deep+raw_5-fold_results.json',
        f'{dataset.upper()}_{model.lower()}_deep_5-fold_results.json',
        f'{dataset.upper()}_{model.lower()}_raw_5-fold_results.json',
    ]
    
    results = {}
    for pattern in patterns:
        filepath = SURV_STACK_RESULTS_DIR / pattern
        if filepath.exists():
            with open(filepath, 'r') as f:
                data = json.load(f)
                mode = data.get('mode', 'deep+raw')
                results[mode] = {
                    'c_index_q25': data['mean_metrics'].get('c_index_q25', 0),
                    'c_index_q50': data['mean_metrics'].get('c_index_q50', 0),
                    'c_index_q75': data['mean_metrics'].get('c_index_q75', 0),
                    'ibs': data['mean_metrics'].get('ibs', 0),
                    'std_c_index_q50': data['std_metrics'].get('c_index_q50', 0),
                    'std_ibs': data['std_metrics'].get('ibs', 0),
                }
    
    return results if results else None


def load_tfm_baseline_results(dataset: str, model: str) -> Optional[Dict]:
    """Load TFM baseline results for comparison."""
    config = DATASET_CONFIG.get(dataset.lower())
    if not config:
        return None
    
    cv_dir = TFM_RESULTS_DIR / f'{model.lower()}' / config['tfm_cv_dir']
    if not cv_dir.exists():
        return None
    
    # Find the most recent results file
    final_files = sorted(cv_dir.glob('cv_results_final_*.json'), 
                        key=os.path.getmtime, reverse=True)
    if not final_files:
        final_files = sorted(cv_dir.glob('cv_results_*.json'),
                            key=os.path.getmtime, reverse=True)
    
    if not final_files:
        return None
    
    with open(final_files[0], 'r') as f:
        raw_results = json.load(f)
    
    # Organize by model and mode
    results = {}
    for r in raw_results:
        if r.get('status') != 'success':
            continue
        
        model = r.get('model')
        mode = r.get('mode', 'raw')
        
        if model not in results:
            results[model] = {}
        
        results[model][mode] = {
            'c_index_q25': r.get('c_index_q25', 0),
            'c_index_q50': r.get('c_index_q50', 0),
            'c_index_q75': r.get('c_index_q75', 0),
            'ibs': r.get('ibs', 0),
        }
    
    return results


def plot1_method_comparison(all_data: Dict, plots_dir: Path, tfm: str):
    """
    PLOT 1: Method Comparison Across Datasets
    
    Compares Survival Stacking (deep+raw) vs best baseline for each dataset.
    """
    print("\n--- Plot 1: Method Comparison by Dataset ---")
    
    datasets = [d for d in ['metabric', 'pbc', 'support'] if d in all_data]
    if not datasets:
        print("No data available for plotting")
        return
    
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_facecolor('#fafafa')
    
    x = np.arange(len(datasets))
    width = 0.12
    n_methods = 6  # SurvStack + 4 baselines + best baseline indicator
    
    # Collect data for each method
    methods = ['SurvivalStacking', 'CoxPH', 'RSF', 'XGBoost', 'DeepSurv']
    method_data = {m: [] for m in methods}
    
    for dataset in datasets:
        data = all_data[dataset]
        
        # Survival Stacking (deep+raw mode)
        ss_data = data.get('survival_stacking', {}).get('deep+raw', {})
        method_data['SurvivalStacking'].append(ss_data.get('c_index_q50', 0))
        
        # Baselines (use deep+raw mode, fallback to raw)
        baselines = data.get('tfm_baselines', {})
        for model in ['CoxPH', 'RSF', 'XGBoost', 'DeepSurv']:
            model_data = baselines.get(model, {})
            # Prefer deep+raw, then deep, then raw
            val = (model_data.get('deep+raw', {}).get('c_index_q50', 0) or
                   model_data.get('deep', {}).get('c_index_q50', 0) or
                   model_data.get('raw', {}).get('c_index_q50', 0))
            method_data[model].append(val)
    
    # Plot bars
    colors = [MODEL_COLORS.get(m, '#888888') for m in methods]
    bars_list = []
    
    for i, method in enumerate(methods):
        offset = (i - len(methods)/2 + 0.5) * width
        bars = ax.bar(x + offset, method_data[method], width, 
                     label=method, color=colors[i], edgecolor='white', linewidth=1.5)
        bars_list.append(bars)
        
        # Add value labels
        for j, (bar, val) in enumerate(zip(bars, method_data[method])):
            if val > 0:
                ax.annotate(f'{val:.3f}',
                           xy=(bar.get_x() + bar.get_width() / 2, val),
                           xytext=(0, 3), textcoords="offset points",
                           ha='center', va='bottom', fontsize=8, fontweight='bold',
                           rotation=45)
    
    # Formatting
    ax.set_title(f'Survival Stacking vs {tfm} Baselines\n(C-Index at Median Survival Time, 5-Fold CV)',
                 fontsize=14, fontweight='bold', pad=15)
    ax.set_ylabel('C-Index (↑ better)', fontsize=12)
    ax.set_xlabel('Dataset', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels([DATASET_CONFIG[d]['name'] for d in datasets], fontsize=11, fontweight='medium')
    ax.legend(loc='upper right', fontsize=9, ncol=2, framealpha=0.9)
    
    # Dynamic y-axis
    all_vals = [v for m in methods for v in method_data[m] if v > 0]
    if all_vals:
        ax.set_ylim(min(all_vals) - 0.03, max(all_vals) + 0.05)
    
    ax.grid(axis='y', linestyle='--', alpha=0.4, zorder=0)
    ax.set_axisbelow(True)
    
    # Add insight box
    insights = []
    for i, dataset in enumerate(datasets):
        ss_val = method_data['SurvivalStacking'][i]
        best_baseline = max((method_data[m][i], m) for m in ['CoxPH', 'RSF', 'XGBoost', 'DeepSurv'])
        if ss_val > 0 and best_baseline[0] > 0:
            diff = (ss_val - best_baseline[0]) * 100
            status = "✓" if diff > 0 else "✗"
            insights.append(f"{DATASET_CONFIG[dataset]['name']}: {status} {diff:+.1f}% vs {best_baseline[1]}")
    
    if insights:
        insight_text = "Survival Stacking vs Best Baseline:\n" + "\n".join(insights)
        ax.text(0.02, 0.02, insight_text, transform=ax.transAxes, fontsize=9,
                verticalalignment='bottom', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    output_path = plots_dir / f'plot1_method_comparison_{tfm}.png'
    plt.savefig(output_path, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot2_temporal_performance(all_data: Dict, plots_dir: Path, tfm: str):
    """
    PLOT 2: Performance Across Time Horizons
    
    Compares C-Index at q25, q50, q75 for Survival Stacking vs baselines.
    """
    print("\n--- Plot 2: Performance Across Time Horizons ---")
    
    datasets = [d for d in ['metabric', 'pbc', 'support'] if d in all_data]
    if not datasets:
        print("No data available for plotting")
        return
    
    fig, axes = plt.subplots(1, len(datasets), figsize=(6*len(datasets), 6))
    if len(datasets) == 1:
        axes = [axes]
    
    quantiles = ['q25', 'q50', 'q75']
    quantile_labels = ['Early (Q25)', 'Median (Q50)', 'Late (Q75)']
    
    for ax, dataset in zip(axes, datasets):
        ax.set_facecolor('#fafafa')
        data = all_data[dataset]
        
        # Methods to compare: SurvivalStacking + top baselines
        methods = ['SurvivalStacking', 'DeepSurv', 'RSF', 'CoxPH']
        x = np.arange(len(methods))
        width = 0.25
        
        for i, (q, ql) in enumerate(zip(quantiles, quantile_labels)):
            vals = []
            for method in methods:
                if method == 'SurvivalStacking':
                    ss_data = data.get('survival_stacking', {}).get('deep+raw', {})
                    vals.append(ss_data.get(f'c_index_{q}', 0))
                else:
                    baselines = data.get('tfm_baselines', {}).get(method, {})
                    val = (baselines.get('deep+raw', {}).get(f'c_index_{q}', 0) or
                           baselines.get('deep', {}).get(f'c_index_{q}', 0) or
                           baselines.get('raw', {}).get(f'c_index_{q}', 0))
                    vals.append(val)
            
            bars = ax.bar(x + (i - 1) * width, vals, width, label=ql,
                         color=QUANTILE_COLORS[q], edgecolor='white', alpha=0.85)
            
            # Add value labels
            for bar, val in zip(bars, vals):
                if val > 0:
                    ax.annotate(f'{val:.3f}',
                               xy=(bar.get_x() + bar.get_width() / 2, val),
                               xytext=(0, 2), textcoords="offset points",
                               ha='center', va='bottom', fontsize=7, rotation=45)
        
        ax.set_title(f'{DATASET_CONFIG[dataset]["name"]}', fontsize=13, fontweight='bold')
        ax.set_ylabel('C-Index', fontsize=11)
        ax.set_xticks(x)
        ax.set_xticklabels(methods, fontsize=9, rotation=15, ha='right')
        ax.legend(fontsize=8, loc='lower right')
        ax.grid(axis='y', linestyle='--', alpha=0.4)
        ax.set_axisbelow(True)
        
        # Dynamic y-limits
        all_vals = []
        for method in methods:
            if method == 'SurvivalStacking':
                ss_data = data.get('survival_stacking', {}).get('deep+raw', {})
                all_vals.extend([ss_data.get(f'c_index_{q}', 0) for q in quantiles])
            else:
                baselines = data.get('tfm_baselines', {}).get(method, {})
                for mode in ['deep+raw', 'deep', 'raw']:
                    if mode in baselines:
                        all_vals.extend([baselines[mode].get(f'c_index_{q}', 0) for q in quantiles])
                        break
        
        valid_vals = [v for v in all_vals if v > 0]
        if valid_vals:
            ax.set_ylim(min(valid_vals) - 0.02, max(valid_vals) + 0.03)
    
    plt.suptitle('C-Index Across Time Horizons: Early vs Median vs Late Survival\n(5-Fold CV)', 
                 fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    output_path = plots_dir / f'plot2_temporal_performance_{tfm}.png'
    plt.savefig(output_path, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot3_discrimination_vs_calibration(all_data: Dict, plots_dir: Path, tfm: str):
    """
    PLOT 3: Discrimination vs Calibration Trade-off
    
    Scatter plot of C-Index vs IBS for all methods across datasets.
    """
    print("\n--- Plot 3: Discrimination vs Calibration Trade-off ---")
    
    fig, ax = plt.subplots(figsize=(12, 9))
    ax.set_facecolor('#fafafa')
    
    # Markers for different datasets
    dataset_markers = {'metabric': 'o', 'pbc': 's', 'support': '^'}
    
    all_c_indices = []
    all_ibs_vals = []
    
    # Plot each method for each dataset
    for dataset, data in all_data.items():
        marker = dataset_markers.get(dataset, 'o')
        dataset_name = DATASET_CONFIG[dataset]['name']
        
        # Survival Stacking
        ss_data = data.get('survival_stacking', {}).get('deep+raw', {})
        if ss_data.get('c_index_q50', 0) > 0 and ss_data.get('ibs', 0) > 0:
            c_idx = ss_data['c_index_q50']
            ibs = ss_data['ibs']
            ax.scatter(ibs, c_idx, c=MODEL_COLORS['SurvivalStacking'], marker=marker,
                      s=200, edgecolors='black', linewidths=2, zorder=5,
                      label=f'SurvStack ({dataset_name})' if dataset == list(all_data.keys())[0] else '')
            ax.annotate(f'SS-{dataset_name}', (ibs, c_idx), xytext=(5, 5),
                       textcoords='offset points', fontsize=8, fontweight='bold')
            all_c_indices.append(c_idx)
            all_ibs_vals.append(ibs)
        
        # Baselines
        baselines = data.get('tfm_baselines', {})
        for model in BASELINE_MODELS:
            model_data = baselines.get(model, {})
            # Use best mode available
            for mode in ['deep+raw', 'deep', 'raw']:
                if mode in model_data:
                    c_idx = model_data[mode].get('c_index_q50', 0)
                    ibs = model_data[mode].get('ibs', 0)
                    if c_idx > 0 and ibs > 0:
                        ax.scatter(ibs, c_idx, c=MODEL_COLORS.get(model, '#888888'),
                                  marker=marker, s=120, edgecolors='white', linewidths=1.5,
                                  alpha=0.7, zorder=3)
                        ax.annotate(f'{model[:3]}', (ibs, c_idx), xytext=(3, 3),
                                   textcoords='offset points', fontsize=7, alpha=0.8)
                        all_c_indices.append(c_idx)
                        all_ibs_vals.append(ibs)
                    break
    
    # Create legend for methods
    legend_elements = [plt.scatter([], [], c=MODEL_COLORS.get(m, '#888888'), 
                                   s=100, label=m, edgecolors='white')
                      for m in ['SurvivalStacking'] + BASELINE_MODELS]
    ax.legend(handles=legend_elements, loc='lower left', fontsize=10, framealpha=0.9)
    
    # Add dataset marker legend
    for dataset, marker in dataset_markers.items():
        if dataset in all_data:
            ax.scatter([], [], marker=marker, c='gray', s=80, 
                      label=DATASET_CONFIG[dataset]['name'])
    
    # Formatting
    ax.set_xlabel('Integrated Brier Score (↓ better calibration)', fontsize=12)
    ax.set_ylabel('C-Index at Median Time (↑ better discrimination)', fontsize=12)
    ax.set_title(f'Discrimination vs Calibration Trade-off\nSurvival Stacking vs {tfm} Baselines',
                 fontsize=14, fontweight='bold', pad=15)
    
    # Dynamic axis limits with padding
    if all_c_indices and all_ibs_vals:
        c_min, c_max = min(all_c_indices), max(all_c_indices)
        ibs_min, ibs_max = min(all_ibs_vals), max(all_ibs_vals)
        
        c_range = c_max - c_min
        ibs_range = ibs_max - ibs_min
        
        ax.set_xlim(ibs_min - 0.15 * ibs_range, ibs_max + 0.15 * ibs_range)
        ax.set_ylim(c_min - 0.1 * c_range, c_max + 0.1 * c_range)
        
        # Quadrant lines
        c_mid = (c_min + c_max) / 2
        ibs_mid = (ibs_min + ibs_max) / 2
        ax.axhline(y=c_mid, color='gray', linestyle='--', alpha=0.5, zorder=1)
        ax.axvline(x=ibs_mid, color='gray', linestyle='--', alpha=0.5, zorder=1)
        
        # Ideal region
        ax.annotate('IDEAL\n(High C-Index, Low IBS)', 
                   xy=(ibs_min + 0.05 * ibs_range, c_max - 0.02 * c_range), 
                   fontsize=10, ha='left', color='green', fontweight='bold',
                   bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3))
    
    ax.grid(True, linestyle='--', alpha=0.3, zorder=0)
    ax.set_axisbelow(True)
    
    plt.tight_layout()
    output_path = plots_dir / f'plot3_discrimination_vs_calibration_{tfm}.png'
    plt.savefig(output_path, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def print_summary_table(all_data: Dict, model: str):
    """Print a comprehensive summary table of all results."""
    print("\n" + "="*100)
    print(f"COMPREHENSIVE RESULTS SUMMARY: Survival Stacking vs {model} Baselines")
    print("="*100)
    
    for dataset, data in all_data.items():
        config = DATASET_CONFIG[dataset]
        print(f"\n{'─'*100}")
        print(f"DATASET: {config['name']} ({config['description']})")
        print(f"{'─'*100}")
        
        header = f"{'Method':<20} {'Mode':<12} {'C-Index (q50)':<15} {'IBS':<12} {'Notes'}"
        print(header)
        print("-"*80)
        
        # Survival Stacking
        ss_results = data.get('survival_stacking', {})
        for mode, metrics in ss_results.items():
            c_idx = metrics.get('c_index_q50', 0)
            ibs = metrics.get('ibs', 0)
            if c_idx > 0:
                print(f"{'SurvivalStacking':<20} {mode:<12} {c_idx:<15.4f} {ibs:<12.4f} {'★ Our Method'}")
        
        # Baselines
        baselines = data.get('tfm_baselines', {})
        for model in BASELINE_MODELS:
            model_data = baselines.get(model, {})
            for mode in ['raw', 'deep', 'deep+raw']:
                if mode in model_data:
                    c_idx = model_data[mode].get('c_index_q50', 0)
                    ibs = model_data[mode].get('ibs', 0)
                    if c_idx > 0:
                        print(f"{model:<20} {mode:<12} {c_idx:<15.4f} {ibs:<12.4f}")
        
        # Best results summary
        print(f"\n{'Best Results for ' + config['name'] + ':'}")
        
        all_results = []
        for mode, metrics in ss_results.items():
            if metrics.get('c_index_q50', 0) > 0:
                all_results.append(('SurvivalStacking', mode, metrics['c_index_q50'], metrics['ibs']))
        
        for model, model_data in baselines.items():
            for mode, metrics in model_data.items():
                if metrics.get('c_index_q50', 0) > 0:
                    all_results.append((model, mode, metrics['c_index_q50'], metrics['ibs']))
        
        if all_results:
            # Best C-Index
            best_cidx = max(all_results, key=lambda x: x[2])
            print(f"  Best C-Index: {best_cidx[0]} ({best_cidx[1]}) = {best_cidx[2]:.4f}")
            
            # Best IBS
            best_ibs = min(all_results, key=lambda x: x[3])
            print(f"  Best IBS:     {best_ibs[0]} ({best_ibs[1]}) = {best_ibs[3]:.4f}")
    
    print("\n" + "="*100)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Visualize Survival Stacking Results',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python visualize_results.py                     # All datasets
    python visualize_results.py --dataset pbc       # Single dataset
        """
    )
    parser.add_argument(
        '--dataset', '-d',
        type=str,
        choices=['metabric', 'pbc', 'support', 'all'],
        default='all',
        help='Dataset to visualize (default: all)'
    )
    parser.add_argument(
        '--model', '-m',
        type=str,
        choices=['tarte', 'tabicl', 'tabpfn'],
        default='all',
        help='Model to look at'
    )
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()
    
    # Determine which datasets to process
    if args.dataset == 'all':
        datasets = ['metabric', 'pbc', 'support']
    else:
        datasets = [args.dataset]

    # Determine which model to look at
    if args.model == 'all':
        models = ['tarte', 'tabicl'] #, 'tabpfn']
    else:
        models = [args.model]
    
    # Create plots directory
    plots_dir = SURV_STACK_RESULTS_DIR / 'plots'
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*70)
    print("Survival Stacking Results Visualization")
    print("="*70)

    for model in models:
        # Load all data
        all_data = {}
        for dataset in datasets:
            print(f"\nLoading {dataset.upper()}...")

            data = {
                'survival_stacking': load_survival_stacking_results(dataset, model),
                'tfm_baselines': load_tfm_baseline_results(dataset, model)
            }

            if data['survival_stacking'] or data['tfm_baselines']:
                all_data[dataset] = data
                ss_modes = list(data['survival_stacking'].keys()) if data['survival_stacking'] else []
                bl_models = list(data['tfm_baselines'].keys()) if data['tfm_baselines'] else []
                print(f"  Survival Stacking modes: {ss_modes}")
                print(f"  Baseline models: {bl_models}")
            else:
                print(f"  No results found for {dataset}")

        if not all_data:
            print("\nNo results found! Run experiments first:")
            print("  sbatch survivalStacking/run_experiment.sbatch METABRIC deep+raw 5 20")
            return

        # Print summary table
        print_summary_table(all_data, model)

        # Generate plots
        plot1_method_comparison(all_data, plots_dir, model)
        plot2_temporal_performance(all_data, plots_dir, model)
        plot3_discrimination_vs_calibration(all_data, plots_dir,model)
    
    print(f"\n{'='*70}")
    print(f"All plots saved to: {plots_dir}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
