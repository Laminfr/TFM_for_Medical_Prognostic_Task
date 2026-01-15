#!/usr/bin/env python
"""
Survival Stacking Results Visualization - Full Benchmark Version
================================================================

This script visualizes results from the full benchmark comparing:

Survival Stacking Methods:
  1. SurvStack-Raw: Survival Stacking with raw features + XGBoost
  2. SurvStack-Emb: Survival Stacking with raw + TabICL embeddings + XGBoost  
  3. SurvStack-TabICL: Survival Stacking with raw features + TabICL classifier

Baselines (non-stacked):
  4. CoxPH: Standard Cox Proportional Hazards
  5. XGBoost: XGBoost survival model
  6. DeepSurv: Neural network survival model

Plot:
  - 3 dataset groups (METABRIC, PBC, SUPPORT)
  - 6 columns per group (3 SurvStack variants + 3 baselines)
  - Primary metric: C-Index at median survival time (q50)

Usage:
    python -m survivalStacking.visualize_benchmark_results
    python -m survivalStacking.visualize_benchmark_results --dataset pbc
"""

import argparse
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ==========================================
# CONFIGURATION
# ==========================================
PROJECT_ROOT = Path(__file__).parent.parent
RESULTS_DIR = PROJECT_ROOT / 'results' / 'survival_stacking'

# Dataset configurations
DATASET_CONFIG = {
    'metabric': {
        'name': 'METABRIC',
        'description': 'Breast Cancer',
        'n_features': 9,
    },
    'pbc': {
        'name': 'PBC',
        'description': 'Primary Biliary Cirrhosis',
        'n_features': 25,
    },
    'support': {
        'name': 'SUPPORT',
        'description': 'ICU Study',
        'n_features': 24,
    }
}

# Method display names and order
METHODS = [
    ('survstack_raw', 'SurvStack-Raw'),
    ('survstack_emb', 'SurvStack-Emb'),
    ('survstack_tabicl', 'SurvStack-TabICL'),
    ('coxph', 'CoxPH'),
    ('xgboost', 'XGBoost'),
    ('deepsurv', 'DeepSurv'),
]

# Colors - distinguish survival stacking (warm) from baselines (cool)
METHOD_COLORS = {
    'survstack_raw': '#e74c3c',      # Red
    'survstack_emb': '#c0392b',      # Dark Red
    'survstack_tabicl': '#f39c12',   # Orange
    'coxph': '#3498db',              # Blue
    'xgboost': '#9b59b6',            # Purple
    'deepsurv': '#2ecc71',           # Green
}

# Hatching patterns for baselines (to distinguish from survival stacking)
METHOD_HATCHES = {
    'survstack_raw': '',
    'survstack_emb': '',
    'survstack_tabicl': '',
    'coxph': '///',
    'xgboost': '///',
    'deepsurv': '///',
}


def load_benchmark_results(dataset: str) -> Optional[Dict]:
    """Load full benchmark results for a dataset."""
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


def load_all_results() -> Dict[str, Dict]:
    """Load results for all datasets."""
    all_data = {}
    
    for dataset in ['metabric', 'pbc', 'support']:
        data = load_benchmark_results(dataset)
        if data:
            all_data[dataset] = data
            print(f"Loaded {dataset}: {list(data.get('results', {}).keys())}")
        else:
            print(f"No results found for {dataset}")
    
    return all_data


def plot_main_comparison(all_data: Dict, plots_dir: Path):
    """
    Main comparison plot: 3 dataset groups x 6 methods
    
    This is the primary visualization showing C-Index at median time for all methods.
    """
    print("\n=== Generating Main Comparison Plot ===")
    
    datasets = [d for d in ['metabric', 'pbc', 'support'] if d in all_data]
    if not datasets:
        print("No data available for plotting")
        return
    
    fig, ax = plt.subplots(figsize=(16, 8))
    ax.set_facecolor('#fafafa')
    
    n_datasets = len(datasets)
    n_methods = len(METHODS)
    
    # Bar positioning
    bar_width = 0.12
    group_width = bar_width * n_methods + 0.15  # Extra space between groups
    
    # Collect data
    method_data = {m[0]: {'values': [], 'stds': []} for m in METHODS}
    
    for dataset in datasets:
        results = all_data[dataset].get('results', {})
        for method_key, method_name in METHODS:
            if method_key in results:
                method_data[method_key]['values'].append(
                    results[method_key]['mean'].get('c_index_q50', 0)
                )
                method_data[method_key]['stds'].append(
                    results[method_key]['std'].get('c_index_q50', 0)
                )
            else:
                method_data[method_key]['values'].append(0)
                method_data[method_key]['stds'].append(0)
    
    # Plot bars
    x = np.arange(n_datasets) * group_width
    
    for i, (method_key, method_name) in enumerate(METHODS):
        offset = (i - n_methods / 2 + 0.5) * bar_width
        values = method_data[method_key]['values']
        stds = method_data[method_key]['stds']
        
        bars = ax.bar(
            x + offset, values, bar_width,
            yerr=stds,
            label=method_name,
            color=METHOD_COLORS[method_key],
            hatch=METHOD_HATCHES[method_key],
            edgecolor='white',
            linewidth=1.5,
            capsize=3,
            error_kw={'linewidth': 1, 'capthick': 1}
        )
        
        # Add value labels on bars
        for bar, val, std in zip(bars, values, stds):
            if val > 0:
                ax.annotate(
                    f'{val:.3f}',
                    xy=(bar.get_x() + bar.get_width() / 2, val + std + 0.005),
                    ha='center', va='bottom',
                    fontsize=7, fontweight='bold',
                    rotation=45
                )
    
    # Formatting
    ax.set_title(
        'Survival Stacking Benchmark: C-Index at Median Survival Time\n'
        '(5-Fold Cross-Validation, Balanced Class Weighting)',
        fontsize=14, fontweight='bold', pad=15
    )
    ax.set_ylabel('C-Index (↑ better)', fontsize=12)
    ax.set_xlabel('Dataset', fontsize=12)
    
    # X-axis labels
    ax.set_xticks(x)
    ax.set_xticklabels(
        [f"{DATASET_CONFIG[d]['name']}\n({DATASET_CONFIG[d]['description']})" 
         for d in datasets],
        fontsize=11, fontweight='medium'
    )
    
    # Legend - split into two rows
    survstack_patches = [
        mpatches.Patch(color=METHOD_COLORS[m[0]], label=m[1])
        for m in METHODS[:3]
    ]
    baseline_patches = [
        mpatches.Patch(color=METHOD_COLORS[m[0]], hatch='///', label=m[1])
        for m in METHODS[3:]
    ]
    
    legend1 = ax.legend(
        handles=survstack_patches,
        title='Survival Stacking',
        loc='upper left',
        fontsize=9,
        title_fontsize=10
    )
    ax.add_artist(legend1)
    ax.legend(
        handles=baseline_patches,
        title='Baselines',
        loc='upper right',
        fontsize=9,
        title_fontsize=10
    )
    
    # Y-axis limits
    all_vals = [v for m in method_data.values() for v in m['values'] if v > 0]
    if all_vals:
        y_min = min(all_vals) - 0.05
        y_max = max(all_vals) + 0.08
        ax.set_ylim(y_min, y_max)
    
    ax.grid(axis='y', linestyle='--', alpha=0.4, zorder=0)
    ax.set_axisbelow(True)
    
    # Add summary box
    summary_lines = []
    for dataset in datasets:
        results = all_data[dataset].get('results', {})
        survstack_vals = []
        baseline_vals = []
        
        for method_key, _ in METHODS[:3]:
            if method_key in results:
                survstack_vals.append(results[method_key]['mean'].get('c_index_q50', 0))
        for method_key, _ in METHODS[3:]:
            if method_key in results:
                baseline_vals.append(results[method_key]['mean'].get('c_index_q50', 0))
        
        if survstack_vals and baseline_vals:
            best_ss = max(survstack_vals)
            best_bl = max(baseline_vals)
            diff = (best_ss - best_bl) * 100
            icon = '✓' if diff > 0 else '✗'
            summary_lines.append(
                f"{DATASET_CONFIG[dataset]['name']}: {icon} {diff:+.1f}% vs best baseline"
            )
    
    if summary_lines:
        summary_text = "Best SurvStack vs Best Baseline:\n" + "\n".join(summary_lines)
        ax.text(
            0.02, 0.02, summary_text,
            transform=ax.transAxes, fontsize=9,
            verticalalignment='bottom',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        )
    
    plt.tight_layout()
    output_path = plots_dir / 'benchmark_comparison.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_temporal_performance(all_data: Dict, plots_dir: Path):
    """
    Plot C-Index at different time quantiles (q25, q50, q75) for SurvStack methods.
    """
    print("\n=== Generating Temporal Performance Plot ===")
    
    datasets = [d for d in ['metabric', 'pbc', 'support'] if d in all_data]
    if not datasets:
        return
    
    fig, axes = plt.subplots(1, len(datasets), figsize=(5 * len(datasets), 6))
    if len(datasets) == 1:
        axes = [axes]
    
    quantiles = ['q25', 'q50', 'q75']
    quantile_labels = ['Early (Q25)', 'Median (Q50)', 'Late (Q75)']
    quantile_colors = ['#ff7f0e', '#2ca02c', '#9467bd']
    
    survstack_methods = METHODS[:3]  # Only SurvStack variants
    
    for ax, dataset in zip(axes, datasets):
        ax.set_facecolor('#fafafa')
        results = all_data[dataset].get('results', {})
        
        x = np.arange(len(survstack_methods))
        width = 0.25
        
        for i, (q, ql, qc) in enumerate(zip(quantiles, quantile_labels, quantile_colors)):
            vals = []
            stds = []
            for method_key, _ in survstack_methods:
                if method_key in results:
                    vals.append(results[method_key]['mean'].get(f'c_index_{q}', 0))
                    stds.append(results[method_key]['std'].get(f'c_index_{q}', 0))
                else:
                    vals.append(0)
                    stds.append(0)
            
            bars = ax.bar(
                x + (i - 1) * width, vals, width,
                yerr=stds,
                label=ql, color=qc,
                edgecolor='white', alpha=0.85,
                capsize=3
            )
            
            for bar, val in zip(bars, vals):
                if val > 0:
                    ax.annotate(
                        f'{val:.3f}',
                        xy=(bar.get_x() + bar.get_width() / 2, val),
                        xytext=(0, 3), textcoords='offset points',
                        ha='center', va='bottom', fontsize=7, rotation=45
                    )
        
        ax.set_title(f"{DATASET_CONFIG[dataset]['name']}", fontsize=13, fontweight='bold')
        ax.set_ylabel('C-Index', fontsize=11)
        ax.set_xticks(x)
        ax.set_xticklabels([m[1] for m in survstack_methods], fontsize=9, rotation=15, ha='right')
        ax.legend(fontsize=8, loc='lower right')
        ax.grid(axis='y', linestyle='--', alpha=0.4)
        ax.set_axisbelow(True)
        
        # Dynamic y-limits
        all_vals = []
        for method_key, _ in survstack_methods:
            if method_key in results:
                for q in quantiles:
                    all_vals.append(results[method_key]['mean'].get(f'c_index_{q}', 0))
        valid_vals = [v for v in all_vals if v > 0]
        if valid_vals:
            ax.set_ylim(min(valid_vals) - 0.03, max(valid_vals) + 0.05)
    
    plt.suptitle(
        'C-Index Across Time Horizons\n(Early vs Median vs Late Survival)',
        fontsize=14, fontweight='bold'
    )
    plt.tight_layout()
    
    output_path = plots_dir / 'benchmark_temporal.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_discrimination_vs_calibration(all_data: Dict, plots_dir: Path):
    """
    Scatter plot of C-Index vs IBS for all methods.
    """
    print("\n=== Generating Discrimination vs Calibration Plot ===")
    
    fig, ax = plt.subplots(figsize=(12, 9))
    ax.set_facecolor('#fafafa')
    
    dataset_markers = {'metabric': 'o', 'pbc': 's', 'support': '^'}
    marker_sizes = {'survstack_raw': 200, 'survstack_emb': 200, 'survstack_tabicl': 200,
                    'coxph': 120, 'xgboost': 120, 'deepsurv': 120}
    
    all_c_indices = []
    all_ibs_vals = []
    
    for dataset, data in all_data.items():
        marker = dataset_markers.get(dataset, 'o')
        results = data.get('results', {})
        
        for method_key, method_name in METHODS:
            if method_key not in results:
                continue
            
            c_idx = results[method_key]['mean'].get('c_index_q50', 0)
            ibs = results[method_key]['mean'].get('ibs', 0)
            
            if c_idx > 0 and ibs > 0:
                ax.scatter(
                    ibs, c_idx,
                    c=METHOD_COLORS[method_key],
                    marker=marker,
                    s=marker_sizes.get(method_key, 150),
                    edgecolors='black' if method_key.startswith('survstack') else 'white',
                    linewidths=2 if method_key.startswith('survstack') else 1.5,
                    zorder=5 if method_key.startswith('survstack') else 3,
                    alpha=0.9
                )
                
                # Label
                label = f"{method_name[:3]}"
                ax.annotate(
                    label, (ibs, c_idx),
                    xytext=(5, 5), textcoords='offset points',
                    fontsize=7, alpha=0.8
                )
                
                all_c_indices.append(c_idx)
                all_ibs_vals.append(ibs)
    
    # Create legends
    # Method legend
    method_handles = [
        plt.scatter([], [], c=METHOD_COLORS[m[0]], s=100, label=m[1],
                   edgecolors='black' if m[0].startswith('survstack') else 'white')
        for m in METHODS
    ]
    legend1 = ax.legend(
        handles=method_handles,
        title='Methods',
        loc='lower left',
        fontsize=9
    )
    ax.add_artist(legend1)
    
    # Dataset legend
    dataset_handles = [
        plt.scatter([], [], c='gray', marker=m, s=80, label=DATASET_CONFIG[d]['name'])
        for d, m in dataset_markers.items() if d in all_data
    ]
    ax.legend(
        handles=dataset_handles,
        title='Datasets',
        loc='lower right',
        fontsize=9
    )
    
    # Formatting
    ax.set_xlabel('Integrated Brier Score (↓ better calibration)', fontsize=12)
    ax.set_ylabel('C-Index at Median Time (↑ better discrimination)', fontsize=12)
    ax.set_title(
        'Discrimination vs Calibration Trade-off\nSurvival Stacking vs Baselines',
        fontsize=14, fontweight='bold', pad=15
    )
    
    # Axis limits and ideal region
    if all_c_indices and all_ibs_vals:
        c_min, c_max = min(all_c_indices), max(all_c_indices)
        ibs_min, ibs_max = min(all_ibs_vals), max(all_ibs_vals)
        
        c_range = c_max - c_min
        ibs_range = ibs_max - ibs_min
        
        ax.set_xlim(ibs_min - 0.15 * ibs_range, ibs_max + 0.15 * ibs_range)
        ax.set_ylim(c_min - 0.1 * c_range, c_max + 0.1 * c_range)
        
        # Ideal region annotation
        ax.annotate(
            'IDEAL\n(High C-Index, Low IBS)',
            xy=(ibs_min + 0.02 * ibs_range, c_max),
            fontsize=10, ha='left', color='green', fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3)
        )
    
    ax.grid(True, linestyle='--', alpha=0.3, zorder=0)
    ax.set_axisbelow(True)
    
    plt.tight_layout()
    output_path = plots_dir / 'benchmark_tradeoff.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def print_summary_table(all_data: Dict):
    """Print comprehensive results table."""
    print(f"\n{'='*100}")
    print("COMPREHENSIVE BENCHMARK RESULTS")
    print(f"{'='*100}")
    
    for dataset, data in all_data.items():
        config = DATASET_CONFIG[dataset]
        results = data.get('results', {})
        
        print(f"\n{'─'*100}")
        print(f"DATASET: {config['name']} ({config['description']})")
        print(f"{'─'*100}")
        
        print(f"{'Method':<25} {'C-Index (q50)':<18} {'IBS':<15} {'Notes'}")
        print("-" * 80)
        
        for method_key, method_name in METHODS:
            if method_key not in results:
                continue
            
            c_idx = results[method_key]['mean'].get('c_index_q50', 0)
            c_std = results[method_key]['std'].get('c_index_q50', 0)
            ibs = results[method_key]['mean'].get('ibs', 0)
            ibs_std = results[method_key]['std'].get('ibs', 0)
            
            note = ''
            if method_key.startswith('survstack'):
                note = '★ Our Method'
            
            print(f"{method_name:<25} {c_idx:.4f} ± {c_std:.4f}    {ibs:.4f} ± {ibs_std:.4f}   {note}")
        
        # Best results
        print(f"\nBest Results for {config['name']}:")
        
        all_results = []
        for method_key, method_name in METHODS:
            if method_key in results:
                c_idx = results[method_key]['mean'].get('c_index_q50', 0)
                ibs = results[method_key]['mean'].get('ibs', 0)
                if c_idx > 0:
                    all_results.append((method_name, c_idx, ibs))
        
        if all_results:
            best_cidx = max(all_results, key=lambda x: x[1])
            best_ibs = min(all_results, key=lambda x: x[2])
            print(f"  Best C-Index: {best_cidx[0]} = {best_cidx[1]:.4f}")
            print(f"  Best IBS:     {best_ibs[0]} = {best_ibs[2]:.4f}")
    
    print(f"\n{'='*100}")


def parse_args():
    parser = argparse.ArgumentParser(description='Visualize Survival Stacking Benchmark Results')
    parser.add_argument('--dataset', '-d', type=str, choices=['metabric', 'pbc', 'support', 'all'],
                        default='all', help='Dataset to visualize')
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Create plots directory
    plots_dir = RESULTS_DIR / 'plots'
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("Survival Stacking Benchmark Visualization")
    print("=" * 70)
    
    # Load results
    all_data = load_all_results()
    
    if not all_data:
        print("\nNo results found! Run the benchmark first:")
        print("  python -m survivalStacking.run_full_benchmark --dataset all --cv 5")
        return
    
    # Filter by dataset if specified
    if args.dataset != 'all':
        if args.dataset in all_data:
            all_data = {args.dataset: all_data[args.dataset]}
        else:
            print(f"No results for {args.dataset}")
            return
    
    # Print summary table
    print_summary_table(all_data)
    
    # Generate plots
    plot_main_comparison(all_data, plots_dir)
    plot_temporal_performance(all_data, plots_dir)
    plot_discrimination_vs_calibration(all_data, plots_dir)
    
    print(f"\n{'='*70}")
    print(f"All plots saved to: {plots_dir}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
