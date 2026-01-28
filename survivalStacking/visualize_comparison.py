#!/usr/bin/env python
"""
Survival Stacking Results Visualization
========================================

This script generates key insight plots comparing TabICl predictions against baselines.

1. PLOT 1: "Method Comparison by Dataset" (Discrimination)
   - Compares TabICL (direct inference) against baseline models:
     XGBoost, Logistic Regression, and LightGBM
   - Each baseline is evaluated on raw features and TabICL embeddings
   - Primary metric: C-Index at median survival time (q50)
   - Results are grouped by dataset (e.g. METABRIC, PBC, SUPPORT)

2. PLOT 2: "Discrimination vs Calibration Trade-off"
   - Scatter plot of C-Index (discrimination) vs Integrated Brier Score (IBS)
   - Each point corresponds to a method–dataset pair
   - Visualizes the trade-off between predictive accuracy and calibration
   - Ideal region: high C-Index and low IBS (upper-left corner)
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
SURV_STACK_RESULTS_DIR = PROJECT_ROOT / 'results' / 'tabicl_comparison'

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
BASELINE_MODELS = ['XGBoost', 'Logistic Regression', 'LGBMClassifier']

# Colors
MODEL_COLORS = {
    'tabicl_direct': '#e74c3c',  # Red - our method
    'xgboost_raw': '#ff7f0e',
    'xgboost_embeddings': '#ffbb78',
    'logistic_raw': '#2ca02c',
    'logistic_embeddings': '#baff6d',
    'lightgbm_raw': '#1f77b4',
    'lightgbm_embeddings': '#98eb85'
}

def load_survival_stacking_results(dataset: str, model: str) -> Optional[Dict]:
    """
    Load TabICL / XGBoost experiment results.
    """
    results = {}
    pattern = f'{dataset.upper()}_comparison_*.json'
    matches = list(SURV_STACK_RESULTS_DIR.glob(pattern))
    if matches:
        filepath = matches[0]
        with open(filepath, 'r') as f:
            data = json.load(f)
            for method, metrics in data['summary'].items():
                results[method] = {
                    'c_index_q50': metrics.get('c_index_q50_mean', 0),
                    'ibs': metrics.get('ibs_mean', 0),
                    'std_c_index_q50': metrics.get('c_index_q50_std', 0),
                    'std_ibs': metrics.get('ibs_std', 0),
                }

    return results if results else None

def plot1_method_comparison(all_data: Dict, plots_dir: Path):
    """
    PLOT 1: Baseline Comparison Across Datasets

    Compares Survival Stacking results TFM vs baseline for each dataset.
    """
    print("\n--- Plot 1: Baseline Comparison by Dataset ---")

    datasets = [d for d in ['metabric', 'pbc', 'support'] if d in all_data]
    if not datasets:
        print("No data available for plotting")
        return

    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_facecolor('#fafafa')

    x = np.arange(len(datasets))
    width = 0.12

    # List of methods in the order you want to appear in the plot
    methods = [
        'tabicl_direct',
        'xgboost_raw', 'xgboost_embeddings',
        'logistic_raw', 'logistic_embeddings',
        'lightgbm_raw', 'lightgbm_embeddings'
    ]

    # Initialize dictionary to hold C-Index values per dataset
    method_data = {
        m: {
            'mean': [],
            'std': []
        }
        for m in methods
    }

    datasets = list(all_data.keys())

    for dataset in datasets:
        data = all_data[dataset]
        for m in methods:
            # Get c_index_q50 if exists, else 0
            mean_val = data.get(m, {}).get('c_index_q50', 0)
            method_data[m]['mean'].append(mean_val)
            std_val = data.get(m, {}).get('std_c_index_q50', 0)
            method_data[m]['std'].append(std_val)

    # Plot bars
    colors = [MODEL_COLORS.get(m, '#888888') for m in methods]
    bars_list = []

    for i, method in enumerate(methods):
        offset = (i - len(methods) / 2 + 0.5) * width
        bars = ax.bar(x + offset, method_data[method]['mean'], width, yerr=method_data[method]['std'], capsize=3,
                      label=method, color=colors[i], edgecolor='white', linewidth=1.5)
        bars_list.append(bars)

        # Add value labels
        for j, (bar, mean, std) in enumerate(zip(bars, method_data[method]['mean'], method_data[method]['std'])):
            if mean > 0:
                y = mean + std
                ax.annotate(f'{mean:.3f}',
                            xy=(bar.get_x() + bar.get_width() / 2, y),
                            xytext=(0, 5), textcoords="offset points",
                            ha='center', va='bottom', fontsize=8, fontweight='bold',
                            rotation=45)

    # Formatting
    ax.set_title(f'TFM vs Baselines\n(C-Index at Median Survival Time, 5-Fold CV)',
                 fontsize=14, fontweight='bold', pad=15)
    ax.set_ylabel('C-Index (↑ better)', fontsize=12)
    ax.set_xlabel('Dataset', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels([DATASET_CONFIG[d]['name'] for d in datasets], fontsize=11, fontweight='medium')
    ax.legend(loc='upper right', fontsize=9, ncol=2, framealpha=0.9)

    # Dynamic y-axis
    all_vals = [v for m in methods for v in method_data[m]['mean'] if v > 0]
    if all_vals:
        ax.set_ylim(min(all_vals) - 0.03, max(all_vals) + 0.05)

    ax.grid(axis='y', linestyle='--', alpha=0.4, zorder=0)
    ax.set_axisbelow(True)

    # Add insight box
    insights = []
    for i, dataset in enumerate(datasets):
        tabicl_val = method_data['tabicl_direct']['mean'][i]

        # Compare each baseline to tabicl_direct and find the best one
        best_baseline = max((method_data[m]['mean'][i], m) for m in ['xgboost_raw', 'xgboost_embeddings',
                                                             'logistic_raw', 'logistic_embeddings',
                                                             'lightgbm_raw', 'lightgbm_embeddings'])
        if best_baseline[0] > 0:
            diff = (tabicl_val - best_baseline[0]) * 100
            status = "✓" if diff > 0 else "✗"
            insights.append(f"{DATASET_CONFIG[dataset]['name']}: {status} {diff:+.1f}% vs {best_baseline[1]}")

    if insights:
        insight_text = "Survival Stacking vs Best Baseline:\n" + "\n".join(insights)
        ax.text(0.02, 0.02, insight_text, transform=ax.transAxes, fontsize=9,
                verticalalignment='bottom', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.tight_layout()
    output_path = plots_dir / f'plot1_method_comparison.png'
    plt.savefig(output_path, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()

def plot2_discrimination_vs_calibration(all_data: Dict, plots_dir: Path):
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
        ss_data = data.get('tabicl_direct', {})
        if ss_data.get('c_index_q50', 0) > 0 and ss_data.get('ibs', 0) > 0:
            c_idx = ss_data['c_index_q50']
            ibs = ss_data['ibs']
            ax.scatter(ibs, c_idx, c=MODEL_COLORS['tabicl_direct'], marker=marker,
                       s=200, edgecolors='black', linewidths=2, zorder=5,
                       label=f'SurvStack ({dataset_name})' if dataset == list(all_data.keys())[0] else '')
            ax.annotate(f'SS-{dataset_name}', (ibs, c_idx), xytext=(5, 5),
                        textcoords='offset points', fontsize=8, fontweight='bold')
            all_c_indices.append(c_idx)
            all_ibs_vals.append(ibs)

        # Baselines
        for model in ['xgboost_raw', 'xgboost_embeddings',
                      'logistic_raw', 'logistic_embeddings',
                      'lightgbm_raw', 'lightgbm_embeddings']:
            model_data = data.get(model, {})
            # Use best mode available
            c_idx = model_data.get('c_index_q50', 0)
            ibs = model_data.get('ibs', 0)
            if c_idx > 0 and ibs > 0:
                ax.scatter(ibs, c_idx, c=MODEL_COLORS.get(model, '#888888'),
                           marker=marker, s=120, edgecolors='white', linewidths=1.5,
                           alpha=0.7, zorder=3)
                ax.annotate(f'{model}', (ibs, c_idx), xytext=(3, 3),
                            textcoords='offset points', fontsize=7, alpha=0.8)
                all_c_indices.append(c_idx)
                all_ibs_vals.append(ibs)

    # Create legend for methods
    legend_elements = [plt.scatter([], [], c=MODEL_COLORS.get(m, '#888888'),
                                   s=100, label=m, edgecolors='white')
                       for m in ['tabicl_direct'] + ['xgboost_raw', 'xgboost_embeddings',
                                                     'logistic_raw', 'logistic_embeddings',
                                                     'lightgbm_raw', 'lightgbm_embeddings']]
    ax.legend(handles=legend_elements, loc='lower left', fontsize=10, framealpha=0.9)

    # Add dataset marker legend
    for dataset, marker in dataset_markers.items():
        if dataset in all_data:
            ax.scatter([], [], marker=marker, c='gray', s=80,
                       label=DATASET_CONFIG[dataset]['name'])

    # Formatting
    ax.set_xlabel('Integrated Brier Score (↓ better calibration)', fontsize=12)
    ax.set_ylabel('C-Index at Median Time (↑ better discrimination)', fontsize=12)
    ax.set_title(f'Discrimination vs Calibration Trade-off\nSurvival Stacking vs Baselines',
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
    output_path = plots_dir / f'plot3_discrimination_vs_calibration.png'
    plt.savefig(output_path, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def print_summary_table(all_data: Dict, model: str):
    """Print a comprehensive summary table of all results."""
    print("\n" + "=" * 100)
    print(f"COMPREHENSIVE RESULTS SUMMARY: Survival Stacking vs {model} Baselines")
    print("=" * 100)

    for dataset, data in all_data.items():
        config = DATASET_CONFIG[dataset]
        print(f"\n{'─' * 100}")
        print(f"DATASET: {config['name']} ({config['description']})")
        print(f"{'─' * 100}")

        header = f"{'Mode':<20} {'C-Index (q50)':<15} {'IBS':<12}"
        print(header)
        print("-" * 80)

        # Survival Stacking Results
        for mode, metrics in data.items():
            c_idx = metrics.get('c_index_q50', 0)
            ibs = metrics.get('ibs', 0)
            if c_idx > 0:
                print(f"{mode:<20} {c_idx:<15.4f} {ibs:<12.4f}")

        # Best results summary
        print(f"\n{'Best Results for ' + config['name'] + ':'}")

        all_results = []
        for mode, metrics in data.items():
            if metrics.get('c_index_q50', 0) > 0:
                all_results.append((mode, metrics['c_index_q50'], metrics['ibs']))

        if all_results:
            # Best C-Index
            best_cidx = max(all_results, key=lambda x: x[2])
            print(f"  Best C-Index: {best_cidx[0]} = {best_cidx[1]:.4f}")

            # Best IBS
            best_ibs = min(all_results, key=lambda x: x[2])
            print(f"  Best IBS:     {best_ibs[0]} = {best_ibs[2]:.4f}")

    print("\n" + "=" * 100)


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
        models = ['tabicl']  # , 'tabpfn']
    else:
        models = [args.model]

    # Create plots directory
    plots_dir = SURV_STACK_RESULTS_DIR / 'plots'
    plots_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Survival Stacking Results Visualization")
    print("=" * 70)

    for model in models:
        # Load all data
        all_data = {}
        for dataset in datasets:
            print(f"\nLoading {dataset.upper()}...")

            data = load_survival_stacking_results(dataset, model)

            if data:
                all_data[dataset] = data
                ss_modes = list(data.keys()) if data else []
                print(f"  Modes: {ss_modes}")
            else:
                print(f"  No results found for {dataset}")

        if not all_data:
            print("\nNo results found! Run experiments first:")
            print("  sbatch survivalStacking/run_experiment.sbatch METABRIC deep+raw 5 20")
            return

        # Print summary table
        print_summary_table(all_data, model)

        # Generate plots
        plot1_method_comparison(all_data, plots_dir)
        plot2_discrimination_vs_calibration(all_data, plots_dir)

    print(f"\n{'=' * 70}")
    print(f"All plots saved to: {plots_dir}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
