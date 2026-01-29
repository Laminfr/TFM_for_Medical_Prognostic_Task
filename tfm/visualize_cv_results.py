#!/usr/bin/env python
"""
Unified CV Results Visualization for Embedding Methods
======================================================

This script generates 3 key insight plots from the cross-validation results:

1. PLOT 1: "Embedding Impact by Model" (Discrimination)
   - Compares Raw vs Deep vs Deep+Raw for each model
   - Shows which models benefit from embeddings
   - Uses C-Index at median time (q50) as the primary metric
   
2. PLOT 2: "Performance Across Time Horizons" (Temporal Robustness)
   - Shows C-Index at early (q25), median (q50), and late (q75) times
   - Reveals if models maintain performance across different time horizons
   - Important for clinical utility (short vs long-term predictions)
   
3. PLOT 3: "Discrimination vs Calibration Trade-off"
   - Plots C-Index (discrimination) against IBS (calibration)
   - Shows which models achieve good ranking AND probability estimates
   - Ideal: high C-Index AND low IBS (upper-left corner)

Supported embedding methods:
- tabicl: TabICL embeddings (512D)
- tarte: TARTE embeddings

Usage:
    python visualize_cv_results.py --method tabicl --dataset metabric
    python visualize_cv_results.py --method tarte --dataset pbc
    python visualize_cv_results.py --method tabicl --dataset support
"""

import argparse
import matplotlib.pyplot as plt
import numpy as np
import json
import os
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ==========================================
# CONFIGURATION
# ==========================================

# Dataset-specific configurations
DATASET_CONFIG = {
    'metabric': {
        'name': 'METABRIC',
        'cv_dir': 'cv_metabric',
        'n_features': 9,
        'description': 'Breast Cancer Dataset'
    },
    'pbc': {
        'name': 'PBC',
        'cv_dir': 'cv_pbc',
        'n_features': 25,
        'description': 'Primary Biliary Cirrhosis'
    },
    'support': {
        'name': 'SUPPORT',
        'cv_dir': 'cv_support',
        'n_features': 24,
        'description': 'SUPPORT Study Dataset'
    },
    'gbsg': {
        'name': 'GBSG',
        'cv_dir': 'cv_gbsg',
        'n_features': 7,
        'description': 'German Breast Cancer Study'
    },
    'seer': {
        'name': 'SEER',
        'cv_dir': 'cv_seer',
        'n_features': 26,
        'description': 'U.S. cancer incidence data'
    }
}

# Models that completed successfully with all modes
MODELS_EMBEDDING = ['CoxPH', 'RSF', 'XGBoost', 'DeepSurv']
# All models (including those that only have raw results)
MODELS_ALL = ['CoxPH', 'RSF', 'XGBoost', 'NFG', 'DeSurv', 'DeepSurv']

# Model name mapping
MODEL_NAME_MAP = {
    'NFG': 'NeuralFineGray',
    'NeuralFineGray': 'NeuralFineGray',
    'CoxPH': 'CoxPH',
    'RSF': 'RSF',
    'XGBoost': 'XGBoost',
    'DeSurv': 'DeSurv',
    'DeepSurv': 'DeepSurv'
}

# Colors
MODE_COLORS = {
    'Raw': '#4a90e2',       # Blue
    'Deep': '#d9534f',      # Red  
    'Deep+Raw': '#5cb85c'   # Green
}

QUANTILE_COLORS = {
    'q25': '#ff7f0e',  # Orange - Early
    'q50': '#2ca02c',  # Green - Median
    'q75': '#9467bd'   # Purple - Late
}

# Embedding dimension info
EMBEDDING_INFO = {
    'tabicl': {'dim': 512, 'display_name': 'TabICL'},
    'tarte': {'dim': 256, 'display_name': 'TARTE'}  # Adjust if different
}


def get_results_dir(embedding_method):
    """Get the results directory for an embedding method."""
    return PROJECT_ROOT / f'results/{embedding_method}'


def load_cv_results(cv_results_dir):
    """
    Load cross-validation results from JSON files.
    Returns a nested dict: data[Mode][Model][Metric]
    """
    data = {'Raw': {}, 'Deep': {}, 'Deep+Raw': {}}
    
    if not cv_results_dir.exists():
        print(f"CV Results directory not found: {cv_results_dir}")
        return None
    
    # Find the most recent final results file
    final_files = sorted(cv_results_dir.glob('cv_results_final_*.json'), 
                        key=os.path.getmtime, reverse=True)
    
    if not final_files:
        final_files = sorted(cv_results_dir.glob('cv_results_*.json'),
                            key=os.path.getmtime, reverse=True)
    
    if not final_files:
        print("No CV results files found!")
        return None
    
    results_file = final_files[0]
    print(f"Loading CV results from: {results_file.name}")
    
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    # Organize by mode and model
    for r in results:
        if r.get('status') != 'success':
            continue
            
        mode = r.get('mode', 'raw')
        model = r.get('model')
        
        mode_key = {'raw': 'Raw', 'deep': 'Deep', 'deep+raw': 'Deep+Raw'}.get(mode, mode)
        
        if mode_key in data:
            data[mode_key][model] = {k: v for k, v in r.items() 
                                      if k not in ['model', 'mode', 'dataset', 'timestamp', 'status', 'embedding_method']}
    
    return data


def plot1_embedding_impact(data, plots_dir, dataset_config, embedding_method):
    """
    PLOT 1: Embedding Impact by Model
    
    Shows the core research question: Does the embedding improve survival models?
    Compares 3 feature representations for each model.
    """
    method_info = EMBEDDING_INFO.get(embedding_method, {'dim': 512, 'display_name': embedding_method.upper()})
    method_name = method_info['display_name']
    emb_dim = method_info['dim']
    
    print(f"\n--- Plot 1: {method_name} Impact by Model ---")
    
    dataset_name = dataset_config['name']
    n_features = dataset_config['n_features']
    
    models = MODELS_EMBEDDING
    x = np.arange(len(models))
    width = 0.25
    
    raw_vals = [data['Raw'].get(m, {}).get('c_index_q50', 0) for m in models]
    deep_vals = [data['Deep'].get(m, {}).get('c_index_q50', 0) for m in models]
    dr_vals = [data['Deep+Raw'].get(m, {}).get('c_index_q50', 0) for m in models]
    
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.set_facecolor('#fafafa')
    
    # Plot bars
    bars1 = ax.bar(x - width, raw_vals, width, label=f'Raw Features ({n_features}D)', 
                   color=MODE_COLORS['Raw'], edgecolor='white', linewidth=1.5)
    bars2 = ax.bar(x, deep_vals, width, label=f'{method_name} ({emb_dim}D)', 
                   color=MODE_COLORS['Deep'], edgecolor='white', linewidth=1.5)
    bars3 = ax.bar(x + width, dr_vals, width, label=f'{method_name} + Raw ({emb_dim+n_features}D)', 
                   color=MODE_COLORS['Deep+Raw'], edgecolor='white', linewidth=1.5)
    
    # Formatting
    ax.set_title(f'{method_name} Impact on Survival Models - {dataset_name}\n(C-Index at Median Survival Time, 5-Fold CV)', 
                 fontsize=14, fontweight='bold', pad=15)
    ax.set_ylabel('C-Index (↑ better)', fontsize=12)
    ax.set_xlabel('Survival Model', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=11, fontweight='medium')
    ax.legend(loc='upper right', fontsize=10, framealpha=0.9)
    
    # Y-axis limits
    all_vals = [v for v in raw_vals + deep_vals + dr_vals if v > 0]
    if all_vals:
        y_min = min(all_vals) - 0.02
        y_max = max(all_vals) + 0.03
        ax.set_ylim(y_min, y_max)
    
    ax.grid(axis='y', linestyle='--', alpha=0.4, zorder=0)
    ax.set_axisbelow(True)
    
    # Add value labels with delta indicators
    def add_labels_with_delta(bars, vals, ref_vals=None):
        for i, (bar, val) in enumerate(zip(bars, vals)):
            if val == 0:
                continue
            ax.annotate(f'{val:.3f}',
                       xy=(bar.get_x() + bar.get_width() / 2, val),
                       xytext=(0, 3), textcoords="offset points",
                       ha='center', va='bottom', fontsize=9, fontweight='bold')
            if ref_vals is not None and ref_vals[i] > 0:
                delta = (val - ref_vals[i]) * 100
                color = 'green' if delta > 0 else 'red'
                sign = '+' if delta > 0 else ''
                ax.annotate(f'{sign}{delta:.1f}%',
                           xy=(bar.get_x() + bar.get_width() / 2, val),
                           xytext=(0, 14), textcoords="offset points",
                           ha='center', va='bottom', fontsize=7, color=color)
    
    add_labels_with_delta(bars1, raw_vals)
    add_labels_with_delta(bars2, deep_vals, raw_vals)
    add_labels_with_delta(bars3, dr_vals, raw_vals)
    
    # Generate insight
    best_model = models[np.argmax(raw_vals)] if any(v > 0 for v in raw_vals) else 'N/A'
    best_raw = max(raw_vals) if any(v > 0 for v in raw_vals) else 0
    
    improvements = []
    for i, m in enumerate(models):
        if raw_vals[i] > 0 and dr_vals[i] > 0:
            improvements.append((m, (dr_vals[i] - raw_vals[i]) * 100))
    
    if improvements:
        best_improve = max(improvements, key=lambda x: x[1])
        insight_text = f"Best Model: {best_model} (C-Index: {best_raw:.3f})\nBest {method_name} Improvement: {best_improve[0]} ({best_improve[1]:+.1f}%)"
    else:
        insight_text = f"Best Model: {best_model} (C-Index: {best_raw:.3f})"
    
    ax.text(0.02, 0.02, insight_text, transform=ax.transAxes, fontsize=9,
            verticalalignment='bottom', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    output_path = plots_dir / f'plot1_{embedding_method}_impact.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot2_temporal_performance(data, plots_dir, dataset_config, embedding_method):
    """
    PLOT 2: Performance Across Time Horizons
    
    Shows C-Index at early (q25), median (q50), and late (q75) times.
    """
    method_info = EMBEDDING_INFO.get(embedding_method, {'display_name': embedding_method.upper()})
    method_name = method_info['display_name']
    
    print(f"\n--- Plot 2: Temporal Performance ({method_name}) ---")
    
    dataset_name = dataset_config['name']
    models = MODELS_EMBEDDING
    
    # Get Deep+Raw mode data (best mode typically)
    mode_data = data.get('Deep+Raw', {})
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    
    for idx, mode in enumerate(['Raw', 'Deep', 'Deep+Raw']):
        ax = axes[idx]
        mode_data = data.get(mode, {})
        
        x = np.arange(len(models))
        width = 0.25
        
        q25_vals = [mode_data.get(m, {}).get('c_index_q25', 0) for m in models]
        q50_vals = [mode_data.get(m, {}).get('c_index_q50', 0) for m in models]
        q75_vals = [mode_data.get(m, {}).get('c_index_q75', 0) for m in models]
        
        bars1 = ax.bar(x - width, q25_vals, width, label='Early (q25)', color=QUANTILE_COLORS['q25'])
        bars2 = ax.bar(x, q50_vals, width, label='Median (q50)', color=QUANTILE_COLORS['q50'])
        bars3 = ax.bar(x + width, q75_vals, width, label='Late (q75)', color=QUANTILE_COLORS['q75'])
        
        mode_label = mode if mode == 'Raw' else f'{method_name}' if mode == 'Deep' else f'{method_name}+Raw'
        ax.set_title(f'{mode_label}', fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(models, fontsize=10, rotation=45, ha='right')
        ax.grid(axis='y', linestyle='--', alpha=0.4)
        
        if idx == 0:
            ax.set_ylabel('C-Index', fontsize=11)
            ax.legend(loc='lower right', fontsize=9)
    
    fig.suptitle(f'Performance Across Time Horizons - {dataset_name}\n({method_name} Embeddings)', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    output_path = plots_dir / f'plot2_temporal_performance.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot3_discrimination_vs_calibration(data, plots_dir, dataset_config, embedding_method):
    """
    PLOT 3: Discrimination vs Calibration Trade-off
    
    Plots C-Index (discrimination) against IBS (calibration).
    Ideal: high C-Index AND low IBS (upper-left corner).
    """
    method_info = EMBEDDING_INFO.get(embedding_method, {'display_name': embedding_method.upper()})
    method_name = method_info['display_name']
    
    print(f"\n--- Plot 3: Discrimination vs Calibration ({method_name}) ---")
    
    dataset_name = dataset_config['name']
    
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_facecolor('#fafafa')
    
    markers = {'Raw': 'o', 'Deep': 's', 'Deep+Raw': '^'}
    
    for mode in ['Raw', 'Deep', 'Deep+Raw']:
        mode_data = data.get(mode, {})
        
        for model in MODELS_EMBEDDING:
            c_idx = mode_data.get(model, {}).get('c_index_q50', 0)
            ibs = mode_data.get(model, {}).get('ibs', 0)
            
            if c_idx > 0 and ibs > 0:
                color = MODE_COLORS[mode]
                marker = markers[mode]
                
                mode_label = mode if mode == 'Raw' else f'{method_name}' if mode == 'Deep' else f'{method_name}+Raw'
                ax.scatter(ibs, c_idx, c=color, s=150, marker=marker, 
                          alpha=0.8, edgecolors='white', linewidths=1.5,
                          label=f'{model} ({mode_label})')
                
                # Add model label
                ax.annotate(model, (ibs, c_idx), xytext=(5, 5), 
                           textcoords='offset points', fontsize=8)
    
    ax.set_xlabel('IBS (↓ better - lower is better calibration)', fontsize=12)
    ax.set_ylabel('C-Index (↑ better - higher is better discrimination)', fontsize=12)
    ax.set_title(f'Discrimination vs Calibration Trade-off - {dataset_name}\n({method_name} Embeddings)', 
                 fontsize=14, fontweight='bold')
    
    # Add ideal region indicator
    ax.axhline(y=0.7, color='gray', linestyle='--', alpha=0.5, label='Good C-Index threshold')
    ax.axvline(x=0.2, color='gray', linestyle=':', alpha=0.5, label='Good IBS threshold')
    
    ax.grid(True, alpha=0.3)
    
    # Custom legend (avoid duplicates)
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc='lower left', fontsize=8, ncol=2)
    
    plt.tight_layout()
    output_path = plots_dir / f'plot3_discrimination_vs_calibration.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def print_summary_table(data, dataset_config, embedding_method):
    """Print a summary table of all results."""
    method_info = EMBEDDING_INFO.get(embedding_method, {'display_name': embedding_method.upper()})
    method_name = method_info['display_name']
    
    print(f"\n{'='*80}")
    print(f"SUMMARY: {method_name} on {dataset_config['name']}")
    print(f"{'='*80}")
    print(f"{'Model':<12} {'Mode':<12} {'C-Index q50':<15} {'IBS':<10}")
    print("-" * 50)
    
    for mode in ['Raw', 'Deep', 'Deep+Raw']:
        mode_data = data.get(mode, {})
        for model in MODELS_EMBEDDING:
            c_idx = mode_data.get(model, {}).get('c_index_q50', 0)
            ibs = mode_data.get(model, {}).get('ibs', 0)
            if c_idx > 0:
                mode_label = mode if mode == 'Raw' else f'{method_name}' if mode == 'Deep' else f'{method_name}+Raw'
                print(f"{model:<12} {mode_label:<12} {c_idx:<15.4f} {ibs:<10.4f}")
    
    print(f"{'='*80}")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Visualize CV Results for Embedding Methods',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python visualize_cv_results.py --method tabicl --dataset metabric
    python visualize_cv_results.py --method tarte --dataset pbc
    python visualize_cv_results.py --method tabicl --dataset support
    
This replaces the separate tfm/TabICL/visualize_cv_results.py and 
tfm/TARTE/visualize_cv_results.py scripts.
        """
    )
    parser.add_argument(
        '--method', '-m',
        type=str,
        required=True,
        choices=['tabicl', 'tarte'],
        help='Embedding method to visualize'
    )
    parser.add_argument(
        '--dataset', '-d',
        type=str,
        choices=['metabric', 'pbc', 'support', 'gbsg', 'seer'],
        default='metabric',
        help='Dataset to visualize (default: metabric)'
    )
    return parser.parse_args()


def main():
    """Main entry point - generates all 3 insight plots."""
    args = parse_args()
    
    embedding_method = args.method.lower()
    dataset_key = args.dataset.lower()
    
    if dataset_key not in DATASET_CONFIG:
        print(f"Unknown dataset: {args.dataset}")
        print(f"Available datasets: {list(DATASET_CONFIG.keys())}")
        return
    
    dataset_config = DATASET_CONFIG[dataset_key]
    method_info = EMBEDDING_INFO.get(embedding_method, {'display_name': embedding_method.upper()})
    
    # Set paths
    base_results_dir = get_results_dir(embedding_method)
    cv_results_dir = base_results_dir / dataset_config['cv_dir']
    plots_dir = cv_results_dir / 'plots'
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print(f"{method_info['display_name']} CV Results Visualization - {dataset_config['name']}")
    print(f"Dataset: {dataset_config['description']}")
    print(f"Results dir: {cv_results_dir}")
    print("=" * 60)
    
    data = load_cv_results(cv_results_dir)
    if data is None:
        print(f"No CV results found for {dataset_config['name']}.")
        print(f"Run: python tfm/run_cv_analysis.py --method {embedding_method} --dataset {args.dataset}")
        return
    
    # Print summary table
    print_summary_table(data, dataset_config, embedding_method)
    
    # Generate the 3 key insight plots
    plot1_embedding_impact(data, plots_dir, dataset_config, embedding_method)
    plot2_temporal_performance(data, plots_dir, dataset_config, embedding_method)
    plot3_discrimination_vs_calibration(data, plots_dir, dataset_config, embedding_method)
    
    print(f"\n{'='*60}")
    print(f"All plots saved to: {plots_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
