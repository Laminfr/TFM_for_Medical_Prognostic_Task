#!/usr/bin/env python
"""
TARTE CV Results Visualization (Modular Version)
=================================================

This script generates 3 key insight plots from the cross-validation results:

1. PLOT 1: "TARTE Impact by Model" (Discrimination)
   - Compares Raw vs Deep vs Deep+Raw for each model
   - Shows which models benefit from TARTE embeddings
   - Uses C-Index at median time (q50) as the primary metric
   
2. PLOT 2: "Performance Across Time Horizons" (Temporal Robustness)
   - Shows C-Index at early (q25), median (q50), and late (q75) times
   - Reveals if models maintain performance across different time horizons
   - Important for clinical utility (short vs long-term predictions)
   
3. PLOT 3: "Discrimination vs Calibration Trade-off"
   - Plots C-Index (discrimination) against IBS (calibration)
   - Shows which models achieve good ranking AND probability estimates
   - Ideal: high C-Index AND low IBS (upper-left corner)

"""

import argparse
import matplotlib.pyplot as plt
import numpy as np
import json
import os
from pathlib import Path
dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ==========================================
# CONFIGURATION
# ==========================================
BASE_RESULTS_DIR = Path(os.path.join(dir, "tfm", "TARTE", "results"))

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
MODELS_TARTE = ['CoxPH', 'RSF', 'XGBoost', 'DeepSurv']
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
        model = r.get('model')  # Keep original short name
        
        mode_key = {'raw': 'Raw', 'deep': 'Deep', 'deep+raw': 'Deep+Raw'}.get(mode, mode)
        
        if mode_key in data:
            data[mode_key][model] = {k: v for k, v in r.items() 
                                      if k not in ['model', 'mode', 'dataset', 'timestamp', 'status']}
    
    return data


def plot1_tarte_impact(data, plots_dir, dataset_config):
    """
    PLOT 1: TARTE Impact by Model
    
    WHY THIS COMPARISON:
    - Shows the core research question: Does TARTE improve survival models?
    - Compares 3 feature representations for each model
    - Uses C-Index at median survival time (most clinically relevant)
    - Only shows models where TARTE completed successfully
    """
    print("\n--- Plot 1: TARTE Impact by Model ---")
    
    dataset_name = dataset_config['name']
    n_features = dataset_config['n_features']
    
    models = MODELS_TARTE
    x = np.arange(len(models))
    width = 0.25
    
    raw_vals = [data['Raw'].get(m, {}).get('c_index_q50', 0) for m in models]
    deep_vals = [data['Deep'].get(m, {}).get('c_index_q50', 0) for m in models]
    dr_vals = [data['Deep+Raw'].get(m, {}).get('c_index_q50', 0) for m in models]
    
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.set_facecolor('#fafafa')
    
    # Plot bars
    bars1 = ax.bar(x - width, raw_vals, width, label=f'Raw Features',
                   color=MODE_COLORS['Raw'], edgecolor='white', linewidth=1.5)
    bars2 = ax.bar(x, deep_vals, width, label='TARTE',
                   color=MODE_COLORS['Deep'], edgecolor='white', linewidth=1.5)
    bars3 = ax.bar(x + width, dr_vals, width, label=f'TARTE + Raw',
                   color=MODE_COLORS['Deep+Raw'], edgecolor='white', linewidth=1.5)
    
    # Formatting
    ax.set_title(f'TARTE Impact on Survival Models - {dataset_name}\n(C-Index at Median Survival Time, 5-Fold CV)',
                 fontsize=14, fontweight='bold', pad=15)
    ax.set_ylabel('C-Index (↑ better)', fontsize=12)
    ax.set_xlabel('Survival Model', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=11, fontweight='medium')
    ax.legend(loc='upper right', fontsize=10, framealpha=0.9)
    
    # Y-axis limits (zoom in on relevant range - dynamic based on data)
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
            # Value label
            ax.annotate(f'{val:.3f}',
                       xy=(bar.get_x() + bar.get_width() / 2, val),
                       xytext=(0, 3), textcoords="offset points",
                       ha='center', va='bottom', fontsize=9, fontweight='bold')
            # Delta indicator (compared to raw)
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
    
    # Generate insight based on data
    best_model = models[np.argmax(raw_vals)] if any(v > 0 for v in raw_vals) else 'N/A'
    best_raw = max(raw_vals) if any(v > 0 for v in raw_vals) else 0
    
    # Find model with best improvement from TARTE
    improvements = []
    for i, m in enumerate(models):
        if raw_vals[i] > 0 and dr_vals[i] > 0:
            improvements.append((m, (dr_vals[i] - raw_vals[i]) * 100))
    
    if improvements:
        best_improve = max(improvements, key=lambda x: x[1])
        insight_text = f"Best Model: {best_model} (C-Index: {best_raw:.3f})\nBest TARTE Improvement: {best_improve[0]} ({best_improve[1]:+.1f}%)"
    else:
        insight_text = f"Best Model: {best_model} (C-Index: {best_raw:.3f})"
    
    ax.text(0.02, 0.02, insight_text, transform=ax.transAxes, fontsize=9,
            verticalalignment='bottom', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    output_path = plots_dir / 'plot1_tarte_impact.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot2_temporal_performance(data, plots_dir, dataset_config):
    """
    PLOT 2: Performance Across Time Horizons
    
    WHY THIS COMPARISON:
    - Clinical relevance: Early predictions (1yr) vs Late predictions (5yr)
    - Shows temporal robustness of predictions
    - q25 = early survival time (short-term risk)
    - q50 = median survival time
    - q75 = late survival time (long-term risk)
    """
    print("\n--- Plot 2: Performance Across Time Horizons ---")
    
    dataset_name = dataset_config['name']
    n_features = dataset_config['n_features']
    
    # Focus on models that work with TARTE, compare Raw vs Deep+Raw
    models = MODELS_TARTE
    quantiles = ['q25', 'q50', 'q75']
    quantile_labels = ['Early (Q25)', 'Median (Q50)', 'Late (Q75)']
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Collect all values to set consistent y-axis limits
    all_raw_vals = []
    all_dr_vals = []
    
    for q in quantiles:
        all_raw_vals.extend([data['Raw'].get(m, {}).get(f'c_index_{q}', 0) for m in models])
        all_dr_vals.extend([data['Deep+Raw'].get(m, {}).get(f'c_index_{q}', 0) for m in models])
    
    all_vals = [v for v in all_raw_vals + all_dr_vals if v > 0]
    if all_vals:
        y_min = min(all_vals) - 0.02
        y_max = max(all_vals) + 0.02
    else:
        y_min, y_max = 0.5, 0.7
    
    # Left: Raw features
    ax1 = axes[0]
    x = np.arange(len(models))
    width = 0.25
    
    for i, (q, ql) in enumerate(zip(quantiles, quantile_labels)):
        vals = [data['Raw'].get(m, {}).get(f'c_index_{q}', 0) for m in models]
        ax1.bar(x + (i - 1) * width, vals, width, label=ql, 
               color=QUANTILE_COLORS[q], edgecolor='white', alpha=0.85)
    
    ax1.set_title(f'Raw Features', fontsize=13, fontweight='bold')
    ax1.set_ylabel('C-Index', fontsize=11)
    ax1.set_xticks(x)
    ax1.set_xticklabels(models, fontsize=10)
    ax1.legend(fontsize=9)
    ax1.set_ylim(y_min, y_max)
    ax1.grid(axis='y', linestyle='--', alpha=0.4)
    ax1.set_axisbelow(True)
    
    # Right: TARTE + Raw features
    ax2 = axes[1]
    
    for i, (q, ql) in enumerate(zip(quantiles, quantile_labels)):
        vals = [data['Deep+Raw'].get(m, {}).get(f'c_index_{q}', 0) for m in models]
        ax2.bar(x + (i - 1) * width, vals, width, label=ql,
               color=QUANTILE_COLORS[q], edgecolor='white', alpha=0.85)
    
    ax2.set_title(f'TARTE + Raw Features', fontsize=13, fontweight='bold')
    ax2.set_ylabel('C-Index', fontsize=11)
    ax2.set_xticks(x)
    ax2.set_xticklabels(models, fontsize=10)
    ax2.legend(fontsize=9)
    ax2.set_ylim(y_min, y_max)
    ax2.grid(axis='y', linestyle='--', alpha=0.4)
    ax2.set_axisbelow(True)
    
    plt.suptitle(f'C-Index Across Time Horizons: Early vs Median vs Late Survival\n(5-Fold CV on {dataset_name})', 
                 fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    output_path = plots_dir / 'plot2_temporal_performance.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot3_discrimination_vs_calibration(data, plots_dir, dataset_config):
    """
    PLOT 3: Discrimination vs Calibration Trade-off
    
    WHY THIS COMPARISON:
    - C-Index measures ranking (discrimination): "Can we rank patients by risk?"
    - IBS measures calibration: "Are predicted probabilities accurate?"
    - A model can rank well but give wrong probabilities (or vice versa)
    - Ideal model: HIGH C-Index + LOW IBS (upper-left quadrant)
    """
    print("\n--- Plot 3: Discrimination vs Calibration Trade-off ---")
    
    dataset_name = dataset_config['name']
    
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_facecolor('#fafafa')
    
    # Markers for different modes
    markers = {'Raw': 'o', 'Deep': 's', 'Deep+Raw': '^'}
    n_features = dataset_config['n_features']
    mode_labels = {'Raw': f'Raw', 'Deep': f'TARTE', 'Deep+Raw': f'TARTE+Raw'}
    
    # Collect all values for setting axis limits
    all_c_indices = []
    all_ibs_vals = []
    
    # Plot all models
    all_models = MODELS_ALL
    
    for mode in ['Raw', 'Deep', 'Deep+Raw']:
        c_indices = []
        ibs_vals = []
        labels = []
        
        for model in all_models:
            c_idx = data[mode].get(model, {}).get('c_index_q50', 0)
            ibs = data[mode].get(model, {}).get('ibs', 0)
            
            if c_idx > 0 and ibs > 0:  # Only plot if we have both metrics
                c_indices.append(c_idx)
                ibs_vals.append(ibs)
                labels.append(model)
                all_c_indices.append(c_idx)
                all_ibs_vals.append(ibs)
        
        if c_indices:
            scatter = ax.scatter(ibs_vals, c_indices, 
                               c=MODE_COLORS[mode], 
                               marker=markers[mode],
                               s=150, 
                               label=mode_labels[mode],
                               edgecolors='white',
                               linewidths=1.5,
                               alpha=0.85,
                               zorder=3)
            
            # Add model labels
            for i, label in enumerate(labels):
                ax.annotate(label, (ibs_vals[i], c_indices[i]),
                           xytext=(5, 5), textcoords='offset points',
                           fontsize=8, alpha=0.8)
    
    # Formatting
    ax.set_xlabel('Integrated Brier Score (↓ better calibration)', fontsize=12)
    ax.set_ylabel('C-Index at Median Time (↑ better discrimination)', fontsize=12)
    ax.set_title(f'Discrimination vs Calibration Trade-off\n(5-Fold CV on {dataset_name})', 
                 fontsize=14, fontweight='bold', pad=15)
    
    # Set dynamic axis limits
    if all_c_indices and all_ibs_vals:
        c_min, c_max = min(all_c_indices), max(all_c_indices)
        ibs_min, ibs_max = min(all_ibs_vals), max(all_ibs_vals)
        
        c_range = c_max - c_min
        ibs_range = ibs_max - ibs_min
        
        ax.set_xlim(ibs_min - 0.1 * ibs_range, ibs_max + 0.1 * ibs_range)
        ax.set_ylim(c_min - 0.1 * c_range, c_max + 0.1 * c_range)
        
        # Add quadrant lines at median values
        c_mid = (c_min + c_max) / 2
        ibs_mid = (ibs_min + ibs_max) / 2
        ax.axhline(y=c_mid, color='gray', linestyle='--', alpha=0.5, zorder=1)
        ax.axvline(x=ibs_mid, color='gray', linestyle='--', alpha=0.5, zorder=1)
        
        # Ideal region annotation (upper-left)
        ax.annotate('IDEAL\n(High C-Index, Low IBS)', 
                   xy=(ibs_min + 0.1 * ibs_range, c_max - 0.05 * c_range), 
                   fontsize=9, ha='center', color='green', fontweight='bold',
                   bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3))
    
    ax.legend(loc='lower left', fontsize=10, framealpha=0.9)
    ax.grid(True, linestyle='--', alpha=0.3, zorder=0)
    ax.set_axisbelow(True)
    
    plt.tight_layout()
    output_path = plots_dir / 'plot3_discrimination_vs_calibration.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def print_summary_table(data, dataset_config):
    """Print a formatted summary table of results."""
    dataset_name = dataset_config['name']
    
    print("\n" + "="*80)
    print(f"SUMMARY TABLE: {dataset_name} - C-Index (q50) and IBS by Model and Mode")
    print("="*80)
    
    header = f"{'Model':<12} | {'Raw C-Idx':<10} {'Raw IBS':<10} | {'Deep C-Idx':<10} {'Deep IBS':<10} | {'D+R C-Idx':<10} {'D+R IBS':<10}"
    print(header)
    print("-"*80)
    
    for model in MODELS_ALL:
        raw_c = data['Raw'].get(model, {}).get('c_index_q50', 0)
        raw_i = data['Raw'].get(model, {}).get('ibs', 0)
        deep_c = data['Deep'].get(model, {}).get('c_index_q50', 0)
        deep_i = data['Deep'].get(model, {}).get('ibs', 0)
        dr_c = data['Deep+Raw'].get(model, {}).get('c_index_q50', 0)
        dr_i = data['Deep+Raw'].get(model, {}).get('ibs', 0)
        
        # Format with N/A for missing values
        raw_c_str = f"{raw_c:.4f}" if raw_c > 0 else "N/A"
        raw_i_str = f"{raw_i:.4f}" if raw_i > 0 else "N/A"
        deep_c_str = f"{deep_c:.4f}" if deep_c > 0 else "N/A"
        deep_i_str = f"{deep_i:.4f}" if deep_i > 0 else "N/A"
        dr_c_str = f"{dr_c:.4f}" if dr_c > 0 else "N/A"
        dr_i_str = f"{dr_i:.4f}" if dr_i > 0 else "N/A"
        
        print(f"{model:<12} | {raw_c_str:<10} {raw_i_str:<10} | {deep_c_str:<10} {deep_i_str:<10} | {dr_c_str:<10} {dr_i_str:<10}")
    
    print("="*80)
    
    # Generate dynamic insights based on data
    print("\nKey Insights:")
    
    # Find best model for raw features
    best_raw_model = None
    best_raw_cidx = 0
    for model in MODELS_ALL:
        c_idx = data['Raw'].get(model, {}).get('c_index_q50', 0)
        if c_idx > best_raw_cidx:
            best_raw_cidx = c_idx
            best_raw_model = model
    
    if best_raw_model:
        print(f"1. {best_raw_model} achieves best C-Index ({best_raw_cidx:.4f}) with raw features")
    
    # Find model with best TARTE improvement
    best_improvement = 0
    best_improve_model = None
    for model in MODELS_TARTE:
        raw_c = data['Raw'].get(model, {}).get('c_index_q50', 0)
        dr_c = data['Deep+Raw'].get(model, {}).get('c_index_q50', 0)
        if raw_c > 0 and dr_c > 0:
            improvement = (dr_c - raw_c) / raw_c * 100
            if improvement > best_improvement:
                best_improvement = improvement
                best_improve_model = model
    
    if best_improve_model:
        raw_c = data['Raw'].get(best_improve_model, {}).get('c_index_q50', 0)
        dr_c = data['Deep+Raw'].get(best_improve_model, {}).get('c_index_q50', 0)
        print(f"2. {best_improve_model} benefits most from TARTE: {raw_c:.4f} → {dr_c:.4f} (+{best_improvement:.1f}%)")
    
    # Check for models with poor calibration
    poor_calib_models = []
    for model in MODELS_ALL:
        ibs = data['Raw'].get(model, {}).get('ibs', 0)
        if ibs > 0.25:
            poor_calib_models.append((model, ibs))
    
    if poor_calib_models:
        model, ibs = poor_calib_models[0]
        print(f"3. {model} has high IBS (~{ibs:.2f}) indicating calibration issues")
    
    print("="*80)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Visualize TARTE CV Results',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python visualize_cv_results.py                     # METABRIC (default)
    python visualize_cv_results.py --dataset pbc       # PBC dataset
    python visualize_cv_results.py --dataset support   # SUPPORT dataset
        """
    )
    parser.add_argument(
        '--dataset', '-d',
        type=str,
        choices=['metabric', 'pbc', 'support', 'gbsg'],
        default='metabric',
        help='Dataset to visualize (default: metabric)'
    )
    return parser.parse_args()


def main(dataset='metabric'):
    """Main entry point - generates all 3 insight plots."""
    
    # Get dataset configuration
    dataset_key = dataset
    if dataset_key not in DATASET_CONFIG:
        print(f"Unknown dataset: {dataset}")
        print(f"Available datasets: {list(DATASET_CONFIG.keys())}")
        return
    
    dataset_config = DATASET_CONFIG[dataset_key]
    
    # Set paths
    cv_results_dir = BASE_RESULTS_DIR / dataset_config['cv_dir']
    plots_dir = cv_results_dir / 'plots'
    plots_dir.mkdir(exist_ok=True)
    
    print("="*60)
    print(f"TARTE CV Results Visualization - {dataset_config['name']}")
    print(f"Dataset: {dataset_config['description']}")
    print(f"Results dir: {cv_results_dir}")
    print("="*60)
    
    data = load_cv_results(cv_results_dir)
    if data is None:
        print(f"No CV results found for {dataset_config['name']}.")
        return
    
    # Print summary table
    print_summary_table(data, dataset_config)
    
    # Generate the 3 key insight plots
    plot1_tarte_impact(data, plots_dir, dataset_config)
    plot2_temporal_performance(data, plots_dir, dataset_config)
    plot3_discrimination_vs_calibration(data, plots_dir, dataset_config)
    
    print(f"\n{'='*60}")
    print(f"All plots saved to: {plots_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main(dataset='support')
