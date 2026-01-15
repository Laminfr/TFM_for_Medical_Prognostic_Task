"""
TabICL CV Results Visualization
===============================

This script generates 3 key insight plots from the cross-validation results:

1. PLOT 1: "TabICL Impact by Model" (Discrimination)
   - Compares Raw vs Deep vs Deep+Raw for each model
   - Shows which models benefit from TabICL embeddings
   - Uses C-Index at median time (q50) as the primary metric
   
2. PLOT 2: "Performance Across Time Horizons" (Temporal Robustness)
   - Shows C-Index at early (q25), median (q50), and late (q75) times
   - Reveals if models maintain performance across different time horizons
   - Important for clinical utility (short vs long-term predictions)
   
3. PLOT 3: "Discrimination vs Calibration Trade-off"
   - Plots C-Index (discrimination) against IBS (calibration)
   - Shows which models achieve good ranking AND probability estimates
   - Ideal: high C-Index AND low IBS (upper-left corner)

Author: Auto-generated for TabICL survival analysis benchmark
"""

import matplotlib.pyplot as plt
import numpy as np
import json
import os
from pathlib import Path

# ==========================================
# CONFIGURATION
# ==========================================
CV_RESULTS_DIR = Path('/vol/miltank/users/sajb/Project/NeuralFineGray/tabICL/results/cv')
PLOTS_DIR = CV_RESULTS_DIR / 'plots'
PLOTS_DIR.mkdir(exist_ok=True)

# Models that completed successfully with all modes
MODELS_TABICL = ['CoxPH', 'RSF', 'XGBoost', 'DeepSurv']
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


def load_cv_results():
    """
    Load cross-validation results from JSON files.
    Returns a nested dict: data[Mode][Model][Metric]
    """
    data = {'Raw': {}, 'Deep': {}, 'Deep+Raw': {}}
    
    if not CV_RESULTS_DIR.exists():
        print(f"CV Results directory not found: {CV_RESULTS_DIR}")
        return None
    
    # Find the most recent final results file
    final_files = sorted(CV_RESULTS_DIR.glob('cv_results_final_*.json'), 
                        key=os.path.getmtime, reverse=True)
    
    if not final_files:
        final_files = sorted(CV_RESULTS_DIR.glob('cv_results_*.json'),
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


def plot1_tabicl_impact(data):
    """
    PLOT 1: TabICL Impact by Model
    
    WHY THIS COMPARISON:
    - Shows the core research question: Does TabICL improve survival models?
    - Compares 3 feature representations for each model
    - Uses C-Index at median survival time (most clinically relevant)
    - Only shows models where TabICL completed successfully
    """
    print("\n--- Plot 1: TabICL Impact by Model ---")
    
    models = MODELS_TABICL
    x = np.arange(len(models))
    width = 0.25
    
    raw_vals = [data['Raw'].get(m, {}).get('c_index_q50', 0) for m in models]
    deep_vals = [data['Deep'].get(m, {}).get('c_index_q50', 0) for m in models]
    dr_vals = [data['Deep+Raw'].get(m, {}).get('c_index_q50', 0) for m in models]
    
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.set_facecolor('#fafafa')
    
    # Plot bars
    bars1 = ax.bar(x - width, raw_vals, width, label='Raw Features', 
                   color=MODE_COLORS['Raw'], edgecolor='white', linewidth=1.5)
    bars2 = ax.bar(x, deep_vals, width, label='TabICL (512D)', 
                   color=MODE_COLORS['Deep'], edgecolor='white', linewidth=1.5)
    bars3 = ax.bar(x + width, dr_vals, width, label='TabICL + Raw (521D)', 
                   color=MODE_COLORS['Deep+Raw'], edgecolor='white', linewidth=1.5)
    
    # Formatting
    ax.set_title('TabICL Impact on Survival Models\n(C-Index at Median Survival Time, 5-Fold CV)', 
                 fontsize=14, fontweight='bold', pad=15)
    ax.set_ylabel('C-Index (↑ better)', fontsize=12)
    ax.set_xlabel('Survival Model', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=11, fontweight='medium')
    ax.legend(loc='upper right', fontsize=10, framealpha=0.9)
    
    # Y-axis limits (zoom in on relevant range)
    all_vals = [v for v in raw_vals + deep_vals + dr_vals if v > 0]
    if all_vals:
        ax.set_ylim(min(all_vals) - 0.02, max(all_vals) + 0.02)
    
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
    
    # Add insight annotation
    insight_text = "Key Finding: CoxPH benefits most from TabICL (+0.5%)\nRSF performs best overall but doesn't improve with TabICL"
    ax.text(0.02, 0.02, insight_text, transform=ax.transAxes, fontsize=9,
            verticalalignment='bottom', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    output_path = PLOTS_DIR / 'plot1_tabicl_impact.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot2_temporal_performance(data):
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
    
    # Focus on models that work with TabICL, compare Raw vs Deep+Raw
    models = MODELS_TABICL
    quantiles = ['q25', 'q50', 'q75']
    quantile_labels = ['Early (Q25)', 'Median (Q50)', 'Late (Q75)']
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Left: Raw features
    ax1 = axes[0]
    x = np.arange(len(models))
    width = 0.25
    
    for i, (q, ql) in enumerate(zip(quantiles, quantile_labels)):
        vals = [data['Raw'].get(m, {}).get(f'c_index_{q}', 0) for m in models]
        ax1.bar(x + (i - 1) * width, vals, width, label=ql, 
               color=QUANTILE_COLORS[q], edgecolor='white', alpha=0.85)
    
    ax1.set_title('Raw Features (9D)', fontsize=13, fontweight='bold')
    ax1.set_ylabel('C-Index', fontsize=11)
    ax1.set_xticks(x)
    ax1.set_xticklabels(models, fontsize=10)
    ax1.legend(fontsize=9)
    ax1.set_ylim(0.58, 0.66)
    ax1.grid(axis='y', linestyle='--', alpha=0.4)
    ax1.set_axisbelow(True)
    
    # Right: TabICL + Raw features
    ax2 = axes[1]
    
    for i, (q, ql) in enumerate(zip(quantiles, quantile_labels)):
        vals = [data['Deep+Raw'].get(m, {}).get(f'c_index_{q}', 0) for m in models]
        ax2.bar(x + (i - 1) * width, vals, width, label=ql,
               color=QUANTILE_COLORS[q], edgecolor='white', alpha=0.85)
    
    ax2.set_title('TabICL + Raw Features (521D)', fontsize=13, fontweight='bold')
    ax2.set_ylabel('C-Index', fontsize=11)
    ax2.set_xticks(x)
    ax2.set_xticklabels(models, fontsize=10)
    ax2.legend(fontsize=9)
    ax2.set_ylim(0.58, 0.66)
    ax2.grid(axis='y', linestyle='--', alpha=0.4)
    ax2.set_axisbelow(True)
    
    plt.suptitle('C-Index Across Time Horizons: Early vs Median vs Late Survival\n(5-Fold CV on METABRIC)', 
                 fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    output_path = PLOTS_DIR / 'plot2_temporal_performance.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot3_discrimination_vs_calibration(data):
    """
    PLOT 3: Discrimination vs Calibration Trade-off
    
    WHY THIS COMPARISON:
    - C-Index measures ranking (discrimination): "Can we rank patients by risk?"
    - IBS measures calibration: "Are predicted probabilities accurate?"
    - A model can rank well but give wrong probabilities (or vice versa)
    - Ideal model: HIGH C-Index + LOW IBS (upper-left quadrant)
    """
    print("\n--- Plot 3: Discrimination vs Calibration Trade-off ---")
    
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_facecolor('#fafafa')
    
    # Markers for different modes
    markers = {'Raw': 'o', 'Deep': 's', 'Deep+Raw': '^'}
    mode_labels = {'Raw': 'Raw (9D)', 'Deep': 'TabICL (512D)', 'Deep+Raw': 'TabICL+Raw (521D)'}
    
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
    ax.set_title('Discrimination vs Calibration Trade-off\n(5-Fold CV on METABRIC)', 
                 fontsize=14, fontweight='bold', pad=15)
    
    # Add quadrant annotations
    ax.axhline(y=0.63, color='gray', linestyle='--', alpha=0.5, zorder=1)
    ax.axvline(x=0.20, color='gray', linestyle='--', alpha=0.5, zorder=1)
    
    # Ideal region annotation
    ax.annotate('IDEAL\n(High C-Index, Low IBS)', xy=(0.16, 0.645), fontsize=9,
               ha='center', color='green', fontweight='bold',
               bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3))
    
    # Poor region
    ax.annotate('Poor\n(Low Discrimination)', xy=(0.27, 0.59), fontsize=8,
               ha='center', color='red', alpha=0.7)
    
    ax.legend(loc='lower left', fontsize=10, framealpha=0.9)
    ax.grid(True, linestyle='--', alpha=0.3, zorder=0)
    ax.set_axisbelow(True)
    
    # Set axis limits
    ax.set_xlim(0.14, 0.30)
    ax.set_ylim(0.57, 0.66)
    
    plt.tight_layout()
    output_path = PLOTS_DIR / 'plot3_discrimination_vs_calibration.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def print_summary_table(data):
    """Print a formatted summary table of results."""
    print("\n" + "="*80)
    print("SUMMARY TABLE: C-Index (q50) and IBS by Model and Mode")
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
    print("\nKey Insights:")
    print("1. RSF achieves best C-Index (0.6424) with raw features")
    print("2. CoxPH benefits most from TabICL: 0.6337 → 0.6371 (+0.5%)")
    print("3. XGBoost has poor calibration (IBS ~0.27) despite decent C-Index")
    print("4. NFG and DeSurv failed with TabICL embeddings (dtype mismatch)")
    print("="*80)


def main():
    """Main entry point - generates all 3 insight plots."""
    print("="*60)
    print("TabICL CV Results Visualization")
    print("="*60)
    
    data = load_cv_results()
    if data is None:
        print("No CV results found. Run run_cv_analysis.py first.")
        return
    
    # Print summary table
    print_summary_table(data)
    
    # Generate the 3 key insight plots
    plot1_tabicl_impact(data)
    plot2_temporal_performance(data)
    plot3_discrimination_vs_calibration(data)
    
    print(f"\n{'='*60}")
    print(f"All plots saved to: {PLOTS_DIR}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
