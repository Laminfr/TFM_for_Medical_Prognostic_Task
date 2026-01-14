#!/usr/bin/env python
"""
Competing Risks Benchmark Visualization
========================================

Generates comparison plots for the 3-phase competing risks benchmark:
- Phase 1: Multi-Class Survival Stacking (XGBoost)
- Phase 2: Pure Neural Fine-Gray
- Phase 3: Hybrid (NFG Embeddings + XGBoost)

Creates column charts comparing:
1. Cause-Specific C-Index across methods and datasets
2. CIF-based IBS across methods and datasets

Design follows the same style as Survival Stacking visualizations.
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
RESULTS_DIR = PROJECT_ROOT / 'results' / 'competing_risks'
PLOTS_DIR = RESULTS_DIR / 'plots'

# Dataset configurations
DATASET_CONFIG = {
    'synthetic_competing': {
        'name': 'SYNTHETIC',
        'description': 'Synthetic Competing Risks',
        'color': '#3498db'
    },
    'seer': {
        'name': 'SEER',
        'description': 'SEER Cancer Data',
        'color': '#e74c3c'
    }
}

# Phase/Method colors
PHASE_COLORS = {
    'stacking': '#4a90e2',      # Blue - Multi-Class XGBoost
    'nfg': '#d9534f',           # Red - Neural Fine-Gray
    'hybrid': '#5cb85c',        # Green - Hybrid approach
}

PHASE_LABELS = {
    'stacking': 'Multi-Class XGBoost',
    'nfg': 'Neural Fine-Gray',
    'hybrid': 'Hybrid (NFG + XGB)',
}

# Risk colors for per-risk plots
RISK_COLORS = {
    1: '#e74c3c',  # Red
    2: '#3498db',  # Blue
    3: '#2ecc71',  # Green
}


def load_benchmark_results(results_dir: Path = RESULTS_DIR) -> Dict:
    """Load all benchmark results."""
    results = {}
    
    # Try to load combined results first
    combined_file = results_dir / 'full_benchmark_5fold.json'
    if combined_file.exists():
        with open(combined_file, 'r') as f:
            return json.load(f)
    
    # Otherwise load individual dataset files
    for dataset in DATASET_CONFIG.keys():
        patterns = [
            f'{dataset}_benchmark_5fold.json',
            f'{dataset.upper()}_benchmark_5fold.json',
        ]
        for pattern in patterns:
            filepath = results_dir / pattern
            if filepath.exists():
                with open(filepath, 'r') as f:
                    results[dataset.upper()] = json.load(f)
                break
    
    return results


def plot_cindex_comparison(
    results: Dict,
    output_path: Path,
    per_risk: bool = False
):
    """
    Plot 1: Cause-Specific C-Index Comparison
    
    Column chart comparing C-Index across phases and datasets.
    """
    print("\n--- Plot 1: C-Index Comparison ---")
    
    datasets = [k for k in results.keys() if k.lower() in DATASET_CONFIG]
    phases = ['stacking', 'nfg', 'hybrid']
    
    if not datasets:
        print("No valid datasets found in results")
        return
    
    if per_risk:
        # Get number of risks from first dataset
        n_risks = results[datasets[0]].get('n_risks', 2)
        fig, axes = plt.subplots(1, n_risks, figsize=(6 * n_risks, 6))
        if n_risks == 1:
            axes = [axes]
        
        for risk_idx, ax in enumerate(axes, 1):
            ax.set_facecolor('#fafafa')
            
            x = np.arange(len(datasets))
            width = 0.25
            
            for i, phase in enumerate(phases):
                values = []
                errors = []
                for dataset in datasets:
                    phase_data = results[dataset].get('phases', {}).get(phase, {})
                    mean = phase_data.get('mean_metrics', {}).get(f'c_index_risk{risk_idx}', 0)
                    std = phase_data.get('std_metrics', {}).get(f'c_index_risk{risk_idx}', 0)
                    values.append(mean)
                    errors.append(std)
                
                offset = (i - 1) * width
                bars = ax.bar(x + offset, values, width, yerr=errors, capsize=3,
                             label=PHASE_LABELS[phase], color=PHASE_COLORS[phase],
                             edgecolor='white', linewidth=1.5, alpha=0.9)
                
                # Add value labels
                for bar, val in zip(bars, values):
                    if val > 0:
                        ax.annotate(f'{val:.3f}',
                                   xy=(bar.get_x() + bar.get_width() / 2, val),
                                   xytext=(0, 3), textcoords="offset points",
                                   ha='center', va='bottom', fontsize=8, fontweight='bold')
            
            ax.set_title(f'Risk {risk_idx}', fontsize=12, fontweight='bold')
            ax.set_ylabel('C-Index (↑ better)', fontsize=11)
            ax.set_xlabel('Dataset', fontsize=11)
            ax.set_xticks(x)
            ax.set_xticklabels([DATASET_CONFIG.get(d.lower(), {}).get('name', d) 
                               for d in datasets], fontsize=10)
            ax.legend(fontsize=9, loc='lower right')
            ax.grid(axis='y', linestyle='--', alpha=0.4)
            ax.set_axisbelow(True)
            
            # Dynamic y-limits
            all_vals = [results[d].get('phases', {}).get(p, {}).get('mean_metrics', {}).get(f'c_index_risk{risk_idx}', 0)
                       for d in datasets for p in phases]
            valid_vals = [v for v in all_vals if v > 0]
            if valid_vals:
                ax.set_ylim(min(valid_vals) - 0.05, min(1.0, max(valid_vals) + 0.05))
        
        plt.suptitle('Cause-Specific C-Index by Risk Type\n(5-Fold CV, Higher is Better)',
                    fontsize=14, fontweight='bold')
        
    else:
        # Aggregated plot
        fig, ax = plt.subplots(figsize=(10, 7))
        ax.set_facecolor('#fafafa')
        
        x = np.arange(len(datasets))
        width = 0.25
        
        for i, phase in enumerate(phases):
            values = []
            errors = []
            for dataset in datasets:
                phase_data = results[dataset].get('phases', {}).get(phase, {})
                mean = phase_data.get('mean_metrics', {}).get('c_index_mean', 0)
                std = phase_data.get('std_metrics', {}).get('c_index_mean', 0)
                values.append(mean)
                errors.append(std)
            
            offset = (i - 1) * width
            bars = ax.bar(x + offset, values, width, yerr=errors, capsize=4,
                         label=PHASE_LABELS[phase], color=PHASE_COLORS[phase],
                         edgecolor='white', linewidth=2, alpha=0.9)
            
            # Add value labels
            for j, (bar, val) in enumerate(zip(bars, values)):
                if val > 0:
                    ax.annotate(f'{val:.3f}',
                               xy=(bar.get_x() + bar.get_width() / 2, val),
                               xytext=(0, 4), textcoords="offset points",
                               ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        ax.set_title('Cause-Specific C-Index: Competing Risks Benchmark\n(Mean Across Risks, 5-Fold CV)',
                    fontsize=14, fontweight='bold', pad=15)
        ax.set_ylabel('C-Index (↑ better)', fontsize=12)
        ax.set_xlabel('Dataset', fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels([DATASET_CONFIG.get(d.lower(), {}).get('name', d) 
                           for d in datasets], fontsize=11, fontweight='medium')
        ax.legend(fontsize=10, loc='lower right', framealpha=0.9)
        ax.grid(axis='y', linestyle='--', alpha=0.4)
        ax.set_axisbelow(True)
        
        # Dynamic y-limits
        all_vals = [results[d].get('phases', {}).get(p, {}).get('mean_metrics', {}).get('c_index_mean', 0)
                   for d in datasets for p in phases]
        valid_vals = [v for v in all_vals if v > 0]
        if valid_vals:
            ax.set_ylim(min(valid_vals) - 0.05, min(1.0, max(valid_vals) + 0.05))
        
        # Add insight box
        insights = []
        for dataset in datasets:
            dataset_name = DATASET_CONFIG.get(dataset.lower(), {}).get('name', dataset)
            values = {p: results[dataset].get('phases', {}).get(p, {}).get('mean_metrics', {}).get('c_index_mean', 0)
                     for p in phases}
            best_phase = max(values.items(), key=lambda x: x[1])
            if best_phase[1] > 0:
                insights.append(f"{dataset_name}: Best = {PHASE_LABELS[best_phase[0]]} ({best_phase[1]:.3f})")
        
        if insights:
            insight_text = "Best Methods:\n" + "\n".join(insights)
            ax.text(0.02, 0.98, insight_text, transform=ax.transAxes, fontsize=9,
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', dpi=150)
    print(f"Saved: {output_path}")
    plt.close()


def plot_ibs_comparison(
    results: Dict,
    output_path: Path,
    per_risk: bool = False
):
    """
    Plot 2: CIF-based IBS Comparison
    
    Column chart comparing Integrated Brier Score across phases and datasets.
    """
    print("\n--- Plot 2: IBS Comparison ---")
    
    datasets = [k for k in results.keys() if k.lower() in DATASET_CONFIG]
    phases = ['stacking', 'nfg', 'hybrid']
    
    if not datasets:
        print("No valid datasets found in results")
        return
    
    if per_risk:
        n_risks = results[datasets[0]].get('n_risks', 2)
        fig, axes = plt.subplots(1, n_risks, figsize=(6 * n_risks, 6))
        if n_risks == 1:
            axes = [axes]
        
        for risk_idx, ax in enumerate(axes, 1):
            ax.set_facecolor('#fafafa')
            
            x = np.arange(len(datasets))
            width = 0.25
            
            for i, phase in enumerate(phases):
                values = []
                errors = []
                for dataset in datasets:
                    phase_data = results[dataset].get('phases', {}).get(phase, {})
                    mean = phase_data.get('mean_metrics', {}).get(f'ibs_risk{risk_idx}', 0)
                    std = phase_data.get('std_metrics', {}).get(f'ibs_risk{risk_idx}', 0)
                    values.append(mean)
                    errors.append(std)
                
                offset = (i - 1) * width
                bars = ax.bar(x + offset, values, width, yerr=errors, capsize=3,
                             label=PHASE_LABELS[phase], color=PHASE_COLORS[phase],
                             edgecolor='white', linewidth=1.5, alpha=0.9)
                
                # Add value labels
                for bar, val in zip(bars, values):
                    if val > 0:
                        ax.annotate(f'{val:.3f}',
                                   xy=(bar.get_x() + bar.get_width() / 2, val),
                                   xytext=(0, 3), textcoords="offset points",
                                   ha='center', va='bottom', fontsize=8, fontweight='bold')
            
            ax.set_title(f'Risk {risk_idx}', fontsize=12, fontweight='bold')
            ax.set_ylabel('IBS (↓ better)', fontsize=11)
            ax.set_xlabel('Dataset', fontsize=11)
            ax.set_xticks(x)
            ax.set_xticklabels([DATASET_CONFIG.get(d.lower(), {}).get('name', d) 
                               for d in datasets], fontsize=10)
            ax.legend(fontsize=9, loc='upper right')
            ax.grid(axis='y', linestyle='--', alpha=0.4)
            ax.set_axisbelow(True)
        
        plt.suptitle('CIF-based Integrated Brier Score by Risk Type\n(5-Fold CV, Lower is Better)',
                    fontsize=14, fontweight='bold')
        
    else:
        fig, ax = plt.subplots(figsize=(10, 7))
        ax.set_facecolor('#fafafa')
        
        x = np.arange(len(datasets))
        width = 0.25
        
        for i, phase in enumerate(phases):
            values = []
            errors = []
            for dataset in datasets:
                phase_data = results[dataset].get('phases', {}).get(phase, {})
                mean = phase_data.get('mean_metrics', {}).get('ibs_mean', 0)
                std = phase_data.get('std_metrics', {}).get('ibs_mean', 0)
                values.append(mean)
                errors.append(std)
            
            offset = (i - 1) * width
            bars = ax.bar(x + offset, values, width, yerr=errors, capsize=4,
                         label=PHASE_LABELS[phase], color=PHASE_COLORS[phase],
                         edgecolor='white', linewidth=2, alpha=0.9)
            
            # Add value labels
            for j, (bar, val) in enumerate(zip(bars, values)):
                if val > 0:
                    ax.annotate(f'{val:.3f}',
                               xy=(bar.get_x() + bar.get_width() / 2, val),
                               xytext=(0, 4), textcoords="offset points",
                               ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        ax.set_title('CIF-based IBS: Competing Risks Benchmark\n(Mean Across Risks, 5-Fold CV, Lower is Better)',
                    fontsize=14, fontweight='bold', pad=15)
        ax.set_ylabel('Integrated Brier Score (↓ better)', fontsize=12)
        ax.set_xlabel('Dataset', fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels([DATASET_CONFIG.get(d.lower(), {}).get('name', d) 
                           for d in datasets], fontsize=11, fontweight='medium')
        ax.legend(fontsize=10, loc='upper right', framealpha=0.9)
        ax.grid(axis='y', linestyle='--', alpha=0.4)
        ax.set_axisbelow(True)
        
        # Add insight box
        insights = []
        for dataset in datasets:
            dataset_name = DATASET_CONFIG.get(dataset.lower(), {}).get('name', dataset)
            values = {p: results[dataset].get('phases', {}).get(p, {}).get('mean_metrics', {}).get('ibs_mean', float('inf'))
                     for p in phases}
            # Filter out zeros
            values = {k: v for k, v in values.items() if v > 0}
            if values:
                best_phase = min(values.items(), key=lambda x: x[1])
                insights.append(f"{dataset_name}: Best = {PHASE_LABELS[best_phase[0]]} ({best_phase[1]:.3f})")
        
        if insights:
            insight_text = "Best Methods (lowest IBS):\n" + "\n".join(insights)
            ax.text(0.02, 0.02, insight_text, transform=ax.transAxes, fontsize=9,
                    verticalalignment='bottom', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', dpi=150)
    print(f"Saved: {output_path}")
    plt.close()


def plot_combined_metrics(results: Dict, output_path: Path):
    """
    Plot 3: Combined C-Index and IBS comparison
    
    Side-by-side comparison of both metrics.
    """
    print("\n--- Plot 3: Combined Metrics ---")
    
    datasets = [k for k in results.keys() if k.lower() in DATASET_CONFIG]
    phases = ['stacking', 'nfg', 'hybrid']
    
    if not datasets:
        print("No valid datasets found in results")
        return
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    x = np.arange(len(datasets))
    width = 0.25
    
    # C-Index plot
    ax1.set_facecolor('#fafafa')
    for i, phase in enumerate(phases):
        values = []
        errors = []
        for dataset in datasets:
            phase_data = results[dataset].get('phases', {}).get(phase, {})
            mean = phase_data.get('mean_metrics', {}).get('c_index_mean', 0)
            std = phase_data.get('std_metrics', {}).get('c_index_mean', 0)
            values.append(mean)
            errors.append(std)
        
        offset = (i - 1) * width
        bars = ax1.bar(x + offset, values, width, yerr=errors, capsize=3,
                      label=PHASE_LABELS[phase], color=PHASE_COLORS[phase],
                      edgecolor='white', linewidth=1.5, alpha=0.9)
        
        for bar, val in zip(bars, values):
            if val > 0:
                ax1.annotate(f'{val:.3f}',
                            xy=(bar.get_x() + bar.get_width() / 2, val),
                            xytext=(0, 3), textcoords="offset points",
                            ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    ax1.set_title('Cause-Specific C-Index\n(Higher is Better)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('C-Index', fontsize=11)
    ax1.set_xlabel('Dataset', fontsize=11)
    ax1.set_xticks(x)
    ax1.set_xticklabels([DATASET_CONFIG.get(d.lower(), {}).get('name', d) for d in datasets])
    ax1.legend(fontsize=9, loc='lower right')
    ax1.grid(axis='y', linestyle='--', alpha=0.4)
    ax1.set_axisbelow(True)
    
    # IBS plot
    ax2.set_facecolor('#fafafa')
    for i, phase in enumerate(phases):
        values = []
        errors = []
        for dataset in datasets:
            phase_data = results[dataset].get('phases', {}).get(phase, {})
            mean = phase_data.get('mean_metrics', {}).get('ibs_mean', 0)
            std = phase_data.get('std_metrics', {}).get('ibs_mean', 0)
            values.append(mean)
            errors.append(std)
        
        offset = (i - 1) * width
        bars = ax2.bar(x + offset, values, width, yerr=errors, capsize=3,
                      label=PHASE_LABELS[phase], color=PHASE_COLORS[phase],
                      edgecolor='white', linewidth=1.5, alpha=0.9)
        
        for bar, val in zip(bars, values):
            if val > 0:
                ax2.annotate(f'{val:.3f}',
                            xy=(bar.get_x() + bar.get_width() / 2, val),
                            xytext=(0, 3), textcoords="offset points",
                            ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    ax2.set_title('CIF-based IBS\n(Lower is Better)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Integrated Brier Score', fontsize=11)
    ax2.set_xlabel('Dataset', fontsize=11)
    ax2.set_xticks(x)
    ax2.set_xticklabels([DATASET_CONFIG.get(d.lower(), {}).get('name', d) for d in datasets])
    ax2.legend(fontsize=9, loc='upper right')
    ax2.grid(axis='y', linestyle='--', alpha=0.4)
    ax2.set_axisbelow(True)
    
    plt.suptitle('Competing Risks Benchmark: 3-Phase Comparison\n(5-Fold Cross-Validation)',
                fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', dpi=150)
    print(f"Saved: {output_path}")
    plt.close()


def create_summary_table(results: Dict, output_path: Path):
    """Create a summary table in text format."""
    print("\n--- Creating Summary Table ---")
    
    datasets = [k for k in results.keys() if k.lower() in DATASET_CONFIG]
    phases = ['stacking', 'nfg', 'hybrid']
    
    lines = []
    lines.append("=" * 80)
    lines.append("COMPETING RISKS BENCHMARK SUMMARY")
    lines.append("=" * 80)
    lines.append("")
    
    for dataset in datasets:
        dataset_name = DATASET_CONFIG.get(dataset.lower(), {}).get('name', dataset)
        n_risks = results[dataset].get('n_risks', 2)
        
        lines.append(f"Dataset: {dataset_name}")
        lines.append(f"Risks: {n_risks}")
        lines.append("-" * 60)
        lines.append(f"{'Method':<25} {'C-Index':<20} {'IBS':<20}")
        lines.append("-" * 60)
        
        for phase in phases:
            phase_data = results[dataset].get('phases', {}).get(phase, {})
            mean = phase_data.get('mean_metrics', {})
            std = phase_data.get('std_metrics', {})
            
            c_idx = mean.get('c_index_mean', 0)
            c_std = std.get('c_index_mean', 0)
            ibs = mean.get('ibs_mean', 0)
            ibs_std = std.get('ibs_mean', 0)
            
            c_str = f"{c_idx:.4f} ± {c_std:.4f}" if c_idx > 0 else "N/A"
            ibs_str = f"{ibs:.4f} ± {ibs_std:.4f}" if ibs > 0 else "N/A"
            
            lines.append(f"{PHASE_LABELS[phase]:<25} {c_str:<20} {ibs_str:<20}")
        
        lines.append("")
        
        # Per-risk breakdown
        lines.append("Per-Risk Breakdown:")
        for risk in range(1, n_risks + 1):
            lines.append(f"  Risk {risk}:")
            for phase in phases:
                phase_data = results[dataset].get('phases', {}).get(phase, {})
                mean = phase_data.get('mean_metrics', {})
                c_idx = mean.get(f'c_index_risk{risk}', 0)
                ibs = mean.get(f'ibs_risk{risk}', 0)
                if c_idx > 0 or ibs > 0:
                    lines.append(f"    {PHASE_LABELS[phase]}: C-Index={c_idx:.4f}, IBS={ibs:.4f}")
        
        lines.append("")
        lines.append("=" * 80)
        lines.append("")
    
    summary = "\n".join(lines)
    
    with open(output_path, 'w') as f:
        f.write(summary)
    
    print(summary)
    print(f"\nSaved: {output_path}")


def generate_all_plots(results_dir: Path = RESULTS_DIR, output_dir: Optional[Path] = None):
    """Generate all visualization plots."""
    if output_dir is None:
        output_dir = results_dir / 'plots'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("GENERATING COMPETING RISKS BENCHMARK VISUALIZATIONS")
    print("=" * 70)
    
    # Load results
    results = load_benchmark_results(results_dir)
    
    if not results:
        print(f"No results found in {results_dir}")
        return
    
    print(f"Found results for: {list(results.keys())}")
    
    # Generate plots
    plot_cindex_comparison(results, output_dir / 'cindex_comparison.png', per_risk=False)
    plot_cindex_comparison(results, output_dir / 'cindex_comparison_per_risk.png', per_risk=True)
    plot_ibs_comparison(results, output_dir / 'ibs_comparison.png', per_risk=False)
    plot_ibs_comparison(results, output_dir / 'ibs_comparison_per_risk.png', per_risk=True)
    plot_combined_metrics(results, output_dir / 'combined_metrics.png')
    create_summary_table(results, output_dir / 'summary.txt')
    
    print("\n" + "=" * 70)
    print(f"All plots saved to: {output_dir}")
    print("=" * 70)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Visualize Competing Risks Benchmark Results')
    parser.add_argument('--results-dir', type=str, default=None,
                       help='Directory containing benchmark results')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Directory to save plots')
    
    args = parser.parse_args()
    
    results_dir = Path(args.results_dir) if args.results_dir else RESULTS_DIR
    output_dir = Path(args.output_dir) if args.output_dir else None
    
    generate_all_plots(results_dir, output_dir)
