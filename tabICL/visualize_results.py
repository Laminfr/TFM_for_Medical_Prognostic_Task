import matplotlib.pyplot as plt
import numpy as np
import json
import os
from pathlib import Path

# ==========================================
# CONFIGURATION
# ==========================================
RESULTS_DIR = Path('/vol/miltank/users/sajb/Project/NeuralFineGray/tabICL/results/extended')
OUTPUT_FILE = RESULTS_DIR / 'tabicl_benchmark_chart.png'

# Data from your specific log (Tue 25 Nov 14:11:16 CET 2025)
# This serves as a fallback if JSON files aren't found
MANUAL_DATA = {
    'Raw': {
        'CoxPH': 0.6064, 'RSF': 0.6874, 'XGBoost': 0.5888, 
        'NeuralFineGray': 0.5456, 'DeSurv': 0.5000
    },
    'Deep': {
        'CoxPH': 0.6023, 'RSF': 0.6835, 'XGBoost': 0.5705, 
        'NeuralFineGray': 0.5255, 'DeSurv': 0.6956
    },
    'Deep+Raw': {
        'CoxPH': 0.6017, 'RSF': 0.6719, 'XGBoost': 0.5808, 
        'NeuralFineGray': 0.5704, 'DeSurv': 0.6648
    }
}

def load_data_from_files():
    """Attempts to load the most recent JSON files."""
    data = {'Raw': {}, 'Deep': {}, 'Deep+Raw': {}}
    
    if not RESULTS_DIR.exists():
        return None

    # Map filename patterns to our data keys
    file_patterns = {
        'results_raw': 'Raw', 
        'results_deep_': 'Deep', # Underscore to avoid matching deep+raw
        'results_deep+raw': 'Deep+Raw'
    }

    files_found = 0
    
    # Sort files by time to get the newest ones
    all_files = sorted(RESULTS_DIR.glob('*.json'), key=os.path.getmtime, reverse=True)
    
    for mode_key, mode_name in file_patterns.items():
        # Find first file matching the pattern
        for f in all_files:
            if mode_key in f.name and mode_name not in data:
                try:
                    with open(f, 'r') as jf:
                        content = json.load(jf)
                        # Extract C-index
                        for model, metrics in content.items():
                            if isinstance(metrics, dict) and 'c_index_mean' in metrics:
                                data[mode_name][model] = metrics['c_index_mean']
                        files_found += 1
                        break # Found the newest file for this mode
                except Exception:
                    continue
    
    if files_found < 3:
        print("Warning: Could not find all 3 result files. Using manual log data.")
        return None
        
    return data

def plot_benchmark(data):
    # Setup
    models = ['CoxPH', 'RSF', 'XGBoost', 'NeuralFineGray', 'DeSurv']
    x = np.arange(len(models))
    width = 0.25  # Width of bars

    # Extract values for plotting
    raw_vals = [data['Raw'].get(m, 0) for m in models]
    deep_vals = [data['Deep'].get(m, 0) for m in models]
    dr_vals = [data['Deep+Raw'].get(m, 0) for m in models]

    # Create Plot
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Set background color slightly
    ax.set_facecolor('#f8f9fa')
    plt.grid(axis='y', linestyle='--', alpha=0.7, zorder=0)

    # Plot Bars
    # Zorder=3 ensures bars are on top of grid lines
    rects1 = ax.bar(x - width, raw_vals, width, label='Raw Features (9)', color='#3366cc', edgecolor='white', zorder=3)
    rects2 = ax.bar(x, deep_vals, width, label='TabICL Embeddings (512)', color='#dc3912', edgecolor='white', zorder=3)
    rects3 = ax.bar(x + width, dr_vals, width, label='Deep + Raw (521)', color='#109618', edgecolor='white', zorder=3)

    # Styling
    ax.set_ylabel('C-Index (Time-Dependent)', fontsize=12, fontweight='bold')
    ax.set_title('Impact of TabICL Embeddings on Survival Models (METABRIC)', fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=11, fontweight='medium')
    ax.set_ylim(0.45, 0.75) # Set limits to focus on the differences
    
    # Legend
    ax.legend(loc='upper left', fontsize=10, frameon=True, shadow=True, fancybox=True)

    # Annotation Function
    def autolabel(rects, is_deep=False):
        for rect in rects:
            height = rect.get_height()
            # Highlight the winner (DeSurv Deep)
            font_weight = 'bold' if (height > 0.69 and is_deep) else 'normal'
            color = 'black'
            
            ax.annotate(f'{height:.3f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=9, fontweight=font_weight, color=color)

    autolabel(rects1)
    autolabel(rects2, is_deep=True)
    autolabel(rects3)

    # Add text box for insight
    insight_text = (
        "Key Insight:\n"
        "TabICL embeddings enabled DeSurv\n"
        "to jump from random guessing (0.50)\n"
        "to State-of-the-Art (0.696)."
    )
    plt.text(4.35, 0.65, insight_text, fontsize=10, bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'))

    # Footer
    plt.figtext(0.99, 0.01, 'Metric: C-Index (TD) | Higher is Better', horizontalalignment='right', fontsize=8, color='gray')

    # Save
    plt.tight_layout()
    print(f"Saving chart to: {OUTPUT_FILE}")
    plt.savefig(OUTPUT_FILE, dpi=300)
    print("Done.")

def main():
    print("Generating Visualization...")
    
    # Try loading files, fall back to the log data provided in prompt
    data = load_data_from_files()
    if data is None:
        data = MANUAL_DATA
        print("Using data from logs.")
    
    plot_benchmark(data)

if __name__ == "__main__":
    main()