#!/bin/bash
#SBATCH --job-name=tabicl_visualize
#SBATCH --output=tabICL/logs/%x-%j.out
#SBATCH --error=tabICL/logs/%x-%j.err
#SBATCH --time=0-00:30:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --qos=master-queuesave

# ============================================================
# TabICL Visualization Job (Generates the benchmark bar chart)
# Uses: matplotlib, numpy, json
# No GPU needed
# ============================================================

# --- Load Anaconda module ---
ml python/anaconda3

# --- Activate environment ---
source activate /vol/miltank/users/sajb/Project/tabicl_env

# --- Debug Info ---
echo "=========================================="
echo "TabICL Visualization Script"
echo "Date:   $(date)"
echo "Host:   $(hostname)"
echo "Python: $(which python)"
echo "CPUs:   ${SLURM_CPUS_PER_TASK}"
echo "Mem:    ${SLURM_MEM_PER_NODE} MB"
echo "=========================================="
echo ""

# --- Move to project root ---
cd /vol/miltank/users/sajb/Project/NeuralFineGray

# --- Ensure log + result dirs exist ---
mkdir -p tabICL/logs
mkdir -p tabICL/results/extended

# --- Run visualization ---
echo "Running visualization..."
python tabICL/visualize_results.py
EXIT_CODE=$?

echo ""
echo "=========================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "Visualization completed successfully."
    echo "Output stored in: tabICL/results/extended/tabicl_benchmark_chart.png"
else
    echo "Visualization FAILED with exit code $EXIT_CODE"
fi
echo "Completed at: $(date)"
echo "=========================================="
