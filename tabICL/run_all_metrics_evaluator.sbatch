#!/bin/bash
#SBATCH --job-name=tabicl_extended_metrics
#SBATCH --output=tabICL/logs/%x-%j.out
#SBATCH --error=tabICL/logs/%x-%j.err
#SBATCH --time=0-08:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --qos=master-queuesave
#SBATCH --gres=gpu:1

# =====================================================================
# Extended TabICL Evaluation:
# CoxPH, RSF, XGBoost, NeuralFineGray (NFG), DeSurv
# Dataset: METABRIC, modes: raw / deep / deep+raw
# =====================================================================

# --- Load Anaconda module ---
ml python/anaconda3

# --- Activate experiment environment ---
# This env should contain:
#   tabicl, torch, lifelines, scikit-survival, auton-survival,
#   xgboost, and all NFG / DeSurv deps
source activate /vol/miltank/users/sajb/Project/tabicl_env

# --- Debug info ---
echo "=========================================="
echo "Extended TabICL Metrics Evaluation"
echo "Date:       $(date)"
echo "Host:       $(hostname)"
echo "Job ID:     ${SLURM_JOB_ID}"
echo "CPUs:       ${SLURM_CPUS_PER_TASK}"
echo "Memory:     ${SLURM_MEM_PER_NODE} MB"
echo "GPU(s):     ${CUDA_VISIBLE_DEVICES}"
echo "Python:     $(which python)"
python -c "import torch; print('Torch version:', torch.__version__); print('CUDA available:', torch.cuda.is_available())"
echo "=========================================="
echo ""

# --- Go to project root ---
cd /vol/miltank/users/sajb/Project/NeuralFineGray

# --- Ensure output dirs exist ---
mkdir -p tabICL/logs
mkdir -p tabICL/results/extended

# --- Run extended evaluation ---
echo "Starting extended evaluation (CoxPH, RSF, XGBoost, NFG, DeSurv)..."
python tabICL/all_metrics_evaluator.py
EXIT_CODE=$?

echo ""
echo "=========================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "Extended evaluation finished successfully."
else
    echo "Extended evaluation FAILED with exit code: $EXIT_CODE"
fi
echo "Completed at: $(date)"
echo "=========================================="
