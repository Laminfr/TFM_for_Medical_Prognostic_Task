#!/bin/bash
#SBATCH --job-name=tabicl_cv
#SBATCH --output=tfm/logs/tabicl_cv_%j.out
#SBATCH --error=tfm/logs/tabicl_cv_%j.err
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --qos=master-queuesave
#SBATCH --gres=gpu:1
#SBATCH --partition=universe   # only keep if REQUIRED on your cluster

# =============================================================================
# TabICL CV Analysis - All datasets
# 5-fold CV, seed 42, 20 random search iterations
# Datasets: METABRIC, PBC, SUPPORT
# =============================================================================
###############################################################################
echo "================================================="
echo "Unified Experiment Runner"
echo "Job ID:      $SLURM_JOB_ID"
echo "Node:        $(hostname)"
echo "Date:        $(date)"
echo "CPUs:        $SLURM_CPUS_PER_TASK"
echo "Memory:      $SLURM_MEM_PER_NODE MB"
echo "GPU(s):      ${CUDA_VISIBLE_DEVICES:-none}"
echo "================================================="
###############################################################################
# Load environment variables
###############################################################################
source env.common
echo "CONDA_ENV_PATH=$CONDA_ENV_PATH"
echo "PROJECT_ROOT=$PROJECT_ROOT"
###############################################################################
# Load Conda (NO conda init)
###############################################################################
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate $CONDA_ENV_PATH
##############################################################################
# Move to project root
###############################################################################
cd $PROJECT_ROOT

# Add DeepSurvivalMachines to Python path (it uses internal imports like "from dsm.xxx")
export PYTHONPATH="${PYTHONPATH:-}:$PROJECT_ROOT/DeepSurvivalMachines"

mkdir -p experiments/results

###############################################################################
# Run experiment
###############################################################################
echo "=========================================="
echo "TabICL CV Analysis - All Datasets"
echo "Date: $(date)"
echo "=========================================="

for DATASET in METABRIC PBC SUPPORT; do
    echo ""
    echo ">>> Running TabICL on $DATASET"
    echo "Started: $(date)"
    python tfm/run_cv_analysis.py --method tabicl --dataset $DATASET --n-iter 20
    echo "Completed $DATASET: $(date)"
done

echo ""
echo "=========================================="
echo "All TabICL experiments completed"
echo "Date: $(date)"
echo "=========================================="
