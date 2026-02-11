#!/bin/bash
#SBATCH --job-name=survstack_benchmark
#SBATCH --output=survivalStacking/logs/survstack_benchmark-%j.out
#SBATCH --error=survivalStacking/logs/survstack_benchmark-%j.err
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --qos=master-queuesave
#SBATCH --gres=gpu:1
#SBATCH --partition=universe 

# Survival Stacking Full Benchmark
# Runs all 3 datasets with all methods (6 per dataset)
###############################################################################
echo "=============================================="
echo "Survival Stacking Full Benchmark"
echo "=============================================="
echo "Started: $(date)"
echo "Node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo ""
echo "=============================================="
echo "Unified Experiment Runner"
echo "Job ID:      $SLURM_JOB_ID"
echo "Node:        $(hostname)"
echo "Date:        $(date)"
echo "CPUs:        $SLURM_CPUS_PER_TASK"
echo "Memory:      $SLURM_MEM_PER_NODE MB"
echo "GPU(s):      ${CUDA_VISIBLE_DEVICES:-none}"
echo "=============================================="
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

# HuggingFace token for TabPFN model access
# Load from .env file (keeps secrets out of version control)
if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
fi
###############################################################################
# Move to project root
###############################################################################
cd $PROJECT_ROOT

# Add DeepSurvivalMachines to Python path (it uses internal imports like "from dsm.xxx")
export PYTHONPATH="${PYTHONPATH:-}:$PROJECT_ROOT/DeepSurvivalMachines"

mkdir -p experiments/results

###############################################################################
# ------------------- Experiment switches -------------------
###############################################################################
DATASET=${1:-all}
CV_FOLDS=${2:-5}
N_INTERVALS=${3:-20}
WEIGHTING=${4:-adaptive}

echo "Configuration:"
echo "  Dataset: $DATASET"
echo "  CV Folds: $CV_FOLDS"
echo "  N Intervals: $N_INTERVALS"
echo "  Weighting: $WEIGHTING"
echo ""

###############################################################################
# Run experiment
###############################################################################
python -m survivalStacking.run_full_benchmark \
    --dataset $DATASET \
    --cv $CV_FOLDS \
    --n_intervals $N_INTERVALS \
    --weighting $WEIGHTING \
    --verbose

###############################################################################
# Generate Visualizations
###############################################################################
echo ""
echo "=============================================="
echo "Generating Visualizations..."
echo "=============================================="

python -m survivalStacking.visualize_benchmark_results

echo ""
echo "=============================================="
echo "Completed: $(date)"
echo "=============================================="
