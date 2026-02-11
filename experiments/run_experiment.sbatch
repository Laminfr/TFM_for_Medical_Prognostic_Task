#!/bin/bash
#SBATCH --job-name=exp_run
#SBATCH --output=logs/exp_%x-%j.out
#SBATCH --error=logs/exp_%x-%j.err
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --partition=universe   # only keep if REQUIRED on your cluster

###############################################################################
# Experiment Runner for NeuralFineGray
#
# Usage: sbatch run_experiment.sbatch DATASET MODEL MODE [GRID] [SEED] [FOLD]
#
# Examples:
#   sbatch run_experiment.sbatch METABRIC coxph raw
#   sbatch run_experiment.sbatch all xgboost tabpfn 100 0
#   sbatch run_experiment.sbatch PBC nfg raw 50 42 0
#
# Arguments (positional):
#   DATASET  - METABRIC | SUPPORT | PBC | all
#   MODEL    - coxph | deepsurv | rsf | xgboost | nfg
#   MODE     - raw | tabpfn
#   GRID     - grid search iterations (default: 100)
#   SEED     - random seed (default: 0)
#   FOLD     - specific fold index, or empty for all folds
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
###############################################################################
# Debug: environment sanity
###############################################################################
python - <<EOF
import torch
print("Torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
EOF

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
# You can override ALL of these via:
# sbatch run_experiment.sbatch METABRIC coxph raw 100 0

DATASET=${1:-METABRIC}        # METABRIC | SUPPORT | PBC | all
MODEL=${2:-coxph}             # coxph | deepsurv | rsf | xgboost | nfg
MODE=${3:-raw}                # raw | tabpfn
GRID=${4:-100}                # random search iterations
SEED=${5:-0}                  # random seed
FOLD=${6:-""}                 # optional fold index (leave empty = all folds)

###############################################################################
# Build optional arguments
###############################################################################
FOLD_ARG=""
if [[ -n "$FOLD" ]]; then
    FOLD_ARG="--fold $FOLD"
fi

###############################################################################
# Run experiment
###############################################################################
echo ""
echo "Running configuration:"
echo "  Dataset:     $DATASET"
echo "  Model:       $MODEL"
echo "  Mode:        $MODE"
echo "  Grid search: $GRID"
echo "  Seed:        $SEED"
echo "  Fold:        ${FOLD:-all}"
echo ""

python -m experiments.run_experiment \
    --dataset "$DATASET" \
    --model "$MODEL" \
    --mode "$MODE" \
    --grid-search "$GRID" \
    --seed "$SEED" \
    $FOLD_ARG

###############################################################################
# Final status
###############################################################################
echo ""
echo "================================================="
echo "Experiment finished at: $(date)"
echo "================================================="
