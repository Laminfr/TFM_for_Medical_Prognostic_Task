#!/bin/bash
#SBATCH --job-name=surv_stack
#SBATCH --output=logs/survival_stacking-%j.out
#SBATCH --error=logs/survival_stacking-%j.err
#SBATCH --partition=universe
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8


# =============================================================================
# Survival Stacking Benchmark Script
# Runs Survival Stacking with TabICL embeddings on METABRIC and PBC datasets
# =============================================================================

echo "========================================"
echo "SURVIVAL STACKING BENCHMARK"
echo "========================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Date: $(date)"
echo "========================================"

# Activate conda environment (adjust path as needed)
ml python/anaconda3
source activate /vol/miltank/users/sajb/Project/tabicl_env

# Navigate to project directory
cd /vol/miltank/users/sajb/Project/NeuralFineGray

# Create logs directory
mkdir -p logs
mkdir -p results/survival_stacking

# Dataset and mode from command line or defaults
DATASET=${1:-METABRIC}
MODE=${2:-deep+raw}
CV_FOLDS=${3:-5}
N_INTERVALS=${4:-20}

echo ""
echo "Configuration:"
echo "  Dataset: $DATASET"
echo "  Mode: $MODE"
echo "  CV Folds: $CV_FOLDS"
echo "  Intervals: $N_INTERVALS"
echo ""

# Run the experiment
python -m survivalStacking.run_experiment \
    --dataset "$DATASET" \
    --mode "$MODE" \
    --cv "$CV_FOLDS" \
    --n_intervals "$N_INTERVALS" \
    --verbose

echo ""
echo "========================================"
echo "Completed at: $(date)"
echo "========================================"
