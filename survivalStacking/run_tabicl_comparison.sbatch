#!/bin/bash
#SBATCH --job-name=tabicl_cmp
#SBATCH --output=logs/tabicl_comparison-%j.out
#SBATCH --error=logs/tabicl_comparison-%j.err
#SBATCH --partition=universe
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8


# =============================================================================
# TabICL Comparison Benchmark Script
# Compares: XGBoost raw vs XGBoost+embeddings vs TabICL direct classification
# =============================================================================

echo "========================================"
echo "TABICL COMPARISON BENCHMARK"
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
mkdir -p results/tabicl_comparison

# Dataset from command line or defaults
DATASET=${1:-METABRIC}
CV_FOLDS=${2:-5}
DEVICE=${3:-cuda}

echo ""
echo "Configuration:"
echo "  Dataset: $DATASET"
echo "  CV Folds: $CV_FOLDS"
echo "  Device: $DEVICE"
echo ""

# Run the comparison experiment
python -m survivalStacking.run_tabicl_comparison \
    --dataset "$DATASET" \
    --cv "$CV_FOLDS" \
    --device "$DEVICE"

echo ""
echo "========================================"
echo "Completed at: $(date)"
echo "========================================"
