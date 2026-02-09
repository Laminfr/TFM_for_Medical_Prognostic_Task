#!/bin/bash
#SBATCH --job-name=survstack_benchmark
#SBATCH --output=survivalStacking/logs/survstack_benchmark-%j.out
#SBATCH --error=survivalStacking/logs/survstack_benchmark-%j.err
#SBATCH --partition=universe
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=52:00:00


# Survival Stacking Full Benchmark
# Runs all 3 datasets with all methods (6 per dataset)

echo "=============================================="
echo "Survival Stacking Full Benchmark"
echo "=============================================="
echo "Started: $(date)"
echo "Node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo ""

# Activate conda environment (adjust path as needed)
ml python/anaconda3
source activate /vol/miltank/users/sajb/Project/tabicl_env

# HuggingFace token for TabPFN model access
# Load from .env file (keeps secrets out of version control)
if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
fi

# Navigate to project root
cd /vol/miltank/users/sajb/Project/NeuralFineGray

# Create logs directory if needed
mkdir -p survivalStacking/logs

# Parse arguments
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

# Run the benchmark
python -m survivalStacking.run_full_benchmark \
    --dataset $DATASET \
    --cv $CV_FOLDS \
    --n_intervals $N_INTERVALS \
    --weighting $WEIGHTING \
    --verbose

# Run visualization
echo ""
echo "=============================================="
echo "Generating Visualizations..."
echo "=============================================="

python -m survivalStacking.visualize_benchmark_results

echo ""
echo "=============================================="
echo "Completed: $(date)"
echo "=============================================="
