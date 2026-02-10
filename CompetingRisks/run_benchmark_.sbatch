#!/bin/bash
#SBATCH --job-name=competing_risks_benchmark
#SBATCH --output=CompetingRisks/logs/competing_risks-%j.out
#SBATCH --error=CompetingRisks/logs/competing_risks-%j.err
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --qos=master-queuesave
#SBATCH --gres=gpu:1
#SBATCH --partition=universe   # only keep if REQUIRED on your cluster


# =============================================================================
# Competing Risks Benchmark
# Runs 3 methods: Stacking (XGBoost), NFG (neural), Hybrid (NFG+XGBoost)
# Default: SYNTHETIC_COMPETING dataset (auto-downloaded)
# To add SEER: place seernfg.csv in datasets/seer/ and add to --datasets
# =============================================================================

echo "=================================================="
echo "COMPETING RISKS BENCHMARK"
echo "Date: $(date)"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'None')"
echo "=================================================="
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
# Move to project root
###############################################################################
cd $PROJECT_ROOT

# Add DeepSurvivalMachines to Python path (it uses internal imports like "from dsm.xxx")
export PYTHONPATH="${PYTHONPATH:-}:$PROJECT_ROOT/DeepSurvivalMachines"
# Create logs directory if needed
mkdir -p CompetingRisks/logs
mkdir -p results/competing_risks

# Set seed for reproducibility
export PYTHONHASHSEED=42
###############################################################################
# Run experiment
###############################################################################
# Run benchmark (default: SYNTHETIC_COMPETING only)
# To run with SEER: add --datasets SYNTHETIC_COMPETING SEER_competing_risk
echo ""
echo "Starting benchmark on SYNTHETIC_COMPETING..."
python -m CompetingRisks.run_benchmark --datasets SYNTHETIC_COMPETING "$@"

# Generate visualizations
echo ""
echo "Generating visualizations..."
python -m CompetingRisks.visualize_results

echo ""
echo "=================================================="
echo "Benchmark complete!"
echo "Results saved to: results/competing_risks/"
echo "=================================================="
