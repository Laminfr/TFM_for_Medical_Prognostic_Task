#!/bin/bash
#SBATCH --job-name=competing_risks_benchmark
#SBATCH --output=CompetingRisks/logs/competing_risks-%j.out
#SBATCH --error=CompetingRisks/logs/competing_risks-%j.err
#SBATCH --partition=universe
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --qos=master-queuesave

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

# Navigate to project root
cd /vol/miltank/users/sajb/Project/NeuralFineGray

# Activate environment
ml python/anaconda3
source activate /vol/miltank/users/sajb/Project/tabicl_env

# Create logs directory if needed
mkdir -p CompetingRisks/logs
mkdir -p results/competing_risks

# Set seed for reproducibility
export PYTHONHASHSEED=42

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
