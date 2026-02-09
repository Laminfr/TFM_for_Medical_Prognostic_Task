#!/bin/bash
#SBATCH --job-name=tarte_cv
#SBATCH --output=tfm/logs/tarte_cv_%j.out
#SBATCH --error=tfm/logs/tarte_cv_%j.err
#SBATCH --time=24:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --partition=universe
#SBATCH --qos=master-queuesave

# =============================================================================
# TARTE CV Analysis - All datasets
# 5-fold CV, seed 42, 20 random search iterations
# Datasets: METABRIC, PBC, SUPPORT
# =============================================================================

source activate /vol/miltank/users/sajb/Project/tarte_env
cd /vol/miltank/users/sajb/Project/NeuralFineGray
mkdir -p tfm/logs

echo "=========================================="
echo "TARTE CV Analysis - All Datasets"
echo "Date: $(date)"
echo "=========================================="

for DATASET in METABRIC PBC SUPPORT; do
    echo ""
    echo ">>> Running TARTE on $DATASET"
    echo "Started: $(date)"
    python tfm/run_cv_analysis.py --method tarte --dataset $DATASET --n-iter 20
    echo "Completed $DATASET: $(date)"
done

echo ""
echo "=========================================="
echo "All TARTE experiments completed"
echo "Date: $(date)"
echo "=========================================="
