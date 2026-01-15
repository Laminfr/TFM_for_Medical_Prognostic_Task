#!/bin/bash
#SBATCH --job-name=all_baselines
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --time=0-04:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --qos=master-queuesave

# === Load anaconda ===
ml python/anaconda3

# === Activate env (use source instead of conda) ===
source activate /meta/users/sajb/projectenv

# === Go to project ===
cd /vol/miltank/users/sajb/Project/NeuralFineGray

# === Ensure logs directory exists ===
mkdir -p examples/baselines/logs

echo "=========================================="
echo "Running All Baseline Models on METABRIC"
echo "=========================================="
echo ""

echo "1. Cox Proportional Hazards..."
echo "-------------------------------------------"
python -m examples.baselines.cox_metabric_eval
echo ""

echo "2. XGBoost Survival..."
echo "-------------------------------------------"
python -m examples.baselines.train_xgboost
echo ""

echo "3. DeepSurv (Neural Cox)..."
echo "-------------------------------------------"
python -m examples.baselines.DeepSurv
echo ""

echo "4. Random Survival Forest..."
echo "-------------------------------------------"
python -m examples.baselines.RSF
echo ""

echo "=========================================="
echo "All baselines completed!"
echo "=========================================="
