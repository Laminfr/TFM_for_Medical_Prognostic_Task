#!/bin/bash
#SBATCH --job-name=xgb_surv
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --time=0-02:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --qos=master-queuesave

# === Load Anaconda environment ===
ml python/anaconda3
source activate /meta/users/sajb/projectenv

# === Navigate to your project directory ===
cd /vol/miltank/users/sajb/Project/NeuralFineGray/examples/baselines

# === Ensure logs folder exists ===
mkdir -p logs

# === Run the XGBoost survival training script ===
python train_xgboost.py
