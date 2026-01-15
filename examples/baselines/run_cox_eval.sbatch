#!/bin/bash
#SBATCH --job-name=cox_eval
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --time=0-01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --qos=master-queuesave

# === Load anaconda ===
ml python/anaconda3

# === Activate env (use source instead of conda) ===
source activate /meta/users/sajb/projectenv

# === Go to project ===
cd /vol/miltank/users/sajb/Project/NeuralFineGray

# === Ensure logs directory exists ===
mkdir -p examples/baselines/logs

# === Run Cox baseline evaluation ===
python -m examples.baselines.cox_metabric_eval