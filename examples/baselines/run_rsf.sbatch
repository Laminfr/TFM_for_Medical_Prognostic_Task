#!/bin/bash
#SBATCH --job-name=rsf_eval
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --time=0-02:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --qos=master-queuesave

# === Load anaconda ===
ml python/anaconda3

# === Activate env (use source instead of conda) ===
source activate /meta/users/sajb/projectenv

# === Go to project ===
cd /vol/miltank/users/sajb/Project/NeuralFineGray

# === Ensure logs directory exists ===
mkdir -p examples/baselines/logs

# === Run Random Survival Forest baseline evaluation ===
python -m examples.baselines.RSF
