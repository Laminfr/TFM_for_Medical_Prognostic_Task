#!/bin/bash
# =============================================================================
# Run All Survival Stacking Experiments
# Submits jobs for METABRIC and PBC with all TabICL modes
# =============================================================================

cd /vol/miltank/users/sajb/Project/NeuralFineGray

# Create logs directory
mkdir -p logs
mkdir -p results/survival_stacking

echo "Submitting Survival Stacking experiments..."
echo ""

# METABRIC experiments
echo "=== METABRIC ==="
sbatch survivalStacking/run_experiment.sbatch METABRIC raw 5 20
sbatch survivalStacking/run_experiment.sbatch METABRIC deep 5 20
sbatch survivalStacking/run_experiment.sbatch METABRIC deep+raw 5 20

# PBC experiments
echo "=== PBC ==="
sbatch survivalStacking/run_experiment.sbatch PBC raw 5 20
sbatch survivalStacking/run_experiment.sbatch PBC deep 5 20
sbatch survivalStacking/run_experiment.sbatch PBC deep+raw 5 20

echo ""
echo "All jobs submitted. Check progress with: squeue -u $USER"
echo "Results will be saved to: results/survival_stacking/"
