# TFM: Tabular Foundation Model Embeddings for Survival Analysis

This folder contains the unified pipeline for evaluating survival models enhanced with tabular foundation model embeddings.

## Overview

Three embedding methods are supported:
- **TabICL**: Tabular In-Context Learning embeddings (512D)
- **TabPFN**: Tabular Prior-Data Fitted Network embeddings  
- **TARTE**: Tabular Representation Encoder embeddings (768D)

Each method generates embeddings that are used to enhance 4 baseline survival models:
- CoxPH (Cox Proportional Hazards)
- RSF (Random Survival Forest)
- XGBoost (XGBoost Survival)
- DeepSurv (Neural Cox Model)

## Evaluation Setup

- **Cross-validation**: 5-fold stratified CV
- **Random seed**: 42 (fixed for reproducibility)
- **Hyperparameter tuning**: 20 iterations random search per model
- **Feature modes**:
  - `raw`: Original features only (baseline)
  - `deep`: Embeddings only
  - `deep+raw`: Embeddings concatenated with original features
- **Metrics**: C-Index (at q25, q50, q75 time quantiles) and IBS

## Environments

| Method | Conda Environment | Python Version |
|--------|------------------|----------------|
| TabICL | tab_env | 3.11 |
| TabPFN | tab_env | 3.11 |
| TARTE  | tarte_env | 3.11 |

## Step-by-Step Guide

### 1. Run Experiments

Submit SLURM jobs to run the experiments on all 3 datasets (METABRIC, PBC, SUPPORT):

```bash
# Run TabICL experiments
sbatch tfm/run_tabicl.sbatch

# Run TabPFN experiments
sbatch tfm/run_tabpfn.sbatch

# Run TARTE experiments (uses different environment)
sbatch tfm/run_tarte.sbatch
```
#### Note

make sure to run 
```bash  
cd NeuralFineGray
python -m setup_tarte.py
```
before running TARTE experiments and 
```bash  
cd NeuralFineGray
python -m setup_tabpfn_tabicl.py
```
before running TabPFN or TabICL experiments.

Each job takes approximately 2-6 hours depending on the method and dataset size.

To check job status:
```bash
squeue -u $USER
```

To view logs:
```bash
# Standard output
tail -f tfm/logs/tabicl_cv_<JOBID>.out

# Errors
tail -f tfm/logs/tabicl_cv_<JOBID>.err
```

### 2. Check Results

Results are saved as JSON files in:
```
results/<method>/cv_<dataset>/cv_results_<date>.json
```

For example:
```
results/tabicl/cv_metabric/cv_results_20260203.json
results/tabpfn/cv_support/cv_results_20260203.json
results/tarte/cv_pbc/cv_results_20260203.json
```

### 3. Visualize Results

After experiments complete, generate plots:

```bash
# Activate appropriate environment
conda activate tab_env

# Generate plots for each method and dataset
python tfm/visualize_cv_results.py --method tabicl --dataset metabric
python tfm/visualize_cv_results.py --method tabicl --dataset pbc
python tfm/visualize_cv_results.py --method tabicl --dataset support

python tfm/visualize_cv_results.py --method tabpfn --dataset metabric
python tfm/visualize_cv_results.py --method tabpfn --dataset pbc
python tfm/visualize_cv_results.py --method tabpfn --dataset support

# For TARTE (use tarte_env)
conda activate tarte_env
python tfm/visualize_cv_results.py --method tarte --dataset metabric
python tfm/visualize_cv_results.py --method tarte --dataset pbc
python tfm/visualize_cv_results.py --method tarte --dataset support
```

Plots are saved to:
```
results/<method>/cv_<dataset>/plots/
```

### 4. Output Plots

The visualization script generates 3 plots per dataset:

1. **Embedding Impact by Model**: Compares raw vs deep vs deep+raw for each model
2. **Performance Across Time Horizons**: C-Index at early, median, and late time points
3. **Discrimination vs Calibration Trade-off**: C-Index vs IBS scatter plot

## File Structure

```
tfm/
  README.md                 # This file
  run_cv_analysis.py        # Main experiment script
  visualize_cv_results.py   # Plotting script
  run_tabicl.sbatch         # SLURM job for TabICL
  run_tabpfn.sbatch         # SLURM job for TabPFN
  run_tarte.sbatch          # SLURM job for TARTE
  logs/                     # SLURM output logs
```

## Running Individual Experiments

To run a single dataset manually (not via SLURM):

```bash
# Activate environment
conda activate tab_env

# Run TabICL on METABRIC
python tfm/run_cv_analysis.py --method tabicl --dataset METABRIC --n-iter 20

# Run TabPFN on PBC
python tfm/run_cv_analysis.py --method tabpfn --dataset PBC --n-iter 20
```

For TARTE, use the tarte_env environment.

## Troubleshooting

**Job fails immediately**: Check the .err log file for import errors or missing packages.

**Out of memory**: Reduce batch size in the hyperparameter grid in run_cv_analysis.py.

**TARTE import error**: Ensure you are using tarte_env (Python 3.11+).
