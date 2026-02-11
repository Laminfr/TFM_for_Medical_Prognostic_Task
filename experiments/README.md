# Experiments

Unified experiment runner for survival analysis models with hyperparameter search.

## Quick Start

```bash
# Run CoxPH on all datasets
sbatch experiments/run_experiment.sbatch all coxph raw

# Run specific model on specific dataset
sbatch experiments/run_experiment.sbatch METABRIC deepsurv raw
sbatch experiments/run_experiment.sbatch SUPPORT rsf raw
sbatch experiments/run_experiment.sbatch PBC xgboost raw
sbatch experiments/run_experiment.sbatch METABRIC nfg raw

# Run with TabPFN embeddings
sbatch experiments/run_experiment.sbatch METABRIC coxph tabpfn
```

## Usage

```bash
sbatch experiments/run_experiment.sbatch DATASET MODEL MODE [GRID] [SEED] [FOLD]
```

### Arguments (positional)

| Argument | Values | Description |
|----------|--------|-------------|
| DATASET | METABRIC, SUPPORT, PBC, all | Dataset to run on |
| MODEL | coxph, deepsurv, rsf, xgboost, nfg | Survival model |
| MODE | raw, tabpfn | Feature mode (raw or TabPFN embeddings) |
| GRID | integer (default: 100) | Random search iterations |
| SEED | integer (default: 0) | Random seed |
| FOLD | integer or empty | Specific fold (0-4), or empty for all |

## Models

| Model | Description |
|-------|-------------|
| coxph | Cox Proportional Hazards (lifelines) |
| deepsurv | Deep Cox neural network (PyTorch) |
| rsf | Random Survival Forest (scikit-survival) |
| xgboost | XGBoost with Cox objective |
| nfg | Neural Fine-Gray |


## Output

Results saved to `results/` as CSV files:
- `{DATASET}_raw_{MODEL}.csv` - raw features
- `{DATASET}_tabpfn_{MODEL}.csv` - TabPFN embeddings

Logs saved to `logs/`:
- `exp_exp_run-{JOBID}.out` - stdout
- `exp_exp_run-{JOBID}.err` - stderr

## Evaluation Metrics

| Metric | Description |
|--------|-------------|
| C-index | Concordance index (discrimination, higher is better) |
| IBS | Integrated Brier Score (calibration, lower is better) |


## Direct Python Usage

```bash
# Run directly without SLURM
python -m experiments.run_experiment --dataset METABRIC --model coxph --mode raw
python -m experiments.run_experiment --dataset all --model xgboost --mode tabpfn --grid-search 50
```

### Python Arguments

```
--dataset, -d     Dataset name (required)
--model, -m       Model type (required)
--mode            Feature mode: raw or tabpfn (default: raw)
--fold, -f        Specific fold to run
--grid-search, -g Random search iterations (default: 100)
--seed, -s        Random seed (default: 0)
--output-dir, -o  Custom output directory
```
