# TFM for Medical Prognostic Task

Survival analysis framework based on [NeuralFineGray](https://github.com/Jeanselme/NeuralFineGray.git) models, Survival Stacking, and foundation model embeddings (TabICL, TabPFN, TARTE).

## Environment Setup

Two environments are available depending on the embedding method:

### TabICL/TabPFN Environment

```bash
# Create environment
conda create -n tab_env python=3.11
conda activate tab_env

# setup environment
cd NeuralFineGray
python -m setup_tabpfn_tabicl --install-deps
```

### TARTE Environment

```bash
# Create environment
conda create -n tarte_env python=3.11
conda activate tarte_env

# setup environment
cd NeuralFineGray
python -m setup_tarte --install-deps
```

### HuggingFace Token (for TabPFN)

```bash
export HF_TOKEN="your_token_here"
# Or create .env file with: HF_TOKEN=your_token_here
```
### Note

make sure to run 
```bash  
cd NeuralFineGray
python -m setup_tarte
```
before running TARTE experiments and 
```bash  
cd NeuralFineGray
python -m setup_tabpfn_tabicl
```
otherwise.

## Core Concepts

This framework provides four main experimental pipelines:

1. **Baseline Experiments** - Individual survival models (CoxPH, DeepSurv, RSF, XGBoost, NFG) with hyperparameter tuning
2. **Tabular Foundation Model Embeddings** - Enhance survival models with TabICL, TabPFN, or TARTE embeddings
3. **Survival Stacking** - Ensemble methods combining multiple base learners with optional embeddings
4. **Competing Risks Analysis** - Multi-event survival models using discrete-time approaches

## Available Datasets

| Dataset | Type | Description |
|---------|------|-------------|
| METABRIC | Binary survival | Breast cancer, ~2000 samples |
| SUPPORT | Binary survival | ICU mortality, ~9000 samples |
| PBC | Binary survival | Primary biliary cirrhosis, ~418 samples |
| SYNTHETIC_COMPETING | Competing risks | Synthetic data with 2 event types |
| SEER_competing_risk | Competing risks | Cancer registry (requires local file) |

## Experiment Guides

Each pipeline has detailed step-by-step instructions in its own README:

### 1. Baseline Experiments
📖 See [experiments/README.md](experiments/README.md) for detailed instructions on:
- Running individual models (CoxPH, DeepSurv, RSF, XGBoost, NFG)
- Hyperparameter search configuration
- Using raw features or TabPFN embeddings
- SLURM batch job submission

### 2. Tabular Foundation Model Embeddings
📖 See [tfm/README.md](tfm/README.md) for detailed instructions on:
- Generating TabICL, TabPFN, or TARTE embeddings
- Running cross-validation experiments
- Comparing raw vs deep vs deep+raw feature modes
- Environment-specific requirements

### 3. Survival Stacking
📖 See [survivalStacking/README.md](survivalStacking/README.md) for detailed instructions on:
- Running ensemble stacking benchmarks
- Combining base learners with embeddings
- Statistical significance testing
- Visualization of results

### 4. Competing Risks Analysis
📖 See [CompetingRisks/README.md](CompetingRisks/README.md) for detailed instructions on:
- Discrete-time multiclass approaches
- Hybrid NFG models
- Benchmarking on synthetic and real datasets

## Quick Examples

```bash
# Baseline experiment
python -m experiments.run_experiment --dataset METABRIC --model coxph --mode raw

# Survival stacking
python -m survivalStacking.run_full_benchmark --dataset METABRIC --cv 5

# Competing risks
python -m CompetingRisks.run_benchmark --datasets SYNTHETIC_COMPETING
```

## Results

All experiments save results to `results/` with organized subdirectories:
- `results/experiments/` - Baseline model results
- `results/tabicl/`, `results/tabpfn/`, `results/tarte/` - Embedding experiments
- `results/survival_stacking/` - Stacking ensemble results
- `results/competing_risks/` - Competing risks benchmarks

Plots are saved in `plots/` subdirectories within each results folder
https://github.com/SajbenDani
## Credits
This project is based on [NeuralFineGray](https://github.com/Jeanselme/NeuralFineGray.git) Copyright (c) 2021 Vincent Jeanselme, developed at TUM (Lab for AI in Medicine) by [Dániel Sajben](https://github.com/SajbenDani), [Amelie Trautwein](https://github.com/ATrtwn) and [Mohamed Amine Frouja](https://github.com/Laminfr) and supervised by Dmitrii Seletkov.

