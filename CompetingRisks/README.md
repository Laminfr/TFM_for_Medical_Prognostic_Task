# Competing Risks Benchmark

This package implements a 3-phase benchmark for competing risks analysis, comparing different modeling approaches on datasets with multiple competing event types.

## Overview

### The 3 Phases

1. **Phase 1: Multi-Class Survival Stacking (XGBoost)**
   - Transforms survival data to discrete-time person-period format
   - Uses multi-class XGBoost classifier (`objective='multi:softprob'`)
   - Targets: 0 = survived interval, 1 = Event A, 2 = Event B, etc.
   - Proves how well simple ML can handle competing risks

2. **Phase 2: Pure Neural Fine-Gray (NFG)**
   - Deep learning approach using the existing NFG implementation
   - Uses pure (X, T, E) data
   - Optimizes Fine-Gray subdistribution loss directly
   - Establishes the deep learning SOTA baseline

3. **Phase 3: Hybrid (NFG Embeddings + XGBoost)**
   - Trains NFG model to learn patient representations
   - Extracts embeddings from the shared representation layer
   - Feeds embeddings (+ optional raw features) to multi-class XGBoost
   - Tests whether deep features help tree-based classifiers

### Datasets

Uses the existing `datasets/datasets.py` module:
- **SYNTHETIC_COMPETING**: Synthetic data with 2 competing events from DeepHit repository (loads from URL)
- **SEER**: Real-world cancer survival data with competing causes of death (requires local file)

#### Adding SEER Dataset

The SEER dataset requires a local CSV file. To use it:

1. Place your SEER data file at:
   ```
   datasets/seer/seernfg.csv
   ```

2. The file should contain survival data with competing causes of death (e.g., cancer-specific death vs other causes).

3. Once the file exists, run the benchmark with:
   ```bash
   python -m CompetingRisks.run_benchmark --datasets SYNTHETIC_COMPETING SEER
   ```

**Note:** On first run, the loader will create a cleaned/reduced version at `datasets/seer/seernfg_cleaned.csv` for faster subsequent loading.

### Metrics

Uses the existing `metrics/` module:
- **Cause-Specific C-Index**: `metrics.discrimination.truncated_concordance_td`
- **CIF-based IBS**: `metrics.calibration.integrated_brier_score`

## Package Structure

```
CompetingRisks/
├── __init__.py                  # Package exports (uses existing datasets/metrics)
├── utils.py                     # Helper functions for competing risks
├── discrete_time_multiclass.py  # Multi-class discrete time transformer
├── stacking_multiclass.py       # Phase 1: Multi-class XGBoost stacking
├── nfg_wrapper.py              # Phase 2: NFG wrapper
├── hybrid_model.py             # Phase 3: NFG + XGBoost hybrid
├── run_benchmark.py            # Main benchmark runner
├── visualize_results.py        # Visualization scripts
└── run_benchmark.sbatch        # SLURM job script
```

**Note:** This package reuses the existing `datasets/` and `metrics/` modules to avoid code duplication.

## Usage

### Quick Start

```python
from CompetingRisks import run_full_benchmark

# Run complete benchmark
results = run_full_benchmark(
    datasets=['SYNTHETIC_COMPETING', 'SEER'],
    n_folds=5,
    phases=['stacking', 'nfg', 'hybrid']
)
```

### Individual Models

```python
from datasets.datasets import load_dataset
from CompetingRisks import (
    MultiClassSurvivalStacking,
    NFGCompetingRisks,
    HybridNFGStacking,
    split_data
)

# Load data using existing datasets module
X, T, E, features = load_dataset('SYNTHETIC_COMPETING')
n_risks = int(E.max())
splits = split_data(X, T, E)
X_train, T_train, E_train = splits['train']
X_test, T_test, E_test = splits['test']

# Phase 1: Multi-class Stacking
model1 = MultiClassSurvivalStacking(n_intervals=20)
model1.fit(X_train, T_train, E_train)
cif = model1.predict_cif(X_test)

# Phase 2: Neural Fine-Gray
model2 = NFGCompetingRisks(layers=[100, 100, 100])
model2.fit(X_train, T_train, E_train, n_iter=1000)
cif = model2.predict_cif(X_test)

# Phase 3: Hybrid
model3 = HybridNFGStacking(use_raw_features=True)
model3.fit(X_train, T_train, E_train)
cif = model3.predict_cif(X_test)
```

### Evaluation

```python
from metrics.discrimination import truncated_concordance_td
from metrics.calibration import integrated_brier_score

# Get CIF predictions and times
times = model.get_times()
cif = model.predict_cif(X_test)

# Compute Cause-Specific C-Index (using existing metrics)
t_median = np.median(T_test[E_test > 0])
c_idx, _ = truncated_concordance_td(
    E_test, T_test, cif[1], times, t_median,
    km=(E_train, T_train), competing_risk=1
)

# Compute IBS (using existing metrics)
ibs, _ = integrated_brier_score(
    E_test, T_test, cif[1], times,
    km=(E_train, T_train), competing_risk=1
)
```

### Running on Cluster

```bash
# Submit to SLURM
sbatch CompetingRisks/run_benchmark.sbatch

# Or run directly
python -m CompetingRisks.run_benchmark --datasets SYNTHETIC_COMPETING SEER --n-folds 5

# Generate visualizations
python -m CompetingRisks.visualize_results
```

## Results

Results are saved to `results/competing_risks/`:
- `{dataset}_benchmark_5fold.json` - Per-dataset results
- `full_benchmark_5fold.json` - Combined results
- `plots/` - Visualization plots

### Visualization Outputs

1. **cindex_comparison.png**: Column chart comparing C-Index across phases
2. **ibs_comparison.png**: Column chart comparing IBS across phases
3. **combined_metrics.png**: Side-by-side C-Index and IBS
4. **summary.txt**: Text summary table

## Key Design Decisions

1. **Discrete Time Multi-Class**: Instead of binary (event/no-event), we use multi-class classification where each event type gets its own class. This naturally handles competing risks.

2. **CIF Computation**: The Cumulative Incidence Function is computed from multi-class probabilities using the product-limit formula, properly accounting for competing events.

3. **Embedding Extraction**: NFG's shared representation layer provides learned patient embeddings that capture complex relationships for the hybrid approach.

4. **IPCW Weighting**: All metrics use Inverse Probability of Censoring Weighting for proper handling of censored observations.

## Dependencies

- numpy, pandas, scikit-learn
- xgboost
- torch
- lifelines
- matplotlib
- nfg (from parent project)

## References

- Neural Fine-Gray model for competing risks
- Survival Stacking discrete-time approach
- DeepHit synthetic competing risks data
