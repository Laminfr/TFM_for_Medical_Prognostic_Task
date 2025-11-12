# Baseline Models for Single-Event Survival Analysis on METABRIC

This directory contains baseline model implementations for single-event survival analysis on the METABRIC dataset.

## Models Implemented

1. **Cox Proportional Hazards** (`cox_metabric_eval.py`)
   - Classic semi-parametric model
   - Uses `lifelines` library
   - L2 regularization (penalizer=0.01)

2. **XGBoost Survival** (`train_xgboost.py`)
   - Gradient boosted Cox model
   - Uses `xgboost` with `survival:cox` objective
   - 200 trees, learning_rate=0.05, max_depth=4

3. **DeepSurv** (`DeepSurv.py`)
   - Deep neural network Cox model
   - PyTorch implementation
   - Architecture: [64, 32, 16] hidden layers with BatchNorm and Dropout
   - Trained for 100 epochs with Adam optimizer

4. **Random Survival Forest** (`RSF.py`)
   - Ensemble of survival trees
   - Uses `scikit-survival` implementation
   - 200 trees with max_depth=10

## Data Loading

All models use the shared `data_loader.py` which:
- Loads METABRIC dataset from `pycox`
- Performs 80/20 train/validation split (random_state=42)
- Stratifies by event
- Standardizes features using StandardScaler

## Evaluation Metrics

All models report:
- **C-index** (Concordance Index): Discrimination metric (↑ better)
  - Computed on full validation set
- **IBS** (Integrated Brier Score): Calibration metric (↓ better)
  - Computed on validation samples with time < max(training time)
  - Time grid uses 50 quantiles of event times

## Running the Models

### Locally
```bash
cd /vol/miltank/users/sajb/Project/NeuralFineGray
python -m examples.baselines.cox_metabric_eval
python -m examples.baselines.train_xgboost
python -m examples.baselines.DeepSurv
python -m examples.baselines.RSF
```

### Using SLURM
```bash
cd /vol/miltank/users/sajb/Project/NeuralFineGray/examples/baselines
sbatch run_cox_eval.sbatch
sbatch run_xgboost.sbatch
sbatch run_deepsurv.sbatch
sbatch run_rsf.sbatch
```

## File Structure

```
baselines/
├── __init__.py                 # Package initialization
├── data_loader.py             # Shared data loading/preprocessing
├── cox_metabric_eval.py       # Cox PH model
├── train_xgboost.py           # XGBoost survival model
├── DeepSurv.py                # Deep Cox model
├── RSF.py                     # Random Survival Forest
├── run_cox_eval.sbatch        # SLURM script for Cox
├── run_xgboost.sbatch         # SLURM script for XGBoost
├── run_deepsurv.sbatch        # SLURM script for DeepSurv
├── run_rsf.sbatch             # SLURM script for RSF
└── logs/                      # Output logs from SLURM jobs
```

## Dependencies

- `numpy`
- `pandas`
- `scikit-learn`
- `pycox`
- `scikit-survival` (sksurv)
- `lifelines`
- `xgboost`
- `torch` (PyTorch)

## Implementation Notes

### IBS Calculation
All models follow the same pattern for IBS:
1. Filter validation samples to time < max(training time)
2. Create safe time grid at 95% of max training time
3. Use quantiles of event times when sufficient events exist
4. Interpolate survival predictions to the time grid
5. Compute IBS using `sksurv.metrics.integrated_brier_score`

### Risk Score Clipping
For Cox-based models (Cox PH, XGBoost, DeepSurv), risk scores are clipped to [-10, 10] to prevent numerical overflow when computing `exp(risk_score)`.

### Survival Prediction
- **Cox PH, XGBoost, DeepSurv**: Use Breslow estimator with baseline survival from Kaplan-Meier
- **RSF**: Directly predicts survival functions, interpolated to evaluation time grid

## Expected Performance

Typical performance on METABRIC (for reference):
- Cox PH: C-index ~0.63-0.65
- XGBoost: C-index ~0.64-0.66
- DeepSurv: C-index ~0.63-0.66
- RSF: C-index ~0.65-0.67

Results may vary due to random initialization and hyperparameter choices.
