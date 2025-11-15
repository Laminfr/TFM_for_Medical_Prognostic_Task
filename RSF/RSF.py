import numpy as np
from sksurv.ensemble import RandomSurvivalForest

# Import the shared data loader
from datasets.data_loader import load_and_preprocess_data

# Import metrics from neuralfg repository
import sys
sys.path.insert(0, '/vol/miltank/users/sajb/Project/NeuralFineGray')
from metrics.calibration import integrated_brier_score
from metrics.discrimination import truncated_concordance_td


def train_rsf_model(X_train, y_train, n_estimators=200, max_depth=10, min_samples_split=20, 
                    min_samples_leaf=10, random_state=42):
    """Train Random Survival Forest model."""
    print(f"Training RSF with {n_estimators} trees...")
    
    model = RandomSurvivalForest(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        random_state=random_state,
        n_jobs=-1,  # Use all available cores
        verbose=1
    )
    
    model.fit(X_train, y_train)
    return model


def evaluate_rsf_model(model, X_train, X_val, y_train, y_val):
    """Evaluate Random Survival Forest model using NeuralFineGray metrics."""
    
    # Extract time and event arrays from structured arrays
    t_train = y_train['time']
    e_train = y_train['event'].astype(int)
    t_val = y_val['time']
    e_val = y_val['event'].astype(int)
    
    # Filter validation set to be within the time range of the training data
    max_time_train = t_train.max()
    valid_mask = t_val < max_time_train
    t_val_filtered = t_val[valid_mask]
    e_val_filtered = e_val[valid_mask]
    X_val_filtered = X_val[valid_mask]
    
    print(f"Validation samples for evaluation: {len(t_val_filtered)}/{len(t_val)} " +
          f"({100*len(t_val_filtered)/len(t_val):.1f}%)")
    
    if len(t_val_filtered) == 0:
        print("WARNING: No validation samples within training time range!")
        return {
            "c_index_train": np.nan,
            "c_index_val": np.nan,
            "ibs_val": np.nan
        }
    
    # Create safe time grid within the range of filtered validation data
    max_time_safe = max_time_train * 0.95
    event_times = t_val_filtered[e_val_filtered.astype(bool)]
    event_times = event_times[event_times < max_time_safe]
    
    if len(event_times) > 50:
        time_grid = np.quantile(event_times, np.linspace(0.1, 0.9, 50))
    else:
        time_grid = np.linspace(max(t_val_filtered.min(), 1), max_time_safe, 50)
    
    time_grid = np.unique(time_grid)
    
    print(f"Time grid range: [{time_grid.min():.2f}, {time_grid.max():.2f}]")
    print(f"Max time (train): {max_time_train:.2f}")
    
    # Get survival predictions from RSF
    # RSF returns survival functions directly
    survival_probs_train = model.predict_survival_function(X_train, return_array=True)
    survival_probs_val = model.predict_survival_function(X_val_filtered, return_array=True)
    
    # The survival functions are evaluated at the unique event times from training
    # We need to interpolate to our time grid
    train_times = model.unique_times_
    
    # Interpolate survival probabilities to our time grid
    survival_probs_train_interpolated = np.zeros((len(X_train), len(time_grid)))
    for i in range(len(X_train)):
        survival_probs_train_interpolated[i] = np.interp(
            time_grid, 
            train_times, 
            survival_probs_train[i],
            left=1.0,  # Before first time point, survival = 1
            right=survival_probs_train[i][-1]  # After last time point, use last value
        )
    
    survival_probs_val_interpolated = np.zeros((len(X_val_filtered), len(time_grid)))
    for i in range(len(X_val_filtered)):
        survival_probs_val_interpolated[i] = np.interp(
            time_grid, 
            train_times, 
            survival_probs_val[i],
            left=1.0,
            right=survival_probs_val[i][-1]
        )
    
    # Convert survival to cumulative incidence (risk) for the metrics
    # For single event survival: F(t) = 1 - S(t)
    risk_predicted_train = 1 - survival_probs_train_interpolated
    risk_predicted_val = 1 - survival_probs_val_interpolated
    
    # Prepare IPCW estimator (Kaplan-Meier on censoring distribution)
    # For your metrics: pass tuple (e_train, t_train)
    km = (e_train, t_train)
    
    # For single event survival analysis, competing_risk=1 represents THE event
    # e=0 represents censoring, e=1 represents the event of interest
    competing_risk = 1
    
    # --- C-index using truncated concordance ---
    c_index_train, km = truncated_concordance_td(
        e_train, 
        t_train, 
        risk_predicted_train, 
        time_grid, 
        time_grid[-1],
        km=km,
        competing_risk=competing_risk
    )
    
    c_index_val, km = truncated_concordance_td(
        e_val_filtered, 
        t_val_filtered, 
        risk_predicted_val, 
        time_grid, 
        time_grid[-1],
        km=km,
        competing_risk=competing_risk
    )
    
    # --- Integrated Brier Score ---
    ibs_val, km = integrated_brier_score(
        e_val_filtered,
        t_val_filtered,
        risk_predicted_val,
        time_grid,
        t_eval=time_grid,
        km=km,
        competing_risk=competing_risk
    )
    
    return {
        "c_index_train": c_index_train,
        "c_index_val": c_index_val,
        "ibs_val": ibs_val
    }


def main(dataset='METABRIC', normalize=True, test_size=0.2, random_state=42):
    print(f"Loading and preprocessing {dataset} dataset...")
    X_train, X_val, y_train, y_val = load_and_preprocess_data(
        dataset=dataset,
        normalize=normalize,
        test_size=test_size,
        random_state=random_state,
        as_sksurv_y=True
    )
    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    print(f"Features: {X_train.shape[1]}")
    print(f"Time range (train): [{y_train['time'].min():.2f}, {y_train['time'].max():.2f}]")
    print(f"Time range (val):   [{y_val['time'].min():.2f}, {y_val['time'].max():.2f}]")
    print(f"Event rate (train): {y_train['event'].mean():.2%}")
    print(f"Event rate (val):   {y_val['event'].mean():.2%}")
    
    print("\nTraining Random Survival Forest model...")
    model = train_rsf_model(X_train, y_train, n_estimators=200)
    
    print("\nEvaluating model...")
    metrics = evaluate_rsf_model(model, X_train, X_val, y_train, y_val)
    
    print("\n" + "="*50)
    print("RANDOM SURVIVAL FOREST RESULTS")
    print("="*50)
    if not np.isnan(metrics['c_index_train']):
        print(f"C-index (train): {metrics['c_index_train']:.4f}")
    else:
        print(f"C-index (train): N/A")
    
    if not np.isnan(metrics['c_index_val']):
        print(f"C-index (val):   {metrics['c_index_val']:.4f}")
    else:
        print(f"C-index (val):   N/A")
    
    if not np.isnan(metrics['ibs_val']):
        print(f"IBS (val):       {metrics['ibs_val']:.4f}")
    else:
        print(f"IBS (val):       N/A")
    print("="*50)
    print("\nNote: C-index and IBS computed using NeuralFineGray metrics")
    print("      for single event survival analysis (competing_risk=1)")
    print("      Validation metrics computed on samples with time < max(train time)")


if __name__ == "__main__":
    main()