import numpy as np
from sksurv.ensemble import RandomSurvivalForest
from sksurv.metrics import concordance_index_censored, integrated_brier_score
from sksurv.util import Surv

# Import the shared data loader
from .data_loader import load_and_preprocess_data


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
    """Evaluate Random Survival Forest model."""
    
    # --- C-index using scikit-survival ---
    c_index_train = model.score(X_train, y_train)
    c_index_val = model.score(X_val, y_val)
    
    print(f"C-index (train): {c_index_train:.4f}")
    print(f"C-index (val):   {c_index_val:.4f}")

    # --- Integrated Brier Score ---
    # Filter validation set to be within the time range of the training data
    max_time_train = y_train['time'].max()
    valid_mask = y_val['time'] < max_time_train
    y_val_filtered = y_val[valid_mask]
    X_val_filtered = X_val[valid_mask]
    
    print(f"Validation samples for IBS: {len(y_val_filtered)}/{len(y_val)} " +
          f"({100*len(y_val_filtered)/len(y_val):.1f}%)")
    
    if len(y_val_filtered) == 0:
        print("WARNING: No validation samples within training time range!")
        return {
            "c_index_train": c_index_train,
            "c_index_val": c_index_val,
            "ibs_val": np.nan
        }
    
    # Create safe time grid within the range of filtered validation data
    max_time_safe = max_time_train * 0.95
    event_times = y_val_filtered['time'][y_val_filtered['event'].astype(bool)]
    event_times = event_times[event_times < max_time_safe]
    
    if len(event_times) > 50:
        time_grid = np.quantile(event_times, np.linspace(0.1, 0.9, 50))
    else:
        time_grid = np.linspace(y_val_filtered['time'].min() + 1, max_time_safe, 50)
    
    time_grid = np.unique(time_grid)
    
    print(f"Time grid range: [{time_grid.min():.2f}, {time_grid.max():.2f}]")
    print(f"Max time (train): {max_time_train:.2f}")
    
    # Get survival predictions for filtered validation set
    # RSF returns survival functions directly
    survival_probs = model.predict_survival_function(X_val_filtered, return_array=True)
    
    # The survival functions are evaluated at the unique event times from training
    # We need to interpolate to our time grid
    train_times = model.unique_times_
    
    # Interpolate survival probabilities to our time grid
    survival_probs_interpolated = np.zeros((len(y_val_filtered), len(time_grid)))
    for i in range(len(y_val_filtered)):
        survival_probs_interpolated[i] = np.interp(
            time_grid, 
            train_times, 
            survival_probs[i],
            left=1.0,  # Before first time point, survival = 1
            right=survival_probs[i][-1]  # After last time point, use last value
        )
    
    # Calculate IBS
    ibs = integrated_brier_score(y_train, y_val_filtered, survival_probs_interpolated, time_grid)
    
    return {"c_index_train": c_index_train, "c_index_val": c_index_val, "ibs_val": ibs}


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
    print(f"C-index (train): {metrics['c_index_train']:.4f}")
    print(f"C-index (val):   {metrics['c_index_val']:.4f}")
    if not np.isnan(metrics['ibs_val']):
        print(f"IBS (val):       {metrics['ibs_val']:.4f}")
    else:
        print(f"IBS (val):       N/A (no valid samples)")
    print("="*50)
    print("\nNote: C-index computed on full validation set,")
    print("      IBS computed on validation samples with time < max(train time)")


if __name__ == "__main__":
    main()
