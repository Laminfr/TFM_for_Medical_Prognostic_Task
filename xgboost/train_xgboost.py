import numpy as np
import xgboost as xgb
from sksurv.metrics import concordance_index_censored, integrated_brier_score
from sksurv.nonparametric import kaplan_meier_estimator

# Import the shared data loader
from datasets.data_loader import load_and_preprocess_data

def train_xgboost_model(X_train, y_train):
    """Train XGBoost Survival model."""
    # The Cox objective expects numeric time as the label and the event
    # indicator as the sample weight.
    y_time = y_train["time"]
    y_event = y_train["event"].astype(int)

    # Suggested hyperparameters are a good starting point
    model = xgb.XGBRegressor(
        objective="survival:cox",
        eval_metric="cox-nloglik",
        n_estimators=200,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
    )
    model.fit(X_train, y_time, sample_weight=y_event)
    return model

def evaluate_xgboost_model(model, X_train, X_val, y_train, y_val):
    """Evaluate XGBoost model."""
    
    # --- C-index using scikit-survival ---
    risk_scores_train = model.predict(X_train)
    risk_scores_val = model.predict(X_val)
    
    c_index_train = concordance_index_censored(
        y_train['event'], y_train['time'], risk_scores_train
    )[0]
    c_index_val = concordance_index_censored(
        y_val['event'], y_val['time'], risk_scores_val
    )[0]

    # --- Integrated Brier Score ---
    # Step 1: Estimate the baseline survival function from training data
    km_times, km_surv_probs = kaplan_meier_estimator(y_train["event"], y_train["time"])
    
    # Step 2: Filter validation set to be within the time range of the training data
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
    
    # Apply the same mask to the validation risk scores
    risk_scores_val_filtered = risk_scores_val[valid_mask]
    
    # Step 3: Create safe time grid within the range of filtered validation data
    # CRITICAL: Time grid must be strictly within the follow-up time of test data
    max_time_safe = max_time_train * 0.95  # Use 95% of max training time for safety
    event_times = y_val_filtered['time'][y_val_filtered['event'].astype(bool)]
    event_times = event_times[event_times < max_time_safe]
    
    if len(event_times) > 50:
        time_grid = np.quantile(event_times, np.linspace(0.1, 0.9, 50))
    else:
        time_grid = np.linspace(y_val_filtered['time'].min() + 1, max_time_safe, 50)
    
    time_grid = np.unique(time_grid)
    
    print(f"Time grid range: [{time_grid.min():.2f}, {time_grid.max():.2f}]")
    print(f"Max time (train): {max_time_train:.2f}")
    
    # Interpolate baseline survival at time grid points
    baseline_surv = np.interp(time_grid, km_times, km_surv_probs)

    # Step 4: Calculate survival probabilities for each sample in the filtered validation set
    # S(t|x) = S_0(t) ^ exp(risk_score)
    # Clip risk scores to prevent overflow
    risk_scores_clipped = np.clip(risk_scores_val_filtered, -10, 10)
    
    survival_probs = np.row_stack([
        baseline_surv ** np.exp(risk)
        for risk in risk_scores_clipped
    ])
    
    ibs = integrated_brier_score(y_train, y_val_filtered, survival_probs, time_grid)
    
    return {"c_index_train": c_index_train, "c_index_val": c_index_val, "ibs_val": ibs}

def main():
    print("Loading and preprocessing METABRIC dataset for sksurv...")
    X_train, X_val, y_train, y_val = load_and_preprocess_data(as_sksurv_y=True)

    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    print(f"Features: {X_train.shape[1]}")
    print(f"Time range (train): [{y_train['time'].min():.2f}, {y_train['time'].max():.2f}]")
    print(f"Time range (val):   [{y_val['time'].min():.2f}, {y_val['time'].max():.2f}]")
    print(f"Event rate (train): {y_train['event'].mean():.2%}")
    print(f"Event rate (val):   {y_val['event'].mean():.2%}")

    print("\nTraining XGBoost Survival model...")
    model = train_xgboost_model(X_train, y_train)
    
    print("\nEvaluating model...")
    metrics = evaluate_xgboost_model(model, X_train, X_val, y_train, y_val)
    
    print("\n" + "="*50)
    print("XGBOOST MODEL RESULTS")
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