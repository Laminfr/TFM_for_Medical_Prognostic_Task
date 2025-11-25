import numpy as np
import pandas as pd
from lifelines import CoxPHFitter

# Import metrics from neuralfg repository
import sys
from metrics.calibration import integrated_brier_score
from metrics.discrimination import truncated_concordance_td


def train_cox_model(X_train, T_train, E_train, penalizer=0.01):
    """Train Cox Proportional Hazards model."""

    # Prepare dataframe for lifelines
    df_train = X_train.copy()
    df_train['duration'] = T_train.values
    df_train['event'] = E_train.values
    
    # Fit Cox model
    cph = CoxPHFitter(penalizer=penalizer)
    cph.fit(df_train, duration_col='duration', event_col='event')
    
    return cph


def concordance_index_from_risk_scores(e, t, risk_scores, tied_tol=1e-8):
    """
    Compute C-index directly from risk scores (for Cox-like models).
    Higher risk score should correspond to higher risk (shorter survival).
    """
    event = e.values.astype(bool) if hasattr(e, 'values') else e.astype(bool)
    t = t.values if hasattr(t, 'values') else t
    n_events = event.sum()
    
    if n_events == 0:
        return np.nan
    
    concordant = 0
    permissible = 0
    
    for i in range(len(t)):
        if not event[i]:
            continue
            
        # Compare with all samples at risk at time t[i]
        at_risk = t > t[i]
        
        # Higher risk score means higher risk (shorter time to event)
        concordant += (risk_scores[at_risk] < risk_scores[i]).sum()
        concordant += 0.5 * (np.abs(risk_scores[at_risk] - risk_scores[i]) <= tied_tol).sum()
        permissible += at_risk.sum()
    
    if permissible == 0:
        return np.nan
    
    return concordant / permissible


def evaluate_model(cph, X_train, X_val, t_train, t_val, e_train, e_val):
    """Evaluate Cox model using NeuralFineGray metrics."""
    
    # Get risk scores (partial hazard = exp(X * beta))
    risk_scores_train = cph.predict_partial_hazard(X_train).values.flatten()
    risk_scores_val = cph.predict_partial_hazard(X_val).values.flatten()
    
    # --- C-index using risk scores directly ---
    c_index_train = concordance_index_from_risk_scores(e_train, t_train, risk_scores_train)
    c_index_val = concordance_index_from_risk_scores(e_val, t_val, risk_scores_val)
    
    # --- Integrated Brier Score ---
    # Filter validation set to be within the time range of the training data
    max_time_train = t_train.max()
    valid_mask = t_val.values < max_time_train
    t_val_filtered = t_val.values[valid_mask]
    e_val_filtered = e_val.values[valid_mask]
    X_val_filtered = X_val[valid_mask]
    
    print(f"Validation samples for IBS: {len(t_val_filtered)}/{len(t_val)} " +
          f"({100*len(t_val_filtered)/len(t_val):.1f}%)")
    
    if len(t_val_filtered) == 0:
        print("WARNING: No validation samples within training time range!")
        return {
            "c_index_train": c_index_train,
            "c_index_val": c_index_val,
            "ibs_val": np.nan
        }
    
    # Create safe time grid
    max_time_safe = max_time_train * 0.95
    event_times = t_val_filtered[e_val_filtered > 0]
    event_times = event_times[event_times < max_time_safe]
    
    if len(event_times) > 50:
        time_grid = np.quantile(event_times, np.linspace(0.1, 0.9, 50))
    else:
        time_grid = np.linspace(max(t_val_filtered.min(), 1), max_time_safe, 50)
    
    time_grid = np.unique(time_grid)
    
    print(f"Time grid range: [{time_grid.min():.2f}, {time_grid.max():.2f}]")
    print(f"Max time (train): {max_time_train:.2f}")
    
    # Get survival predictions for filtered validation set
    surv_funcs = cph.predict_survival_function(X_val_filtered, times=time_grid)
    surv_probs = surv_funcs.T.values
    
    # Convert survival to cumulative incidence (risk) for the IBS metric
    risk_predicted_val = 1 - surv_probs
    
    # Prepare IPCW estimator (Kaplan-Meier on censoring distribution)
    km = (e_train.values, t_train.values)
    
    # For single event survival analysis, competing_risk=1 represents THE event
    competing_risk = 1
    
    # Calculate IBS using NeuralFineGray metric
    ibs_val, km = integrated_brier_score(
        e_val_filtered.astype(int),
        t_val_filtered,
        risk_predicted_val,
        time_grid,
        t_eval=time_grid,
        km=km,
        competing_risk=competing_risk
    )
    
    return {
        "surv_probs": surv_probs,
        "c_index_train": c_index_train,
        "c_index_val": c_index_val,
        "ibs_val": ibs_val
    }

def summary_output(model, X_train, t_train, e_train, X_val, t_val, e_val, eval_params):
    print("Loading and preprocessing METABRIC dataset...")
    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    print(f"Features: {X_train.shape[1]}")
    print(f"Time range (train): [{t_train.min():.2f}, {t_train.max():.2f}]")
    print(f"Time range (val):   [{t_val.min():.2f}, {t_val.max():.2f}]")
    print(f"Event rate (train): {e_train.mean():.2%}")
    print(f"Event rate (val):   {e_val.mean():.2%}")
    
    print("\nTraining Cox Proportional Hazards model...")

    print("\nModel Summary:")
    print(model.summary[['coef', 'exp(coef)', 'p']])
    
    print("\nEvaluating model...")
    
    print("\n" + "="*50)
    print("RESULTS")
    print("="*50)
    print(f"C-index (train): {eval_params['c_index_train']:.4f}")
    print(f"C-index (val):   {eval_params['c_index_val']:.4f}")
    if not np.isnan(eval_params['ibs_val']):
        print(f"IBS (val):       {eval_params['ibs_val']:.4f}")
    else:
        print(f"IBS (val):       N/A (no valid samples)")
    print("="*50)
    print("\nNote: C-index computed directly from risk scores")
    print("      IBS computed using NeuralFineGray metrics")
    print("      Validation IBS computed on samples with time < max(train time)")
