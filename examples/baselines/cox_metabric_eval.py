import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index
from pycox.datasets import metabric
from sksurv.metrics import integrated_brier_score, concordance_index_censored
from sksurv.util import Surv


def load_and_preprocess_data():
    """Load METABRIC dataset and preprocess."""
    df = metabric.read_df()
    
    # Split features and targets
    X = df.drop(['duration', 'event'], axis=1)
    t = df['duration']
    e = df['event']
    
    # Train/val split
    X_train, X_val, t_train, t_val, e_train, e_val = train_test_split(
        X, t, e, test_size=0.2, random_state=42, stratify=e
    )
    
    # Standardize features
    scaler = StandardScaler()
    X_train = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=X_train.columns,
        index=X_train.index
    )
    X_val = pd.DataFrame(
        scaler.transform(X_val),
        columns=X_val.columns,
        index=X_val.index
    )
    
    return X_train, X_val, t_train, t_val, e_train, e_val


def train_cox_model(X_train, t_train, e_train):
    """Train Cox Proportional Hazards model."""
    # Prepare dataframe for lifelines
    df_train = X_train.copy()
    df_train['duration'] = t_train.values
    df_train['event'] = e_train.values
    
    # Fit Cox model
    cph = CoxPHFitter(penalizer=0.01)
    cph.fit(df_train, duration_col='duration', event_col='event')
    
    return cph


def evaluate_model(cph, X_train, X_val, t_train, t_val, e_train, e_val):
    """Evaluate Cox model using scikit-survival metrics."""
    
    # --- C-index using lifelines (on full validation set) ---
    c_index_train = cph.score(
        pd.concat([X_train, t_train.rename('duration'), e_train.rename('event')], axis=1),
        scoring_method="concordance_index"
    )
    c_index_val = cph.score(
        pd.concat([X_val, t_val.rename('duration'), e_val.rename('event')], axis=1),
        scoring_method="concordance_index"
    )
    
    # --- Integrated Brier Score ---
    # CRITICAL: Filter validation samples to only include times within training range
    # This is because the censoring distribution is estimated from training data
    max_time_train = t_train.max()
    
    # Filter validation set
    valid_mask = t_val < max_time_train
    X_val_filtered = X_val[valid_mask]
    t_val_filtered = t_val[valid_mask]
    e_val_filtered = e_val[valid_mask]
    
    print(f"Validation samples for IBS: {len(t_val_filtered)}/{len(t_val)} " +
          f"({100*len(t_val_filtered)/len(t_val):.1f}%)")
    
    if len(t_val_filtered) == 0:
        print("WARNING: No validation samples within training time range!")
        return {
            "c_index_train": c_index_train,
            "c_index_val": c_index_val,
            "ibs_val": np.nan
        }
    
    # Create structured arrays
    y_train = Surv.from_arrays(event=e_train.values.astype(bool), time=t_train.values)
    y_val_filtered = Surv.from_arrays(event=e_val_filtered.values.astype(bool), 
                                      time=t_val_filtered.values)
    
    # Create time grid within safe range
    max_time_safe = max_time_train * 0.95
    event_times = t_val_filtered[e_val_filtered.astype(bool)].values
    event_times = event_times[event_times < max_time_safe]
    
    if len(event_times) > 50:
        time_grid = np.quantile(event_times, np.linspace(0.1, 0.9, 50))
    else:
        time_grid = np.linspace(t_val_filtered.min() + 1, max_time_safe, 50)
    
    time_grid = np.unique(time_grid)
    
    print(f"Time grid range: [{time_grid.min():.2f}, {time_grid.max():.2f}]")
    print(f"Max time (train): {max_time_train:.2f}")
    
    # Get survival predictions for filtered validation set
    surv_funcs = cph.predict_survival_function(X_val_filtered, times=time_grid)
    surv_probs = surv_funcs.T.values
    
    # Calculate IBS
    ibs = integrated_brier_score(y_train, y_val_filtered, surv_probs, time_grid)
    
    return {
        "c_index_train": c_index_train,
        "c_index_val": c_index_val,
        "ibs_val": ibs
    }


def main():
    print("Loading and preprocessing METABRIC dataset...")
    X_train, X_val, t_train, t_val, e_train, e_val = load_and_preprocess_data()
    
    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    print(f"Features: {X_train.shape[1]}")
    print(f"Time range (train): [{t_train.min():.2f}, {t_train.max():.2f}]")
    print(f"Time range (val):   [{t_val.min():.2f}, {t_val.max():.2f}]")
    print(f"Event rate (train): {e_train.mean():.2%}")
    print(f"Event rate (val):   {e_val.mean():.2%}")
    
    print("\nTraining Cox Proportional Hazards model...")
    cph = train_cox_model(X_train, t_train, e_train)
    
    print("\nModel Summary:")
    print(cph.summary[['coef', 'exp(coef)', 'p']])
    
    print("\nEvaluating model...")
    metrics = evaluate_model(cph, X_train, X_val, t_train, t_val, e_train, e_val)
    
    print("\n" + "="*50)
    print("RESULTS")
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