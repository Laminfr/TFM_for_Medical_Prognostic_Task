import numpy as np
import pandas as pd

import xgboost as xgb
from sksurv.util import Surv

from lifelines import KaplanMeierFitter

# Import the shared data loader
try:
    from datasets.data_loader import load_and_preprocess_data
except ImportError:
    from datasets.data_loader import load_and_preprocess_data


from metrics.calibration import integrated_brier_score
from metrics.discrimination import truncated_concordance_td


def wrap_np_to_pandas(X, index=None, prefix="x"):
    if isinstance(X, (pd.DataFrame, pd.Series)):
        return X

    X = np.asarray(X)

    if X.ndim == 1:
        return pd.Series(X, index=index)

    if X.ndim == 2:
        n_cols = X.shape[1]
        cols = [f"{prefix}{i}" for i in range(n_cols)]
        return pd.DataFrame(X, columns=cols, index=index)

    raise ValueError("Input must be 1D or 2D numpy array or pandas object.")


def train_xgboost_model(
    X_train,
    t_train,
    e_train,
    t_val,
    e_val,
    objective="survival:cox",
    eval_metric="cox-nloglik",
    n_estimators=200,
    learning_rate=0.05,
    max_depth=4,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
):
    """Train XGBoost Survival model."""
    # The Cox objective expects numeric time as the label and the event
    # indicator as the sample weight.
    X_train = wrap_np_to_pandas(X_train)
    t_train = wrap_np_to_pandas(t_train, prefix="t")
    e_train = wrap_np_to_pandas(e_train, prefix="e")
    t_val = wrap_np_to_pandas(t_val, prefix="t")
    e_val = wrap_np_to_pandas(e_val, prefix="e")

    y_train = Surv.from_arrays(event=e_train.values, time=t_train.values)

    y_time = y_train["time"]
    y_event = y_train["event"].astype(int)

    # Suggested hyperparameters are a good starting point
    model = xgb.train(
        objective=objective,
        eval_metric=eval_metric,
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        random_state=random_state,
    )
    model.fit(X_train, y_time, sample_weight=y_event)
    return model


def estimate_survival_from_cox(
    risk_scores_test,
    t_train,
    e_train,
    time_grid,
):
    """
    Estimate survival probabilities using Breslow estimator.
    S(t|x) = S_0(t) ^ exp(risk_score)
    """
    # Estimate baseline survival using Kaplan-Meier on training data
    kmf = KaplanMeierFitter()
    # For single event survival: event_observed=True when e_train > 0
    kmf.fit(t_train, event_observed=(e_train > 0))

    # Get baseline survival at time grid points
    baseline_surv = kmf.survival_function_at_times(time_grid).values

    # Clip risk scores to prevent overflow
    risk_scores_clipped = np.clip(risk_scores_test, -10, 10)

    # Calculate survival probabilities for each sample
    # S(t|x) = S_0(t) ^ exp(risk_score)
    survival_probs = np.vstack(
        [baseline_surv ** np.exp(risk) for risk in risk_scores_clipped]
    )

    return survival_probs


def concordance_index_from_risk_scores(e, t, risk_scores, tied_tol=1e-8):
    """
    Compute C-index directly from risk scores (for Cox-like models).
    Higher risk score should correspond to higher risk (shorter survival).

    This is a simpler version that works with scalar risk scores.
    """
    event = e.astype(bool)
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
        # So we want risk_scores[at_risk] < risk_scores[i] for concordance
        concordant += (risk_scores[at_risk] < risk_scores[i]).sum()
        concordant += (
            0.5 * (np.abs(risk_scores[at_risk] - risk_scores[i]) <= tied_tol).sum()
        )
        permissible += at_risk.sum()

    if permissible == 0:
        return np.nan

    return concordant / permissible


def evaluate_xgboost_model(model, X_train, t_train, e_train, X_val, t_val, e_val):
    """Evaluate XGBoost model using NeuralFineGray metrics."""

    X_train = wrap_np_to_pandas(X_train)
    t_train = wrap_np_to_pandas(t_train, prefix="t")
    e_train = wrap_np_to_pandas(e_train, prefix="e")
    X_val = wrap_np_to_pandas(X_val)
    t_val = wrap_np_to_pandas(t_val, prefix="t")
    e_val = wrap_np_to_pandas(e_val, prefix="e")

    y_train = Surv.from_arrays(event=e_train.values, time=t_train.values)
    y_val = Surv.from_arrays(event=e_val.values, time=t_val.values)

    # Convert to numpy if needed
    if isinstance(X_train, pd.DataFrame):
        X_train = X_train.values
    if isinstance(t_train, pd.Series):
        t_train = t_train.values
    if isinstance(e_train, pd.Series):
        e_train = e_train.values

    y_xgb_train = np.where(e_train > 0, t_train, -t_train)
    dtrain = xgb.DMatrix(X_train, label=y_xgb_train)
    y_xgb_val = np.where(e_val > 0, t_val, -t_val)
    dval = xgb.DMatrix(X_val, label=y_xgb_val)

    # Get risk scores from XGBoost (these are log-hazard ratios)
    risk_scores_train = model.predict(dtrain)
    risk_scores_val = model.predict(dval)

    # Extract time and event arrays from structured arrays
    t_train = y_train["time"]
    e_train = y_train["event"].astype(int)
    t_val = y_val["time"]
    e_val = y_val["event"].astype(int)

    # --- C-index using risk scores directly ---
    # For Cox models, risk scores can be used directly for concordance
    c_index_train = concordance_index_from_risk_scores(
        e_train, t_train, risk_scores_train
    )
    c_index_val = concordance_index_from_risk_scores(e_val, t_val, risk_scores_val)

    print(f"C-index (train): {c_index_train:.4f}")
    print(f"C-index (val):   {c_index_val:.4f}")

    # --- Integrated Brier Score ---
    # Filter validation set to be within the time range of the training data
    max_time_train = t_train.max()
    valid_mask = t_val < max_time_train
    t_val_filtered = t_val[valid_mask]
    e_val_filtered = e_val[valid_mask]
    X_val_filtered = X_val[valid_mask]

    print(
        f"Validation samples for IBS: {len(t_val_filtered)}/{len(t_val)} "
        + f"({100 * len(t_val_filtered) / len(t_val):.1f}%)"
    )

    if len(t_val_filtered) == 0:
        print("WARNING: No validation samples within training time range!")
        return {
            "c_index_train": c_index_train,
            "c_index_val": c_index_val,
            "ibs_val": np.nan,
        }

    # Apply the same mask to the validation risk scores
    risk_scores_val_filtered = risk_scores_val[valid_mask]

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

    # Estimate survival probabilities for IBS
    survival_probs_val = estimate_survival_from_cox(
        risk_scores_val_filtered, t_train, e_train, time_grid
    )

    # Convert survival to cumulative incidence (risk) for the IBS metric
    risk_predicted_val = 1 - survival_probs_val

    # Prepare IPCW estimator (Kaplan-Meier on censoring distribution)
    km = (e_train, t_train)

    # For single event survival analysis, competing_risk=1 represents THE event
    competing_risk = 1

    # --- Integrated Brier Score ---
    ibs_val, km = integrated_brier_score(
        e_val_filtered,
        t_val_filtered,
        risk_predicted_val,
        time_grid,
        t_eval=time_grid,
        km=km,
        competing_risk=competing_risk,
    )

    return {
        "c_index_predict": None,
        "surv_probs_predict": None,
        "surv_probs_val": survival_probs_val,
        "c_index_train": c_index_train,
        "c_index_val": c_index_val,
        "ibs_val": ibs_val,
    }


def summary_output(X_train, t_train, e_train, X_val, t_val, e_val, metrics):
    X_train = wrap_np_to_pandas(X_train)
    t_train = wrap_np_to_pandas(t_train, prefix="t")
    e_train = wrap_np_to_pandas(e_train, prefix="e")
    X_val = wrap_np_to_pandas(X_val)
    t_val = wrap_np_to_pandas(t_val, prefix="t")
    e_val = wrap_np_to_pandas(e_val, prefix="e")

    y_train = Surv.from_arrays(event=e_train.values, time=t_train.values)
    y_val = Surv.from_arrays(event=e_val.values, time=t_val.values)

    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    print(f"Features: {X_train.shape[1]}")
    print(
        f"Time range (train): [{y_train['time'].min():.2f}, {y_train['time'].max():.2f}]"
    )
    print(f"Time range (val):   [{y_val['time'].min():.2f}, {y_val['time'].max():.2f}]")
    print(f"Event rate (train): {y_train['event'].mean():.2%}")
    print(f"Event rate (val):   {y_val['event'].mean():.2%}")

    print("\n" + "=" * 50)
    print("XGBOOST MODEL RESULTS")
    print("=" * 50)
    if not np.isnan(metrics["c_index_train"]):
        print(f"C-index (train): {metrics['c_index_train']:.4f}")
    else:
        print(f"C-index (train): N/A")

    if not np.isnan(metrics["c_index_val"]):
        print(f"C-index (val):   {metrics['c_index_val']:.4f}")
    else:
        print(f"C-index (val):   N/A")

    if not np.isnan(metrics["ibs_val"]):
        print(f"IBS (val):       {metrics['ibs_val']:.4f}")
    else:
        print(f"IBS (val):       N/A")
    print("=" * 50)
    print("\nNote: C-index computed directly from risk scores")
    print("      IBS computed using NeuralFineGray metrics with survival curves")
    print("      Validation IBS computed on samples with time < max(train time)")
