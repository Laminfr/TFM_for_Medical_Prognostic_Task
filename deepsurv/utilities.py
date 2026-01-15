import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from lifelines import KaplanMeierFitter

# Import the shared data loader
from datasets.data_loader import load_and_preprocess_data


# Import metrics from neuralfg repository
import sys
sys.path.insert(0, '/vol/miltank/users/sajb/Project/NeuralFineGray')
from metrics.calibration import integrated_brier_score
from metrics.discrimination import truncated_concordance_td


class DeepSurvModel(nn.Module):
    """Deep Cox Proportional Hazards Model (DeepSurv)."""
    
    def __init__(self, input_dim, hidden_dims=[64, 32, 16]):
        super(DeepSurvModel, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.3))
            prev_dim = hidden_dim
        
        # Output layer: single risk score
        layers.append(nn.Linear(prev_dim, 1))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)


def cox_loss(risk_scores, times, events):
    """Negative log partial likelihood for Cox model."""
    # Sort by time (descending)
    sorted_indices = torch.argsort(times, descending=True)
    risk_scores = risk_scores[sorted_indices]
    events = events[sorted_indices]
    
    # Compute log risk and cumulative sum
    log_risk = risk_scores - torch.max(risk_scores)  # For numerical stability
    cumsum_risk = torch.logcumsumexp(log_risk, dim=0)
    
    # Only consider events (not censored)
    log_likelihood = (log_risk - cumsum_risk) * events
    
    # Return negative log likelihood
    return -torch.sum(log_likelihood) / torch.sum(events)


def train_deepsurv_model(X_train, t_train, e_train, epochs=100, batch_size=64, lr=0.001):
    """Train DeepSurv model."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Convert to tensors
    X_train_tensor = torch.FloatTensor(X_train.values).to(device)
    t_train_tensor = torch.FloatTensor(t_train.values).to(device)
    e_train_tensor = torch.FloatTensor(e_train.values).to(device)
    
    # Create data loader
    train_dataset = TensorDataset(X_train_tensor, t_train_tensor, e_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # Initialize model
    input_dim = X_train.shape[1]
    model = DeepSurvModel(input_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    
    # Training loop
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch_X, batch_t, batch_e in train_loader:
            optimizer.zero_grad()
            
            risk_scores = model(batch_X).squeeze()
            loss = cox_loss(risk_scores, batch_t, batch_e)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        if (epoch + 1) % 20 == 0:
            avg_loss = total_loss / len(train_loader)
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")
    
    return model, device


def predict_risk_scores(model, X, device):
    """Predict risk scores for given samples."""
    model.eval()
    with torch.no_grad():
        X_tensor = torch.FloatTensor(X.values).to(device)
        risk_scores = model(X_tensor).squeeze().cpu().numpy()
    return risk_scores


def estimate_survival_from_cox(risk_scores_train, risk_scores_test, t_train, e_train, time_grid):
    """
    Estimate survival probabilities using Breslow estimator.
    S(t|x) = S_0(t) ^ exp(risk_score)
    """
    from lifelines import KaplanMeierFitter
    
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
    survival_probs = np.row_stack([
        baseline_surv ** np.exp(risk)
        for risk in risk_scores_clipped
    ])
    
    return survival_probs


def evaluate_deepsurv_model(model, device, X_train, X_val, t_train, t_val, e_train, e_val):
    """Evaluate DeepSurv model using NeuralFineGray metrics for single event survival."""
    
    # Get risk scores
    risk_scores_train = predict_risk_scores(model, X_train, device)
    risk_scores_val = predict_risk_scores(model, X_val, device)
    
    # Create time grid for evaluation
    max_time_train = t_train.max()
    valid_mask = t_val < max_time_train
    
    # Filter validation set to be within the time range of the training data
    t_val_filtered = t_val[valid_mask].values
    e_val_filtered = e_val[valid_mask].values
    
    print(f"Validation samples for evaluation: {len(t_val_filtered)}/{len(t_val)} " +
          f"({100*len(t_val_filtered)/len(t_val):.1f}%)")
    
    if len(t_val_filtered) == 0:
        print("WARNING: No validation samples within training time range!")
        return {
            "c_index_train": np.nan,
            "c_index_val": np.nan,
            "ibs_val": np.nan
        }
    
    # Create safe time grid
    max_time_safe = max_time_train * 0.95
    event_times_val = t_val_filtered[e_val_filtered > 0]
    event_times_val = event_times_val[event_times_val < max_time_safe]
    
    if len(event_times_val) > 50:
        time_grid = np.quantile(event_times_val, np.linspace(0.1, 0.9, 50))
    else:
        time_grid = np.linspace(max(t_val_filtered.min(), 1), max_time_safe, 50)
    
    time_grid = np.unique(time_grid)
    
    print(f"Time grid range: [{time_grid.min():.2f}, {time_grid.max():.2f}]")
    print(f"Max time (train): {max_time_train:.2f}")
    
    # Estimate survival probabilities
    survival_probs_val = estimate_survival_from_cox(
        risk_scores_train, 
        risk_scores_val[valid_mask], 
        t_train.values, 
        e_train.values, 
        time_grid
    )
    
    survival_probs_train = estimate_survival_from_cox(
        risk_scores_train,
        risk_scores_train,
        t_train.values,
        e_train.values,
        time_grid
    )
    
    # Convert survival to cumulative incidence (risk) for the metrics
    # For single event survival: F(t) = 1 - S(t)
    risk_predicted_val = 1 - survival_probs_val
    risk_predicted_train = 1 - survival_probs_train
    
    # Prepare IPCW estimator (Kaplan-Meier on censoring distribution)
    # For your metrics: pass tuple (e_train, t_train)
    km = (e_train.values, t_train.values)
    
    # For single event survival analysis, competing_risk=1 represents THE event
    # e=0 represents censoring, e=1 represents the event of interest
    competing_risk = 1
    
    # --- C-index using truncated concordance ---
    c_index_train, km = truncated_concordance_td(
        e_train.values, 
        t_train.values, 
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
    X_train, X_val, t_train, t_val, e_train, e_val = load_and_preprocess_data(
        dataset=dataset,
        normalize=normalize,
        test_size=test_size,
        random_state=random_state
    )
    
    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    print(f"Features: {X_train.shape[1]}")
    print(f"Time range (train): [{t_train.min():.2f}, {t_train.max():.2f}]")
    print(f"Time range (val):   [{t_val.min():.2f}, {t_val.max():.2f}]")
    print(f"Event rate (train): {(e_train > 0).mean():.2%}")
    print(f"Event rate (val):   {(e_val > 0).mean():.2%}")
    
    print("\nTraining DeepSurv model...")
    model, device = train_deepsurv_model(X_train, t_train, e_train, epochs=100, batch_size=64)
    
    print("\nEvaluating model...")
    metrics = evaluate_deepsurv_model(model, device, X_train, X_val, t_train, t_val, e_train, e_val)
    
    print("\n" + "="*50)
    print("DEEPSURV MODEL RESULTS")
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