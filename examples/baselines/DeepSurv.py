import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sksurv.metrics import concordance_index_censored, integrated_brier_score
from sksurv.util import Surv

# Import the shared data loader
from .data_loader import load_and_preprocess_data


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


def estimate_survival_from_cox(risk_scores_train, risk_scores_test, y_train, time_grid):
    """
    Estimate survival probabilities using Breslow estimator.
    S(t|x) = S_0(t) ^ exp(risk_score)
    """
    from sksurv.linear_model import CoxPHSurvivalAnalysis
    from sksurv.nonparametric import kaplan_meier_estimator
    
    # Estimate baseline survival using Kaplan-Meier on training data
    km_times, km_surv_probs = kaplan_meier_estimator(
        y_train["event"].astype(bool), 
        y_train["time"]
    )
    
    # Interpolate baseline survival at time grid points
    baseline_surv = np.interp(time_grid, km_times, km_surv_probs)
    
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
    """Evaluate DeepSurv model."""
    
    # Get risk scores
    risk_scores_train = predict_risk_scores(model, X_train, device)
    risk_scores_val = predict_risk_scores(model, X_val, device)
    
    # --- C-index using scikit-survival ---
    c_index_train = concordance_index_censored(
        e_train.values.astype(bool), t_train.values, risk_scores_train
    )[0]
    c_index_val = concordance_index_censored(
        e_val.values.astype(bool), t_val.values, risk_scores_val
    )[0]

    # --- Integrated Brier Score ---
    # Filter validation set to be within the time range of the training data
    max_time_train = t_train.max()
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
    
    # Apply the same mask to the validation risk scores
    risk_scores_val_filtered = risk_scores_val[valid_mask]
    
    # Create safe time grid within the range of filtered validation data
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
    
    # Create structured arrays for IBS
    y_train = Surv.from_arrays(event=e_train.values.astype(bool), time=t_train.values)
    y_val_filtered = Surv.from_arrays(event=e_val_filtered.values.astype(bool), 
                                      time=t_val_filtered.values)
    
    # Estimate survival probabilities
    survival_probs = estimate_survival_from_cox(
        risk_scores_train, risk_scores_val_filtered, y_train, time_grid
    )
    
    # Calculate IBS
    ibs = integrated_brier_score(y_train, y_val_filtered, survival_probs, time_grid)
    
    return {"c_index_train": c_index_train, "c_index_val": c_index_val, "ibs_val": ibs}


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
    print(f"Event rate (train): {e_train.mean():.2%}")
    print(f"Event rate (val):   {e_val.mean():.2%}")
    
    print("\nTraining DeepSurv model...")
    model, device = train_deepsurv_model(X_train, t_train, e_train, epochs=100, batch_size=64)
    
    print("\nEvaluating model...")
    metrics = evaluate_deepsurv_model(model, device, X_train, X_val, t_train, t_val, e_train, e_val)
    
    print("\n" + "="*50)
    print("DEEPSURV MODEL RESULTS")
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
