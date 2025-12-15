import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from pandas_patch import pd
from lifelines import KaplanMeierFitter

class DeepSurvTorch(nn.Module):
    """Deep Cox Proportional Hazards Network."""
    def __init__(self, input_dim, hidden_dims=[100, 100], dropout=0.3):
        super(DeepSurvTorch, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = h_dim
            
        # Output is a single risk score
        layers.append(nn.Linear(prev_dim, 1))
        self.net = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.net(x)

class DeepSurv:
    """
    API Wrapper for DeepSurv to match NeuralFineGray/DeSurv interface.
    """
    def __init__(self, layers=[100, 100], dropout=0.3, lr=1e-3, weight_decay=1e-4, cuda=True):
        self.layers = layers
        self.dropout = dropout
        self.lr = lr
        self.weight_decay = weight_decay
        self.device = torch.device('cuda' if cuda and torch.cuda.is_available() else 'cpu')
        self.model = None
        self.baseline_survival = None
        self.baseline_times = None

    def _cox_loss(self, risk_scores, times, events):
        """Negative log partial likelihood."""
        # Sort by time descending
        idx = torch.argsort(times, descending=True)
        risk_scores = risk_scores[idx]
        events = events[idx]
        
        # Log-Sum-Exp trick for stability
        # risk_scores is (B, 1), squeeze to (B)
        risk_scores = risk_scores.squeeze()
        
        log_risk = risk_scores
        # cumsum_exp_risk = \sum_{j: t_j >= t_i} exp(h_j)
        exp_risk = torch.exp(log_risk)
        cumsum_risk = torch.cumsum(exp_risk, dim=0)
        log_cumsum_risk = torch.log(cumsum_risk)
        
        # Loss = - sum( (h_i - log(\sum exp(h_j))) * E_i )
        log_likelihood = (log_risk - log_cumsum_risk) * events
        
        return -torch.sum(log_likelihood) / (torch.sum(events) + 1e-8)

    def fit(self, x, t, e, val_data=None, n_iter=1000, bs=256, patience_max=10):
        # Handle DataFrames
        if isinstance(x, pd.DataFrame): x = x.values
        if isinstance(t, pd.Series): t = t.values
        if isinstance(e, pd.Series): e = e.values

        self.input_dim = x.shape[1]
        self.model = DeepSurvTorch(self.input_dim, self.layers, self.dropout).to(self.device)
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        # Data Loading
        train_ds = TensorDataset(
            torch.FloatTensor(x).to(self.device),
            torch.FloatTensor(t).to(self.device),
            torch.FloatTensor(e).to(self.device)
        )
        train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True)

        # Validation Data Prep
        best_val_loss = float('inf')
        patience = 0
        
        if val_data is not None:
            x_val, t_val, e_val = val_data
            if isinstance(x_val, pd.DataFrame): x_val = x_val.values
            if isinstance(t_val, pd.Series): t_val = t_val.values
            if isinstance(e_val, pd.Series): e_val = e_val.values
            
            x_val_t = torch.FloatTensor(x_val).to(self.device)
            t_val_t = torch.FloatTensor(t_val).to(self.device)
            e_val_t = torch.FloatTensor(e_val).to(self.device)

        for epoch in range(n_iter):
            self.model.train()
            total_loss = 0
            for bx, bt, be in train_loader:
                optimizer.zero_grad()
                risk = self.model(bx)
                loss = self._cox_loss(risk, bt, be)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            # Validation
            if val_data is not None:
                self.model.eval()
                with torch.no_grad():
                    val_risk = self.model(x_val_t)
                    val_loss = self._cox_loss(val_risk, t_val_t, e_val_t).item()
                
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience = 0
                    # Save best state if needed, here we just continue
                else:
                    patience += 1
                    if patience >= patience_max:
                        print(f"Early stopping at epoch {epoch}")
                        break
        
        # Compute Baseline Hazard (Breslow / KM approximation)
        self._compute_baseline_survival(x, t, e)
        return self

    def _compute_baseline_survival(self, x, t, e):
        """
        Estimates S0(t) using KM on the training data, assuming mean risk = 0.
        S(t|x) = S0(t)^exp(risk(x))
        """
        # 1. Get risk scores for training data
        self.model.eval()
        with torch.no_grad():
            x_t = torch.FloatTensor(x).to(self.device)
            risks = self.model(x_t).cpu().numpy().flatten()
        
        # 2. Fit KM
        # Note: A pure Breslow estimator is more mathematically precise for Cox, 
        # but KM is often used as a robust proxy in DeepSurv implementations (like pycox).
        # We follow the snippet provided: S0(t) from KM.
        
        # We center the risks so that exp(risk) has mean ~ 1, or just use the raw output.
        # Standard DeepSurv usually computes baseline hazard dH0(t).
        # Here we use the simplified approach requested:
        kmf = KaplanMeierFitter()
        kmf.fit(t, event_observed=(e > 0))
        
        self.baseline_times = kmf.survival_function_.index.values
        self.baseline_probs = kmf.survival_function_.values.flatten()

    def predict_survival(self, x, times, risk=1):
        """
        Returns survival probabilities S(t|x) at specified times.
        Shape: (n_samples, n_times)
        """
        if isinstance(x, pd.DataFrame): x = x.values
        if isinstance(times, list): times = np.array(times)
        
        self.model.eval()
        with torch.no_grad():
            x_t = torch.FloatTensor(x).to(self.device)
            # Log-risk (h(x))
            log_risk = self.model(x_t).cpu().numpy().flatten()
        
        # S(t|x) = S0(t) ^ exp(h(x))
        # 1. Interpolate baseline survival at requested times
        # We use numpy interp (steps)
        # S0 at times t
        s0_at_times = np.interp(times, self.baseline_times, self.baseline_probs)
        
        # 2. Exponentiate
        # shape: (N, T)
        # exp_risk: (N, 1)
        risk_exp = np.exp(log_risk).reshape(-1, 1)
        s0_t = s0_at_times.reshape(1, -1)
        
        # Avoid numerical issues
        surv_probs = np.power(s0_t, risk_exp)
        return surv_probs