"""
Phase 2: Neural Fine-Gray Wrapper for Competing Risks Benchmark

Wraps the existing NFG implementation to provide a consistent interface
for the competing risks benchmark.

Key features:
- Uses pure (X, T, E) data
- Optimizes Fine-Gray loss directly
- Provides CIF predictions for all competing risks
- Extracts embeddings for hybrid model (Phase 3)
"""

import numpy as np
import torch
from typing import Dict, Optional, Tuple, List
import warnings

from nfg.nfg_api import NeuralFineGray
from nfg.nfg_torch import NeuralFineGrayTorch


class NFGCompetingRisks:
    """
    Neural Fine-Gray wrapper for competing risks benchmark.
    
    Provides a unified interface for the NFG model that matches
    the MultiClassSurvivalStacking API for fair comparison.
    
    Parameters
    ----------
    layers : list, default=[100, 100, 100]
        Hidden layer sizes for the embedding network
    layers_surv : list, default=[100]
        Hidden layer sizes for the survival prediction heads
    dropout : float, default=0.0
        Dropout rate
    cuda : bool, default=True
        Whether to use GPU if available (uses cuda=2 for full GPU support)
    cause_specific : bool, default=False
        Whether to use cause-specific loss (vs subdistribution)
    random_state : int, default=42
        Random seed
    """
    
    def __init__(
        self,
        layers: List[int] = [100, 100, 100],
        layers_surv: List[int] = [100],
        dropout: float = 0.0,
        cuda: bool = True,
        cause_specific: bool = False,
        random_state: int = 42,
        **kwargs
    ):
        self.layers = layers
        self.layers_surv = layers_surv
        self.dropout = dropout
        # NFG uses cuda=2 for full GPU (model + data), cuda=1 for model only
        self.cuda = 2 if (cuda and torch.cuda.is_available()) else 0
        self.cause_specific = cause_specific
        self.random_state = random_state
        self.kwargs = kwargs
        
        # Fitted attributes
        self.model_ = None
        self.n_risks_ = None
        self.times_ = None
        self.T_train_ = None
        
    def fit(
        self,
        X: np.ndarray,
        T: np.ndarray,
        E: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        T_val: Optional[np.ndarray] = None,
        E_val: Optional[np.ndarray] = None,
        n_iter: int = 1000,
        learning_rate: float = 1e-3,
        batch_size: int = 256,
        verbose: bool = True
    ) -> 'NFGCompetingRisks':
        """
        Fit the Neural Fine-Gray model.
        
        Parameters
        ----------
        X : np.ndarray
            Training feature matrix
        T : np.ndarray
            Training times
        E : np.ndarray
            Training events (0=censored, 1,2,...=event types)
        X_val, T_val, E_val : optional
            Validation data
        n_iter : int
            Maximum number of training iterations
        learning_rate : float
            Learning rate
        batch_size : int
            Batch size for training
        verbose : bool
            Whether to print progress
            
        Returns
        -------
        self
        """
        X = np.asarray(X).astype(np.float64)
        T = np.asarray(T).astype(np.float64).flatten()
        E = np.asarray(E).astype(np.int64).flatten()
        
        self.n_risks_ = int(E.max())
        self.T_train_ = T.copy()
        
        # Generate evaluation times based on training data
        event_times = T[E > 0]
        if len(event_times) > 0:
            percentiles = np.linspace(5, 95, 20)
            self.times_ = np.percentile(event_times, percentiles)
        else:
            self.times_ = np.linspace(T.min(), T.max(), 20)
        
        if verbose:
            print(f"Fitting NFGCompetingRisks...")
            print(f"  Samples: {len(X)}, Features: {X.shape[1]}, Risks: {self.n_risks_}")
            print(f"  Using CUDA: {self.cuda > 0} (mode={self.cuda})")
        
        # Create NFG model
        self.model_ = NeuralFineGray(
            layers=self.layers,
            layers_surv=self.layers_surv,
            dropout=self.dropout,
            cuda=self.cuda,
            cause_specific=self.cause_specific,
            **self.kwargs
        )
        
        # Prepare validation data
        val_data = None
        if X_val is not None and T_val is not None and E_val is not None:
            val_data = (
                np.asarray(X_val).astype(np.float64),
                np.asarray(T_val).astype(np.float64),
                np.asarray(E_val).astype(np.int64)
            )
        
        # Fit model
        # Note: train_nfg uses n_iter, lr, bs (not iters, learning_rate, batch_size)
        self.model_.fit(
            X, T, E,
            val_data=val_data,
            n_iter=n_iter,
            lr=learning_rate,
            bs=batch_size
        )
        
        if verbose:
            print("  Training complete.")
        
        return self
    
    def predict_cif(
        self,
        X: np.ndarray,
        times: Optional[np.ndarray] = None
    ) -> Dict[int, np.ndarray]:
        """
        Predict Cumulative Incidence Functions for each risk.
        
        Parameters
        ----------
        X : np.ndarray
            Feature matrix
        times : np.ndarray, optional
            Time points for prediction
            
        Returns
        -------
        cif : dict
            CIF for each risk: {risk: array (n_samples, n_times)}
        """
        if self.model_ is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        X = np.asarray(X).astype(np.float64)
        
        if times is None:
            times = self.times_
        else:
            times = np.asarray(times)
        
        cif = {}
        for risk in range(1, self.n_risks_ + 1):
            # predict_survival returns survival probability for that risk
            # CIF = 1 - Survival for that cause
            survival = self._predict_survival_internal(X, times, risk=risk)
            cif[risk] = 1 - survival
        
        return cif
    
    def _predict_survival_internal(
        self, 
        X: np.ndarray, 
        times: np.ndarray, 
        risk: int = 1
    ) -> np.ndarray:
        """
        Internal predict_survival that handles GPU device placement correctly.
        
        The original NFG predict_survival has a bug where _preprocess_test_data
        doesn't move data to GPU when cuda=2, causing device mismatch.
        """
        # Preprocess to tensor
        X_tensor = self.model_._preprocess_test_data(X)
        
        # Move to GPU if needed
        if self.cuda > 0:
            X_tensor = X_tensor.cuda()
        
        times_list = list(times)
        scores = []
        
        for t_ in times_list:
            t_normalized = self.model_._normalise(
                torch.DoubleTensor([t_] * len(X_tensor))
            ).to(X_tensor.device)
            
            with torch.no_grad():
                log_sr, log_beta, _ = self.model_.torch_model(X_tensor, t_normalized)
                beta = 1 if self.model_.cause_specific else log_beta.exp()
                outcomes = 1 - beta * (1 - torch.exp(log_sr))
                scores.append(outcomes[:, int(risk) - 1].unsqueeze(1).cpu().numpy())
        
        return np.concatenate(scores, axis=1)
    
    def predict_survival(self, X: np.ndarray, risk: int = 1) -> np.ndarray:
        """
        Predict survival function for a specific risk.
        
        Parameters
        ----------
        X : np.ndarray
            Feature matrix
        risk : int
            Which competing risk
            
        Returns
        -------
        survival : np.ndarray
            Survival probabilities (n_samples, n_times)
        """
        if self.model_ is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        X = np.asarray(X).astype(np.float64)
        return self._predict_survival_internal(X, self.times_, risk=risk)
    
    def get_risk_scores(
        self,
        X: np.ndarray,
        risk: int = 1,
        time_quantile: float = 0.5
    ) -> np.ndarray:
        """
        Get risk scores for a specific cause at a specific time quantile.
        
        Parameters
        ----------
        X : np.ndarray
            Feature matrix
        risk : int
            Which competing risk
        time_quantile : float
            Which time quantile
            
        Returns
        -------
        scores : np.ndarray
            Risk scores (higher = higher risk)
        """
        cif = self.predict_cif(X)
        
        # Get time index for the quantile
        t_idx = int(time_quantile * (len(self.times_) - 1))
        
        return cif[risk][:, t_idx]
    
    def extract_embeddings(self, X: np.ndarray) -> np.ndarray:
        """
        Extract learned embeddings from the NFG model.
        
        The embeddings are the output of the shared representation layer
        before the cause-specific prediction heads.
        
        Parameters
        ----------
        X : np.ndarray
            Feature matrix
            
        Returns
        -------
        embeddings : np.ndarray
            Learned embeddings (n_samples, embedding_dim)
        """
        if self.model_ is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        X = np.asarray(X).astype(np.float64)
        
        # Get the torch model
        torch_model = self.model_.torch_model
        torch_model.eval()
        
        # Preprocess X like the model does
        X_tensor = self.model_._preprocess_test_data(X)
        
        # Move to GPU if model is on GPU
        if self.cuda > 0:
            X_tensor = X_tensor.cuda()
        
        with torch.no_grad():
            # Pass through embedding layer only
            embeddings = torch_model.embed(X_tensor)
        
        return embeddings.cpu().numpy()
    
    def get_times(self) -> np.ndarray:
        """Return the time points used for predictions."""
        if self.times_ is None:
            raise ValueError("Model not fitted.")
        return self.times_
    
    def get_torch_model(self) -> NeuralFineGrayTorch:
        """Return the underlying PyTorch model for advanced usage."""
        if self.model_ is None:
            raise ValueError("Model not fitted.")
        return self.model_.torch_model


if __name__ == '__main__':
    # Test the wrapper
    from datasets.datasets import load_dataset
    from .utils import split_data
    
    print("Testing NFGCompetingRisks...")
    
    # Load data using existing datasets module
    X, T, E, features = load_dataset('SYNTHETIC_COMPETING')
    n_risks = int(E.max())
    splits = split_data(X, T, E)
    
    X_train, T_train, E_train = splits['train']
    X_val, T_val, E_val = splits['val']
    X_test, T_test, E_test = splits['test']
    
    # Fit model
    model = NFGCompetingRisks(
        layers=[50, 50],
        layers_surv=[50],
        cuda=torch.cuda.is_available()
    )
    model.fit(
        X_train, T_train, E_train,
        X_val, T_val, E_val,
        n_iter=100,  # Short for testing
        verbose=True
    )
    
    # Predict CIF
    cif = model.predict_cif(X_test)
    print(f"\nCIF shapes: {[f'Risk {k}: {v.shape}' for k, v in cif.items()]}")
    
    # Extract embeddings
    embeddings = model.extract_embeddings(X_test)
    print(f"Embeddings shape: {embeddings.shape}")
    
    # Get risk scores
    scores = model.get_risk_scores(X_test, risk=1)
    print(f"Risk scores range: [{scores.min():.3f}, {scores.max():.3f}]")
