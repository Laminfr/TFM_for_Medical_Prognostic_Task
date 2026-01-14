"""
Phase 3: Hybrid NFG + Stacking Model

Combines the learned representations from Neural Fine-Gray (Phase 2)
with the multi-class XGBoost classifier (Phase 1).

Key idea:
- Train NFG model to learn deep feature representations
- Extract embeddings from the trained NFG model
- Use embeddings (with optional raw features) as input to multi-class XGBoost
- This tests whether deep features help tree-based classifiers
"""

import numpy as np
from typing import Dict, Optional, Tuple, Any, List
from sklearn.base import BaseEstimator
import warnings

from .nfg_wrapper import NFGCompetingRisks
from .stacking_multiclass import MultiClassSurvivalStacking


class HybridNFGStacking(BaseEstimator):
    """
    Hybrid model combining NFG embeddings with Multi-Class Survival Stacking.
    
    This two-stage approach:
    1. Trains NFG to learn patient representations
    2. Extracts embeddings and trains XGBoost on these features
    
    Parameters
    ----------
    nfg_layers : list, default=[100, 100, 100]
        Hidden layers for NFG embedding network
    nfg_layers_surv : list, default=[100]
        Hidden layers for NFG survival heads
    nfg_dropout : float, default=0.0
        Dropout for NFG
    nfg_n_iter : int, default=1000
        Training iterations for NFG
    n_intervals : int, default=20
        Number of time intervals for stacking
    use_raw_features : bool, default=True
        Whether to concatenate raw features with embeddings
    xgb_params : dict, optional
        XGBoost parameters
    random_state : int, default=42
        Random seed
    cuda : bool, default=True
        Whether to use GPU for NFG
    """
    
    def __init__(
        self,
        nfg_layers: List[int] = [100, 100, 100],
        nfg_layers_surv: List[int] = [100],
        nfg_dropout: float = 0.0,
        nfg_n_iter: int = 1000,
        n_intervals: int = 20,
        use_raw_features: bool = True,
        xgb_params: Optional[Dict[str, Any]] = None,
        random_state: int = 42,
        cuda: bool = True
    ):
        self.nfg_layers = nfg_layers
        self.nfg_layers_surv = nfg_layers_surv
        self.nfg_dropout = nfg_dropout
        self.nfg_n_iter = nfg_n_iter
        self.n_intervals = n_intervals
        self.use_raw_features = use_raw_features
        self.xgb_params = xgb_params or {}
        self.random_state = random_state
        self.cuda = cuda
        
        # Fitted components
        self.nfg_model_ = None
        self.stacking_model_ = None
        self.n_risks_ = None
        self.times_ = None
        self.embedding_dim_ = None
        
    def fit(
        self,
        X: np.ndarray,
        T: np.ndarray,
        E: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        T_val: Optional[np.ndarray] = None,
        E_val: Optional[np.ndarray] = None,
        verbose: bool = True
    ) -> 'HybridNFGStacking':
        """
        Fit the hybrid model in two stages.
        
        Stage 1: Train NFG model
        Stage 2: Extract embeddings and train XGBoost
        
        Parameters
        ----------
        X : np.ndarray
            Training feature matrix
        T : np.ndarray
            Training times
        E : np.ndarray
            Training events
        X_val, T_val, E_val : optional
            Validation data
        verbose : bool
            Whether to print progress
            
        Returns
        -------
        self
        """
        X = np.asarray(X)
        T = np.asarray(T).flatten()
        E = np.asarray(E).flatten()
        
        self.n_risks_ = int(E.max())
        
        if verbose:
            print("=" * 60)
            print("STAGE 1: Training Neural Fine-Gray for embeddings...")
            print("=" * 60)
        
        # Stage 1: Train NFG
        self.nfg_model_ = NFGCompetingRisks(
            layers=self.nfg_layers,
            layers_surv=self.nfg_layers_surv,
            dropout=self.nfg_dropout,
            cuda=self.cuda,
            random_state=self.random_state
        )
        
        self.nfg_model_.fit(
            X, T, E,
            X_val, T_val, E_val,
            n_iter=self.nfg_n_iter,
            verbose=verbose
        )
        
        # Extract embeddings
        if verbose:
            print("\nExtracting embeddings...")
        
        embeddings_train = self.nfg_model_.extract_embeddings(X)
        self.embedding_dim_ = embeddings_train.shape[1]
        
        if verbose:
            print(f"  Embedding dimension: {self.embedding_dim_}")
        
        # Prepare features for stacking
        if self.use_raw_features:
            X_stacking_train = np.concatenate([embeddings_train, X], axis=1)
            if verbose:
                print(f"  Using embeddings + raw features: {X_stacking_train.shape[1]} features")
        else:
            X_stacking_train = embeddings_train
            if verbose:
                print(f"  Using embeddings only: {X_stacking_train.shape[1]} features")
        
        # Prepare validation embeddings
        X_stacking_val = None
        if X_val is not None:
            embeddings_val = self.nfg_model_.extract_embeddings(X_val)
            if self.use_raw_features:
                X_stacking_val = np.concatenate([embeddings_val, X_val], axis=1)
            else:
                X_stacking_val = embeddings_val
        
        if verbose:
            print("\n" + "=" * 60)
            print("STAGE 2: Training Multi-Class XGBoost on embeddings...")
            print("=" * 60)
        
        # Stage 2: Train XGBoost on embeddings
        self.stacking_model_ = MultiClassSurvivalStacking(
            n_intervals=self.n_intervals,
            classifier_params=self.xgb_params,
            random_state=self.random_state
        )
        
        self.stacking_model_.fit(
            X_stacking_train, T, E,
            X_stacking_val, T_val, E_val,
            verbose=verbose
        )
        
        self.times_ = self.stacking_model_.times_
        
        if verbose:
            print("\n" + "=" * 60)
            print("Hybrid model training complete!")
            print("=" * 60)
        
        return self
    
    def _prepare_features(self, X: np.ndarray) -> np.ndarray:
        """Prepare features by extracting embeddings and optionally adding raw features."""
        embeddings = self.nfg_model_.extract_embeddings(X)
        
        if self.use_raw_features:
            return np.concatenate([embeddings, X], axis=1)
        return embeddings
    
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
        if self.stacking_model_ is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        X = np.asarray(X)
        X_embedded = self._prepare_features(X)
        
        return self.stacking_model_.predict_cif(X_embedded, times)
    
    def predict_survival(self, X: np.ndarray) -> np.ndarray:
        """
        Predict overall survival function.
        
        Parameters
        ----------
        X : np.ndarray
            Feature matrix
            
        Returns
        -------
        survival : np.ndarray
            Survival probabilities
        """
        X = np.asarray(X)
        X_embedded = self._prepare_features(X)
        
        return self.stacking_model_.predict_survival(X_embedded)
    
    def get_risk_scores(
        self,
        X: np.ndarray,
        risk: int = 1,
        time_quantile: float = 0.5
    ) -> np.ndarray:
        """
        Get risk scores for a specific cause.
        
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
            Risk scores
        """
        cif = self.predict_cif(X)
        t_idx = int(time_quantile * (len(self.times_) - 1))
        return cif[risk][:, t_idx]
    
    def get_times(self) -> np.ndarray:
        """Return the time points used for predictions."""
        if self.times_ is None:
            raise ValueError("Model not fitted.")
        return self.times_
    
    def get_nfg_model(self) -> NFGCompetingRisks:
        """Return the underlying NFG model."""
        if self.nfg_model_ is None:
            raise ValueError("Model not fitted.")
        return self.nfg_model_
    
    def get_stacking_model(self) -> MultiClassSurvivalStacking:
        """Return the underlying stacking model."""
        if self.stacking_model_ is None:
            raise ValueError("Model not fitted.")
        return self.stacking_model_


if __name__ == '__main__':
    # Test the hybrid model
    import torch
    from datasets.datasets import load_dataset
    from .utils import split_data
    
    print("Testing HybridNFGStacking...")
    
    # Load data using existing datasets module
    X, T, E, features = load_dataset('SYNTHETIC_COMPETING')
    n_risks = int(E.max())
    splits = split_data(X, T, E)
    
    X_train, T_train, E_train = splits['train']
    X_val, T_val, E_val = splits['val']
    X_test, T_test, E_test = splits['test']
    
    # Fit hybrid model
    model = HybridNFGStacking(
        nfg_layers=[50, 50],
        nfg_layers_surv=[50],
        nfg_n_iter=100,  # Short for testing
        n_intervals=15,
        use_raw_features=True,
        cuda=torch.cuda.is_available()
    )
    
    model.fit(X_train, T_train, E_train, X_val, T_val, E_val, verbose=True)
    
    # Predict
    cif = model.predict_cif(X_test)
    print(f"\nCIF shapes: {[f'Risk {k}: {v.shape}' for k, v in cif.items()]}")
    
    # Get risk scores
    scores = model.get_risk_scores(X_test, risk=1)
    print(f"Risk scores range: [{scores.min():.3f}, {scores.max():.3f}]")
