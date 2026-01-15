"""
TabICL Embedding Extractor for Survival Analysis
Updated version with improved deep encoder embedding extraction.
Fixes shape mismatch by correctly handling (1, N, D) output tensors.

Features:
- One-hot encoding for categorical features in deep+raw mode
- Optional PCA compression for tree-based models
- Proper handling of string columns in raw data
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional, Union, List
import warnings
import torch
from sklearn.preprocessing import OneHotEncoder
from sklearn.decomposition import PCA

try:
    from tabicl import TabICLClassifier
    TABICL_AVAILABLE = True
except ImportError:
    TABICL_AVAILABLE = False
    warnings.warn("tabicl package not found. Install with: pip install tabicl")


def _extract_row_embeddings(clf, X_df: pd.DataFrame, split_name: str, device: str = 'cpu') -> Optional[np.ndarray]:
    """
    Capture deep row embeddings by hooking TabICL's row_interactor module.
    Correctly handles TabICL's batching behavior (1, N_rows, Dim).
    """
    try:
        if not hasattr(clf, 'model_'):
            if hasattr(clf, '_load_model'):
                clf._load_model()
            else:
                return None

        model = clf.model_
        if not hasattr(model, 'row_interactor'):
            return None

        model.eval()
        captured_batches: List[torch.Tensor] = []

        def capture_hook(module, inputs, outputs):
            # Output is typically the row embeddings H
            tensor = outputs[0] if isinstance(outputs, (tuple, list)) else outputs
            if not isinstance(tensor, torch.Tensor):
                return
            tensor = tensor.detach().cpu()

            # Fix: TabICL row_interactor often returns (1, N_rows, Dim)
            # We need to squeeze the batch dimension if it's 1 and Dim 2 is the embedding dim.
            if tensor.dim() == 3 and tensor.shape[0] == 1:
                # Shape (1, N, D) -> Squeeze to (N, D)
                tensor = tensor.squeeze(0)
            
            # If shape is already (N, D), keep it. 
            # If shape is (Batch, Seq, Dim) with Batch > 1, we append as is and concat later.
            
            if tensor.dim() == 2:
                captured_batches.append(tensor)
            elif tensor.dim() == 3:
                # Unexpected 3D shape with Batch > 1. 
                # This might happen if TabICL batches the data.
                # Assuming (Batch, N_rows_in_batch, Dim) -> Flatten to (Batch*N, Dim)
                # But usually row_interactor dim 1 is sequence length.
                # If we are here, we just append and hope torch.cat handles it or we process later.
                captured_batches.append(tensor)

        # Register hook
        hook_handle = model.row_interactor.register_forward_hook(capture_hook)

        try:
            with torch.no_grad():
                _ = clf.predict_proba(X_df)
        except Exception:
            return None
        finally:
            hook_handle.remove()

        if not captured_batches:
            return None

        # Concatenate all captured batches
        # Note: If predict_proba loops, we might have multiple tensors.
        embeddings = torch.cat(captured_batches, dim=0)
        
        # If we still have 3D tensor after concat (e.g. from the weird fallback), flatten it?
        # But our hook logic tries to ensure 2D. 
        if embeddings.dim() == 3 and embeddings.shape[0] == 1:
             embeddings = embeddings.squeeze(0)

        expected_rows = len(X_df)

        # Handle Context + Query case
        # If TabICL includes context rows in the forward pass, we might get N_context + N_query rows.
        # We only want the last expected_rows (the queries).
        if embeddings.shape[0] > expected_rows:
            embeddings = embeddings[-expected_rows:]
        
        if embeddings.shape[0] != expected_rows:
            return None

        return embeddings.numpy()

    except Exception:
        return None


def _encode_categorical_features(X_train: np.ndarray, X_val: np.ndarray, X_test: np.ndarray, 
                                  feature_names: list, verbose: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray, list]:
    """
    Detect and one-hot encode categorical (string) features in raw data.
    Returns encoded arrays and updated feature names.
    """
    # Convert to DataFrame to detect dtypes
    df_train = pd.DataFrame(X_train, columns=feature_names)
    df_val = pd.DataFrame(X_val, columns=feature_names)
    df_test = pd.DataFrame(X_test, columns=feature_names)
    
    # Find string/object columns
    cat_cols = []
    num_cols = []
    for col in df_train.columns:
        if df_train[col].dtype == 'object' or df_train[col].dtype.name == 'category':
            cat_cols.append(col)
        else:
            # Try to convert to numeric, if it fails it's categorical
            try:
                pd.to_numeric(df_train[col].iloc[0])
                num_cols.append(col)
            except (ValueError, TypeError):
                cat_cols.append(col)
    
    if not cat_cols:
        # No categorical columns, return as-is
        return X_train.astype(float), X_val.astype(float), X_test.astype(float), feature_names
    
    if verbose:
        print(f"  → One-hot encoding {len(cat_cols)} categorical columns: {cat_cols}")
    
    # Fit OneHotEncoder on training data
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    X_train_cat = encoder.fit_transform(df_train[cat_cols])
    X_val_cat = encoder.transform(df_val[cat_cols])
    X_test_cat = encoder.transform(df_test[cat_cols])
    
    # Get encoded feature names
    encoded_names = encoder.get_feature_names_out(cat_cols).tolist()
    
    # Extract numeric columns
    X_train_num = df_train[num_cols].values.astype(float)
    X_val_num = df_val[num_cols].values.astype(float)
    X_test_num = df_test[num_cols].values.astype(float)
    
    # Combine: numeric + encoded categorical
    X_train_enc = np.column_stack([X_train_num, X_train_cat]) if len(num_cols) > 0 else X_train_cat
    X_val_enc = np.column_stack([X_val_num, X_val_cat]) if len(num_cols) > 0 else X_val_cat
    X_test_enc = np.column_stack([X_test_num, X_test_cat]) if len(num_cols) > 0 else X_test_cat
    
    new_feature_names = num_cols + encoded_names
    
    return X_train_enc, X_val_enc, X_test_enc, new_feature_names


def _apply_pca_compression(X_train_emb: np.ndarray, X_val_emb: np.ndarray, X_test_emb: np.ndarray,
                           n_components: int = 32, verbose: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray, PCA]:
    """
    Apply PCA to compress high-dimensional embeddings.
    Fits on training data only to prevent leakage.
    
    Args:
        n_components: Target dimensions (default 32 for tree models)
        
    Returns:
        Compressed embeddings and fitted PCA object
    """
    pca = PCA(n_components=min(n_components, X_train_emb.shape[1], X_train_emb.shape[0]))
    
    X_train_pca = pca.fit_transform(X_train_emb)
    X_val_pca = pca.transform(X_val_emb)
    X_test_pca = pca.transform(X_test_emb)
    
    if verbose:
        explained_var = pca.explained_variance_ratio_.sum() * 100
        print(f"  → PCA: {X_train_emb.shape[1]}D → {X_train_pca.shape[1]}D ({explained_var:.1f}% variance)")
    
    return X_train_pca, X_val_pca, X_test_pca, pca


def apply_tabicl_embedding(
    X_train: np.ndarray,
    X_val: np.ndarray, 
    X_test: np.ndarray,
    E_train: np.ndarray,
    feature_names: list,
    verbose: bool = True,
    use_deep_embeddings: bool = True,
    concat_with_raw: bool = True,
    pca_for_trees: bool = False,
    pca_n_components: int = 32,
    **tabicl_kwargs
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, object]:
    """
    Apply TabICL embedding extraction to train/val/test splits.
    
    Args:
        X_train, X_val, X_test: Feature arrays
        E_train: Event indicator for training
        feature_names: List of feature names
        use_deep_embeddings: Extract 512D embeddings from row_interactor
        concat_with_raw: Concatenate embeddings with raw features (deep+raw mode)
        pca_for_trees: Apply PCA compression (for tree models like RSF, XGBoost)
        pca_n_components: Target PCA dimensions (default 32)
        
    Returns: (X_train_emb, X_val_emb, X_test_emb, classifier)
    """
    if not TABICL_AVAILABLE:
        raise ImportError("tabicl is not installed. Install with: pip install tabicl")
    
    if verbose:
        print(f"TabICL: {X_train.shape[0]} samples, {X_train.shape[1]} features")
    
    # Convert to DataFrames
    X_train_df = pd.DataFrame(X_train, columns=feature_names)
    X_val_df = pd.DataFrame(X_val, columns=feature_names)
    X_test_df = pd.DataFrame(X_test, columns=feature_names)
    
    # Binary event indicator for TabICL
    y_train_cls = E_train.astype(int)
    
    # Initialize TabICL
    clf = TabICLClassifier(
        checkpoint_version=tabicl_kwargs.get('checkpoint_version', 'tabicl-classifier-v1.1-0506.ckpt'),
        n_estimators=tabicl_kwargs.get('n_estimators', 1),
        random_state=tabicl_kwargs.get('random_state', 42),
        device=tabicl_kwargs.get('device', 'cpu'),
        verbose=tabicl_kwargs.get('verbose', False),
        n_jobs=tabicl_kwargs.get('n_jobs', 1)
    )
    
    clf.fit(X_train_df, y_train_cls)
    
    device = tabicl_kwargs.get('device', 'cpu')
    
    # Try to extract deep embeddings using row_interactor hooks
    X_train_emb = X_val_emb = X_test_emb = None
    if use_deep_embeddings:
        splits = [
            ('train', X_train_df),
            ('val', X_val_df),
            ('test', X_test_df),
        ]
        deep_embeddings = {}
        for split_name, df in splits:
            emb = _extract_row_embeddings(clf, df, split_name, device)
            
            # Check for validity: must not be None, must match rows, must be high-dim (>20)
            if emb is None or emb.shape[1] < 32:
                use_deep_embeddings = False
                deep_embeddings = {}
                break
            deep_embeddings[split_name] = emb
        
        if use_deep_embeddings:
            X_train_emb = deep_embeddings['train']
            X_val_emb = deep_embeddings['val']
            X_test_emb = deep_embeddings['test']
            if verbose:
                print(f"  → Deep embeddings: {X_train_emb.shape[1]}D")
        else:
            if verbose:
                print(f"  → Fallback to predict_proba features")
    
    # Fallback to predict_proba-based features if deep extraction failed or was not requested
    if not use_deep_embeddings:
        train_proba = clf.predict_proba(X_train_df)
        val_proba = clf.predict_proba(X_val_df)
        test_proba = clf.predict_proba(X_test_df)
        
        eps = 1e-10
        def proba_to_enhanced_features(proba):
            p_event = np.clip(proba[:, 1], eps, 1 - eps)
            p_censored = np.clip(proba[:, 0], eps, 1 - eps)
            return np.column_stack([
                p_event, np.log(p_event / p_censored), p_event ** 2, 
                np.sqrt(p_event), -np.log(p_censored),
                p_event * (1 - p_event), np.abs(p_event - 0.5)
            ])
        
        X_train_emb = proba_to_enhanced_features(train_proba)
        X_val_emb = proba_to_enhanced_features(val_proba)
        X_test_emb = proba_to_enhanced_features(test_proba)
    
    # Apply PCA compression if requested (for tree models)
    if pca_for_trees and use_deep_embeddings:
        X_train_emb, X_val_emb, X_test_emb, _ = _apply_pca_compression(
            X_train_emb, X_val_emb, X_test_emb, 
            n_components=pca_n_components, verbose=verbose
        )
    
    # Combine with raw features if requested
    if concat_with_raw:
        # One-hot encode categorical features in raw data before concatenation
        X_train_raw_enc, X_val_raw_enc, X_test_raw_enc, _ = _encode_categorical_features(
            X_train, X_val, X_test, feature_names, verbose=verbose
        )
        X_train_final = np.column_stack([X_train_emb, X_train_raw_enc])
        X_val_final = np.column_stack([X_val_emb, X_val_raw_enc])
        X_test_final = np.column_stack([X_test_emb, X_test_raw_enc])
    else:
        X_train_final = X_train_emb
        X_val_final = X_val_emb
        X_test_final = X_test_emb
        
    if verbose:
        print(f"  → Final shape: {X_train_final.shape[1]} features")
    
    return X_train_final, X_val_final, X_test_final, clf