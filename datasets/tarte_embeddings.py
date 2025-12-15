"""
TARTE Embedding Extractor for Survival Analysis
Updated version with improved deep encoder embedding extraction.
Fixes shape mismatch by correctly handling (1, N, D) output tensors.

Features:
- One-hot encoding for categorical features in deep+raw mode
- Optional PCA compression for tree-based models
- Proper handling of string columns in raw data
"""
from pandas_patch import pd
import numpy as np
np.seterr(over='ignore', invalid='ignore')
from typing import Tuple, Optional, Union, List
import warnings
import torch
from sklearn.preprocessing import OneHotEncoder
from sklearn.decomposition import PCA

from tarte_ai import TARTE_TableEncoder, TARTE_TablePreprocessor
from sklearn.pipeline import Pipeline

def _encode_categorical_features(X_train: np.ndarray, X_dev: np.ndarray, X_val: np.ndarray, X_test: np.ndarray,
                                 feature_names: list, verbose: bool = False) -> Tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, list]:
    """
    Detect and one-hot encode categorical (string) features in raw data.
    Returns encoded arrays and updated feature names.
    """
    # Convert to DataFrame to detect dtypes
    df_train = pd.DataFrame(X_train, columns=feature_names)
    df_dev = pd.DataFrame(X_dev, columns=feature_names)
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
        return X_train.astype(float), X_dev.astype(float), X_val.astype(float), X_test.astype(float), feature_names

    if verbose:
        print(f"  → One-hot encoding {len(cat_cols)} categorical columns: {cat_cols}")

    # Fit OneHotEncoder on training data
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    X_train_cat = encoder.fit_transform(df_train[cat_cols])
    X_dev_cat = encoder.transform(df_dev[cat_cols])
    X_val_cat = encoder.transform(df_val[cat_cols])
    X_test_cat = encoder.transform(df_test[cat_cols])

    # Get encoded feature names
    encoded_names = encoder.get_feature_names_out(cat_cols).tolist()

    # Extract numeric columns
    X_train_num = df_train[num_cols].values.astype(float)
    X_dev_num = df_dev[num_cols].values.astype(float)
    X_val_num = df_val[num_cols].values.astype(float)
    X_test_num = df_test[num_cols].values.astype(float)

    # Combine: numeric + encoded categorical
    X_train_enc = np.column_stack([X_train_num, X_train_cat]) if len(num_cols) > 0 else X_train_cat
    X_dev_enc = np.column_stack([X_dev_num, X_dev_cat]) if len(num_cols) > 0 else X_dev_cat
    X_val_enc = np.column_stack([X_val_num, X_val_cat]) if len(num_cols) > 0 else X_val_cat
    X_test_enc = np.column_stack([X_test_num, X_test_cat]) if len(num_cols) > 0 else X_test_cat

    new_feature_names = num_cols + encoded_names

    return X_train_enc, X_dev_enc, X_val_enc, X_test_enc, new_feature_names


def _apply_pca_compression(X_train_emb: np.ndarray, X_dev_emb: np.ndarray, X_val_emb: np.ndarray, X_test_emb: np.ndarray,
                           n_components: int = 32, verbose: bool = False) -> Tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, PCA]:
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
    X_dev_pca = pca.transform(X_dev_emb)
    X_val_pca = pca.transform(X_val_emb)
    X_test_pca = pca.transform(X_test_emb)

    if verbose:
        explained_var = pca.explained_variance_ratio_.sum() * 100
        print(f"  → PCA: {X_train_emb.shape[1]}D → {X_train_pca.shape[1]}D ({explained_var:.1f}% variance)")

    return X_train_pca, X_dev_pca, X_val_pca, X_test_pca, pca


def apply_tarte_embedding(
        X_train: np.ndarray,
        X_dev: np.ndarray,
        X_val: np.ndarray,
        X_test: np.ndarray,
        E_train: np.ndarray,
        T_train: np.ndarray,
        feature_names: list,
        verbose: bool = True,
        use_deep_embeddings: bool = True,
        concat_with_raw: bool = True,
        pca_for_trees: bool = False,
        pca_n_components: int = 32,
        **tarte_kwargs
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Apply TARTE embedding extraction to train/val/test splits.

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
    if verbose:
        print(f"TARTE: {X_train.shape[0]} samples, {X_train.shape[1]} features")

    # Convert to DataFrames
    X_train_df = pd.DataFrame(X_train, columns=feature_names)
    X_dev_df = pd.DataFrame(X_dev, columns=feature_names)
    X_val_df = pd.DataFrame(X_val, columns=feature_names)
    X_test_df = pd.DataFrame(X_test, columns=feature_names)

    # Binary event indicator (optional)
    #y_train_cls_E = E_train.astype(int)
    #y_train_cls_T = T_train.astype(int)
    # dummy variable
    y_dummy = pd.Series(1, index=np.arange(len(X_train_df)))

    torch.cuda.empty_cache()

    tarte_tab_prepper = TARTE_TablePreprocessor()
    tarte_tab_encoder = TARTE_TableEncoder()
    prep_pipe = Pipeline([("prep", tarte_tab_prepper), ("tabenc", tarte_tab_encoder)])
    # get embeddings
    train_emb = prep_pipe.fit_transform(X_train_df, None) # Binary event indicator could be used here
    dev_emb = prep_pipe.transform(X_dev_df)
    val_emb = prep_pipe.transform(X_val_df)
    test_emb = prep_pipe.transform(X_test_df)
    # Wrap embeddings in DataFrames
    X_train_emb = pd.DataFrame(
        train_emb, columns=[f"x{i}" for i in range(train_emb.shape[1])],
        index=np.arange(X_train.shape[0])
    )
    X_dev_emb = pd.DataFrame(
        dev_emb, columns=[f"x{i}" for i in range(dev_emb.shape[1])],
        index=np.arange(X_dev.shape[0])
    )
    X_val_emb = pd.DataFrame(
        val_emb, columns=[f"x{i}" for i in range(val_emb.shape[1])],
        index=np.arange(X_val.shape[0])
    )
    X_test_emb = pd.DataFrame(
        test_emb, columns=[f"x{i}" for i in range(test_emb.shape[1])],
        index=np.arange(X_test.shape[0])
    )
    X_train_emb = X_train_emb.to_numpy()
    X_dev_emb = X_dev_emb.to_numpy()
    X_val_emb = X_val_emb.to_numpy()
    X_test_emb = X_test_emb.to_numpy()

    # Apply PCA compression if requested (for tree models)
    if pca_for_trees and use_deep_embeddings:
        X_train_emb, X_dev_emb, X_val_emb, X_test_emb, _ = _apply_pca_compression(
            X_train_emb, X_dev_emb, X_val_emb, X_test_emb,
            n_components=pca_n_components, verbose=verbose
        )

    # Combine with raw features if requested
    if concat_with_raw:
        # One-hot encode categorical features in raw data before concatenation
        X_train_raw_enc, X_dev_raw_enc, X_val_raw_enc, X_test_raw_enc, _ = _encode_categorical_features(
            X_train, X_dev, X_val, X_test, feature_names, verbose=verbose
        )
        X_train_final = np.column_stack([X_train_emb, X_train_raw_enc])
        X_dev_final = np.column_stack([X_dev_emb, X_dev_raw_enc])
        X_val_final = np.column_stack([X_val_emb, X_val_raw_enc])
        X_test_final = np.column_stack([X_test_emb, X_test_raw_enc])
    else:
        X_train_final = X_train_emb
        X_dev_final = X_val_emb
        X_val_final = X_val_emb
        X_test_final = X_test_emb

    if verbose:
        print(f"  → Final shape: {X_train_final.shape[1]} features")

    return X_train_final, X_dev_final, X_val_final, X_test_final