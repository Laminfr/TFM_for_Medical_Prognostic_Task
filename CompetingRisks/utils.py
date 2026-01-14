"""
Utility functions for Competing Risks Benchmark.

These are helper functions that wrap the existing datasets and metrics modules
for the specific needs of the competing risks benchmark.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from sklearn.model_selection import train_test_split


def get_competing_risks_datasets() -> List[str]:
    """Return list of datasets suitable for competing risks analysis."""
    return ['SEER', 'SYNTHETIC_COMPETING']


def split_data(
    X: np.ndarray,
    T: np.ndarray,
    E: np.ndarray,
    train_size: float = 0.7,
    val_size: float = 0.15,
    test_size: float = 0.15,
    random_state: int = 42,
    stratify: bool = True
) -> Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Split data into train/val/test sets, optionally stratified by event type.
    
    Args:
        X, T, E: Feature matrix, times, events
        train_size, val_size, test_size: Split proportions
        random_state: Random seed
        stratify: Whether to stratify by event type
        
    Returns:
        Dictionary with 'train', 'val', 'test' keys, each containing (X, T, E)
    """
    assert abs(train_size + val_size + test_size - 1.0) < 1e-6
    
    # First split: train vs (val + test)
    stratify_col = E if stratify else None
    X_train, X_temp, T_train, T_temp, E_train, E_temp = train_test_split(
        X, T, E, 
        train_size=train_size,
        random_state=random_state,
        stratify=stratify_col
    )
    
    # Second split: val vs test
    val_ratio = val_size / (val_size + test_size)
    stratify_col = E_temp if stratify else None
    X_val, X_test, T_val, T_test, E_val, E_test = train_test_split(
        X_temp, T_temp, E_temp,
        train_size=val_ratio,
        random_state=random_state,
        stratify=stratify_col
    )
    
    return {
        'train': (X_train, T_train, E_train),
        'val': (X_val, T_val, E_val),
        'test': (X_test, T_test, E_test)
    }


def get_evaluation_times(T: np.ndarray, E: np.ndarray, n_times: int = 20) -> np.ndarray:
    """
    Get evaluation time points for metrics computation.
    
    Uses quantiles of event times for better coverage.
    """
    event_times = T[E > 0]
    if len(event_times) < n_times:
        event_times = T
    
    # Use quantile-based times to ensure good coverage
    percentiles = np.linspace(5, 95, n_times)
    times = np.percentile(event_times, percentiles)
    return np.unique(times)


def evaluate_competing_risks_model(
    T_test: np.ndarray,
    E_test: np.ndarray,
    cif_predictions: Dict[int, np.ndarray],
    times: np.ndarray,
    T_train: Optional[np.ndarray] = None,
    E_train: Optional[np.ndarray] = None
) -> Dict[str, Dict[int, float]]:
    """
    Evaluate competing risks predictions using existing metrics.
    
    Args:
        T_test: Test set times
        E_test: Test set events
        cif_predictions: Dict mapping risk -> CIF matrix (n_samples, n_times)
        times: Evaluation time points
        T_train, E_train: Training data for IPCW
        
    Returns:
        Dictionary with metrics for each risk
    """
    from metrics.discrimination import truncated_concordance_td
    from metrics.calibration import integrated_brier_score
    
    results = {'c_index': {}, 'ibs': {}}
    
    # Use training data for IPCW if provided
    km_data = (E_train, T_train) if T_train is not None else (E_test, T_test)
    
    # Get median time for C-index evaluation
    event_times = T_test[E_test > 0]
    t_median = np.median(event_times) if len(event_times) > 0 else np.median(T_test)
    
    for risk, cif_matrix in cif_predictions.items():
        # Cause-specific C-index using existing function
        c_idx, _ = truncated_concordance_td(
            E_test, T_test, cif_matrix, times, t_median,
            km=km_data, competing_risk=risk
        )
        results['c_index'][risk] = c_idx
        
        # Integrated Brier Score using existing function
        ibs, _ = integrated_brier_score(
            E_test, T_test, cif_matrix, times,
            km=km_data, competing_risk=risk
        )
        results['ibs'][risk] = ibs
    
    return results


def aggregate_metrics(
    metrics: Dict[str, Dict[int, float]],
    n_risks: int
) -> Dict[str, float]:
    """
    Aggregate per-risk metrics into summary metrics.
    
    Args:
        metrics: Per-risk metrics from evaluate_competing_risks_model
        n_risks: Number of competing risks
        
    Returns:
        Aggregated metrics (mean across risks)
    """
    aggregated = {}
    
    for metric_name in ['c_index', 'ibs']:
        values = [metrics[metric_name].get(r, np.nan) for r in range(1, n_risks + 1)]
        values = [v for v in values if not np.isnan(v)]
        if values:
            aggregated[f'{metric_name}_mean'] = np.mean(values)
            # Also store individual values
            for r, v in metrics[metric_name].items():
                aggregated[f'{metric_name}_risk{r}'] = v
    
    return aggregated
