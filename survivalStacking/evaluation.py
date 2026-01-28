"""
Evaluation utilities for Survival Stacking

Computes standard survival analysis metrics:
- C-Index (Concordance Index): Discrimination ability
- IBS (Integrated Brier Score): Calibration quality
- Brier Score at specific times
"""

import numpy as np
from typing import Tuple, Dict, Optional, Union
from scipy.stats import ttest_rel
import warnings


def concordance_index(
    T: np.ndarray,
    E: np.ndarray,
    risk_scores: np.ndarray
) -> float:
    """
    Compute Harrell's concordance index.
    
    Parameters
    ----------
    T : np.ndarray
        True event/censoring times
    E : np.ndarray
        Event indicators (1=event, 0=censored)
    risk_scores : np.ndarray
        Predicted risk scores (higher = more risk)
        
    Returns
    -------
    c_index : float
        Concordance index in [0, 1], 0.5 = random
    """
    try:
        from sksurv.metrics import concordance_index_censored
        c_idx = concordance_index_censored(E > 0, T, risk_scores)[0]
        return float(c_idx)
    except ImportError:
        # Fallback implementation
        return _concordance_index_numpy(T, E, risk_scores)


def _concordance_index_numpy(
    T: np.ndarray,
    E: np.ndarray, 
    risk_scores: np.ndarray
) -> float:
    """Pure numpy implementation of concordance index."""
    T = np.asarray(T).flatten()
    E = np.asarray(E).flatten()
    risk_scores = np.asarray(risk_scores).flatten()
    
    n = len(T)
    concordant = 0
    discordant = 0
    tied_risk = 0
    
    for i in range(n):
        for j in range(i + 1, n):
            # Only consider comparable pairs
            if E[i] == 0 and E[j] == 0:
                continue
            
            # Determine which patient had event first
            if T[i] < T[j] and E[i] == 1:
                # Patient i had event before j (j was still at risk)
                if risk_scores[i] > risk_scores[j]:
                    concordant += 1
                elif risk_scores[i] < risk_scores[j]:
                    discordant += 1
                else:
                    tied_risk += 0.5
            elif T[j] < T[i] and E[j] == 1:
                # Patient j had event before i
                if risk_scores[j] > risk_scores[i]:
                    concordant += 1
                elif risk_scores[j] < risk_scores[i]:
                    discordant += 1
                else:
                    tied_risk += 0.5
            elif T[i] == T[j] and E[i] == 1 and E[j] == 1:
                # Both had events at same time
                if risk_scores[i] == risk_scores[j]:
                    tied_risk += 1
                else:
                    concordant += 0.5
                    discordant += 0.5
    
    total = concordant + discordant + tied_risk
    if total == 0:
        return 0.5
    return (concordant + 0.5 * tied_risk) / total


def brier_score(
    T: np.ndarray,
    E: np.ndarray,
    survival_probs: np.ndarray,
    eval_time: float,
    T_train: Optional[np.ndarray] = None,
    E_train: Optional[np.ndarray] = None
) -> float:
    """
    Compute Brier Score at a specific time point with IPCW correction.
    
    Parameters
    ----------
    T : np.ndarray
        True event/censoring times (test set)
    E : np.ndarray
        Event indicators (test set)
    survival_probs : np.ndarray
        Predicted survival probabilities at eval_time
    eval_time : float
        Time at which to evaluate
    T_train, E_train : optional
        Training data for estimating censoring distribution
        If None, uses test data
        
    Returns
    -------
    bs : float
        Brier score at eval_time (lower is better)
    """
    T = np.asarray(T).flatten()
    E = np.asarray(E).flatten()
    survival_probs = np.asarray(survival_probs).flatten()
    
    n = len(T)
    
    # Estimate censoring distribution using Kaplan-Meier
    if T_train is None:
        T_train = T
        E_train = E
    
    # Censoring KM: treat censoring as "event"
    G = _kaplan_meier_censoring(T_train, E_train, eval_time)
    G_individual = _kaplan_meier_censoring_at_times(T_train, E_train, np.minimum(T, eval_time))
    
    # Avoid division by zero
    G = max(G, 1e-10)
    G_individual = np.maximum(G_individual, 1e-10)
    
    bs = 0.0
    weights_sum = 0.0
    
    for i in range(n):
        if T[i] <= eval_time and E[i] == 1:
            # Event before eval_time: prediction should be 0
            weight = 1.0 / G_individual[i]
            bs += weight * (survival_probs[i] ** 2)
            weights_sum += weight
        elif T[i] > eval_time:
            # Still at risk at eval_time: prediction should be 1
            weight = 1.0 / G
            bs += weight * ((1 - survival_probs[i]) ** 2)
            weights_sum += weight
        # Censored before eval_time: excluded (IPCW)
    
    if weights_sum == 0:
        return 0.0
    
    return bs / weights_sum


def _kaplan_meier_censoring(T: np.ndarray, E: np.ndarray, t: float) -> float:
    """Estimate censoring survival probability at time t."""
    # Treat censoring as event (flip E)
    E_cens = 1 - E
    
    unique_times = np.sort(np.unique(T[E_cens > 0]))
    unique_times = unique_times[unique_times <= t]
    
    G = 1.0
    for time in unique_times:
        at_risk = np.sum(T >= time)
        events = np.sum((T == time) & (E_cens > 0))
        if at_risk > 0:
            G *= (1 - events / at_risk)
    
    return G


def _kaplan_meier_censoring_at_times(T: np.ndarray, E: np.ndarray, times: np.ndarray) -> np.ndarray:
    """Estimate censoring survival probability at multiple times."""
    return np.array([_kaplan_meier_censoring(T, E, t) for t in times])


def integrated_brier_score(
    T: np.ndarray,
    E: np.ndarray,
    survival_matrix: np.ndarray,
    times: np.ndarray,
    T_train: Optional[np.ndarray] = None,
    E_train: Optional[np.ndarray] = None
) -> float:
    """
    Compute Integrated Brier Score over multiple time points.
    
    Parameters
    ----------
    T : np.ndarray
        True event/censoring times
    E : np.ndarray
        Event indicators
    survival_matrix : np.ndarray
        Predicted survival probabilities (n_samples, n_times)
    times : np.ndarray
        Time points corresponding to columns of survival_matrix
    T_train, E_train : optional
        Training data for censoring distribution
        
    Returns
    -------
    ibs : float
        Integrated Brier Score (lower is better)
    """
    T = np.asarray(T).flatten()
    E = np.asarray(E).flatten()
    survival_matrix = np.asarray(survival_matrix)
    times = np.asarray(times).flatten()
    
    if T_train is None:
        T_train = T
        E_train = E
    
    # Compute Brier score at each time point
    bs_values = []
    valid_times = []
    
    for i, t in enumerate(times):
        # Skip times beyond observed range
        if t < T.min() or t > T.max():
            continue
        
        bs = brier_score(T, E, survival_matrix[:, i], t, T_train, E_train)
        bs_values.append(bs)
        valid_times.append(t)
    
    if len(bs_values) < 2:
        return 0.25  # Default if not enough valid times
    
    # Integrate using trapezoidal rule
    valid_times = np.array(valid_times)
    bs_values = np.array(bs_values)
    
    ibs = np.trapz(bs_values, valid_times) / (valid_times[-1] - valid_times[0])
    
    return float(ibs)


def compute_survival_metrics(
    T: np.ndarray,
    E: np.ndarray,
    survival_matrix: np.ndarray,
    times: np.ndarray,
    T_train: Optional[np.ndarray] = None,
    E_train: Optional[np.ndarray] = None,
    quantiles: Tuple[float, ...] = (0.25, 0.50, 0.75)
) -> Dict[str, float]:
    """
    Compute comprehensive survival metrics.
    
    Parameters
    ----------
    T : np.ndarray
        True event/censoring times
    E : np.ndarray
        Event indicators
    survival_matrix : np.ndarray
        Predicted survival probabilities (n_samples, n_times)
    times : np.ndarray
        Time points
    T_train, E_train : optional
        Training data for metrics
    quantiles : tuple
        Event time quantiles for C-Index evaluation
        
    Returns
    -------
    metrics : dict
        Dictionary with c_index_q25, c_index_q50, c_index_q75, ibs
    """
    T = np.asarray(T).flatten()
    E = np.asarray(E).flatten()
    survival_matrix = np.asarray(survival_matrix)
    times = np.asarray(times).flatten()
    
    if T_train is None:
        T_train = T
        E_train = E
    
    metrics = {}
    
    # Compute C-Index at quantile times
    event_times = T[E > 0]
    quantile_times = np.percentile(event_times, [q * 100 for q in quantiles])
    
    for q, q_time in zip(quantiles, quantile_times):
        q_name = f"q{int(q*100):02d}"
        
        # Find closest time in prediction
        closest_idx = np.argmin(np.abs(times - q_time))
        
        # Risk score = 1 - survival probability
        risk_at_q = 1.0 - survival_matrix[:, closest_idx]
        
        c_idx = concordance_index(T, E, risk_at_q)
        metrics[f'c_index_{q_name}'] = c_idx
    
    # Compute IBS
    ibs = integrated_brier_score(T, E, survival_matrix, times, T_train, E_train)
    metrics['ibs'] = ibs
    
    return metrics


def print_metrics(metrics: Dict[str, float], model_name: str = "Model"):
    """Pretty print metrics."""
    print(f"\n{model_name} Results:")
    print("-" * 40)
    for key, value in sorted(metrics.items()):
        if 'c_index' in key:
            print(f"  {key}: {value:.4f}")
    print(f"  IBS: {metrics.get('ibs', 0):.4f}")
    print("-" * 40)


def pairwise_t_test_vs_baseline(fold_results, baseline='TabICL', baseline_method=None, metric='c_index_q50'):
    """
    Perform pairwise t-tests of all approaches against a baseline method.
    """
    if baseline_method is None:
        raise ValueError(
            "baseline_method must be provided (e.g., 'tabpfn_direct' or 'xgboost_raw')."
        )
    # 1. Identify all competing approaches (excluding baseline)
    all_approaches = set()
    baselines_approaches = set()
    for fold in fold_results:
        for model_family, methods in fold.items():
            if model_family != baseline:
                for method_name in methods.keys():
                    all_approaches.add((model_family, method_name))


    # 2. Collect paired fold-wise values
    p_values = {}

    for model_family, method_name in all_approaches:
        baseline_array = []
        approach_array = []

        for fold in fold_results:
            try:
                baseline_metrics = fold[baseline][baseline_method][metric]
                approach_metrics = fold[model_family][method_name][metric]

                if (not isinstance(baseline_metrics, dict)) or ('error' in baseline_metrics):
                    continue
                if (not isinstance(approach_metrics, dict)) or ('error' in approach_metrics):
                    continue

                baseline_val = baseline_metrics[metric]
                approach_val = approach_metrics[metric]

                baseline_array.append(baseline_val)
                approach_array.append(approach_val)
            except KeyError:
                # skip folds where one of the values is missing
                continue

        # 3. Run paired t-test if enough folds
        if len(baseline_array) >= 2:
            t_stat, p_val = ttest_rel(baseline_array, approach_array)
            p_values[f'{model_family}:{method_name}'] = {
                'baseline_mean': np.mean(baseline_array),
                'method_mean': np.mean(approach_array),
                'mean_diff': np.mean(np.array(approach_array) - np.array(baseline_array)),
                't_stat': t_stat,
                'p_value': p_val,
                'n_folds': len(baseline_array)
            }
        else:
            p_values[f'{model_family}:{method_name}'] = {
                'p_value': np.nan,
                'n_folds': len(baseline_array)
            }

    return p_values
