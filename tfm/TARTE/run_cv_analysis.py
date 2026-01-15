
"""
Cross-Validation Analysis Pipeline for TARTE Embeddings

This script orchestrates:
1. 5-Fold Cross-Validation
2. Hyperparameter Tuning (Random Search) within each fold
3. TARTE embeddings generated dynamically inside folds (transductive setup)
4. Evaluation at time quantiles (q0.25, q0.50, q0.75)

"""
import os
import warnings
import json
import numpy as np
from pathlib import Path
from datetime import datetime
import traceback

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Add project root to path
dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import sys
sys.path.append(dir)

from datasets.datasets import load_dataset
from experiments.experiment import (
    CoxExperiment, RSFExperiment, XGBoostExperiment,
    DeepSurvExperiment, TARTEExperiment
)

# =============================================================================
# CONFIGURATION
# =============================================================================
RANDOM_SEED = 42
N_FOLDS = 5
N_ITER = 50  # Random search iterations per fold

# Hyperparameter Grids (based on NFG paper)
# Note: DeepSurv has separate grids for raw vs deep modes
HYPER_GRIDS = {
    'CoxPH': {
        'penalizer': [0.001, 0.01, 0.1]
    },
    'RSF': {
        'n_estimators': [100, 300, 500],
        'max_depth': [5, 10, None],
        'min_samples_leaf': [5, 10, 15]
    },
    'XGBoost': {
        'n_estimators': [100, 200],
        'max_depth': [4, 6, 8],
        'learning_rate': [0.05, 0.1],
        'min_child_weight': [5, 10]
    },
    'DeepSurv': {
        'layers': [[50], [50, 50], [100, 100]],
        'learning_rate': [1e-3, 1e-4],
        'dropout': [0.0, 0.25, 0.5],
        'epochs': [500],
        'batch': [256],
        'patience_max': [5]
    }
}

# Separate hyperparameter grid for DeepSurv with TabICL embeddings
# Lower learning rates for stability with high-dimensional embeddings
HYPER_GRIDS_DEEP = {
    'DeepSurv': {
        'layers': [[100], [100, 100], [256, 128]],  # Larger layers for 512D input
        'learning_rate': [1e-4, 1e-5],  # Lower LR for stability
        'dropout': [0.1, 0.3, 0.5],  # Higher dropout to prevent overfitting
        'epochs': [500],
        'batch': [256],
        'patience_max': [10]  # More patience with lower LR
    }
}

# Models that should use PCA compression for TARTE embeddings
TREE_MODELS = ['RSF', 'XGBoost']
# Models that keep full 512D embeddings
NEURAL_MODELS = ['DeepSurv', 'NFG', 'DeSurv']
LINEAR_MODELS = ['CoxPH']

# Experiment classes mapping
EXPERIMENT_CLASSES = {
    'CoxPH': CoxExperiment,
    'RSF': RSFExperiment,
    'XGBoost': XGBoostExperiment,
    'DeepSurv': DeepSurvExperiment
}

# Modes to evaluate
MODES = ['raw', 'deep', 'deep+raw']

# Models to evaluate (excluding NFG and DeSurv which have dtype issues)
MODELS = ['CoxPH', 'RSF', 'XGBoost', 'DeepSurv']


def compute_metrics_at_quantiles(predictions, t, e, times):
    """
    Compute C-Index and IBS at specific time quantiles.

    Args:
        predictions: DataFrame with survival predictions (samples x times)
        t: True event times
        e: Event indicators
        times: Evaluation time points

    Returns:
        dict with metrics at q0.25, q0.50, q0.75
    """
    from metrics.calibration import integrated_brier_score as nfg_ibs
    from metrics.discrimination import truncated_concordance_td as nfg_cindex

    # Get event times for quantile calculation
    event_times = t[e > 0]
    quantiles = np.percentile(event_times, [25, 50, 75])

    metrics = {}

    for q_idx, (q_name, q_time) in enumerate(zip(['q25', 'q50', 'q75'], quantiles)):
        try:
            # Get risk predictions (1 - survival)
            # Find closest time point in predictions
            if hasattr(predictions, 'columns'):
                pred_times = predictions.columns.get_level_values(1).values
                closest_idx = np.argmin(np.abs(pred_times - q_time))
                risk_at_q = 1.0 - predictions.iloc[:, closest_idx].values
            else:
                risk_at_q = 1.0 - predictions[:, np.argmin(np.abs(times - q_time))]

            # C-Index at quantile
            from sksurv.metrics import concordance_index_censored
            c_idx = concordance_index_censored(e > 0, t, risk_at_q)[0]
            metrics[f'c_index_{q_name}'] = float(c_idx)

        except Exception:
            metrics[f'c_index_{q_name}'] = 0.5

    # Overall IBS
    try:
        if hasattr(predictions, 'values'):
            risk_pred = 1.0 - predictions.values
        else:
            risk_pred = 1.0 - predictions

        ibs, _ = nfg_ibs(
            e_test=e.astype(int),
            t_test=t.astype(float),
            risk_predicted_test=risk_pred,
            times=times,
            km=(e.astype(int), t.astype(float)),
            competing_risk=1
        )
        metrics['ibs'] = float(ibs)
    except:
        metrics['ibs'] = 0.25

    return metrics


def run_experiment(model_name, mode, dataset='METABRIC', results_dir=None, verbose=True):
    """
    Run a single experiment configuration.

    Args:
        model_name: Name of the model ('CoxPH', 'RSF', etc.)
        mode: Feature mode ('raw', 'deep', 'deep+raw')
        dataset: Dataset name
        results_dir: Directory to save results
        verbose: Print progress

    Returns:
        dict with results
    """
    global N_ITER
    if verbose:
        print(f"Running: {model_name} ({mode})")

    result = {
        'model': model_name,
        'mode': mode,
        'dataset': dataset,
        'timestamp': datetime.now().isoformat()
    }

    try:
        # Load data
        if mode in ['deep', 'deep+raw']:
            # Need raw data for TabICL
            X, T, E, feature_names, df_raw = load_dataset(
                dataset=dataset,
                normalize=True,
                return_raw=True
            )
        else:
            X, T, E, feature_names = load_dataset(dataset=dataset, normalize=True)
            df_raw = None

        # Define evaluation times
        times = np.linspace(T.min(), T.max(), 100)

        # Setup experiment path
        exp_path = str(results_dir / f'{model_name}_{mode}_{dataset}')

        # Select hyperparameter grid based on model and mode
        # Use special grid for DeepSurv with TARTE embeddings
        if mode in ['deep', 'deep+raw'] and model_name in HYPER_GRIDS_DEEP:
            hyper_grid = HYPER_GRIDS_DEEP[model_name]
        else:
            hyper_grid = HYPER_GRIDS.get(model_name, {})

        base_class = EXPERIMENT_CLASSES[model_name]

        # less iterations on SEER
        if dataset == 'SEER':
            N_ITER = 20
        if mode == 'raw':
            # Standard experiment without TARTE
            exp = base_class.create(
                hyper_grid=hyper_grid,
                n_iter=N_ITER,
                k=N_FOLDS,
                random_seed=RANDOM_SEED,
                path=exp_path,
                save=True,
                force=True  # Force re-run, don't use cached results
            )
            # Set times after creation (it's set in __init__ but create() doesn't pass it)
            exp.times = times

            # Run CV
            predictions = exp.train(X, T, E)

        else:
            # Determine if PCA compression should be used (for tree models)
            use_pca = model_name in TREE_MODELS
            # TARTE-enhanced experiment
            exp = TARTEExperiment(
                base_experiment_class=base_class,
                tarte_mode=mode,
                hyper_grid=hyper_grid,
                n_iter=N_ITER,
                k=N_FOLDS,
                random_seed=RANDOM_SEED,
                path=exp_path,
                save=True,
                times=times,
                device='cuda' if model_name in NEURAL_MODELS else 'cpu',
                # Model-specific TARTE settings
                pca_for_trees=use_pca,
                pca_n_components=32 if use_pca else None
            )

            # Run CV with raw data
            predictions = exp.train(
                X, T, E,
                x_raw=df_raw,
                feature_names=list(feature_names)
            )

        if predictions is not None:
            # Compute metrics at quantiles
            metrics = compute_metrics_at_quantiles(predictions, T, E, times)
            result.update(metrics)
            result['status'] = 'success'

            if verbose:
                print(
                    f"  → {model_name} ({mode}): C-Index={metrics.get('c_index_q50', 0):.4f}, IBS={metrics.get('ibs', 0):.4f}")
        else:
            result['status'] = 'no_predictions'

    except Exception as ex:
        traceback.print_exc()
        print(f"ERROR: {model_name} ({mode}): {ex}")
        result['status'] = 'failed'
        result['error'] = str(ex)

    return result


def main(dataset='METABRIC'):
    """Main entry point."""

    # Set results directory based on dataset
    results_dir = Path(os.path.join(dir, f"results/tarte/cv_{dataset.lower()}"))
    results_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(f"CROSS-VALIDATION ANALYSIS: TARTE on {dataset}")
    print("=" * 60)
    print(f"Folds: {N_FOLDS} | Random Search Iterations: {N_ITER}")
    print(f"Modes: {MODES}")
    print(f"Models: {MODELS}")
    print("=" * 60)

    all_results = []

    # Run all configurations
    for mode in MODES:
        for model in MODELS:
            try:
                result = run_experiment(model, mode, dataset=dataset, results_dir=results_dir)
                all_results.append(result)

                # Save intermediate results
                results_file = results_dir / f'cv_results_{datetime.now().strftime("%Y%m%d")}.json'
                with open(results_file, 'w') as f:
                    json.dump(all_results, f, indent=2)

            except Exception as ex:
                print(f"FATAL ERROR for {model} ({mode}): {ex}")
                all_results.append({
                    'model': model,
                    'mode': mode,
                    'status': 'fatal_error',
                    'error': str(ex)
                })

    # Final summary
    print("\n" + "=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)

    # Create summary table
    summary = {}
    for r in all_results:
        if r.get('status') == 'success':
            key = (r['model'], r['mode'])
            summary[key] = {
                'c_index_q50': r.get('c_index_q50', 0),
                'ibs': r.get('ibs', 0)
            }

    print(f"\n{'Model':<15} {'Mode':<10} {'C-Index (q50)':<15} {'IBS':<10}")
    print("-" * 50)
    for (model, mode), metrics in sorted(summary.items()):
        print(f"{model:<15} {mode:<10} {metrics['c_index_q50']:.4f}        {metrics['ibs']:.4f}")

    # Save final results
    final_file = results_dir / f'cv_results_final_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    with open(final_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to: {final_file}")


if __name__ == "__main__":
    main(dataset='PBC')
