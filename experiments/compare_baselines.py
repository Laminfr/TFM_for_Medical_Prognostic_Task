"""
Compare all baseline models by running them sequentially and collecting results.
"""
import time
import numpy as np
from tabulate import tabulate

# Import all baseline models
try:
    from ..datasets.data_loader import load_and_preprocess_data
    from ..coxph.utilities import train_cox_model, evaluate_model as evaluate_cox
    from ..XGB.utilities import train_xgboost_model, evaluate_xgboost_model
    from ..deepsurv.utilities import train_deepsurv_model, evaluate_deepsurv_model
    from ..rsf.utilities import train_rsf_model, evaluate_rsf_model
except ImportError:
    from ..datasets.data_loader import load_and_preprocess_data
    from ..coxph.utilities import train_cox_model, evaluate_model as evaluate_cox
    from ..XGB.utilities import train_xgboost_model, evaluate_xgboost_model
    from ..deepsurv.utilities import train_deepsurv_model, evaluate_deepsurv_model
    from ..rsf.utilities import train_rsf_model, evaluate_rsf_model


def run_comparison(dataset='METABRIC', normalize=True, test_size=0.2, random_state=42):
    """Run all baselines and collect results."""
    print("="*70)
    print(f"BASELINE COMPARISON ON {dataset} DATASET")
    print("="*70)
    
    # Load data once
    print(f"\nLoading and preprocessing {dataset} dataset...")
    X_train, X_val, t_train, t_val, e_train, e_val = load_and_preprocess_data(
        dataset=dataset,
        normalize=normalize,
        test_size=test_size,
        random_state=random_state
    )
    X_train_sk, X_val_sk, y_train_sk, y_val_sk = load_and_preprocess_data(
        dataset=dataset,
        normalize=normalize,
        test_size=test_size,
        random_state=random_state,
        as_sksurv_y=True
    )
    
    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    print(f"Features: {X_train.shape[1]}")
    print(f"Event rate (train): {e_train.mean():.2%}")
    print(f"Event rate (val):   {e_val.mean():.2%}")
    
    results = []
    
    # 1. Cox Proportional Hazards
    print("\n" + "="*70)
    print("1. COX PROPORTIONAL HAZARDS")
    print("="*70)
    start_time = time.time()
    try:
        cph = train_cox_model(X_train, t_train, e_train)
        print("\nModel Summary:")
        print(cph.summary[['coef', 'exp(coef)', 'p']])
        metrics_cox = evaluate_cox(cph, X_train, X_val, t_train, t_val, e_train, e_val)
        train_time = time.time() - start_time
        results.append({
            "Model": "Cox PH",
            "C-index (train)": f"{metrics_cox['c_index_train']:.4f}",
            "C-index (val)": f"{metrics_cox['c_index_val']:.4f}",
            "IBS (val)": f"{metrics_cox['ibs_val']:.4f}" if not np.isnan(metrics_cox['ibs_val']) else "N/A",
            "Time (s)": f"{train_time:.1f}"
        })
        print(f"✓ Completed in {train_time:.1f}s")
    except Exception as e:
        print(f"✗ Failed: {e}")
        import traceback
        traceback.print_exc()
        results.append({"Model": "Cox PH", "C-index (train)": "Error", "C-index (val)": "Error", 
                       "IBS (val)": "Error", "Time (s)": "-"})
    
    # 2. XGBoost Survival
    print("\n" + "="*70)
    print("2. XGBOOST SURVIVAL")
    print("="*70)
    start_time = time.time()
    try:
        model_xgb = train_xgboost_model(X_train_sk, y_train_sk)
        metrics_xgb = evaluate_xgboost_model(model_xgb, X_train_sk, X_val_sk, y_train_sk, y_val_sk)
        train_time = time.time() - start_time
        results.append({
            "Model": "XGBoost",
            "C-index (train)": f"{metrics_xgb['c_index_train']:.4f}",
            "C-index (val)": f"{metrics_xgb['c_index_val']:.4f}",
            "IBS (val)": f"{metrics_xgb['ibs_val']:.4f}" if not np.isnan(metrics_xgb['ibs_val']) else "N/A",
            "Time (s)": f"{train_time:.1f}"
        })
        print(f"✓ Completed in {train_time:.1f}s")
    except Exception as e:
        print(f"✗ Failed: {e}")
        import traceback
        traceback.print_exc()
        results.append({"Model": "XGBoost", "C-index (train)": "Error", "C-index (val)": "Error",
                       "IBS (val)": "Error", "Time (s)": "-"})
    
    # 3. DeepSurv
    print("\n" + "="*70)
    print("3. DEEPSURV (NEURAL COX)")
    print("="*70)
    start_time = time.time()
    try:
        model_ds, device = train_deepsurv_model(X_train, t_train, e_train, epochs=100)
        metrics_ds = evaluate_deepsurv_model(model_ds, device, X_train, X_val, t_train, t_val, e_train, e_val)
        train_time = time.time() - start_time
        results.append({
            "Model": "DeepSurv",
            "C-index (train)": f"{metrics_ds['c_index_train']:.4f}",
            "C-index (val)": f"{metrics_ds['c_index_val']:.4f}",
            "IBS (val)": f"{metrics_ds['ibs_val']:.4f}" if not np.isnan(metrics_ds['ibs_val']) else "N/A",
            "Time (s)": f"{train_time:.1f}"
        })
        print(f"✓ Completed in {train_time:.1f}s")
    except Exception as e:
        print(f"✗ Failed: {e}")
        import traceback
        traceback.print_exc()
        results.append({"Model": "DeepSurv", "C-index (train)": "Error", "C-index (val)": "Error",
                       "IBS (val)": "Error", "Time (s)": "-"})
    
    # 4. Random Survival Forest
    print("\n" + "="*70)
    print("4. RANDOM SURVIVAL FOREST")
    print("="*70)
    start_time = time.time()
    try:
        model_rsf = train_rsf_model(X_train_sk, y_train_sk, n_estimators=200)
        metrics_rsf = evaluate_rsf_model(model_rsf, X_train_sk, X_val_sk, y_train_sk, y_val_sk)
        train_time = time.time() - start_time
        results.append({
            "Model": "RSF",
            "C-index (train)": f"{metrics_rsf['c_index_train']:.4f}",
            "C-index (val)": f"{metrics_rsf['c_index_val']:.4f}",
            "IBS (val)": f"{metrics_rsf['ibs_val']:.4f}" if not np.isnan(metrics_rsf['ibs_val']) else "N/A",
            "Time (s)": f"{train_time:.1f}"
        })
        print(f"✓ Completed in {train_time:.1f}s")
    except Exception as e:
        print(f"✗ Failed: {e}")
        import traceback
        traceback.print_exc()
        results.append({"Model": "RSF", "C-index (train)": "Error", "C-index (val)": "Error",
                       "IBS (val)": "Error", "Time (s)": "-"})
    
    # Print summary table
    print("\n" + "="*70)
    print("SUMMARY OF RESULTS")
    print("="*70)
    print(tabulate(results, headers="keys", tablefmt="grid"))
    print("\nNote: All metrics computed using NeuralFineGray framework")
    print("      C-index ↑ is better, IBS ↓ is better")
    print("="*70)


if __name__ == "__main__":
    run_comparison()  # Default: METABRIC
    # run_comparison(dataset='GBSG')  # To test on GBSG dataset