
"""
Experiment with TARTE embedding extraction.
Compare all baseline models by running them sequentially and collecting results.
"""
import time
from pandas_patch import pd
import warnings
warnings.filterwarnings("ignore", message="TypedStorage is deprecated")
import os
import sys
sys.path.append("/")

from datasets.data_loader import load_and_preprocess_data
from coxph.utilities import train_cox_model, evaluate_model as evaluate_cox
from xgb_survival.utilities import train_xgboost_model, evaluate_xgboost_model
from deepsurv.utilities import train_deepsurv_model, evaluate_deepsurv_model
from rsf.utilities import train_rsf_model, evaluate_rsf_model
from tfm.TARTE.legacy.legacy_embedding_strategies import get_embeddings_tarte, get_embeddings_dummy_tarte, get_embeddings_combination_tarte
from images.utilities import plot_results_relative, plot_results_absolute


def baselines_evaluate_embeddings(dataset='METABRIC', normalize=True, test_size=0.2, random_state=42, embeddings_flag=None):
    """Run all baselines on raw data and embeddings and collect results."""
    print("= " *70)
    print(f"BASELINE COMPARISON ON {dataset} DATASET")
    print("= " *70)

    # Load data once
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

    if embeddings_flag=="emb":
        print(f"\n>>> TARTE Embeddings with no target\n")
        print("Extract TARTE embeddings ...")
        X_train, X_val = get_embeddings_tarte(X_train, X_val)
        X_train_sk, X_val_sk = X_train, X_val
        dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        X_train.to_pickle(os.path.join(dir, "tfm", "TARTE", "X_train_tarte.pkl"))
        X_val.to_pickle("X_val_tarte.pkl")
        print("Run Baselines on embeddings")
    elif embeddings_flag=="dummy":
        print(f"\n>>> TARTE Embeddings with dummy target\n")
        print("Extract TARTE embeddings with dummy y ...")
        X_train, X_val = get_embeddings_dummy_tarte(X_train, X_val)
        X_train_sk, X_val_sk = X_train, X_val
        dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        X_train.to_pickle(os.path.join(dir, "tfm", "TARTE", "X_train_tarte.pkl"))
        X_val.to_pickle("X_val_tarte.pkl")
        print("Run Baselines on embeddings")
    elif embeddings_flag == "combi":
        print(f"\n>>> TARTE Embeddings for time and event combined\n")
        print("Extract TARTE embeddings with time and event combined...")
        X_train, X_val = get_embeddings_combination_tarte(X_train, X_val, t_train, e_train)
        X_train_sk, X_val_sk = X_train, X_val
        dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        X_train.to_pickle(os.path.join(dir, "tfm", "TARTE", "X_train_tarte.pkl"))
        X_val.to_pickle("X_val_tarte.pkl")
        print("Run Baselines on embeddings")
    else:
        print(f"\n>>> Use raw data for baseline predictions\n")

    results = []

    # 1. Cox Proportional Hazards
    print("Cox")
    start_time = time.time()
    try:
        cph = train_cox_model(X_train, t_train, e_train)
        metrics_cox = evaluate_cox(cph, X_train, X_val, t_train, t_val, e_train, e_val)
        train_time = time.time() - start_time
        results.append({
            "Model": "Cox PH",
            "C-index (train)": metrics_cox['c_index_train'],
            "C-index (val)": metrics_cox['c_index_val'],
            "IBS (val)": metrics_cox['ibs_val'],
            "Time (s)": train_time
        })
    except Exception as e:
        print(f"✗ Failed: {e}")
        import traceback
        traceback.print_exc()
        results.append({"Model": "Cox PH", "C-index (train)": "Error", "C-index (val)": "Error",
                        "IBS (val)": "Error", "Time (s)": "-"})

    # 2. XGBoost Survival
    print("XGBoost")
    start_time = time.time()
    try:
        model_xgb = train_xgboost_model(X_train_sk, y_train_sk)
        metrics_xgb = evaluate_xgboost_model(model_xgb, X_train_sk, X_val_sk, y_train_sk, y_val_sk)
        train_time = time.time() - start_time
        results.append({
            "Model": "XGBoost",
            "C-index (train)": metrics_xgb['c_index_train'],
            "C-index (val)": metrics_xgb['c_index_val'],
            "IBS (val)": metrics_xgb['ibs_val'],
            "Time (s)": train_time
        })
    except Exception as e:
        print(f"✗ Failed: {e}")
        import traceback
        traceback.print_exc()
        results.append({"Model": "XGBoost", "C-index (train)": "Error", "C-index (val)": "Error",
                        "IBS (val)": "Error", "Time (s)": "-"})

    # 3. DeepSurv
    print("DeepSurv")
    start_time = time.time()
    try:
        model_ds, device = train_deepsurv_model(X_train, t_train, e_train, epochs=100)
        metrics_ds = evaluate_deepsurv_model(model_ds, device, X_train, X_val, t_train, t_val, e_train, e_val)
        train_time = time.time() - start_time
        results.append({
            "Model": "DeepSurv",
            "C-index (train)": metrics_ds['c_index_train'],
            "C-index (val)": metrics_ds['c_index_val'],
            "IBS (val)": metrics_ds['ibs_val'],
            "Time (s)": train_time
        })
    except Exception as e:
        print(f"✗ Failed: {e}")
        import traceback
        traceback.print_exc()
        results.append({"Model": "DeepSurv", "C-index (train)": "Error", "C-index (val)": "Error",
                        "IBS (val)": "Error", "Time (s)": "-"})

    # 4. Random Survival Forest
    print("RSF")
    start_time = time.time()
    try:
        model_rsf = train_rsf_model(X_train_sk, y_train_sk, n_estimators=200)
        metrics_rsf = evaluate_rsf_model(model_rsf, X_train_sk, X_val_sk, y_train_sk, y_val_sk)
        train_time = time.time() - start_time
        results.append({
            "Model": "RSF",
            "C-index (train)": metrics_rsf['c_index_train'],
            "C-index (val)": metrics_rsf['c_index_val'],
            "IBS (val)": metrics_rsf['ibs_val'],
            "Time (s)": train_time
        })
    except Exception as e:
        print(f"✗ Failed: {e}")
        import traceback
        traceback.print_exc()
        results.append({"Model": "RSF", "C-index (train)": "Error", "C-index (val)": "Error",
                        "IBS (val)": "Error", "Time (s)": "-"})

    return results

if __name__ == "__main__":
    raw_results = baselines_evaluate_embeddings(embeddings_flag=None, dataset='SEER')
    #emb_results = baselines_evaluate_embeddings(embeddings_flag="emb", dataset='SEER')
    #dummy_results = baselines_evaluate_embeddings(embeddings_flag="dummy", dataset='SEER')
    #combi_results = baselines_evaluate_embeddings(embeddings_flag="combi", dataset='SEER')

    # Combine results with a method column
    def combine_results(results_dicts, method_name):
        df = pd.DataFrame(results_dicts)
        df["Method"] = method_name
        return df

    all_results = pd.concat([
        combine_results(raw_results, "Raw"),
        #combine_results(emb_results, "TARTE Embeddings"),
        #combine_results(dummy_results, "TARTE Embeddings with dummy y"),
        #combine_results(combi_results, "TARTE Embeddings with time and event combined")
    ])

    # Define metrics
    metrics = ["C-index (train)", "C-index (val)", "IBS (val)", "Time (s)"]

    # extract raw rows
    df_raw = all_results[all_results["Method"] == "Raw"].set_index("Model")

    # overwrite column in-place
    for metric in metrics:
        all_results[f"{metric} to Baseline"] = all_results.apply(
            lambda row: row[metric] - df_raw.loc[row["Model"], metric]
            if row["Method"] != "Raw" else 0.0,
            axis=1
        )

    plot_results_absolute(all_results)
    plot_results_relative(all_results)