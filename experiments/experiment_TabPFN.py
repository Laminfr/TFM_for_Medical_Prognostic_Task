import time
import warnings
import sys

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", message="TypedStorage is deprecated")

from ..datasets.data_loader import load_and_preprocess_data
from ..coxph.utilities import train_cox_model, evaluate_model as evaluate_cox
from ..xgboost.utilities import train_xgboost_model, evaluate_xgboost_model
from ..deepsurv.utilities import train_deepsurv_model, evaluate_deepsurv_model
from ..rsf.utilities import train_rsf_model, evaluate_rsf_model
from ..tfm.TabPFN.extract_embeddings import get_embeddings_tabpfn

def find_repo_root():
    from pathlib import Path
    start = start or Path(__file__).resolve()
    for parent in [start] + list(start.parents):
        if (parent / ".git").exists():
            return parent
    raise FileNotFoundError("No .git folder found in any parent directory")

sys.path.append(find_repo_root())
sys.path.append(find_repo_root()+ "/DeepSurvivalMachines/")




def plot_results_absolute(df_all, results_path):

    sns.set_theme(style="whitegrid", font_scale=1.2)

    # --- Plot C-index (train) ---
    plt.figure(figsize=(8, 5))
    sns.barplot(data=df_all, x="Model", y="C-index (train)", hue="Method")
    plt.title("Train C-index Comparison (Raw vs TabPFN)")
    plt.legend(title="Method", framealpha=0.0)
    plt.tight_layout()
    plt.savefig(results_path + "/tabpfn_C_train.png")

    # --- Plot C-index (val) ---
    plt.figure(figsize=(8, 5))
    sns.barplot(data=df_all, x="Model", y="C-index (val)", hue="Method")
    plt.title("Validation C-index Comparison (Raw vs TabPFN)")
    plt.legend(title="Method", framealpha=0.0)
    plt.tight_layout()
    plt.savefig(results_path + "/tabpfn_C_val.png")

    # --- Plot IBS (val) ---
    plt.figure(figsize=(8, 5))
    sns.barplot(data=df_all, x="Model", y="IBS (val)", hue="Method")
    plt.title("Validation IBS Comparison (lower is better)")
    plt.legend(title="Method", framealpha=0.0)
    plt.tight_layout()
    plt.savefig(results_path + "/tabpfn_IBS.png")

    # --- Plot training time ---
    plt.figure(figsize=(8, 5))
    sns.barplot(data=df_all, x="Model", y="Time (s)", hue="Method")
    plt.title("Training Time by Method (Raw vs TabPFN)")
    plt.legend(title="Method", framealpha=0.0)
    plt.tight_layout()
    plt.savefig(results_path + "/tabpfn_Time.png")


def plot_results_relative(df_all, results_path):
    sns.set_theme(style="whitegrid", font_scale=1.2)

    # --- Plot C-index (train) diff ---
    plt.figure(figsize=(8, 5))
    sns.barplot(
        data=df_all,
        x="Model",
        y="C-index (train) to Baseline",
        hue="Method",
    )
    plt.title("Train C-index Difference to Raw Baseline")
    plt.legend(title="Method", framealpha=0.0)
    plt.tight_layout()
    plt.savefig(results_path + "/tabpfn_C_train_rel.png")

    # --- Plot C-index (val) diff ---
    plt.figure(figsize=(8, 5))
    sns.barplot(
        data=df_all,
        x="Model",
        y="C-index (val) to Baseline",
        hue="Method",
    )
    plt.title("Validation C-index Difference to Raw Baseline")
    plt.legend(title="Method", framealpha=0.0)
    plt.tight_layout()
    plt.savefig(results_path + "/tabpfn_C_val_rel.png")

    # --- Plot IBS (val) diff ---
    plt.figure(figsize=(8, 5))
    sns.barplot(
        data=df_all,
        x="Model",
        y="IBS (val) to Baseline",
        hue="Method",
    )
    plt.title("Validation IBS Difference to Raw Baseline (negative is better)")
    plt.legend(title="Method", framealpha=0.0)
    plt.tight_layout()
    plt.savefig(results_path + "/tabpfn_IBS_rel.png")

    # --- Plot training time diff ---
    plt.figure(figsize=(8, 5))
    sns.barplot(
        data=df_all,
        x="Model",
        y="Time (s) to Baseline",
        hue="Method",
    )
    plt.title("Training Time Difference to Raw Baseline")
    plt.legend(title="Method", framealpha=0.0)
    plt.tight_layout()
    plt.savefig(results_path + "/tabpfn_Time_rel.png")


def baselines_evaluate_tabpfn(
    dataset="METABRIC",
    normalize=True,
    test_size=0.2,
    random_state=42,
    use_tabpfn_embeddings=False,
):
    """
    Run all baseline survival models either on:
        - raw data (use_tabpfn_embeddings=False), or
        - TabPFN embeddings (use_tabpfn_embeddings=True).

    Returns:
        list[dict]: one dict per model with metrics and training time.
    """
    print("= " * 70)
    mode_str = "TabPFN EMBEDDINGS" if use_tabpfn_embeddings else "RAW FEATURES"
    print(f"BASELINE COMPARISON ON {dataset} DATASET ({mode_str})")
    print("= " * 70)

    # Load data (standard format)
    X_train, X_val, t_train, t_val, e_train, e_val = load_and_preprocess_data(
        dataset=dataset,
        normalize=normalize,
        test_size=test_size,
        random_state=random_state,
    )

    # Load data for scikit-survival compatible targets (for XGB & RSF)
    X_train_sk, X_val_sk, y_train_sk, y_val_sk = load_and_preprocess_data(
        dataset=dataset,
        normalize=normalize,
        test_size=test_size,
        random_state=random_state,
        as_sksurv_y=True,
    )

    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    print(f"Features: {X_train.shape[1]}")
    print(f"Event rate (train): {e_train.mean():.2%}")
    print(f"Event rate (val):   {e_val.mean():.2%}")

    # ------------------------------------------------------------------
    # Replace X by TabPFN embeddings if requested
    # ------------------------------------------------------------------
    if use_tabpfn_embeddings:
        print("\n>>> Using TabPFN embeddings (time+event pseudo-label)\n")
        print("Extract TabPFN embeddings ...")
        X_train_emb, X_val_emb = get_embeddings_tabpfn(X_train, X_val, t_train, e_train)
        # Overwrite features for all models with embeddings
        X_train, X_val = X_train_emb, X_val_emb
        X_train_sk, X_val_sk = X_train_emb, X_val_emb
        print("Run baselines on TabPFN embeddings.")
    else:
        print("\n>>> Using raw features for baseline predictions\n")

    results = []

    # ------------------------------------------------------------------
    # 1. Cox Proportional Hazards
    # ------------------------------------------------------------------
    start_time = time.time()
    try:
        cph = train_cox_model(X_train, t_train, e_train)
        metrics_cox = evaluate_cox(cph, X_train, X_val, t_train, t_val, e_train, e_val)
        train_time = time.time() - start_time
        results.append(
            {
                "Model": "Cox PH",
                "C-index (train)": metrics_cox["c_index_train"],
                "C-index (val)": metrics_cox["c_index_val"],
                "IBS (val)": metrics_cox["ibs_val"],
                "Time (s)": train_time,
            }
        )
    except Exception as e:
        print(f"✗ Cox PH failed: {e}")
        import traceback

        traceback.print_exc()
        results.append(
            {
                "Model": "Cox PH",
                "C-index (train)": "Error",
                "C-index (val)": "Error",
                "IBS (val)": "Error",
                "Time (s)": "-",
            }
        )

    # ------------------------------------------------------------------
    # 2. XGBoost Survival
    # ------------------------------------------------------------------
    start_time = time.time()
    try:
        model_xgb = train_xgboost_model(X_train_sk, y_train_sk)
        metrics_xgb = evaluate_xgboost_model(
            model_xgb, X_train_sk, X_val_sk, y_train_sk, y_val_sk
        )
        train_time = time.time() - start_time
        results.append(
            {
                "Model": "XGBoost",
                "C-index (train)": metrics_xgb["c_index_train"],
                "C-index (val)": metrics_xgb["c_index_val"],
                "IBS (val)": metrics_xgb["ibs_val"],
                "Time (s)": train_time,
            }
        )
    except Exception as e:
        print(f"✗ XGBoost failed: {e}")
        import traceback

        traceback.print_exc()
        results.append(
            {
                "Model": "XGBoost",
                "C-index (train)": "Error",
                "C-index (val)": "Error",
                "IBS (val)": "Error",
                "Time (s)": "-",
            }
        )

    # ------------------------------------------------------------------
    # 3. DeepSurv
    # ------------------------------------------------------------------
    start_time = time.time()
    try:
        model_ds, device = train_deepsurv_model(X_train, t_train, e_train, epochs=100)
        metrics_ds = evaluate_deepsurv_model(
            model_ds,
            device,
            X_train,
            X_val,
            t_train,
            t_val,
            e_train,
            e_val,
        )
        train_time = time.time() - start_time
        results.append(
            {
                "Model": "DeepSurv",
                "C-index (train)": metrics_ds["c_index_train"],
                "C-index (val)": metrics_ds["c_index_val"],
                "IBS (val)": metrics_ds["ibs_val"],
                "Time (s)": train_time,
            }
        )
    except Exception as e:
        print(f"✗ DeepSurv failed: {e}")
        import traceback

        traceback.print_exc()
        results.append(
            {
                "Model": "DeepSurv",
                "C-index (train)": "Error",
                "C-index (val)": "Error",
                "IBS (val)": "Error",
                "Time (s)": "-",
            }
        )

    # ------------------------------------------------------------------
    # 4. Random Survival Forest
    # ------------------------------------------------------------------
    start_time = time.time()
    try:
        model_rsf = train_rsf_model(X_train_sk, y_train_sk, n_estimators=200)
        metrics_rsf = evaluate_rsf_model(
            model_rsf, X_train_sk, X_val_sk, y_train_sk, y_val_sk
        )
        train_time = time.time() - start_time
        results.append(
            {
                "Model": "RSF",
                "C-index (train)": metrics_rsf["c_index_train"],
                "C-index (val)": metrics_rsf["c_index_val"],
                "IBS (val)": metrics_rsf["ibs_val"],
                "Time (s)": train_time,
            }
        )
    except Exception as e:
        print(f"✗ RSF failed: {e}")
        import traceback

        traceback.print_exc()
        results.append(
            {
                "Model": "RSF",
                "C-index (train)": "Error",
                "C-index (val)": "Error",
                "IBS (val)": "Error",
                "Time (s)": "-",
            }
        )

    return results


if __name__ == "__main__":
    from dotenv import load_dotenv
    import os
    load_dotenv()

    current_dir = (os.path.dirname(os.path.abspath(__file__)))
    results_dir = os.path.join(current_dir, "TabPFN_results")
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    # 1) Raw features
    raw_results = baselines_evaluate_tabpfn(use_tabpfn_embeddings=False)

    # 2) TabPFN embeddings
    tabpfn_results = baselines_evaluate_tabpfn(use_tabpfn_embeddings=True)

    # Helper to convert list-of-dicts to DataFrame with "Method" column
    def combine_results(results_dicts, method_name):
        df = pd.DataFrame(results_dicts)
        df["Method"] = method_name
        return df

    # Combine all results
    all_results = pd.concat(
        [
            combine_results(raw_results, "Raw"),
            combine_results(tabpfn_results, "TabPFN Embeddings"),
        ],
        ignore_index=True,
    )

    # Metrics to compare
    metrics = ["C-index (train)", "C-index (val)", "IBS (val)", "Time (s)"]

    # Extract raw baseline rows and index by model
    df_raw = all_results[all_results["Method"] == "Raw"].set_index("Model")

    # Compute difference to baseline for each metric
    for metric in metrics:
        all_results[f"{metric} to Baseline"] = all_results.apply(
            lambda row: (
                row[metric] - df_raw.loc[row["Model"], metric]
                if row["Method"] != "Raw"
                else 0.0
            ),
            axis=1,
        )

    # Make plots
    plot_results_absolute(all_results, results_dir)
    plot_results_relative(all_results, results_dir)
