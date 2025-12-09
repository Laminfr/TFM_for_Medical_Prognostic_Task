if __name__ == "__main__":
    import sys
    from datasets import datasets
    from experiments.experiment import XGBoost_TabPFN_embeddings_Experiment
    random_seed = 0

    # Open dataset
    dataset = sys.argv[1] # METABRIC, SUPPORT, PBC

    # Specific fold selection
    fold = None
    if len(sys.argv) == 3:
        fold = int(sys.argv[2])

    print("Script running experiments on ", dataset)
    x, t, e, covariates = datasets.load_dataset(dataset, competing = False) 

    # Hyperparameters
    grid_search = 100
    max_epochs = 1000
    batch = [100, 250]

    param_grid_cox = {
        "n_estimators" : [100, 200],
        "learning_rate" :[0.05, 0.1],
        "max_depth" : [4, 6, 8],
        "subsample" : [0.8],
        "colsample_bytree" : [0.8],
        "random_state" : [42],
        'min_child_weight': [5, 10],
        "epochs": [max_epochs],
        "batch": batch,
    }

    experiment = XGBoost_TabPFN_embeddings_Experiment.create(
        param_grid_cox,
        n_iter=grid_search,
        path="Results/{}_TabPFN_rsf_nfg".format(dataset),
        random_seed=random_seed,
        fold=fold
    )

    experiment.train(x, t, e, covariates)  



    print ("*********************this is evaluations parameters : *********************************************\n")
    results = experiment.get_eval_metrics()
    none_report = []
    print("\n=== Model Metrics ===\n")

    for i, d in enumerate(results, 1):
        c_tr = d.get("c_index_train")
        c_val = d.get("c_index_val")
        c_pre = d.get("c_index_predict")
        ibs   = d.get("ibs_val")

        metrics = {
            "C-index Train": c_tr,
            "C-index Val": c_val,
            "C-index Predict": c_pre,
            "IBS Val": ibs
        }

        none_fields = [name for name, val in metrics.items() if val is None]
        if none_fields:
            none_report.append((i, none_fields))

        formatted_parts = []
        for name, val in metrics.items():
            if val is None:
                continue
            formatted_parts.append(f"{name}: {val:.4f}")

        if not formatted_parts:
            formatted_parts.append("NO METRICS AVAILABLE (all None)")

        print(f"Model {i:2d} | " + " | ".join(formatted_parts))

    print("\n=== None Value Report ===")
    if not none_report:
        print("No None values found. All metrics are valid.")
    else:
        for model_idx, fields in none_report:
            print(f"Model {model_idx}: {', '.join(fields)} were None")


    print ("*********************this is evaluations parameters : *********************************************\n")