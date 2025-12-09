if __name__ == "__main__":
    import sys
    from datasets import datasets
    from experiments.experiment import CoxPH_TabPFN_embeddings
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
        "penalizer": [0.0, 0.01, 0.1, 1.0],
        'epochs': [max_epochs],
        'learning_rate' : [1e-3, 1e-4],
        'batch': batch,
    }


    experiment = CoxPH_TabPFN_embeddings.create(
        param_grid_cox,
        n_iter=grid_search,
        path="Results/{}_TabPFN_cox_nfg".format(dataset),
        random_seed=random_seed,
        fold=fold
    )

    experiment.train(x, t, e, covariates)  



    print ("*********************this is evaluations parameters : *********************************************\n")
    # results = experiment.get_eval_metrics()
    # c_index_train = [d["c_index_train"] for d in results]
    # c_index_val   = [d["c_index_val"]   for d in results]
    # ibs_val       = [d["ibs_val"]       for d in results]

    # print("=== Model Metrics ===")
    # for i, (c_tr, c_val, ibs) in enumerate(zip(c_index_train, c_index_val, ibs_val), 1):
    #     print(f"Model {i:2d} | C-index Train: {c_tr:.4f} | C-index Val: {c_val:.4f} | IBS Val: {ibs:.4f}")

    print ("*********************this is evaluations parameters : *********************************************\n")