if __name__ == "__main__":
    import sys
    from datasets import datasets
    from experiments.experiment import DeepSurvExperiment

    random_seed = 0

    # Open dataset
    dataset = sys.argv[1]  # METABRIC, SUPPORT, PBC

    # Specific fold selection
    fold = None
    if len(sys.argv) == 3:
        fold = int(sys.argv[2])

    print("Script running experiments on ", dataset)
    x, t, e, covariates = datasets.load_dataset(dataset, competing=False)

    # Hyperparameters
    grid_search = 100
    max_epochs = 1000
    batch = [100, 250]

    param_grid_deepsurv = {
        'layers': [[50], [50, 50], [100, 100]],
        'learning_rate': [1e-3, 1e-4],
        'dropout': [0.0, 0.25, 0.5],
        'epochs': [500],
        'batch': [256],
        'patience_max': [5]
    }

    experiment = DeepSurvExperiment.create(
        param_grid_deepsurv,
        n_iter=grid_search,
        path="Results/{}_raw_deepsurv".format(dataset),
        random_seed=random_seed,
        fold=fold,
    )

    experiment.train(x, t, e, covariates)

    print(
        "*********************this is evaluations parameters : *********************************************\n"
    )
    print(
        "*********************this is evaluations parameters : *********************************************\n"
    )
