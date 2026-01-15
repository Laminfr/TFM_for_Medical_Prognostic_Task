import sys
from datasets import datasets
from experiments.experiment import NFGExperiment, CoxExperiment

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

param_grid_cox = {
    "penalizer": [0.0, 0.01, 0.1, 1.0],
}

CoxExperiment.create(
    param_grid_cox,
    n_iter=grid_search,
    path="Results/{}_cox".format(dataset),
    random_seed=random_seed,
    fold=fold
).train(x, t, e, covariates)
