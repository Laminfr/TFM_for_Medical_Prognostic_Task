import sys
from datasets import datasets
from experiments.experiment import NFGExperiment

random_seed = 0

# Open dataset
dataset = sys.argv[1] # METABRIC, SUPPORT, PBC

# Specific fold selection
fold = None
if len(sys.argv) == 3:
    fold = int(sys.argv[2])

print("Script running experiments on ", dataset)
x, t, e, covariates = datasets.load_dataset(dataset, competing = False) 

import numpy as np
if np.max(e) > 1:               # datasets with competing risks (e.g., PBC)
    e = (e == 1).astype(int)    # keep primary event as 1, others → 0

# Hyperparameters
max_epochs = 1000
grid_search = 100
layers = [[i] * (j + 1) for i in [25, 50] for j in range(4)]
layers_large = [[i] * (j + 1) for i in [25, 50] for j in range(8)]

batch = [100, 250]

# NFG single risk
param_grid = {
    'epochs': [max_epochs],
    'learning_rate' : [1e-3, 1e-4],
    'batch': batch,
    
    'dropout': [0., 0.25, 0.5, 0.75],

    'layers_surv': layers,
    'layers' : layers,
    'act': ['Tanh'],
}


NFGExperiment.create(param_grid, 
                     n_iter = grid_search, 
                     path = 'Results/{}_nfg'.format(dataset), 
                     random_seed = random_seed, 
                     fold = fold).train(x, t, e)