"""
CompetingRisks Benchmark Package

This package implements a 3-phase benchmark for competing risks analysis:

Phase 1: Survival Stacking V2 (Multi-Class XGBoost)
    - Discrete-time multi-class classification for competing events
    - Uses XGBClassifier with objective='multi:softprob'

Phase 2: Pure NeuralFineGray
    - Deep learning approach optimizing Fine-Gray loss directly
    - Uses the existing NFG implementation

Phase 3: Hybrid (NFG Embeddings + Stacking)
    - Extract embeddings from trained NFG model
    - Feed embeddings to Multi-Class XGBoost

Datasets:
    - SEER: Real-world cancer competing risks data
    - SYNTHETIC_COMPETING: Synthetic competing risks data

Metrics:
    - Cause-Specific C-Index: Discrimination for each cause (truncated_concordance_td)
    - CIF-based IBS: Calibration based on Cumulative Incidence Function
"""

# Use existing modules - no duplicate code
from datasets.datasets import load_dataset
from metrics.discrimination import truncated_concordance_td
from metrics.calibration import integrated_brier_score

from .stacking_multiclass import MultiClassSurvivalStacking
from .nfg_wrapper import NFGCompetingRisks
from .hybrid_model import HybridNFGStacking
from .run_benchmark import run_full_benchmark, run_single_phase
from .utils import get_competing_risks_datasets, split_data, get_evaluation_times

__all__ = [
    # Datasets (from existing datasets module)
    'load_dataset',
    'get_competing_risks_datasets',
    'split_data',
    'get_evaluation_times',
    # Metrics (from existing metrics module)
    'truncated_concordance_td',
    'integrated_brier_score',
    # Models
    'MultiClassSurvivalStacking',
    'NFGCompetingRisks', 
    'HybridNFGStacking',
    # Benchmark
    'run_full_benchmark',
    'run_single_phase',
]
