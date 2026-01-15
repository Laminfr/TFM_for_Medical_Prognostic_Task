"""
Survival Stacking: Discrete-Time Survival Analysis

This module implements a survival stacking approach that transforms
continuous survival analysis into a binary classification problem.

Key Components:
- DiscreteTimeTransformer: Converts data to person-period format
- SurvivalStackingModel: XGBoost-based survival prediction
- TabICLBinaryClassifier: Direct TabICL binary classification (no embeddings)
- Evaluation utilities for C-Index and IBS

Classifier options for SurvivalStackingModel:
- 'xgboost': XGBoost gradient boosting (default)
- 'lightgbm': LightGBM gradient boosting
- 'logistic': Logistic regression
- 'tabicl': TabICL in-context learning (direct classification, no embeddings)

Reference:
- Tutz & Schmid (2016): Modeling Discrete Time-to-Event Data
- Kvamme et al. (2019): Time-to-Event Prediction with Neural Networks
"""

from .discrete_time import DiscreteTimeTransformer
from .stacking_model import SurvivalStackingModel
from .evaluation import compute_survival_metrics

# Optional TabICL classifier import
try:
    from .tabicl_classifier import TabICLBinaryClassifier, TabICLBatchClassifier
    __all__ = [
        'DiscreteTimeTransformer',
        'SurvivalStackingModel', 
        'compute_survival_metrics',
        'TabICLBinaryClassifier',
        'TabICLBatchClassifier'
    ]
except ImportError:
    __all__ = [
        'DiscreteTimeTransformer',
        'SurvivalStackingModel', 
        'compute_survival_metrics'
    ]
