"""
Survival Stacking: Discrete-Time Survival Analysis

This module implements a survival stacking approach that transforms
continuous survival analysis into a binary classification problem.

Key Components:
- DiscreteTimeTransformer: Converts data to person-period format
- SurvivalStackingModel: XGBoost-based survival prediction
- Evaluation utilities for C-Index and IBS

Reference:
- Tutz & Schmid (2016): Modeling Discrete Time-to-Event Data
- Kvamme et al. (2019): Time-to-Event Prediction with Neural Networks
"""

from .discrete_time import DiscreteTimeTransformer
from .stacking_model import SurvivalStackingModel
from .evaluation import compute_survival_metrics

__all__ = [
    'DiscreteTimeTransformer',
    'SurvivalStackingModel', 
    'compute_survival_metrics'
]
