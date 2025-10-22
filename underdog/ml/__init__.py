"""
Machine Learning Components for UNDERDOG

Traditional ML models and feature engineering:
- Regime-Switching Classification (GMM + XGBoost)
- Volatility models (HAR, GARCH)
- Feature Engineering
- Preprocessing pipelines

Note: DRL agents moved to underdog/rl/
Note: Sentiment analysis moved to underdog/sentiment/

Papers:
- arXiv:2510.03236v1 (Regime-Switching)
"""

# Import from new locations for backward compatibility
from underdog.ml.models.regime_classifier import RegimeSwitchingModel

__all__ = [
    'RegimeSwitchingModel',
]
