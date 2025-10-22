"""
ML Models Submodule

Traditional machine learning models for trading:
- regime_classifier.py: Market regime detection (GMM + XGBoost)
- volatility_har.py: HAR volatility forecasting
- volatility_garch.py: GARCH models (future)
"""

from underdog.ml.models.regime_classifier import RegimeSwitchingModel

__all__ = [
    'RegimeSwitchingModel',
]
