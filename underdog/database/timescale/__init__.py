"""
TimescaleDB Integration Module

High-performance time-series database for trading data.

Features:
- Hypertables for OHLCV, sentiment, macro indicators
- Continuous aggregates (1min → 5min → 1h → 1d)
- Retention policies (5 years)
- Compression (automatic)

Components:
- timescale_connector.py: Database connection + schema management
- timescale_ingestion.py: Async data ingestion pipelines

Schema:
- ohlcv_data: Price data from MT5
- sentiment_scores: Reddit + LLM sentiment
- macro_indicators: FRED data (DFF, VIX, etc)
- regime_predictions: Regime classifier output
"""

__all__ = [
    'TimescaleConnector',
    'TimescaleIngestion',
]
