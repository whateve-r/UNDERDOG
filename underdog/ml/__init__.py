"""
Machine Learning Components for UNDERDOG

Deep Reinforcement Learning (DRL) architecture for adaptive trading:
- Regime-Switching Classification (GMM + XGBoost)
- DRL Agents (TD3, PPO) for signal integration
- Sentiment Analysis (FinGPT, FinBERT)
- Feature Engineering

Papers:
- arXiv:2510.04952v2 (Safe Execution)
- arXiv:2510.10526v1 (LLM + RL)
- arXiv:2510.03236v1 (Regime-Switching)
"""

__all__ = [
    'RegimeSwitchingModel',
    'TradingDRLAgent',
    'FinGPTSentimentAnalyzer',
    'TradingEnvironment',
]
