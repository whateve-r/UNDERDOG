"""
Deep Reinforcement Learning (DRL) Module

Implements trading agents using state-of-the-art RL algorithms:
- TD3 (Twin Delayed Deep Deterministic Policy Gradient)
- PPO (Proximal Policy Optimization)

Based on papers:
- arXiv:2510.10526v1: LLM + RL Integration (TD3)
- arXiv:2510.04952v2: Safe Execution (Constrained RL)

Components:
- agents.py: TD3, PPO implementations
- environments.py: Trading environment (Gymnasium)
- train_drl.py: Training pipeline
- drl_models.py: Neural network architectures (Actor-Critic)

Model checkpoints: data/models/drl_checkpoints/
"""

__all__ = [
    'TradingDRLAgent',
    'TradingEnvironment',
    'train_drl_agent',
]
