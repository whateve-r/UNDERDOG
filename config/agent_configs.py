"""
UNDERDOG TRADING SYSTEMS
Heterogeneous MARL Agent Configuration System

This module provides asset-specific DRL configurations for the HMARL architecture.
Each configuration is optimized for the microstructure and dynamics of its target market.

Architecture Strategy:
- EURUSD: TD3 + LSTM (trend-following, temporal dependencies)
- USDJPY: PPO + CNN1D (breakout detection, policy stability)
- XAUUSD: SAC + Transformer (regime changes, high exploration)
- GBPUSD: DDPG + Attention (momentum trading, whipsaw resistance)
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Literal

# ==============================================================================
# BASE CONFIGURATION
# ==============================================================================

@dataclass
class AgentConfig:
    """Base configuration for all DRL agents"""
    
    # State/Action space
    state_dim: int = 31  # Financial Intelligence: 29 base + 2 staleness features (overridden by trainer)
    action_dim: int = 1  # Will be overridden by trainer
    
    # Core hyperparameters
    lr_actor: float = 3e-4
    lr_critic: float = 3e-4
    gamma: float = 0.99
    tau: float = 0.005
    batch_size: int = 256
    buffer_size: int = 1_000_000
    
    # Network architecture
    hidden_dims: tuple = (256, 256)
    hidden_dim: int = 256  # For compatibility with existing agents
    activation: str = 'relu'
    architecture: Literal['MLP', 'LSTM', 'CNN1D', 'Transformer', 'Attention'] = 'MLP'
    
    # Training dynamics
    warmup_steps: int = 10_000
    update_freq: int = 1
    gradient_clip: float = 1.0
    
    # Device
    device: str = 'cuda' if __import__('torch').cuda.is_available() else 'cpu'
    
    # Additional params (can be overridden)
    extras: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'lr_actor': self.lr_actor,
            'lr_critic': self.lr_critic,
            'gamma': self.gamma,
            'tau': self.tau,
            'batch_size': self.batch_size,
            'buffer_size': self.buffer_size,
            'hidden_dims': self.hidden_dims,
            'activation': self.activation,
            'architecture': self.architecture,
            'warmup_steps': self.warmup_steps,
            'update_freq': self.update_freq,
            'gradient_clip': self.gradient_clip,
            **self.extras
        }


# ==============================================================================
# TD3 CONFIGURATION (EURUSD - TREND FOLLOWING)
# ==============================================================================

@dataclass
class TD3Config(AgentConfig):
    """
    Twin Delayed DDPG Configuration
    
    Optimized for: EURUSD (low volatility, strong trends, mean-reversion)
    Architecture: LSTM for temporal pattern recognition
    
    Key Features:
    - Policy delay to reduce variance
    - Target policy smoothing for robustness
    - Sequential state processing (60-120 steps history)
    """
    
    # TD3-specific
    policy_delay: int = 2
    policy_noise: float = 0.2
    noise_clip: float = 0.5
    
    # Architecture
    architecture: Literal['LSTM'] = 'LSTM'
    hidden_dims: tuple = (256, 256)
    hidden_dim: int = 256  # For compatibility with TD3Agent
    
    # LSTM-specific
    sequence_length: int = 60  # 60 candles lookback for trend detection
    lstm_layers: int = 2
    lstm_hidden_size: int = 128
    dropout: float = 0.1
    
    # Training
    lr_actor: float = 1e-4  # Lower LR for stable EURUSD
    lr_critic: float = 3e-4
    gamma: float = 0.99
    
    # Exploration
    exploration_noise: float = 0.1  # Conservative for low volatility
    expl_noise: float = 0.1  # Alias for compatibility with TD3Agent
    
    # TD3 specific parameters for compatibility
    policy_freq: int = 2  # Same as policy_delay
    use_cbrl: bool = False  # Chaos-based RL disabled for LSTM
    
    def __post_init__(self):
        self.extras = {
            'policy_delay': self.policy_delay,
            'policy_noise': self.policy_noise,
            'noise_clip': self.noise_clip,
            'sequence_length': self.sequence_length,
            'lstm_layers': self.lstm_layers,
            'lstm_hidden_size': self.lstm_hidden_size,
            'dropout': self.dropout,
            'exploration_noise': self.exploration_noise,
        }


# ==============================================================================
# PPO CONFIGURATION (USDJPY - BREAKOUT TRADING)
# ==============================================================================

@dataclass
class PPOConfig(AgentConfig):
    """
    Proximal Policy Optimization Configuration
    
    Optimized for: USDJPY (moderate volatility, frequent breakouts)
    Architecture: CNN1D for local pattern detection
    
    Key Features:
    - Clipped surrogate objective for stability
    - Sharpe ratio reward shaping
    - Convolutional feature extraction for breakout patterns
    """
    
    # PPO-specific
    clip_ratio: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01  # Moderate exploration
    max_grad_norm: float = 0.5
    
    # Architecture
    architecture: Literal['CNN1D'] = 'CNN1D'
    hidden_dims: tuple = (256, 128)
    hidden_dim: int = 256  # For compatibility
    
    # CNN-specific
    conv_channels: tuple = (32, 64, 128)  # Progressive feature extraction
    kernel_sizes: tuple = (5, 3, 3)  # Detect patterns at different scales
    pooling: str = 'max'
    
    # Training
    lr_actor: float = 3e-4
    lr_critic: float = 1e-3  # Higher for value estimation stability
    gamma: float = 0.99
    gae_lambda: float = 0.95  # GAE for advantage estimation
    
    # PPO-specific training
    ppo_epochs: int = 10
    num_minibatches: int = 4
    
    # Reward shaping
    reward_type: str = 'sharpe'  # Sharpe ratio instead of raw PnL
    sharpe_window: int = 50  # Rolling window for Sharpe calculation
    
    def __post_init__(self):
        self.extras = {
            'clip_ratio': self.clip_ratio,
            'value_coef': self.value_coef,
            'entropy_coef': self.entropy_coef,
            'max_grad_norm': self.max_grad_norm,
            'conv_channels': self.conv_channels,
            'kernel_sizes': self.kernel_sizes,
            'pooling': self.pooling,
            'gae_lambda': self.gae_lambda,
            'ppo_epochs': self.ppo_epochs,
            'num_minibatches': self.num_minibatches,
            'reward_type': self.reward_type,
            'sharpe_window': self.sharpe_window,
        }


# ==============================================================================
# SAC CONFIGURATION (XAUUSD - REGIME ADAPTATION)
# ==============================================================================

@dataclass
class SACConfig(AgentConfig):
    """
    Soft Actor-Critic Configuration
    
    Optimized for: XAUUSD (high volatility, regime changes, safe-haven flows)
    Architecture: Transformer for long-range dependencies
    
    Key Features:
    - Maximum entropy RL for robust exploration
    - Adaptive temperature coefficient
    - Transformer attention for regime detection
    """
    
    # SAC-specific
    alpha: float = 0.2  # Initial entropy coefficient
    auto_alpha: bool = True  # Adaptive temperature
    target_entropy: float = -4.0  # -dim(action_space) - will be set dynamically
    
    # Architecture
    architecture: Literal['Transformer'] = 'Transformer'
    hidden_dims: tuple = (512, 256)  # Larger capacity for complex patterns
    hidden_dim: int = 512  # For compatibility
    
    # Transformer-specific
    num_heads: int = 8
    num_layers: int = 4
    d_model: int = 256
    d_ff: int = 1024
    sequence_length: int = 120  # Longer context for regime detection
    dropout: float = 0.1
    
    # Training
    lr_actor: float = 3e-4
    lr_critic: float = 3e-4
    lr_alpha: float = 3e-4  # Learning rate for temperature
    gamma: float = 0.99
    
    # Exploration (critical for XAUUSD)
    exploration_bonus: float = 0.1  # Additional entropy bonus
    
    # Reward shaping
    reward_type: str = 'exploration_enhanced'  # Bonus for state diversity
    
    def __post_init__(self):
        self.extras = {
            'alpha': self.alpha,
            'auto_alpha': self.auto_alpha,
            'target_entropy': self.target_entropy,
            'num_heads': self.num_heads,
            'num_layers': self.num_layers,
            'd_model': self.d_model,
            'd_ff': self.d_ff,
            'sequence_length': self.sequence_length,
            'dropout': self.dropout,
            'lr_alpha': self.lr_alpha,
            'exploration_bonus': self.exploration_bonus,
            'reward_type': self.reward_type,
        }


# ==============================================================================
# DDPG CONFIGURATION (GBPUSD - MOMENTUM TRADING)
# ==============================================================================

@dataclass
class DDPGConfig(AgentConfig):
    """
    Deep Deterministic Policy Gradient Configuration
    
    Optimized for: GBPUSD (high volatility, whipsaws, false breakouts)
    Architecture: Attention mechanism for feature weighting
    
    Key Features:
    - Deterministic policy for decisive actions
    - Attention mechanism to filter noise
    - Asymmetric reward with whipsaw penalty
    """
    
    # DDPG-specific
    ou_noise_theta: float = 0.15  # Ornstein-Uhlenbeck noise
    ou_noise_sigma: float = 0.2
    
    # Architecture
    architecture: Literal['Attention'] = 'Attention'
    hidden_dims: tuple = (256, 256)
    hidden_dim: int = 256  # For compatibility
    
    # Attention-specific
    attention_heads: int = 4
    attention_dim: int = 128
    attention_dropout: float = 0.1
    
    # Training
    lr_actor: float = 1e-4
    lr_critic: float = 3e-4
    gamma: float = 0.99
    
    # Exploration decay
    noise_decay: float = 0.9995  # Aggressive decay for volatile market
    noise_min: float = 0.05
    
    # Reward shaping (whipsaw resistance)
    reward_type: str = 'asymmetric'
    whipsaw_penalty: float = 2.0  # 2x penalty for false breakouts
    hold_bonus: float = 0.1  # Bonus for position conviction
    
    def __post_init__(self):
        self.extras = {
            'ou_noise_theta': self.ou_noise_theta,
            'ou_noise_sigma': self.ou_noise_sigma,
            'attention_heads': self.attention_heads,
            'attention_dim': self.attention_dim,
            'attention_dropout': self.attention_dropout,
            'noise_decay': self.noise_decay,
            'noise_min': self.noise_min,
            'reward_type': self.reward_type,
            'whipsaw_penalty': self.whipsaw_penalty,
            'hold_bonus': self.hold_bonus,
        }


# ==============================================================================
# AGENT MAPPING SYSTEM
# ==============================================================================

AGENT_MAPPING = {
    'EURUSD': {
        'algorithm': 'TD3',
        'config_class': TD3Config,
        'architecture': 'LSTM',
        'description': 'Trend-following with temporal dependencies',
        'optimal_for': ['low_volatility', 'mean_reversion', 'trending_markets'],
    },
    'USDJPY': {
        'algorithm': 'PPO',
        'config_class': PPOConfig,
        'architecture': 'CNN1D',
        'description': 'Breakout detection with policy stability',
        'optimal_for': ['moderate_volatility', 'breakouts', 'sharp_ratio_focus'],
    },
    'XAUUSD': {
        'algorithm': 'SAC',
        'config_class': SACConfig,
        'architecture': 'Transformer',
        'description': 'Regime adaptation with maximum entropy',
        'optimal_for': ['high_volatility', 'regime_changes', 'exploration_critical'],
    },
    'GBPUSD': {
        'algorithm': 'DDPG',
        'config_class': DDPGConfig,
        'architecture': 'Attention',
        'description': 'Momentum trading with whipsaw resistance',
        'optimal_for': ['high_volatility', 'false_breakouts', 'noise_filtering'],
    },
}


def get_agent_config(symbol: str) -> AgentConfig:
    """
    Factory function to create asset-specific agent configuration
    
    Args:
        symbol: Trading symbol (EURUSD, USDJPY, XAUUSD, GBPUSD)
        
    Returns:
        AgentConfig instance optimized for the symbol
        
    Raises:
        ValueError: If symbol not in AGENT_MAPPING
    """
    if symbol not in AGENT_MAPPING:
        raise ValueError(
            f"Symbol {symbol} not configured. Available: {list(AGENT_MAPPING.keys())}"
        )
    
    mapping = AGENT_MAPPING[symbol]
    config_class = mapping['config_class']
    
    return config_class()


def get_agent_metadata(symbol: str) -> Dict[str, Any]:
    """
    Get full metadata for an agent including configuration
    
    Args:
        symbol: Trading symbol
        
    Returns:
        Dictionary with algorithm, config, architecture, and description
    """
    if symbol not in AGENT_MAPPING:
        raise ValueError(
            f"Symbol {symbol} not configured. Available: {list(AGENT_MAPPING.keys())}"
        )
    
    mapping = AGENT_MAPPING[symbol].copy()
    mapping['config'] = mapping['config_class']()
    
    return mapping


def print_agent_summary():
    """Print summary of all configured agents"""
    print("\n" + "="*80)
    print("UNDERDOG TRADING SYSTEMS - HETEROGENEOUS MARL ARCHITECTURE")
    print("="*80)
    
    for symbol, mapping in AGENT_MAPPING.items():
        print(f"\n[{symbol}]")
        print(f"  Algorithm:    {mapping['algorithm']}")
        print(f"  Architecture: {mapping['architecture']}")
        print(f"  Strategy:     {mapping['description']}")
        print(f"  Optimal For:  {', '.join(mapping['optimal_for'])}")
    
    print("\n" + "="*80)


if __name__ == '__main__':
    # Demo: Print configuration summary
    print_agent_summary()
    
    # Demo: Get specific config
    print("\nExample: EURUSD TD3 Configuration")
    eurusd_config = get_agent_config('EURUSD')
    print(f"LR Actor: {eurusd_config.lr_actor}")
    print(f"Architecture: {eurusd_config.architecture}")
    print(f"Sequence Length: {eurusd_config.extras.get('sequence_length')}")
