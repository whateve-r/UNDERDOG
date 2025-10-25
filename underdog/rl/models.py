"""
UNDERDOG TRADING SYSTEMS
Neural Network Architectures for Heterogeneous MARL

This module implements specialized neural network architectures for different
asset classes and market dynamics:

- LSTM: Temporal dependencies (EURUSD trend-following)
- CNN1D: Local pattern detection (USDJPY breakouts)
- Transformer: Long-range dependencies (XAUUSD regime changes)
- Attention: Feature weighting (GBPUSD noise filtering)

All architectures are compatible with TD3, PPO, SAC, and DDPG agents.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional
import math


# ==============================================================================
# LSTM ARCHITECTURE (EURUSD - TREND FOLLOWING)
# ==============================================================================

class LSTMPolicy(nn.Module):
    """
    LSTM-based Policy Network
    
    Optimized for: EURUSD (low volatility, strong trends, temporal dependencies)
    
    Architecture:
        Input: State sequence (batch, seq_len, state_dim)
        LSTM: 2 layers with hidden_size=128
        Output: Action (batch, action_dim)
    
    Key Features:
    - Captures temporal patterns in FX trends
    - Bidirectional processing for context
    - Dropout for regularization
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.1,
        max_action: float = 1.0
    ):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.max_action = max_action
        
        # LSTM for temporal processing
        self.lstm = nn.LSTM(
            input_size=state_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=False  # Causal for trading
        )
        
        # Policy head
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, action_dim),
            nn.Tanh()
        )
    
    def forward(self, state: torch.Tensor, hidden: Optional[Tuple] = None) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            state: (batch, seq_len, state_dim) or (batch, state_dim)
            hidden: Optional LSTM hidden state
        
        Returns:
            action: (batch, action_dim) in [-max_action, max_action]
        """
        # Handle single timestep input
        if state.dim() == 2:
            state = state.unsqueeze(1)  # (batch, 1, state_dim)
        
        # LSTM processing
        lstm_out, _ = self.lstm(state, hidden)  # (batch, seq_len, hidden_size)
        
        # Take last timestep output
        last_output = lstm_out[:, -1, :]  # (batch, hidden_size)
        
        # Policy head
        action = self.fc(last_output)
        
        return self.max_action * action


class LSTMCritic(nn.Module):
    """
    LSTM-based Critic Network (Q-function)
    
    Twin Q-networks for TD3/SAC compatibility
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM for state processing
        self.lstm = nn.LSTM(
            input_size=state_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Q1 network
        self.q1 = nn.Sequential(
            nn.Linear(hidden_size + action_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1)
        )
        
        # Q2 network
        self.q2 = nn.Sequential(
            nn.Linear(hidden_size + action_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1)
        )
    
    def forward(self, state: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass (both Q-networks)
        
        Args:
            state: (batch, seq_len, state_dim) or (batch, state_dim)
            action: (batch, action_dim)
        
        Returns:
            (Q1, Q2): Both (batch, 1)
        """
        # Handle single timestep
        if state.dim() == 2:
            state = state.unsqueeze(1)
        
        # LSTM processing
        lstm_out, _ = self.lstm(state)
        last_output = lstm_out[:, -1, :]  # (batch, hidden_size)
        
        # Concatenate with action
        sa = torch.cat([last_output, action], dim=1)
        
        return self.q1(sa), self.q2(sa)


# ==============================================================================
# CNN1D ARCHITECTURE (USDJPY - BREAKOUT DETECTION)
# ==============================================================================

class CNN1DPolicy(nn.Module):
    """
    1D Convolutional Policy Network
    
    Optimized for: USDJPY (breakouts, local patterns, momentum)
    
    Architecture:
        Input: State sequence (batch, seq_len, state_dim)
        Conv1D: Multi-scale feature extraction (5, 3, 3 kernels)
        Output: Action (batch, action_dim)
    
    Key Features:
    - Detects local price patterns (support/resistance breaks)
    - Multi-scale convolutions for different timeframes
    - Efficient for short-term trading signals
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        conv_channels: Tuple[int, int, int] = (32, 64, 128),
        kernel_sizes: Tuple[int, int, int] = (5, 3, 3),
        max_action: float = 1.0
    ):
        super().__init__()
        
        self.max_action = max_action
        
        # Convolutional layers
        self.conv1 = nn.Conv1d(state_dim, conv_channels[0], kernel_size=kernel_sizes[0], padding=kernel_sizes[0]//2)
        self.bn1 = nn.BatchNorm1d(conv_channels[0])
        
        self.conv2 = nn.Conv1d(conv_channels[0], conv_channels[1], kernel_size=kernel_sizes[1], padding=kernel_sizes[1]//2)
        self.bn2 = nn.BatchNorm1d(conv_channels[1])
        
        self.conv3 = nn.Conv1d(conv_channels[1], conv_channels[2], kernel_size=kernel_sizes[2], padding=kernel_sizes[2]//2)
        self.bn3 = nn.BatchNorm1d(conv_channels[2])
        
        # Global average pooling
        self.pool = nn.AdaptiveAvgPool1d(1)
        
        # Policy head
        self.fc = nn.Sequential(
            nn.Linear(conv_channels[2], 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, action_dim),
            nn.Tanh()
        )
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            state: (batch, seq_len, state_dim) or (batch, state_dim)
        
        Returns:
            action: (batch, action_dim) in [-max_action, max_action]
        """
        # Handle single timestep
        if state.dim() == 2:
            state = state.unsqueeze(1)  # (batch, 1, state_dim)
        
        # Transpose for Conv1D: (batch, state_dim, seq_len)
        x = state.transpose(1, 2)
        
        # Convolutional layers
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        
        # Global pooling
        x = self.pool(x).squeeze(-1)  # (batch, conv_channels[2])
        
        # Policy head
        action = self.fc(x)
        
        return self.max_action * action


class CNN1DCritic(nn.Module):
    """
    1D Convolutional Critic Network
    
    Twin Q-networks for PPO value estimation
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        conv_channels: Tuple[int, int, int] = (32, 64, 128),
        kernel_sizes: Tuple[int, int, int] = (5, 3, 3)
    ):
        super().__init__()
        
        # Shared convolutional backbone
        self.conv1 = nn.Conv1d(state_dim, conv_channels[0], kernel_size=kernel_sizes[0], padding=kernel_sizes[0]//2)
        self.bn1 = nn.BatchNorm1d(conv_channels[0])
        
        self.conv2 = nn.Conv1d(conv_channels[0], conv_channels[1], kernel_size=kernel_sizes[1], padding=kernel_sizes[1]//2)
        self.bn2 = nn.BatchNorm1d(conv_channels[1])
        
        self.conv3 = nn.Conv1d(conv_channels[1], conv_channels[2], kernel_size=kernel_sizes[2], padding=kernel_sizes[2]//2)
        self.bn3 = nn.BatchNorm1d(conv_channels[2])
        
        self.pool = nn.AdaptiveAvgPool1d(1)
        
        # Q1 network
        self.q1 = nn.Sequential(
            nn.Linear(conv_channels[2] + action_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 1)
        )
        
        # Q2 network
        self.q2 = nn.Sequential(
            nn.Linear(conv_channels[2] + action_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 1)
        )
    
    def forward(self, state: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass (both Q-networks)"""
        # Handle single timestep
        if state.dim() == 2:
            state = state.unsqueeze(1)
        
        # Transpose for Conv1D
        x = state.transpose(1, 2)
        
        # Convolutional processing
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        
        # Global pooling
        x = self.pool(x).squeeze(-1)
        
        # Concatenate with action
        sa = torch.cat([x, action], dim=1)
        
        return self.q1(sa), self.q2(sa)


# ==============================================================================
# TRANSFORMER ARCHITECTURE (XAUUSD - REGIME ADAPTATION)
# ==============================================================================

class PositionalEncoding(nn.Module):
    """Positional encoding for Transformer"""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_model)
        Returns:
            x with positional encoding added
        """
        return x + self.pe[:, :x.size(1), :]


class TransformerPolicy(nn.Module):
    """
    Transformer-based Policy Network
    
    Optimized for: XAUUSD (regime changes, long-range dependencies, volatility)
    
    Architecture:
        Input: State sequence (batch, seq_len, state_dim)
        Transformer: Multi-head attention with 8 heads, 4 layers
        Output: Action (batch, action_dim)
    
    Key Features:
    - Captures long-range dependencies (regime shifts)
    - Self-attention for pattern recognition
    - Suitable for high-volatility assets
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        max_action: float = 1.0
    ):
        super().__init__()
        
        self.max_action = max_action
        self.d_model = d_model
        
        # Input projection
        self.input_projection = nn.Linear(state_dim, d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Policy head
        self.fc = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, action_dim),
            nn.Tanh()
        )
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            state: (batch, seq_len, state_dim) or (batch, state_dim)
        
        Returns:
            action: (batch, action_dim) in [-max_action, max_action]
        """
        # Handle single timestep
        if state.dim() == 2:
            state = state.unsqueeze(1)
        
        # Project to d_model
        x = self.input_projection(state)  # (batch, seq_len, d_model)
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Transformer processing
        x = self.transformer(x)  # (batch, seq_len, d_model)
        
        # Take last timestep (or mean pooling)
        x = x[:, -1, :]  # (batch, d_model)
        
        # Policy head
        action = self.fc(x)
        
        return self.max_action * action


class TransformerCritic(nn.Module):
    """Transformer-based Critic Network"""
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 1024,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.d_model = d_model
        
        # Input projection
        self.input_projection = nn.Linear(state_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Q1 network
        self.q1 = nn.Sequential(
            nn.Linear(d_model + action_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1)
        )
        
        # Q2 network
        self.q2 = nn.Sequential(
            nn.Linear(d_model + action_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1)
        )
    
    def forward(self, state: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass (both Q-networks)"""
        if state.dim() == 2:
            state = state.unsqueeze(1)
        
        # Project and encode
        x = self.input_projection(state)
        x = self.pos_encoder(x)
        
        # Transformer processing
        x = self.transformer(x)
        x = x[:, -1, :]  # Last timestep
        
        # Concatenate with action
        sa = torch.cat([x, action], dim=1)
        
        return self.q1(sa), self.q2(sa)


# ==============================================================================
# ATTENTION ARCHITECTURE (GBPUSD - NOISE FILTERING)
# ==============================================================================

class AttentionPolicy(nn.Module):
    """
    Multi-Head Attention Policy Network
    
    Optimized for: GBPUSD (whipsaws, false breakouts, noise)
    
    Architecture:
        Input: State sequence (batch, seq_len, state_dim)
        Multi-Head Attention: 4 heads for feature weighting
        Output: Action (batch, action_dim)
    
    Key Features:
    - Attention mechanism filters noise
    - Focuses on important features (ignores whipsaws)
    - Suitable for high-frequency false breakouts
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        attention_dim: int = 128,
        num_heads: int = 4,
        dropout: float = 0.1,
        max_action: float = 1.0
    ):
        super().__init__()
        
        self.max_action = max_action
        self.attention_dim = attention_dim
        
        # Input projection
        self.input_projection = nn.Linear(state_dim, attention_dim)
        
        # Multi-head attention
        self.attention = nn.MultiheadAttention(
            embed_dim=attention_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Feedforward network
        self.ffn = nn.Sequential(
            nn.Linear(attention_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 256),
            nn.LayerNorm(256),
            nn.ReLU()
        )
        
        # Policy head
        self.fc = nn.Sequential(
            nn.Linear(256, action_dim),
            nn.Tanh()
        )
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            state: (batch, seq_len, state_dim) or (batch, state_dim)
        
        Returns:
            action: (batch, action_dim) in [-max_action, max_action]
        """
        # Handle single timestep
        if state.dim() == 2:
            state = state.unsqueeze(1)
        
        # Project input
        x = self.input_projection(state)  # (batch, seq_len, attention_dim)
        
        # Multi-head attention (self-attention)
        attn_output, _ = self.attention(x, x, x)  # (batch, seq_len, attention_dim)
        
        # Take last timestep
        x = attn_output[:, -1, :]  # (batch, attention_dim)
        
        # Feedforward
        x = self.ffn(x)  # (batch, 256)
        
        # Policy head
        action = self.fc(x)
        
        return self.max_action * action


class AttentionCritic(nn.Module):
    """Attention-based Critic Network"""
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        attention_dim: int = 128,
        num_heads: int = 4,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.attention_dim = attention_dim
        
        # Input projection
        self.input_projection = nn.Linear(state_dim, attention_dim)
        
        # Multi-head attention
        self.attention = nn.MultiheadAttention(
            embed_dim=attention_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Feedforward
        self.ffn = nn.Sequential(
            nn.Linear(attention_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 256),
            nn.LayerNorm(256),
            nn.ReLU()
        )
        
        # Q1 network
        self.q1 = nn.Sequential(
            nn.Linear(256 + action_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1)
        )
        
        # Q2 network
        self.q2 = nn.Sequential(
            nn.Linear(256 + action_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1)
        )
    
    def forward(self, state: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass (both Q-networks)"""
        if state.dim() == 2:
            state = state.unsqueeze(1)
        
        # Project and attend
        x = self.input_projection(state)
        attn_output, _ = self.attention(x, x, x)
        x = attn_output[:, -1, :]
        
        # Feedforward
        x = self.ffn(x)
        
        # Concatenate with action
        sa = torch.cat([x, action], dim=1)
        
        return self.q1(sa), self.q2(sa)


# ==============================================================================
# ARCHITECTURE FACTORY
# ==============================================================================

def create_policy_network(
    architecture: str,
    state_dim: int,
    action_dim: int,
    **kwargs
) -> nn.Module:
    """
    Factory function to create policy network
    
    Args:
        architecture: 'LSTM', 'CNN1D', 'Transformer', 'Attention', 'MLP'
        state_dim: State dimension
        action_dim: Action dimension
        **kwargs: Architecture-specific parameters
    
    Returns:
        Policy network instance
    """
    # Map generic hidden_dim to architecture-specific parameter names
    if 'hidden_dim' in kwargs:
        if architecture == 'LSTM':
            kwargs['hidden_size'] = kwargs.pop('hidden_dim')
        elif architecture == 'Transformer':
            kwargs['d_model'] = kwargs.pop('hidden_dim')
        elif architecture == 'Attention':
            kwargs['attention_dim'] = kwargs.pop('hidden_dim')
        elif architecture == 'CNN1D':
            # CNN1D doesn't use hidden_dim, remove it
            kwargs.pop('hidden_dim')
    
    if architecture == 'LSTM':
        return LSTMPolicy(state_dim, action_dim, **kwargs)
    elif architecture == 'CNN1D':
        return CNN1DPolicy(state_dim, action_dim, **kwargs)
    elif architecture == 'Transformer':
        return TransformerPolicy(state_dim, action_dim, **kwargs)
    elif architecture == 'Attention':
        return AttentionPolicy(state_dim, action_dim, **kwargs)
    elif architecture == 'MLP':
        # Use standard Actor from agents.py
        from underdog.rl.agents import Actor
        return Actor(state_dim, action_dim, kwargs.get('hidden_dim', 256))
    else:
        raise ValueError(f"Unknown architecture: {architecture}")


def create_critic_network(
    architecture: str,
    state_dim: int,
    action_dim: int,
    **kwargs
) -> nn.Module:
    """
    Factory function to create critic network
    
    Args:
        architecture: 'LSTM', 'CNN1D', 'Transformer', 'Attention', 'MLP'
        state_dim: State dimension
        action_dim: Action dimension
        **kwargs: Architecture-specific parameters
    
    Returns:
        Critic network instance
    """
    # Map generic hidden_dim to architecture-specific parameter names
    if 'hidden_dim' in kwargs:
        if architecture == 'LSTM':
            kwargs['hidden_size'] = kwargs.pop('hidden_dim')
        elif architecture == 'Transformer':
            kwargs['d_model'] = kwargs.pop('hidden_dim')
        elif architecture == 'Attention':
            kwargs['attention_dim'] = kwargs.pop('hidden_dim')
        elif architecture == 'CNN1D':
            # CNN1D doesn't use hidden_dim, remove it
            kwargs.pop('hidden_dim')
    
    if architecture == 'LSTM':
        return LSTMCritic(state_dim, action_dim, **kwargs)
    elif architecture == 'CNN1D':
        return CNN1DCritic(state_dim, action_dim, **kwargs)
    elif architecture == 'Transformer':
        return TransformerCritic(state_dim, action_dim, **kwargs)
    elif architecture == 'Attention':
        return AttentionCritic(state_dim, action_dim, **kwargs)
    elif architecture == 'MLP':
        # Use standard Critic from agents.py
        from underdog.rl.agents import Critic
        return Critic(state_dim, action_dim, kwargs.get('hidden_dim', 256))
    else:
        raise ValueError(f"Unknown architecture: {architecture}")
