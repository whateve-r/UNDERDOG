"""
A3C Meta-Agent for Multi-Asset Portfolio Coordination

This module implements the Meta-Agent (Level 2) of the MARL architecture.
The A3C agent coordinates 4 local TD3 agents by assigning risk limits.

Architecture:
    - Algorithm: A3C (Asynchronous Advantage Actor-Critic) without lock
    - Input: Meta-State (15D) - Global portfolio metrics
    - Output: Meta-Action (4D) - Risk limits for each local agent
    
Coordination Strategy:
    - Meta-Action modulates maximum position size for each TD3 agent
    - Allows risk reduction during high Turbulence periods
    - Prevents over-allocation across correlated pairs
    
References:
    - 2405.19982v1.pdf: A3C superior to PPO for multi-currency Forex
    - ALA2017_Gupta.pdf: CTDE (Centralized Training, Decentralized Execution)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Beta
import numpy as np
from typing import Tuple, Optional, Dict, Any
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class A3CConfig:
    """Configuration for A3C Meta-Agent"""
    
    # Network architecture
    meta_state_dim: int = 23      # Meta-State dimension (expanded: 15 + 8 new features)
    meta_action_dim: int = 4      # Meta-Action dimension (4 risk limits)
    hidden_dim: int = 128         # Hidden layer size
    
    # Learning
    lr: float = 3e-4              # Learning rate
    gamma: float = 0.99           # Discount factor
    entropy_coef: float = 0.01    # Entropy coefficient (exploration)
    value_loss_coef: float = 0.5  # Value loss coefficient
    max_grad_norm: float = 0.5    # Gradient clipping
    
    # A3C specific
    n_steps: int = 5              # Steps before update
    
    # Action distribution
    action_min: float = 0.1       # Minimum risk limit
    action_max: float = 1.0       # Maximum risk limit


class A3CMetaAgent(nn.Module):
    """
    A3C Meta-Agent for coordinating 4 TD3 local agents
    
    Architecture:
        Meta-State (15D) → Shared Network (128D) → Actor + Critic
        
        Actor: Outputs Beta distribution parameters (alpha, beta) for each
               of the 4 risk limits. Beta distribution is bounded [0, 1]
               which naturally fits risk limits.
        
        Critic: Outputs state value V(s) for advantage calculation
    
    Training:
        - A3C: Asynchronous updates without locks
        - Advantage Actor-Critic: Policy gradient with baseline
        - Entropy regularization: Encourages exploration
    """
    
    def __init__(self, config: Optional[A3CConfig] = None):
        """
        Initialize A3C Meta-Agent
        
        Args:
            config: A3C configuration
        """
        super().__init__()
        
        self.config = config or A3CConfig()
        
        # Shared feature extractor
        self.shared = nn.Sequential(
            nn.Linear(self.config.meta_state_dim, self.config.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.config.hidden_dim, self.config.hidden_dim),
            nn.ReLU()
        )
        
        # Actor head: Beta distribution parameters
        # For each of 4 agents, output (alpha, beta) parameters
        # Beta(alpha, beta) is bounded [0, 1] - perfect for risk limits
        self.actor_alpha = nn.Sequential(
            nn.Linear(self.config.hidden_dim, self.config.meta_action_dim),
            nn.Softplus()  # Ensure positive (alpha > 0)
        )
        
        self.actor_beta = nn.Sequential(
            nn.Linear(self.config.hidden_dim, self.config.meta_action_dim),
            nn.Softplus()  # Ensure positive (beta > 0)
        )
        
        # Critic head: State value
        self.critic = nn.Sequential(
            nn.Linear(self.config.hidden_dim, 1)
        )
        
        # Optimizer
        self.optimizer = optim.Adam(self.parameters(), lr=self.config.lr)
        
        # Episode tracking
        self.episode_rewards = []
        self.episode_values = []
        self.episode_log_probs = []
        self.episode_entropies = []
        
        logger.info(
            f"A3CMetaAgent initialized: state_dim={self.config.meta_state_dim}, "
            f"action_dim={self.config.meta_action_dim}, hidden_dim={self.config.hidden_dim}"
        )
    
    def forward(self, meta_state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through network
        
        Args:
            meta_state: Meta-State tensor (batch_size, 15)
        
        Returns:
            alpha: Beta distribution alpha parameters (batch_size, 4)
            beta: Beta distribution beta parameters (batch_size, 4)
            value: State value (batch_size, 1)
        """
        # Shared features
        shared_features = self.shared(meta_state)
        
        # Actor: Beta distribution parameters
        alpha = self.actor_alpha(shared_features) + 1.0  # Alpha >= 1
        beta = self.actor_beta(shared_features) + 1.0    # Beta >= 1
        
        # Critic: State value
        value = self.critic(shared_features)
        
        return alpha, beta, value
    
    def select_meta_action(
        self,
        meta_state: np.ndarray,
        deterministic: bool = False
    ) -> Tuple[np.ndarray, Optional[float], Optional[float]]:
        """
        Select Meta-Action (risk limits for 4 agents)
        
        Args:
            meta_state: Meta-State vector (15D)
            deterministic: If True, use mean of Beta distribution
        
        Returns:
            meta_action: Risk limits [0, 1]⁴
            log_prob: Log probability of action (for training)
            entropy: Entropy of distribution (for exploration)
        """
        with torch.no_grad():
            # Convert to tensor
            state_tensor = torch.FloatTensor(meta_state).unsqueeze(0)
            
            # Forward pass
            alpha, beta, value = self.forward(state_tensor)
            
            # Create Beta distributions
            dist = Beta(alpha, beta)
            
            if deterministic:
                # Use mean of Beta distribution
                meta_action = dist.mean
            else:
                # Sample from Beta distribution
                meta_action = dist.sample()
            
            # Scale to [action_min, action_max]
            meta_action_scaled = (
                self.config.action_min + 
                meta_action * (self.config.action_max - self.config.action_min)
            )
            
            # Convert to numpy
            meta_action_np = meta_action_scaled.squeeze(0).numpy()
            
            # Calculate log_prob and entropy for training
            if not deterministic:
                log_prob = dist.log_prob(meta_action).sum().item()
                entropy = dist.entropy().sum().item()
            else:
                log_prob = None
                entropy = None
            
            return meta_action_np, log_prob, entropy
    
    def evaluate_action(
        self,
        meta_state: torch.Tensor,
        meta_action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate action for training (compute log_prob, entropy, value)
        
        Args:
            meta_state: Meta-State tensor (batch_size, 15)
            meta_action: Meta-Action tensor (batch_size, 4) in [action_min, action_max]
        
        Returns:
            log_prob: Log probability of action (batch_size,)
            entropy: Entropy of distribution (batch_size,)
            value: State value (batch_size, 1)
        """
        # Forward pass
        alpha, beta, value = self.forward(meta_state)
        
        # Create Beta distributions
        dist = Beta(alpha, beta)
        
        # Unscale action to [0, 1]
        meta_action_unscaled = (
            (meta_action - self.config.action_min) / 
            (self.config.action_max - self.config.action_min)
        )
        
        # Calculate log probability (sum over 4 dimensions)
        log_prob = dist.log_prob(meta_action_unscaled).sum(dim=-1)
        
        # Calculate entropy (sum over 4 dimensions)
        entropy = dist.entropy().sum(dim=-1)
        
        return log_prob, entropy, value
    
    def compute_returns(
        self,
        rewards: list,
        dones: list,
        next_value: float
    ) -> torch.Tensor:
        """
        Compute discounted returns (Monte Carlo)
        
        Args:
            rewards: List of rewards
            dones: List of done flags
            next_value: Value of next state (bootstrap if not done)
        
        Returns:
            returns: Discounted returns tensor
        """
        returns = []
        R = next_value
        
        for reward, done in zip(reversed(rewards), reversed(dones)):
            if done:
                R = 0.0
            R = reward + self.config.gamma * R
            returns.insert(0, R)
        
        return torch.FloatTensor(returns)
    
    def update(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        returns: torch.Tensor
    ) -> Dict[str, float]:
        """
        Update A3C Meta-Agent with collected experience
        
        Args:
            states: Meta-States (batch_size, 15)
            actions: Meta-Actions (batch_size, 4)
            returns: Discounted returns (batch_size,)
        
        Returns:
            metrics: Training metrics dict
        """
        # Evaluate actions
        log_probs, entropies, values = self.evaluate_action(states, actions)
        
        values = values.squeeze(-1)
        
        # Compute advantages
        advantages = returns - values.detach()
        
        # Actor loss: Policy gradient with entropy regularization
        actor_loss = -(log_probs * advantages).mean()
        actor_loss -= self.config.entropy_coef * entropies.mean()
        
        # Critic loss: MSE between value and return
        critic_loss = F.mse_loss(values, returns)
        
        # Total loss
        total_loss = actor_loss + self.config.value_loss_coef * critic_loss
        
        # Optimize
        self.optimizer.zero_grad()
        total_loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(
            self.parameters(),
            self.config.max_grad_norm
        )
        
        self.optimizer.step()
        
        # Metrics
        metrics = {
            'actor_loss': actor_loss.item(),
            'critic_loss': critic_loss.item(),
            'total_loss': total_loss.item(),
            'entropy': entropies.mean().item(),
            'value_mean': values.mean().item(),
            'advantage_mean': advantages.mean().item(),
        }
        
        return metrics
    
    def save(self, path: str):
        """Save model checkpoint"""
        torch.save({
            'state_dict': self.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'config': self.config,
        }, path)
        logger.info(f"A3CMetaAgent saved to {path}")
    
    def load(self, path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(path, weights_only=False)
        self.load_state_dict(checkpoint['state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        logger.info(f"A3CMetaAgent loaded from {path}")


def test_a3c_meta_agent():
    """Test function for A3CMetaAgent"""
    
    # Create agent
    config = A3CConfig(
        meta_state_dim=23,  # Expanded with correlations, sentiment, opportunity
        meta_action_dim=4,
        hidden_dim=128
    )
    
    agent = A3CMetaAgent(config=config)
    
    # Test forward pass
    meta_state = np.random.randn(15).astype(np.float32)
    
    # Select action
    meta_action, log_prob, entropy = agent.select_meta_action(meta_state, deterministic=False)
    
    print(f"Meta-State shape: {meta_state.shape}")
    print(f"Meta-Action: {meta_action}")
    print(f"Log Prob: {log_prob:.4f}")
    print(f"Entropy: {entropy:.4f}")
    
    # Test training update
    states = torch.randn(10, 15)
    actions = torch.rand(10, 4) * 0.9 + 0.1  # [0.1, 1.0]
    returns = torch.randn(10)
    
    metrics = agent.update(states, actions, returns)
    
    print(f"\nTraining Metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")
    
    # Test save/load
    agent.save("test_a3c_meta_agent.pth")
    agent.load("test_a3c_meta_agent.pth")
    print("\nSave/Load successful!")


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
    )
    
    # Run test
    print("Testing A3CMetaAgent...")
    test_a3c_meta_agent()
