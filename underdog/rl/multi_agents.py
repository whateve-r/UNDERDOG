"""
UNDERDOG TRADING SYSTEMS
PPO, SAC, and DDPG Agent Implementations

This module extends the existing TD3 agent with additional DRL algorithms
for heterogeneous MARL architecture.

Agents:
- PPOAgent: Proximal Policy Optimization (USDJPY - breakouts, stability)
- SACAgent: Soft Actor-Critic (XAUUSD - exploration, regime adaptation)
- DDPGAgent: Deep Deterministic Policy Gradient (GBPUSD - momentum, whipsaw resistance)

All agents support custom neural network architectures from models.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass
import logging

from underdog.rl.models import create_policy_network, create_critic_network
from underdog.rl.agents import ReplayBuffer

logger = logging.getLogger(__name__)


# ==============================================================================
# PPO AGENT (USDJPY - BREAKOUT DETECTION)
# ==============================================================================

@dataclass
class PPOConfig:
    """PPO Agent Configuration"""
    # Network architecture
    state_dim: int = 14
    action_dim: int = 2
    hidden_dim: int = 256
    architecture: str = 'CNN1D'  # Default for USDJPY
    
    # Training hyperparameters
    lr_actor: float = 3e-4
    lr_critic: float = 1e-3
    gamma: float = 0.99
    gae_lambda: float = 0.95
    
    # PPO-specific
    clip_ratio: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5
    
    # Training
    ppo_epochs: int = 10
    batch_size: int = 64
    buffer_size: int = 2048  # Smaller on-policy buffer
    
    # Device
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'


class PPOAgent:
    """
    Proximal Policy Optimization Agent
    
    Optimized for: USDJPY (breakouts, policy stability, Sharpe optimization)
    
    Key Features:
    - Clipped surrogate objective for stable updates
    - GAE for variance reduction
    - On-policy learning (buffer cleared after training)
    - Entropy bonus for exploration
    
    Compatible Architectures:
    - CNN1D (default for USDJPY breakout detection)
    - LSTM, Transformer, Attention, MLP
    """
    
    def __init__(self, config: Optional[PPOConfig] = None):
        """Initialize PPO agent"""
        self.config = config or PPOConfig()
        self.device = torch.device(self.config.device)
        
        # Create networks using architecture factory
        max_lot_size = 0.1
        
        # Policy network
        self.actor = create_policy_network(
            architecture=self.config.architecture,
            state_dim=self.config.state_dim,
            action_dim=self.config.action_dim,
            hidden_dim=self.config.hidden_dim,
            max_action=max_lot_size
        ).to(self.device)
        
        # Value network (single output, not twin critics)
        self.critic = nn.Sequential(
            nn.Linear(self.config.state_dim, self.config.hidden_dim),
            nn.LayerNorm(self.config.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.config.hidden_dim, self.config.hidden_dim),
            nn.LayerNorm(self.config.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.config.hidden_dim, 1)
        ).to(self.device)
        
        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.config.lr_actor)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.config.lr_critic)
        
        # Rollout buffer (on-policy)
        self.buffer = {
            'states': [],
            'actions': [],
            'rewards': [],
            'dones': [],
            'log_probs': [],
            'values': []
        }
        
        # Action distribution (for continuous actions)
        self.log_std = nn.Parameter(torch.zeros(self.config.action_dim).to(self.device))
        
        # Training iteration counter
        self.total_it = 0
        
        logger.info(f"PPOAgent initialized with {self.config.architecture} architecture")
    
    def select_action(self, state: np.ndarray, explore: bool = True) -> np.ndarray:
        """
        Select action using policy network
        
        Args:
            state: Environment state
            explore: If True, sample from policy; if False, use mean
        
        Returns:
            action: Action vector
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            # Get mean action from policy
            action_mean = self.actor(state_tensor)
            
            if explore:
                # Sample from Gaussian distribution
                std = torch.exp(self.log_std)
                dist = torch.distributions.Normal(action_mean, std)
                action = dist.sample()
                action = torch.clamp(action, -1.0, 1.0)
            else:
                action = action_mean
        
        return action.cpu().numpy()[0]
    
    def store_transition(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        done: bool
    ):
        """Store transition in rollout buffer"""
        state_tensor = torch.FloatTensor(state).to(self.device)
        action_tensor = torch.FloatTensor(action).to(self.device)
        
        with torch.no_grad():
            # Get value estimate (reduce sequential output to scalar)
            value = self.critic(state_tensor.unsqueeze(0)).mean().item()
            
            # Get log probability
            action_mean = self.actor(state_tensor.unsqueeze(0))
            std = torch.exp(self.log_std)
            dist = torch.distributions.Normal(action_mean, std)
            log_prob = dist.log_prob(action_tensor).sum().item()
        
        self.buffer['states'].append(state)
        self.buffer['actions'].append(action)
        self.buffer['rewards'].append(reward)
        self.buffer['dones'].append(done)
        self.buffer['log_probs'].append(log_prob)
        self.buffer['values'].append(value)
    
    def train(self, replay_buffer=None) -> Dict[str, float]:
        """
        Train PPO agent using rollout buffer
        
        Returns:
            Training metrics
        """
        if len(self.buffer['states']) < self.config.batch_size:
            return {}  # Not enough data
        
        self.total_it += 1
        
        # Convert buffer to tensors
        states = torch.FloatTensor(np.array(self.buffer['states'])).to(self.device)
        actions = torch.FloatTensor(np.array(self.buffer['actions'])).to(self.device)
        old_log_probs = torch.FloatTensor(self.buffer['log_probs']).to(self.device)
        rewards = self.buffer['rewards']
        dones = self.buffer['dones']
        values = self.buffer['values']
        
        # Compute advantages using GAE
        advantages = []
        returns = []
        gae = 0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0
            else:
                next_value = values[t + 1]
            
            delta = rewards[t] + self.config.gamma * next_value * (1 - dones[t]) - values[t]
            gae = delta + self.config.gamma * self.config.gae_lambda * (1 - dones[t]) * gae
            
            advantages.insert(0, gae)
            returns.insert(0, gae + values[t])
        
        advantages = torch.FloatTensor(advantages).to(self.device)
        returns = torch.FloatTensor(returns).to(self.device)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # PPO update for K epochs
        total_actor_loss = 0
        total_critic_loss = 0
        total_entropy = 0
        
        for _ in range(self.config.ppo_epochs):
            # Compute current log probs and values
            action_mean = self.actor(states)
            std = torch.exp(self.log_std)
            dist = torch.distributions.Normal(action_mean, std)
            log_probs = dist.log_prob(actions).sum(dim=1)
            entropy = dist.entropy().sum(dim=1).mean()
            
            current_values = self.critic(states).squeeze()
            
            # Compute ratio and clipped surrogate
            ratio = torch.exp(log_probs - old_log_probs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.config.clip_ratio, 1 + self.config.clip_ratio) * advantages
            
            # Actor loss (policy loss)
            actor_loss = -torch.min(surr1, surr2).mean() - self.config.entropy_coef * entropy
            
            # Critic loss (value loss)
            critic_loss = F.mse_loss(current_values, returns)
            
            # Total loss
            total_loss = actor_loss + self.config.value_coef * critic_loss
            
            # Update networks
            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            total_loss.backward()
            
            # Gradient clipping
            nn.utils.clip_grad_norm_(self.actor.parameters(), self.config.max_grad_norm)
            nn.utils.clip_grad_norm_(self.critic.parameters(), self.config.max_grad_norm)
            
            self.actor_optimizer.step()
            self.critic_optimizer.step()
            
            total_actor_loss += actor_loss.item()
            total_critic_loss += critic_loss.item()
            total_entropy += entropy.item()
        
        # Clear buffer (on-policy)
        self.buffer = {
            'states': [],
            'actions': [],
            'rewards': [],
            'dones': [],
            'log_probs': [],
            'values': []
        }
        
        return {
            'actor_loss': total_actor_loss / self.config.ppo_epochs,
            'critic_loss': total_critic_loss / self.config.ppo_epochs,
            'entropy': total_entropy / self.config.ppo_epochs
        }
    
    def save(self, filename: str):
        """Save agent"""
        torch.save({
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict(),
            'log_std': self.log_std,
            'config': self.config,
        }, filename)
    
    def load(self, filename: str):
        """Load agent"""
        checkpoint = torch.load(filename, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic.load_state_dict(checkpoint['critic'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])
        self.log_std = checkpoint['log_std']


# ==============================================================================
# SAC AGENT (XAUUSD - REGIME ADAPTATION)
# ==============================================================================

@dataclass
class SACConfig:
    """SAC Agent Configuration"""
    # Network architecture
    state_dim: int = 14
    action_dim: int = 2
    hidden_dim: int = 512
    architecture: str = 'Transformer'  # Default for XAUUSD
    
    # Training hyperparameters
    lr_actor: float = 3e-4
    lr_critic: float = 3e-4
    lr_alpha: float = 3e-4
    gamma: float = 0.99
    tau: float = 0.005
    
    # SAC-specific
    alpha: float = 0.2  # Entropy coefficient
    auto_alpha: bool = True  # Automatic temperature tuning
    target_entropy: float = -2.0  # -dim(action_space)
    
    # Replay buffer
    buffer_size: int = 1_000_000
    batch_size: int = 256
    
    # Device
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'


class SACAgent:
    """
    Soft Actor-Critic Agent
    
    Optimized for: XAUUSD (high volatility, regime changes, exploration)
    
    Key Features:
    - Maximum entropy RL (robust exploration)
    - Automatic temperature tuning
    - Twin Q-networks
    - Stochastic policy
    
    Compatible Architectures:
    - Transformer (default for XAUUSD regime detection)
    - LSTM, CNN1D, Attention, MLP
    """
    
    def __init__(self, config: Optional[SACConfig] = None):
        """Initialize SAC agent"""
        self.config = config or SACConfig()
        self.device = torch.device(self.config.device)
        
        # Set target entropy
        if self.config.auto_alpha:
            self.target_entropy = self.config.target_entropy or -self.config.action_dim
        
        # Create networks
        max_lot_size = 0.1
        
        # Policy network
        self.actor = create_policy_network(
            architecture=self.config.architecture,
            state_dim=self.config.state_dim,
            action_dim=self.config.action_dim,
            hidden_dim=self.config.hidden_dim,
            max_action=max_lot_size
        ).to(self.device)
        
        # Critic networks (twin Q-functions)
        self.critic = create_critic_network(
            architecture=self.config.architecture,
            state_dim=self.config.state_dim,
            action_dim=self.config.action_dim,
            hidden_dim=self.config.hidden_dim
        ).to(self.device)
        
        self.critic_target = create_critic_network(
            architecture=self.config.architecture,
            state_dim=self.config.state_dim,
            action_dim=self.config.action_dim,
            hidden_dim=self.config.hidden_dim
        ).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        
        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.config.lr_actor)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.config.lr_critic)
        
        # Temperature (alpha)
        if self.config.auto_alpha:
            self.log_alpha = torch.tensor(np.log(self.config.alpha), requires_grad=True, device=self.device)
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=self.config.lr_alpha)
            self.alpha = self.log_alpha.exp()
        else:
            self.alpha = self.config.alpha
        
        # Action distribution
        self.log_std = nn.Parameter(torch.zeros(self.config.action_dim).to(self.device))
        
        self.total_it = 0
        
        logger.info(f"SACAgent initialized with {self.config.architecture} architecture")
    
    def select_action(self, state: np.ndarray, explore: bool = True) -> np.ndarray:
        """Select action using stochastic policy"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            action_mean = self.actor(state_tensor)
            
            if explore:
                std = torch.exp(self.log_std)
                dist = torch.distributions.Normal(action_mean, std)
                action = dist.sample()
                action = torch.tanh(action)  # Squash to [-1, 1]
            else:
                action = torch.tanh(action_mean)
        
        return action.cpu().numpy()[0]
    
    def train(self, replay_buffer: ReplayBuffer) -> Dict[str, float]:
        """Train SAC agent"""
        self.total_it += 1
        
        # Sample batch
        state, action, reward, next_state, done = replay_buffer.sample(self.config.batch_size)
        
        state = torch.FloatTensor(state).to(self.device)
        action = torch.FloatTensor(action).to(self.device)
        reward = torch.FloatTensor(reward).unsqueeze(1).to(self.device)
        next_state = torch.FloatTensor(next_state).to(self.device)
        done = torch.FloatTensor(done).unsqueeze(1).to(self.device)
        
        # Update critics
        with torch.no_grad():
            # Get next action from policy
            next_action_mean = self.actor(next_state)
            std = torch.exp(self.log_std)
            dist = torch.distributions.Normal(next_action_mean, std)
            next_action = dist.rsample()
            next_action_tanh = torch.tanh(next_action)
            next_log_prob = dist.log_prob(next_action).sum(dim=1, keepdim=True)
            
            # Apply tanh correction
            next_log_prob -= torch.log(1 - next_action_tanh.pow(2) + 1e-6).sum(dim=1, keepdim=True)
            
            # Compute target Q-value
            q1_next, q2_next = self.critic_target(next_state, next_action_tanh)
            q_next = torch.min(q1_next, q2_next)
            
            if isinstance(self.alpha, torch.Tensor):
                alpha_val = self.alpha.detach()
            else:
                alpha_val = self.alpha
            
            target_q = reward + (1 - done) * self.config.gamma * (q_next - alpha_val * next_log_prob)
        
        # Current Q-values
        q1, q2 = self.critic(state, action)
        
        # Critic loss
        critic_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)
        
        # Update critics
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # Update actor
        action_mean = self.actor(state)
        std = torch.exp(self.log_std)
        dist = torch.distributions.Normal(action_mean, std)
        action_sample = dist.rsample()
        action_tanh = torch.tanh(action_sample)
        log_prob = dist.log_prob(action_sample).sum(dim=1, keepdim=True)
        log_prob -= torch.log(1 - action_tanh.pow(2) + 1e-6).sum(dim=1, keepdim=True)
        
        q1_pi, q2_pi = self.critic(state, action_tanh)
        q_pi = torch.min(q1_pi, q2_pi)
        
        if isinstance(self.alpha, torch.Tensor):
            alpha_val = self.alpha.detach()
        else:
            alpha_val = self.alpha
        
        actor_loss = (alpha_val * log_prob - q_pi).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # Update temperature
        if self.config.auto_alpha:
            alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()
            
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            
            self.alpha = self.log_alpha.exp()
        
        # Soft update target networks
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.config.tau * param.data + (1 - self.config.tau) * target_param.data)
        
        return {
            'actor_loss': actor_loss.item(),
            'critic_loss': critic_loss.item(),
            'alpha': self.alpha.item() if isinstance(self.alpha, torch.Tensor) else self.alpha
        }
    
    def save(self, filename: str):
        """Save agent"""
        torch.save({
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'critic_target': self.critic_target.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict(),
            'log_std': self.log_std,
            'log_alpha': self.log_alpha if self.config.auto_alpha else None,
            'config': self.config,
        }, filename)
    
    def load(self, filename: str):
        """Load agent"""
        checkpoint = torch.load(filename, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic.load_state_dict(checkpoint['critic'])
        self.critic_target.load_state_dict(checkpoint['critic_target'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])
        self.log_std = checkpoint['log_std']
        if self.config.auto_alpha and checkpoint['log_alpha'] is not None:
            self.log_alpha = checkpoint['log_alpha']
            self.alpha = self.log_alpha.exp()


# ==============================================================================
# DDPG AGENT (GBPUSD - MOMENTUM TRADING)
# ==============================================================================

@dataclass
class DDPGConfig:
    """DDPG Agent Configuration"""
    # Network architecture
    state_dim: int = 14
    action_dim: int = 2
    hidden_dim: int = 256
    architecture: str = 'Attention'  # Default for GBPUSD
    
    # Training hyperparameters
    lr_actor: float = 1e-4
    lr_critic: float = 3e-4
    gamma: float = 0.99
    tau: float = 0.005
    
    # Exploration (Ornstein-Uhlenbeck noise)
    ou_noise_theta: float = 0.15
    ou_noise_sigma: float = 0.2
    noise_decay: float = 0.9995
    noise_min: float = 0.05
    
    # Replay buffer
    buffer_size: int = 1_000_000
    batch_size: int = 256
    
    # Device
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'


class OUNoise:
    """Ornstein-Uhlenbeck noise process"""
    
    def __init__(self, size: int, theta: float = 0.15, sigma: float = 0.2):
        self.size = size
        self.theta = theta
        self.sigma = sigma
        self.state = np.zeros(size)
        self.reset()
    
    def reset(self):
        self.state = np.zeros(self.size)
    
    def sample(self):
        dx = self.theta * (0 - self.state) + self.sigma * np.random.randn(self.size)
        self.state += dx
        return self.state


class DDPGAgent:
    """
    Deep Deterministic Policy Gradient Agent
    
    Optimized for: GBPUSD (whipsaws, false breakouts, momentum)
    
    Key Features:
    - Deterministic policy
    - OU noise for exploration
    - Twin Q-networks (optional)
    - Attention mechanism for noise filtering
    
    Compatible Architectures:
    - Attention (default for GBPUSD whipsaw resistance)
    - LSTM, CNN1D, Transformer, MLP
    """
    
    def __init__(self, config: Optional[DDPGConfig] = None):
        """Initialize DDPG agent"""
        self.config = config or DDPGConfig()
        self.device = torch.device(self.config.device)
        
        # Create networks
        max_lot_size = 0.1
        
        # Policy network
        self.actor = create_policy_network(
            architecture=self.config.architecture,
            state_dim=self.config.state_dim,
            action_dim=self.config.action_dim,
            hidden_dim=self.config.hidden_dim,
            max_action=max_lot_size
        ).to(self.device)
        
        self.actor_target = create_policy_network(
            architecture=self.config.architecture,
            state_dim=self.config.state_dim,
            action_dim=self.config.action_dim,
            hidden_dim=self.config.hidden_dim,
            max_action=max_lot_size
        ).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        
        # Critic networks
        self.critic = create_critic_network(
            architecture=self.config.architecture,
            state_dim=self.config.state_dim,
            action_dim=self.config.action_dim,
            hidden_dim=self.config.hidden_dim
        ).to(self.device)
        
        self.critic_target = create_critic_network(
            architecture=self.config.architecture,
            state_dim=self.config.state_dim,
            action_dim=self.config.action_dim,
            hidden_dim=self.config.hidden_dim
        ).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        
        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.config.lr_actor)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.config.lr_critic)
        
        # OU Noise
        self.noise = OUNoise(self.config.action_dim, self.config.ou_noise_theta, self.config.ou_noise_sigma)
        self.noise_scale = 1.0
        
        self.total_it = 0
        
        logger.info(f"DDPGAgent initialized with {self.config.architecture} architecture")
    
    def select_action(self, state: np.ndarray, explore: bool = True) -> np.ndarray:
        """Select action using deterministic policy + OU noise"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            action = self.actor(state_tensor).cpu().numpy()[0]
        
        if explore:
            noise = self.noise.sample() * self.noise_scale
            action = np.clip(action + noise, -1.0, 1.0)
            
            # Decay noise
            self.noise_scale = max(self.config.noise_min, self.noise_scale * self.config.noise_decay)
        
        return action
    
    def train(self, replay_buffer: ReplayBuffer) -> Dict[str, float]:
        """Train DDPG agent"""
        self.total_it += 1
        
        # Sample batch
        state, action, reward, next_state, done = replay_buffer.sample(self.config.batch_size)
        
        state = torch.FloatTensor(state).to(self.device)
        action = torch.FloatTensor(action).to(self.device)
        reward = torch.FloatTensor(reward).unsqueeze(1).to(self.device)
        next_state = torch.FloatTensor(next_state).to(self.device)
        done = torch.FloatTensor(done).unsqueeze(1).to(self.device)
        
        # Update critic
        with torch.no_grad():
            next_action = self.actor_target(next_state)
            q1_next, q2_next = self.critic_target(next_state, next_action)
            q_next = torch.min(q1_next, q2_next)
            target_q = reward + (1 - done) * self.config.gamma * q_next
        
        q1, q2 = self.critic(state, action)
        critic_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # Update actor
        actor_loss = -self.critic.forward(state, self.actor(state))[0].mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # Soft update targets
        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(self.config.tau * param.data + (1 - self.config.tau) * target_param.data)
        
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.config.tau * param.data + (1 - self.config.tau) * target_param.data)
        
        return {
            'actor_loss': actor_loss.item(),
            'critic_loss': critic_loss.item(),
            'noise_scale': self.noise_scale
        }
    
    def save(self, filename: str):
        """Save agent"""
        torch.save({
            'actor': self.actor.state_dict(),
            'actor_target': self.actor_target.state_dict(),
            'critic': self.critic.state_dict(),
            'critic_target': self.critic_target.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict(),
            'noise_scale': self.noise_scale,
            'config': self.config,
        }, filename)
    
    def load(self, filename: str):
        """Load agent"""
        checkpoint = torch.load(filename, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor'])
        self.actor_target.load_state_dict(checkpoint['actor_target'])
        self.critic.load_state_dict(checkpoint['critic'])
        self.critic_target.load_state_dict(checkpoint['critic_target'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])
        self.noise_scale = checkpoint.get('noise_scale', 1.0)
