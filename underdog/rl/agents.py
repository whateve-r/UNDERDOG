"""
Deep Reinforcement Learning Agents - TD3 Implementation

TD3 (Twin Delayed Deep Deterministic Policy Gradient) agent for forex trading.

Papers:
- arXiv:2510.10526v1: LLM + RL Integration
- Fujimoto et al. 2018: "Addressing Function Approximation Error in Actor-Critic Methods"

Key Features:
- Continuous action space: [position_size, entry/exit decision]
- 14-dimensional state space (StateVectorBuilder)
- Twin Q-networks (reduces overestimation)
- Delayed policy updates (improves stability)
- Target policy smoothing (noise for exploration)

Action Space:
- Action[0]: Position size [-1, 1] (short to long)
- Action[1]: Entry/Exit [-1, 1] (close to open)

Reward Function:
- Primary: Sharpe Ratio (rolling 30 periods)
- Penalty: Drawdown (exponential penalty)
- Bonus: Prop Firm compliance (no violations)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class TD3Config:
    """TD3 Agent configuration"""
    # Network architecture
    state_dim: int = 14
    action_dim: int = 2
    hidden_dim: int = 256
    
    # Training hyperparameters
    lr_actor: float = 3e-4
    lr_critic: float = 3e-4
    gamma: float = 0.99  # Discount factor
    tau: float = 0.005  # Soft update rate
    policy_noise: float = 0.2
    noise_clip: float = 0.5
    policy_freq: int = 2  # Delayed policy updates
    
    # Exploration
    expl_noise: float = 0.1
    
    # Replay buffer
    buffer_size: int = 1_000_000
    batch_size: int = 256
    
    # Device
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'


class Actor(nn.Module):
    """
    Actor network (policy)
    
    Maps state → action
    Input: state vector (14,)
    Output: action vector (2,) in [-1, 1]
    """
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh()  # Output in [-1, 1]
        )
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            state: State tensor (batch_size, state_dim)
        
        Returns:
            Action tensor (batch_size, action_dim) in [-1, 1]
        """
        return self.net(state)


class Critic(nn.Module):
    """
    Twin Critic networks (Q-functions)
    
    Maps (state, action) → Q-value
    Two networks to reduce overestimation bias
    """
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        
        # Q1 network
        self.q1 = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        # Q2 network
        self.q2 = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, state: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass (both Q-networks)
        
        Args:
            state: State tensor (batch_size, state_dim)
            action: Action tensor (batch_size, action_dim)
        
        Returns:
            (Q1 value, Q2 value)
        """
        sa = torch.cat([state, action], dim=1)
        return self.q1(sa), self.q2(sa)
    
    def q1_forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Q1 only (for policy update)"""
        sa = torch.cat([state, action], dim=1)
        return self.q1(sa)


class TD3Agent:
    """
    TD3 Agent for forex trading
    
    Usage:
        agent = TD3Agent(config)
        action = agent.select_action(state, explore=True)
        agent.train(replay_buffer)
    """
    
    def __init__(self, config: Optional[TD3Config] = None):
        """
        Initialize TD3 agent
        
        Args:
            config: TD3 configuration (uses defaults if None)
        """
        self.config = config or TD3Config()
        self.device = torch.device(self.config.device)
        
        # Networks
        self.actor = Actor(
            self.config.state_dim,
            self.config.action_dim,
            self.config.hidden_dim
        ).to(self.device)
        
        self.actor_target = Actor(
            self.config.state_dim,
            self.config.action_dim,
            self.config.hidden_dim
        ).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        
        self.critic = Critic(
            self.config.state_dim,
            self.config.action_dim,
            self.config.hidden_dim
        ).to(self.device)
        
        self.critic_target = Critic(
            self.config.state_dim,
            self.config.action_dim,
            self.config.hidden_dim
        ).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        
        # Optimizers
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.config.lr_actor)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.config.lr_critic)
        
        # Training state
        self.total_it = 0
        
        logger.info(f"TD3Agent initialized (device: {self.device})")
        logger.info(f"  State dim: {self.config.state_dim}, Action dim: {self.config.action_dim}")
        logger.info(f"  Hidden dim: {self.config.hidden_dim}")
    
    def select_action(self, state: np.ndarray, explore: bool = True) -> np.ndarray:
        """
        Select action for given state
        
        Args:
            state: State vector (state_dim,)
            explore: If True, add exploration noise
        
        Returns:
            Action vector (action_dim,) in [-1, 1]
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            action = self.actor(state_tensor).cpu().numpy()[0]
        
        if explore:
            noise = np.random.normal(0, self.config.expl_noise, size=self.config.action_dim)
            action = np.clip(action + noise, -1, 1)
        
        return action
    
    def train(self, replay_buffer, batch_size: Optional[int] = None) -> Dict[str, float]:
        """
        Train agent on batch from replay buffer
        
        Args:
            replay_buffer: ReplayBuffer instance
            batch_size: Batch size (uses config if None)
        
        Returns:
            Training metrics: {'critic_loss': float, 'actor_loss': float}
        """
        self.total_it += 1
        batch_size = batch_size or self.config.batch_size
        
        # Sample batch
        state, action, reward, next_state, done = replay_buffer.sample(batch_size)
        
        state = torch.FloatTensor(state).to(self.device)
        action = torch.FloatTensor(action).to(self.device)
        reward = torch.FloatTensor(reward).unsqueeze(1).to(self.device)
        next_state = torch.FloatTensor(next_state).to(self.device)
        done = torch.FloatTensor(done).unsqueeze(1).to(self.device)
        
        # ====================================================================
        # Train Critic
        # ====================================================================
        
        with torch.no_grad():
            # Target policy smoothing (add noise to target action)
            noise = (torch.randn_like(action) * self.config.policy_noise).clamp(
                -self.config.noise_clip, self.config.noise_clip
            )
            next_action = (self.actor_target(next_state) + noise).clamp(-1, 1)
            
            # Compute target Q-value (min of two Q-networks)
            target_q1, target_q2 = self.critic_target(next_state, next_action)
            target_q = torch.min(target_q1, target_q2)
            target_q = reward + (1 - done) * self.config.gamma * target_q
        
        # Current Q-values
        current_q1, current_q2 = self.critic(state, action)
        
        # Critic loss (MSE)
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)
        
        # Optimize critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        metrics = {'critic_loss': critic_loss.item()}
        
        # ====================================================================
        # Train Actor (Delayed)
        # ====================================================================
        
        if self.total_it % self.config.policy_freq == 0:
            # Actor loss (maximize Q1)
            actor_loss = -self.critic.q1_forward(state, self.actor(state)).mean()
            
            # Optimize actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            
            metrics['actor_loss'] = actor_loss.item()
            
            # Soft update target networks
            self._soft_update(self.actor, self.actor_target)
            self._soft_update(self.critic, self.critic_target)
        
        return metrics
    
    def _soft_update(self, source: nn.Module, target: nn.Module):
        """Soft update target network parameters"""
        for param, target_param in zip(source.parameters(), target.parameters()):
            target_param.data.copy_(
                self.config.tau * param.data + (1 - self.config.tau) * target_param.data
            )
    
    def save(self, path: str):
        """Save agent checkpoint"""
        torch.save({
            'actor': self.actor.state_dict(),
            'actor_target': self.actor_target.state_dict(),
            'critic': self.critic.state_dict(),
            'critic_target': self.critic_target.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict(),
            'total_it': self.total_it,
            'config': self.config,
        }, path)
        logger.info(f"Agent saved to {path}")
    
    def load(self, path: str):
        """Load agent checkpoint"""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        
        self.actor.load_state_dict(checkpoint['actor'])
        self.actor_target.load_state_dict(checkpoint['actor_target'])
        self.critic.load_state_dict(checkpoint['critic'])
        self.critic_target.load_state_dict(checkpoint['critic_target'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])
        self.total_it = checkpoint['total_it']
        
        logger.info(f"Agent loaded from {path} (iteration {self.total_it})")


class ReplayBuffer:
    """
    Experience replay buffer
    
    Stores (state, action, reward, next_state, done) transitions
    """
    
    def __init__(self, state_dim: int, action_dim: int, max_size: int = 1_000_000):
        """Initialize replay buffer"""
        self.max_size = max_size
        self.ptr = 0
        self.size = 0
        
        self.state = np.zeros((max_size, state_dim))
        self.action = np.zeros((max_size, action_dim))
        self.reward = np.zeros(max_size)
        self.next_state = np.zeros((max_size, state_dim))
        self.done = np.zeros(max_size)
    
    def add(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: bool
    ):
        """Add transition to buffer"""
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.reward[self.ptr] = reward
        self.next_state[self.ptr] = next_state
        self.done[self.ptr] = done
        
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)
    
    def sample(self, batch_size: int) -> Tuple[np.ndarray, ...]:
        """Sample random batch"""
        idx = np.random.randint(0, self.size, size=batch_size)
        
        return (
            self.state[idx],
            self.action[idx],
            self.reward[idx],
            self.next_state[idx],
            self.done[idx]
        )
    
    def __len__(self) -> int:
        return self.size


# ============================================================================
# CLI Test
# ============================================================================

if __name__ == "__main__":
    print("Testing TD3Agent...")
    
    # Create agent
    config = TD3Config(device='cpu')  # Force CPU for test
    agent = TD3Agent(config)
    
    # Test action selection
    state = np.random.randn(14)
    action = agent.select_action(state, explore=True)
    
    print(f"\nState: {state[:3]}... (14 dims)")
    print(f"Action: {action} (shape: {action.shape})")
    print(f"  Position size: {action[0]:.3f} [-1=short, +1=long]")
    print(f"  Entry/Exit: {action[1]:.3f} [-1=close, +1=open]")
    
    # Test training
    replay_buffer = ReplayBuffer(state_dim=14, action_dim=2, max_size=10000)
    
    # Fill buffer with random transitions
    for _ in range(1000):
        s = np.random.randn(14)
        a = agent.select_action(s, explore=True)
        r = np.random.randn()
        s_next = np.random.randn(14)
        done = np.random.rand() > 0.95
        replay_buffer.add(s, a, r, s_next, done)
    
    print(f"\nReplay buffer size: {len(replay_buffer)}")
    
    # Train for 10 iterations
    print("\nTraining for 10 iterations...")
    for i in range(10):
        metrics = agent.train(replay_buffer)
        if 'actor_loss' in metrics:
            print(f"  Iteration {i+1}: Critic Loss = {metrics['critic_loss']:.4f}, "
                  f"Actor Loss = {metrics['actor_loss']:.4f}")
        else:
            print(f"  Iteration {i+1}: Critic Loss = {metrics['critic_loss']:.4f} (actor delayed)")
    
    # Test save/load
    agent.save('test_td3_agent.pth')
    agent.load('test_td3_agent.pth')
    
    print("\n✓ Test complete")
