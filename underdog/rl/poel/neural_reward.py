"""
Neural Reward Functions (NRF) for Autonomous Skill Discovery.

Based on the DISCOVER algorithm for unsupervised skill learning.
Enables Open-Ended Learning by generating curriculum of increasing complexity.

Key Idea:
- Train reward network R_ψ to assign high reward to frontier states (unvisited)
- Train policy π_θ to maximize R_ψ reward → discovers new skills
- Iterate: R_ψ pushes π_θ toward unexplored regions → OEL emerges
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import deque
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class NRFConfig:
    """Configuration for Neural Reward Function."""
    state_dim: int
    hidden_dims: List[int] = None  # [128, 64] default
    learning_rate: float = 3e-4
    buffer_size: int = 50000  # Visited state buffer
    batch_size: int = 256
    reward_scale: float = 10.0  # Scale NRF rewards to match PnL magnitude
    frontier_threshold: float = 0.7  # Distance to consider state "novel"
    update_frequency: int = 100  # Update R_ψ every N steps
    
    def __post_init__(self):
        if self.hidden_dims is None:
            self.hidden_dims = [128, 64]


class RewardNetwork(nn.Module):
    """
    Neural Reward Function R_ψ(s).
    
    Architecture: MLP that maps state → scalar reward
    Training: Maximize reward for frontier states, minimize for visited states
    """
    
    def __init__(self, state_dim: int, hidden_dims: List[int] = [128, 64]):
        """
        Args:
            state_dim: Dimension of state space
            hidden_dims: List of hidden layer sizes
        """
        super().__init__()
        
        layers = []
        prev_dim = state_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.LayerNorm(hidden_dim))  # Stabilize training
            prev_dim = hidden_dim
        
        # Output layer: scalar reward
        layers.append(nn.Linear(prev_dim, 1))
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize network weights."""
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
            if module.bias is not None:
                nn.init.constant_(module.bias, 0.0)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Compute reward for state.
        
        Args:
            state: (batch_size, state_dim) tensor
        
        Returns:
            rewards: (batch_size, 1) tensor
        """
        return self.network(state)
    
    def compute_reward(self, state: np.ndarray) -> float:
        """
        Compute reward for single state (inference).
        
        Args:
            state: (state_dim,) numpy array
        
        Returns:
            Scalar reward value
        """
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            reward = self.forward(state_tensor)
            return reward.item()


class VisitedStateBuffer:
    """
    Buffer of visited states for NRF training.
    
    Used to create negative samples: states that should get low reward
    because they're already explored.
    """
    
    def __init__(self, max_size: int = 50000, state_dim: int = 31):
        """
        Args:
            max_size: Maximum buffer capacity
            state_dim: Dimension of state vectors
        """
        self.max_size = max_size
        self.state_dim = state_dim
        self.buffer = deque(maxlen=max_size)
        
        # For efficient distance calculations
        self.states_array: Optional[np.ndarray] = None
        self.dirty = True  # Flag to rebuild array
    
    def add(self, state: np.ndarray):
        """Add state to buffer. Always flatten to 1D."""
        # CRITICAL: Flatten state to ensure consistent shape
        flattened_state = state.flatten()
        self.buffer.append(flattened_state.copy())
        self.dirty = True
    
    def sample(self, batch_size: int) -> np.ndarray:
        """
        Sample random states from buffer.
        
        Args:
            batch_size: Number of states to sample
        
        Returns:
            (batch_size, state_dim) array
        """
        if len(self.buffer) == 0:
            return np.zeros((0, self.state_dim))
        
        n_samples = min(batch_size, len(self.buffer))
        indices = np.random.choice(len(self.buffer), size=n_samples, replace=False)
        
        return np.array([self.buffer[i] for i in indices])
    
    def compute_distances(self, state: np.ndarray) -> np.ndarray:
        """
        Compute L2 distances from state to all buffered states.
        
        Args:
            state: (state_dim,) array or (time, state_dim) array
        
        Returns:
            (buffer_size,) array of distances
        """
        if len(self.buffer) == 0:
            return np.array([])
        
        # CRITICAL: Flatten state to ensure consistent shape
        state_flat = state.flatten()
        
        # Rebuild array if needed
        if self.dirty or self.states_array is None:
            try:
                self.states_array = np.array(list(self.buffer))
                self.dirty = False
            except ValueError:
                # RECOVERY: Buffer has inhomogeneous shapes - clear it
                logger.warning(f"[NRF] Clearing buffer due to shape inconsistency (size={len(self.buffer)})")
                self.buffer.clear()
                self.states_array = None
                self.dirty = True
                return np.array([])
        
        # L2 distance
        distances = np.linalg.norm(self.states_array - state_flat, axis=1)
        return distances
    
    def is_frontier_state(self, state: np.ndarray, threshold: float = 0.7) -> bool:
        """
        Check if state is on the frontier (far from visited states).
        
        Args:
            state: State vector
            threshold: Distance threshold for "frontier" classification
        
        Returns:
            True if state is novel (frontier)
        """
        if len(self.buffer) < 10:
            return True  # Early episodes: everything is novel
        
        distances = self.compute_distances(state)
        
        # Handle empty distances (buffer was cleared due to shape mismatch)
        if len(distances) == 0:
            return True  # Treat as novel when buffer is empty
        
        min_distance = np.min(distances)
        
        # Normalize by state dimension
        normalized_distance = min_distance / np.sqrt(self.state_dim)
        
        return normalized_distance > threshold
    
    def __len__(self):
        return len(self.buffer)


class NeuralRewardFunction:
    """
    Main NRF training and inference system.
    
    Implements the DISCOVER algorithm:
    1. Collect states during policy execution
    2. Label visited states as negative samples
    3. Label frontier states as positive samples
    4. Train R_ψ to maximize reward for frontier, minimize for visited
    5. Use R_ψ rewards to train policy → skill discovery
    """
    
    def __init__(self, config: NRFConfig):
        """
        Args:
            config: NRFConfig object
        """
        self.config = config
        
        # Reward network
        self.reward_net = RewardNetwork(
            state_dim=config.state_dim,
            hidden_dims=config.hidden_dims,
        )
        
        # Optimizer
        self.optimizer = torch.optim.Adam(
            self.reward_net.parameters(),
            lr=config.learning_rate,
        )
        
        # State buffer
        self.visited_buffer = VisitedStateBuffer(
            max_size=config.buffer_size,
            state_dim=config.state_dim,
        )
        
        # Training statistics
        self.update_count = 0
        self.frontier_count = 0
        self.visited_count = 0
        
        # Loss history
        self.loss_history = deque(maxlen=100)
    
    def compute_reward(self, state: np.ndarray) -> float:
        """
        Compute NRF reward for state (use during policy training).
        
        Args:
            state: State vector
        
        Returns:
            Scaled reward value
        """
        reward = self.reward_net.compute_reward(state)
        return reward * self.config.reward_scale
    
    def add_state(self, state: np.ndarray):
        """
        Add state to visited buffer.
        
        Call this after every environment step during NRF mode.
        
        Args:
            state: State vector
        """
        self.visited_buffer.add(state)
        
        # Check if frontier
        if self.visited_buffer.is_frontier_state(state, self.config.frontier_threshold):
            self.frontier_count += 1
        else:
            self.visited_count += 1
    
    def update_reward_function(self) -> Dict[str, float]:
        """
        Update R_ψ network using visited state buffer.
        
        Training objective:
        - Maximize reward for frontier states (far from visited)
        - Minimize reward for visited states (already explored)
        
        Returns:
            Dict with training metrics
        """
        if len(self.visited_buffer) < self.config.batch_size:
            return {'loss': 0.0, 'skipped': True}
        
        self.update_count += 1
        
        # Sample visited states (negative samples)
        visited_states = self.visited_buffer.sample(self.config.batch_size // 2)
        visited_tensor = torch.FloatTensor(visited_states)
        
        # Generate frontier states (positive samples)
        # Strategy: Add noise to visited states to push toward unexplored regions
        noise = np.random.randn(*visited_states.shape) * 0.5
        frontier_states = visited_states + noise
        frontier_tensor = torch.FloatTensor(frontier_states)
        
        # Compute rewards
        visited_rewards = self.reward_net(visited_tensor)
        frontier_rewards = self.reward_net(frontier_tensor)
        
        # Loss: Maximize gap between frontier and visited
        # L = -mean(R(frontier)) + mean(R(visited))
        # Equivalent to: push frontier rewards high, visited rewards low
        loss_frontier = -frontier_rewards.mean()
        loss_visited = visited_rewards.mean()
        
        # Add regularization to prevent reward explosion
        reg_loss = 0.01 * (frontier_rewards.pow(2).mean() + visited_rewards.pow(2).mean())
        
        total_loss = loss_frontier + loss_visited + reg_loss
        
        # Backprop
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.reward_net.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        # Log
        self.loss_history.append(total_loss.item())
        
        return {
            'loss': total_loss.item(),
            'frontier_reward': -loss_frontier.item(),
            'visited_reward': loss_visited.item(),
            'reg_loss': reg_loss.item(),
            'buffer_size': len(self.visited_buffer),
            'update_count': self.update_count,
        }
    
    def should_update(self, step: int) -> bool:
        """
        Check if R_ψ should be updated at current step.
        
        Args:
            step: Current training step
        
        Returns:
            True if update should occur
        """
        return step % self.config.update_frequency == 0
    
    def get_statistics(self) -> Dict[str, any]:
        """Get NRF training statistics."""
        return {
            'update_count': self.update_count,
            'frontier_states': self.frontier_count,
            'visited_states': self.visited_count,
            'buffer_size': len(self.visited_buffer),
            'avg_loss': np.mean(self.loss_history) if self.loss_history else 0.0,
            'frontier_ratio': (
                self.frontier_count / max(1, self.frontier_count + self.visited_count)
            ),
        }
    
    def reset_statistics(self):
        """Reset episode-level statistics."""
        self.frontier_count = 0
        self.visited_count = 0
    
    def save(self, path: str):
        """Save reward network weights."""
        torch.save({
            'model_state_dict': self.reward_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'update_count': self.update_count,
        }, path)
    
    def load(self, path: str):
        """Load reward network weights."""
        checkpoint = torch.load(path)
        self.reward_net.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.update_count = checkpoint['update_count']


class NRFSkillDiscovery:
    """
    Orchestrates the full NRF skill discovery cycle.
    
    Alternates between:
    1. Skill Generation: Train π_θ with R_ψ rewards
    2. Reward Update: Train R_ψ to push toward frontier
    """
    
    def __init__(
        self,
        nrf: NeuralRewardFunction,
        skill_generation_epochs: int = 5,
        reward_update_frequency: int = 100,
    ):
        """
        Args:
            nrf: NeuralRewardFunction instance
            skill_generation_epochs: How many epochs to train policy per cycle
            reward_update_frequency: Update R_ψ every N steps
        """
        self.nrf = nrf
        self.skill_generation_epochs = skill_generation_epochs
        self.reward_update_frequency = reward_update_frequency
        
        # Cycle tracking
        self.cycle_count = 0
        self.current_phase = 'skill_generation'  # or 'reward_update'
    
    def start_skill_generation(self) -> Dict[str, any]:
        """
        Start skill generation phase.
        
        Returns:
            Dict with phase config
        """
        self.current_phase = 'skill_generation'
        
        return {
            'phase': 'skill_generation',
            'epochs': self.skill_generation_epochs,
            'use_nrf_rewards': True,
            'alpha': 0.0,  # 100% NRF rewards, 0% PnL
            'message': f'Cycle {self.cycle_count}: Generating new skill with R_ψ rewards',
        }
    
    def start_reward_update(self) -> Dict[str, any]:
        """
        Start reward update phase.
        
        Returns:
            Dict with phase config
        """
        self.current_phase = 'reward_update'
        
        return {
            'phase': 'reward_update',
            'update_frequency': self.reward_update_frequency,
            'message': f'Cycle {self.cycle_count}: Updating R_ψ to push frontier',
        }
    
    def step(self, training_step: int) -> Optional[Dict[str, any]]:
        """
        Execute one step of NRF cycle.
        
        Args:
            training_step: Current training step
        
        Returns:
            Update metrics if R_ψ was updated, else None
        """
        if self.current_phase == 'skill_generation':
            # Policy is being trained with NRF rewards
            # Just collect states
            return None
        
        elif self.current_phase == 'reward_update':
            # Update R_ψ periodically
            if self.nrf.should_update(training_step):
                metrics = self.nrf.update_reward_function()
                return metrics
        
        return None
    
    def end_cycle(self) -> Dict[str, any]:
        """
        Complete current cycle and prepare for next.
        
        Returns:
            Cycle summary
        """
        self.cycle_count += 1
        stats = self.nrf.get_statistics()
        
        return {
            'cycle': self.cycle_count,
            'completed_phase': self.current_phase,
            'nrf_stats': stats,
            'message': f'Cycle {self.cycle_count} complete. Discovered skill ready for evaluation.',
        }


def create_nrf_system(state_dim: int, **kwargs) -> NRFSkillDiscovery:
    """
    Factory function to create complete NRF system.
    
    Args:
        state_dim: Dimension of state space
        **kwargs: Override NRFConfig defaults
    
    Returns:
        NRFSkillDiscovery instance ready for training
    """
    config = NRFConfig(state_dim=state_dim, **kwargs)
    nrf = NeuralRewardFunction(config)
    discovery = NRFSkillDiscovery(nrf)
    
    return discovery
