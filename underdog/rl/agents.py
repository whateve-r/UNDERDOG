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
    noise_clip: float = 0.1  # CRITICAL: Reduced from 0.5 to 0.1 for stability
    policy_freq: int = 2  # Delayed policy updates
    
    # Exploration
    expl_noise: float = 0.1
    use_cbrl: bool = True  # 🌪️ Use Chaos-Based RL (vs Gaussian noise)
    
    # Replay buffer
    buffer_size: int = 1_000_000
    batch_size: int = 256
    
    # Device
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'


class Actor(nn.Module):
    """
    Heterogeneous Actor network (policy)
    
    Supports multiple architectures:
    - MLP: Standard feedforward (batch, state_dim)
    - LSTM: Temporal sequences (batch, seq_len, feature_dim)
    - CNN1D: Convolutional patterns (batch, seq_len, feature_dim)
    - Transformer: Self-attention (batch, seq_len, feature_dim)
    - Attention: Attention mechanism (batch, feature_dim)
    
    Maps state → action in [-max_action, max_action]
    """
    
    def __init__(
        self, 
        state_dim: int, 
        action_dim: int, 
        hidden_dim: int = 256, 
        max_action: float = 1.0,
        architecture: str = 'MLP',
        sequence_length: int = 60,
        feature_dim: int = 31
    ):
        super().__init__()
        
        self.max_action = max_action
        self.architecture = architecture
        self.sequence_length = sequence_length
        self.feature_dim = feature_dim
        
        if architecture == 'LSTM':
            # LSTM for temporal sequences: (batch, seq_len, feature_dim) -> (batch, hidden_dim)
            self.lstm = nn.LSTM(
                input_size=feature_dim,
                hidden_size=hidden_dim // 2,  # Bidirectional doubles output
                num_layers=2,
                batch_first=True,
                dropout=0.1,
                bidirectional=True
            )
            self.fc = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, action_dim),
                nn.Tanh()
            )
        
        elif architecture == 'CNN1D':
            # CNN1D for local patterns: (batch, seq_len, feature_dim) -> (batch, hidden_dim)
            # Transpose to (batch, feature_dim, seq_len) for Conv1d
            self.conv = nn.Sequential(
                nn.Conv1d(feature_dim, 64, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv1d(64, 128, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool1d(1)  # (batch, 128, 1)
            )
            self.fc = nn.Sequential(
                nn.Linear(128, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, action_dim),
                nn.Tanh()
            )
        
        elif architecture == 'Transformer':
            # Transformer for self-attention: (batch, seq_len, feature_dim) -> (batch, hidden_dim)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=feature_dim,
                nhead=4,
                dim_feedforward=hidden_dim,
                batch_first=True
            )
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
            self.fc = nn.Sequential(
                nn.Linear(feature_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, action_dim),
                nn.Tanh()
            )
        
        elif architecture == 'Attention':
            # Attention mechanism: (batch, feature_dim) -> (batch, hidden_dim)
            self.attention = nn.MultiheadAttention(
                embed_dim=feature_dim,
                num_heads=4,
                batch_first=True
            )
            self.fc = nn.Sequential(
                nn.Linear(feature_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, action_dim),
                nn.Tanh()
            )
        
        else:  # MLP (default)
            # Standard feedforward: (batch, state_dim) -> (batch, action_dim)
            self.net = nn.Sequential(
                nn.Linear(state_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, action_dim),
                nn.Tanh()
            )
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with architecture-specific processing
        
        Args:
            state: State tensor
                - MLP/Attention: (batch, state_dim)
                - LSTM/CNN1D/Transformer: (batch, seq_len, feature_dim)
        
        Returns:
            action: Action tensor (batch, action_dim) in [-max_action, max_action]
        """
        if self.architecture == 'LSTM':
            # LSTM: (batch, seq_len, feature_dim) -> (batch, hidden_dim)
            lstm_out, _ = self.lstm(state)  # (batch, seq_len, hidden_dim)
            last_hidden = lstm_out[:, -1, :]  # Use last timestep
            return self.max_action * self.fc(last_hidden)
        
        elif self.architecture == 'CNN1D':
            # CNN1D: (batch, seq_len, feature_dim) -> (batch, feature_dim, seq_len)
            state_transposed = state.transpose(1, 2)
            conv_out = self.conv(state_transposed).squeeze(-1)  # (batch, 128)
            return self.max_action * self.fc(conv_out)
        
        elif self.architecture == 'Transformer':
            # Transformer: (batch, seq_len, feature_dim) -> (batch, hidden_dim)
            transformer_out = self.transformer(state)  # (batch, seq_len, feature_dim)
            pooled = transformer_out.mean(dim=1)  # Average pooling over sequence
            return self.max_action * self.fc(pooled)
        
        elif self.architecture == 'Attention':
            # Attention: (batch, feature_dim) -> (batch, hidden_dim)
            # Need to add sequence dimension for MultiheadAttention
            state_seq = state.unsqueeze(1)  # (batch, 1, feature_dim)
            attn_out, _ = self.attention(state_seq, state_seq, state_seq)
            attn_pooled = attn_out.squeeze(1)  # (batch, feature_dim)
            return self.max_action * self.fc(attn_pooled)
        
        else:  # MLP
            return self.max_action * self.net(state)


class Critic(nn.Module):
    """
    Heterogeneous Twin Critic networks (Q-functions)
    
    Supports multiple architectures for state encoding:
    - MLP: Standard feedforward (batch, state_dim)
    - LSTM: Temporal sequences (batch, seq_len, feature_dim)
    - CNN1D: Convolutional patterns (batch, seq_len, feature_dim)
    - Transformer: Self-attention (batch, seq_len, feature_dim)
    - Attention: Attention mechanism (batch, feature_dim)
    
    Maps (state, action) → Q-value with twin networks for reduced bias
    """
    
    def __init__(
        self, 
        state_dim: int, 
        action_dim: int, 
        hidden_dim: int = 256,
        architecture: str = 'MLP',
        sequence_length: int = 60,
        feature_dim: int = 31
    ):
        super().__init__()
        
        self.architecture = architecture
        self.sequence_length = sequence_length
        self.feature_dim = feature_dim
        
        # State encoder (shared architecture for both Q-networks)
        if architecture == 'LSTM':
            self.state_encoder1 = nn.LSTM(
                input_size=feature_dim,
                hidden_size=hidden_dim // 2,
                num_layers=2,
                batch_first=True,
                dropout=0.1,
                bidirectional=True
            )
            self.state_encoder2 = nn.LSTM(
                input_size=feature_dim,
                hidden_size=hidden_dim // 2,
                num_layers=2,
                batch_first=True,
                dropout=0.1,
                bidirectional=True
            )
            state_embed_dim = hidden_dim
        
        elif architecture == 'CNN1D':
            self.state_encoder1 = nn.Sequential(
                nn.Conv1d(feature_dim, 64, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv1d(64, 128, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool1d(1)
            )
            self.state_encoder2 = nn.Sequential(
                nn.Conv1d(feature_dim, 64, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv1d(64, 128, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool1d(1)
            )
            state_embed_dim = 128
        
        elif architecture == 'Transformer':
            encoder_layer1 = nn.TransformerEncoderLayer(
                d_model=feature_dim,
                nhead=4,
                dim_feedforward=hidden_dim,
                batch_first=True
            )
            self.state_encoder1 = nn.TransformerEncoder(encoder_layer1, num_layers=2)
            
            encoder_layer2 = nn.TransformerEncoderLayer(
                d_model=feature_dim,
                nhead=4,
                dim_feedforward=hidden_dim,
                batch_first=True
            )
            self.state_encoder2 = nn.TransformerEncoder(encoder_layer2, num_layers=2)
            state_embed_dim = feature_dim
        
        elif architecture == 'Attention':
            self.state_encoder1 = nn.MultiheadAttention(
                embed_dim=feature_dim,
                num_heads=4,
                batch_first=True
            )
            self.state_encoder2 = nn.MultiheadAttention(
                embed_dim=feature_dim,
                num_heads=4,
                batch_first=True
            )
            state_embed_dim = feature_dim
        
        else:  # MLP
            state_embed_dim = state_dim
            self.state_encoder1 = None
            self.state_encoder2 = None
        
        # Q1 network (takes encoded state + action)
        self.q1 = nn.Sequential(
            nn.Linear(state_embed_dim + action_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        # Q2 network (takes encoded state + action)
        self.q2 = nn.Sequential(
            nn.Linear(state_embed_dim + action_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def _encode_state(self, state: torch.Tensor, encoder_idx: int = 1) -> torch.Tensor:
        """
        Encode state using architecture-specific encoder
        
        Args:
            state: Raw state tensor
            encoder_idx: Which encoder to use (1 or 2)
        
        Returns:
            Encoded state vector (batch, state_embed_dim)
        """
        encoder = self.state_encoder1 if encoder_idx == 1 else self.state_encoder2
        
        if self.architecture == 'LSTM':
            # (batch, seq_len, feature_dim) -> (batch, hidden_dim)
            lstm_out, _ = encoder(state)
            return lstm_out[:, -1, :]  # Last timestep
        
        elif self.architecture == 'CNN1D':
            # (batch, seq_len, feature_dim) -> (batch, feature_dim, seq_len)
            state_transposed = state.transpose(1, 2)
            conv_out = encoder(state_transposed).squeeze(-1)  # (batch, 128)
            return conv_out
        
        elif self.architecture == 'Transformer':
            # (batch, seq_len, feature_dim) -> (batch, feature_dim)
            transformer_out = encoder(state)
            return transformer_out.mean(dim=1)  # Average pooling
        
        elif self.architecture == 'Attention':
            # (batch, feature_dim) -> (batch, feature_dim)
            state_seq = state.unsqueeze(1)  # Add sequence dimension
            attn_out, _ = encoder(state_seq, state_seq, state_seq)
            return attn_out.squeeze(1)
        
        else:  # MLP - no encoding needed
            return state
    
    def forward(self, state: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass (both Q-networks)
        
        Args:
            state: State tensor
                - MLP/Attention: (batch, state_dim)
                - LSTM/CNN1D/Transformer: (batch, seq_len, feature_dim)
            action: Action tensor (batch, action_dim)
        
        Returns:
            (Q1 value, Q2 value) - each (batch, 1)
        """
        # Encode state with respective encoders
        state_encoded_1 = self._encode_state(state, encoder_idx=1)
        state_encoded_2 = self._encode_state(state, encoder_idx=2)
        
        # Concatenate with action
        sa1 = torch.cat([state_encoded_1, action], dim=1)
        sa2 = torch.cat([state_encoded_2, action], dim=1)
        
        return self.q1(sa1), self.q2(sa2)
    
    def q1_forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Q1 only (for policy update)
        
        Args:
            state: State tensor (same format as forward)
            action: Action tensor (batch, action_dim)
        
        Returns:
            Q1 value (batch, 1)
        """
        state_encoded = self._encode_state(state, encoder_idx=1)
        sa = torch.cat([state_encoded, action], dim=1)
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
        # Use max_action=0.1 to match compliance_shield lot size limit
        max_lot_size = 0.1
        
        # Extract architecture params from config
        architecture = getattr(self.config, 'architecture', 'MLP')
        
        # Import factory functions
        from underdog.rl.models import create_policy_network, create_critic_network
        
        # Create policy networks using architecture factory
        self.actor = create_policy_network(
            architecture=architecture,
            state_dim=self.config.state_dim,
            action_dim=self.config.action_dim,
            hidden_dim=self.config.hidden_dim,
            max_action=max_lot_size
        ).to(self.device)
        
        self.actor_target = create_policy_network(
            architecture=architecture,
            state_dim=self.config.state_dim,
            action_dim=self.config.action_dim,
            hidden_dim=self.config.hidden_dim,
            max_action=max_lot_size
        ).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        
        # Create critic networks using architecture factory
        self.critic = create_critic_network(
            architecture=architecture,
            state_dim=self.config.state_dim,
            action_dim=self.config.action_dim,
            hidden_dim=self.config.hidden_dim
        ).to(self.device)
        
        self.critic_target = create_critic_network(
            architecture=architecture,
            state_dim=self.config.state_dim,
            action_dim=self.config.action_dim,
            hidden_dim=self.config.hidden_dim
        ).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        
        # Optimizers
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.config.lr_actor)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.config.lr_critic)
        
        # Training state
        self.total_it = 0
        
        # 🌪️ CBRL: Chaotic noise generator for intelligent exploration
        from underdog.rl.cbrl_noise import ChaoticNoiseGenerator
        self.chaotic_noise = ChaoticNoiseGenerator(
            x0=0.314159,  # π-inspired initialization
            output_range=(-1.0, 1.0)
        )
        self.use_cbrl = getattr(self.config, 'use_cbrl', True)  # Default: use CBRL
        
        # 🎯 Epsilon decay based on agent confidence (TD-error)
        self.td_error_history = []
        self.epsilon = 1.0  # Exploration strength [0, 1]
        self.epsilon_min = 0.1  # Minimum exploration
        
        logger.info(f"TD3Agent initialized (device: {self.device})")
        logger.info(f"  Architecture: {architecture}")
        logger.info(f"  State dim: {self.config.state_dim}, Action dim: {self.config.action_dim}")
        logger.info(f"  CBRL enabled: {self.use_cbrl}")
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
        # DEBUG: Check for NaN in state
        if np.isnan(state).any():
            logger.error(f"NaN detected in state! state={state}")
            return np.zeros(self.config.action_dim, dtype=np.float32)
        
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            action = self.actor(state_tensor).cpu().numpy()[0]
        
        # DEBUG: Check for NaN in action
        if np.isnan(action).any():
            logger.error(f"NaN detected in action! action={action}, state_mean={state.mean():.4f}")
            return np.zeros(self.config.action_dim, dtype=np.float32)
        
        if explore:
            # 🌪️ CBRL: Use chaotic noise instead of Gaussian
            # CRITICAL: Scale noise to match Actor's output range [-0.1, 0.1]
            max_lot_size = 0.1
            noise_scale = max_lot_size * self.config.expl_noise  # e.g., 0.1 * 0.1 = 0.01
            
            if self.use_cbrl:
                # Generate chaotic noise scaled by epsilon (adaptive exploration)
                noise = self.chaotic_noise.get_noise(shape=(self.config.action_dim,))
                noise = noise * self.epsilon * noise_scale
            else:
                # Standard Gaussian noise (baseline)
                noise = np.random.normal(0, noise_scale, size=self.config.action_dim)
            
            action = np.clip(action + noise, -max_lot_size, max_lot_size)
        
        return action
    
    def train(self, replay_buffer, batch_size: Optional[int] = None) -> Dict[str, float]:
        """
        Train agent on batch from replay buffer
        
        Args:
            replay_buffer: ReplayBuffer or PrioritizedReplayBuffer instance
            batch_size: Batch size (uses config if None)
        
        Returns:
            Training metrics: {'critic_loss': float, 'actor_loss': float}
        """
        self.total_it += 1
        batch_size = batch_size or self.config.batch_size
        
        # Sample batch (supports both regular and prioritized replay)
        use_per = isinstance(replay_buffer, PrioritizedReplayBuffer)
        
        if use_per:
            state, action, reward, next_state, done, indices, weights = replay_buffer.sample(batch_size)
            weights = torch.FloatTensor(weights).unsqueeze(1).to(self.device)
        else:
            state, action, reward, next_state, done = replay_buffer.sample(batch_size)
            indices = None
            weights = torch.ones((batch_size, 1)).to(self.device)
        
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
            # CRITICAL: Scale noise to match Actor's output range [-0.1, 0.1]
            # Noise should be proportional to action scale to avoid overwhelming the smoothing
            max_lot_size = 0.1  # Match Actor's max_action
            noise_scale = max_lot_size * self.config.policy_noise  # e.g., 0.1 * 0.2 = 0.02
            noise_clip = max_lot_size * self.config.noise_clip      # e.g., 0.1 * 0.1 = 0.01 (10% of action range)
            
            noise = (torch.randn_like(action) * noise_scale).clamp(-noise_clip, noise_clip)
            next_action = (self.actor_target(next_state) + noise).clamp(-max_lot_size, max_lot_size)
            
            # Compute target Q-value (min of two Q-networks)
            target_q1, target_q2 = self.critic_target(next_state, next_action)
            target_q = torch.min(target_q1, target_q2)
            target_q = reward + (1 - done) * self.config.gamma * target_q
        
        # Current Q-values
        current_q1, current_q2 = self.critic(state, action)
        
        # Critic loss (MSE with importance sampling weights for PER)
        td_error1 = current_q1 - target_q
        td_error2 = current_q2 - target_q
        
        critic_loss = (weights * (td_error1 ** 2)).mean() + (weights * (td_error2 ** 2)).mean()
        
        # 🎯 CBRL: Track TD-error for epsilon decay
        with torch.no_grad():
            td_error = torch.abs(td_error1).mean().item()
            td_errors_batch = torch.abs(td_error1).squeeze().cpu().numpy()
            
            self.td_error_history.append(td_error)
            
            # Update priorities if using PER
            if use_per and indices is not None:
                replay_buffer.update_priorities(indices, td_errors_batch)
            
            # Keep only last 100 TD-errors for moving average
            if len(self.td_error_history) > 100:
                self.td_error_history = self.td_error_history[-100:]
            
            # Update epsilon based on agent confidence (inverse of TD-error)
            # High TD-error → Low confidence → High exploration (epsilon)
            # Low TD-error → High confidence → Low exploration (epsilon)
            if len(self.td_error_history) >= 10:
                avg_td_error = np.mean(self.td_error_history[-10:])
                # Normalize TD-error to [0, 1] (typical range: 0.01 to 10.0)
                td_error_norm = np.clip(avg_td_error / 10.0, 0.0, 1.0)
                # Epsilon inversely proportional to confidence
                self.epsilon = max(self.epsilon_min, td_error_norm)
        
        # Optimize critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        metrics = {'critic_loss': critic_loss.item(), 'td_error': td_error, 'epsilon': self.epsilon}
        
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
    
    def state_dict(self) -> Dict:
        """Get agent state dict (for checkpointing)"""
        return {
            'actor': self.actor.state_dict(),
            'actor_target': self.actor_target.state_dict(),
            'critic': self.critic.state_dict(),
            'critic_target': self.critic_target.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict(),
            'total_it': self.total_it,
            'config': self.config,
        }
    
    def load_state_dict(self, state_dict: Dict):
        """Load agent state dict (from checkpoint)"""
        self.actor.load_state_dict(state_dict['actor'])
        self.actor_target.load_state_dict(state_dict['actor_target'])
        self.critic.load_state_dict(state_dict['critic'])
        self.critic_target.load_state_dict(state_dict['critic_target'])
        self.actor_optimizer.load_state_dict(state_dict['actor_optimizer'])
        self.critic_optimizer.load_state_dict(state_dict['critic_optimizer'])
        self.total_it = state_dict['total_it']
    
    def save(self, path: str):
        """Save agent checkpoint"""
        torch.save(self.state_dict(), path)
        logger.info(f"Agent saved to {path}")
    
    def load(self, path: str):
        """Load agent checkpoint"""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.load_state_dict(checkpoint)
        logger.info(f"Agent loaded from {path} (iteration {self.total_it})")


class ReplayBuffer:
    """
    Experience replay buffer
    
    Stores (state, action, reward, next_state, done) transitions
    
    Supports both flat states and sequential states:
    - Flat: state_dim=31 → shape=(max_size, 31)
    - Sequential: state_dim=(60, 31) → shape=(max_size, 60, 31)
    """
    
    def __init__(self, state_dim, action_dim: int, max_size: int = 1_000_000):
        """
        Initialize replay buffer
        
        Args:
            state_dim: State shape (int for flat, tuple for sequential)
            action_dim: Action dimension
            max_size: Maximum buffer size
        """
        self.max_size = max_size
        self.ptr = 0
        self.size = 0
        
        # Handle both flat and sequential state dimensions
        if isinstance(state_dim, int):
            state_shape = (max_size, state_dim)
        else:
            # state_dim is tuple (seq_len, feature_dim)
            state_shape = (max_size,) + tuple(state_dim)
        
        self.state = np.zeros(state_shape, dtype=np.float32)
        self.action = np.zeros((max_size, action_dim), dtype=np.float32)
        self.reward = np.zeros(max_size, dtype=np.float32)
        self.next_state = np.zeros(state_shape, dtype=np.float32)
        self.done = np.zeros(max_size, dtype=np.float32)
    
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


class PrioritizedReplayBuffer(ReplayBuffer):
    """
    Prioritized Experience Replay Buffer for CMDP
    
    Prioritizes:
    1. Catastrophic DD penalties (-1000 reward)
    2. Transitions near DD limit (high-risk states)
    3. Recent experiences (for non-stationary environments)
    
    Based on:
    - Schaul et al. 2015: Prioritized Experience Replay
    - Applied to CMDP constraint learning
    """
    
    def __init__(
        self, 
        state_dim: int, 
        action_dim: int, 
        max_size: int = 1_000_000,
        alpha: float = 0.6,  # Priority exponent (0=uniform, 1=full priority)
        beta: float = 0.4,   # Importance sampling (0=no correction, 1=full correction)
        beta_increment: float = 0.001  # Anneal beta to 1.0 over training
    ):
        """
        Initialize prioritized replay buffer
        
        Args:
            state_dim: State dimension
            action_dim: Action dimension
            max_size: Maximum buffer size
            alpha: Priority exponent (how much prioritization to use)
            beta: Importance sampling exponent (bias correction)
            beta_increment: Beta annealing rate
        """
        super().__init__(state_dim, action_dim, max_size)
        
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.max_priority = 1.0
        
        # Priority storage
        self.priorities = np.zeros(max_size, dtype=np.float32)
    
    def add(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: bool
    ):
        """Add transition with maximum priority (will be sampled soon)"""
        super().add(state, action, reward, next_state, done)
        
        # Assign maximum priority to new transition
        # This ensures catastrophic failures are sampled immediately
        self.priorities[self.ptr] = self.max_priority
    
    def sample(self, batch_size: int) -> Tuple[np.ndarray, ...]:
        """
        Sample batch with priorities
        
        Returns:
            (states, actions, rewards, next_states, dones, indices, weights)
            - indices: for updating priorities
            - weights: importance sampling weights
        """
        if self.size < batch_size:
            batch_size = self.size
        
        # Calculate sampling probabilities from priorities
        priorities = self.priorities[:self.size] ** self.alpha
        probs = priorities / priorities.sum()
        
        # Count non-zero probabilities
        non_zero_count = np.count_nonzero(probs)
        
        # Always use replace=True to avoid numpy sampling errors
        # This is standard practice in PER implementations
        indices = np.random.choice(self.size, batch_size, p=probs, replace=True)
        
        # Importance sampling weights (for bias correction)
        weights = (self.size * probs[indices]) ** (-self.beta)
        weights = weights / weights.max()  # Normalize
        
        # Anneal beta towards 1.0
        self.beta = min(1.0, self.beta + self.beta_increment)
        
        return (
            self.state[indices],
            self.action[indices],
            self.reward[indices],
            self.next_state[indices],
            self.done[indices],
            indices,
            weights
        )
    
    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray):
        """
        Update priorities based on TD-errors
        
        Args:
            indices: Batch indices from sample()
            td_errors: Absolute TD-errors |Q - target_Q|
        """
        # Priority = |TD-error| + epsilon (avoid zero priority)
        priorities = np.abs(td_errors) + 1e-6
        
        # CRITICAL: Boost priority for catastrophic DD penalties
        for i, idx in enumerate(indices):
            reward = self.reward[idx]
            
            # If catastrophic penalty, boost priority 10x
            if reward <= -1000:
                priorities[i] *= 10.0
            
            # If near DD limit (high risk state), boost priority 5x
            # Check if next_state has high DD ratio (dims [19] or [20])
            next_state = self.next_state[idx]
            if len(next_state) >= 21:
                daily_dd_ratio = next_state[19]  # [19]: Daily DD used / limit
                total_dd_ratio = next_state[20]  # [20]: Total DD used / limit
                
                if daily_dd_ratio > 0.8 or total_dd_ratio > 0.8:
                    priorities[i] *= 5.0
        
        self.priorities[indices] = priorities
        self.max_priority = max(self.max_priority, priorities.max())


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
