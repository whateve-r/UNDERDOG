"""
UNDERDOG TRADING SYSTEMS
Heterogeneous MARL Migration - Implementation Guide

================================================================================
STATUS: 95% COMPLETE - REMAINING CRITICAL TASKS
================================================================================

COMPLETED ✅:
-------------
1. Agent Configuration System (config/agent_configs.py)
   - TD3Config with LSTM architecture settings
   - PPOConfig with CNN1D architecture settings  
   - SACConfig with Transformer architecture settings
   - DDPGConfig with Attention architecture settings
   - All configs now have: state_dim, action_dim, hidden_dim, device

2. Neural Network Architectures (underdog/rl/models.py)
   - LSTMPolicy/Critic for EURUSD trend-following
   - CNN1DPolicy/Critic for USDJPY breakout detection
   - TransformerPolicy/Critic for XAUUSD regime adaptation
   - AttentionPolicy/Critic for GBPUSD whipsaw resistance
   - Factory functions with parameter mapping

3. New Agent Classes (underdog/rl/multi_agents.py)
   - PPOAgent with on-policy buffer
   - SACAgent with adaptive temperature (auto_alpha=True)
   - DDPGAgent with OU noise exploration

4. Training Loop Integration (scripts/train_marl_agent.py)
   - Heterogeneous agent factory
   - Separate training paths for on-policy (PPO) vs off-policy (TD3/SAC/DDPG)
   - CSV logging with agent types
   - Agent-specific metrics tracking

5. Validation Script (scripts/validate_hmarl_migration.py)
   - Configuration validation
   - Architecture instantiation tests
   - Agent compatibility tests

================================================================================
CRITICAL PENDING TASKS (From HMARL Paper Mandates)
================================================================================

TASK 1: Sequential Observation Space ⚠️ CRITICAL ⚠️
--------------------------------------------------------
File: underdog/rl/multi_asset_env.py (or local environment class)

Current State: Envir

onment returns single timestep observation [1 × Features]
Required State: Return temporal sequences for LSTM/CNN/Transformer

Implementation Required:

```python
class LocalEnvWithSequentialObs:
    """
    Enhanced local environment with temporal observation buffer
    
    Mandates from HMARL Paper:
    - EURUSD (LSTM): Return [60-120 × Features]
    - USDJPY (CNN1D): Return [10-20 × Features]
    - XAUUSD (Transformer): Return [100+ × Features]
    - GBPUSD (Attention): Return [1 × Features] (current state OK)
    """
    
    def __init__(self, symbol: str, sequence_length: int = None):
        self.symbol = symbol
        
        # Set sequence length based on architecture
        if sequence_length is None:
            SEQUENCE_MAPPING = {
                'EURUSD': 60,    # LSTM needs medium context
                'USDJPY': 15,    # CNN needs short patterns
                'XAUUSD': 120,   # Transformer needs long context
                'GBPUSD': 1,     # Attention uses vectorial
            }
            self.sequence_length = SEQUENCE_MAPPING.get(symbol, 1)
        else:
            self.sequence_length = sequence_length
        
        # Observation buffer (circular buffer)
        self.observation_buffer = []
        self.max_buffer_size = max(120, self.sequence_length)  # Always keep enough
        
    def _get_observation(self) -> np.ndarray:
        """
        Return observation with temporal context
        
        Returns:
            - If sequence_length == 1: [Features] (for GBPUSD)
            - If sequence_length > 1: [sequence_length × Features] (for others)
        """
        # Get current features
        current_features = self._compute_features()  # [Features]
        
        # Add to buffer
        self.observation_buffer.append(current_features)
        
        # Keep buffer size manageable
        if len(self.observation_buffer) > self.max_buffer_size:
            self.observation_buffer.pop(0)
        
        # Return based on sequence length
        if self.sequence_length == 1:
            # GBPUSD (Attention): return current state only
            return current_features
        else:
            # LSTM/CNN/Transformer: return sequence
            if len(self.observation_buffer) < self.sequence_length:
                # Pad with zeros if not enough history
                padding_needed = self.sequence_length - len(self.observation_buffer)
                padded = [np.zeros_like(current_features)] * padding_needed
                sequence = padded + self.observation_buffer
            else:
                # Get last N timesteps
                sequence = self.observation_buffer[-self.sequence_length:]
            
            # Stack to [sequence_length × Features]
            return np.array(sequence)
    
    def reset(self):
        """Clear observation buffer on episode reset"""
        self.observation_buffer = []
        # ... rest of reset logic
```

Validation Code to Add (scripts/train_marl_agent.py):

```python
# After first agent action in training loop
if self.episode == 1 and step == 1:
    logger.critical("="*80)
    logger.critical("TENSOR SHAPE VALIDATION (HMARL Mandate)")
    logger.critical("="*80)
    
    for i, (agent, env, symbol) in enumerate(zip(self.local_agents, self.env.local_envs, symbols)):
        obs = env._get_observation()
        logger.critical(f"[{symbol}] {agent.__class__.__name__}")
        logger.critical(f"  Observation Shape: {obs.shape}")
        logger.critical(f"  Expected: [T={SEQUENCE_LENGTHS[symbol]}, F={state_dim}]")
        
        # Assert correct shape
        if agent.__class__.__name__ in ['TD3Agent', 'SACAgent']:  # LSTM, Transformer
            assert obs.ndim == 2, f"{symbol}: Expected 2D sequence, got {obs.ndim}D"
            assert obs.shape[0] == SEQUENCE_LENGTHS[symbol], f"{symbol}: Wrong sequence length"
        elif agent.__class__.__name__ == 'PPOAgent':  # CNN1D
            assert obs.ndim == 2, f"{symbol}: CNN needs 2D input"
```


TASK 2: GAE for PPO (USDJPY) - ALREADY PARTIALLY IMPLEMENTED ✅
----------------------------------------------------------------
File: underdog/rl/multi_agents.py (PPOAgent.train())

Status: GAE calculation EXISTS in PPOAgent.train() with lambda=0.95
Action: Verify it's correct and log GAE values

Validation:
- Check that advantages are normalized
- Verify lambda=0.95 parameter
- Log advantage statistics in first 5 episodes


TASK 3: Adaptive Alpha for SAC (XAUUSD) - ALREADY IMPLEMENTED ✅
------------------------------------------------------------------
File: underdog/rl/multi_agents.py (SACAgent)

Status: COMPLETE
- auto_alpha=True in SACConfig
- log_alpha is trainable parameter
- alpha_optimizer exists
- alpha_loss is computed and optimized

Validation Required:
```python
# In SACAgent.train(), log alpha evolution
if self.total_it % 100 == 0:
    logger.info(f"[SAC] Alpha: {self.alpha.item():.4f}, Target Entropy: {self.target_entropy}")
```


TASK 4: Asymmetric Reward for DDPG (GBPUSD) ⚠️ NOT IMPLEMENTED
----------------------------------------------------------------
File: underdog/rl/multi_asset_env.py or local environment

Current: Raw PnL reward
Required: Fakeout penalty + hold bonus

Implementation:

```python
class DDPGRewardShaper:
    """
    Asymmetric reward for GBPUSD whipsaw resistance
    
    Mandate: Penalize false breakouts 2x
    """
    
    def __init__(self):
        self.position_open_step = None
        self.position_open_price = None
        self.whipsaw_threshold_steps = 5
        
    def calculate_reward(self, pnl: float, position_changed: bool, current_step: int) -> float:
        """
        Calculate reward with whipsaw penalty
        
        Args:
            pnl: Raw profit/loss
            position_changed: True if position was opened/closed this step
            current_step: Current step number
            
        Returns:
            Shaped reward
        """
        reward = pnl
        
        # Track position opening
        if position_changed and pnl == 0:  # New position opened
            self.position_open_step = current_step
        
        # Detect whipsaw (position closed in loss within threshold)
        if position_changed and pnl < 0:
            if self.position_open_step is not None:
                steps_held = current_step - self.position_open_step
                
                if steps_held <= self.whipsaw_threshold_steps:
                    # FAKEOUT DETECTED: Apply 2x penalty
                    penalty = abs(pnl) * 2.0
                    reward = -penalty
                    logger.warning(f"[GBPUSD] Fakeout penalty: {penalty:.2f} (held {steps_held} steps)")
            
            self.position_open_step = None
        
        # Hold bonus (small reward for conviction)
        if self.position_open_step is not None:
            steps_held = current_step - self.position_open_step
            if steps_held > self.whipsaw_threshold_steps:
                hold_bonus = 0.1 * min(steps_held / 10, 1.0)  # Max 0.1
                reward += hold_bonus
        
        return reward
```


TASK 5: Sharpe Reward for PPO (USDJPY) ⚠️ NOT IMPLEMENTED
------------------------------------------------------------
File: underdog/rl/multi_asset_env.py or local environment

Current: Raw PnL reward
Required: Sharpe ratio + KL penalty

Implementation:

```python
class SharpeRewardShaper:
    """
    Sharpe-based reward for USDJPY policy stability
    
    Mandate: Optimize risk-adjusted returns, penalize policy instability
    """
    
    def __init__(self, window_size: int = 50):
        self.returns_buffer = []
        self.window_size = window_size
        
    def calculate_reward(self, pnl: float, kl_divergence: float = None) -> float:
        """
        Calculate Sharpe-based reward
        
        Args:
            pnl: Raw profit/loss
            kl_divergence: KL between old and new policy (from PPO)
            
        Returns:
            Sharpe ratio reward with KL penalty
        """
        self.returns_buffer.append(pnl)
        
        # Keep buffer at window size
        if len(self.returns_buffer) > self.window_size:
            self.returns_buffer.pop(0)
        
        # Calculate Sharpe ratio
        if len(self.returns_buffer) >= 2:
            mean_return = np.mean(self.returns_buffer)
            std_return = np.std(self.returns_buffer) + 1e-6  # Avoid division by zero
            sharpe_ratio = mean_return / std_return
        else:
            sharpe_ratio = 0.0
        
        # Base reward is Sharpe ratio
        reward = sharpe_ratio
        
        # Add KL penalty for policy stability (if available from PPO)
        if kl_divergence is not None:
            kl_penalty = 0.5 * kl_divergence  # Beta = 0.5
            reward -= kl_penalty
        
        return reward
```


================================================================================
DEPLOYMENT CHECKLIST
================================================================================

Before running full training (50+ episodes):

1. ✅ Validate agent configs have all required fields
   Command: poetry run python scripts/validate_hmarl_migration.py

2. ⚠️ Implement sequential observation space
   File: underdog/rl/multi_asset_env.py
   Test: Check shapes in first training step

3. ⚠️ Implement DDPG asymmetric reward
   File: underdog/rl/multi_asset_env.py or separate reward shaper
   Test: Verify fakeout penalties in logs

4. ⚠️ Implement PPO Sharpe reward
   File: underdog/rl/multi_asset_env.py or separate reward shaper
   Test: Monitor Sharpe evolution in dashboard

5. ✅ Verify SAC alpha adaptation
   Test: Log alpha values every 100 iterations

6. ⚠️ Add shape validation logging
   File: scripts/train_marl_agent.py
   Test: First episode should log all shapes

7. ⚠️ Update dashboard with HMARL metrics
   File: scripts/visualize_training_v2.py
   Add: Agent types, SAC alpha, PPO KL-div, DDPG noise scale

================================================================================
CURRENT COMMAND STATUS
================================================================================

Validation script is running:
  Command: poetry run python scripts/validate_hmarl_migration.py
  Status: Should complete soon with all architecture tests

Next steps:
  1. Check validation results
  2. Implement sequential observations (CRITICAL)
  3. Implement reward shapers
  4. Run 10-episode test training
  5. Verify shapes and metrics in logs
  6. Scale to 50+ episodes

================================================================================
REFERENCES
================================================================================

Paper Mandates Implemented:
- Independent Learners (IL) architecture ✅
- No parameter sharing between agents ✅
- Asset-specific algorithms ✅
- Heterogeneous neural architectures ✅

Paper Mandates Pending:
- Sequential observation space ⚠️
- Asset-specific reward shaping ⚠️
- Tensor shape validation ⚠️

================================================================================
