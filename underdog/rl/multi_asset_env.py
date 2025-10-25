"""
Multi-Asset Forex Trading Environment for MARL (Multi-Agent RL)

This environment orchestrates 4 TD3 local agents (one per currency pair)
and coordinates them through an A3C Meta-Agent for centralized training
and decentralized execution (CTDE).

Architecture:
    - LEVEL 1 (Local): 4× TD3 Agents (EURUSD, GBPUSD, USDJPY, USDCHF)
    - LEVEL 2 (Global): 1× A3C Meta-Agent (Coordinator)
    
Integration:
    - MetaTrader 5 (MT5) via ZMQ for async real-time data and execution
    
References:
    - 2405.19982v1.pdf: A3C for multi-currency Forex trading
    - ALA2017_Gupta.pdf: CTDE (Centralized Training, Decentralized Execution)
"""

import gymnasium as gym
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import logging
from dataclasses import dataclass, field

from underdog.rl.environments import ForexTradingEnv, TradingEnvConfig
from underdog.rl.observation_buffer import MultiAssetObservationManager

logger = logging.getLogger(__name__)


@dataclass
class MultiAssetConfig:
    """Configuration for Multi-Asset Environment"""
    
    # Symbols to trade
    symbols: List[str] = field(default_factory=lambda: ["EURUSD", "GBPUSD", "USDJPY", "USDCHF"])
    
    # Capital allocation
    initial_balance: float = 100000.0  # Total portfolio balance
    equal_allocation: bool = True      # Split capital equally across agents
    
    # Risk management (global)
    max_global_dd_pct: float = 0.10    # 10% max portfolio drawdown
    max_daily_dd_pct: float = 0.05     # 5% max daily drawdown
    
    # Coordination
    meta_action_mode: str = "risk_limit"  # Meta-Action controls risk limits
    meta_action_clip: Tuple[float, float] = (0.1, 1.0)  # Min/max risk allocation
    
    # Data source
    data_source: str = "historical"    # "historical" or "mt5"
    timeframe: str = "M1"              # M1, M5, M15, H1, H4, D1
    
    # MT5 integration (for live trading)
    mt5_enabled: bool = False
    mt5_symbols_mapping: Dict[str, str] = field(default_factory=lambda: {
        "EURUSD": "EURUSD",
        "GBPUSD": "GBPUSD", 
        "USDJPY": "USDJPY",
        "USDCHF": "USDCHF"
    })


class MultiAssetEnv(gym.Env):
    """
    Multi-Asset Forex Trading Environment for MARL
    
    Coordinates 4 ForexTradingEnv instances (one per currency pair)
    and provides Meta-State/Meta-Action interface for A3C coordinator.
    
    State Space (Meta-State):
        - Meta-State: 15D vector with global portfolio metrics
            [0]: Global DD ratio (%)
            [1]: Total balance (normalized)
            [2]: Turbulence Index Global (average of 4 local turbulences)
            [3-6]: Local DD ratios (4 agents)
            [7-10]: Local position sizes (4 agents)
            [11-14]: Local balance ratios (4 agents)
    
    Action Space (Meta-Action):
        - Meta-Action: 4D vector ∈ [0, 1]⁴
            Each dimension controls max risk/exposure for one agent
            [0]: Risk limit for EURUSD agent
            [1]: Risk limit for GBPUSD agent
            [2]: Risk limit for USDJPY agent
            [3]: Risk limit for USDCHF agent
    
    Execution Flow:
        1. Meta-Agent selects Meta-Action (risk limits)
        2. Apply risk limits to each local agent
        3. Each local TD3 agent selects its own action (position)
        4. Execute all actions in parallel (async if MT5)
        5. Aggregate rewards (cooperative MARL)
        6. Return Meta-State for next step
    """
    
    metadata = {"render_modes": []}
    
    def __init__(
        self,
        config: Optional[MultiAssetConfig] = None,
        base_config_path: Optional[str] = None
    ):
        """
        Initialize Multi-Asset Environment
        
        Args:
            config: MultiAsset configuration
            base_config_path: Path to base YAML config for ForexTradingEnv
        """
        super().__init__()
        
        self.config = config or MultiAssetConfig()
        self.num_agents = len(self.config.symbols)
        
        # Portfolio tracking (MUST be before _create_local_environments)
        self.initial_balance = self.config.initial_balance
        self.peak_balance = self.initial_balance
        self.current_balance = self.initial_balance
        
        # Load base configuration for local environments
        # NOTE: ConfigLoader not available in test mode, using defaults
        self.base_config = None  # Will use ForexTradingEnv defaults
        
        # Create local environments (4× ForexTradingEnv)
        self.local_envs: List[ForexTradingEnv] = []
        self._create_local_environments()
        
        # Meta-State space (23D: expanded with correlations, macro sentiment, opportunity)
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(23,),
            dtype=np.float32
        )
        
        # Meta-Action space (4D - risk limits for each agent)
        self.action_space = gym.spaces.Box(
            low=self.config.meta_action_clip[0],
            high=self.config.meta_action_clip[1],
            shape=(self.num_agents,),
            dtype=np.float32
        )
        
        # Episode tracking
        self.episode_count = 0
        self.step_count = 0
        
        # 🔥 CRITICAL: Temporal Observation Buffers for HMARL
        # Each agent needs different sequence lengths for temporal architectures:
        # - EURUSD (TD3+LSTM): 60 timesteps
        # - USDJPY (PPO+CNN1D): 15 timesteps
        # - XAUUSD (SAC+Transformer): 120 timesteps
        # - GBPUSD (DDPG+Attention): 1 timestep (no buffer)
        # Determine observation dimension from first local env
        if len(self.local_envs) > 0:
            obs_dim = self.local_envs[0].observation_space.shape[0]
        else:
            obs_dim = 24  # Default from ForexTradingEnv
        
        self.obs_manager = MultiAssetObservationManager(
            symbols=self.config.symbols,
            feature_dim=obs_dim
        )
        
        logger.info(
            f"MultiAssetEnv initialized: {self.num_agents} agents, "
            f"balance=${self.initial_balance:,.0f}, mode={self.config.data_source}"
        )
        logger.info(f"Observation buffers initialized: {self.obs_manager.get_sequence_shapes()}")
    
    def _create_local_environments(self):
        """Create 4 local ForexTradingEnv instances (one per symbol)"""
        
        # Calculate capital allocation
        if self.config.equal_allocation:
            balance_per_agent = self.initial_balance / self.num_agents
        else:
            # TODO: Support custom allocation weights
            balance_per_agent = self.initial_balance / self.num_agents
        
        # Data path
        data_dir = Path(__file__).parent.parent.parent / "data" / "histdata"
        
        for symbol in self.config.symbols:
            # Load historical data for this symbol
            csv_file = data_dir / f"{symbol}_2024_full.csv"
            if not csv_file.exists():
                raise FileNotFoundError(f"Historical data not found: {csv_file}")
            
            # Load CSV (assuming format: timestamp, open, high, low, close, volume)
            df = pd.read_csv(csv_file)
            # Ensure datetime column
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
            elif 'time' in df.columns:
                df['timestamp'] = pd.to_datetime(df['time'])
            
            # Build per-symbol TradingEnvConfig and pass to ForexTradingEnv
            env_config = TradingEnvConfig(
                symbol=symbol,
                initial_balance=balance_per_agent,
                timeframe=self.config.timeframe
            )

            env = ForexTradingEnv(
                config=env_config,
                db_connector=None,
                historical_data=df,  # Pass loaded data
                render_mode=None
            )
            
            self.local_envs.append(env)
            logger.debug(f"Created local env for {symbol} with balance=${balance_per_agent:,.0f} ({len(df)} bars)")
    
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reset all local environments and return initial Meta-State
        
        🔥 CRITICAL: Initializes observation buffers with first observations.
        After reset, get_local_observations() returns temporal sequences.
        
        Returns:
            meta_state: Meta-State vector (15D)
            info: Episode info dict
        """
        super().reset(seed=seed)
        
        # Reset all local environments and collect initial observations
        local_states = []
        for env in self.local_envs:
            state, info = env.reset(seed=seed)
            local_states.append(state)
        
        # 🔥 Initialize observation buffers with initial observations
        self.obs_manager.reset(local_states)
        
        # Reset portfolio tracking
        self.current_balance = self.initial_balance
        self.peak_balance = self.initial_balance
        self.step_count = 0
        self.episode_count += 1
        
        # Build Meta-State from local states
        meta_state = self._build_meta_state(local_states)
        
        info = {
            'episode': self.episode_count,
            'symbols': self.config.symbols,
            'balance': self.current_balance,
        }
        
        logger.info(
            f"Episode {self.episode_count} reset: {self.num_agents} agents, "
            f"balance=${self.current_balance:,.0f}"
        )
        
        return meta_state, info
    
    def step(
        self,
        meta_action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Execute one coordinated step across all agents
        
        Workflow:
            1. Apply Meta-Action (risk limits) to each local agent
            2. Each local TD3 agent selects its own action
            3. Execute all actions in parallel
            4. Aggregate rewards (cooperative MARL)
            5. Check termination conditions
            6. Return Meta-State for next step
        
        Args:
            meta_action: 4D vector of risk limits [0, 1]⁴
        
        Returns:
            meta_state: Next Meta-State (15D)
            total_reward: Aggregated reward from all agents
            terminated: Episode terminated (DD breach or manual stop)
            truncated: Episode truncated (max steps)
            info: Step info dict
        """
        self.step_count += 1
        
        # 1. Apply Meta-Action to each local agent (set risk limits)
        self._apply_meta_action(meta_action)
        
        # 2. Let each local agent take its own action (TD3 policy)
        # NOTE: In actual training, TD3 agents will select actions
        # For now, we'll let the environment handle action selection
        local_states = []
        local_rewards = []
        local_dones = []
        local_infos = []
        
        for i, env in enumerate(self.local_envs):
            # Get action from TD3 agent (this will be called externally)
            # For environment step, we assume action is provided by agent
            # Here we just execute the environment step with a dummy action
            # In actual training loop, this will be replaced with agent.select_action()
            
            # PLACEHOLDER: Random action (will be replaced by TD3 agent)
            local_action = np.random.uniform(-1.0, 1.0, size=(1,))
            
            # Execute step in local environment
            state, reward, done, truncated, info = env.step(local_action)
            
            local_states.append(state)
            local_rewards.append(reward)
            local_dones.append(done or truncated)
            local_infos.append(info)
        
        # 🔥 CRITICAL: Update observation buffers with new observations
        self.obs_manager.add(local_states)
        
        # 3. Aggregate rewards (cooperative MARL)
        # Paper: "new+Multi-Agent+RL..." suggests sum of individual rewards
        total_reward = sum(local_rewards)
        
        # 4. Update portfolio balance (using equity = balance + unrealized P&L)
        # CRITICAL: Use equity instead of balance to reflect open position P&L
        # This ensures drawdown calculations and dashboard metrics are accurate
        old_balance = self.current_balance
        self.current_balance = sum(env.equity for env in self.local_envs)
        self.peak_balance = max(self.peak_balance, self.current_balance)
        
        # DEBUG: Log significant balance changes
        if abs(self.current_balance - old_balance) > 100:
            logger.debug(
                f"Portfolio balance update: ${old_balance:.2f} -> ${self.current_balance:.2f} "
                f"(equities: {[f'${e:.2f}' for e in [env.equity for env in self.local_envs]]})"
            )
        
        # 5. Build Meta-State
        meta_state = self._build_meta_state(local_states)
        
        # 6. Check global termination conditions
        global_dd = self._calculate_global_dd()
        terminated = any(local_dones) or global_dd > self.config.max_global_dd_pct
        truncated = False  # TODO: Add max_steps truncation
        
        # Calculate local DD ratios (not stored as attribute in ForexTradingEnv)
        local_dd_ratios = []
        for env in self.local_envs:
            dd_usd = env.peak_balance - env.equity
            dd_ratio = dd_usd / env.peak_balance if env.peak_balance > 0 else 0.0
            local_dd_ratios.append(dd_ratio)
        
        # 7. Build info dict
        info = {
            'step': self.step_count,
            'local_rewards': local_rewards,
            'local_dds': local_dd_ratios,
            'global_dd': global_dd,
            'balance': self.current_balance,
            'peak_balance': self.peak_balance,
            'meta_action': meta_action.tolist(),
            'local_infos': local_infos,
        }
        
        if terminated:
            logger.warning(
                f"Episode {self.episode_count} terminated at step {self.step_count}: "
                f"global_dd={global_dd:.2%}, balance=${self.current_balance:,.0f}"
            )
        
        return meta_state, total_reward, terminated, truncated, info
    
    def _apply_meta_action(self, meta_action: np.ndarray):
        """
        Apply Meta-Action (risk limits) to each local agent
        
        Meta-Action modulates the maximum position size each agent can take.
        This is the core coordination mechanism of CTDE.
        
        Args:
            meta_action: 4D vector of risk limits [0, 1]⁴
                        1.0 = full risk allowed
                        0.5 = 50% of normal risk
                        0.1 = 10% of normal risk (minimum)
        """
        for i, env in enumerate(self.local_envs):
            risk_limit = float(meta_action[i])
            
            # Modulate max position size based on risk limit
            # Original max_position_size is stored in config (TradingEnvConfig is a dataclass)
            if not hasattr(env, '_original_max_position_size'):
                env._original_max_position_size = env.config.max_position_size
            
            # Apply risk limit
            env.config.max_position_size = env._original_max_position_size * risk_limit
            
            logger.debug(
                f"{self.config.symbols[i]}: risk_limit={risk_limit:.2f}, "
                f"max_position={env.config.max_position_size:.2f}"
            )
    
    def _build_meta_state(self, local_states: List[np.ndarray]) -> np.ndarray:
        """
        📊 Build Meta-State from local states with MARL awareness
        
        Meta-State (EXPANDED TO 23D):
            [0]: Global DD ratio (%)
            [1]: Total balance (normalized by initial)
            [2]: Turbulence Index Global (average of 4 local turbulences)
            [3-6]: Local DD ratios (4 agents)
            [7-10]: Local position sizes (4 agents) - PEER INFLUENCE
            [11-14]: Local balance ratios (4 agents, normalized)
            [15-18]: Cross-correlation vector (4 pairs with EURUSD as anchor)
            [19]: Macro sentiment score (volatility proxy + time risk)
            [20-22]: Opportunity scores (avg, min, max across agents)
        
        Args:
            local_states: List of 4 state vectors (29D each, one per TD3 agent)
        
        Returns:
            meta_state: 23D numpy array
        """
        # Extract features from local states (now 29D)
        # State indices: turbulence[21], market_power[24], behavioral_bias[25], 
        #                liquidity_phase[26], opportunity[27], self_confidence[28]
        
        turbulences = []
        market_powers = []
        opportunity_scores = []
        
        for state in local_states:
            if len(state) >= 29:
                turbulences.append(float(state[21]))  # Turbulence index
                market_powers.append(float(state[24]))  # Market power
                opportunity_scores.append(float(state[27]))  # Opportunity score
            else:
                # Fallback for old state format
                turbulences.append(0.0)
                market_powers.append(0.0)
                opportunity_scores.append(0.5)
        
        # Calculate global metrics
        global_dd = self._calculate_global_dd()
        total_balance_norm = self.current_balance / self.initial_balance
        turbulence_global = np.mean(turbulences) if turbulences else 0.0
        
        # Extract local metrics from environments
        local_dds = []
        local_positions = []  # 💰 PEER INFLUENCE
        local_balances = []
        
        for env in self.local_envs:
            # Calculate DD ratio (not stored as attribute)
            dd_usd = env.peak_balance - env.equity
            dd_ratio = dd_usd / env.peak_balance if env.peak_balance > 0 else 0.0
            local_dds.append(np.clip(dd_ratio, 0.0, 1.0))
            
            # Position size and balance
            local_positions.append(env.position_size)  # MARL awareness
            local_balances.append(
                env.balance / (self.initial_balance / self.num_agents)
            )
        
        # 💰 CROSS-CORRELATION VECTOR: Correlate returns of 4 pairs
        cross_corr = self._calculate_cross_correlation()
        
        # 💰 MACRO SENTIMENT: Proxy using global turbulence + time risk
        macro_sentiment = self._calculate_macro_sentiment(turbulence_global)
        
        # 💰 OPPORTUNITY AGGREGATION: Meta-awareness of agent clarity
        avg_opportunity = np.mean(opportunity_scores)
        min_opportunity = np.min(opportunity_scores)
        max_opportunity = np.max(opportunity_scores)
        
        # Construct Meta-State vector (23D)
        meta_state = np.array([
            global_dd,           # [0]
            total_balance_norm,  # [1]
            turbulence_global,   # [2]
            *local_dds,          # [3-6]
            *local_positions,    # [7-10] 💰 PEER INFLUENCE
            *local_balances,     # [11-14]
            *cross_corr,         # [15-18] 💰 CROSS-CORRELATION (4 values)
            macro_sentiment,     # [19] 💰 MACRO SENTIMENT
            avg_opportunity,     # [20] 💰 OPPORTUNITY (avg)
            min_opportunity,     # [21] 💰 OPPORTUNITY (min)
            max_opportunity,     # [22] 💰 OPPORTUNITY (max)
        ], dtype=np.float32)
        
        return meta_state
    
    def _calculate_cross_correlation(self) -> List[float]:
        """
        💰 Calculate Cross-Correlation Vector between currency pairs
        
        Detects decoupling of capital flows (anomalous correlations)
        
        Returns correlation of each pair with EURUSD (anchor):
        - corr(EURUSD, GBPUSD)
        - corr(EURUSD, USDJPY)
        - corr(EURUSD, USDCHF)
        - corr(GBPUSD, USDJPY) (bonus: EUR-GBP divergence)
        
        Returns:
            List of 4 correlation values [-1, 1]
        """
        correlations = []
        
        # Need at least 20 data points for meaningful correlation
        min_window = 20
        
        if not hasattr(self, 'price_history'):
            self.price_history = {symbol: [] for symbol in self.config.symbols}
        
        # Collect current prices
        for i, env in enumerate(self.local_envs):
            symbol = self.config.symbols[i]
            if env.df_ohlcv is not None and len(env.df_ohlcv) > 0:
                current_price = env.df_ohlcv['close'].values[-1]
                self.price_history[symbol].append(current_price)
                
                # Keep only last N prices
                if len(self.price_history[symbol]) > 100:
                    self.price_history[symbol] = self.price_history[symbol][-100:]
        
        # Calculate correlations if enough data
        if all(len(prices) >= min_window for prices in self.price_history.values()):
            symbols = self.config.symbols
            
            # Get log-returns for correlation
            returns = {}
            for symbol in symbols:
                prices = np.array(self.price_history[symbol][-min_window:])
                returns[symbol] = np.diff(np.log(prices))
            
            # CRITICAL: Check for zero variance before correlation (prevents divide-by-zero)
            # If stddev is zero, correlation is undefined (returns NaN)
            def safe_corrcoef(arr1, arr2):
                """Safe correlation that handles zero-variance arrays"""
                std1 = np.std(arr1)
                std2 = np.std(arr2)
                
                # If either series has zero variance, return 0.0 (uncorrelated)
                if std1 < 1e-8 or std2 < 1e-8:
                    return 0.0
                
                # Otherwise, compute correlation normally
                corr_matrix = np.corrcoef(arr1, arr2)
                corr = corr_matrix[0, 1]
                return 0.0 if np.isnan(corr) else float(corr)
            
            # Correlation 1: EURUSD vs GBPUSD
            corr_1 = safe_corrcoef(returns[symbols[0]], returns[symbols[1]])
            
            # Correlation 2: EURUSD vs USDJPY
            corr_2 = safe_corrcoef(returns[symbols[0]], returns[symbols[2]])
            
            # Correlation 3: EURUSD vs USDCHF
            corr_3 = safe_corrcoef(returns[symbols[0]], returns[symbols[3]])
            
            # Correlation 4: GBPUSD vs USDJPY (EUR-GBP divergence proxy)
            corr_4 = safe_corrcoef(returns[symbols[1]], returns[symbols[2]])
            
            correlations = [corr_1, corr_2, corr_3, corr_4]
        else:
            # Not enough data: return neutral correlations
            correlations = [0.0, 0.0, 0.0, 0.0]
        
        # Handle NaN (can occur with constant prices) - DEPRECATED: safe_corrcoef handles this
        correlations = [0.0 if np.isnan(c) else float(c) for c in correlations]
        
        return correlations
    
    def _calculate_macro_sentiment(self, turbulence_global: float) -> float:
        """
        💰 Calculate Macro Sentiment Score (Volatility Proxy + Time Risk)
        
        Represents exogenous risk factors:
        - High global turbulence = high macro risk
        - Friday close / Weekend = increased risk
        - News events (simplified: detected via volatility spikes)
        
        Args:
            turbulence_global: Average turbulence across 4 pairs [0, 1]
        
        Returns:
            Macro sentiment score [0, 1] where:
            - 0.0 = Low risk (calm markets, mid-week)
            - 1.0 = High risk (volatile markets, Friday close)
        """
        sentiment = 0.0
        
        # Component 1: Turbulence-based risk (60% weight)
        turbulence_risk = turbulence_global * 0.6
        
        # Component 2: Time-based risk (40% weight)
        time_risk = 0.0
        
        # Check if we can detect day/time from environment data
        if self.local_envs and self.local_envs[0].df_ohlcv is not None:
            df = self.local_envs[0].df_ohlcv
            if 'time' in df.columns and len(df) > 0:
                current_time = df['time'].iloc[-1]
                
                # Friday risk (day 4 in weekday())
                if hasattr(current_time, 'weekday'):
                    if current_time.weekday() == 4:  # Friday
                        time_risk += 0.3
                    
                    # Friday afternoon (hour >= 16 UTC)
                    if current_time.weekday() == 4 and current_time.hour >= 16:
                        time_risk += 0.2  # Extra risk before weekend
                
                # End of day risk (hour >= 20 or <= 2 UTC)
                if hasattr(current_time, 'hour'):
                    if current_time.hour >= 20 or current_time.hour <= 2:
                        time_risk += 0.1
        
        time_risk = min(time_risk, 1.0) * 0.4
        
        # Combined sentiment
        sentiment = turbulence_risk + time_risk
        
        return float(np.clip(sentiment, 0.0, 1.0))
    
    def _calculate_global_dd(self) -> float:
        """
        Calculate global portfolio drawdown
        
        Returns:
            global_dd: Portfolio drawdown ratio [0, 1]
        """
        if self.peak_balance <= 0:
            return 0.0
        
        global_dd = (self.peak_balance - self.current_balance) / self.peak_balance
        return max(0.0, global_dd)
    
    def render(self, mode: str = "human"):
        """Render environment (optional for debugging)"""
        if mode == "human":
            print(f"\n{'='*60}")
            print(f"Episode {self.episode_count} - Step {self.step_count}")
            print(f"{'='*60}")
            print(f"Portfolio Balance: ${self.current_balance:,.2f}")
            print(f"Peak Balance:      ${self.peak_balance:,.2f}")
            print(f"Global DD:         {self._calculate_global_dd():.2%}")
            print(f"\nLocal Agents:")
            for i, env in enumerate(self.local_envs):
                symbol = self.config.symbols[i]
                # Calculate DD ratio manually
                dd_usd = env.peak_balance - env.equity
                dd_ratio = dd_usd / env.peak_balance if env.peak_balance > 0 else 0.0
                print(f"  {symbol:8s}: ${env.balance:10,.2f}  DD={dd_ratio:6.2%}  Pos={env.position_size:+.2f}")
            print(f"{'='*60}\n")
    
    
    def get_local_observations(self) -> List[np.ndarray]:
        """
        🔥 CRITICAL: Get temporal observation sequences for all agents.
        
        This method MUST be used instead of accessing local_states directly
        during training. Returns sequences with correct shapes for each architecture:
        
        Returns:
            List of observation sequences:
            - EURUSD: [60, Features] for LSTM
            - USDJPY: [15, Features] for CNN1D
            - XAUUSD: [120, Features] for Transformer
            - GBPUSD: [Features] for Attention (no temporal dimension)
        
        Example:
            >>> meta_state, info = env.reset()
            >>> observations = env.get_local_observations()
            >>> # observations[0] has shape [60, 24] for EURUSD LSTM
            >>> # observations[1] has shape [15, 24] for USDJPY CNN1D
            >>> # observations[2] has shape [120, 24] for XAUUSD Transformer
            >>> # observations[3] has shape [24] for GBPUSD Attention
        
        Raises:
            RuntimeError: If called before reset()
        """
        return self.obs_manager.get_sequences()
    
    def get_observation_shapes(self) -> Dict[str, tuple]:
        """
        Get expected observation shapes for each agent (debugging).
        
        Returns:
            Dict mapping symbol to expected shape tuple.
        """
        return self.obs_manager.get_sequence_shapes()
    
    def close(self):
        """Close all local environments"""
        for env in self.local_envs:
            env.close()
        logger.info(f"MultiAssetEnv closed: {self.num_agents} agents")


def test_multi_asset_env():
    """Test function for MultiAssetEnv"""
    
    # Create environment
    config = MultiAssetConfig(
        symbols=["EURUSD", "GBPUSD", "USDJPY", "USDCHF"],
        initial_balance=100000.0,
        data_source="historical"
    )
    
    env = MultiAssetEnv(config=config)
    
    # Reset
    meta_state, info = env.reset()
    print(f"Meta-State shape: {meta_state.shape}")
    print(f"Meta-State: {meta_state}")
    
    # Take 5 steps with random Meta-Actions
    for step in range(5):
        # Random Meta-Action (risk limits)
        meta_action = np.random.uniform(0.5, 1.0, size=(4,))
        
        meta_state, reward, terminated, truncated, info = env.step(meta_action)
        
        print(f"\nStep {step + 1}:")
        print(f"  Meta-Action: {meta_action}")
        print(f"  Reward: {reward:.2f}")
        print(f"  Global DD: {info['global_dd']:.2%}")
        print(f"  Balance: ${info['balance']:,.2f}")
        
        if terminated or truncated:
            print("Episode terminated!")
            break
    
    env.close()


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
    )
    
    # Run test
    print("Testing MultiAssetEnv...")
    test_multi_asset_env()
