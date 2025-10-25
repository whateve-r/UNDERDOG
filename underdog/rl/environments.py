"""
Trading Environment - Gymnasium Compatible

Forex trading environment for DRL training with:
- 🧠 24-dimensional EXPANDED state space (StateVectorBuilder)
  * [0-16] Standard features (price, technicals, sentiment, regime, macro)
  * [17-20] 🛡️ CRITICAL CMDP features (position, cash, daily DD, total DD)
  * [21-23] Enhanced features (turbulence, stochastic, momentum)
- 2-dimensional continuous action space (TD3)
- 🛡️ CMDP Reward with Catastrophic DD Penalties
  * PENALTY_DAILY = -1000 (5% breach)
  * PENALTY_TOTAL = -10000 (10% breach)
  * Episode termination on violations
- Prop Firm safety constraints (via PropFirmSafetyShield)

Papers:
- arXiv:2510.04952v2: Safe Trade Execution (Shield integration)
- arXiv:2510.10526v1: LLM + RL Integration
- arXiv:3745133.3745185: Technical Analysis & Turbulence Index

Action Space:
- Action[0]: Position size [-1, 1] (short to long)
- Action[1]: Entry/Exit decision [-1, 1] (close to open)

Reward Function:
- Primary: Sharpe Ratio (rolling 30 periods)
- 🛡️ CRITICAL: Catastrophic DD penalties (-1000 daily, -10000 total)
- Soft penalty: Quadratic DD penalty (before limits)
- Bonus: Prop Firm compliance (no violations)

Integration:
- StateVectorBuilder for observations
- PropFirmSafetyShield for action validation
- TimescaleDB for historical data
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional, Any
from datetime import datetime, timedelta
from dataclasses import dataclass
import logging

from underdog.ml.feature_engineering import StateVectorBuilder, StateVectorConfig
from underdog.risk_management.compliance.compliance_shield import PropFirmSafetyShield, SafetyConstraints
from underdog.database.timescale.timescale_connector import TimescaleDBConnector

logger = logging.getLogger(__name__)


@dataclass
class TradingEnvConfig:
    """Trading environment configuration"""
    # Environment
    initial_balance: float = 100_000.0
    max_position_size: float = 1.0  # Max 1 lot
    commission_pct: float = 0.0003  # 0.03% per trade (3 pips on EURUSD)
    spread_pips: float = 1.0  # 1 pip spread
    
    # Episode
    max_steps: int = 1000  # Max bars per episode
    lookback_bars: int = 200  # Historical bars for state
    
    # Reward
    sharpe_window: int = 30  # Rolling Sharpe calculation
    dd_penalty_factor: float = 2.0  # Exponential DD penalty
    compliance_bonus: float = 0.1  # Bonus for no violations
    
    # 🛡️ CRITICAL: CMDP Safety Constraints (Prop Firm Rules)
    max_daily_dd_pct: float = 0.05  # 5% daily DD limit
    max_total_dd_pct: float = 0.10  # 10% total DD limit
    penalty_daily_dd: float = -1000.0  # Catastrophic penalty for daily DD breach
    penalty_total_dd: float = -10000.0  # Absolute catastrophic penalty for total DD breach
    terminate_on_dd_breach: bool = True  # Terminate episode on DD violation
    
    # Safety
    use_safety_shield: bool = True
    
    # Data
    symbol: str = 'EURUSD'
    timeframe: str = '1H'


class ForexTradingEnv(gym.Env):
    """
    Forex trading environment for DRL
    
    Observation Space: Box(14,) - Normalized features
    Action Space: Box(2,) - Continuous actions [-1, 1]
    
    Usage:
        env = ForexTradingEnv(config, db_connector)
        obs, info = env.reset()
        action = agent.select_action(obs)
        obs, reward, terminated, truncated, info = env.step(action)
    """
    
    metadata = {'render_modes': ['human', 'rgb_array']}
    
    def __init__(
        self,
        config: Optional[TradingEnvConfig] = None,
        db_connector: Optional[TimescaleDBConnector] = None,
        historical_data: Optional[pd.DataFrame] = None,
        render_mode: Optional[str] = None
    ):
        """
        Initialize trading environment
        
        Args:
            config: Environment configuration
            db_connector: TimescaleDB connector (for live data)
            historical_data: Pre-loaded OHLCV data (for backtesting)
            render_mode: Render mode ('human' or 'rgb_array')
        """
        super().__init__()
        
        self.config = config or TradingEnvConfig()
        self.render_mode = render_mode
        
        # Symbol (store as instance variable for ComplianceShield)
        self.symbol = self.config.symbol
        
        # Data source
        self.db = db_connector
        self.historical_data = historical_data
        
        # State builder (only for live mode, not for backtesting with historical_data)
        state_config = StateVectorConfig(use_redis=False)  # Disable Redis for training
        # Don't create StateVectorBuilder if we have historical_data (backtesting mode)
        self.state_builder = None
        if db_connector and historical_data is None:
            self.state_builder = StateVectorBuilder(db_connector, state_config)
        
        # Safety shield
        self.shield = None
        if self.config.use_safety_shield:
            constraints = SafetyConstraints(
                max_daily_dd_pct=0.05,
                max_total_dd_pct=0.10,
                max_positions=2,
                max_risk_per_trade_pct=0.015  # Corrected parameter name
            )
            self.shield = PropFirmSafetyShield(constraints)
        
        # Define spaces
        # 🧠 EXPANDED TO 31D: CMDP + Technical + Market Awareness + 🔥 STALENESS (Latency Risk)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(31,),
            dtype=np.float32
        )
        
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(1,),  # Single continuous action: position size
            dtype=np.float32
        )
        
        # Episode state
        self.current_step = 0
        self.balance = self.config.initial_balance
        self.equity = self.config.initial_balance
        self.initial_balance_snapshot = self.config.initial_balance
        self.position_size = 0.0  # Current position (-1 to +1)
        self.position_entry_price = 0.0
        self.position_entry_step = 0
        self.position_entry_staleness = 0.0  # 🔥 CRITICAL: Staleness at entry for reward shaping
        
        # 🛡️ CRITICAL: Drawdown tracking for CMDP
        self.peak_balance = self.config.initial_balance  # Peak balance (for total DD)
        self.daily_start_balance = self.config.initial_balance  # Daily starting balance
        self.daily_peak_balance = self.config.initial_balance  # Daily peak (for intraday DD)
        self.current_day = 0  # Track episode day (for daily reset)
        self._dd_termination_flag = False  # CMDP termination flag
        
        # Performance tracking
        self.returns_history = []
        self.equity_history = []
        self.actions_history = []
        self.trades = []
        
        # 💰 Market awareness tracking
        self._last_behavioral_bias = 0.0  # For smart contrarian reward
        
        # Current data
        self.df_ohlcv = None
        self.current_price = 0.0
        
        logger.info(f"ForexTradingEnv initialized (symbol: {self.config.symbol}, max_steps: {self.config.max_steps})")
    
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reset environment to initial state
        
        Args:
            seed: Random seed
            options: Additional options
        
        Returns:
            (observation, info)
        """
        super().reset(seed=seed)
        
        # Reset episode state
        self.current_step = 0
        self.balance = self.config.initial_balance
        self.equity = self.config.initial_balance
        self.initial_balance_snapshot = self.config.initial_balance
        self.position_size = 0.0
        self.position_entry_price = 0.0
        self.position_entry_step = 0
        self.position_entry_staleness = 0.0  # 🔥 Reset staleness tracking
        
        # 🛡️ Reset drawdown tracking
        self.peak_balance = self.config.initial_balance
        self.daily_start_balance = self.config.initial_balance
        self.daily_peak_balance = self.config.initial_balance
        self.current_day = 0
        self._dd_termination_flag = False
        
        # Reset tracking
        self.returns_history = []
        self.equity_history = [self.equity]
        self.actions_history = []
        self.trades = []
        
        # Load data
        if self.historical_data is not None:
            # Use pre-loaded data (backtesting mode)
            self.df_ohlcv = self.historical_data.copy()
        else:
            # TODO: Load from TimescaleDB (live mode)
            raise NotImplementedError("Live data loading not yet implemented")
        
        if len(self.df_ohlcv) < self.config.max_steps + self.config.lookback_bars:
            raise ValueError(f"Insufficient data: need {self.config.max_steps + self.config.lookback_bars}, got {len(self.df_ohlcv)}")
        
        # Get initial observation
        obs = self._get_observation()
        info = self._get_info()
        
        logger.info(f"Episode reset: balance={self.balance:.2f}, obs_shape={obs.shape}, obs_len={len(obs)}")
        
        return obs, info
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Execute one time step
        
        Args:
            action: Continuous action in [-1, 1] representing target position size
                   -1.0 = max short, 0.0 = neutral, +1.0 = max long
        
        Returns:
            (observation, reward, terminated, truncated, info)
        """
        # Clip action to valid range
        action = np.clip(action, -1, 1)
        
        # Get current price
        idx = self.config.lookback_bars + self.current_step
        self.current_price = self.df_ohlcv.iloc[idx]['close']
        
        # Apply safety shield
        if self.shield:
            action = self._apply_shield(action)
        
        # Execute action
        self._execute_action(action)
        
        # Update equity
        self._update_equity()
        
        # Calculate reward
        reward = self._calculate_reward()
        
        # Check termination
        terminated = self._check_termination()
        truncated = self.current_step >= self.config.max_steps
        
        # Advance step
        self.current_step += 1
        
        # Get observation
        obs = self._get_observation()
        info = self._get_info()
        
        # Track history
        self.actions_history.append(action)
        self.equity_history.append(self.equity)
        
        return obs, reward, terminated, truncated, info
    
    def _get_observation(self) -> np.ndarray:
        """
        Get current observation (state vector) - EXPANDED TO 24D
        
        State Vector Layout (24 dimensions):
        [0-2]   Price features (normalized_price, returns, volatility)
        [3-8]   Technical indicators (RSI, MACD, ATR, BB, ADX, CCI)
        [9]     Sentiment score
        [10-12] Regime features (trend, range, transition confidence)
        [13-15] Macro features (VIX, Fed Rate, Yield Curve)
        [16]    Volume ratio
        [17]    🛡️ Current position size [-1, 1]
        [18]    🛡️ Normalized cash balance [0, 1]
        [19]    🛡️ Daily drawdown used ratio [0, 1]
        [20]    🛡️ Total drawdown used ratio [0, 1]
        [21]    🛡️ Market turbulence index [0, 1]
        [22-23] Additional technical (Stochastic, Momentum)
        
        Returns:
            State vector (24,)
        """
        if self.state_builder and self.db:
            # Use StateVectorBuilder (live mode)
            # TODO: Implement async call
            raise NotImplementedError("Live state building not yet implemented")
        else:
            # Build state manually from historical data (backtesting mode)
            idx = self.config.lookback_bars + self.current_step
            df_window = self.df_ohlcv.iloc[:idx+1]
            
            # Use StateVectorBuilder's calculation methods
            close = df_window['close'].values
            high = df_window['high'].values
            low = df_window['low'].values
            open_price = df_window['open'].values
            volume = df_window['volume'].values if 'volume' in df_window.columns else np.ones(len(close))
            
            # [0-2] Price features
            price_norm = (close[-1] - close[-100:].min()) / (close[-100:].max() - close[-100:].min() + 1e-8)
            returns = np.log(close[-1] / close[-2]) if len(close) >= 2 else 0.0
            volatility = np.std(np.diff(np.log(close[-20:]))) if len(close) >= 20 else 0.02
            
            # [3-8] Technical indicators (expanded)
            rsi = self._calc_rsi(close, period=14)
            macd_signal = self._calc_macd(close)
            atr_norm = self._calc_atr(high, low, close, period=14)
            bb_width = self._calc_bb_width(close, period=20)
            adx = self._calc_adx(high, low, close, period=14)
            cci = self._calc_cci(high, low, close, period=20)
            
            # [9] Sentiment (placeholder - would fetch from DB)
            sentiment = 0.0
            
            # [10-12] Regime (placeholder - would fetch from DB)
            regime_trend = 0.33
            regime_range = 0.33
            regime_transition = 0.33
            
            # [13-15] Macro (placeholder - would fetch from DB)
            vix_norm = 0.25
            fed_rate_norm = 0.5
            yield_curve_norm = 0.0
            
            # [16] Volume ratio (safely handle division by zero)
            if len(volume) >= 20:
                mean_vol = np.mean(volume[-20:])
                volume_ratio = (volume[-1] / mean_vol) if mean_vol > 0 else 1.0
            else:
                volume_ratio = 1.0
            
            # 🛡️ [17-20] CRITICAL POSITION & RISK FEATURES (CMDP)
            position_feature = self.position_size  # Already [-1, 1]
            
            cash_norm = self.balance / self.initial_balance_snapshot  # [0, ~1+]
            cash_norm = np.clip(cash_norm, 0.0, 2.0) / 2.0  # Normalize to [0, 1]
            
            # Daily DD used ratio
            daily_dd_usd = self.daily_peak_balance - self.equity
            daily_dd_limit = self.initial_balance_snapshot * self.config.max_daily_dd_pct
            daily_dd_ratio = np.clip(daily_dd_usd / daily_dd_limit, 0.0, 1.5) / 1.5  # [0, 1]
            
            # Total DD used ratio
            total_dd_usd = self.peak_balance - self.equity
            total_dd_limit = self.initial_balance_snapshot * self.config.max_total_dd_pct
            total_dd_ratio = np.clip(total_dd_usd / total_dd_limit, 0.0, 1.5) / 1.5  # [0, 1]
            
            # 🛡️ [21] Market Turbulence Index (Log-Returns Volatility)
            turbulence = self._calculate_turbulence_local(close, window=20)
            
            # [22-23] Additional technical indicators
            stochastic = self._calc_stochastic(high, low, close, period=14)
            momentum = (close[-1] - close[-10]) / close[-10] if len(close) >= 10 else 0.0
            momentum = np.clip(momentum, -0.1, 0.1) / 0.1  # Normalize to [-1, 1]
            
            # 💰 [24-29] MARKET AWARENESS FEATURES (Smart Money Detection)
            market_power = self._calculate_market_power(high, low, volume)
            behavioral_bias = self._calculate_behavioral_bias(open_price, high, low, close)
            
            # Store for reward calculation
            self._last_behavioral_bias = behavioral_bias
            
            liquidity_phase = self._calculate_liquidity_phase(market_power, turbulence, volume)
            liquidity_phase_norm = liquidity_phase / 2.0  # Normalize to [0, 1]
            opportunity_score = self._calculate_opportunity_score()
            self_confidence = self._calculate_self_confidence()
            
            # 🔥 [29-30] STALENESS FEATURES (Financial Intelligence - Latency Risk)
            # Feature 1: Reference Price Deviation (Staleness Proxy)
            # Quantifies how "stale" or "surprising" current price is vs recent trajectory
            staleness_proxy = self._calculate_staleness_proxy(close)
            
            # Feature 2: Short-Term Order Imbalance Proxy
            # Detects impulsive pressure (precursor to slippage/whipsaw)
            order_imbalance_proxy = self._calculate_order_imbalance_proxy(close)
            
            state = np.array([
                price_norm,          # [0]
                returns,             # [1]
                volatility,          # [2]
                rsi,                 # [3]
                macd_signal,         # [4]
                atr_norm,            # [5]
                bb_width,            # [6]
                adx,                 # [7]
                cci,                 # [8]
                sentiment,           # [9]
                regime_trend,        # [10]
                regime_range,        # [11]
                regime_transition,   # [12]
                vix_norm,            # [13]
                fed_rate_norm,       # [14]
                yield_curve_norm,    # [15]
                volume_ratio,        # [16]
                position_feature,    # [17] 🛡️ CRITICAL
                cash_norm,           # [18] 🛡️ CRITICAL
                daily_dd_ratio,      # [19] 🛡️ CRITICAL
                total_dd_ratio,      # [20] 🛡️ CRITICAL
                turbulence,          # [21] 🛡️ CRITICAL
                stochastic,          # [22]
                momentum,            # [23]
                market_power,        # [24] 💰 NEW: VVR (Volume/Volatility Ratio)
                behavioral_bias,     # [25] 💰 NEW: Wick/Range Ratio
                liquidity_phase_norm,# [26] 💰 NEW: Market Cycle (0=Accum, 1=Manip, 2=Dist)
                opportunity_score,   # [27] 💰 NEW: Signal Clarity
                self_confidence,     # [28] 💰 NEW: Rolling Sharpe
                staleness_proxy,     # [29] 🔥 CRITICAL: Price quality/latency risk
                order_imbalance_proxy,# [30] 🔥 CRITICAL: Impulsive pressure
            ], dtype=np.float32)
            
            # 🛡️ Safety: Replace any NaN/Inf values
            if np.any(np.isnan(state)) or np.any(np.isinf(state)):
                state = np.nan_to_num(state, nan=0.0, posinf=1.0, neginf=-1.0)
            
            return state
    
    # 🧠 Technical Indicator Helper Methods (Simplified Implementations)
    
    def _calc_rsi(self, close: np.ndarray, period: int = 14) -> float:
        """Calculate RSI (Relative Strength Index) [0, 1]"""
        if len(close) < period + 1:
            return 0.5
        deltas = np.diff(close[-period-1:])
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        avg_gain = np.mean(gains)
        avg_loss = np.mean(losses)
        if avg_loss == 0:
            return 1.0
        rs = avg_gain / avg_loss
        rsi = 1 - (1 / (1 + rs))
        return float(rsi)
    
    def _calc_macd(self, close: np.ndarray) -> float:
        """Calculate MACD signal [-1, 1] (normalized)"""
        if len(close) < 26:
            return 0.0
        ema_12 = self._ema(close, 12)
        ema_26 = self._ema(close, 26)
        macd_line = ema_12 - ema_26
        # Normalize by recent price
        macd_norm = macd_line / close[-1] if close[-1] > 0 else 0.0
        return float(np.clip(macd_norm * 100, -1, 1))  # Scale to [-1, 1]
    
    def _ema(self, data: np.ndarray, period: int) -> float:
        """Calculate EMA"""
        if len(data) < period:
            return data[-1] if len(data) > 0 else 0.0
        weights = np.exp(np.linspace(-1., 0., period))
        weights /= weights.sum()
        ema = np.convolve(data[-period:], weights, mode='valid')[0]
        return float(ema)
    
    def _calc_atr(self, high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> float:
        """Calculate ATR (Average True Range) normalized [0, 1]"""
        if len(close) < period + 1:
            return 0.02
        tr = np.maximum(high[-period:] - low[-period:], 
                       np.abs(high[-period:] - close[-period-1:-1]))
        atr = np.mean(tr)
        atr_norm = atr / close[-1] if close[-1] > 0 else 0.02
        return float(np.clip(atr_norm, 0, 0.1) / 0.1)  # Normalize to [0, 1]
    
    def _calc_bb_width(self, close: np.ndarray, period: int = 20) -> float:
        """Calculate Bollinger Bands width [0, 1]"""
        if len(close) < period:
            return 0.05
        sma = np.mean(close[-period:])
        std = np.std(close[-period:])
        bb_width = (2 * std) / sma if sma > 0 else 0.05
        return float(np.clip(bb_width, 0, 0.2) / 0.2)  # Normalize to [0, 1]
    
    def _calc_adx(self, high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> float:
        """Calculate ADX (Average Directional Index) [0, 1]"""
        if len(close) < period + 1:
            return 0.25
        # Simplified ADX (full calculation is complex)
        tr = np.maximum(high[-period:] - low[-period:], 
                       np.abs(high[-period:] - close[-period-1:-1]))
        atr = np.mean(tr)
        # Approximate DI+/DI- with price movement
        up_move = high[-period:] - high[-period-1:-1]
        down_move = low[-period-1:-1] - low[-period:]
        di_plus = np.mean(np.maximum(up_move, 0)) / atr if atr > 0 else 0
        di_minus = np.mean(np.maximum(down_move, 0)) / atr if atr > 0 else 0
        dx = np.abs(di_plus - di_minus) / (di_plus + di_minus + 1e-8)
        adx = dx  # Simplified (should be smoothed)
        return float(np.clip(adx, 0, 1))
    
    def _calc_cci(self, high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 20) -> float:
        """Calculate CCI (Commodity Channel Index) [-1, 1]"""
        if len(close) < period:
            return 0.0
        tp = (high[-period:] + low[-period:] + close[-period:]) / 3
        sma_tp = np.mean(tp)
        mean_dev = np.mean(np.abs(tp - sma_tp))
        cci = (tp[-1] - sma_tp) / (0.015 * mean_dev) if mean_dev > 0 else 0.0
        return float(np.clip(cci / 100, -1, 1))  # Normalize to [-1, 1]
    
    def _calc_stochastic(self, high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> float:
        """Calculate Stochastic Oscillator [0, 1]"""
        if len(close) < period:
            return 0.5
        lowest_low = np.min(low[-period:])
        highest_high = np.max(high[-period:])
        if highest_high == lowest_low:
            return 0.5
        k = (close[-1] - lowest_low) / (highest_high - lowest_low)
        return float(np.clip(k, 0, 1))
    
    def _calculate_turbulence_local(self, close: np.ndarray, window: int = 20) -> float:
        """
        📈 Calculate Local Turbulence Index (Volatility of Log-Returns)
        
        Paper: "Multi-Agent Reinforcement Learning for Liquidation Strategy Analysis"
        Formula: Turbulence = σ(log_returns) normalized to [0, 1]
        
        Args:
            close: Array of closing prices
            window: Lookback window (default: 20)
        
        Returns:
            Turbulence index [0, 1] where:
            - 0.0 = Low volatility (calm market)
            - 1.0 = High volatility (turbulent market)
        """
        if len(close) < window + 1:
            return 0.0  # Not enough data, assume calm
        
        # Calculate log-returns over window
        log_returns = np.log(close[-window:] / close[-window-1:-1])
        
        # Volatility = standard deviation of log-returns
        volatility = np.std(log_returns)
        
        # Normalize to [0, 1] using typical Forex volatility range
        # Typical daily volatility: 0.001 (0.1%) to 0.02 (2%)
        # For 1H bars: scale down by sqrt(24) ≈ 4.9
        typical_max_vol = 0.02 / 4.9  # ≈ 0.004
        turbulence_norm = np.clip(volatility / typical_max_vol, 0.0, 1.0)
        
        return float(turbulence_norm)
    
    def _calculate_market_power(
        self, 
        high: np.ndarray, 
        low: np.ndarray, 
        volume: np.ndarray
    ) -> float:
        """
        💰 Calculate Market Power Estimate (VVR - Volume/Volatility Ratio)
        
        Concept: "Smart Money Absorption Detection"
        
        High volume + Low price movement = Institutional absorption (trap)
        Low volume + High price movement = Retail momentum (liquidity gap)
        
        Formula: P_t = Volume / (Range + ε)
        
        Where:
        - High P_t → Large volume absorbed with little movement → Smart Money
        - Low P_t → Movement driven by low volume → Retail/Momentum
        
        Args:
            high: Array of high prices
            low: Array of low prices
            volume: Array of volumes
        
        Returns:
            Market power estimate [0, 1] where:
            - 0.0 = Retail-driven (low volume, high movement)
            - 1.0 = Institution-driven (high volume, low movement)
        """
        if len(volume) < 1 or len(high) < 1:
            return 0.0
        
        # Current candle metrics
        current_volume = volume[-1]
        current_range = high[-1] - low[-1]
        
        # Avoid division by zero
        epsilon = 1e-8
        current_range = max(current_range, epsilon)
        
        # VVR: Volume/Volatility Ratio
        vvr = current_volume / current_range
        
        # Normalize to [0, 1] using historical context
        # Typical VVR range for 1H Forex: 1000 to 100000
        if len(volume) >= 20:
            vvr_mean = np.mean(volume[-20:] / (high[-20:] - low[-20:] + epsilon))
            vvr_std = np.std(volume[-20:] / (high[-20:] - low[-20:] + epsilon))
            
            # Z-score normalization
            if vvr_std > 0:
                vvr_zscore = (vvr - vvr_mean) / vvr_std
                # Map to [0, 1]: high z-score = high institutional presence
                market_power = 1 / (1 + np.exp(-vvr_zscore))  # Sigmoid
            else:
                market_power = 0.5  # Neutral if no variance
        else:
            market_power = 0.5  # Not enough data
        
        return float(np.clip(market_power, 0.0, 1.0))
    
    def _calculate_behavioral_bias(
        self,
        open_price: np.ndarray,
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray
    ) -> float:
        """
        🎯 Calculate Behavioral Bias Index (Wick/Range Ratio)
        
        Concept: "Retail Stop Hunt Detection"
        
        Large wicks = Price rejection = Stop hunt or liquidity grab
        Small wicks = Clean momentum = Retail following trend
        
        Formula: B_t = Wick_Length / Total_Range
        
        Where wick length is the larger of:
        - Upper wick: High - max(Open, Close)
        - Lower wick: min(Open, Close) - Low
        
        Args:
            open_price: Array of open prices
            high: Array of high prices
            low: Array of low prices
            close: Array of close prices
        
        Returns:
            Behavioral bias [0, 1] where:
            - 0.0 = Clean momentum (retail following)
            - 1.0 = Strong rejection (stop hunt, smart money trap)
        """
        if len(close) < 1:
            return 0.0
        
        # Current candle
        o = open_price[-1]
        h = high[-1]
        l = low[-1]
        c = close[-1]
        
        # Calculate wicks
        body_high = max(o, c)
        body_low = min(o, c)
        
        upper_wick = h - body_high
        lower_wick = body_low - l
        
        # Total range
        total_range = h - l
        epsilon = 1e-8
        total_range = max(total_range, epsilon)
        
        # Larger wick indicates stronger rejection
        max_wick = max(upper_wick, lower_wick)
        
        # Wick/Range ratio
        wick_ratio = max_wick / total_range
        
        # Enhance signal: if both wicks are large → strong trap
        both_wicks_large = (upper_wick > total_range * 0.3 and 
                           lower_wick > total_range * 0.3)
        
        if both_wicks_large:
            wick_ratio = min(1.0, wick_ratio * 1.5)  # Amplify signal
        
        return float(np.clip(wick_ratio, 0.0, 1.0))
    
    def _calculate_liquidity_phase(
        self,
        market_power: float,
        turbulence: float,
        volume: np.ndarray
    ) -> int:
        """
        🔄 Calculate Liquidity Phase Index (Market Cycle Classification)
        
        Concept: "Wyckoff Market Phases"
        
        Phases:
        - 0: Accumulation (Low turbulence, moderate-high volume, high absorption)
        - 1: Manipulation (High turbulence, extreme volume, high absorption)
        - 2: Distribution (Low turbulence, low volume, low absorption)
        
        Args:
            market_power: VVR score [0, 1]
            turbulence: Volatility score [0, 1]
            volume: Array of volumes
        
        Returns:
            Phase index: 0, 1, or 2
        """
        # Volume trend (increasing = 1, decreasing = -1)
        if len(volume) >= 5:
            recent_vol = np.mean(volume[-5:])
            past_vol = np.mean(volume[-20:-5]) if len(volume) >= 20 else recent_vol
            vol_trend = 1 if recent_vol > past_vol * 1.1 else -1
        else:
            vol_trend = 0
        
        # Classification logic
        if market_power > 0.6 and turbulence < 0.4:
            # High absorption, low volatility → Accumulation
            phase = 0
        elif market_power > 0.6 and turbulence > 0.6:
            # High absorption, high volatility → Manipulation
            phase = 1
        elif market_power < 0.4 and vol_trend == -1:
            # Low absorption, decreasing volume → Distribution
            phase = 2
        else:
            # Default: assume manipulation (safest assumption)
            phase = 1
        
        return int(phase)
    
    def _calculate_opportunity_score(self) -> float:
        """
        ⏰ Calculate Opportunity Score (Signal Clarity)
        
        Concept: "Meta-Awareness - When NOT to trade"
        
        Low opportunity = High spread volatility or erratic rewards
        High opportunity = Stable environment, clear signals
        
        Formula: Based on reward variance and spread stability
        
        Returns:
            Opportunity score [0, 1] where:
            - 0.0 = Low clarity, avoid trading
            - 1.0 = High clarity, good conditions
        """
        # Check reward stability
        if len(self.returns_history) >= 10:
            recent_returns = np.array(self.returns_history[-10:])
            return_std = np.std(recent_returns)
            
            # Normalize: high std = low opportunity
            # Typical return std: 0.0001 to 0.01
            opportunity_from_returns = 1.0 - np.clip(return_std / 0.01, 0.0, 1.0)
        else:
            opportunity_from_returns = 0.5
        
        # Check price stability (using current data if available)
        if self.df_ohlcv is not None and len(self.df_ohlcv) >= 10:
            recent_close = self.df_ohlcv['close'].values[-10:]
            price_std = np.std(recent_close) / np.mean(recent_close)
            
            # Normalize: high relative std = low opportunity
            opportunity_from_price = 1.0 - np.clip(price_std / 0.02, 0.0, 1.0)
        else:
            opportunity_from_price = 0.5
        
        # Combined score (weighted average)
        opportunity = 0.6 * opportunity_from_returns + 0.4 * opportunity_from_price
        
        return float(np.clip(opportunity, 0.0, 1.0))
    
    def _calculate_self_confidence(self) -> float:
        """
        🎯 Calculate Self-Confidence Score (Rolling Sharpe Ratio)
        
        Concept: "Agent's Self-Assessment"
        
        High confidence = Consistent positive returns, low variance
        Low confidence = Erratic returns, high variance
        
        Formula: C_t = rolling_mean(returns) / rolling_std(returns)
        
        Returns:
            Confidence score [0, 1] where:
            - 0.0 = No confidence (high variance, negative mean)
            - 1.0 = High confidence (consistent positive returns)
        """
        if len(self.returns_history) < 10:
            return 0.5  # Neutral until enough data
        
        # Rolling window (last 30 returns or less)
        window = min(30, len(self.returns_history))
        recent_returns = np.array(self.returns_history[-window:])
        
        mean_return = np.mean(recent_returns)
        std_return = np.std(recent_returns)
        
        if std_return > 0:
            # Sharpe-like ratio
            sharpe = mean_return / std_return
            
            # Map to [0, 1]: typical Sharpe range -2 to +2
            confidence = 1 / (1 + np.exp(-sharpe))  # Sigmoid
        else:
            # No variance → perfect consistency
            confidence = 1.0 if mean_return >= 0 else 0.0
        
        return float(np.clip(confidence, 0.0, 1.0))
    
    def _calculate_staleness_proxy(self, close: np.ndarray) -> float:
        """
        🔥 STALENESS FEATURE 1: Reference Price Deviation
        
        Quantifies how "stale" or "surprising" the current price is relative
        to its recent trajectory. High values indicate potential latency risk
        or abrupt volatility spikes.
        
        Based on: Stale Quotes Arbitrage paper - detects price quality degradation
        
        Formula: (Current Price - WMA_5) / Range_10
        
        Where:
        - WMA_5: Weighted Moving Average of last 5 prices
        - Range_10: Price range (high-low) over last 10 bars
        
        Returns:
            Staleness proxy [-1, 1] where:
            - ~0.0 = Price aligned with recent trajectory (fresh quote)
            - >0.5 = Price significantly above recent average (potential stale/spike)
            - <-0.5 = Price significantly below recent average (potential stale/drop)
        """
        if len(close) < 10:
            return 0.0  # Insufficient data
        
        current_price = close[-1]
        
        # Weighted Moving Average (last 5 prices, linearly weighted)
        lookback_short = min(5, len(close))
        recent_prices = close[-lookback_short:]
        weights = np.linspace(1, lookback_short, lookback_short)
        weights = weights / weights.sum()
        wma_5 = np.sum(recent_prices * weights)
        
        # Price range over last 10 bars
        lookback_range = min(10, len(close))
        price_range = np.max(close[-lookback_range:]) - np.min(close[-lookback_range:])
        
        # Avoid division by zero
        if price_range < 1e-8:
            return 0.0
        
        # Calculate deviation
        deviation = (current_price - wma_5) / price_range
        
        # Clip to reasonable range (±2 sigma equivalent)
        staleness = np.clip(deviation, -1.0, 1.0)
        
        return float(staleness)
    
    def _calculate_order_imbalance_proxy(self, close: np.ndarray) -> float:
        """
        🔥 STALENESS FEATURE 2: Short-Term Order Imbalance Proxy
        
        Measures impulsive buy/sell pressure in the shortest window.
        High imbalance precedes:
        - Slippage (cost of latency arbitrage)
        - Whipsaw risk (sudden reversals)
        - Breakout continuation (for USDJPY/XAUUSD)
        
        Approximation: Standard deviation of returns in ultra-short window
        (True order imbalance requires order book data, unavailable in CSV)
        
        Formula: StdDev(returns[-5:]) normalized
        
        Returns:
            Imbalance proxy [0, 1] where:
            - ~0.0 = Balanced, calm market
            - >0.5 = High imbalance, impulsive moves (caution for GBPUSD, opportunity for XAUUSD)
        """
        if len(close) < 6:
            return 0.0  # Insufficient data
        
        # Calculate returns over last 5 bars
        lookback = min(5, len(close) - 1)
        recent_close = close[-(lookback+1):]
        returns = np.diff(np.log(recent_close))
        
        # Standard deviation of returns (volatility proxy)
        if len(returns) == 0:
            return 0.0
        
        std_returns = np.std(returns)
        
        # Normalize: typical intraday volatility ~0.001 to 0.01 (0.1% to 1%)
        # Map to [0, 1] with sigmoid-like scaling
        imbalance = std_returns / 0.005  # 0.5% is midpoint
        imbalance_norm = np.clip(imbalance, 0.0, 1.0)
        
        return float(imbalance_norm)
    
    def _apply_shield(self, action: np.ndarray) -> np.ndarray:
        """
        Apply safety shield to validate action
        
        Args:
            action: Proposed action
        
        Returns:
            Validated action (may be modified by shield)
        """
        # Extract position size (handle both scalar and array)
        if isinstance(action, np.ndarray) and action.size == 1:
            position_size = float(action[0])
        else:
            position_size = float(action)
        
        # Get account state
        peak_balance = max(self.equity_history) if self.equity_history else self.balance
        dd_pct = (peak_balance - self.equity) / peak_balance if peak_balance > 0 else 0.0
        
        account_state = {
            'balance': self.balance,
            'equity': self.equity,
            'daily_dd_pct': dd_pct,  # Simplified (should be daily)
            'total_dd_pct': dd_pct,
            'open_positions': [{'size': self.position_size}] if abs(self.position_size) > 0.01 else [],
            'daily_high_equity': peak_balance,
            'initial_balance': self.initial_balance_snapshot,
            'margin_level': 100.0,  # Placeholder
        }
        
        # Build order from action (single position size value)
        # Actor outputs [-0.1, 0.1] which we use directly as lot sizes
        # 0.1 lot = ~$10k notional (with 100:1 leverage) = ~10% of $100k capital
        # This matches ComplianceShield's max_lot_size = 0.1 constraint
        
        # Note: ComplianceShield expects 'type' key, not 'action'
        if abs(position_size) < 0.01:
            # Close or wait
            order = {
                'type': 'wait',  # No position
                'symbol': self.symbol,
            }
        else:
            # Open new position or modify existing
            order = {
                'type': 'open',
                'symbol': self.symbol,
                'direction': 'buy' if position_size > 0 else 'sell',
                'lot_size': abs(position_size),  # Use actor output directly as lot size
                'risk_pct': abs(position_size) * 0.02,  # Simplified risk calculation
                'stop_loss': None,  # Optional
                'take_profit': None,  # Optional
            }
        
        # Validate
        is_safe, corrected_order = self.shield.validate_action(order, account_state)
        
        if not is_safe:
            logger.debug(f"Shield blocked action: {corrected_order.get('reason', 'unknown')}")
            
            # Convert corrected order back to action
            corrected_action_type = corrected_order.get('action', corrected_order.get('type', 'wait'))
            
            if corrected_action_type == 'close_all':
                return np.array([0.0])  # Force close
            elif corrected_action_type == 'block_new_trades':
                return np.array([self.position_size])  # Maintain current position
            elif corrected_action_type == 'wait':
                return np.array([0.0])  # Neutral position
            else:
                # Reduce lot size if provided
                new_size = corrected_order.get('lot_size', abs(position_size))
                return np.array([new_size * np.sign(position_size)])
        
        return action
    
    def _execute_action(self, action: np.ndarray):
        """
        Execute trading action
        
        Args:
            action: [position_size] - single continuous value in [-1, 1]
                    Directly sets target position size
        """
        # Extract position size (handle both scalar and array)
        if isinstance(action, np.ndarray) and action.size == 1:
            target_position = float(action[0]) * self.config.max_position_size
        else:
            target_position = float(action) * self.config.max_position_size
        
        # Update position directly (continuous control)
        if abs(target_position) < 0.01:
            # Target is neutral → close position
            if abs(self.position_size) > 0.01:
                self._close_position()
        else:
            # Target is long/short → open or modify position
            if abs(self.position_size) < 0.01:
                self._open_position(target_position)
            else:
                self._modify_position(target_position)
    
    def _open_position(self, size: float):
        """Open new position"""
        self.position_size = size
        self.position_entry_price = self.current_price
        self.position_entry_step = self.current_step
        
        # 🔥 CRITICAL: Track staleness at entry for reward shaping
        # This enables the DDPG fakeout penalty to scale with price quality
        if hasattr(self, 'df_ohlcv') and self.df_ohlcv is not None:
            idx = self.config.lookback_bars + self.current_step
            close = self.df_ohlcv['close'].values[:idx+1]
            self.position_entry_staleness = self._calculate_staleness_proxy(close)
        else:
            self.position_entry_staleness = 0.0
        
        # Deduct commission
        commission = abs(size) * self.current_price * self.config.commission_pct
        old_balance = self.balance
        self.balance -= commission
        
        logger.info(
            f"OPENED position: size={size:.4f}, price={self.current_price:.5f}, "
            f"commission=${commission:.2f}, balance: ${old_balance:.2f} -> ${self.balance:.2f}"
        )
    
    def _close_position(self):
        """Close current position"""
        if abs(self.position_size) < 0.01:
            return
        
        # Calculate P&L
        price_diff = self.current_price - self.position_entry_price
        pnl = self.position_size * price_diff * 100_000  # 1 lot = 100k units
        
        # Deduct commission + spread
        commission = abs(self.position_size) * self.current_price * self.config.commission_pct
        spread_cost = abs(self.position_size) * self.config.spread_pips * 10  # $10 per pip
        
        net_pnl = pnl - commission - spread_cost
        old_balance = self.balance
        self.balance += net_pnl
        
        logger.info(
            f"CLOSED position: size={self.position_size:.4f}, "
            f"entry={self.position_entry_price:.5f}, exit={self.current_price:.5f}, "
            f"pnl=${net_pnl:.2f}, balance: ${old_balance:.2f} -> ${self.balance:.2f}"
        )
        
        # Record trade
        self.trades.append({
            'entry_step': self.position_entry_step,
            'exit_step': self.current_step,
            'entry_price': self.position_entry_price,
            'exit_price': self.current_price,
            'size': self.position_size,
            'pnl': net_pnl,
            'duration': self.current_step - self.position_entry_step,
        })
        
        # Reset position
        self.position_size = 0.0
        self.position_entry_price = 0.0
        self.position_entry_staleness = 0.0  # 🔥 Reset staleness
    
    def _modify_position(self, new_size: float):
        """Modify existing position (close and reopen)"""
        self._close_position()
        self._open_position(new_size)
    
    def _update_equity(self):
        """Update equity (balance + unrealized P&L)"""
        unrealized_pnl = 0.0
        
        if abs(self.position_size) > 0.01:
            price_diff = self.current_price - self.position_entry_price
            unrealized_pnl = self.position_size * price_diff * 100_000
        
        old_equity = self.equity
        self.equity = self.balance + unrealized_pnl
        
        # Log only if there's significant change
        if abs(self.equity - old_equity) > 10:
            logger.debug(
                f"Equity update: balance=${self.balance:.2f}, "
                f"unrealized_pnl=${unrealized_pnl:.2f}, equity=${self.equity:.2f}"
            )
    
    def _calculate_reward(self) -> float:
        """
        Calculate reward with CMDP constraints (Sharpe Ratio - DD penalties)
        
        🛡️ CRITICAL IMPLEMENTATION: Constrained MDP for Prop Firm Safety
        - Daily DD breach → PENALTY_DAILY (-1000)
        - Total DD breach → PENALTY_TOTAL (-10000)
        - Violations trigger episode termination
        
        Returns:
            Reward value (can be heavily negative on violations)
        """
        # Calculate returns
        if len(self.equity_history) >= 2:
            returns = (self.equity - self.equity_history[-1]) / self.equity_history[-1]
            self.returns_history.append(returns)
        
        # Update peaks for DD calculation
        self.peak_balance = max(self.peak_balance, self.equity)
        self.daily_peak_balance = max(self.daily_peak_balance, self.equity)
        
        # 🛡️ CRITICAL: Calculate Drawdowns
        # Total DD: Distance from peak balance
        total_dd_usd = self.peak_balance - self.equity
        total_dd_pct = total_dd_usd / self.peak_balance if self.peak_balance > 0 else 0.0
        
        # Daily DD: Distance from daily peak (resets every 24 bars if 1H, or daily)
        daily_dd_usd = self.daily_peak_balance - self.equity
        daily_dd_pct = daily_dd_usd / self.daily_peak_balance if self.daily_peak_balance > 0 else 0.0
        
        # Daily reset logic (every 24 bars for 1H timeframe)
        if self.current_step % 24 == 0 and self.current_step > 0:
            self.daily_start_balance = self.equity
            self.daily_peak_balance = self.equity
            self.current_day += 1
        
        reward = 0.0
        terminated = False  # Track if we should terminate
        
        # 🛡️ CMDP CONSTRAINT 1: Daily Drawdown Check
        if daily_dd_pct > self.config.max_daily_dd_pct:
            reward = self.config.penalty_daily_dd
            terminated = True
            logger.warning(f"DAILY DD BREACH: {daily_dd_pct:.2%} > {self.config.max_daily_dd_pct:.2%} | Penalty: {reward:.0f}")
        
        # 🛡️ CMDP CONSTRAINT 2: Total Drawdown Check (more severe)
        elif total_dd_pct > self.config.max_total_dd_pct:
            reward = self.config.penalty_total_dd
            terminated = True
            logger.error(f"TOTAL DD BREACH: {total_dd_pct:.2%} > {self.config.max_total_dd_pct:.2%} | Penalty: {reward:.0f}")
        
        # Normal reward calculation (only if no DD breach)
        else:
            # 1. Sharpe Ratio (rolling window) - BASE REWARD
            if len(self.returns_history) >= self.config.sharpe_window:
                recent_returns = self.returns_history[-self.config.sharpe_window:]
                mean_return = np.mean(recent_returns)
                std_return = np.std(recent_returns)
                
                if std_return > 1e-8:
                    sharpe = mean_return / std_return
                    reward += sharpe
            
            # 🔥 1.5. PPO VOLATILITY-ADJUSTED REWARD (USDJPY only): Penalize PnL volatility
            # Encourages stable, consistent returns (higher effective Sharpe)
            if self.config.symbol == 'USDJPY':
                volatility_penalty = self._calculate_volatility_penalty()
                reward += volatility_penalty  # Will be negative based on PnL variance
            
            # 🔥 1.6. DDPG ASYMMETRIC REWARD (GBPUSD only): Fakeout Penalty scaled by Staleness
            # Penalizes closing in loss within N steps, especially if entered on stale quote
            if self.config.symbol == 'GBPUSD':
                fakeout_penalty = self._calculate_fakeout_penalty()
                reward += fakeout_penalty  # Will be negative if applicable
            
            # 2. IMPROVED Soft DD penalty (exponential near limits)
            # Penalize both daily and total DD progressively
            daily_proximity = daily_dd_pct / self.config.max_daily_dd_pct  # 0.0 to 1.0
            total_proximity = total_dd_pct / self.config.max_total_dd_pct  # 0.0 to 1.0
            
            # Exponential penalty that grows rapidly as we approach limits
            # At 80% of limit: -10, at 90%: -50, at 95%: -100
            if daily_proximity > 0.5:  # Start penalizing at 50% of limit
                daily_penalty = -100 * (daily_proximity ** 4)
                reward += daily_penalty
            
            if total_proximity > 0.5:
                total_penalty = -200 * (total_proximity ** 4)  # Heavier penalty for total DD
                reward += total_penalty
            
            # 3. Compliance bonus (staying well below limits)
            if daily_dd_pct < 0.025 and total_dd_pct < 0.05:  # Well below limits
                reward += self.config.compliance_bonus
            
            # 💰 4. SMART CONTRARIAN BONUS: Reward anti-retail behavior
            # If behavioral_bias is high (stop hunt/trap detected) and we're NOT following it
            if hasattr(self, '_last_behavioral_bias'):
                behavioral_bias = self._last_behavioral_bias
                
                # Detect if agent traded AGAINST the retail bias
                # High bias + opposite position = contrarian (good)
                if behavioral_bias > 0.6:  # Strong retail trap detected
                    # Check if position is contrary to apparent momentum
                    if self.position_size != 0:
                        # Simplified: reward exists if we're positioned during high bias
                        # (assumes we're likely trading contrarian in manipulative phase)
                        contrarian_bonus = 0.05 * (1.0 - behavioral_bias)
                        reward += contrarian_bonus
        
        # Store termination flag for step() to handle
        self._dd_termination_flag = terminated
        
        return float(reward)
    
    def _calculate_volatility_penalty(self) -> float:
        """
        🔥 PPO VOLATILITY-ADJUSTED REWARD: Penalize PnL volatility
        
        Encourages PPO agent (USDJPY) to select smoother, more consistent
        return trajectories. This complements GAE (Generalized Advantage Estimation)
        by directly rewarding stability.
        
        Based on: Sharpe Ratio optimization principle
        
        Formula:
            penalty = -β * rolling_volatility(PnL, window=20)
        
        Where:
        - β: Volatility penalty coefficient (default 0.5)
        - rolling_volatility: Std dev of recent PnL changes
        
        Impact:
        - Stable PnL trajectory: low volatility → minimal penalty
        - Erratic PnL trajectory: high volatility → significant penalty
        
        Returns:
            Penalty value (negative or 0)
        """
        # Need sufficient history to calculate volatility
        volatility_window = 20
        if len(self.equity_history) < volatility_window + 1:
            return 0.0  # Insufficient data
        
        # Calculate PnL changes over rolling window
        recent_equity = np.array(self.equity_history[-volatility_window-1:])
        pnl_changes = np.diff(recent_equity)
        
        # Calculate volatility (std dev of PnL changes)
        volatility = np.std(pnl_changes)
        
        # Penalty coefficient (β): controls strength of volatility aversion
        # Higher β = stronger preference for stability
        beta = 0.5
        
        # Normalize volatility by initial balance to make scale-invariant
        normalized_volatility = volatility / self.initial_balance_snapshot
        
        # Calculate penalty
        penalty = -beta * normalized_volatility * 1000  # Scale for reward magnitude
        
        logger.debug(
            f"Volatility penalty: vol={volatility:.2f}, "
            f"norm_vol={normalized_volatility:.4f}, penalty={penalty:.2f}"
        )
        
        return penalty
    
    def _calculate_fakeout_penalty(self) -> float:
        """
        🔥 DDPG ASYMMETRIC REWARD: Fakeout Penalty scaled by Staleness
        
        Penalizes GBPUSD agent for:
        1. Closing position in loss (whipsaw)
        2. Holding < N steps (impulsive trade)
        3. Entering on stale/low-quality price (high staleness)
        
        Based on: Stale Quotes Arbitrage paper + HMARL advisor mandate
        
        Formula:
            penalty = -abs_loss * 2.0 * (1.0 + staleness_at_entry)
        
        Where:
        - abs_loss: Absolute PnL loss on closed trade
        - 2.0: Base whipsaw penalty multiplier
        - staleness_at_entry: Staleness proxy at position open [0, 1]
        
        Impact:
        - Entry on fresh quote (staleness=0): penalty = -2.0 * loss
        - Entry on stale quote (staleness=0.5): penalty = -3.0 * loss
        - Entry on very stale quote (staleness=1.0): penalty = -4.0 * loss
        
        Returns:
            Penalty value (negative or 0)
        """
        # Only apply if we just closed a position
        if abs(self.position_size) > 0.01:
            return 0.0  # Position still open, no penalty
        
        # Check if we closed a position this step
        # (position_size now 0, but position_entry_price != 0)
        if abs(self.position_entry_price) < 1e-6:
            return 0.0  # No recent position
        
        # Get last trade (just closed)
        if len(self.trades) == 0:
            return 0.0
        
        last_trade = self.trades[-1]
        pnl = last_trade.get('pnl', 0.0)
        duration_steps = last_trade.get('duration', 0)
        
        # Only penalize if:
        # 1. Closed in loss (PnL < 0)
        # 2. Duration < fakeout threshold (default 10 steps)
        fakeout_threshold_steps = 10
        
        if pnl < 0 and duration_steps < fakeout_threshold_steps:
            abs_loss = abs(pnl)
            staleness_at_entry = getattr(self, 'position_entry_staleness', 0.0)
            
            # Staleness scaling: (1.0 + staleness) ranges from 1.0 to 2.0
            staleness_multiplier = 1.0 + staleness_at_entry
            
            # Total penalty: base (2.0) * staleness multiplier * loss
            penalty = -abs_loss * 2.0 * staleness_multiplier
            
            logger.debug(
                f"Fakeout penalty: loss=${abs_loss:.2f}, duration={duration_steps}, "
                f"staleness={staleness_at_entry:.3f}, penalty={penalty:.2f}"
            )
            
            return penalty
        
        return 0.0  # No penalty
    
    def _check_termination(self) -> bool:
        """
        Check if episode should terminate
        
        🛡️ CRITICAL: CMDP termination on DD breach
        
        Returns:
            True if episode should terminate
        """
        # 1. Check CMDP DD violation flag (set by _calculate_reward)
        if hasattr(self, '_dd_termination_flag') and self._dd_termination_flag:
            if self.config.terminate_on_dd_breach:
                logger.warning("Episode terminated: Drawdown limit breached (CMDP constraint)")
                return True
        
        # 2. Margin call (equity drops below 50% of initial)
        if self.equity < self.initial_balance_snapshot * 0.5:
            logger.warning(f"Episode terminated: equity={self.equity:.2f} (margin call)")
            return True
        
        return False
    
    def _get_info(self) -> Dict[str, Any]:
        """Get episode info with CMDP metrics + Trading Statistics"""
        peak_equity = max(self.equity_history) if self.equity_history else self.balance
        total_dd_pct = (peak_equity - self.equity) / peak_equity if peak_equity > 0 else 0.0
        
        # Daily DD
        daily_dd_usd = self.daily_peak_balance - self.equity
        daily_dd_pct = daily_dd_usd / self.daily_peak_balance if self.daily_peak_balance > 0 else 0.0
        
        # 📊 Calculate Sharpe Ratio
        sharpe_ratio = 0.0
        if len(self.returns_history) > 1:
            returns_array = np.array(self.returns_history)
            mean_return = np.mean(returns_array)
            std_return = np.std(returns_array)
            if std_return > 0:
                # Simple Sharpe (non-annualized for episode-level reporting)
                sharpe_ratio = mean_return / std_return
        
        # 📊 Calculate Win Rate
        win_rate = 0.0
        if len(self.trades) > 0:
            wins = sum(1 for trade in self.trades if trade.get('pnl', 0) > 0)
            win_rate = wins / len(self.trades)
        
        return {
            'step': self.current_step,
            'balance': self.balance,
            'equity': self.equity,
            'position_size': self.position_size,
            'drawdown_pct': total_dd_pct,  # Total DD
            'max_drawdown': total_dd_pct,  # ✅ Alias for training script
            'daily_dd_pct': daily_dd_pct,  # 🛡️ Daily DD
            'daily_dd_usd': daily_dd_usd,  # 🛡️ Daily DD absolute
            'num_trades': len(self.trades),
            'current_price': self.current_price,
            'peak_balance': peak_equity,
            'daily_peak_balance': self.daily_peak_balance,
            'sharpe_ratio': sharpe_ratio,  # ✅ NEW
            'win_rate': win_rate,  # ✅ NEW
        }
    
    def render(self):
        """Render environment (human-readable)"""
        if self.render_mode == 'human':
            info = self._get_info()
            print(f"Step {info['step']}: Balance=${info['balance']:.2f}, "
                  f"Equity=${info['equity']:.2f}, DD={info['drawdown_pct']*100:.2f}%, "
                  f"Position={info['position_size']:.2f}, Trades={info['num_trades']}")


# ============================================================================
# CLI Test
# ============================================================================

if __name__ == "__main__":
    print("Testing ForexTradingEnv...")
    
    # Generate synthetic OHLCV data
    np.random.seed(42)
    num_bars = 2000
    
    dates = pd.date_range(start='2024-01-01', periods=num_bars, freq='1H')
    price = 1.1000
    prices = [price]
    
    for _ in range(num_bars - 1):
        price += np.random.normal(0, 0.0005)  # Random walk
        prices.append(price)
    
    df = pd.DataFrame({
        'timestamp': dates,
        'open': prices,
        'high': [p * 1.0001 for p in prices],
        'low': [p * 0.9999 for p in prices],
        'close': prices,
        'volume': [1000] * num_bars,
    })
    df.set_index('timestamp', inplace=True)
    
    print(f"\n✓ Generated {len(df)} bars of synthetic OHLCV data")
    
    # Create environment (disable shield for faster testing)
    config = TradingEnvConfig(max_steps=500, use_safety_shield=False)
    env = ForexTradingEnv(config=config, historical_data=df, render_mode='human')
    
    print(f"\n✓ Environment created")
    print(f"  Observation space: {env.observation_space}")
    print(f"  Action space: {env.action_space}")
    
    # Run episode
    obs, info = env.reset()
    print(f"\n✓ Episode started")
    print(f"  Initial observation: {obs[:3]}... (14 dims)")
    print(f"  Initial balance: ${info['balance']:.2f}")
    
    # Random agent
    total_reward = 0
    for step in range(10):  # Run 10 steps
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        if step < 3:  # Print first 3 steps
            env.render()
    
    print(f"\n✓ Episode completed 10 steps")
    print(f"  Total reward: {total_reward:.4f}")
    print(f"  Final equity: ${info['equity']:.2f}")
    print(f"  Trades executed: {info['num_trades']}")
    
    print("\n✓ Test complete")
