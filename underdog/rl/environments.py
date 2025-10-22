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
        
        # Data source
        self.db = db_connector
        self.historical_data = historical_data
        
        # State builder
        state_config = StateVectorConfig(use_redis=False)  # Disable Redis for training
        self.state_builder = StateVectorBuilder(db_connector, state_config) if db_connector else None
        
        # Safety shield
        self.shield = None
        if self.config.use_safety_shield:
            constraints = SafetyConstraints(
                max_daily_dd_pct=0.05,
                max_total_dd_pct=0.10,
                max_positions=2,
                max_risk_per_trade=0.015
            )
            self.shield = PropFirmSafetyShield(constraints)
        
        # Define spaces
        # 🧠 EXPANDED TO 24D for CMDP + Enhanced Technical Analysis
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(24,),
            dtype=np.float32
        )
        
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(2,),
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
        
        logger.debug(f"Episode reset: balance={self.balance:.2f}, steps={self.config.max_steps}")
        
        return obs, info
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Execute one time step
        
        Args:
            action: Action vector [position_size, entry_exit]
        
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
            
            # [16] Volume ratio
            volume_ratio = volume[-1] / np.mean(volume[-20:]) if len(volume) >= 20 else 1.0
            
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
            
            # 🛡️ [21] Market Turbulence Index (ATR-based volatility proxy)
            turbulence = atr_norm  # Simplified: use ATR as proxy
            
            # [22-23] Additional technical indicators
            stochastic = self._calc_stochastic(high, low, close, period=14)
            momentum = (close[-1] - close[-10]) / close[-10] if len(close) >= 10 else 0.0
            momentum = np.clip(momentum, -0.1, 0.1) / 0.1  # Normalize to [-1, 1]
            
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
            ], dtype=np.float32)
            
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
    
    def _apply_shield(self, action: np.ndarray) -> np.ndarray:
        """
        Apply safety shield to validate action
        
        Args:
            action: Proposed action
        
        Returns:
            Validated action (may be modified by shield)
        """
        # Get account state
        peak_balance = max(self.equity_history) if self.equity_history else self.balance
        dd_pct = (peak_balance - self.equity) / peak_balance if peak_balance > 0 else 0.0
        
        account_state = {
            'balance': self.balance,
            'equity': self.equity,
            'daily_dd_pct': dd_pct,  # Simplified (should be daily)
            'total_dd_pct': dd_pct,
            'open_positions': 1 if abs(self.position_size) > 0.01 else 0,
            'margin_level': 100.0,  # Placeholder
        }
        
        # Build order from action
        order = {
            'action': 'open' if action[1] > 0 else 'close',
            'side': 'long' if action[0] > 0 else 'short',
            'lot_size': abs(action[0]),
            'risk_pct': abs(action[0]) * 0.02,  # Simplified risk calculation
        }
        
        # Validate
        is_safe, corrected_order = self.shield.validate_action(order, account_state)
        
        if not is_safe:
            logger.debug(f"Shield blocked action: {corrected_order.get('reason', 'unknown')}")
            
            # Convert corrected order back to action
            if corrected_order['action'] == 'close_all':
                return np.array([0.0, -1.0])  # Force close
            elif corrected_order['action'] == 'block_new_trades':
                return np.array([self.position_size, -1.0])  # Maintain position
            else:
                # Reduce lot size
                new_size = corrected_order.get('lot_size', abs(action[0]))
                return np.array([new_size * np.sign(action[0]), action[1]])
        
        return action
    
    def _execute_action(self, action: np.ndarray):
        """
        Execute trading action
        
        Args:
            action: [position_size, entry_exit]
        """
        target_position = action[0] * self.config.max_position_size
        entry_exit_signal = action[1]
        
        # Determine if we should open/close position
        if entry_exit_signal < -0.5:  # Close signal
            if abs(self.position_size) > 0.01:
                self._close_position()
        elif entry_exit_signal > 0.5:  # Open signal
            if abs(self.position_size) < 0.01:
                self._open_position(target_position)
            else:
                # Modify existing position
                self._modify_position(target_position)
    
    def _open_position(self, size: float):
        """Open new position"""
        self.position_size = size
        self.position_entry_price = self.current_price
        self.position_entry_step = self.current_step
        
        # Deduct commission
        commission = abs(size) * self.current_price * self.config.commission_pct
        self.balance -= commission
        
        logger.debug(f"Opened position: size={size:.4f}, price={self.current_price:.5f}")
    
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
        self.balance += net_pnl
        
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
        
        logger.debug(f"Closed position: PnL={net_pnl:.2f}, duration={self.current_step - self.position_entry_step}")
        
        # Reset position
        self.position_size = 0.0
        self.position_entry_price = 0.0
    
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
        
        self.equity = self.balance + unrealized_pnl
    
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
            logger.warning(f"⚠️ DAILY DD BREACH: {daily_dd_pct:.2%} > {self.config.max_daily_dd_pct:.2%} | Penalty: {reward:.0f}")
        
        # 🛡️ CMDP CONSTRAINT 2: Total Drawdown Check (more severe)
        elif total_dd_pct > self.config.max_total_dd_pct:
            reward = self.config.penalty_total_dd
            terminated = True
            logger.error(f"🚨 TOTAL DD BREACH: {total_dd_pct:.2%} > {self.config.max_total_dd_pct:.2%} | Penalty: {reward:.0f}")
        
        # Normal reward calculation (only if no DD breach)
        else:
            # 1. Sharpe Ratio (rolling window)
            if len(self.returns_history) >= self.config.sharpe_window:
                recent_returns = self.returns_history[-self.config.sharpe_window:]
                mean_return = np.mean(recent_returns)
                std_return = np.std(recent_returns)
                
                if std_return > 1e-8:
                    sharpe = mean_return / std_return
                    reward += sharpe
            
            # 2. Soft DD penalty (quadratic, before hard limits)
            if total_dd_pct > 0:
                reward -= self.config.dd_penalty_factor * (total_dd_pct ** 2)
            
            # 3. Compliance bonus (staying well below limits)
            if daily_dd_pct < 0.03 and total_dd_pct < 0.05:  # Well below limits
                reward += self.config.compliance_bonus
        
        # Store termination flag for step() to handle
        self._dd_termination_flag = terminated
        
        return float(reward)
    
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
                logger.warning("🚨 Episode terminated: Drawdown limit breached (CMDP constraint)")
                return True
        
        # 2. Margin call (equity drops below 50% of initial)
        if self.equity < self.initial_balance_snapshot * 0.5:
            logger.warning(f"Episode terminated: equity={self.equity:.2f} (margin call)")
            return True
        
        return False
    
    def _get_info(self) -> Dict[str, Any]:
        """Get episode info with CMDP metrics"""
        peak_equity = max(self.equity_history) if self.equity_history else self.balance
        total_dd_pct = (peak_equity - self.equity) / peak_equity if peak_equity > 0 else 0.0
        
        # Daily DD
        daily_dd_usd = self.daily_peak_balance - self.equity
        daily_dd_pct = daily_dd_usd / self.daily_peak_balance if self.daily_peak_balance > 0 else 0.0
        
        return {
            'step': self.current_step,
            'balance': self.balance,
            'equity': self.equity,
            'position_size': self.position_size,
            'drawdown_pct': total_dd_pct,  # Total DD
            'daily_dd_pct': daily_dd_pct,  # 🛡️ Daily DD
            'daily_dd_usd': daily_dd_usd,  # 🛡️ Daily DD absolute
            'num_trades': len(self.trades),
            'current_price': self.current_price,
            'peak_balance': peak_equity,
            'daily_peak_balance': self.daily_peak_balance,
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
