"""
Base EA Framework - Abstract Classes for Trading Strategies
===========================================================

PROPÓSITO:
----------
Clases base abstractas para implementar EAs de manera consistente, con:
- Compliance automático (Drawdown Monitor)
- Ejecución robusta (Fault-Tolerant Executor)
- Position sizing adaptativo (Confidence-Weighted)
- Regime filtering (ADX + ATR)

JERARQUÍA DE CLASES:
--------------------
BaseEA (Abstract)
  ├── TrendFollowingEA
  │   ├── SuperTrendRSI
  │   └── ParabolicEMA
  ├── MeanReversionEA
  │   ├── BollingerCCI
  │   └── PairArbitrage
  ├── BreakoutEA
  │   ├── KeltnerBreakout
  │   └── ATRBreakout
  └── ScalpingEA
      └── EmaScalper

INTEGRATION PATTERN:
--------------------
```python
class MyEA(TrendFollowingEA):
    def generate_signal(self) -> Optional[Signal]:
        # Implementar lógica específica
        pass
    
    def manage_positions(self):
        # Implementar trailing stops, exits
        pass

# Usage
ea = MyEA(config)
ea.on_tick()  # Call on every tick
```

AUTHOR: UNDERDOG Development Team
DATE: October 2025
"""

import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, List, Dict, Tuple
from datetime import datetime
from enum import Enum
import logging

from underdog.risk_management.floating_drawdown_monitor import (
    FloatingDrawdownMonitor, DrawdownLevel, DrawdownLimits
)
from underdog.execution.fault_tolerant_executor import (
    FaultTolerantExecutor, OrderResult
)
from underdog.risk_management.position_sizing import (
    PositionSizer, confidence_weighted_sizing
)
from underdog.strategies.filters.regime_filter import (
    RegimeFilter, MarketRegime
)

logger = logging.getLogger(__name__)


# ============================================================================
# ENUMS & DATA CLASSES
# ============================================================================

class SignalType(Enum):
    """Trading signal type"""
    BUY = "BUY"
    SELL = "SELL"
    CLOSE_BUY = "CLOSE_BUY"
    CLOSE_SELL = "CLOSE_SELL"
    NONE = "NONE"


@dataclass
class Signal:
    """Trading signal with entry/exit levels"""
    type: SignalType
    entry_price: float
    sl: float
    tp: float
    confidence: float  # 0.0-1.0 (for position sizing)
    comment: str = ""
    
    def __str__(self):
        return (
            f"Signal({self.type.value}): Entry={self.entry_price:.5f}, "
            f"SL={self.sl:.5f}, TP={self.tp:.5f}, Conf={self.confidence:.2f}"
        )


@dataclass
class EAConfig:
    """Base configuration for all EAs"""
    # Symbol & Timeframe
    symbol: str
    timeframe: int  # MT5 timeframe constant
    
    # Risk Management
    risk_per_trade: float = 1.5  # % of account
    max_daily_dd: float = 5.0    # %
    max_total_dd: float = 10.0   # %
    max_floating_dd: float = 1.0  # % (Instant Funding)
    
    # Position Management
    max_positions: int = 1
    magic_number: int = 0
    
    # Execution
    max_slippage: int = 20  # pips
    max_retries: int = 5
    retry_sleep_ms: int = 300
    
    # Regime Filter
    use_regime_filter: bool = True
    regime_filter_timeframe: int = mt5.TIMEFRAME_M15
    
    # Logging
    log_signals: bool = True
    log_positions: bool = True


# ============================================================================
# BASE EA (Abstract)
# ============================================================================

class BaseEA(ABC):
    """
    Abstract base class for all EAs.
    
    RESPONSIBILITIES:
    -----------------
    1. Check drawdown (CRITICAL)
    2. Manage existing positions
    3. Generate new signals
    4. Execute orders with retry logic
    5. Position sizing with confidence weighting
    6. Regime filtering
    
    SUBCLASS MUST IMPLEMENT:
    ------------------------
    - generate_signal(): Signal generation logic
    - manage_positions(): Position management (trailing, exits)
    """
    
    def __init__(self, config: EAConfig):
        """
        Initialize EA with configuration.
        
        Args:
            config: EA configuration
        """
        self.config = config
        
        # Core components
        self.dd_monitor = FloatingDrawdownMonitor(
            limits=DrawdownLimits(
                max_daily_dd_pct=config.max_daily_dd,
                max_total_dd_pct=config.max_total_dd,
                max_floating_dd_pct=config.max_floating_dd
            )
        )
        
        self.executor = FaultTolerantExecutor()
        self.sizer = PositionSizer()
        
        # Regime filter (optional)
        if config.use_regime_filter:
            self.regime_filter = RegimeFilter(
                symbol=config.symbol,
                timeframe=config.regime_filter_timeframe
            )
        else:
            self.regime_filter = None
        
        # State tracking
        self.positions: Dict[int, Dict] = {}  # ticket → position info
        self.signals_history: List[Signal] = []
        self.stats = {
            'total_signals': 0,
            'executed_trades': 0,
            'failed_trades': 0,
            'total_profit': 0.0
        }
        
        logger.info(
            f"[{self.__class__.__name__}] Initialized for {config.symbol} "
            f"(Magic={config.magic_number})"
        )
    
    def on_tick(self):
        """
        Main tick handler - Called on every price update.
        
        **CRITICAL ORDER**:
        1. Check drawdown (FIRST)
        2. Check regime filter
        3. Manage existing positions
        4. Generate new signals
        5. Execute entries
        """
        # 1. CHECK DRAWDOWN (CRITICAL)
        dd_state = self.dd_monitor.check_drawdown()
        
        if dd_state.is_breached:
            logger.error(
                f"[{self.__class__.__name__}] BREACH DETECTED - HALTING TRADING"
            )
            return
        
        if dd_state.level == DrawdownLevel.CRITICAL:
            logger.warning(
                f"[{self.__class__.__name__}] CRITICAL DD - Only managing positions"
            )
            self.manage_positions()
            return
        
        # Adjust risk if in WARNING state
        risk_multiplier = 1.0
        if dd_state.level == DrawdownLevel.WARNING:
            risk_multiplier = 0.5  # Reduce risk to 50%
            logger.warning(f"[{self.__class__.__name__}] DD Warning - Risk reduced")
        
        # 2. CHECK REGIME FILTER
        if self.regime_filter is not None:
            regime_state = self.regime_filter.get_current_regime()
            
            if not self._should_trade_in_regime(regime_state.regime):
                logger.debug(
                    f"[{self.__class__.__name__}] Trading disabled in "
                    f"{regime_state.regime.value} regime"
                )
                self.manage_positions()  # Still manage open positions
                return
        
        # 3. MANAGE EXISTING POSITIONS
        self.manage_positions()
        
        # 4. CHECK POSITION LIMIT
        if len(self.positions) >= self.config.max_positions:
            return
        
        # 5. GENERATE SIGNAL
        signal = self.generate_signal()
        
        if signal is None or signal.type == SignalType.NONE:
            return
        
        # 6. EXECUTE SIGNAL
        self._execute_signal(signal, risk_multiplier)
    
    @abstractmethod
    def generate_signal(self) -> Optional[Signal]:
        """
        Generate trading signal based on strategy logic.
        
        MUST BE IMPLEMENTED BY SUBCLASS.
        
        Returns:
            Signal object or None if no signal
        """
        pass
    
    @abstractmethod
    def manage_positions(self):
        """
        Manage existing positions (trailing stops, exits, etc.).
        
        MUST BE IMPLEMENTED BY SUBCLASS.
        """
        pass
    
    def _should_trade_in_regime(self, regime: MarketRegime) -> bool:
        """
        Determine if EA should trade in current regime.
        
        OVERRIDE in subclasses if needed.
        
        Args:
            regime: Current market regime
        
        Returns:
            True if should trade, False otherwise
        """
        # Default: trade in all regimes
        return True
    
    def _execute_signal(self, signal: Signal, risk_multiplier: float = 1.0):
        """
        Execute trading signal with fault-tolerant executor.
        
        Args:
            signal: Trading signal to execute
            risk_multiplier: Risk adjustment factor (0.0-1.0)
        """
        # Log signal
        if self.config.log_signals:
            logger.info(f"[{self.__class__.__name__}] {signal}")
        
        self.signals_history.append(signal)
        self.stats['total_signals'] += 1
        
        # Calculate position size
        lot_size = self._calculate_position_size(signal, risk_multiplier)
        
        if lot_size <= 0:
            logger.warning("Position size = 0, skipping trade")
            return
        
        # Determine order type
        if signal.type == SignalType.BUY:
            order_type = mt5.ORDER_TYPE_BUY
        elif signal.type == SignalType.SELL:
            order_type = mt5.ORDER_TYPE_SELL
        else:
            logger.error(f"Invalid signal type: {signal.type}")
            return
        
        # Execute with retry logic
        result = self.executor.open_position(
            symbol=self.config.symbol,
            order_type=order_type,
            volume=lot_size,
            sl=signal.sl,
            tp=signal.tp,
            comment=f"{self.__class__.__name__}_{signal.comment}",
            magic=self.config.magic_number
        )
        
        if result.success:
            # Store position info
            self.positions[result.ticket] = {
                'signal': signal,
                'lot_size': lot_size,
                'open_time': datetime.now(),
                'open_price': signal.entry_price
            }
            
            self.stats['executed_trades'] += 1
            
            logger.info(
                f"[{self.__class__.__name__}] Position opened: "
                f"#{result.ticket} ({lot_size} lots)"
            )
        else:
            self.stats['failed_trades'] += 1
            logger.error(
                f"[{self.__class__.__name__}] Failed to open position: "
                f"{result.comment}"
            )
    
    def _calculate_position_size(
        self,
        signal: Signal,
        risk_multiplier: float = 1.0
    ) -> float:
        """
        Calculate position size with confidence weighting.
        
        Args:
            signal: Trading signal
            risk_multiplier: Risk adjustment (0.0-1.0)
        
        Returns:
            Lot size (volume)
        """
        # Get account info
        account_info = mt5.account_info()
        if account_info is None:
            logger.error("Failed to get account info")
            return 0.0
        
        balance = account_info.balance
        
        # Calculate base size
        base_size = self.sizer.calculate_size(
            account_balance=balance,
            entry_price=signal.entry_price,
            stop_loss=signal.sl,
            confidence_score=signal.confidence
        )
        
        # Apply risk multiplier (DD state adjustment)
        final_size = base_size['final_size'] * risk_multiplier
        
        # Apply symbol limits
        symbol_info = mt5.symbol_info(self.config.symbol)
        if symbol_info is None:
            logger.error(f"Failed to get symbol info for {self.config.symbol}")
            return 0.0
        
        # Round to symbol's volume step
        volume_step = symbol_info.volume_step
        final_size = round(final_size / volume_step) * volume_step
        
        # Clamp to min/max
        final_size = max(symbol_info.volume_min, final_size)
        final_size = min(symbol_info.volume_max, final_size)
        
        return final_size
    
    def close_position(self, ticket: int, reason: str = "Manual"):
        """
        Close specific position.
        
        Args:
            ticket: Position ticket
            reason: Reason for closure
        """
        if ticket not in self.positions:
            logger.warning(f"Position #{ticket} not found in tracking")
            return
        
        # Get position info
        positions = mt5.positions_get(ticket=ticket)
        if positions is None or len(positions) == 0:
            logger.warning(f"Position #{ticket} not found in MT5")
            del self.positions[ticket]
            return
        
        position = positions[0]
        
        # Close with retry logic
        result = self.executor.close_position(
            ticket=ticket,
            volume=position.volume
        )
        
        if result.success:
            logger.info(
                f"[{self.__class__.__name__}] Position closed: "
                f"#{ticket} (Reason: {reason})"
            )
            
            # Update stats
            if position.profit > 0:
                self.stats['total_profit'] += position.profit
            
            # Remove from tracking
            del self.positions[ticket]
        else:
            logger.error(
                f"[{self.__class__.__name__}] Failed to close #{ticket}: "
                f"{result.comment}"
            )
    
    def close_all_positions(self, reason: str = "Manual"):
        """
        Close all positions tracked by this EA.
        
        Args:
            reason: Reason for closure
        """
        tickets = list(self.positions.keys())
        
        for ticket in tickets:
            self.close_position(ticket, reason)
    
    def get_statistics(self) -> Dict:
        """
        Get EA performance statistics.
        
        Returns:
            Dict with stats
        """
        return {
            'ea_name': self.__class__.__name__,
            'symbol': self.config.symbol,
            'total_signals': self.stats['total_signals'],
            'executed_trades': self.stats['executed_trades'],
            'failed_trades': self.stats['failed_trades'],
            'open_positions': len(self.positions),
            'total_profit': self.stats['total_profit'],
            'execution_stats': self.executor.get_stats()
        }
    
    def print_statistics(self):
        """Print EA statistics to console"""
        stats = self.get_statistics()
        
        print(f"\n{'='*60}")
        print(f"EA STATISTICS: {stats['ea_name']}")
        print(f"{'='*60}")
        print(f"Symbol: {stats['symbol']}")
        print(f"Total Signals: {stats['total_signals']}")
        print(f"Executed Trades: {stats['executed_trades']}")
        print(f"Failed Trades: {stats['failed_trades']}")
        print(f"Open Positions: {stats['open_positions']}")
        print(f"Total Profit: ${stats['total_profit']:.2f}")
        print(f"{'='*60}")
        
        # Execution stats
        exec_stats = stats['execution_stats']
        print("\nEXECUTION STATS:")
        print(f"  Total Attempts: {exec_stats['total_attempts']}")
        print(f"  Success Rate: {exec_stats['success_rate']:.1f}%")
        print(f"  Avg Retries: {exec_stats['avg_retries']:.2f}")
        print(f"{'='*60}\n")


# ============================================================================
# SPECIALIZED BASE CLASSES
# ============================================================================

class TrendFollowingEA(BaseEA):
    """
    Base class for trend-following EAs.
    
    COMPATIBLE REGIMES: TREND
    
    EXAMPLES:
    - SuperTrendRSI
    - ParabolicEMA
    """
    
    def _should_trade_in_regime(self, regime: MarketRegime) -> bool:
        """Only trade in TREND regime"""
        return regime == MarketRegime.TREND
    
    def _check_trend_strength(self, adx: float, threshold: float = 25.0) -> bool:
        """Helper: Check if trend is strong enough"""
        return adx >= threshold


class MeanReversionEA(BaseEA):
    """
    Base class for mean reversion EAs.
    
    COMPATIBLE REGIMES: RANGE
    
    EXAMPLES:
    - BollingerCCI
    - PairArbitrage
    """
    
    def _should_trade_in_regime(self, regime: MarketRegime) -> bool:
        """Only trade in RANGE regime"""
        return regime == MarketRegime.RANGE
    
    def _check_overbought_oversold(
        self,
        rsi: float,
        overbought: float = 70.0,
        oversold: float = 30.0
    ) -> str:
        """Helper: Check if market is overbought/oversold"""
        if rsi >= overbought:
            return "OVERBOUGHT"
        elif rsi <= oversold:
            return "OVERSOLD"
        return "NEUTRAL"


class BreakoutEA(BaseEA):
    """
    Base class for breakout EAs.
    
    COMPATIBLE REGIMES: TREND, VOLATILE
    
    EXAMPLES:
    - KeltnerBreakout
    - ATRBreakout
    """
    
    def _should_trade_in_regime(self, regime: MarketRegime) -> bool:
        """Trade in TREND or VOLATILE regimes"""
        return regime in [MarketRegime.TREND, MarketRegime.VOLATILE]
    
    def _calculate_range(
        self,
        highs: np.ndarray,
        lows: np.ndarray,
        lookback: int
    ) -> Tuple[float, float]:
        """Helper: Calculate range high/low"""
        range_high = np.max(highs[-lookback:])
        range_low = np.min(lows[-lookback:])
        return range_high, range_low


class ScalpingEA(BaseEA):
    """
    Base class for scalping EAs.
    
    COMPATIBLE REGIMES: ALL (with adjustments)
    
    EXAMPLES:
    - EmaScalper
    """
    
    def _should_trade_in_regime(self, regime: MarketRegime) -> bool:
        """Scalp in all regimes (adjust TP/SL per regime)"""
        return True
    
    def _adjust_targets_for_regime(
        self,
        regime: MarketRegime,
        base_sl_pips: float,
        base_tp_pips: float
    ) -> Tuple[float, float]:
        """Helper: Adjust SL/TP based on regime"""
        if regime == MarketRegime.VOLATILE:
            # Wider targets in volatile markets
            return base_sl_pips * 1.5, base_tp_pips * 1.5
        elif regime == MarketRegime.RANGE:
            # Tighter targets in ranging markets
            return base_sl_pips * 0.8, base_tp_pips * 0.8
        else:
            # Normal targets
            return base_sl_pips, base_tp_pips


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)
    
    # Example: Create a simple trend-following EA
    class SimpleEA(TrendFollowingEA):
        def generate_signal(self) -> Optional[Signal]:
            # Placeholder
            return None
        
        def manage_positions(self):
            # Placeholder
            pass
    
    # Initialize
    config = EAConfig(
        symbol="EURUSD",
        timeframe=mt5.TIMEFRAME_M15,
        magic_number=12345
    )
    
    ea = SimpleEA(config)
    
    print(f"EA initialized: {ea.__class__.__name__}")
    print(f"Symbol: {config.symbol}")
    print(f"Timeframe: {config.timeframe}")
