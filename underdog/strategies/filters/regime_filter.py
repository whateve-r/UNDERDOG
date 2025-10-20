"""
Regime Filter Híbrido - ADX + ATR con Hysteresis Logic
======================================================

PROPÓSITO:
----------
Clasificar el régimen de mercado (TREND, RANGE, VOLATILE) para desactivar
EAs inapropiados y mejorar la tasa de win rate general del portfolio.

REGÍMENES:
----------
1. **TREND**: ADX > 25 AND ATR > ATR_avg * 1.2
   - Activar: Trend-following EAs (SuperTrend, Parabolic, Keltner)
   - Desactivar: Mean reversion EAs (PairArbitrage, BollingerCCI)

2. **RANGE**: ADX < 20 AND BB_width < BB_avg
   - Activar: Mean reversion EAs (BollingerCCI, PairArbitrage)
   - Desactivar: Breakout EAs (Keltner, ATRBreakout)

3. **VOLATILE**: ATR > ATR_avg * 1.5 AND ADX < 20
   - Activar: Scalping EAs (EmaScalper) con TP reducidos
   - Precaución: Aumentar slippage tolerance

HYSTERESIS LOGIC:
-----------------
Para evitar "flapping" (cambios rápidos de régimen), usamos diferentes
thresholds para ENTRY vs EXIT:

- TREND Entry: ADX > 25
- TREND Exit: ADX < 20  (5-point buffer)

- RANGE Entry: ADX < 20
- RANGE Exit: ADX > 25  (5-point buffer)

Ejemplo:
  Estado actual: TREND
  ADX baja a 22 → Permanece TREND (no cruza threshold exit de 20)
  ADX baja a 18 → Cambia a RANGE

RATIONALE (Scientific Backing):
--------------------------------
- ADX > 25: Wilder's threshold para trending markets
- ATR ratio: Volatility expansion indica breakout potencial
- Hysteresis: Evita whipsaws (López de Prado, "Advances in Financial ML")
- Bollinger Band width: Measure of market compression

INTEGRATION:
------------
```python
regime_filter = RegimeFilter(symbol="EURUSD", timeframe=mt5.TIMEFRAME_M15)

def on_tick():
    regime = regime_filter.get_current_regime()
    
    if regime == "TREND":
        # Ejecutar Trend EAs
        supertrend_ea.on_tick()
        parabolic_ea.on_tick()
    elif regime == "RANGE":
        # Ejecutar Mean Reversion EAs
        bollinger_cci_ea.on_tick()
    elif regime == "VOLATILE":
        # Solo scalping con TP reducidos
        ema_scalper_ea.on_tick()
```

AUTHOR: UNDERDOG Development Team
DATE: October 2025
"""

import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple
from enum import Enum
from dataclasses import dataclass
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class MarketRegime(Enum):
    """Market regime classification"""
    TREND = "TREND"          # ADX > 25, ATR high
    RANGE = "RANGE"          # ADX < 20, BB narrow
    VOLATILE = "VOLATILE"    # ATR high, ADX low (choppy)
    UNKNOWN = "UNKNOWN"      # Not enough data


@dataclass
class RegimeConfig:
    """Configuration for regime detection"""
    # ADX thresholds
    adx_trend_entry: float = 25.0    # ADX > 25 → TREND
    adx_trend_exit: float = 20.0     # ADX < 20 → Exit TREND (hysteresis)
    adx_range_entry: float = 20.0    # ADX < 20 → RANGE
    adx_range_exit: float = 25.0     # ADX > 25 → Exit RANGE
    
    # ATR thresholds (ratio vs moving average)
    atr_expansion_ratio: float = 1.2  # ATR > ATR_avg * 1.2 → Expansion
    atr_volatile_ratio: float = 1.5   # ATR > ATR_avg * 1.5 → Volatile
    
    # Bollinger Band width (for RANGE detection)
    bb_narrow_ratio: float = 0.8      # BB_width < BB_avg * 0.8 → Narrow
    
    # Indicator periods
    adx_period: int = 14
    atr_period: int = 14
    bb_period: int = 20
    bb_deviation: float = 2.0
    
    # Lookback for moving averages
    atr_ma_period: int = 50
    bb_width_ma_period: int = 50
    
    # Minimum bars for regime detection
    min_bars: int = 100


@dataclass
class RegimeState:
    """Current regime state"""
    regime: MarketRegime
    adx: float
    atr: float
    atr_avg: float
    bb_width: float
    bb_width_avg: float
    confidence: float  # 0.0-1.0
    timestamp: datetime
    
    def __str__(self):
        return (
            f"Regime: {self.regime.value} (confidence={self.confidence:.2f})\n"
            f"  ADX: {self.adx:.2f}\n"
            f"  ATR: {self.atr:.5f} (avg={self.atr_avg:.5f}, ratio={self.atr/self.atr_avg:.2f})\n"
            f"  BB Width: {self.bb_width:.5f} (avg={self.bb_width_avg:.5f})"
        )


class RegimeFilter:
    """
    Market regime classifier with hysteresis logic.
    
    USAGE:
    ------
    ```python
    # Initialize
    filter = RegimeFilter("EURUSD", mt5.TIMEFRAME_M15)
    
    # Get current regime
    regime_state = filter.get_current_regime()
    
    # Check if EA should trade
    if filter.should_ea_trade("TREND_FOLLOWING", regime_state.regime):
        # Execute EA logic
        pass
    ```
    """
    
    def __init__(
        self,
        symbol: str,
        timeframe: int = mt5.TIMEFRAME_M15,
        config: Optional[RegimeConfig] = None
    ):
        """
        Initialize regime filter.
        
        Args:
            symbol: Trading symbol (EURUSD, GBPUSD, etc.)
            timeframe: MT5 timeframe constant
            config: Optional custom configuration
        """
        self.symbol = symbol
        self.timeframe = timeframe
        self.config = config or RegimeConfig()
        
        # Current state
        self.current_regime = MarketRegime.UNKNOWN
        self.regime_history = []  # For debugging
        
        # Precompute compatibility matrix
        self._init_ea_compatibility()
        
        logger.info(
            f"RegimeFilter initialized for {symbol} "
            f"(TF={self._timeframe_str(timeframe)})"
        )
    
    def _timeframe_str(self, tf: int) -> str:
        """Convert MT5 timeframe to string"""
        mapping = {
            mt5.TIMEFRAME_M5: "M5",
            mt5.TIMEFRAME_M15: "M15",
            mt5.TIMEFRAME_M30: "M30",
            mt5.TIMEFRAME_H1: "H1",
            mt5.TIMEFRAME_H4: "H4",
            mt5.TIMEFRAME_D1: "D1"
        }
        return mapping.get(tf, f"TF_{tf}")
    
    def _init_ea_compatibility(self):
        """
        Define which EAs should trade in which regimes.
        
        EA TYPE → ALLOWED REGIMES
        """
        self.ea_compatibility = {
            # Trend-following EAs
            "TREND_FOLLOWING": {
                MarketRegime.TREND: True,
                MarketRegime.RANGE: False,
                MarketRegime.VOLATILE: False
            },
            
            # Mean reversion EAs
            "MEAN_REVERSION": {
                MarketRegime.TREND: False,  # Dangerous in trends
                MarketRegime.RANGE: True,
                MarketRegime.VOLATILE: False
            },
            
            # Breakout EAs
            "BREAKOUT": {
                MarketRegime.TREND: True,   # Good for continuation
                MarketRegime.RANGE: False,  # False breakouts
                MarketRegime.VOLATILE: True  # ATR expansion
            },
            
            # Scalping EAs
            "SCALPING": {
                MarketRegime.TREND: True,   # Can scalp trends
                MarketRegime.RANGE: True,   # Can scalp ranges
                MarketRegime.VOLATILE: True  # Fast profits (wide TP)
            }
        }
    
    def get_current_regime(self) -> RegimeState:
        """
        Get current market regime based on latest data.
        
        Returns:
            RegimeState with regime classification and indicators
        """
        # Get historical bars
        bars = mt5.copy_rates_from_pos(
            self.symbol,
            self.timeframe,
            0,  # Start from current bar
            self.config.min_bars
        )
        
        if bars is None or len(bars) < self.config.min_bars:
            logger.warning(f"Not enough data for {self.symbol}")
            return RegimeState(
                regime=MarketRegime.UNKNOWN,
                adx=0, atr=0, atr_avg=0, bb_width=0, bb_width_avg=0,
                confidence=0.0,
                timestamp=datetime.now()
            )
        
        df = pd.DataFrame(bars)
        
        # Calculate indicators
        adx = self._calculate_adx(df, self.config.adx_period)
        atr = self._calculate_atr(df, self.config.atr_period)
        atr_avg = df['atr'].rolling(self.config.atr_ma_period).mean().iloc[-1]
        bb_width, bb_width_avg = self._calculate_bb_width(df)
        
        # Classify regime with hysteresis
        new_regime = self._classify_regime(adx, atr, atr_avg, bb_width, bb_width_avg)
        
        # Calculate confidence (how strongly does data support this regime)
        confidence = self._calculate_confidence(adx, atr, atr_avg, bb_width, bb_width_avg, new_regime)
        
        # Update state
        self.current_regime = new_regime
        
        # Create state object
        state = RegimeState(
            regime=new_regime,
            adx=adx,
            atr=atr,
            atr_avg=atr_avg,
            bb_width=bb_width,
            bb_width_avg=bb_width_avg,
            confidence=confidence,
            timestamp=datetime.now()
        )
        
        # Log regime changes
        if len(self.regime_history) == 0 or self.regime_history[-1].regime != new_regime:
            logger.info(f"Regime changed to {new_regime.value} (confidence={confidence:.2f})")
            self.regime_history.append(state)
        
        return state
    
    def _classify_regime(
        self,
        adx: float,
        atr: float,
        atr_avg: float,
        bb_width: float,
        bb_width_avg: float
    ) -> MarketRegime:
        """
        Classify market regime with hysteresis logic.
        
        HYSTERESIS RULES:
        -----------------
        - If currently TREND: Need ADX < 20 to exit (not just < 25)
        - If currently RANGE: Need ADX > 25 to exit (not just > 20)
        - This prevents rapid flipping between regimes
        """
        atr_ratio = atr / atr_avg if atr_avg > 0 else 1.0
        bb_ratio = bb_width / bb_width_avg if bb_width_avg > 0 else 1.0
        
        # TREND detection
        if adx >= self.config.adx_trend_entry and atr_ratio >= self.config.atr_expansion_ratio:
            return MarketRegime.TREND
        
        # RANGE detection
        if adx <= self.config.adx_range_entry and bb_ratio <= self.config.bb_narrow_ratio:
            return MarketRegime.RANGE
        
        # VOLATILE detection (high ATR, low ADX = choppy)
        if atr_ratio >= self.config.atr_volatile_ratio and adx < self.config.adx_range_entry:
            return MarketRegime.VOLATILE
        
        # HYSTERESIS: Stay in current regime if not clearly exited
        if self.current_regime == MarketRegime.TREND:
            if adx >= self.config.adx_trend_exit:  # Still above exit threshold
                return MarketRegime.TREND
        
        if self.current_regime == MarketRegime.RANGE:
            if adx <= self.config.adx_range_exit:  # Still below exit threshold
                return MarketRegime.RANGE
        
        # Default: RANGE (safest assumption)
        return MarketRegime.RANGE
    
    def _calculate_confidence(
        self,
        adx: float,
        atr: float,
        atr_avg: float,
        bb_width: float,
        bb_width_avg: float,
        regime: MarketRegime
    ) -> float:
        """
        Calculate confidence score (0.0-1.0) for regime classification.
        
        Higher confidence = Stronger signals supporting this regime.
        """
        if regime == MarketRegime.UNKNOWN:
            return 0.0
        
        atr_ratio = atr / atr_avg if atr_avg > 0 else 1.0
        bb_ratio = bb_width / bb_width_avg if bb_width_avg > 0 else 1.0
        
        if regime == MarketRegime.TREND:
            # Confidence based on ADX strength and ATR expansion
            adx_score = min((adx - 20) / 20, 1.0)  # Normalize (20-40 → 0.0-1.0)
            atr_score = min((atr_ratio - 1.0) / 0.5, 1.0)  # (1.0-1.5 → 0.0-1.0)
            return (adx_score + atr_score) / 2.0
        
        elif regime == MarketRegime.RANGE:
            # Confidence based on low ADX and narrow BB
            adx_score = max(1.0 - (adx / 20), 0.0)  # Lower ADX = higher score
            bb_score = max(1.0 - bb_ratio, 0.0)     # Narrower BB = higher score
            return (adx_score + bb_score) / 2.0
        
        elif regime == MarketRegime.VOLATILE:
            # Confidence based on high ATR and low ADX
            atr_score = min((atr_ratio - 1.0) / 1.0, 1.0)
            adx_score = max(1.0 - (adx / 25), 0.0)
            return (atr_score + adx_score) / 2.0
        
        return 0.5  # Default medium confidence
    
    def should_ea_trade(self, ea_type: str, regime: MarketRegime) -> bool:
        """
        Determine if EA should trade in current regime.
        
        Args:
            ea_type: EA type (TREND_FOLLOWING, MEAN_REVERSION, BREAKOUT, SCALPING)
            regime: Current market regime
        
        Returns:
            True if EA should trade, False otherwise
        
        Example:
            if filter.should_ea_trade("TREND_FOLLOWING", regime_state.regime):
                supertrend_ea.on_tick()
        """
        if ea_type not in self.ea_compatibility:
            logger.warning(f"Unknown EA type: {ea_type}")
            return True  # Default: allow trading
        
        allowed = self.ea_compatibility[ea_type].get(regime, False)
        
        if not allowed:
            logger.debug(f"{ea_type} EA disabled in {regime.value} regime")
        
        return allowed
    
    # ========================================================================
    # INDICATOR CALCULATIONS
    # ========================================================================
    
    def _calculate_adx(self, df: pd.DataFrame, period: int) -> float:
        """
        Calculate ADX (Average Directional Index).
        
        ADX measures trend strength (0-100):
        - < 20: Weak trend (ranging)
        - 20-25: Moderate trend
        - > 25: Strong trend
        """
        high = df['high'].values
        low = df['low'].values
        close = df['close'].values
        
        # True Range
        tr1 = high - low
        tr2 = np.abs(high - np.roll(close, 1))
        tr3 = np.abs(low - np.roll(close, 1))
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        
        # Directional Movement
        up_move = high - np.roll(high, 1)
        down_move = np.roll(low, 1) - low
        
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
        
        # Smooth with Wilder's MA
        atr = pd.Series(tr).ewm(span=period, adjust=False).mean()
        plus_di = 100 * pd.Series(plus_dm).ewm(span=period, adjust=False).mean() / atr
        minus_di = 100 * pd.Series(minus_dm).ewm(span=period, adjust=False).mean() / atr
        
        # ADX
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.ewm(span=period, adjust=False).mean()
        
        return adx.iloc[-1]
    
    def _calculate_atr(self, df: pd.DataFrame, period: int) -> float:
        """Calculate ATR (Average True Range)"""
        high = df['high'].values
        low = df['low'].values
        close = df['close'].values
        
        tr1 = high - low
        tr2 = np.abs(high - np.roll(close, 1))
        tr3 = np.abs(low - np.roll(close, 1))
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        
        atr = pd.Series(tr).rolling(period).mean()
        df['atr'] = atr
        
        return atr.iloc[-1]
    
    def _calculate_bb_width(self, df: pd.DataFrame) -> Tuple[float, float]:
        """
        Calculate Bollinger Band width and its moving average.
        
        BB Width = (Upper Band - Lower Band) / Middle Band
        """
        period = self.config.bb_period
        deviation = self.config.bb_deviation
        
        close = df['close']
        
        # Bollinger Bands
        middle = close.rolling(period).mean()
        std = close.rolling(period).std()
        upper = middle + (std * deviation)
        lower = middle - (std * deviation)
        
        # Width (normalized)
        bb_width = (upper - lower) / middle
        bb_width_avg = bb_width.rolling(self.config.bb_width_ma_period).mean()
        
        return bb_width.iloc[-1], bb_width_avg.iloc[-1]
    
    def get_regime_statistics(self) -> Dict:
        """
        Get statistics about regime history.
        
        Returns:
            Dict with regime counts and durations
        """
        if len(self.regime_history) == 0:
            return {}
        
        regime_counts = {}
        for state in self.regime_history:
            regime = state.regime.value
            regime_counts[regime] = regime_counts.get(regime, 0) + 1
        
        total = len(self.regime_history)
        regime_pcts = {k: (v / total) * 100 for k, v in regime_counts.items()}
        
        return {
            'total_samples': total,
            'regime_counts': regime_counts,
            'regime_percentages': regime_pcts,
            'current_regime': self.current_regime.value
        }


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)
    
    # Initialize MT5
    if not mt5.initialize():
        print("MT5 initialization failed")
        exit()
    
    # Create regime filter
    filter = RegimeFilter(
        symbol="EURUSD",
        timeframe=mt5.TIMEFRAME_M15
    )
    
    # Get current regime
    regime_state = filter.get_current_regime()
    print("\n" + "="*60)
    print("CURRENT MARKET REGIME")
    print("="*60)
    print(regime_state)
    print("="*60)
    
    # Check EA compatibility
    print("\nEA COMPATIBILITY:")
    print(f"  Trend Following: {filter.should_ea_trade('TREND_FOLLOWING', regime_state.regime)}")
    print(f"  Mean Reversion: {filter.should_ea_trade('MEAN_REVERSION', regime_state.regime)}")
    print(f"  Breakout: {filter.should_ea_trade('BREAKOUT', regime_state.regime)}")
    print(f"  Scalping: {filter.should_ea_trade('SCALPING', regime_state.regime)}")
    
    # Shutdown MT5
    mt5.shutdown()
