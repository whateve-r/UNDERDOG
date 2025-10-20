"""
FxSuperTrendRSI - Trend Following EA with RSI Filter
====================================================

ESTRATEGIA:
-----------
SuperTrend + RSI momentum filter para entradas de alta calidad.

LÓGICA DE ENTRADA:
------------------
BUY:
  - SuperTrend direction = UP (close > ST_Line)
  - RSI > 65 (momentum alcista confirmado)
  - Confidence = 1.0 (señal clara)

SELL:
  - SuperTrend direction = DOWN (close < ST_Line)
  - RSI < 35 (momentum bajista confirmado)
  - Confidence = 1.0 (señal clara)

GESTIÓN DE SALIDA:
------------------
- SL Inicial: max(ATR * multiplier, ST_Line)
- Trailing Stop: Activado después de +15 pips
- Exit por ADX: Cerrar si ADX < 15 (pérdida de tendencia)
- Exit por SuperTrend: Cerrar si ST cambia de dirección

PARÁMETROS CLAVE:
-----------------
- SuperTrend_Multiplier: 3.5-4.0 (vs 3.0 común)
- SuperTrend_Period: 10
- RSI_Period: 14
- RSI_Buy_Level: 65 (vs 60 más agresivo)
- RSI_Sell_Level: 35 (vs 40 más agresivo)
- TrailingStart_Pips: 15
- ADX_Exit_Threshold: 15

TIMEFRAME: M5
R:R: 1:1.5 (dinámico según ATR)

MEJORAS VS VERSIÓN BÁSICA:
---------------------------
1. RSI threshold más estricto (65/35 vs 60/40)
2. SL dinámico basado en SuperTrend Line
3. Trailing stop con hysteresis
4. Exit por debilitamiento de tendencia (ADX < 15)

AUTHOR: UNDERDOG Development Team
DATE: October 2025
"""

import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Optional, Dict
import logging

from underdog.strategies.base_ea import (
    TrendFollowingEA, Signal, SignalType, EAConfig
)

logger = logging.getLogger(__name__)


@dataclass
class SuperTrendRSIConfig(EAConfig):
    """Configuration for SuperTrend RSI EA"""
    # SuperTrend parameters
    supertrend_period: int = 10
    supertrend_multiplier: float = 3.5
    
    # RSI parameters
    rsi_period: int = 14
    rsi_buy_level: float = 65.0
    rsi_sell_level: float = 35.0
    
    # Exit parameters
    trailing_start_pips: float = 15.0
    adx_exit_threshold: float = 15.0
    adx_period: int = 14
    
    # Risk:Reward
    tp_multiplier: float = 1.5  # TP = SL * 1.5


class FxSuperTrendRSI(TrendFollowingEA):
    """
    SuperTrend + RSI trend-following EA.
    
    USAGE:
    ------
    ```python
    config = SuperTrendRSIConfig(
        symbol="EURUSD",
        timeframe=mt5.TIMEFRAME_M5,
        magic_number=101
    )
    
    ea = FxSuperTrendRSI(config)
    
    while True:
        ea.on_tick()
        time.sleep(1)
    ```
    """
    
    def __init__(self, config: SuperTrendRSIConfig):
        """Initialize SuperTrend RSI EA"""
        super().__init__(config)
        self.config: SuperTrendRSIConfig = config
        
        # Trailing stop tracking
        self.trailing_stops: Dict[int, float] = {}  # ticket → trailing SL
        
        logger.info(
            f"[SuperTrendRSI] Initialized: ST_mult={config.supertrend_multiplier}, "
            f"RSI={config.rsi_buy_level}/{config.rsi_sell_level}"
        )
    
    def generate_signal(self) -> Optional[Signal]:
        """
        Generate SuperTrend + RSI signal.
        
        Returns:
            Signal object or None
        """
        # Get market data
        bars = mt5.copy_rates_from_pos(
            self.config.symbol,
            self.config.timeframe,
            0,  # Start from current bar
            100  # Need enough bars for indicators
        )
        
        if bars is None or len(bars) < 50:
            return None
        
        df = pd.DataFrame(bars)
        
        # Calculate indicators
        df = self._calculate_supertrend(df)
        df = self._calculate_rsi(df)
        df = self._calculate_adx(df)
        
        # Current values
        current_close = df['close'].iloc[-1]
        st_line = df['supertrend_line'].iloc[-1]
        st_direction = df['supertrend_direction'].iloc[-1]
        rsi = df['rsi'].iloc[-1]
        adx = df['adx'].iloc[-1]
        atr = df['atr'].iloc[-1]
        
        # Get current price
        tick = mt5.symbol_info_tick(self.config.symbol)
        if tick is None:
            return None
        
        # Check for BUY signal
        if st_direction == 1 and rsi >= self.config.rsi_buy_level:
            # Calculate SL/TP
            sl_distance = max(
                atr * self.config.supertrend_multiplier,
                abs(current_close - st_line)
            )
            
            sl = tick.ask - sl_distance
            tp = tick.ask + (sl_distance * self.config.tp_multiplier)
            
            logger.info(
                f"[SuperTrendRSI] BUY Signal: RSI={rsi:.1f}, ADX={adx:.1f}, "
                f"ST_Line={st_line:.5f}"
            )
            
            return Signal(
                type=SignalType.BUY,
                entry_price=tick.ask,
                sl=sl,
                tp=tp,
                confidence=1.0,
                comment="ST_UP_RSI_BULL"
            )
        
        # Check for SELL signal
        elif st_direction == -1 and rsi <= self.config.rsi_sell_level:
            # Calculate SL/TP
            sl_distance = max(
                atr * self.config.supertrend_multiplier,
                abs(current_close - st_line)
            )
            
            sl = tick.bid + sl_distance
            tp = tick.bid - (sl_distance * self.config.tp_multiplier)
            
            logger.info(
                f"[SuperTrendRSI] SELL Signal: RSI={rsi:.1f}, ADX={adx:.1f}, "
                f"ST_Line={st_line:.5f}"
            )
            
            return Signal(
                type=SignalType.SELL,
                entry_price=tick.bid,
                sl=sl,
                tp=tp,
                confidence=1.0,
                comment="ST_DOWN_RSI_BEAR"
            )
        
        return None
    
    def manage_positions(self):
        """
        Manage open positions:
        1. Trailing stop (after +15 pips profit)
        2. Exit if ADX < 15 (trend weakening)
        3. Exit if SuperTrend reverses
        """
        # Get positions for this EA
        positions = mt5.positions_get(
            symbol=self.config.symbol,
            magic=self.config.magic_number
        )
        
        if positions is None or len(positions) == 0:
            return
        
        # Get current market data
        bars = mt5.copy_rates_from_pos(
            self.config.symbol,
            self.config.timeframe,
            0,
            50
        )
        
        if bars is None:
            return
        
        df = pd.DataFrame(bars)
        df = self._calculate_supertrend(df)
        df = self._calculate_adx(df)
        
        current_adx = df['adx'].iloc[-1]
        current_st_direction = df['supertrend_direction'].iloc[-1]
        
        tick = mt5.symbol_info_tick(self.config.symbol)
        if tick is None:
            return
        
        for position in positions:
            ticket = position.ticket
            
            # 1. Check ADX exit (trend weakening)
            if current_adx < self.config.adx_exit_threshold:
                logger.info(
                    f"[SuperTrendRSI] Closing #{ticket}: ADX too low ({current_adx:.1f})"
                )
                self.close_position(ticket, "ADX_EXIT")
                continue
            
            # 2. Check SuperTrend reversal
            if position.type == mt5.ORDER_TYPE_BUY and current_st_direction == -1:
                logger.info(
                    f"[SuperTrendRSI] Closing #{ticket}: SuperTrend reversed DOWN"
                )
                self.close_position(ticket, "ST_REVERSAL")
                continue
            
            elif position.type == mt5.ORDER_TYPE_SELL and current_st_direction == 1:
                logger.info(
                    f"[SuperTrendRSI] Closing #{ticket}: SuperTrend reversed UP"
                )
                self.close_position(ticket, "ST_REVERSAL")
                continue
            
            # 3. Trailing stop
            self._apply_trailing_stop(position, tick)
    
    def _apply_trailing_stop(self, position, tick):
        """
        Apply trailing stop after profit threshold reached.
        
        Args:
            position: MT5 position object
            tick: Current tick
        """
        ticket = position.ticket
        
        # Calculate current profit in pips
        point = mt5.symbol_info(self.config.symbol).point
        
        if position.type == mt5.ORDER_TYPE_BUY:
            profit_pips = (tick.bid - position.price_open) / point
        else:
            profit_pips = (position.price_open - tick.ask) / point
        
        # Check if profit exceeds trailing start threshold
        if profit_pips < self.config.trailing_start_pips:
            return
        
        # Calculate trailing SL
        if position.type == mt5.ORDER_TYPE_BUY:
            # Move SL up
            new_sl = tick.bid - (self.config.trailing_start_pips * point)
            
            # Only move SL up (never down)
            if new_sl > position.sl:
                # Check if SL needs updating
                if ticket not in self.trailing_stops or new_sl > self.trailing_stops[ticket]:
                    result = self.executor.modify_position(
                        ticket=ticket,
                        sl=new_sl,
                        tp=position.tp
                    )
                    
                    if result.success:
                        self.trailing_stops[ticket] = new_sl
                        logger.info(
                            f"[SuperTrendRSI] Trailing SL updated for #{ticket}: "
                            f"{new_sl:.5f} (+{profit_pips:.1f} pips profit)"
                        )
        
        else:  # SELL position
            # Move SL down
            new_sl = tick.ask + (self.config.trailing_start_pips * point)
            
            # Only move SL down (never up)
            if new_sl < position.sl:
                if ticket not in self.trailing_stops or new_sl < self.trailing_stops[ticket]:
                    result = self.executor.modify_position(
                        ticket=ticket,
                        sl=new_sl,
                        tp=position.tp
                    )
                    
                    if result.success:
                        self.trailing_stops[ticket] = new_sl
                        logger.info(
                            f"[SuperTrendRSI] Trailing SL updated for #{ticket}: "
                            f"{new_sl:.5f} (+{profit_pips:.1f} pips profit)"
                        )
    
    # ========================================================================
    # INDICATOR CALCULATIONS
    # ========================================================================
    
    def _calculate_supertrend(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate SuperTrend indicator.
        
        SuperTrend Logic:
        -----------------
        1. Basic Bands = (High + Low) / 2 ± (ATR * multiplier)
        2. Final Bands = Track with trend direction
        3. Direction = 1 (UP) if close > upper_band, -1 (DOWN) if close < lower_band
        """
        period = self.config.supertrend_period
        multiplier = self.config.supertrend_multiplier
        
        # Calculate ATR
        high = df['high'].values
        low = df['low'].values
        close = df['close'].values
        
        tr1 = high - low
        tr2 = np.abs(high - np.roll(close, 1))
        tr3 = np.abs(low - np.roll(close, 1))
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        
        atr = pd.Series(tr).rolling(period).mean()
        df['atr'] = atr
        
        # Basic bands
        hl2 = (df['high'] + df['low']) / 2
        upper_band = hl2 + (multiplier * atr)
        lower_band = hl2 - (multiplier * atr)
        
        # Final bands (track with trend)
        supertrend_line = pd.Series(index=df.index, dtype=float)
        direction = pd.Series(index=df.index, dtype=int)
        
        supertrend_line.iloc[0] = upper_band.iloc[0]
        direction.iloc[0] = 1
        
        for i in range(1, len(df)):
            # Update final upper band
            if close[i] > supertrend_line.iloc[i-1]:
                supertrend_line.iloc[i] = lower_band.iloc[i]
                direction.iloc[i] = 1  # UP
            else:
                supertrend_line.iloc[i] = upper_band.iloc[i]
                direction.iloc[i] = -1  # DOWN
        
        df['supertrend_line'] = supertrend_line
        df['supertrend_direction'] = direction
        
        return df
    
    def _calculate_rsi(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate RSI (Relative Strength Index)"""
        period = self.config.rsi_period
        
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        df['rsi'] = rsi
        return df
    
    def _calculate_adx(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate ADX (Average Directional Index)"""
        period = self.config.adx_period
        
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
        
        # Smooth with EMA
        atr = pd.Series(tr).ewm(span=period, adjust=False).mean()
        plus_di = 100 * pd.Series(plus_dm).ewm(span=period, adjust=False).mean() / atr
        minus_di = 100 * pd.Series(minus_dm).ewm(span=period, adjust=False).mean() / atr
        
        # ADX
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.ewm(span=period, adjust=False).mean()
        
        df['adx'] = adx
        return df


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    import time
    import logging
    logging.basicConfig(level=logging.INFO)
    
    # Initialize MT5
    if not mt5.initialize():
        print("MT5 initialization failed")
        exit()
    
    # Create EA
    config = SuperTrendRSIConfig(
        symbol="EURUSD",
        timeframe=mt5.TIMEFRAME_M5,
        magic_number=101,
        risk_per_trade=1.5
    )
    
    ea = FxSuperTrendRSI(config)
    
    print("="*60)
    print("SuperTrendRSI EA Running")
    print("="*60)
    print(f"Symbol: {config.symbol}")
    print(f"Timeframe: M5")
    print(f"Magic: {config.magic_number}")
    print("="*60)
    print("Press Ctrl+C to stop\n")
    
    try:
        while True:
            ea.on_tick()
            time.sleep(1)  # Check every second
    
    except KeyboardInterrupt:
        print("\nStopping EA...")
        ea.print_statistics()
    
    finally:
        mt5.shutdown()
