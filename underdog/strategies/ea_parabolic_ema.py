"""
FxParabolicEMA - Adaptive Trend Following EA
============================================

STRATEGY: Parabolic SAR + Adaptive EMA (period 12-70 based on ATR/ADX)

ENTRY LOGIC:
- BUY: SAR below price + Price above EMA + ADX > 20
- SELL: SAR above price + Price below EMA + ADX > 20

EMA ADAPTIVE LOGIC (Dual-Regime):
- Trending market (ADX > 25, ATR > avg*1.2) → Fast EMA (period 12-30)
- Ranging market (ADX < 20) → Slow EMA (period 50-70)

EXIT:
- SL: SAR level (with ATR fallback)
- TP: Dynamic (2x SL distance)
- Exit if SAR reverses

TIMEFRAME: M15
PARAMETERS:
- K_EMA_Min: 12, K_EMA_Max: 70
- SAR_Step: 0.015, SAR_Max: 0.15
- ADX_Trend_Threshold: 25
"""

import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Optional
import logging

from underdog.strategies.base_ea import TrendFollowingEA, Signal, SignalType, EAConfig

logger = logging.getLogger(__name__)


@dataclass
class ParabolicEMAConfig(EAConfig):
    """Configuration for Parabolic EMA EA"""
    sar_step: float = 0.015
    sar_max: float = 0.15
    
    ema_min_period: int = 12
    ema_max_period: int = 70
    
    adx_period: int = 14
    adx_trend_threshold: float = 25.0
    
    atr_period: int = 14
    atr_expansion_ratio: float = 1.2


class FxParabolicEMA(TrendFollowingEA):
    """Parabolic SAR + Adaptive EMA trend-following EA"""
    
    def __init__(self, config: ParabolicEMAConfig):
        super().__init__(config)
        self.config: ParabolicEMAConfig = config
        logger.info("[ParabolicEMA] Initialized")
    
    def generate_signal(self) -> Optional[Signal]:
        bars = mt5.copy_rates_from_pos(self.config.symbol, self.config.timeframe, 0, 100)
        if bars is None or len(bars) < 50:
            return None
        
        df = pd.DataFrame(bars)
        df = self._calculate_parabolic_sar(df)
        df = self._calculate_adaptive_ema(df)
        df = self._calculate_adx(df)
        
        current_close = df['close'].iloc[-1]
        sar = df['sar'].iloc[-1]
        ema = df['ema_adaptive'].iloc[-1]
        adx = df['adx'].iloc[-1]
        atr = df['atr'].iloc[-1]
        
        tick = mt5.symbol_info_tick(self.config.symbol)
        if tick is None:
            return None
        
        # BUY Signal: SAR below price + Price above EMA + ADX > 20
        if current_close > sar and current_close > ema and adx > 20:
            sl_distance = max(abs(current_close - sar), atr * 2.0)
            sl = tick.ask - sl_distance
            tp = tick.ask + (sl_distance * 2.0)
            
            return Signal(SignalType.BUY, tick.ask, sl, tp, 1.0, "SAR_UP_EMA_BULL")
        
        # SELL Signal: SAR above price + Price below EMA + ADX > 20
        elif current_close < sar and current_close < ema and adx > 20:
            sl_distance = max(abs(current_close - sar), atr * 2.0)
            sl = tick.bid + sl_distance
            tp = tick.bid - (sl_distance * 2.0)
            
            return Signal(SignalType.SELL, tick.bid, sl, tp, 1.0, "SAR_DOWN_EMA_BEAR")
        
        return None
    
    def manage_positions(self):
        positions = mt5.positions_get(symbol=self.config.symbol, magic=self.config.magic_number)
        if positions is None or len(positions) == 0:
            return
        
        bars = mt5.copy_rates_from_pos(self.config.symbol, self.config.timeframe, 0, 50)
        if bars is None:
            return
        
        df = pd.DataFrame(bars)
        df = self._calculate_parabolic_sar(df)
        
        current_close = df['close'].iloc[-1]
        current_sar = df['sar'].iloc[-1]
        
        for position in positions:
            # Exit if SAR reverses
            if position.type == mt5.ORDER_TYPE_BUY and current_close < current_sar:
                self.close_position(position.ticket, "SAR_REVERSAL")
            elif position.type == mt5.ORDER_TYPE_SELL and current_close > current_sar:
                self.close_position(position.ticket, "SAR_REVERSAL")
    
    def _calculate_parabolic_sar(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate Parabolic SAR"""
        high = df['high'].values
        low = df['low'].values
        
        sar = np.zeros(len(df))
        ep = np.zeros(len(df))
        af = np.zeros(len(df))
        trend = np.zeros(len(df))
        
        # Initialize
        sar[0] = low[0]
        ep[0] = high[0]
        af[0] = self.config.sar_step
        trend[0] = 1  # 1 = up, -1 = down
        
        for i in range(1, len(df)):
            # Update SAR
            sar[i] = sar[i-1] + af[i-1] * (ep[i-1] - sar[i-1])
            
            # Check for trend reversal
            if trend[i-1] == 1:  # Uptrend
                if low[i] < sar[i]:
                    # Reversal to downtrend
                    trend[i] = -1
                    sar[i] = ep[i-1]
                    ep[i] = low[i]
                    af[i] = self.config.sar_step
                else:
                    trend[i] = 1
                    if high[i] > ep[i-1]:
                        ep[i] = high[i]
                        af[i] = min(af[i-1] + self.config.sar_step, self.config.sar_max)
                    else:
                        ep[i] = ep[i-1]
                        af[i] = af[i-1]
            else:  # Downtrend
                if high[i] > sar[i]:
                    # Reversal to uptrend
                    trend[i] = 1
                    sar[i] = ep[i-1]
                    ep[i] = high[i]
                    af[i] = self.config.sar_step
                else:
                    trend[i] = -1
                    if low[i] < ep[i-1]:
                        ep[i] = low[i]
                        af[i] = min(af[i-1] + self.config.sar_step, self.config.sar_max)
                    else:
                        ep[i] = ep[i-1]
                        af[i] = af[i-1]
        
        df['sar'] = sar
        df['sar_trend'] = trend
        return df
    
    def _calculate_adaptive_ema(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate adaptive EMA period based on market regime.
        
        LOGIC:
        - Trending (ADX > 25, ATR > avg*1.2) → Fast EMA (12-30)
        - Ranging (ADX < 20) → Slow EMA (50-70)
        """
        # Calculate ADX and ATR
        df = self._calculate_adx(df)
        
        high = df['high'].values
        low = df['low'].values
        close = df['close'].values
        
        # Calculate ATR
        tr1 = high - low
        tr2 = np.abs(high - np.roll(close, 1))
        tr3 = np.abs(low - np.roll(close, 1))
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        atr = pd.Series(tr).rolling(self.config.atr_period).mean()
        df['atr'] = atr
        
        atr_avg = atr.rolling(50).mean()
        atr_ratio = atr / atr_avg
        
        # Determine EMA period
        adx = df['adx']
        ema_period = pd.Series(index=df.index, dtype=float)
        
        for i in range(len(df)):
            if i < 50:
                ema_period.iloc[i] = self.config.ema_min_period
            else:
                # Trending market → Fast EMA
                if adx.iloc[i] > self.config.adx_trend_threshold and atr_ratio.iloc[i] > self.config.atr_expansion_ratio:
                    ema_period.iloc[i] = self.config.ema_min_period
                # Ranging market → Slow EMA
                elif adx.iloc[i] < 20:
                    ema_period.iloc[i] = self.config.ema_max_period
                # Transition → Medium EMA
                else:
                    ema_period.iloc[i] = (self.config.ema_min_period + self.config.ema_max_period) / 2
        
        # Calculate EMA with adaptive period (use average period for simplicity)
        avg_period = int(ema_period.mean())
        ema = df['close'].ewm(span=avg_period, adjust=False).mean()
        df['ema_adaptive'] = ema
        
        return df
    
    def _calculate_adx(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate ADX"""
        period = self.config.adx_period
        
        high = df['high'].values
        low = df['low'].values
        close = df['close'].values
        
        tr1 = high - low
        tr2 = np.abs(high - np.roll(close, 1))
        tr3 = np.abs(low - np.roll(close, 1))
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        
        up_move = high - np.roll(high, 1)
        down_move = np.roll(low, 1) - low
        
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
        
        atr = pd.Series(tr).ewm(span=period, adjust=False).mean()
        plus_di = 100 * pd.Series(plus_dm).ewm(span=period, adjust=False).mean() / atr
        minus_di = 100 * pd.Series(minus_dm).ewm(span=period, adjust=False).mean() / atr
        
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.ewm(span=period, adjust=False).mean()
        
        df['adx'] = adx
        return df


if __name__ == "__main__":
    import time
    import logging
    logging.basicConfig(level=logging.INFO)
    
    if not mt5.initialize():
        exit()
    
    config = ParabolicEMAConfig(
        symbol="EURUSD",
        timeframe=mt5.TIMEFRAME_M15,
        magic_number=102
    )
    
    ea = FxParabolicEMA(config)
    print("ParabolicEMA EA Running (Ctrl+C to stop)")
    
    try:
        while True:
            ea.on_tick()
            time.sleep(1)
    except KeyboardInterrupt:
        ea.print_statistics()
    finally:
        mt5.shutdown()
