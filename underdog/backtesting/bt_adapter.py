"""
Backtrader Strategy Adapter for UNDERDOG EAs
=============================================

Converts UNDERDOG TA-Lib based EAs to Backtrader strategies.

Architecture:
-------------
1. Base adapter class that wraps UNDERDOG EA logic
2. Specific adapters for each EA type (ATRBreakout, SuperTrendRSI, etc.)
3. Signal conversion: EA.check_signals() → Backtrader orders
4. Risk management: PropFirmRiskManager integration
5. Metrics: Trade tracking for Monte Carlo validation

Usage:
------
    from underdog.backtesting.bt_adapter import ATRBreakoutBT
    
    cerebro = bt.Cerebro()
    cerebro.addstrategy(ATRBreakoutBT, 
                       atr_period=14,
                       atr_multiplier_entry=1.5)
    cerebro.run()

Author: UNDERDOG Development Team
Date: October 2025
"""

import backtrader as bt
import numpy as np
import talib
from typing import Dict, Optional, List
import logging

from underdog.risk.prop_firm_rme import PropFirmRiskManager

logger = logging.getLogger(__name__)


class UnderdogStrategyBase(bt.Strategy):
    """Base class for converting UNDERDOG EAs to Backtrader."""
    
    # Default parameters (override in subclasses)
    params = (
        ('risk_per_trade', 0.02),        # 2% risk per trade
        ('daily_dd_limit', 0.05),        # 5% daily DD
        ('total_dd_limit', 0.10),        # 10% total DD
        ('use_kelly', False),             # Kelly Criterion
        ('kelly_fraction', 0.5),         # Half-Kelly
        ('min_bars', 50),                # Min bars before trading
    )
    
    def __init__(self):
        """Initialize strategy with risk manager."""
        # Initialize risk manager
        self.rme = PropFirmRiskManager(
            daily_dd_limit=self.params.daily_dd_limit,
            total_dd_limit=self.params.total_dd_limit,
            risk_per_trade=self.params.risk_per_trade,
            use_kelly=self.params.use_kelly,
            kelly_fraction=self.params.kelly_fraction
        )
        
        # Track state
        self.order = None
        self.bar_counter = 0
        self.trades = []  # For Monte Carlo validation
        self.daily_start_value = self.broker.getvalue()
        self.high_water_mark = self.broker.getvalue()
        
        # Entry tracking
        self.entry_price = None
        self.entry_size = None
        self.stop_loss = None
        self.take_profit = None
        
    def log(self, txt, dt=None):
        """Log message with timestamp."""
        dt = dt or self.datas[0].datetime.date(0)
        print(f'[{dt}] {txt}')
    
    def notify_order(self, order):
        """Handle order notifications."""
        if order.status in [order.Submitted, order.Accepted]:
            return
        
        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(f'BUY EXECUTED @ {order.executed.price:.5f}, Size: {order.executed.size:.2f}')
                self.entry_price = order.executed.price
                self.entry_size = order.executed.size
            else:
                self.log(f'SELL EXECUTED @ {order.executed.price:.5f}, Size: {order.executed.size:.2f}')
        
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log(f'Order Failed: {order.getstatusname()}')
        
        self.order = None
    
    def notify_trade(self, trade):
        """Handle trade close notifications."""
        if trade.isclosed:
            pnl = trade.pnl
            self.log(f'TRADE CLOSED - P&L: ${pnl:.2f}, Bars: {trade.barlen}')
            
            # Record for risk manager and Monte Carlo
            self.rme.record_trade(pnl)
            self.trades.append({
                'pnl': pnl,
                'bars_held': trade.barlen,
                'size': self.entry_size
            })
            
            # Reset entry tracking
            self.entry_price = None
            self.entry_size = None
            self.stop_loss = None
            self.take_profit = None
    
    def check_daily_drawdown(self) -> bool:
        """Check if daily DD limit breached."""
        current_value = self.broker.getvalue()
        daily_dd = (self.daily_start_value - current_value) / self.daily_start_value
        
        if daily_dd > self.rme.daily_dd_limit:
            self.log(f'⚠️ DAILY DD BREACHED ({daily_dd:.1%}) - PANIC CLOSE')
            if self.position:
                self.close()
            return True
        
        return False
    
    def calculate_position_size(self, signal_price: float, stop_loss: float) -> float:
        """Calculate position size using risk manager."""
        current_value = self.broker.getvalue()
        sl_distance = abs(signal_price - stop_loss)
        
        if sl_distance == 0:
            return 0.01
        
        # Calculate base size from risk amount
        risk_amount = current_value * self.rme.risk_per_trade
        size = risk_amount / sl_distance
        
        # Apply Kelly if enabled
        if self.rme.use_kelly and len(self.trades) >= 10:
            stats = self.rme.get_statistics()
            kelly_fraction = stats['kelly_fraction']
            size *= kelly_fraction
        
        # Clamp to reasonable limits (0.01 - 10 lots)
        size = max(0.01, min(size, 10.0))
        
        return size
    
    def next(self):
        """Override in subclass to implement strategy logic."""
        raise NotImplementedError("Subclass must implement next()")
    
    def get_trades_for_validation(self) -> List[Dict]:
        """Return trades for Monte Carlo validation."""
        return self.trades


class ATRBreakoutBT(UnderdogStrategyBase):
    """
    ATR Breakout Strategy - Backtrader Adapter
    
    Logic:
    ------
    BUY: Candle range > 1.5× ATR + Close > Open + RSI > 55
    SELL: Candle range > 1.5× ATR + Close < Open + RSI < 45
    SL: 1.5× ATR, TP: 2.5× ATR (R:R = 1:1.67)
    """
    
    params = (
        ('atr_period', 14),
        ('atr_multiplier_entry', 1.5),
        ('atr_multiplier_sl', 1.5),
        ('atr_multiplier_tp', 2.5),
        ('rsi_period', 14),
        ('rsi_bullish', 55),
        ('rsi_bearish', 45),
        ('adx_period', 14),
        ('adx_threshold', 20),
    )
    
    def __init__(self):
        super().__init__()
        
        # Indicators (using Backtrader built-ins)
        self.atr = bt.indicators.ATR(self.data, period=self.params.atr_period)
        self.rsi = bt.indicators.RSI(self.data.close, period=self.params.rsi_period)
        self.adx = bt.indicators.ADX(self.data, period=self.params.adx_period)
        
    def next(self):
        """Strategy logic on each bar."""
        self.bar_counter += 1
        
        # Skip initial bars
        if self.bar_counter < self.params.min_bars:
            return
        
        # Check daily DD limit
        if self.check_daily_drawdown():
            return
        
        # Get current bar data
        close = self.data.close[0]
        open_ = self.data.open[0]
        high = self.data.high[0]
        low = self.data.low[0]
        
        candle_range = high - low
        atr_value = self.atr[0]
        rsi_value = self.rsi[0]
        adx_value = self.adx[0]
        
        # Check if already in position
        if self.position:
            # Exit conditions
            # 1. Opposite signal with large range
            opposite_large_range = candle_range > self.params.atr_multiplier_entry * atr_value
            
            if self.position.size > 0:  # Long position
                if close < open_ and opposite_large_range:
                    self.log(f'EXIT LONG - Opposite breakout')
                    self.close()
            else:  # Short position
                if close > open_ and opposite_large_range:
                    self.log(f'EXIT SHORT - Opposite breakout')
                    self.close()
            
            # 2. Weak trend
            if adx_value < 15:
                self.log(f'EXIT - Weak trend (ADX={adx_value:.1f})')
                self.close()
            
            return
        
        # Entry conditions
        if self.order:
            return  # Pending order exists
        
        # BUY Signal
        if (candle_range > self.params.atr_multiplier_entry * atr_value and
            close > open_ and
            rsi_value > self.params.rsi_bullish and
            adx_value > self.params.adx_threshold):
            
            # Calculate position size
            stop_loss = close - (self.params.atr_multiplier_sl * atr_value)
            size = self.calculate_position_size(close, stop_loss)
            
            self.log(f'BUY SIGNAL - Range: {candle_range:.5f}, ATR: {atr_value:.5f}, RSI: {rsi_value:.1f}, Size: {size:.2f}')
            self.order = self.buy(size=size)
            self.stop_loss = stop_loss
            self.take_profit = close + (self.params.atr_multiplier_tp * atr_value)
        
        # SELL Signal
        elif (candle_range > self.params.atr_multiplier_entry * atr_value and
              close < open_ and
              rsi_value < self.params.rsi_bearish and
              adx_value > self.params.adx_threshold):
            
            # Calculate position size
            stop_loss = close + (self.params.atr_multiplier_sl * atr_value)
            size = self.calculate_position_size(close, stop_loss)
            
            self.log(f'SELL SIGNAL - Range: {candle_range:.5f}, ATR: {atr_value:.5f}, RSI: {rsi_value:.1f}, Size: {size:.2f}')
            self.order = self.sell(size=size)
            self.stop_loss = stop_loss
            self.take_profit = close - (self.params.atr_multiplier_tp * atr_value)


class SuperTrendRSIBT(UnderdogStrategyBase):
    """
    SuperTrend + RSI Strategy - Backtrader Adapter
    
    Logic:
    ------
    BUY: Price > SuperTrend + RSI < 35 (oversold bounce)
    SELL: Price < SuperTrend + RSI > 65 (overbought reversal)
    SL: SuperTrend line, TP: 2× ATR
    """
    
    params = (
        ('atr_period', 14),
        ('atr_multiplier', 2.0),
        ('rsi_period', 14),
        ('rsi_overbought', 65),
        ('rsi_oversold', 35),
        ('adx_period', 14),
        ('adx_threshold', 20),
    )
    
    def __init__(self):
        super().__init__()
        
        # Indicators
        self.atr = bt.indicators.ATR(self.data, period=self.params.atr_period)
        self.rsi = bt.indicators.RSI(self.data.close, period=self.params.rsi_period)
        self.adx = bt.indicators.ADX(self.data, period=self.params.adx_period)
        
        # SuperTrend calculation (manual - not in Backtrader by default)
        self.hl2 = (self.data.high + self.data.low) / 2
        
    def calculate_supertrend(self):
        """Calculate SuperTrend indicator."""
        hl2 = (self.data.high[0] + self.data.low[0]) / 2
        atr = self.atr[0]
        
        upper_band = hl2 + (self.params.atr_multiplier * atr)
        lower_band = hl2 - (self.params.atr_multiplier * atr)
        
        # Simplified: Use current close to determine trend
        if self.data.close[0] > upper_band:
            return lower_band, 1  # Uptrend
        elif self.data.close[0] < lower_band:
            return upper_band, -1  # Downtrend
        else:
            return hl2, 0  # Neutral
    
    def next(self):
        """Strategy logic on each bar."""
        self.bar_counter += 1
        
        if self.bar_counter < self.params.min_bars:
            return
        
        if self.check_daily_drawdown():
            return
        
        # Calculate SuperTrend
        st_value, st_direction = self.calculate_supertrend()
        
        rsi_value = self.rsi[0]
        adx_value = self.adx[0]
        close = self.data.close[0]
        
        # Exit logic
        if self.position:
            if self.position.size > 0 and st_direction == -1:
                self.log(f'EXIT LONG - Trend reversal')
                self.close()
            elif self.position.size < 0 and st_direction == 1:
                self.log(f'EXIT SHORT - Trend reversal')
                self.close()
            return
        
        if self.order:
            return
        
        # BUY Signal: Uptrend + Oversold RSI
        if (st_direction == 1 and
            rsi_value < self.params.rsi_oversold and
            adx_value > self.params.adx_threshold):
            
            stop_loss = st_value
            size = self.calculate_position_size(close, stop_loss)
            
            self.log(f'BUY SIGNAL - ST: {st_value:.5f}, RSI: {rsi_value:.1f}, Size: {size:.2f}')
            self.order = self.buy(size=size)
            self.stop_loss = stop_loss
        
        # SELL Signal: Downtrend + Overbought RSI
        elif (st_direction == -1 and
              rsi_value > self.params.rsi_overbought and
              adx_value > self.params.adx_threshold):
            
            stop_loss = st_value
            size = self.calculate_position_size(close, stop_loss)
            
            self.log(f'SELL SIGNAL - ST: {st_value:.5f}, RSI: {rsi_value:.1f}, Size: {size:.2f}')
            self.order = self.sell(size=size)
            self.stop_loss = stop_loss


class BollingerCCIBT(UnderdogStrategyBase):
    """
    Bollinger Bands + CCI Strategy - Backtrader Adapter
    
    Logic:
    ------
    BUY: Price < Lower BB + CCI < -100 (oversold)
    SELL: Price > Upper BB + CCI > +100 (overbought)
    SL: Middle BB, TP: Opposite BB
    """
    
    params = (
        ('bb_period', 20),
        ('bb_stddev', 2.0),
        ('cci_period', 20),
        ('cci_oversold', -100),
        ('cci_overbought', 100),
    )
    
    def __init__(self):
        super().__init__()
        
        # Bollinger Bands
        self.bb = bt.indicators.BollingerBands(
            self.data.close,
            period=self.params.bb_period,
            devfactor=self.params.bb_stddev
        )
        
        # CCI
        self.cci = bt.indicators.CCI(
            self.data,
            period=self.params.cci_period
        )
    
    def next(self):
        """Strategy logic on each bar."""
        self.bar_counter += 1
        
        if self.bar_counter < self.params.min_bars:
            return
        
        if self.check_daily_drawdown():
            return
        
        close = self.data.close[0]
        cci_value = self.cci[0]
        
        bb_top = self.bb.top[0]
        bb_mid = self.bb.mid[0]
        bb_bot = self.bb.bot[0]
        
        # Exit logic: Mean reversion to middle band
        if self.position:
            if self.position.size > 0 and close >= bb_mid:
                self.log(f'EXIT LONG - Mean reversion')
                self.close()
            elif self.position.size < 0 and close <= bb_mid:
                self.log(f'EXIT SHORT - Mean reversion')
                self.close()
            return
        
        if self.order:
            return
        
        # BUY Signal: Below lower BB + Oversold CCI
        if close < bb_bot and cci_value < self.params.cci_oversold:
            stop_loss = bb_bot - (bb_mid - bb_bot)
            size = self.calculate_position_size(close, stop_loss)
            
            self.log(f'BUY SIGNAL - Price: {close:.5f}, BB_Bot: {bb_bot:.5f}, CCI: {cci_value:.1f}')
            self.order = self.buy(size=size)
            self.stop_loss = stop_loss
            self.take_profit = bb_top
        
        # SELL Signal: Above upper BB + Overbought CCI
        elif close > bb_top and cci_value > self.params.cci_overbought:
            stop_loss = bb_top + (bb_top - bb_mid)
            size = self.calculate_position_size(close, stop_loss)
            
            self.log(f'SELL SIGNAL - Price: {close:.5f}, BB_Top: {bb_top:.5f}, CCI: {cci_value:.1f}')
            self.order = self.sell(size=size)
            self.stop_loss = stop_loss
            self.take_profit = bb_bot


# Strategy registry for easy lookup
STRATEGY_REGISTRY = {
    'ATRBreakout': ATRBreakoutBT,
    'SuperTrendRSI': SuperTrendRSIBT,
    'BollingerCCI': BollingerCCIBT,
}


def get_strategy_class(strategy_name: str):
    """Get strategy class by name."""
    if strategy_name not in STRATEGY_REGISTRY:
        raise ValueError(f"Unknown strategy: {strategy_name}. Available: {list(STRATEGY_REGISTRY.keys())}")
    return STRATEGY_REGISTRY[strategy_name]
