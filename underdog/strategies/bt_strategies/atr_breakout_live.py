"""
Example Live Strategy - ATR Breakout with MT5 Integration

This is an example of how to adapt a Backtrader strategy for live trading with MT5.
The strategy works in both backtest and live mode automatically.

Strategy Logic:
- BUY when price breaks above 20-period SMA + ATR threshold
- SELL when price breaks below 20-period SMA - ATR threshold
- Use ATR for dynamic SL/TP

Usage:
    # Backtest mode
    cerebro = bt.Cerebro()
    cerebro.addstrategy(ATRBreakoutLive)
    cerebro.run()
    
    # Live mode
    executor = MT5Executor(account=..., password=..., server=...)
    bridge = BacktraderMT5Bridge(executor=executor)
    
    cerebro = bt.Cerebro()
    cerebro.addstrategy(ATRBreakoutLive, mt5_bridge=bridge, symbol="EURUSD")
    cerebro.run()

Author: Underdog Trading System
"""

import backtrader as bt
from underdog.bridges.bt_to_mt5 import LiveStrategy


class ATRBreakoutLive(LiveStrategy):
    """
    ATR Breakout strategy adapted for live trading
    
    Inherits from LiveStrategy to automatically handle MT5 bridge integration
    """
    
    params = (
        ('sma_period', 20),
        ('atr_period', 14),
        ('atr_multiplier', 2.0),
        ('symbol', 'EURUSD'),
        ('volume', 0.1),
        ('atr_sl_multiplier', 2.0),  # SL = ATR * 2
        ('atr_tp_multiplier', 4.0),  # TP = ATR * 4
        ('mt5_bridge', None),
    )
    
    def __init__(self):
        super().__init__()
        
        # Indicators
        self.sma = bt.indicators.SMA(self.data.close, period=self.params.sma_period)
        self.atr = bt.indicators.ATR(self.data, period=self.params.atr_period)
        
        # Track position
        self.order = None
        self.in_position = False
    
    def next(self):
        """
        Strategy logic - executes on each bar
        
        Works in both backtest and live mode automatically
        """
        # Skip if we have a pending order
        if self.order:
            return
        
        # Get current values
        close = self.data.close[0]
        sma = self.sma[0]
        atr = self.atr[0]
        
        # Calculate dynamic SL/TP based on ATR
        sl_pips = self._price_to_pips(atr * self.params.atr_sl_multiplier)
        tp_pips = self._price_to_pips(atr * self.params.atr_tp_multiplier)
        
        # BUY signal: price breaks above SMA + ATR threshold
        if not self.in_position and close > sma + (atr * self.params.atr_multiplier):
            self.log(f"BUY SIGNAL - Close: {close:.5f}, SMA: {sma:.5f}, ATR: {atr:.5f}")
            
            # Execute BUY (works in both backtest and live)
            self.execute_buy(
                symbol=self.params.symbol,
                volume=self.params.volume,
                sl_pips=sl_pips,
                tp_pips=tp_pips
            )
            self.in_position = True
        
        # SELL signal: price breaks below SMA - ATR threshold
        elif not self.in_position and close < sma - (atr * self.params.atr_multiplier):
            self.log(f"SELL SIGNAL - Close: {close:.5f}, SMA: {sma:.5f}, ATR: {atr:.5f}")
            
            # Execute SELL (works in both backtest and live)
            self.execute_sell(
                symbol=self.params.symbol,
                volume=self.params.volume,
                sl_pips=sl_pips,
                tp_pips=tp_pips
            )
            self.in_position = True
    
    def notify_order(self, order):
        """Handle order notifications (backtest mode only)"""
        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(f"BUY EXECUTED - Price: {order.executed.price:.5f}")
            elif order.issell():
                self.log(f"SELL EXECUTED - Price: {order.executed.price:.5f}")
            self.order = None
        
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log(f"ORDER REJECTED/CANCELED")
            self.order = None
            self.in_position = False
    
    def notify_trade(self, trade):
        """Handle trade notifications (backtest mode only)"""
        if trade.isclosed:
            self.log(f"TRADE CLOSED - Profit: {trade.pnl:.2f}")
            self.in_position = False
    
    def _price_to_pips(self, price_diff: float) -> float:
        """
        Convert price difference to pips for EURUSD
        
        For other pairs, adjust accordingly:
        - JPY pairs: * 100
        - Other 5-digit pairs: * 10000
        """
        return price_diff * 10000  # EURUSD standard
    
    def log(self, txt: str):
        """Logging helper"""
        dt = self.data.datetime.date(0)
        print(f"{dt.isoformat()} - {txt}")


# Example usage for backtesting
if __name__ == "__main__":
    import backtrader as bt
    from datetime import datetime
    
    print("=" * 80)
    print("ATR Breakout Live Strategy - BACKTEST MODE")
    print("=" * 80)
    
    # Create cerebro
    cerebro = bt.Cerebro()
    
    # Add strategy (no bridge = backtest mode)
    cerebro.addstrategy(ATRBreakoutLive, symbol='EURUSD', volume=0.1)
    
    # Add data (you would normally load real data here)
    # For demonstration, we'll skip data loading
    # cerebro.adddata(data)
    
    # Set broker parameters
    cerebro.broker.setcash(10000.0)
    cerebro.broker.setcommission(commission=0.0001)  # 1 pip spread
    
    print(f"Starting Portfolio Value: ${cerebro.broker.getvalue():.2f}")
    
    # Run backtest (will fail without data, but shows the pattern)
    # cerebro.run()
    
    print("\nTo run in LIVE mode with MT5:")
    print("1. Create MT5Executor instance")
    print("2. Create BacktraderMT5Bridge with executor")
    print("3. Pass bridge to strategy: cerebro.addstrategy(ATRBreakoutLive, mt5_bridge=bridge)")
    print("4. Run cerebro.run() - strategy will execute in MT5")
