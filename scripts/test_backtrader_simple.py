"""
Simple Backtrader test with synthetic data.
Test basic functionality before HF integration.
"""

import backtrader as bt
import pandas as pd
from datetime import datetime, timedelta

class SimpleStrategy(bt.Strategy):
    params = (
        ('fast_period', 10),
        ('slow_period', 30),
    )
    
    def __init__(self):
        print(f"[INIT] Strategy initialized with fast={self.params.fast_period}, slow={self.params.slow_period}")
        self.fast_ma = bt.indicators.SMA(self.data.close, period=self.params.fast_period)
        self.slow_ma = bt.indicators.SMA(self.data.close, period=self.params.slow_period)
        self.order = None
        
    def log(self, txt):
        dt = self.data.datetime.date(0)
        print(f'[{dt}] {txt}')
    
    def notify_order(self, order):
        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(f'BUY EXECUTED, Price: {order.executed.price:.5f}')
            elif order.issell():
                self.log(f'SELL EXECUTED, Price: {order.executed.price:.5f}')
        self.order = None
    
    def next(self):
        # Simple crossover strategy
        if not self.position:
            if self.fast_ma[0] > self.slow_ma[0]:
                self.log(f'BUY SIGNAL - Fast MA: {self.fast_ma[0]:.5f} > Slow MA: {self.slow_ma[0]:.5f}')
                self.order = self.buy()
        else:
            if self.fast_ma[0] < self.slow_ma[0]:
                self.log(f'SELL SIGNAL - Fast MA: {self.fast_ma[0]:.5f} < Slow MA: {self.slow_ma[0]:.5f}')
                self.order = self.sell()

def create_synthetic_data():
    """Create simple synthetic OHLCV data."""
    print("\n[DATA] Creating synthetic data...")
    
    dates = pd.date_range(start='2024-01-01', end='2024-03-31', freq='1H')
    
    # Simple trending price with noise
    import numpy as np
    np.random.seed(42)
    
    base_price = 1.1000
    trend = np.linspace(0, 0.01, len(dates))  # Upward trend
    noise = np.random.normal(0, 0.0005, len(dates))
    close = base_price + trend + np.cumsum(noise)
    
    df = pd.DataFrame({
        'datetime': dates,
        'open': close * 0.9999,
        'high': close * 1.0001,
        'low': close * 0.9998,
        'close': close,
        'volume': np.random.uniform(100, 1000, len(dates))
    })
    
    df.set_index('datetime', inplace=True)
    print(f"[DATA] Created {len(df)} bars from {df.index[0]} to {df.index[-1]}")
    print(f"[DATA] Price range: {df['close'].min():.5f} to {df['close'].max():.5f}")
    
    return df

def run_backtest():
    print("\n" + "="*80)
    print("BACKTRADER SIMPLE TEST")
    print("="*80)
    
    # Create cerebro
    cerebro = bt.Cerebro()
    print("\n[SETUP] Cerebro created")
    
    # Add strategy
    cerebro.addstrategy(SimpleStrategy)
    print("[SETUP] Strategy added")
    
    # Create data
    df = create_synthetic_data()
    
    # Convert to Backtrader format
    data = bt.feeds.PandasData(dataname=df)
    cerebro.adddata(data)
    print("[SETUP] Data feed added")
    
    # Set initial capital
    start_cash = 10000.0
    cerebro.broker.setcash(start_cash)
    print(f"[SETUP] Initial capital: ${start_cash:,.2f}")
    
    # Set commission
    cerebro.broker.setcommission(commission=0.0001)  # 0.01% = 1 pip spread
    print("[SETUP] Commission: 0.01%")
    
    # Run backtest
    print("\n" + "="*80)
    print("RUNNING BACKTEST...")
    print("="*80 + "\n")
    
    cerebro.run()
    
    # Results
    final_value = cerebro.broker.getvalue()
    pnl = final_value - start_cash
    pnl_pct = (pnl / start_cash) * 100
    
    print("\n" + "="*80)
    print("RESULTS")
    print("="*80)
    print(f"Initial Capital: ${start_cash:,.2f}")
    print(f"Final Value:     ${final_value:,.2f}")
    print(f"P&L:             ${pnl:,.2f} ({pnl_pct:+.2f}%)")
    print("="*80)

if __name__ == '__main__':
    run_backtest()
