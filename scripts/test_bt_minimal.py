"""
Minimal Backtrader test - 100 bars only.
"""

import backtrader as bt
import pandas as pd
import numpy as np

class TestStrategy(bt.Strategy):
    def __init__(self):
        self.sma = bt.indicators.SMA(self.data.close, period=5)
        print("Strategy initialized")
        
    def next(self):
        if len(self) % 10 == 0:  # Print every 10 bars
            print(f"Bar {len(self)}: Close={self.data.close[0]:.5f}, SMA={self.sma[0]:.5f}")

# Create minimal data
np.random.seed(42)
dates = pd.date_range('2024-01-01', periods=100, freq='H')
close = 1.10 + np.cumsum(np.random.randn(100) * 0.001)

df = pd.DataFrame({
    'datetime': dates,
    'open': close,
    'high': close * 1.001,
    'low': close * 0.999,
    'close': close,
    'volume': 1000
})
df.set_index('datetime', inplace=True)

print(f"Data: {len(df)} bars")
print(f"Date range: {df.index[0]} to {df.index[-1]}")

# Run backtest
cerebro = bt.Cerebro()
cerebro.addstrategy(TestStrategy)
cerebro.adddata(bt.feeds.PandasData(dataname=df))
cerebro.broker.setcash(10000)

print("\nRunning backtest...")
cerebro.run()

print(f"\nFinal value: ${cerebro.broker.getvalue():,.2f}")
print("Done!")
