"""
Backtrader evaluation with HF data.
Test: SMA Crossover with realistic spread/slippage.
"""

import backtrader as bt
import pandas as pd
from datetime import datetime

from underdog.data.hf_loader import HuggingFaceDataHandler


class SMACrossover(bt.Strategy):
    """Simple SMA Crossover strategy."""
    
    params = (
        ('fast_period', 10),
        ('slow_period', 50),
    )
    
    def __init__(self):
        self.sma_fast = bt.indicators.SMA(self.data.close, period=self.params.fast_period)
        self.sma_slow = bt.indicators.SMA(self.data.close, period=self.params.slow_period)
        self.crossover = bt.indicators.CrossOver(self.sma_fast, self.sma_slow)
    
    def next(self):
        if not self.position:
            if self.crossover > 0:  # Golden cross
                self.buy()
        else:
            if self.crossover < 0:  # Death cross
                self.close()


def test_backtrader():
    """Test Backtrader with HF data."""
    print("="*80)
    print("BACKTRADER EVALUATION")
    print("="*80)
    
    # Load data
    print("\n[1] Loading data from HF...")
    handler = HuggingFaceDataHandler(
        dataset_id='elthariel/histdata_fx_1m',
        symbol='EURUSD',
        start_date='2024-01-01',
        end_date='2024-03-31'  # 3 months for quick test
    )
    
    # Convert to Backtrader format
    df = handler.df.copy()
    df['openinterest'] = 0  # Required by Backtrader
    
    data = bt.feeds.PandasData(dataname=df)
    
    # Create Cerebro
    print("\n[2] Setting up Backtrader...")
    cerebro = bt.Cerebro()
    cerebro.adddata(data)
    cerebro.addstrategy(SMACrossover)
    
    # Set initial capital
    initial_capital = 10000.0
    cerebro.broker.setcash(initial_capital)
    
    # Set realistic commission (spread-based)
    # EURUSD spread: ~0.2 pips = 0.00002
    # For 1 standard lot (100k units), that's $2 per trade
    cerebro.broker.setcommission(
        commission=0.00002,  # 0.2 pips
        mult=100000,  # 1 standard lot
        margin=100,  # 1:100 leverage
    )
    
    print(f"  Initial capital: ${initial_capital:,.2f}")
    print(f"  Commission: 0.2 pips (realistic EURUSD spread)")
    
    # Run backtest
    print("\n[3] Running backtest...")
    cerebro.run()
    
    # Results
    final_value = cerebro.broker.getvalue()
    pnl = final_value - initial_capital
    pnl_pct = (pnl / initial_capital) * 100
    
    print("\n[4] Results:")
    print(f"  Final portfolio value: ${final_value:,.2f}")
    print(f"  P&L: ${pnl:,.2f} ({pnl_pct:+.2f}%)")
    
    # Analyzers
    print("\n[5] Adding analyzers...")
    cerebro = bt.Cerebro()
    cerebro.adddata(data)
    cerebro.addstrategy(SMACrossover)
    cerebro.broker.setcash(initial_capital)
    cerebro.broker.setcommission(commission=0.00002, mult=100000, margin=100)
    
    # Add analyzers
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
    cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')
    
    results = cerebro.run()
    strat = results[0]
    
    # Print analyzer results
    print("\n  Sharpe Ratio:", strat.analyzers.sharpe.get_analysis().get('sharperatio', 'N/A'))
    
    dd = strat.analyzers.drawdown.get_analysis()
    print(f"  Max Drawdown: {dd.max.drawdown:.2f}%")
    
    trades = strat.analyzers.trades.get_analysis()
    print(f"\n  Total trades: {trades.total.closed if trades.total.closed else 0}")
    if trades.total.closed:
        won = trades.won.total if hasattr(trades.won, 'total') else 0
        lost = trades.lost.total if hasattr(trades.lost, 'total') else 0
        print(f"  Won: {won}, Lost: {lost}")
        if trades.total.closed > 0:
            print(f"  Win rate: {(won / trades.total.closed) * 100:.1f}%")
    
    print("\n" + "="*80)
    print("✓ BACKTRADER TEST COMPLETE")
    print("="*80)
    
    return cerebro, results


if __name__ == '__main__':
    cerebro, results = test_backtrader()
