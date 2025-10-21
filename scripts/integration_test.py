"""
Complete Integration Test - UNDERDOG Trading System

Demonstrates:
1. Data loading (HuggingFace with synthetic fallback)
2. Strategy execution (SMA Crossover)
3. Risk Management (PropFirmRiskManager)
4. Backtesting (Backtrader)
5. Validation (Monte Carlo)
"""

import backtrader as bt
import pandas as pd
import numpy as np
from datetime import datetime

# Import UNDERDOG components
from underdog.risk.prop_firm_rme import PropFirmRiskManager
from underdog.validation.monte_carlo import validate_backtest


class SMACrossoverWithRisk(bt.Strategy):
    """SMA Crossover strategy with integrated risk management."""
    
    params = (
        ('fast_period', 10),
        ('slow_period', 30),
    )
    
    def __init__(self):
        self.fast_ma = bt.indicators.SMA(self.data.close, period=self.params.fast_period)
        self.slow_ma = bt.indicators.SMA(self.data.close, period=self.params.slow_period)
        self.atr = bt.indicators.ATR(self.data, period=14)
        self.order = None
        self.trades = []
        
        # Initialize Risk Manager
        self.rme = PropFirmRiskManager(
            daily_dd_limit=0.05,
            total_dd_limit=0.10,
            risk_per_trade=0.02,
            use_kelly=True
        )
        
        self.daily_start_value = self.broker.getvalue()
        self.high_water_mark = self.broker.getvalue()
        
    def log(self, txt):
        dt = self.data.datetime.date(0)
        print(f'[{dt}] {txt}')
    
    def notify_order(self, order):
        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(f'BUY EXECUTED @ {order.executed.price:.5f}')
            elif order.issell():
                self.log(f'SELL EXECUTED @ {order.executed.price:.5f}')
        self.order = None
    
    def notify_trade(self, trade):
        if trade.isclosed:
            pnl = trade.pnl
            self.log(f'TRADE CLOSED - P&L: ${pnl:.2f}')
            
            # Record trade in RME
            self.rme.record_trade(pnl)
            self.trades.append({'pnl': pnl})
    
    def next(self):
        # Check daily DD limit
        current_value = self.broker.getvalue()
        daily_dd = (self.daily_start_value - current_value) / self.daily_start_value
        
        if daily_dd > self.rme.daily_dd_limit:
            if self.position:
                self.log(f'⚠️ DAILY DD BREACHED ({daily_dd:.1%}) - CLOSING POSITION')
                self.close()
            return
        
        # Update high water mark
        self.high_water_mark = max(self.high_water_mark, current_value)
        
        # Trading logic
        if not self.position:
            if self.fast_ma[0] > self.slow_ma[0]:
                # Calculate position size
                stop_distance = self.atr[0] * 2.0
                equity = current_value
                risk_amount = equity * self.rme.risk_per_trade
                size = risk_amount / stop_distance if stop_distance > 0 else 0.01
                size = max(0.01, min(size, 1.0))  # Clamp to 0.01-1.0 lots
                
                self.order = self.buy(size=size)
                self.log(f'BUY SIGNAL - Size: {size:.2f} lots (ATR: {self.atr[0]:.5f})')
        else:
            if self.fast_ma[0] < self.slow_ma[0]:
                self.order = self.close()
                self.log(f'SELL SIGNAL')


def create_synthetic_data():
    """Create synthetic EURUSD data."""
    np.random.seed(42)
    dates = pd.date_range(start='2023-01-01', end='2024-12-31', freq='1h')
    
    base_price = 1.1000
    trend = np.linspace(0, 0.02, len(dates))
    noise = np.cumsum(np.random.normal(0, 0.0005, len(dates)))
    close = base_price + trend + noise
    
    df = pd.DataFrame({
        'datetime': dates,
        'open': close * 0.9999,
        'high': close * 1.0002,
        'low': close * 0.9998,
        'close': close,
        'volume': np.random.uniform(100, 1000, len(dates))
    })
    
    df.set_index('datetime', inplace=True)
    return df


def run_integrated_backtest():
    """Run complete integrated backtest."""
    
    print("\n" + "="*80)
    print("UNDERDOG INTEGRATED BACKTEST")
    print("="*80)
    print("Components:")
    print("  • Data: Synthetic EURUSD (2023-2024)")
    print("  • Strategy: SMA Crossover (10/30)")
    print("  • Risk: PropFirmRiskManager (5% daily DD, 2% risk/trade)")
    print("  • Framework: Backtrader")
    print("  • Validation: Monte Carlo (10k iterations)")
    print("="*80 + "\n")
    
    # Create data
    print("Creating synthetic data...")
    df = create_synthetic_data()
    print(f"Data: {len(df):,} bars from {df.index[0]} to {df.index[-1]}")
    
    # Setup Backtrader
    cerebro = bt.Cerebro()
    cerebro.addstrategy(SMACrossoverWithRisk)
    
    data = bt.feeds.PandasData(dataname=df)
    cerebro.adddata(data)
    
    start_cash = 10000.0
    cerebro.broker.setcash(start_cash)
    cerebro.broker.setcommission(commission=0.0001)
    
    # Add analyzers
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
    cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')
    
    print("\n" + "="*80)
    print("RUNNING BACKTEST...")
    print("="*80 + "\n")
    
    # Run
    results = cerebro.run()
    strat = results[0]
    
    # Results
    final_value = cerebro.broker.getvalue()
    pnl = final_value - start_cash
    pnl_pct = (pnl / start_cash) * 100
    
    dd_analyzer = strat.analyzers.drawdown.get_analysis()
    max_dd = dd_analyzer.max.drawdown / 100.0 if dd_analyzer.max.drawdown else 0.01
    
    returns_analyzer = strat.analyzers.returns.get_analysis()
    total_return = returns_analyzer.get('rtot', 0)
    
    calmar = (total_return / abs(max_dd)) if max_dd != 0 else 0
    
    sharpe_analyzer = strat.analyzers.sharpe.get_analysis()
    sharpe = sharpe_analyzer.get('sharperatio', 0) or 0
    
    trades_analyzer = strat.analyzers.trades.get_analysis()
    total_trades = trades_analyzer.total.total if hasattr(trades_analyzer.total, 'total') else 0
    
    print("\n" + "="*80)
    print("BACKTEST RESULTS")
    print("="*80)
    print(f"Initial Capital:  ${start_cash:,.2f}")
    print(f"Final Value:      ${final_value:,.2f}")
    print(f"P&L:              ${pnl:,.2f} ({pnl_pct:+.2f}%)")
    print(f"Max Drawdown:     {max_dd*100:.2f}%")
    print(f"Calmar Ratio:     {calmar:.2f}")
    print(f"Sharpe Ratio:     {sharpe:.2f}")
    print(f"Total Trades:     {total_trades}")
    print("="*80)
    
    # Risk Management Stats
    rme_stats = strat.rme.get_statistics()
    print("\n" + "="*80)
    print("RISK MANAGEMENT STATS")
    print("="*80)
    print(f"Trades Recorded:  {rme_stats['total_trades']}")
    print(f"Win Rate:         {rme_stats['win_rate']:.1%}")
    print(f"Avg Win/Loss:     {rme_stats['avg_win_loss']:.2f}")
    print(f"Kelly Fraction:   {rme_stats['kelly_fraction']:.2%}")
    print("="*80)
    
    # Monte Carlo Validation
    if len(strat.trades) >= 10:
        print("\n")
        is_robust = validate_backtest(
            strat.trades,
            mc_iterations=10000,
            confidence=0.05
        )
        
        if not is_robust:
            print("\n⚠️ WARNING: Strategy failed Monte Carlo validation")
            print("   Do NOT trade this strategy in production")
    else:
        print(f"\n⚠️ Too few trades ({len(strat.trades)}) for Monte Carlo validation")
    
    print("\n" + "="*80)
    print("INTEGRATION TEST COMPLETE")
    print("="*80)


if __name__ == '__main__':
    run_integrated_backtest()
