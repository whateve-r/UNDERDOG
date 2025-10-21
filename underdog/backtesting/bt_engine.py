"""
UNDERDOG Backtesting Engine - Backtrader Integration
====================================================

Complete backtesting pipeline with:
- Backtrader execution
- PropFirmRiskManager integration
- Monte Carlo validation
- Walk-Forward Optimization
- Standardized results for Streamlit

Usage:
------
    from underdog.backtesting.bt_engine import run_backtest
    
    results = run_backtest(
        strategy_name='ATRBreakout',
        symbol='EURUSD',
        start_date='2023-01-01',
        end_date='2024-12-31',
        strategy_params={'atr_period': 14},
        validate_monte_carlo=True
    )
    
    # Results dict contains:
    # - equity_curve: DataFrame
    # - trades: DataFrame
    # - metrics: Dict
    # - monte_carlo: Dict (if enabled)
    # - wfo_results: DataFrame (if enabled)

Author: UNDERDOG Development Team
Date: October 2025
"""

import backtrader as bt
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, Optional, Tuple, List
import logging

from underdog.backtesting.bt_adapter import get_strategy_class
from underdog.data.hf_loader import HuggingFaceDataHandler
from underdog.validation.monte_carlo import validate_backtest
from underdog.validation.wfo import WalkForwardOptimizer

logger = logging.getLogger(__name__)


def load_data_for_backtest(
    symbol: str,
    start_date: str,
    end_date: str,
    use_hf: bool = False,
    use_synthetic: bool = True
) -> pd.DataFrame:
    """
    Load data for backtesting.
    
    Priority:
    1. HuggingFace (if use_hf=True and authenticated)
    2. Synthetic data (if use_synthetic=True)
    3. Error
    
    Returns:
    --------
    DataFrame with columns: datetime, open, high, low, close, volume
    """
    if use_hf:
        try:
            logger.info(f"Attempting to load HuggingFace data for {symbol}")
            
            # HuggingFaceDataHandler has built-in synthetic fallback
            handler = HuggingFaceDataHandler(
                dataset_id='elthariel/histdata_fx_1m',
                symbol=symbol,
                start_date=start_date,
                end_date=end_date
            )
            
            # Get data from handler
            df = handler.df.copy()
            
            # Ensure datetime is a column (not index)
            if isinstance(df.index, pd.DatetimeIndex):
                df = df.reset_index()
                if 'timestamp' in df.columns:
                    df.rename(columns={'timestamp': 'datetime'}, inplace=True)
                elif df.index.name == 'timestamp':
                    df['datetime'] = df.index
                else:
                    # Index is already datetime
                    df['datetime'] = df.index
            
            # Ensure we have required columns
            required_cols = ['datetime', 'open', 'high', 'low', 'close', 'volume']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}. Available: {df.columns.tolist()}")
            
            # Set datetime as index for Backtrader
            df.set_index('datetime', inplace=True)
            
            # Keep only OHLCV columns (drop bid/ask if present)
            ohlcv_cols = ['open', 'high', 'low', 'close', 'volume']
            df = df[ohlcv_cols]
            
            logger.info(f"✓ Loaded {len(df):,} bars from HuggingFace ({df.index[0]} to {df.index[-1]})")
            return df
            
        except Exception as e:
            logger.warning(f"HuggingFace load failed: {e}")
            if not use_synthetic:
                raise
    
    if use_synthetic:
        # Generate synthetic data with realistic volatility
        logger.info(f"Generating synthetic data for {symbol}")
        dates = pd.date_range(start=start_date, end=end_date, freq='1h')
        
        np.random.seed(42)
        base_price = 1.1000 if 'EUR' in symbol else 1.2500
        
        # Generate realistic price movement (GBM with jumps)
        n = len(dates)
        returns = np.random.normal(0.00005, 0.002, n)  # Higher volatility
        
        # Add trending periods
        trend_changes = np.random.choice([0, 1, -1], n, p=[0.7, 0.15, 0.15])
        returns += trend_changes * 0.0003
        
        # Add volatility clusters
        volatility = np.abs(np.random.normal(1.0, 0.3, n))
        returns *= volatility
        
        close = base_price * np.exp(np.cumsum(returns))
        
        # Generate OHLC from close
        # High/Low ranges based on ATR-like volatility
        atr = np.abs(returns) * close * 1.5
        high = close + np.random.uniform(0.3, 0.7, n) * atr
        low = close - np.random.uniform(0.3, 0.7, n) * atr
        open_ = close + np.random.uniform(-0.5, 0.5, n) * atr
        
        df = pd.DataFrame({
            'datetime': dates,
            'open': open_,
            'high': high,
            'low': low,
            'close': close,
            'volume': np.random.uniform(100, 1000, len(dates))
        })
        
        df.set_index('datetime', inplace=True)
        logger.info(f"Generated {len(df):,} synthetic bars with realistic volatility")
        return df
    
    raise ValueError("No data source available")


class BacktestAnalyzer(bt.Analyzer):
    """Custom analyzer to extract detailed trade data."""
    
    def __init__(self):
        self.trades = []
        self.equity_curve = []
    
    def notify_trade(self, trade):
        """Record trade when closed."""
        if trade.isclosed:
            exit_price = trade.price + (trade.pnl / trade.size if trade.size != 0 else 0)
            self.trades.append({
                'entry_date': bt.num2date(trade.dtopen),
                'exit_date': bt.num2date(trade.dtclose),
                'entry_price': trade.price,
                'exit_price': exit_price,
                'size': trade.size,
                'pnl': trade.pnl,
                'pnl_net': trade.pnlcomm,
                'bars_held': trade.barlen
            })
    
    def next(self):
        """Record equity at each bar."""
        self.equity_curve.append({
            'datetime': self.strategy.datetime.datetime(),
            'equity': self.strategy.broker.getvalue()
        })
    
    def get_analysis(self):
        """Return collected data."""
        return {
            'trades': self.trades,
            'equity_curve': self.equity_curve
        }


def run_backtest(
    strategy_name: str,
    symbol: str,
    start_date: str,
    end_date: str,
    strategy_params: Optional[Dict] = None,
    initial_capital: float = 10000.0,
    commission: float = 0.0001,
    use_hf_data: bool = False,
    validate_monte_carlo: bool = False,
    mc_iterations: int = 10000,
    run_wfo: bool = False,
    wfo_param_grid: Optional[Dict] = None
) -> Dict:
    """
    Run complete backtest with validation.
    
    Parameters:
    -----------
    strategy_name : str
        Name of strategy ('ATRBreakout', 'SuperTrendRSI', 'BollingerCCI')
    symbol : str
        Trading symbol (e.g., 'EURUSD')
    start_date : str
        Start date (ISO format: 'YYYY-MM-DD')
    end_date : str
        End date (ISO format: 'YYYY-MM-DD')
    strategy_params : dict, optional
        Strategy parameters to override defaults
    initial_capital : float
        Starting capital (default: $10,000)
    commission : float
        Commission per trade (default: 0.0001 = 0.01%)
    use_hf_data : bool
        Use HuggingFace data (default: False = synthetic)
    validate_monte_carlo : bool
        Run Monte Carlo validation (default: False)
    mc_iterations : int
        Monte Carlo iterations (default: 10,000)
    run_wfo : bool
        Run Walk-Forward Optimization (default: False)
    wfo_param_grid : dict, optional
        Parameter grid for WFO
    
    Returns:
    --------
    dict with keys:
        - equity_curve: DataFrame(datetime, equity, drawdown_pct)
        - trades: DataFrame(all trade details)
        - metrics: Dict(performance metrics)
        - monte_carlo: Dict(validation results) [if enabled]
        - wfo_results: DataFrame [if enabled]
        - strategy: Strategy instance (for accessing trades)
    """
    logger.info(f"Starting backtest: {strategy_name} on {symbol} ({start_date} to {end_date})")
    
    # Load data
    df = load_data_for_backtest(
        symbol=symbol,
        start_date=start_date,
        end_date=end_date,
        use_hf=use_hf_data,
        use_synthetic=not use_hf_data
    )
    
    # Initialize Cerebro
    cerebro = bt.Cerebro()
    
    # Add strategy
    strategy_class = get_strategy_class(strategy_name)
    
    # Merge default params with user params
    params = strategy_params or {}
    cerebro.addstrategy(strategy_class, **params)
    
    # Add data
    data = bt.feeds.PandasData(dataname=df)
    cerebro.adddata(data)
    
    # Set broker
    cerebro.broker.setcash(initial_capital)
    cerebro.broker.setcommission(commission=commission)
    
    # Add analyzers
    cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trade_analyzer')
    cerebro.addanalyzer(BacktestAnalyzer, _name='custom')
    
    # Run backtest
    logger.info("Running backtest...")
    results = cerebro.run()
    strat = results[0]
    
    # Extract results
    final_value = cerebro.broker.getvalue()
    
    # Get analyzer results
    returns_analysis = strat.analyzers.returns.get_analysis()
    drawdown_analysis = strat.analyzers.drawdown.get_analysis()
    sharpe_analysis = strat.analyzers.sharpe.get_analysis()
    trade_analysis = strat.analyzers.trade_analyzer.get_analysis()
    custom_analysis = strat.analyzers.custom.get_analysis()
    
    # Build equity curve DataFrame
    equity_df = pd.DataFrame(custom_analysis['equity_curve'])
    if len(equity_df) > 0:
        equity_df['drawdown_pct'] = calculate_drawdown(equity_df['equity'].values)
    else:
        equity_df['drawdown_pct'] = 0.0
    
    # Build trades DataFrame
    trades_df = pd.DataFrame(custom_analysis['trades'])
    if len(trades_df) > 0:
        trades_df['symbol'] = symbol
        trades_df['side'] = trades_df['size'].apply(lambda x: 'BUY' if x > 0 else 'SELL')
    
    # Calculate metrics
    total_return = ((final_value / initial_capital) - 1) * 100
    max_dd = drawdown_analysis.max.drawdown / 100 if hasattr(drawdown_analysis.max, 'drawdown') else 0
    sharpe = sharpe_analysis.get('sharperatio', 0) or 0
    
    # Trade statistics
    num_trades = len(trades_df)
    winning_trades = trades_df[trades_df['pnl'] > 0] if len(trades_df) > 0 else pd.DataFrame()
    losing_trades = trades_df[trades_df['pnl'] < 0] if len(trades_df) > 0 else pd.DataFrame()
    
    win_rate = (len(winning_trades) / num_trades * 100) if num_trades > 0 else 0
    avg_win = winning_trades['pnl'].mean() if len(winning_trades) > 0 else 0
    avg_loss = losing_trades['pnl'].mean() if len(losing_trades) > 0 else 0
    profit_factor = abs(winning_trades['pnl'].sum() / losing_trades['pnl'].sum()) if len(losing_trades) > 0 else 0
    
    metrics = {
        'initial_capital': initial_capital,
        'final_capital': final_value,
        'total_return_pct': total_return,
        'max_drawdown_pct': max_dd * 100,
        'sharpe_ratio': sharpe,
        'num_trades': num_trades,
        'win_rate_pct': win_rate,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'profit_factor': profit_factor,
        'avg_pnl': trades_df['pnl'].mean() if len(trades_df) > 0 else 0,
        'total_commission': trades_df['pnl'].sum() - trades_df['pnl_net'].sum() if len(trades_df) > 0 else 0,
        'avg_slippage_pips': 0.0  # Placeholder
    }
    
    logger.info(f"Backtest complete: {num_trades} trades, {total_return:.2f}% return")
    
    # Build results dict
    results_dict = {
        'equity_curve': equity_df,
        'trades': trades_df,
        'metrics': metrics,
        'strategy': strat  # For accessing internal state
    }
    
    # Monte Carlo validation
    if validate_monte_carlo and len(trades_df) >= 10:
        logger.info("Running Monte Carlo validation...")
        
        trades_for_mc = [{'pnl': row['pnl']} for _, row in trades_df.iterrows()]
        is_robust = validate_backtest(
            trades=trades_for_mc,
            mc_iterations=mc_iterations,
            confidence=0.05
        )
        
        results_dict['monte_carlo'] = {
            'is_robust': is_robust,
            'iterations': mc_iterations,
            'num_trades': len(trades_for_mc)
        }
        
        logger.info(f"Monte Carlo: {'✓ ROBUST' if is_robust else '✗ NOT ROBUST'}")
    
    # Walk-Forward Optimization
    if run_wfo and wfo_param_grid:
        logger.info("Running Walk-Forward Optimization...")
        
        # TODO: Implement WFO integration
        # This requires running multiple backtests with different date windows
        # For now, placeholder
        results_dict['wfo_results'] = pd.DataFrame()
        logger.warning("WFO not yet implemented in bt_engine")
    
    return results_dict


def calculate_drawdown(equity: np.ndarray) -> np.ndarray:
    """Calculate drawdown percentage from equity curve."""
    running_max = np.maximum.accumulate(equity)
    drawdown = (equity - running_max) / running_max * 100
    return drawdown


def run_parameter_sweep(
    strategy_name: str,
    symbol: str,
    start_date: str,
    end_date: str,
    param_grid: Dict[str, List],
    initial_capital: float = 10000.0
) -> pd.DataFrame:
    """
    Run parameter sweep for sensitivity analysis.
    
    Parameters:
    -----------
    param_grid : dict
        Dict of param_name -> list of values
        Example: {'atr_period': [10, 14, 20], 'rsi_period': [10, 14, 20]}
    
    Returns:
    --------
    DataFrame with columns: param1, param2, ..., sharpe_ratio, total_return, max_dd
    """
    from itertools import product
    
    logger.info(f"Running parameter sweep with {len(param_grid)} parameters")
    
    # Generate all combinations
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())
    combinations = list(product(*param_values))
    
    logger.info(f"Testing {len(combinations)} parameter combinations")
    
    results = []
    
    for i, combo in enumerate(combinations):
        if i % 10 == 0:
            logger.info(f"Progress: {i}/{len(combinations)}")
        
        # Build params dict
        params = dict(zip(param_names, combo))
        
        try:
            # Run backtest
            result = run_backtest(
                strategy_name=strategy_name,
                symbol=symbol,
                start_date=start_date,
                end_date=end_date,
                strategy_params=params,
                initial_capital=initial_capital,
                validate_monte_carlo=False
            )
            
            # Record results
            row = params.copy()
            row['sharpe_ratio'] = result['metrics']['sharpe_ratio']
            row['total_return_pct'] = result['metrics']['total_return_pct']
            row['max_drawdown_pct'] = result['metrics']['max_drawdown_pct']
            row['num_trades'] = result['metrics']['num_trades']
            row['win_rate_pct'] = result['metrics']['win_rate_pct']
            
            results.append(row)
            
        except Exception as e:
            logger.error(f"Parameter combo {params} failed: {e}")
            continue
    
    df_results = pd.DataFrame(results)
    logger.info(f"Parameter sweep complete: {len(df_results)} successful runs")
    
    return df_results


# Example usage
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    
    print("="*80)
    print("UNDERDOG Backtesting Engine - Test")
    print("="*80)
    
    # Test ATRBreakout strategy
    results = run_backtest(
        strategy_name='ATRBreakout',
        symbol='EURUSD',
        start_date='2023-01-01',
        end_date='2024-12-31',
        strategy_params={
            'atr_period': 14,
            'atr_multiplier_entry': 1.5,
            'risk_per_trade': 0.02
        },
        initial_capital=10000.0,
        validate_monte_carlo=True,
        mc_iterations=1000  # Reduced for testing
    )
    
    print("\n" + "="*80)
    print("RESULTS")
    print("="*80)
    print(f"Initial Capital: ${results['metrics']['initial_capital']:,.2f}")
    print(f"Final Capital:   ${results['metrics']['final_capital']:,.2f}")
    print(f"Total Return:    {results['metrics']['total_return_pct']:.2f}%")
    print(f"Max Drawdown:    {results['metrics']['max_drawdown_pct']:.2f}%")
    print(f"Sharpe Ratio:    {results['metrics']['sharpe_ratio']:.2f}")
    print(f"Num Trades:      {results['metrics']['num_trades']}")
    print(f"Win Rate:        {results['metrics']['win_rate_pct']:.1f}%")
    
    if 'monte_carlo' in results:
        print("\nMonte Carlo:")
        print(f"  Robust: {results['monte_carlo']['is_robust']}")
    
    print("\nEquity Curve Preview:")
    print(results['equity_curve'].head())
    
    print("\nTrades Preview:")
    print(results['trades'].head())
