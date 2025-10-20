"""
Simple Backtesting Runner - Streamlit Integration
==================================================

Simplified wrapper around EventDrivenBacktest for easy Streamlit integration.

Usage from Streamlit:
    from underdog.backtesting.simple_runner import run_simple_backtest
    
    results = run_simple_backtest(
        ea_class=FxSuperTrendRSI,
        ea_config=config,
        ohlcv_data=df,
        initial_capital=100000
    )
    
    # results contains:
    # - equity_curve: pd.DataFrame
    # - trades: pd.DataFrame
    # - metrics: Dict (Sharpe, win_rate, max_dd, etc.)
    # - final_capital: float
"""

from typing import Dict, Any, Type
import pandas as pd
import numpy as np
from datetime import datetime
import asyncio

from underdog.backtesting.engines.event_driven import (
    EventDrivenBacktest,
    BacktestConfig,
    SignalEvent
)


def run_simple_backtest(
    ea_class: Type,
    ea_config: Any,
    ohlcv_data: pd.DataFrame,
    initial_capital: float = 100000.0,
    commission_pct: float = 0.0,
    slippage_pips: float = 0.5
) -> Dict[str, Any]:
    """
    Run backtest with simplified interface for Streamlit.
    
    Args:
        ea_class: EA class (e.g., FxSuperTrendRSI)
        ea_config: EA configuration instance
        ohlcv_data: DataFrame with columns [timestamp, open, high, low, close, volume]
                    Optional: bid, ask (will be calculated from close if missing)
        initial_capital: Starting capital (default: $100,000)
        commission_pct: Commission as percentage (default: 0.0)
        slippage_pips: Slippage in pips (default: 0.5)
    
    Returns:
        Dict with:
            - equity_curve: pd.DataFrame [timestamp, equity, cash, unrealized_pnl, drawdown_pct]
            - trades: pd.DataFrame [timestamp, symbol, side, entry, exit, pnl, etc.]
            - metrics: Dict with performance metrics
            - final_capital: float
    """
    
    print(f"\n{'='*80}")
    print(f"SIMPLE BACKTEST RUNNER")
    print(f"{'='*80}")
    print(f"EA: {ea_class.__name__}")
    print(f"Symbol: {ea_config.symbol}")
    print(f"Timeframe: {ea_config.timeframe}")
    print(f"Initial Capital: ${initial_capital:,.2f}")
    print(f"Data: {len(ohlcv_data):,} bars from {ohlcv_data.iloc[0]['timestamp']} to {ohlcv_data.iloc[-1]['timestamp']}")
    print(f"{'='*80}\n")
    
    # =========================================================================
    # STEP 1: Convert OHLCV to tick format (bid/ask)
    # =========================================================================
    
    tick_data = _ohlcv_to_ticks(ohlcv_data, ea_config.symbol)
    
    # =========================================================================
    # STEP 2: Create EA instance and strategy wrapper
    # =========================================================================
    
    ea_instance = ea_class(ea_config)
    
    # Initialize EA (synchronously for simplicity)
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    
    loop.run_until_complete(ea_instance.initialize())
    
    # Create strategy function for event-driven backtest
    def strategy_wrapper(tick_event, positions, current_capital):
        """Wrapper to convert tick to signal"""
        
        # Get recent bars around this tick
        # (In real backtest, we'd maintain a rolling window)
        # For simplicity, we'll just call generate_signal with full data
        
        # Check if we should generate signal (every N ticks to avoid overtrading)
        if not hasattr(strategy_wrapper, 'tick_count'):
            strategy_wrapper.tick_count = 0
            strategy_wrapper.last_signal_bar = 0
        
        strategy_wrapper.tick_count += 1
        
        # Only check for signals every 15 minutes worth of ticks (assuming M15)
        # This prevents checking on every single tick
        bars_per_signal = 1  # Check every bar
        
        if strategy_wrapper.tick_count % bars_per_signal != 0:
            return None
        
        # Get bar index
        bar_idx = strategy_wrapper.tick_count // bars_per_signal
        
        if bar_idx <= strategy_wrapper.last_signal_bar:
            return None
        
        # Get recent bars
        lookback = min(100, bar_idx)
        recent_bars = ohlcv_data.iloc[max(0, bar_idx - lookback):bar_idx + 1].copy()
        
        if len(recent_bars) < 50:  # Need minimum bars for indicators
            return None
        
        # Generate signal from EA
        try:
            signal = loop.run_until_complete(ea_instance.generate_signal(recent_bars))
            
            if signal:
                strategy_wrapper.last_signal_bar = bar_idx
                
                # Convert to SignalEvent
                return SignalEvent(
                    timestamp=tick_event.timestamp,
                    strategy_id=ea_class.__name__,
                    symbol=tick_event.symbol,
                    side='buy' if signal.type.name == 'BUY' else 'sell',
                    confidence=signal.confidence,
                    size_suggestion=0.01  # Fixed lot size for simplicity
                )
        except Exception as e:
            # Silently ignore signal generation errors
            pass
        
        return None
    
    # =========================================================================
    # STEP 3: Run event-driven backtest
    # =========================================================================
    
    backtest_config = BacktestConfig(
        initial_capital=initial_capital,
        commission_pct=commission_pct,
        base_slippage_pips=slippage_pips,
        use_dynamic_spread=True,
        use_dynamic_slippage=True
    )
    
    backtest = EventDrivenBacktest(backtest_config)
    
    results = backtest.run(
        tick_data=tick_data,
        strategy=strategy_wrapper,
        bar_duration_minutes=15  # Assuming M15
    )
    
    # =========================================================================
    # STEP 4: Format results for Streamlit
    # =========================================================================
    
    formatted_results = {
        'equity_curve': results['equity_curve'],
        'trades': results['trades'],
        'metrics': {
            'final_capital': results['final_capital'],
            'total_return_pct': results['total_return_pct'],
            'num_trades': results['num_trades'],
            'win_rate_pct': results['win_rate_pct'],
            'sharpe_ratio': results['sharpe_ratio'],
            'max_drawdown_pct': results['max_drawdown_pct'],
            'avg_pnl': results['avg_pnl'],
            'total_commission': results['total_commission'],
            'avg_slippage_pips': results['avg_slippage_pips']
        },
        'final_capital': results['final_capital']
    }
    
    # Shutdown EA
    loop.run_until_complete(ea_instance.shutdown())
    
    print(f"\n{'='*80}")
    print(f"BACKTEST COMPLETE")
    print(f"{'='*80}")
    print(f"Final Capital: ${results['final_capital']:,.2f}")
    print(f"Total Return: {results['total_return_pct']:.2f}%")
    print(f"Trades: {results['num_trades']}")
    print(f"Win Rate: {results['win_rate_pct']:.2f}%")
    print(f"Sharpe Ratio: {results['sharpe_ratio']:.3f}")
    print(f"Max Drawdown: {results['max_drawdown_pct']:.2f}%")
    print(f"{'='*80}\n")
    
    return formatted_results


def _ohlcv_to_ticks(ohlcv: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """
    Convert OHLCV data to tick format (bid/ask) for event-driven backtest.
    
    Args:
        ohlcv: DataFrame with OHLCV data
        symbol: Trading symbol
    
    Returns:
        DataFrame with columns [timestamp, symbol, bid, ask, volume]
    """
    
    # Ensure timestamp column
    if 'timestamp' not in ohlcv.columns:
        if isinstance(ohlcv.index, pd.DatetimeIndex):
            ohlcv = ohlcv.reset_index()
            ohlcv.rename(columns={'index': 'timestamp'}, inplace=True)
        else:
            raise ValueError("OHLCV data must have 'timestamp' column or DatetimeIndex")
    
    # Calculate bid/ask from close if not present
    if 'bid' not in ohlcv.columns or 'ask' not in ohlcv.columns:
        # Assume 1 pip spread (0.0001 for EURUSD-like pairs)
        spread = 0.0001
        
        ohlcv['bid'] = ohlcv['close'] - spread / 2
        ohlcv['ask'] = ohlcv['close'] + spread / 2
    
    # Create tick data (use close as tick)
    tick_data = pd.DataFrame({
        'timestamp': ohlcv['timestamp'],
        'symbol': symbol,
        'bid': ohlcv['bid'],
        'ask': ohlcv['ask'],
        'volume': ohlcv.get('volume', 0.0)
    })
    
    # Set timestamp as index
    tick_data = tick_data.set_index('timestamp')
    
    return tick_data


def load_historical_data(filepath: str) -> pd.DataFrame:
    """
    Load historical OHLCV data from CSV.
    
    Args:
        filepath: Path to CSV file
    
    Returns:
        DataFrame with parsed timestamps
    """
    df = pd.read_csv(filepath, parse_dates=['timestamp'])
    
    print(f"Loaded {len(df):,} bars from {filepath}")
    print(f"  Date range: {df['timestamp'].min()} → {df['timestamp'].max()}")
    print(f"  Columns: {list(df.columns)}")
    
    return df


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":
    """Example backtest execution"""
    
    import sys
    from pathlib import Path
    import MetaTrader5 as mt5
    
    # Add project root to path
    sys.path.append(str(Path(__file__).parent.parent.parent))
    
    from underdog.strategies.ea_supertrend_rsi_v4 import FxSuperTrendRSI, SuperTrendRSIConfig
    
    print("\n" + "="*80)
    print("SIMPLE BACKTEST RUNNER - EXAMPLE")
    print("="*80 + "\n")
    
    # Check if historical data exists
    data_file = Path("data/historical/EURUSD_M15.csv")
    
    if not data_file.exists():
        print(f"❌ Data file not found: {data_file}")
        print("\nPlease generate data first:")
        print("  poetry run python scripts/generate_synthetic_data.py")
        sys.exit(1)
    
    # Load data
    print("Loading historical data...")
    ohlcv_data = load_historical_data(str(data_file))
    
    # Configure EA
    ea_config = SuperTrendRSIConfig(
        symbol="EURUSD",
        timeframe=mt5.TIMEFRAME_M15,
        magic_number=101,
        risk_per_trade=0.01,
        enable_events=False
    )
    
    # Run backtest
    results = run_simple_backtest(
        ea_class=FxSuperTrendRSI,
        ea_config=ea_config,
        ohlcv_data=ohlcv_data,
        initial_capital=100000
    )
    
    # Display results
    print("\n" + "="*80)
    print("RESULTS SUMMARY")
    print("="*80)
    
    for key, value in results['metrics'].items():
        if isinstance(value, float):
            print(f"  {key}: {value:.2f}")
        else:
            print(f"  {key}: {value}")
    
    print("\n" + "="*80 + "\n")
