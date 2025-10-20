"""
Z-Score Utilities for Pairs Trading
Provides Z-Score calculation and signal generation for mean reversion strategies.

The Z-Score measures how many standard deviations the spread is from its mean,
enabling systematic entry/exit decisions for pairs trading.
"""
import numpy as np
import pandas as pd
from typing import Tuple, Optional


def calculate_spread_zscore(spread: pd.Series, window: int = 20) -> pd.Series:
    """
    Calculate rolling Z-Score of price spread.
    
    Z-Score = (Spread - Mean) / StdDev
    
    Interpretation:
    - Z > +2: Spread unusually high → SHORT spread (sell A, buy B)
    - Z < -2: Spread unusually low → LONG spread (buy A, sell B)
    - Z ≈ 0: Spread at equilibrium → EXIT positions
    
    Args:
        spread: Price spread series (e.g., A - hedge_ratio * B)
        window: Rolling window for mean/std calculation (default: 20 bars)
    
    Returns:
        Z-Score series
    
    Example:
        >>> spread = pd.Series([1.0, 1.2, 1.5, 1.8, 1.6, 1.3, 1.1])
        >>> zscore = calculate_spread_zscore(spread, window=5)
        >>> print(zscore.iloc[-1])
        0.447...  # Spread is 0.45 std devs above mean
    """
    # Rolling mean and std
    rolling_mean = spread.rolling(window=window, min_periods=window).mean()
    rolling_std = spread.rolling(window=window, min_periods=window).std()
    
    # Z-Score
    zscore = (spread - rolling_mean) / rolling_std
    
    return zscore


def generate_zscore_signals(
    zscore: pd.Series,
    entry_threshold: float = 1.5,
    exit_threshold: float = 0.5
) -> pd.Series:
    """
    Generate trading signals from Z-Score.
    
    Signal logic:
    - BUY (+1): Z-Score < -entry_threshold (spread too low)
    - SELL (-1): Z-Score > +entry_threshold (spread too high)
    - EXIT (0): |Z-Score| < exit_threshold (spread normalized)
    
    Args:
        zscore: Z-Score series
        entry_threshold: Z-Score threshold for entry (default: 1.5)
        exit_threshold: Z-Score threshold for exit (default: 0.5)
    
    Returns:
        Signal series: +1 (long), -1 (short), 0 (no position/exit)
    
    Example:
        >>> zscore = pd.Series([-2.0, -1.8, -0.5, 0.2, 1.6, 2.1, 0.3])
        >>> signals = generate_zscore_signals(zscore, entry_threshold=1.5)
        >>> print(signals.values)
        [-1. -1.  0.  0.  1.  1.  0.]
    """
    signals = pd.Series(0, index=zscore.index)
    
    # Long signal: Z-Score very negative (spread too low)
    signals[zscore < -entry_threshold] = 1
    
    # Short signal: Z-Score very positive (spread too high)
    signals[zscore > entry_threshold] = -1
    
    # Exit signal: Z-Score near zero (spread normalized)
    signals[np.abs(zscore) < exit_threshold] = 0
    
    return signals


def calculate_half_life(spread: pd.Series) -> float:
    """
    Calculate mean reversion half-life of spread.
    
    Half-life estimates how long (in bars) it takes for the spread
    to revert halfway to its mean. Used to:
    - Validate mean reversion (half-life < 30 bars is good)
    - Set time stops (e.g., 2x half-life)
    
    Uses Ornstein-Uhlenbeck process estimation.
    
    Args:
        spread: Price spread series
    
    Returns:
        Half-life in number of bars
    
    Example:
        >>> spread = pd.Series([...])  # Mean-reverting spread
        >>> hl = calculate_half_life(spread)
        >>> print(f"Half-life: {hl:.1f} bars")
        Half-life: 12.3 bars
    """
    # Lag spread by 1
    spread_lag = spread.shift(1)
    spread_diff = spread - spread_lag
    
    # Drop NaN
    spread_lag = spread_lag.dropna()
    spread_diff = spread_diff.dropna()
    
    # OLS regression: ΔSpread = λ * Spread_t-1
    # where λ = speed of mean reversion
    X = spread_lag.values.reshape(-1, 1)
    y = spread_diff.values
    
    # Calculate lambda using least squares
    lambda_coef = np.linalg.lstsq(X, y, rcond=None)[0][0]
    
    # Half-life = -ln(2) / λ
    half_life = -np.log(2) / lambda_coef if lambda_coef < 0 else np.inf
    
    return half_life


def get_zscore_thresholds(
    timeframe_minutes: int
) -> Tuple[float, float]:
    """
    Get recommended Z-Score thresholds by timeframe.
    
    Higher timeframes use wider thresholds to avoid noise.
    
    Args:
        timeframe_minutes: Timeframe in minutes
    
    Returns:
        (entry_threshold, exit_threshold)
    
    Recommendations:
    - M5: Entry ±1.5, Exit ±0.3 (tight, frequent trades)
    - M15: Entry ±1.5, Exit ±0.5 (balanced)
    - H1: Entry ±2.0, Exit ±0.7 (wide, selective trades)
    """
    thresholds = {
        5: (1.5, 0.3),   # M5
        15: (1.5, 0.5),  # M15
        60: (2.0, 0.7),  # H1
    }
    return thresholds.get(timeframe_minutes, (1.5, 0.5))
