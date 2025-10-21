"""
Monte Carlo validation for backtesting robustness.

Detects "lucky" backtests by shuffling trade order and comparing
original equity curve to simulated distributions.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple
from scipy.stats import percentileofscore


def monte_carlo_shuffle(
    trades: List[Dict],
    iterations: int = 10000,
    confidence_level: float = 0.05
) -> Dict:
    """
    Validate backtest robustness via trade shuffling.
    
    PRINCIPLE: If strategy is robust, original equity curve should
    be in upper percentiles of shuffled simulations. If original
    performance is < 5th percentile → "lucky backtest", not robust.
    
    Args:
        trades: List of trade dicts with 'pnl' key
        iterations: Number of Monte Carlo shuffles (default 10,000)
        confidence_level: Confidence threshold (default 0.05 = 5%)
        
    Returns:
        Dict with validation results
    """
    print("\n" + "="*80)
    print("MONTE CARLO VALIDATION")
    print("="*80)
    print(f"Trades: {len(trades)}")
    print(f"Iterations: {iterations:,}")
    print(f"Confidence Level: {confidence_level*100:.0f}%")
    
    if len(trades) < 10:
        print("\n⚠️ WARNING: Too few trades (<10) for reliable MC validation")
        return {
            'valid': False,
            'reason': 'Insufficient trades'
        }
    
    # Extract PnL from trades
    pnls = [t['pnl'] for t in trades]
    
    # Calculate original equity curve
    original_equity = np.cumsum(pnls)
    original_final = original_equity[-1]
    
    print(f"\nOriginal Performance:")
    print(f"  Final P&L: ${original_final:,.2f}")
    print(f"  Win Rate: {sum(p > 0 for p in pnls) / len(pnls):.1%}")
    
    # Run Monte Carlo simulations
    print(f"\nRunning {iterations:,} simulations...")
    simulated_finals = []
    
    for i in range(iterations):
        # Shuffle trade order
        shuffled_pnls = np.random.permutation(pnls)
        shuffled_equity = np.cumsum(shuffled_pnls)
        simulated_finals.append(shuffled_equity[-1])
        
        if (i + 1) % 1000 == 0:
            print(f"  Progress: {i+1:,} / {iterations:,}")
    
    simulated_finals = np.array(simulated_finals)
    
    # Calculate percentile of original result
    percentile = percentileofscore(simulated_finals, original_final)
    
    # Calculate statistics
    sim_mean = simulated_finals.mean()
    sim_std = simulated_finals.std()
    sim_min = simulated_finals.min()
    sim_max = simulated_finals.max()
    sim_5th = np.percentile(simulated_finals, 5)
    sim_95th = np.percentile(simulated_finals, 95)
    
    print("\n" + "="*80)
    print("SIMULATION RESULTS")
    print("="*80)
    print(f"Simulated Mean:   ${sim_mean:,.2f}")
    print(f"Simulated Std:    ${sim_std:,.2f}")
    print(f"Simulated Range:  ${sim_min:,.2f} to ${sim_max:,.2f}")
    print(f"5th Percentile:   ${sim_5th:,.2f}")
    print(f"95th Percentile:  ${sim_95th:,.2f}")
    print(f"\nOriginal Result:  ${original_final:,.2f}")
    print(f"Percentile Rank:  {percentile:.1f}%")
    
    # Validation decision
    is_robust = percentile >= (confidence_level * 100)
    
    print("\n" + "="*80)
    if is_robust:
        print("✓ ROBUST: Strategy passes Monte Carlo validation")
        print(f"  Original result is above {confidence_level*100:.0f}th percentile")
        print("  Performance is NOT due to lucky trade sequence")
    else:
        print("✗ NOT ROBUST: Strategy fails Monte Carlo validation")
        print(f"  Original result is below {confidence_level*100:.0f}th percentile")
        print("  ⚠️ WARNING: Performance may be due to lucky trade sequence")
        print("  Recommendation: Do NOT trade this strategy")
    print("="*80)
    
    return {
        'valid': True,
        'is_robust': is_robust,
        'original_final': original_final,
        'percentile': percentile,
        'simulated_mean': sim_mean,
        'simulated_std': sim_std,
        'simulated_5th': sim_5th,
        'simulated_95th': sim_95th,
        'confidence_level': confidence_level,
        'iterations': iterations,
        'num_trades': len(trades)
    }


def analyze_equity_curve_stability(
    trades: List[Dict],
    iterations: int = 1000
) -> Dict:
    """
    Analyze equity curve stability via multiple shuffles.
    
    Generates distribution of equity curves to visualize robustness.
    
    Args:
        trades: List of trade dicts with 'pnl' key
        iterations: Number of simulations
        
    Returns:
        Dict with equity curve statistics
    """
    pnls = [t['pnl'] for t in trades]
    original_equity = np.cumsum(pnls)
    
    # Simulate equity curves
    equity_curves = []
    for _ in range(iterations):
        shuffled_pnls = np.random.permutation(pnls)
        equity_curve = np.cumsum(shuffled_pnls)
        equity_curves.append(equity_curve)
    
    equity_curves = np.array(equity_curves)
    
    # Calculate percentile bands
    equity_5th = np.percentile(equity_curves, 5, axis=0)
    equity_25th = np.percentile(equity_curves, 25, axis=0)
    equity_50th = np.percentile(equity_curves, 50, axis=0)
    equity_75th = np.percentile(equity_curves, 75, axis=0)
    equity_95th = np.percentile(equity_curves, 95, axis=0)
    
    return {
        'original': original_equity,
        'percentile_5': equity_5th,
        'percentile_25': equity_25th,
        'percentile_50': equity_50th,
        'percentile_75': equity_75th,
        'percentile_95': equity_95th,
        'num_simulations': iterations
    }


def validate_backtest(
    trades: List[Dict],
    mc_iterations: int = 10000,
    confidence: float = 0.05
) -> bool:
    """
    Complete backtest validation workflow.
    
    Args:
        trades: List of trade dicts with 'pnl' key
        mc_iterations: Monte Carlo iterations
        confidence: Confidence threshold
        
    Returns:
        True if backtest passes validation, False otherwise
    """
    result = monte_carlo_shuffle(trades, mc_iterations, confidence)
    
    if not result['valid']:
        print(f"\n⚠️ Validation skipped: {result['reason']}")
        return False
    
    return result['is_robust']


# Example usage
if __name__ == '__main__':
    # Example: Simulate trades
    np.random.seed(42)
    
    # Good strategy (consistent wins)
    print("\nExample 1: Consistent Strategy")
    good_trades = [
        {'pnl': np.random.normal(100, 50)} for _ in range(50)
    ]
    validate_backtest(good_trades, mc_iterations=10000)
    
    # Lucky strategy (one big win)
    print("\n\nExample 2: Lucky Strategy (One Big Win)")
    lucky_trades = [{'pnl': -10} for _ in range(49)]
    lucky_trades.append({'pnl': 1000})  # Lucky big win
    validate_backtest(lucky_trades, mc_iterations=10000)
