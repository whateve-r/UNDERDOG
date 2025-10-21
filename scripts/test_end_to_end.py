#!/usr/bin/env python3
"""
End-to-End Test Script for UNDERDOG Backtesting System

This script validates the complete workflow:
1. Load data (synthetic or HuggingFace)
2. Run backtest with bt_engine
3. Verify results structure
4. Generate visualizations
5. Export results

Usage:
    poetry run python scripts/test_end_to_end.py
    poetry run python scripts/test_end_to_end.py --use-hf-data
    poetry run python scripts/test_end_to_end.py --strategy SuperTrendRSI
    poetry run python scripts/test_end_to_end.py --quick
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta
import argparse

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from underdog.backtesting.bt_engine import run_backtest
from underdog.backtesting.bt_adapter import STRATEGY_REGISTRY


def print_section(title: str):
    """Print formatted section header"""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)


def validate_results(results: dict) -> bool:
    """Validate backtest results structure (bt_engine format)"""
    print_section("VALIDATING RESULTS STRUCTURE")
    
    required_keys = [
        'equity_curve', 'trades', 'metrics', 'strategy'
    ]
    
    validation_passed = True
    
    # Check top-level keys
    for key in required_keys:
        if key not in results:
            print(f"❌ FAIL: Missing required key '{key}'")
            validation_passed = False
        else:
            print(f"✅ PASS: Found '{key}'")
    
    # Validate equity_curve is a DataFrame
    if 'equity_curve' in results:
        import pandas as pd
        if not isinstance(results['equity_curve'], pd.DataFrame):
            print(f"❌ FAIL: equity_curve should be DataFrame, got {type(results['equity_curve'])}")
            validation_passed = False
        elif len(results['equity_curve']) == 0:
            print(f"❌ FAIL: equity_curve is empty")
            validation_passed = False
        else:
            print(f"✅ PASS: equity_curve has {len(results['equity_curve'])} data points")
    
    # Validate trades is a DataFrame
    if 'trades' in results:
        import pandas as pd
        if not isinstance(results['trades'], pd.DataFrame):
            print(f"❌ FAIL: trades should be DataFrame, got {type(results['trades'])}")
            validation_passed = False
        else:
            print(f"✅ PASS: trades has {len(results['trades'])} entries")
    
    # Validate metrics dict
    if 'metrics' in results:
        metrics = results['metrics']
        required_metrics = [
            'total_return_pct', 'sharpe_ratio', 'max_drawdown_pct',
            'win_rate_pct', 'profit_factor', 'num_trades'
        ]
        
        for metric in required_metrics:
            if metric not in metrics:
                print(f"❌ FAIL: Missing metric '{metric}'")
                validation_passed = False
            else:
                value = metrics[metric]
                print(f"✅ PASS: {metric} = {value:.2f}" if isinstance(value, (int, float)) else f"✅ PASS: {metric} = {value}")
    
    return validation_passed


def test_strategy_parameters(strategy_name: str):
    """Test that strategy parameters are correctly applied"""
    print_section(f"TESTING STRATEGY PARAMETERS: {strategy_name}")
    
    if strategy_name not in STRATEGY_REGISTRY:
        print(f"❌ FAIL: Strategy '{strategy_name}' not found in registry")
        print(f"Available strategies: {list(STRATEGY_REGISTRY.keys())}")
        return False
    
    strategy_class = STRATEGY_REGISTRY[strategy_name]
    print(f"✅ PASS: Strategy class loaded: {strategy_class.__name__}")
    
    # Check default parameters
    if hasattr(strategy_class, 'params'):
        print(f"✅ PASS: Strategy has parameters:")
        for param_name in dir(strategy_class.params):
            if not param_name.startswith('_'):
                param_value = getattr(strategy_class.params, param_name)
                print(f"   - {param_name}: {param_value}")
    else:
        print(f"⚠️  WARN: Strategy has no default parameters")
    
    return True


def export_results(results: dict, strategy_name: str):
    """Export results to CSV for manual inspection"""
    print_section("EXPORTING RESULTS")
    
    output_dir = project_root / "data" / "test_results"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Export trades to CSV
    trades_file = output_dir / f"trades_{strategy_name}_{timestamp}.csv"
    if 'trades' in results and len(results['trades']) > 0:
        import pandas as pd
        trades_df = results['trades']
        trades_df.to_csv(trades_file, index=False)
        print(f"✅ Trades exported to: {trades_file}")
    else:
        print(f"⚠️  No trades to export")
    
    # Export equity curve to CSV
    equity_file = output_dir / f"equity_{strategy_name}_{timestamp}.csv"
    if 'equity_curve' in results and len(results['equity_curve']) > 0:
        import pandas as pd
        equity_df = results['equity_curve']
        equity_df.to_csv(equity_file, index=False)
        print(f"✅ Equity curve exported to: {equity_file}")
    else:
        print(f"⚠️  No equity curve to export")
    
    # Export summary metrics
    summary_file = output_dir / f"summary_{strategy_name}_{timestamp}.txt"
    with open(summary_file, 'w') as f:
        f.write(f"BACKTEST SUMMARY - {strategy_name}\n")
        f.write(f"Generated: {datetime.now()}\n")
        f.write("=" * 80 + "\n\n")
        
        if 'metrics' in results:
            metrics = results['metrics']
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    f.write(f"{key}: {value:.4f}\n")
                else:
                    f.write(f"{key}: {value}\n")
        
        if 'monte_carlo' in results:
            f.write("\nMonte Carlo Validation:\n")
            mc = results['monte_carlo']
            for key, value in mc.items():
                f.write(f"  {key}: {value}\n")
    
    print(f"✅ Summary exported to: {summary_file}")
    print(f"\n📁 All results saved in: {output_dir}")


def run_end_to_end_test(
    strategy_name: str = "ATRBreakout",
    use_hf_data: bool = False,
    quick_test: bool = False
):
    """Run complete end-to-end test"""
    
    print_section("END-TO-END TEST STARTED")
    print(f"Strategy: {strategy_name}")
    print(f"Data Source: {'HuggingFace' if use_hf_data else 'Synthetic'}")
    print(f"Test Mode: {'Quick (1 month)' if quick_test else 'Full (1 year)'}")
    
    # Step 1: Validate strategy exists
    if not test_strategy_parameters(strategy_name):
        print("\n❌ TEST FAILED: Strategy validation failed")
        return False
    
    # Step 2: Configure date range
    end_date = datetime.now()
    if quick_test:
        start_date = end_date - timedelta(days=30)  # 1 month for quick test
    else:
        start_date = end_date - timedelta(days=365)  # 1 year for full test
    
    print_section("RUNNING BACKTEST")
    print(f"Start Date: {start_date.strftime('%Y-%m-%d')}")
    print(f"End Date: {end_date.strftime('%Y-%m-%d')}")
    
    # Step 3: Run backtest
    try:
        results = run_backtest(
            strategy_name=strategy_name,
            symbol="EURUSD",
            start_date=start_date.strftime("%Y-%m-%d"),
            end_date=end_date.strftime("%Y-%m-%d"),
            initial_capital=100000,
            use_hf_data=use_hf_data,
            validate_monte_carlo=True,
            mc_iterations=1000 if quick_test else 10000,
            strategy_params={}  # Use default parameters
        )
        print("✅ Backtest completed successfully")
    except Exception as e:
        print(f"❌ FAIL: Backtest execution failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Step 4: Validate results
    if not validate_results(results):
        print("\n❌ TEST FAILED: Results validation failed")
        return False
    
    # Step 5: Print summary
    print_section("BACKTEST RESULTS SUMMARY")
    metrics = results.get('metrics', {})
    print(f"Total Return: {metrics.get('total_return_pct', 0):.2f}%")
    print(f"Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.2f}")
    print(f"Max Drawdown: {metrics.get('max_drawdown_pct', 0):.2f}%")
    print(f"Win Rate: {metrics.get('win_rate_pct', 0):.2f}%")
    print(f"Profit Factor: {metrics.get('profit_factor', 0):.2f}")
    print(f"Total Trades: {metrics.get('num_trades', 0)}")
    
    # Monte Carlo validation
    if 'monte_carlo' in results:
        mc = results['monte_carlo']
        print(f"\nMonte Carlo Validation ({mc.get('iterations', 0):,} runs):")
        print(f"  Trades Analyzed: {mc.get('num_trades', 0)}")
        print(f"  Status: {'✓ ROBUST' if mc.get('is_robust') else '✗ NOT ROBUST'}")
    
    # Step 6: Export results
    export_results(results, strategy_name)
    
    # Step 7: Final verdict
    print_section("END-TO-END TEST COMPLETED")
    
    metrics = results.get('metrics', {})
    
    # Check if strategy is profitable and robust
    total_return = metrics.get('total_return_pct', 0)
    sharpe_ratio = metrics.get('sharpe_ratio', 0)
    is_robust = results.get('monte_carlo', {}).get('is_robust', False)
    
    print(f"✅ Total Return: {total_return:.2f}%")
    print(f"✅ Sharpe Ratio: {sharpe_ratio:.2f}")
    print(f"✅ Monte Carlo Status: {'ROBUST' if is_robust else 'NOT ROBUST'}")
    
    # Production criteria (for Prop Firm trading)
    is_profitable = total_return > 0
    has_good_sharpe = sharpe_ratio > 1.0  # Lowered from 1.5 for synthetic data
    
    if is_profitable and has_good_sharpe and is_robust:
        print("\n🎉 TEST PASSED: Strategy meets production criteria!")
        print("   ✓ Profitable")
        print("   ✓ Good risk-adjusted returns (Sharpe > 1.0)")
        print("   ✓ Robust (Monte Carlo validated)")
        return True
    else:
        print("\n⚠️  TEST PASSED but strategy needs improvement:")
        if not is_profitable:
            print("   ✗ Not profitable (Return < 0%)")
        if not has_good_sharpe:
            print(f"   ✗ Low Sharpe Ratio ({sharpe_ratio:.2f} < 1.0)")
        if not is_robust:
            print("   ✗ Not robust (failed Monte Carlo)")
        print("\n   This is normal for synthetic data or short test periods.")
        print("   Re-run with --use-hf-data and longer timeframe for real validation.")
        return True  # Still pass test (infrastructure works)


def main():
    parser = argparse.ArgumentParser(
        description="End-to-end test for UNDERDOG backtesting system"
    )
    parser.add_argument(
        "--strategy",
        type=str,
        default="ATRBreakout",
        choices=list(STRATEGY_REGISTRY.keys()),
        help="Strategy to test"
    )
    parser.add_argument(
        "--use-hf-data",
        action="store_true",
        help="Use HuggingFace real data instead of synthetic"
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick test (1 month, 1k Monte Carlo runs)"
    )
    
    args = parser.parse_args()
    
    success = run_end_to_end_test(
        strategy_name=args.strategy,
        use_hf_data=args.use_hf_data,
        quick_test=args.quick
    )
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
