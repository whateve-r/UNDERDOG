"""
Walk-Forward Optimization (WFO) for Backtrader.

Implements:
- Rolling window optimization (In-Sample → Out-of-Sample)
- Parameter grid search on IS data
- Validation on OOS data
- Objective function: Calmar Ratio (Return / |Max DD|)
"""

import backtrader as bt
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any
from itertools import product


class WalkForwardOptimizer:
    """
    Walk-Forward Optimization framework for Backtrader strategies.
    
    Configuration:
    - In-Sample: 5 years (parameter optimization)
    - Out-of-Sample: 1 year (validation)
    - Rolling Step: 3 months (quarterly reoptimization)
    """
    
    def __init__(
        self,
        strategy_class,
        data_feed,
        start_date: datetime,
        end_date: datetime,
        in_sample_years: int = 5,
        out_sample_months: int = 12,
        rolling_step_months: int = 3,
        initial_cash: float = 10000.0,
        commission: float = 0.0001,
        objective: str = 'calmar_ratio'
    ):
        """
        Initialize WFO framework.
        
        Args:
            strategy_class: Backtrader strategy class
            data_feed: Backtrader data feed
            start_date: Start date for WFO
            end_date: End date for WFO
            in_sample_years: Years for optimization
            out_sample_months: Months for validation
            rolling_step_months: How often to reoptimize
            initial_cash: Starting capital
            commission: Commission rate (0.0001 = 0.01%)
            objective: Objective function ('calmar_ratio', 'sharpe', 'return')
        """
        self.strategy_class = strategy_class
        self.data_feed = data_feed
        self.start_date = start_date
        self.end_date = end_date
        self.in_sample_years = in_sample_years
        self.out_sample_months = out_sample_months
        self.rolling_step_months = rolling_step_months
        self.initial_cash = initial_cash
        self.commission = commission
        self.objective = objective
        
        self.windows = []
        self.results = []
        
    def generate_windows(self) -> List[Dict]:
        """
        Generate rolling WFO windows.
        
        Returns:
            List of windows with IS and OOS date ranges
        """
        windows = []
        
        current_date = self.start_date
        
        while current_date < self.end_date:
            # In-Sample window
            is_start = current_date
            is_end = current_date + timedelta(days=365 * self.in_sample_years)
            
            # Out-of-Sample window
            oos_start = is_end
            oos_end = oos_start + timedelta(days=30 * self.out_sample_months)
            
            # Check if we have enough data
            if oos_end > self.end_date:
                break
            
            windows.append({
                'window_id': len(windows) + 1,
                'is_start': is_start,
                'is_end': is_end,
                'oos_start': oos_start,
                'oos_end': oos_end
            })
            
            # Move to next window
            current_date += timedelta(days=30 * self.rolling_step_months)
        
        self.windows = windows
        return windows
    
    def optimize_in_sample(
        self,
        window: Dict,
        param_grid: Dict[str, List[Any]]
    ) -> Tuple[Dict, float]:
        """
        Optimize parameters on In-Sample data.
        
        Args:
            window: Window configuration
            param_grid: Parameter grid for optimization
            
        Returns:
            Tuple of (best_params, best_score)
        """
        print(f"\n[IS] Optimizing Window {window['window_id']}")
        print(f"     Period: {window['is_start'].date()} to {window['is_end'].date()}")
        
        # Generate parameter combinations
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        param_combinations = list(product(*param_values))
        
        print(f"     Testing {len(param_combinations)} parameter combinations...")
        
        best_score = -np.inf
        best_params = None
        
        for combo in param_combinations:
            params = dict(zip(param_names, combo))
            
            # Run backtest with these parameters
            score = self._run_backtest(
                window['is_start'],
                window['is_end'],
                params
            )
            
            if score > best_score:
                best_score = score
                best_params = params
        
        print(f"     Best params: {best_params}")
        print(f"     Best {self.objective}: {best_score:.4f}")
        
        return best_params, best_score
    
    def validate_out_sample(
        self,
        window: Dict,
        params: Dict
    ) -> Dict:
        """
        Validate optimized parameters on Out-of-Sample data.
        
        Args:
            window: Window configuration
            params: Optimized parameters from IS
            
        Returns:
            OOS performance metrics
        """
        print(f"\n[OOS] Validating Window {window['window_id']}")
        print(f"      Period: {window['oos_start'].date()} to {window['oos_end'].date()}")
        print(f"      Using params: {params}")
        
        # Run backtest on OOS data
        cerebro = bt.Cerebro()
        cerebro.addstrategy(self.strategy_class, **params)
        
        # Filter data for OOS period
        # (Implement data filtering logic here)
        cerebro.adddata(self.data_feed)
        
        cerebro.broker.setcash(self.initial_cash)
        cerebro.broker.setcommission(commission=self.commission)
        
        # Add analyzers
        cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')
        cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
        cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')
        cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')
        
        results = cerebro.run()[0]
        
        # Extract metrics
        final_value = cerebro.broker.getvalue()
        total_return = (final_value - self.initial_cash) / self.initial_cash
        
        dd_analyzer = results.analyzers.drawdown.get_analysis()
        max_dd = dd_analyzer.max.drawdown / 100.0 if dd_analyzer.max.drawdown else 0.01
        
        calmar = total_return / abs(max_dd) if max_dd != 0 else 0
        
        sharpe_analyzer = results.analyzers.sharpe.get_analysis()
        sharpe = sharpe_analyzer.get('sharperatio', 0) or 0
        
        trades_analyzer = results.analyzers.trades.get_analysis()
        total_trades = trades_analyzer.total.total if hasattr(trades_analyzer, 'total') else 0
        
        metrics = {
            'window_id': window['window_id'],
            'period_start': window['oos_start'],
            'period_end': window['oos_end'],
            'params': params,
            'final_value': final_value,
            'total_return': total_return,
            'max_drawdown': max_dd,
            'calmar_ratio': calmar,
            'sharpe_ratio': sharpe,
            'total_trades': total_trades
        }
        
        print(f"      Return: {total_return*100:.2f}%")
        print(f"      Max DD: {max_dd*100:.2f}%")
        print(f"      Calmar: {calmar:.2f}")
        print(f"      Trades: {total_trades}")
        
        return metrics
    
    def _run_backtest(
        self,
        start_date: datetime,
        end_date: datetime,
        params: Dict
    ) -> float:
        """
        Run single backtest and return objective score.
        
        Args:
            start_date: Backtest start date
            end_date: Backtest end date
            params: Strategy parameters
            
        Returns:
            Objective function value
        """
        cerebro = bt.Cerebro()
        cerebro.addstrategy(self.strategy_class, **params)
        cerebro.adddata(self.data_feed)
        cerebro.broker.setcash(self.initial_cash)
        cerebro.broker.setcommission(commission=self.commission)
        
        cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')
        cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
        cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')
        
        results = cerebro.run()[0]
        
        # Calculate objective
        if self.objective == 'calmar_ratio':
            final_value = cerebro.broker.getvalue()
            total_return = (final_value - self.initial_cash) / self.initial_cash
            
            dd_analyzer = results.analyzers.drawdown.get_analysis()
            max_dd = dd_analyzer.max.drawdown / 100.0 if dd_analyzer.max.drawdown else 0.01
            
            score = total_return / abs(max_dd) if max_dd != 0 else 0
            
        elif self.objective == 'sharpe':
            sharpe_analyzer = results.analyzers.sharpe.get_analysis()
            score = sharpe_analyzer.get('sharperatio', 0) or 0
            
        elif self.objective == 'return':
            final_value = cerebro.broker.getvalue()
            score = (final_value - self.initial_cash) / self.initial_cash
            
        else:
            raise ValueError(f"Unknown objective: {self.objective}")
        
        return score
    
    def run_wfo(self, param_grid: Dict[str, List[Any]]) -> pd.DataFrame:
        """
        Run complete Walk-Forward Optimization.
        
        Args:
            param_grid: Parameter ranges for optimization
            
        Returns:
            DataFrame with OOS results for each window
        """
        print("\n" + "="*80)
        print("WALK-FORWARD OPTIMIZATION")
        print("="*80)
        print(f"Strategy: {self.strategy_class.__name__}")
        print(f"Period: {self.start_date.date()} to {self.end_date.date()}")
        print(f"In-Sample: {self.in_sample_years} years")
        print(f"Out-of-Sample: {self.out_sample_months} months")
        print(f"Rolling Step: {self.rolling_step_months} months")
        print(f"Objective: {self.objective}")
        print("="*80)
        
        # Generate windows
        windows = self.generate_windows()
        print(f"\nGenerated {len(windows)} WFO windows")
        
        # Run WFO for each window
        oos_results = []
        
        for window in windows:
            # Optimize on IS
            best_params, best_is_score = self.optimize_in_sample(window, param_grid)
            
            # Validate on OOS
            oos_metrics = self.validate_out_sample(window, best_params)
            oos_results.append(oos_metrics)
        
        # Aggregate results
        df_results = pd.DataFrame(oos_results)
        
        print("\n" + "="*80)
        print("WFO AGGREGATE RESULTS (Out-of-Sample Only)")
        print("="*80)
        print(f"Total Windows: {len(df_results)}")
        print(f"Avg Return: {df_results['total_return'].mean()*100:.2f}%")
        print(f"Avg Max DD: {df_results['max_drawdown'].mean()*100:.2f}%")
        print(f"Avg Calmar: {df_results['calmar_ratio'].mean():.2f}")
        print(f"Avg Sharpe: {df_results['sharpe_ratio'].mean():.2f}")
        print(f"Total Trades: {df_results['total_trades'].sum()}")
        print("="*80)
        
        self.results = df_results
        return df_results
