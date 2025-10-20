"""
Walk-Forward Optimization (WFO) Pipeline
Automated IS/OS segmentation with parameter optimization and comprehensive metrics.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable, Tuple, Any
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from pathlib import Path
import json
import pickle
import warnings


@dataclass
class WFOConfig:
    """Walk-Forward Optimization configuration"""
    # Time windows
    in_sample_days: int = 252  # 1 year IS
    out_sample_days: int = 63  # 3 months OS
    step_days: int = 63  # Step forward 3 months
    
    # Optimization
    max_iterations: int = 100
    optimization_metric: str = "sharpe_ratio"  # sharpe_ratio, profit_factor, sortino, cagr
    
    # Thresholds
    min_trades_is: int = 30  # Minimum trades in IS period
    min_trades_os: int = 10  # Minimum trades in OS period
    min_sharpe_threshold: float = 0.5  # Minimum acceptable Sharpe
    
    # Output
    save_results: bool = True
    results_dir: str = "results/wfo"


@dataclass
class WFOFold:
    """Single Walk-Forward fold"""
    fold_id: int
    is_start: datetime
    is_end: datetime
    os_start: datetime
    os_end: datetime
    
    # Optimization results
    best_params: Optional[Dict[str, Any]] = None
    is_metrics: Optional[Dict[str, float]] = None
    os_metrics: Optional[Dict[str, float]] = None
    
    # Trade details
    is_trades: List[Dict] = field(default_factory=list)
    os_trades: List[Dict] = field(default_factory=list)
    
    # Performance
    optimization_time: float = 0.0
    backtest_time: float = 0.0


@dataclass
class WFOResult:
    """Complete WFO result"""
    config: WFOConfig
    folds: List[WFOFold]
    
    # Aggregate metrics
    avg_is_sharpe: float = 0.0
    avg_os_sharpe: float = 0.0
    sharpe_degradation: float = 0.0
    
    passing_folds: int = 0
    total_folds: int = 0
    pass_rate: float = 0.0
    
    # Summary statistics
    summary_metrics: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization"""
        return {
            'config': {
                'in_sample_days': self.config.in_sample_days,
                'out_sample_days': self.config.out_sample_days,
                'step_days': self.config.step_days,
                'optimization_metric': self.config.optimization_metric
            },
            'folds': [
                {
                    'fold_id': f.fold_id,
                    'is_start': f.is_start.isoformat(),
                    'is_end': f.is_end.isoformat(),
                    'os_start': f.os_start.isoformat(),
                    'os_end': f.os_end.isoformat(),
                    'best_params': f.best_params,
                    'is_metrics': f.is_metrics,
                    'os_metrics': f.os_metrics,
                    'is_trades_count': len(f.is_trades),
                    'os_trades_count': len(f.os_trades)
                }
                for f in self.folds
            ],
            'aggregate_metrics': {
                'avg_is_sharpe': self.avg_is_sharpe,
                'avg_os_sharpe': self.avg_os_sharpe,
                'sharpe_degradation': self.sharpe_degradation,
                'passing_folds': self.passing_folds,
                'total_folds': self.total_folds,
                'pass_rate': self.pass_rate
            },
            'summary_metrics': self.summary_metrics
        }


class WalkForwardOptimizer:
    """
    Walk-Forward Optimization engine for strategy validation.
    
    Implements anchored or rolling window WFO with automated parameter optimization
    and comprehensive out-of-sample validation.
    """
    
    def __init__(self, config: Optional[WFOConfig] = None):
        """
        Initialize WFO engine.
        
        Args:
            config: WFO configuration
        """
        self.config = config or WFOConfig()
        
        # Results storage
        self.results: Optional[WFOResult] = None
        
        # MEJORA CIENTÍFICA: Purging & Embargo configuration
        self.embargo_pct = 0.01  # 1% del train set como gap (default)
    
    def purge_and_embargo(self, 
                          train_data: pd.DataFrame, 
                          val_data: pd.DataFrame,
                          embargo_pct: Optional[float] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Purging & Embargo para eliminar data leakage en time-series CV.
        
        Literatura: "Advances in Financial Machine Learning" - López de Prado
        
        Problema: En time-series, features usan rolling windows que pueden incluir
                 datos del validation set, causando leakage.
        
        Solución:
        - Purging: Remover del val set observaciones cuyo label se calculó con datos de train
        - Embargo: Añadir gap temporal entre train y val
        
        Args:
            train_data: Conjunto de entrenamiento
            val_data: Conjunto de validación
            embargo_pct: Porcentaje del train set para usar como gap (default: self.embargo_pct)
        
        Returns:
            Tuple de (train_clean, val_clean) sin overlap
        """
        if embargo_pct is None:
            embargo_pct = self.embargo_pct
        
        # Calcular número de periodos de embargo
        embargo_periods = int(len(train_data) * embargo_pct)
        
        if embargo_periods == 0:
            warnings.warn("Embargo periods = 0. Considereusar embargo_pct mayor.")
            return train_data, val_data
        
        # Purging: Remover últimos embargo_periods del train set
        train_clean = train_data.iloc[:-embargo_periods] if embargo_periods > 0 else train_data
        
        # Embargo: Remover primeros embargo_periods del val set
        val_clean = val_data.iloc[embargo_periods:] if embargo_periods < len(val_data) else val_data
        
        removed_train = len(train_data) - len(train_clean)
        removed_val = len(val_data) - len(val_clean)
        
        if removed_train > 0 or removed_val > 0:
            print(f"[Purging & Embargo] Train: {len(train_data)} → {len(train_clean)} "
                  f"(-{removed_train}), Val: {len(val_data)} → {len(val_clean)} (-{removed_val})")
        
        return train_clean, val_clean
    
    def run(self,
            data: pd.DataFrame,
            strategy_func: Callable,
            param_grid: Dict[str, List],
            start_date: Optional[datetime] = None,
            end_date: Optional[datetime] = None,
            anchored: bool = False) -> WFOResult:
        """
        Run Walk-Forward Optimization.
        
        Args:
            data: Historical price data (OHLCV) with datetime index
            strategy_func: Strategy function(data, **params) -> trades_df
            param_grid: Parameter grid for optimization {param_name: [values]}
            start_date: WFO start date (default: first date in data)
            end_date: WFO end date (default: last date in data)
            anchored: Use anchored window (True) or rolling window (False)
        
        Returns:
            WFOResult with complete optimization results
        """
        # Validate data
        if not isinstance(data.index, pd.DatetimeIndex):
            raise ValueError("Data must have DatetimeIndex")
        
        # Set date range
        start_date = start_date or data.index[0]
        end_date = end_date or data.index[-1]
        
        print(f"[WFO] Starting Walk-Forward Optimization")
        print(f"[WFO] Date range: {start_date.date()} to {end_date.date()}")
        print(f"[WFO] Window: IS={self.config.in_sample_days}d, OS={self.config.out_sample_days}d")
        print(f"[WFO] Anchored: {anchored}")
        
        # Generate folds
        folds = self._generate_folds(start_date, end_date, anchored)
        print(f"[WFO] Generated {len(folds)} folds")
        
        # Run optimization for each fold
        completed_folds = []
        for i, fold in enumerate(folds):
            print(f"\n[WFO] Processing Fold {fold.fold_id}/{len(folds)}")
            
            try:
                # Extract IS and OS data
                is_data = data[(data.index >= fold.is_start) & (data.index <= fold.is_end)]
                os_data = data[(data.index >= fold.os_start) & (data.index <= fold.os_end)]
                
                # MEJORA CIENTÍFICA: Aplicar Purging & Embargo
                is_data_clean, os_data_clean = self.purge_and_embargo(is_data, os_data)
                
                if len(is_data_clean) < self.config.min_trades_is:
                    print(f"[WFO] Skipping fold {fold.fold_id}: Insufficient IS data after purging")
                    continue
                
                if len(os_data_clean) < self.config.min_trades_os:
                    print(f"[WFO] Skipping fold {fold.fold_id}: Insufficient OS data after purging")
                    continue
                
                # Optimize on IS (cleaned)
                import time
                opt_start = time.time()
                
                best_params, is_metrics = self._optimize_parameters(
                    is_data_clean, strategy_func, param_grid
                )
                
                fold.optimization_time = time.time() - opt_start
                fold.best_params = best_params
                fold.is_metrics = is_metrics
                
                print(f"[WFO] IS Optimization: {self.config.optimization_metric}={is_metrics.get(self.config.optimization_metric, 0):.3f}")
                print(f"[WFO] Best params: {best_params}")
                
                # Test on OS (cleaned)
                bt_start = time.time()
                os_trades = strategy_func(os_data_clean, **best_params)
                os_metrics = self._calculate_metrics(os_trades)
                
                fold.backtest_time = time.time() - bt_start
                fold.os_metrics = os_metrics
                fold.os_trades = os_trades.to_dict('records') if hasattr(os_trades, 'to_dict') else []
                
                print(f"[WFO] OS Performance: {self.config.optimization_metric}={os_metrics.get(self.config.optimization_metric, 0):.3f}")
                
                completed_folds.append(fold)
                
            except Exception as e:
                print(f"[WFO] Error in fold {fold.fold_id}: {e}")
                continue
        
        # Aggregate results
        self.results = self._aggregate_results(completed_folds)
        
        # Save results
        if self.config.save_results:
            self._save_results()
        
        return self.results
    
    def _generate_folds(self, start_date: datetime, end_date: datetime, anchored: bool) -> List[WFOFold]:
        """Generate WFO folds"""
        folds = []
        fold_id = 1
        
        current_start = start_date
        
        while True:
            # IS period
            is_start = current_start
            is_end = is_start + timedelta(days=self.config.in_sample_days)
            
            # OS period
            os_start = is_end + timedelta(days=1)
            os_end = os_start + timedelta(days=self.config.out_sample_days)
            
            # Check if we've reached the end
            if os_end > end_date:
                break
            
            fold = WFOFold(
                fold_id=fold_id,
                is_start=is_start,
                is_end=is_end,
                os_start=os_start,
                os_end=os_end
            )
            
            folds.append(fold)
            fold_id += 1
            
            # Move forward
            if anchored:
                # Anchored: keep IS start fixed, expand window
                current_start = start_date
            else:
                # Rolling: move window forward
                current_start += timedelta(days=self.config.step_days)
        
        return folds
    
    def _optimize_parameters(self,
                            data: pd.DataFrame,
                            strategy_func: Callable,
                            param_grid: Dict[str, List]) -> Tuple[Dict, Dict]:
        """
        Optimize strategy parameters on IS data.
        
        Args:
            data: IS data
            strategy_func: Strategy function
            param_grid: Parameter grid
        
        Returns:
            Tuple of (best_params, best_metrics)
        """
        # Generate parameter combinations
        param_combinations = self._generate_param_combinations(param_grid)
        
        best_score = -np.inf
        best_params = None
        best_metrics = None
        
        for params in param_combinations:
            try:
                # Run strategy with these parameters
                trades = strategy_func(data, **params)
                
                # Calculate metrics
                metrics = self._calculate_metrics(trades)
                
                # Get optimization score
                score = metrics.get(self.config.optimization_metric, -np.inf)
                
                if score > best_score:
                    best_score = score
                    best_params = params
                    best_metrics = metrics
                
            except Exception as e:
                # Skip parameter combinations that fail
                continue
        
        return best_params, best_metrics
    
    def _generate_param_combinations(self, param_grid: Dict[str, List]) -> List[Dict]:
        """Generate all parameter combinations from grid"""
        import itertools
        
        keys = list(param_grid.keys())
        values = list(param_grid.values())
        
        combinations = []
        for combo in itertools.product(*values):
            combinations.append(dict(zip(keys, combo)))
        
        return combinations
    
    def _calculate_metrics(self, trades: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate performance metrics for trades.
        
        Args:
            trades: DataFrame with columns: timestamp, pnl, return_pct
        
        Returns:
            Dict of metrics
        """
        if trades is None or len(trades) == 0:
            return {
                'sharpe_ratio': 0.0,
                'sortino_ratio': 0.0,
                'profit_factor': 0.0,
                'win_rate': 0.0,
                'avg_win': 0.0,
                'avg_loss': 0.0,
                'max_drawdown': 0.0,
                'total_return': 0.0,
                'num_trades': 0
            }
        
        # Extract returns
        if 'return_pct' in trades.columns:
            returns = trades['return_pct'].values
        elif 'pnl' in trades.columns:
            returns = trades['pnl'].values
        else:
            returns = np.zeros(len(trades))
        
        # Basic stats
        num_trades = len(trades)
        total_return = np.sum(returns)
        
        # Win/Loss stats
        wins = returns[returns > 0]
        losses = returns[returns < 0]
        
        win_rate = len(wins) / num_trades if num_trades > 0 else 0.0
        avg_win = np.mean(wins) if len(wins) > 0 else 0.0
        avg_loss = np.mean(np.abs(losses)) if len(losses) > 0 else 0.0
        
        # Profit factor
        gross_profit = np.sum(wins) if len(wins) > 0 else 0.0
        gross_loss = np.sum(np.abs(losses)) if len(losses) > 0 else 0.0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0.0
        
        # Sharpe ratio (annualized)
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        sharpe_ratio = (mean_return / std_return) * np.sqrt(252) if std_return > 0 else 0.0
        
        # Sortino ratio (annualized)
        downside_returns = returns[returns < 0]
        downside_std = np.std(downside_returns) if len(downside_returns) > 0 else 0.0
        sortino_ratio = (mean_return / downside_std) * np.sqrt(252) if downside_std > 0 else 0.0
        
        # Maximum drawdown
        cumulative = np.cumsum(returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = running_max - cumulative
        max_drawdown = np.max(drawdown) if len(drawdown) > 0 else 0.0
        
        return {
            'sharpe_ratio': float(sharpe_ratio),
            'sortino_ratio': float(sortino_ratio),
            'profit_factor': float(profit_factor),
            'win_rate': float(win_rate),
            'avg_win': float(avg_win),
            'avg_loss': float(avg_loss),
            'max_drawdown': float(max_drawdown),
            'total_return': float(total_return),
            'num_trades': int(num_trades),
            'cagr': float(total_return * 252 / num_trades) if num_trades > 0 else 0.0
        }
    
    def _aggregate_results(self, folds: List[WFOFold]) -> WFOResult:
        """Aggregate results across all folds"""
        result = WFOResult(config=self.config, folds=folds)
        
        if len(folds) == 0:
            return result
        
        # Calculate aggregate metrics
        is_sharpes = [f.is_metrics.get('sharpe_ratio', 0) for f in folds if f.is_metrics]
        os_sharpes = [f.os_metrics.get('sharpe_ratio', 0) for f in folds if f.os_metrics]
        
        result.avg_is_sharpe = np.mean(is_sharpes) if is_sharpes else 0.0
        result.avg_os_sharpe = np.mean(os_sharpes) if os_sharpes else 0.0
        result.sharpe_degradation = result.avg_is_sharpe - result.avg_os_sharpe
        
        # Count passing folds (OS Sharpe > threshold)
        result.passing_folds = sum(1 for s in os_sharpes if s >= self.config.min_sharpe_threshold)
        result.total_folds = len(folds)
        result.pass_rate = result.passing_folds / result.total_folds if result.total_folds > 0 else 0.0
        
        # Summary metrics
        result.summary_metrics = {
            'is_sharpe_mean': result.avg_is_sharpe,
            'is_sharpe_std': float(np.std(is_sharpes)) if is_sharpes else 0.0,
            'os_sharpe_mean': result.avg_os_sharpe,
            'os_sharpe_std': float(np.std(os_sharpes)) if os_sharpes else 0.0,
            'sharpe_degradation_pct': (result.sharpe_degradation / result.avg_is_sharpe * 100) if result.avg_is_sharpe > 0 else 0.0,
            'pass_rate_pct': result.pass_rate * 100
        }
        
        print(f"\n[WFO] ===== SUMMARY =====")
        print(f"[WFO] Total folds: {result.total_folds}")
        print(f"[WFO] Passing folds: {result.passing_folds} ({result.pass_rate*100:.1f}%)")
        print(f"[WFO] Avg IS Sharpe: {result.avg_is_sharpe:.3f}")
        print(f"[WFO] Avg OS Sharpe: {result.avg_os_sharpe:.3f}")
        print(f"[WFO] Degradation: {result.sharpe_degradation:.3f} ({result.summary_metrics['sharpe_degradation_pct']:.1f}%)")
        
        return result
    
    def _save_results(self) -> None:
        """Save WFO results to disk"""
        if self.results is None:
            return
        
        # Create results directory
        results_dir = Path(self.config.results_dir)
        results_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"wfo_results_{timestamp}.json"
        filepath = results_dir / filename
        
        # Save as JSON
        with open(filepath, 'w') as f:
            json.dump(self.results.to_dict(), f, indent=2)
        
        print(f"[WFO] Results saved to: {filepath}")
    
    def plot_results(self, show: bool = True, save_path: Optional[str] = None) -> None:
        """
        Plot WFO results.
        
        Args:
            show: Display plot
            save_path: Save plot to file
        """
        if self.results is None or len(self.results.folds) == 0:
            print("[WFO] No results to plot")
            return
        
        try:
            import matplotlib.pyplot as plt
            
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle('Walk-Forward Optimization Results', fontsize=16)
            
            # Extract data
            fold_ids = [f.fold_id for f in self.results.folds]
            is_sharpes = [f.is_metrics.get('sharpe_ratio', 0) for f in self.results.folds if f.is_metrics]
            os_sharpes = [f.os_metrics.get('sharpe_ratio', 0) for f in self.results.folds if f.os_metrics]
            
            # Plot 1: IS vs OS Sharpe
            axes[0, 0].plot(fold_ids, is_sharpes, 'b-o', label='In-Sample')
            axes[0, 0].plot(fold_ids, os_sharpes, 'r-o', label='Out-of-Sample')
            axes[0, 0].axhline(self.config.min_sharpe_threshold, color='g', linestyle='--', label='Threshold')
            axes[0, 0].set_xlabel('Fold')
            axes[0, 0].set_ylabel('Sharpe Ratio')
            axes[0, 0].set_title('IS vs OS Sharpe Ratio')
            axes[0, 0].legend()
            axes[0, 0].grid(True)
            
            # Plot 2: Degradation
            degradation = [is_s - os_s for is_s, os_s in zip(is_sharpes, os_sharpes)]
            axes[0, 1].bar(fold_ids, degradation, color='orange')
            axes[0, 1].axhline(0, color='black', linestyle='-', linewidth=0.5)
            axes[0, 1].set_xlabel('Fold')
            axes[0, 1].set_ylabel('Sharpe Degradation')
            axes[0, 1].set_title('Sharpe Ratio Degradation (IS - OS)')
            axes[0, 1].grid(True)
            
            # Plot 3: Distribution
            axes[1, 0].hist([is_sharpes, os_sharpes], bins=10, label=['IS', 'OS'], alpha=0.7)
            axes[1, 0].set_xlabel('Sharpe Ratio')
            axes[1, 0].set_ylabel('Frequency')
            axes[1, 0].set_title('Sharpe Ratio Distribution')
            axes[1, 0].legend()
            axes[1, 0].grid(True)
            
            # Plot 4: Summary stats
            axes[1, 1].axis('off')
            summary_text = f"""
            Summary Statistics
            ═══════════════════════════
            Total Folds: {self.results.total_folds}
            Passing Folds: {self.results.passing_folds} ({self.results.pass_rate*100:.1f}%)
            
            In-Sample Sharpe: {self.results.avg_is_sharpe:.3f}
            Out-of-Sample Sharpe: {self.results.avg_os_sharpe:.3f}
            
            Degradation: {self.results.sharpe_degradation:.3f}
            Degradation %: {self.results.summary_metrics['sharpe_degradation_pct']:.1f}%
            
            IS Sharpe Std: {self.results.summary_metrics['is_sharpe_std']:.3f}
            OS Sharpe Std: {self.results.summary_metrics['os_sharpe_std']:.3f}
            """
            axes[1, 1].text(0.1, 0.5, summary_text, fontsize=12, family='monospace',
                           verticalalignment='center')
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"[WFO] Plot saved to: {save_path}")
            
            if show:
                plt.show()
            
        except ImportError:
            print("[WFO] matplotlib not installed, skipping plot")


# ========================================
# Utility Functions
# ========================================

def run_wfo_example():
    """Example WFO run with dummy strategy"""
    # Generate dummy data
    dates = pd.date_range('2020-01-01', '2023-12-31', freq='D')
    data = pd.DataFrame({
        'close': np.random.randn(len(dates)).cumsum() + 100,
        'volume': np.random.randint(1000, 10000, len(dates))
    }, index=dates)
    
    # Dummy strategy function
    def dummy_strategy(data: pd.DataFrame, ma_period: int = 20, threshold: float = 0.01) -> pd.DataFrame:
        """Simple MA crossover strategy"""
        data = data.copy()
        data['ma'] = data['close'].rolling(ma_period).mean()
        data['signal'] = (data['close'] > data['ma'] * (1 + threshold)).astype(int)
        data['position_change'] = data['signal'].diff()
        
        # Generate trades
        entries = data[data['position_change'] == 1].index
        exits = data[data['position_change'] == -1].index
        
        trades = []
        for entry, exit_date in zip(entries, exits):
            if exit_date > entry:
                entry_price = data.loc[entry, 'close']
                exit_price = data.loc[exit_date, 'close']
                pnl = exit_price - entry_price
                ret_pct = pnl / entry_price
                
                trades.append({
                    'timestamp': exit_date,
                    'pnl': pnl,
                    'return_pct': ret_pct
                })
        
        return pd.DataFrame(trades)
    
    # Parameter grid
    param_grid = {
        'ma_period': [10, 20, 50],
        'threshold': [0.005, 0.01, 0.02]
    }
    
    # Run WFO
    config = WFOConfig(
        in_sample_days=252,
        out_sample_days=63,
        step_days=63,
        optimization_metric='sharpe_ratio'
    )
    
    wfo = WalkForwardOptimizer(config)
    results = wfo.run(data, dummy_strategy, param_grid)
    
    # Plot results
    wfo.plot_results(show=True)
    
    return results


if __name__ == '__main__':
    print("Running WFO Example...")
    results = run_wfo_example()
