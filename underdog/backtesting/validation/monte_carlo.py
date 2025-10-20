"""
Monte Carlo Simulation Engine
Trade resampling, slippage distributions, parameter jitter, and percentile-based risk analysis.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Callable
import numpy as np
import pandas as pd
from pathlib import Path
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
import warnings


@dataclass
class MonteCarloConfig:
    """Monte Carlo simulation configuration"""
    # Simulation parameters
    n_simulations: int = 10000
    random_seed: Optional[int] = 42
    
    # Resampling methods
    resample_trades: bool = True  # Resample trade sequence
    resample_with_replacement: bool = True
    
    # Perturbations
    add_slippage: bool = True
    slippage_mean: float = 0.0001  # 1 pip average
    slippage_std: float = 0.0002   # 2 pips std
    
    add_latency: bool = False
    latency_mean_ms: float = 50.0
    latency_std_ms: float = 20.0
    
    parameter_jitter: bool = False
    parameter_jitter_pct: float = 0.05  # 5% parameter variation
    
    # Analysis
    confidence_levels: List[float] = field(default_factory=lambda: [0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99])
    
    # Performance
    use_multiprocessing: bool = False
    n_workers: int = 4
    
    # Output
    save_results: bool = True
    results_dir: str = "results/monte_carlo"


@dataclass
class MonteCarloResult:
    """Monte Carlo simulation result"""
    config: MonteCarloConfig
    
    # Original backtest metrics
    original_metrics: Dict[str, float]
    
    # Simulated distributions
    returns_distribution: np.ndarray
    sharpe_distribution: np.ndarray
    drawdown_distribution: np.ndarray
    win_rate_distribution: np.ndarray
    
    # Percentile statistics
    percentiles: Dict[str, Dict[float, float]]
    
    # Risk metrics
    var_95: float = 0.0  # Value at Risk (95%)
    cvar_95: float = 0.0  # Conditional VaR (95%)
    worst_case_dd: float = 0.0
    
    # Probability metrics
    prob_positive_return: float = 0.0
    prob_target_return: float = 0.0
    prob_ruin: float = 0.0
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization"""
        return {
            'config': {
                'n_simulations': self.config.n_simulations,
                'resample_trades': self.config.resample_trades,
                'add_slippage': self.config.add_slippage,
                'parameter_jitter': self.config.parameter_jitter
            },
            'original_metrics': self.original_metrics,
            'percentiles': self.percentiles,
            'risk_metrics': {
                'var_95': float(self.var_95),
                'cvar_95': float(self.cvar_95),
                'worst_case_dd': float(self.worst_case_dd)
            },
            'probability_metrics': {
                'prob_positive_return': float(self.prob_positive_return),
                'prob_target_return': float(self.prob_target_return),
                'prob_ruin': float(self.prob_ruin)
            }
        }


class MonteCarloSimulator:
    """
    Monte Carlo simulation engine for strategy robustness testing.
    
    Implements:
    - Trade sequence resampling
    - Slippage and latency simulation
    - Parameter perturbation
    - Percentile-based risk analysis
    """
    
    def __init__(self, config: Optional[MonteCarloConfig] = None):
        """
        Initialize Monte Carlo simulator.
        
        Args:
            config: Monte Carlo configuration
        """
        self.config = config or MonteCarloConfig()
        
        # Set random seed for reproducibility
        if self.config.random_seed is not None:
            np.random.seed(self.config.random_seed)
        
        self.results: Optional[MonteCarloResult] = None
    
    # MEJORA CIENTÍFICA: Realistic Slippage Model
    def calculate_realistic_slippage(self, 
                                    trade_size: float, 
                                    market_conditions: Dict[str, float]) -> float:
        """
        Modelo realista de slippage basado en microestructura de mercado.
        
        Literatura: Slippage real es función de:
        1. Base spread (bid-ask)
        2. Price impact (proporcional a sqrt(trade_size))
        3. Volatility multiplier
        4. Fat tails (eventos extremos)
        
        Fórmula: slippage = (spread + impact) * vol_multiplier + noise
        
        Args:
            trade_size: Tamaño del trade en unidades (ej: 100k USD)
            market_conditions: Dict con {
                'bid_ask_spread': float (ej: 0.0001 = 1 pip),
                'volume': float (volumen promedio del mercado),
                'volatility': float (volatilidad actual),
                'avg_volatility': float (volatilidad promedio)
            }
        
        Returns:
            Slippage total (como fracción del precio)
        """
        # Componente 1: Base spread
        base_spread = market_conditions.get('bid_ask_spread', 0.0001)
        
        # Componente 2: Price impact (literatura: impact ~ size^0.5)
        avg_market_volume = market_conditions.get('volume', 1000000)
        trade_fraction = trade_size / avg_market_volume
        
        # Fórmula de Kyle (1985): price impact proporcional a sqrt(volume)
        impact_coefficient = 0.1  # Empírico (calibrar con datos reales)
        price_impact = impact_coefficient * np.sqrt(max(trade_fraction, 0))
        
        # Componente 3: Volatility multiplier (mayor slippage en alta vol)
        current_vol = market_conditions.get('volatility', 0.01)
        avg_vol = market_conditions.get('avg_volatility', 0.01)
        vol_multiplier = 1.0 + (current_vol / avg_vol - 1.0)
        vol_multiplier = max(vol_multiplier, 0.5)  # Min 0.5x, no negativo
        
        # Total slippage (determinístico)
        deterministic_slippage = (base_spread + price_impact) * vol_multiplier
        
        # Componente 4: Ruido estocástico con fat tails (t-distribution)
        # Student's t con df=3 captura eventos extremos mejor que Normal
        degrees_of_freedom = 3  # df bajo = fat tails
        stochastic_noise = np.random.standard_t(df=degrees_of_freedom) * 0.0001
        
        total_slippage = deterministic_slippage + stochastic_noise
        
        # Slippage siempre es positivo (adverso)
        return abs(total_slippage)
    
    def apply_realistic_slippage_to_trades(self,
                                           trades: pd.DataFrame,
                                           default_trade_size: float = 100000,
                                           market_vol: float = 0.01) -> pd.DataFrame:
        """
        Aplica slippage realista a cada trade.
        
        Args:
            trades: DataFrame con trades
            default_trade_size: Tamaño por defecto si no está en el DataFrame
            market_vol: Volatilidad del mercado
        
        Returns:
            DataFrame con slippage realista aplicado
        """
        simulated = trades.copy()
        
        for i in range(len(simulated)):
            # Extraer tamaño del trade (si existe)
            trade_size = simulated.iloc[i].get('size', default_trade_size)
            
            # Market conditions (simplificado - idealmente de datos históricos)
            market_conditions = {
                'bid_ask_spread': 0.0001,  # 1 pip para Forex majors
                'volume': 1000000,  # Placeholder
                'volatility': market_vol,
                'avg_volatility': 0.01
            }
            
            # Calcular slippage
            slippage = self.calculate_realistic_slippage(trade_size, market_conditions)
            
            # Aplicar slippage (siempre adverso)
            if 'pnl' in simulated.columns:
                pnl = simulated.iloc[i]['pnl']
                simulated.at[simulated.index[i], 'pnl'] = pnl - abs(pnl) * slippage
            
            if 'return_pct' in simulated.columns:
                ret = simulated.iloc[i]['return_pct']
                simulated.at[simulated.index[i], 'return_pct'] = ret - abs(ret) * slippage
        
        return simulated
    
    def run(self,
            trades: pd.DataFrame,
            initial_capital: float = 100000,
            target_return_pct: float = 20.0,
            ruin_threshold_pct: float = -20.0) -> MonteCarloResult:
        """
        Run Monte Carlo simulation.
        
        Args:
            trades: Historical trades DataFrame with columns: pnl, return_pct, timestamp
            initial_capital: Starting capital
            target_return_pct: Target return for probability calculation
            ruin_threshold_pct: Ruin threshold (e.g., -20% is ruin)
        
        Returns:
            MonteCarloResult with complete simulation results
        """
        print(f"[MC] Starting Monte Carlo Simulation")
        print(f"[MC] Simulations: {self.config.n_simulations:,}")
        print(f"[MC] Original trades: {len(trades)}")
        
        # Calculate original metrics
        original_metrics = self._calculate_metrics(trades, initial_capital)
        print(f"[MC] Original Sharpe: {original_metrics['sharpe_ratio']:.3f}")
        print(f"[MC] Original Max DD: {original_metrics['max_drawdown']:.2f}%")
        
        # Run simulations
        if self.config.use_multiprocessing:
            simulation_results = self._run_parallel(trades, initial_capital)
        else:
            simulation_results = self._run_sequential(trades, initial_capital)
        
        # Extract distributions
        returns_dist = np.array([r['total_return'] for r in simulation_results])
        sharpe_dist = np.array([r['sharpe_ratio'] for r in simulation_results])
        dd_dist = np.array([r['max_drawdown'] for r in simulation_results])
        wr_dist = np.array([r['win_rate'] for r in simulation_results])
        
        # Calculate percentiles
        percentiles = self._calculate_percentiles({
            'total_return': returns_dist,
            'sharpe_ratio': sharpe_dist,
            'max_drawdown': dd_dist,
            'win_rate': wr_dist
        })
        
        # Calculate risk metrics
        var_95 = np.percentile(returns_dist, 5)
        cvar_95 = np.mean(returns_dist[returns_dist <= var_95])
        worst_case_dd = np.max(dd_dist)
        
        # Probability metrics
        prob_positive = np.mean(returns_dist > 0)
        prob_target = np.mean(returns_dist >= target_return_pct)
        prob_ruin = np.mean(returns_dist <= ruin_threshold_pct)
        
        # Create result
        self.results = MonteCarloResult(
            config=self.config,
            original_metrics=original_metrics,
            returns_distribution=returns_dist,
            sharpe_distribution=sharpe_dist,
            drawdown_distribution=dd_dist,
            win_rate_distribution=wr_dist,
            percentiles=percentiles,
            var_95=var_95,
            cvar_95=cvar_95,
            worst_case_dd=worst_case_dd,
            prob_positive_return=prob_positive,
            prob_target_return=prob_target,
            prob_ruin=prob_ruin
        )
        
        # Print summary
        self._print_summary()
        
        # Save results
        if self.config.save_results:
            self._save_results()
        
        return self.results
    
    def _run_sequential(self, trades: pd.DataFrame, initial_capital: float) -> List[Dict]:
        """Run simulations sequentially"""
        results = []
        
        for i in range(self.config.n_simulations):
            if (i + 1) % 1000 == 0:
                print(f"[MC] Progress: {i+1}/{self.config.n_simulations}")
            
            # Simulate
            simulated_trades = self._simulate_trades(trades)
            metrics = self._calculate_metrics(simulated_trades, initial_capital)
            results.append(metrics)
        
        return results
    
    def _run_parallel(self, trades: pd.DataFrame, initial_capital: float) -> List[Dict]:
        """Run simulations in parallel"""
        results = []
        
        with ProcessPoolExecutor(max_workers=self.config.n_workers) as executor:
            futures = [
                executor.submit(self._simulate_and_calculate, trades, initial_capital)
                for _ in range(self.config.n_simulations)
            ]
            
            for i, future in enumerate(as_completed(futures)):
                if (i + 1) % 1000 == 0:
                    print(f"[MC] Progress: {i+1}/{self.config.n_simulations}")
                
                results.append(future.result())
        
        return results
    
    def _simulate_and_calculate(self, trades: pd.DataFrame, initial_capital: float) -> Dict:
        """Single simulation iteration (for parallel execution)"""
        simulated_trades = self._simulate_trades(trades)
        return self._calculate_metrics(simulated_trades, initial_capital)
    
    def _simulate_trades(self, trades: pd.DataFrame) -> pd.DataFrame:
        """
        Apply Monte Carlo perturbations to trades.
        
        Args:
            trades: Original trades
        
        Returns:
            Simulated trades with perturbations
        """
        simulated = trades.copy()
        
        # 1. Resample trade sequence (Block Bootstrap para preservar autocorrelación)
        if self.config.resample_trades:
            simulated = self._block_bootstrap_resample(simulated)
        
        # 2. Add slippage (MEJORADO: Realistic Slippage Model)
        if self.config.add_slippage:
            simulated = self.apply_realistic_slippage_to_trades(
                simulated,
                default_trade_size=100000,
                market_vol=0.01
            )
        
        # 3. Add execution latency (affects order of fills)
        if self.config.add_latency and 'timestamp' in simulated.columns:
            latency = np.random.normal(
                self.config.latency_mean_ms,
                self.config.latency_std_ms,
                len(simulated)
            )
            # Latency affects timestamp (simplified - in reality would affect fill price)
            # This is a placeholder for more complex latency modeling
        
        return simulated
    
    def _block_bootstrap_resample(self, 
                                   trades: pd.DataFrame, 
                                   block_size: int = 10) -> pd.DataFrame:
        """
        Block Bootstrap para preservar autocorrelación en trades.
        
        Literatura: Trade sequences tienen autocorrelación (rachas de wins/losses).
        Bootstrap simple destruye esta estructura.
        
        Args:
            trades: DataFrame de trades
            block_size: Tamaño de bloques (default 10)
        
        Returns:
            DataFrame resampled con estructura temporal preservada
        """
        if not self.config.resample_with_replacement:
            # Shuffle simple sin replacement
            return trades.iloc[np.random.permutation(len(trades))].reset_index(drop=True)
        
        # Block bootstrap
        n_blocks = len(trades) // block_size
        resampled_indices = []
        
        for _ in range(n_blocks):
            # Elegir inicio de bloque aleatorio
            start_idx = np.random.randint(0, len(trades) - block_size + 1)
            block_indices = list(range(start_idx, start_idx + block_size))
            resampled_indices.extend(block_indices)
        
        # Recortar al tamaño original
        resampled_indices = resampled_indices[:len(trades)]
        
        return trades.iloc[resampled_indices].reset_index(drop=True)
    
    def _calculate_metrics(self, trades: pd.DataFrame, initial_capital: float) -> Dict[str, float]:
        """Calculate performance metrics for a trade sequence"""
        if len(trades) == 0:
            return {
                'total_return': 0.0,
                'sharpe_ratio': 0.0,
                'max_drawdown': 0.0,
                'win_rate': 0.0,
                'profit_factor': 0.0,
                'num_trades': 0
            }
        
        # Extract returns
        if 'return_pct' in trades.columns:
            returns = trades['return_pct'].values
        elif 'pnl' in trades.columns:
            returns = trades['pnl'].values / initial_capital * 100
        else:
            returns = np.zeros(len(trades))
        
        # Total return
        total_return = np.sum(returns)
        
        # Sharpe ratio
        mean_ret = np.mean(returns)
        std_ret = np.std(returns)
        sharpe = (mean_ret / std_ret) * np.sqrt(252) if std_ret > 0 else 0.0
        
        # Maximum drawdown
        cumulative = np.cumsum(returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = running_max - cumulative
        max_dd = np.max(drawdown) if len(drawdown) > 0 else 0.0
        
        # Win rate
        wins = returns[returns > 0]
        win_rate = len(wins) / len(returns) if len(returns) > 0 else 0.0
        
        # Profit factor
        gross_profit = np.sum(wins) if len(wins) > 0 else 0.0
        losses = returns[returns < 0]
        gross_loss = np.sum(np.abs(losses)) if len(losses) > 0 else 0.0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0.0
        
        # Calmar Ratio (CAGR / Max Drawdown)
        cagr = total_return * 252 / len(trades) if len(trades) > 0 else 0.0
        calmar_ratio = (cagr / max_dd) if max_dd > 0 else 0.0
        
        return {
            'total_return': float(total_return),
            'sharpe_ratio': float(sharpe),
            'max_drawdown': float(max_dd),
            'win_rate': float(win_rate * 100),
            'profit_factor': float(profit_factor),
            'num_trades': int(len(trades)),
            'cagr': float(cagr),
            'calmar_ratio': float(calmar_ratio)
        }
    
    def _calculate_percentiles(self, distributions: Dict[str, np.ndarray]) -> Dict[str, Dict[float, float]]:
        """Calculate percentiles for all distributions"""
        percentiles = {}
        
        for metric_name, dist in distributions.items():
            percentiles[metric_name] = {
                p: float(np.percentile(dist, p * 100))
                for p in self.config.confidence_levels
            }
        
        return percentiles
    
    def _print_summary(self) -> None:
        """Print Monte Carlo summary"""
        if self.results is None:
            return
        
        print(f"\n[MC] ===== MONTE CARLO SUMMARY =====")
        print(f"[MC] Simulations: {self.config.n_simulations:,}")
        
        print(f"\n[MC] Original vs. Simulated (Median):")
        print(f"[MC]   Total Return: {self.results.original_metrics['total_return']:.2f}% vs {self.results.percentiles['total_return'][0.50]:.2f}%")
        print(f"[MC]   Sharpe Ratio: {self.results.original_metrics['sharpe_ratio']:.3f} vs {self.results.percentiles['sharpe_ratio'][0.50]:.3f}")
        print(f"[MC]   Max DD: {self.results.original_metrics['max_drawdown']:.2f}% vs {self.results.percentiles['max_drawdown'][0.50]:.2f}%")
        
        print(f"\n[MC] Risk Metrics:")
        print(f"[MC]   VaR (95%): {self.results.var_95:.2f}%")
        print(f"[MC]   CVaR (95%): {self.results.cvar_95:.2f}%")
        print(f"[MC]   Worst-case DD: {self.results.worst_case_dd:.2f}%")
        
        print(f"\n[MC] Probability Metrics:")
        print(f"[MC]   P(positive return): {self.results.prob_positive_return*100:.1f}%")
        print(f"[MC]   P(target return): {self.results.prob_target_return*100:.1f}%")
        print(f"[MC]   P(ruin): {self.results.prob_ruin*100:.1f}%")
        
        print(f"\n[MC] Return Percentiles:")
        for p in [0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99]:
            print(f"[MC]   {p*100:.0f}%: {self.results.percentiles['total_return'][p]:.2f}%")
    
    def _save_results(self) -> None:
        """Save Monte Carlo results"""
        if self.results is None:
            return
        
        results_dir = Path(self.config.results_dir)
        results_dir.mkdir(parents=True, exist_ok=True)
        
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save summary JSON
        json_path = results_dir / f"mc_results_{timestamp}.json"
        with open(json_path, 'w') as f:
            json.dump(self.results.to_dict(), f, indent=2)
        
        # Save distributions as numpy arrays
        npz_path = results_dir / f"mc_distributions_{timestamp}.npz"
        np.savez_compressed(
            npz_path,
            returns=self.results.returns_distribution,
            sharpe=self.results.sharpe_distribution,
            drawdown=self.results.drawdown_distribution,
            win_rate=self.results.win_rate_distribution
        )
        
        print(f"[MC] Results saved to: {json_path}")
        print(f"[MC] Distributions saved to: {npz_path}")
    
    def plot_results(self, show: bool = True, save_path: Optional[str] = None) -> None:
        """Plot Monte Carlo results"""
        if self.results is None:
            print("[MC] No results to plot")
            return
        
        try:
            import matplotlib.pyplot as plt
            
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle('Monte Carlo Simulation Results', fontsize=16)
            
            # Plot 1: Returns distribution
            axes[0, 0].hist(self.results.returns_distribution, bins=50, alpha=0.7, edgecolor='black')
            axes[0, 0].axvline(self.results.original_metrics['total_return'], color='r', 
                             linestyle='--', label='Original', linewidth=2)
            axes[0, 0].axvline(self.results.var_95, color='orange', 
                             linestyle='--', label='VaR 95%', linewidth=2)
            axes[0, 0].set_xlabel('Total Return (%)')
            axes[0, 0].set_ylabel('Frequency')
            axes[0, 0].set_title('Total Return Distribution')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
            
            # Plot 2: Sharpe distribution
            axes[0, 1].hist(self.results.sharpe_distribution, bins=50, alpha=0.7, edgecolor='black')
            axes[0, 1].axvline(self.results.original_metrics['sharpe_ratio'], color='r',
                             linestyle='--', label='Original', linewidth=2)
            axes[0, 1].set_xlabel('Sharpe Ratio')
            axes[0, 1].set_ylabel('Frequency')
            axes[0, 1].set_title('Sharpe Ratio Distribution')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
            
            # Plot 3: Drawdown distribution
            axes[1, 0].hist(self.results.drawdown_distribution, bins=50, alpha=0.7, edgecolor='black')
            axes[1, 0].axvline(self.results.original_metrics['max_drawdown'], color='r',
                             linestyle='--', label='Original', linewidth=2)
            axes[1, 0].axvline(self.results.worst_case_dd, color='darkred',
                             linestyle='--', label='Worst Case', linewidth=2)
            axes[1, 0].set_xlabel('Maximum Drawdown (%)')
            axes[1, 0].set_ylabel('Frequency')
            axes[1, 0].set_title('Maximum Drawdown Distribution')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
            
            # Plot 4: Summary statistics
            axes[1, 1].axis('off')
            summary_text = f"""
            Monte Carlo Summary
            ══════════════════════════════
            Simulations: {self.config.n_simulations:,}
            
            Return Median: {self.results.percentiles['total_return'][0.50]:.2f}%
            Return 5th %ile: {self.results.percentiles['total_return'][0.05]:.2f}%
            Return 95th %ile: {self.results.percentiles['total_return'][0.95]:.2f}%
            
            VaR (95%): {self.results.var_95:.2f}%
            CVaR (95%): {self.results.cvar_95:.2f}%
            
            Worst-case DD: {self.results.worst_case_dd:.2f}%
            
            P(positive): {self.results.prob_positive_return*100:.1f}%
            P(target): {self.results.prob_target_return*100:.1f}%
            P(ruin): {self.results.prob_ruin*100:.1f}%
            """
            axes[1, 1].text(0.1, 0.5, summary_text, fontsize=11, family='monospace',
                           verticalalignment='center')
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"[MC] Plot saved to: {save_path}")
            
            if show:
                plt.show()
                
        except ImportError:
            print("[MC] matplotlib not installed, skipping plot")


# ========================================
# Utility Functions
# ========================================

def run_mc_example():
    """Example Monte Carlo simulation"""
    # Generate dummy trades
    np.random.seed(42)
    n_trades = 100
    
    trades = pd.DataFrame({
        'timestamp': pd.date_range('2023-01-01', periods=n_trades, freq='D'),
        'pnl': np.random.normal(50, 200, n_trades),
        'return_pct': np.random.normal(0.5, 2.0, n_trades)
    })
    
    # Run Monte Carlo
    config = MonteCarloConfig(
        n_simulations=5000,
        resample_trades=True,
        add_slippage=True,
        parameter_jitter=False
    )
    
    simulator = MonteCarloSimulator(config)
    results = simulator.run(trades, initial_capital=100000, target_return_pct=20.0)
    
    # Plot results
    simulator.plot_results(show=True)
    
    return results


if __name__ == '__main__':
    print("Running Monte Carlo Example...")
    results = run_mc_example()
