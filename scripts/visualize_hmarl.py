"""
🎯 HMARL Training Auditor - Per-Agent Financial Intelligence Dashboard

Visualizes critical per-agent metrics for heterogeneous MARL training:
- GBPUSD (DDPG): Cumulative Fakeout Penalty (stale quote defense audit)
- USDJPY (PPO): PnL Volatility σ (stability audit)
- XAUUSD (SAC): Net Position Exposure (liquidity opportunity audit)
- EURUSD (TD3): Win Rate (trend-following efficiency audit)

Without per-agent metrics, 50 episodes = black box. This dashboard reveals
which agents learned vs failed, enabling data-driven hyperparameter tuning.

Usage:
    python scripts/visualize_hmarl.py --log-file logs/training_metrics.csv --live
    
    # Static analysis (post-training)
    python scripts/visualize_hmarl.py --log-file logs/training_metrics.csv
"""

import argparse
import time
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


class HMARLAuditor:
    """
    Financial Intelligence Dashboard for Heterogeneous MARL
    
    Implements per-agent audit metrics based on asset-specific reward shapers
    and neural architectures.
    """
    
    # Agent-to-Symbol mapping (matches train_marl_agent.py order)
    AGENT_SYMBOLS = ['EURUSD', 'USDJPY', 'XAUUSD', 'GBPUSD']
    AGENT_ALGOS = ['TD3', 'PPO', 'SAC', 'DDPG']
    AGENT_ARCHS = ['LSTM', 'CNN1D', 'Transformer', 'Attention']
    
    # Colors for visual consistency
    COLORS = {
        'EURUSD': '#1f77b4',  # Blue (stable, trend)
        'USDJPY': '#ff7f0e',  # Orange (volatile, impulsive)
        'XAUUSD': '#2ca02c',  # Green (opportunity, gold)
        'GBPUSD': '#d62728',  # Red (defensive, whipsaw)
    }
    
    def __init__(self, log_file: str, symbols: Optional[List[str]] = None):
        """
        Initialize HMARL auditor
        
        Args:
            log_file: Path to CSV log file from train_marl_agent.py
            symbols: Custom symbol order (default: EURUSD, USDJPY, XAUUSD, GBPUSD)
        """
        self.log_file = Path(log_file)
        self.symbols = symbols or self.AGENT_SYMBOLS
        
        # Create 4x2 subplot grid (4 agents × 2 metrics each)
        self.fig = plt.figure(figsize=(16, 12))
        self.fig.suptitle(
            '🎯 HMARL Financial Intelligence Auditor\n'
            'Per-Agent Reward Shaper Performance',
            fontsize=16, fontweight='bold'
        )
        
        # Create subplot grid
        self.axes = {}
        self._setup_subplots()
        
        # Data cache for rolling calculations
        self.data_cache = {symbol: {'episodes': [], 'metrics': {}} for symbol in self.symbols}
        
        logger.info(f"HMARL Auditor initialized: {self.log_file}")
        logger.info(f"Monitoring agents: {', '.join([f'{s}({a})' for s, a in zip(self.symbols, self.AGENT_ALGOS)])}")
    
    def _setup_subplots(self):
        """Create 4×2 subplot grid with agent-specific metrics"""
        
        for idx, symbol in enumerate(self.symbols):
            algo = self.AGENT_ALGOS[idx]
            arch = self.AGENT_ARCHS[idx]
            color = self.COLORS[symbol]
            
            # Primary metric (left column)
            ax_primary = plt.subplot(4, 2, idx*2 + 1)
            ax_primary.set_title(
                f'{symbol} ({algo}+{arch}) - Primary Metric',
                fontsize=11, fontweight='bold', color=color
            )
            ax_primary.grid(True, alpha=0.3)
            ax_primary.set_xlabel('Episode')
            
            # Secondary metric (right column)
            ax_secondary = plt.subplot(4, 2, idx*2 + 2)
            ax_secondary.set_title(
                f'{symbol} ({algo}+{arch}) - Secondary Metric',
                fontsize=11, fontweight='bold', color=color
            )
            ax_secondary.grid(True, alpha=0.3)
            ax_secondary.set_xlabel('Episode')
            
            self.axes[symbol] = {
                'primary': ax_primary,
                'secondary': ax_secondary
            }
        
        # Configure agent-specific metrics
        self._configure_gbpusd_metrics()
        self._configure_usdjpy_metrics()
        self._configure_xauusd_metrics()
        self._configure_eurusd_metrics()
        
        plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    def _configure_gbpusd_metrics(self):
        """Configure GBPUSD (DDPG) - Fakeout penalty audit"""
        ax_primary = self.axes['GBPUSD']['primary']
        ax_secondary = self.axes['GBPUSD']['secondary']
        
        # Primary: Cumulative Fakeout Penalty
        ax_primary.set_ylabel('Cumulative Fakeout Penalty', fontweight='bold')
        ax_primary.axhline(y=0, color='green', linestyle='--', linewidth=1, alpha=0.5, label='No Penalties')
        ax_primary.legend(loc='lower left')
        
        # Secondary: Whipsaw Rate (trades closed in loss < 10 steps)
        ax_secondary.set_ylabel('Whipsaw Rate (%)', fontweight='bold')
        ax_secondary.axhline(y=20, color='orange', linestyle='--', linewidth=1, alpha=0.5, label='Target <20%')
        ax_secondary.set_ylim(0, 100)
        ax_secondary.legend(loc='upper right')
    
    def _configure_usdjpy_metrics(self):
        """Configure USDJPY (PPO) - Volatility audit"""
        ax_primary = self.axes['USDJPY']['primary']
        ax_secondary = self.axes['USDJPY']['secondary']
        
        # Primary: PnL Volatility (rolling σ of equity changes)
        ax_primary.set_ylabel('PnL Volatility σ (20-ep rolling)', fontweight='bold')
        ax_primary.axhline(y=0, color='blue', linestyle='--', linewidth=1, alpha=0.5, label='Zero Volatility')
        ax_primary.legend(loc='upper right')
        
        # Secondary: Rolling Sharpe Ratio (20 episodes)
        ax_secondary.set_ylabel('Rolling Sharpe Ratio (20-ep)', fontweight='bold')
        ax_secondary.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
        ax_secondary.axhline(y=1.0, color='green', linestyle='--', linewidth=1, alpha=0.5, label='Target Sharpe>1.0')
        ax_secondary.legend(loc='lower right')
    
    def _configure_xauusd_metrics(self):
        """Configure XAUUSD (SAC) - Exposure audit"""
        ax_primary = self.axes['XAUUSD']['primary']
        ax_secondary = self.axes['XAUUSD']['secondary']
        
        # Primary: Net Position Exposure (average position size)
        ax_primary.set_ylabel('Net Position Exposure', fontweight='bold')
        ax_primary.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.5, label='Neutral')
        ax_primary.set_ylim(-1.0, 1.0)
        ax_primary.legend(loc='upper left')
        
        # Secondary: Position Flip Rate (regime adaptation frequency)
        ax_secondary.set_ylabel('Position Flip Rate (flips/episode)', fontweight='bold')
        ax_secondary.axhline(y=5, color='orange', linestyle='--', linewidth=1, alpha=0.5, label='Expected ~5')
        ax_secondary.legend(loc='upper right')
    
    def _configure_eurusd_metrics(self):
        """Configure EURUSD (TD3) - Win rate audit"""
        ax_primary = self.axes['EURUSD']['primary']
        ax_secondary = self.axes['EURUSD']['secondary']
        
        # Primary: Win Rate (rolling 20-episode)
        ax_primary.set_ylabel('Win Rate % (20-ep rolling)', fontweight='bold')
        ax_primary.axhline(y=50, color='gray', linestyle='--', linewidth=1, alpha=0.5, label='Random (50%)')
        ax_primary.axhline(y=40, color='orange', linestyle='--', linewidth=1, alpha=0.5, label='Target >40% (trend)')
        ax_primary.set_ylim(0, 100)
        ax_primary.legend(loc='lower right')
        
        # Secondary: Payoff Ratio (avg_win / avg_loss)
        ax_secondary.set_ylabel('Payoff Ratio (avg_win/avg_loss)', fontweight='bold')
        ax_secondary.axhline(y=1.0, color='gray', linestyle='--', linewidth=1, alpha=0.5, label='Break-even')
        ax_secondary.axhline(y=2.0, color='green', linestyle='--', linewidth=1, alpha=0.5, label='Target >2.0')
        ax_secondary.legend(loc='lower right')
    
    def load_data(self) -> Optional[pd.DataFrame]:
        """
        Load latest training data from CSV
        
        Returns:
            DataFrame or None if file doesn't exist/is empty
        """
        if not self.log_file.exists():
            logger.warning(f"Log file not found: {self.log_file}")
            return None
        
        try:
            df = pd.read_csv(self.log_file)
            if len(df) == 0:
                return None
            return df
        except (pd.errors.EmptyDataError, PermissionError) as e:
            logger.warning(f"Error reading CSV: {e}")
            return None
    
    def calculate_gbpusd_metrics(self, df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """
        Calculate GBPUSD (DDPG) metrics
        
        Metrics:
        1. Cumulative fakeout penalty (estimated from rewards)
        2. Whipsaw rate (trades closed in loss quickly)
        
        Note: Fakeout penalty not directly logged, so we estimate from
        reward deviations when position changes rapidly.
        """
        episodes = df['episode'].values
        agent_idx = self.symbols.index('GBPUSD')
        rewards = df[f'agent{agent_idx}_reward'].values
        
        # Estimate cumulative fakeout penalties
        # (actual penalties would need custom logging, this is approximation)
        # Assume large negative spikes in reward = fakeout penalties
        cumulative_penalty = np.cumsum(np.minimum(rewards, 0))
        
        # Whipsaw rate estimation
        # (would need trade duration data for precise calculation)
        # Proxy: rolling percentage of negative episodes
        window = min(20, len(episodes))
        whipsaw_rates = []
        for i in range(len(episodes)):
            if i < window:
                whipsaw_rates.append(0)
            else:
                recent_rewards = rewards[i-window:i]
                whipsaw_rate = (np.sum(recent_rewards < 0) / window) * 100
                whipsaw_rates.append(whipsaw_rate)
        
        return {
            'episodes': episodes,
            'cumulative_penalty': cumulative_penalty,
            'whipsaw_rate': np.array(whipsaw_rates)
        }
    
    def calculate_usdjpy_metrics(self, df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """
        Calculate USDJPY (PPO) metrics
        
        Metrics:
        1. PnL Volatility (rolling std dev of balance changes)
        2. Rolling Sharpe ratio
        """
        episodes = df['episode'].values
        agent_idx = self.symbols.index('USDJPY')
        balances = df[f'agent{agent_idx}_balance'].values
        
        # Calculate PnL changes
        pnl_changes = np.diff(balances, prepend=balances[0])
        
        # Rolling volatility (20-episode window)
        window = 20
        volatilities = []
        sharpe_ratios = []
        
        for i in range(len(episodes)):
            if i < window:
                volatilities.append(0)
                sharpe_ratios.append(0)
            else:
                recent_changes = pnl_changes[i-window:i]
                vol = np.std(recent_changes)
                mean_return = np.mean(recent_changes)
                
                volatilities.append(vol)
                
                # Sharpe ratio
                if vol > 1e-8:
                    sharpe = mean_return / vol
                else:
                    sharpe = 0
                sharpe_ratios.append(sharpe)
        
        return {
            'episodes': episodes,
            'pnl_volatility': np.array(volatilities),
            'sharpe_ratio': np.array(sharpe_ratios)
        }
    
    def calculate_xauusd_metrics(self, df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """
        Calculate XAUUSD (SAC) metrics
        
        Metrics:
        1. Net position exposure (would need position data, using reward proxy)
        2. Position flip rate (regime adaptation frequency)
        """
        episodes = df['episode'].values
        agent_idx = self.symbols.index('XAUUSD')
        rewards = df[f'agent{agent_idx}_reward'].values
        
        # Exposure proxy: normalize rewards to [-1, 1] range
        # (actual exposure would need position size logging)
        max_abs_reward = np.max(np.abs(rewards)) if np.max(np.abs(rewards)) > 0 else 1.0
        exposure_proxy = rewards / max_abs_reward
        
        # Position flip rate proxy: sign changes in rewards
        # (actual flips would need position delta logging)
        reward_signs = np.sign(rewards)
        flips = np.abs(np.diff(reward_signs, prepend=0))
        
        # Rolling flip count (per episode, no window)
        flip_rates = flips  # Each flip counts as 1
        
        return {
            'episodes': episodes,
            'exposure': exposure_proxy,
            'flip_rate': flip_rates
        }
    
    def calculate_eurusd_metrics(self, df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """
        Calculate EURUSD (TD3) metrics
        
        Metrics:
        1. Win rate (rolling 20-episode)
        2. Payoff ratio (avg_win / avg_loss)
        """
        episodes = df['episode'].values
        agent_idx = self.symbols.index('EURUSD')
        rewards = df[f'agent{agent_idx}_reward'].values
        
        # Win rate calculation
        window = 20
        win_rates = []
        payoff_ratios = []
        
        for i in range(len(episodes)):
            if i < window:
                win_rates.append(0)
                payoff_ratios.append(0)
            else:
                recent_rewards = rewards[i-window:i]
                wins = recent_rewards > 0
                losses = recent_rewards < 0
                
                # Win rate
                win_rate = (np.sum(wins) / window) * 100 if window > 0 else 0
                win_rates.append(win_rate)
                
                # Payoff ratio
                avg_win = np.mean(recent_rewards[wins]) if np.sum(wins) > 0 else 0
                avg_loss = np.abs(np.mean(recent_rewards[losses])) if np.sum(losses) > 0 else 1
                payoff = avg_win / avg_loss if avg_loss > 0 else 0
                payoff_ratios.append(payoff)
        
        return {
            'episodes': episodes,
            'win_rate': np.array(win_rates),
            'payoff_ratio': np.array(payoff_ratios)
        }
    
    def update_plots(self, df: pd.DataFrame):
        """Update all plots with latest data"""
        
        # GBPUSD (DDPG)
        gbp_metrics = self.calculate_gbpusd_metrics(df)
        self.axes['GBPUSD']['primary'].clear()
        self.axes['GBPUSD']['secondary'].clear()
        self._configure_gbpusd_metrics()
        self.axes['GBPUSD']['primary'].plot(
            gbp_metrics['episodes'], gbp_metrics['cumulative_penalty'],
            color=self.COLORS['GBPUSD'], linewidth=2, label='Cumulative Penalty'
        )
        self.axes['GBPUSD']['secondary'].plot(
            gbp_metrics['episodes'], gbp_metrics['whipsaw_rate'],
            color=self.COLORS['GBPUSD'], linewidth=2, label='Whipsaw Rate'
        )
        
        # USDJPY (PPO)
        usd_metrics = self.calculate_usdjpy_metrics(df)
        self.axes['USDJPY']['primary'].clear()
        self.axes['USDJPY']['secondary'].clear()
        self._configure_usdjpy_metrics()
        self.axes['USDJPY']['primary'].plot(
            usd_metrics['episodes'], usd_metrics['pnl_volatility'],
            color=self.COLORS['USDJPY'], linewidth=2, label='PnL Volatility'
        )
        self.axes['USDJPY']['secondary'].plot(
            usd_metrics['episodes'], usd_metrics['sharpe_ratio'],
            color=self.COLORS['USDJPY'], linewidth=2, label='Sharpe Ratio'
        )
        
        # XAUUSD (SAC)
        xau_metrics = self.calculate_xauusd_metrics(df)
        self.axes['XAUUSD']['primary'].clear()
        self.axes['XAUUSD']['secondary'].clear()
        self._configure_xauusd_metrics()
        self.axes['XAUUSD']['primary'].plot(
            xau_metrics['episodes'], xau_metrics['exposure'],
            color=self.COLORS['XAUUSD'], linewidth=2, label='Exposure Proxy'
        )
        self.axes['XAUUSD']['secondary'].plot(
            xau_metrics['episodes'], xau_metrics['flip_rate'],
            color=self.COLORS['XAUUSD'], linewidth=2, marker='o', markersize=3, label='Position Flips'
        )
        
        # EURUSD (TD3)
        eur_metrics = self.calculate_eurusd_metrics(df)
        self.axes['EURUSD']['primary'].clear()
        self.axes['EURUSD']['secondary'].clear()
        self._configure_eurusd_metrics()
        self.axes['EURUSD']['primary'].plot(
            eur_metrics['episodes'], eur_metrics['win_rate'],
            color=self.COLORS['EURUSD'], linewidth=2, label='Win Rate'
        )
        self.axes['EURUSD']['secondary'].plot(
            eur_metrics['episodes'], eur_metrics['payoff_ratio'],
            color=self.COLORS['EURUSD'], linewidth=2, label='Payoff Ratio'
        )
        
        plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    def run_static_analysis(self):
        """Generate static analysis plots (post-training)"""
        logger.info("Running static analysis...")
        
        df = self.load_data()
        if df is None:
            logger.error("No data available for analysis")
            return
        
        self.update_plots(df)
        
        logger.info(f"Analysis complete: {len(df)} episodes")
        logger.info("Displaying plots (close window to exit)...")
        plt.show()
    
    def run_live_monitoring(self, interval_ms: int = 2000):
        """
        Run live monitoring with auto-refresh
        
        Args:
            interval_ms: Refresh interval in milliseconds
        """
        logger.info(f"Starting live monitoring (refresh every {interval_ms}ms)...")
        logger.info("Press Ctrl+C to stop")
        
        def animate(frame):
            df = self.load_data()
            if df is not None:
                self.update_plots(df)
        
        anim = animation.FuncAnimation(
            self.fig, animate, interval=interval_ms, cache_frame_data=False
        )
        
        plt.show()


def main():
    """CLI entry point"""
    parser = argparse.ArgumentParser(
        description='🎯 HMARL Financial Intelligence Auditor',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Live monitoring during training
  python scripts/visualize_hmarl.py --log-file logs/training_metrics.csv --live
  
  # Static analysis after training
  python scripts/visualize_hmarl.py --log-file logs/training_metrics.csv
  
  # Custom refresh rate (live mode)
  python scripts/visualize_hmarl.py --log-file logs/training_metrics.csv --live --interval 5000
        """
    )
    
    parser.add_argument(
        '--log-file',
        type=str,
        default='logs/training_metrics.csv',
        help='Path to training metrics CSV file'
    )
    
    parser.add_argument(
        '--live',
        action='store_true',
        help='Enable live monitoring mode (auto-refresh)'
    )
    
    parser.add_argument(
        '--interval',
        type=int,
        default=2000,
        help='Refresh interval in milliseconds (live mode only)'
    )
    
    args = parser.parse_args()
    
    # Create auditor
    auditor = HMARLAuditor(log_file=args.log_file)
    
    # Run monitoring
    if args.live:
        auditor.run_live_monitoring(interval_ms=args.interval)
    else:
        auditor.run_static_analysis()


if __name__ == '__main__':
    main()
