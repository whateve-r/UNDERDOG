"""
Training Visualization Dashboard

Real-time monitoring of MARL training progress:
- Drawdown Máximo por episodio
- Recompensa Media por episodio
- Balance Final por episodio

Usage:
    python scripts/visualize_training.py --log-file logs/training_metrics.csv
"""

import argparse
import time
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pandas as pd
from pathlib import Path
import numpy as np


class TrainingDashboard:
    """Real-time training visualization dashboard"""
    
    def __init__(self, log_file: str):
        """
        Initialize dashboard
        
        Args:
            log_file: Path to CSV log file with columns:
                - episode: Episode number
                - reward: Episode reward
                - final_balance: Final balance
                - global_dd: Global drawdown (%)
        """
        self.log_file = Path(log_file)
        
        # Create figure with 3 subplots
        self.fig, (self.ax1, self.ax2, self.ax3) = plt.subplots(3, 1, figsize=(12, 10))
        self.fig.suptitle('MARL Training Dashboard', fontsize=16, fontweight='bold')
        
        # Configure subplots
        self._setup_plots()
        
    def _setup_plots(self):
        """Configure subplot aesthetics"""
        # Plot 1: Drawdown Máximo
        self.ax1.set_title('[DD] Drawdown Maximo por Episodio', fontsize=12, fontweight='bold')
        self.ax1.set_xlabel('Episodio')
        self.ax1.set_ylabel('Drawdown (%)')
        self.ax1.axhline(y=5.0, color='red', linestyle='--', linewidth=2, label='Limite DD (5%)')
        self.ax1.axhline(y=4.0, color='orange', linestyle='--', linewidth=1, alpha=0.5, label='Warning (4%)')
        self.ax1.grid(True, alpha=0.3)
        self.ax1.legend(loc='upper right')
        self.ax1.set_ylim(0, 6)
        
        # Plot 2: Recompensa Media
        self.ax2.set_title('[REWARD] Recompensa Media por Episodio', fontsize=12, fontweight='bold')
        self.ax2.set_xlabel('Episodio')
        self.ax2.set_ylabel('Reward')
        self.ax2.axhline(y=-1000, color='red', linestyle='--', linewidth=1, alpha=0.5, label='Catastrofico (-1000)')
        self.ax2.axhline(y=0, color='green', linestyle='--', linewidth=1, alpha=0.5, label='Break-even (0)')
        self.ax2.grid(True, alpha=0.3)
        self.ax2.legend(loc='upper right')
        
        # Plot 3: Balance Final
        self.ax3.set_title('[BALANCE] Balance Final por Episodio', fontsize=12, fontweight='bold')
        self.ax3.set_xlabel('Episodio')
        self.ax3.set_ylabel('Balance ($)')
        self.ax3.axhline(y=100000, color='blue', linestyle='--', linewidth=1, alpha=0.5, label='Inicial ($100K)')
        self.ax3.grid(True, alpha=0.3)
        self.ax3.legend(loc='upper right')
        
        plt.tight_layout()
    
    def update(self, frame):
        """Update plots with new data (for animation)"""
        if not self.log_file.exists():
            return
        
        try:
            # Read latest data (with retry for locked files)
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    df = pd.read_csv(self.log_file)
                    break
                except (pd.errors.EmptyDataError, PermissionError):
                    if attempt < max_retries - 1:
                        time.sleep(0.1)  # Wait 100ms and retry
                        continue
                    else:
                        return  # Give up silently
            
            if len(df) == 0:
                return
            
            episodes = df['episode'].values
            rewards = df['reward'].values
            balances = df['final_balance'].values
            dds = df['global_dd'].values * 100  # Convert to percentage
            
            # Clear axes
            self.ax1.clear()
            self.ax2.clear()
            self.ax3.clear()
            
            # Re-setup (to restore labels/grids)
            self._setup_plots()
            
            # Plot 1: Drawdown
            self.ax1.plot(episodes, dds, 'o-', color='darkred', linewidth=2, markersize=4, label='DD Máximo')
            self.ax1.fill_between(episodes, 0, dds, alpha=0.3, color='red')
            
            # Highlight violations
            violations = dds > 5.0
            if violations.any():
                self.ax1.scatter(episodes[violations], dds[violations], 
                               color='red', s=100, marker='X', zorder=5, label='Violación')
            
            # Plot 2: Reward
            self.ax2.plot(episodes, rewards, 'o-', color='darkblue', linewidth=2, markersize=4, label='Reward')
            self.ax2.fill_between(episodes, rewards, 0, alpha=0.3, 
                                 where=(rewards >= 0), color='green', interpolate=True)
            self.ax2.fill_between(episodes, rewards, 0, alpha=0.3, 
                                 where=(rewards < 0), color='red', interpolate=True)
            
            # Plot 3: Balance
            self.ax3.plot(episodes, balances, 'o-', color='darkgreen', linewidth=2, markersize=4, label='Balance')
            self.ax3.fill_between(episodes, 100000, balances, alpha=0.3, 
                                 where=(balances >= 100000), color='green', interpolate=True)
            self.ax3.fill_between(episodes, 100000, balances, alpha=0.3, 
                                 where=(balances < 100000), color='red', interpolate=True)
            
            # Statistics (text annotations)
            last_episode = episodes[-1]
            last_reward = rewards[-1]
            last_balance = balances[-1]
            last_dd = dds[-1]
            
            avg_reward = np.mean(rewards[-10:]) if len(rewards) >= 10 else np.mean(rewards)
            avg_dd = np.mean(dds[-10:]) if len(dds) >= 10 else np.mean(dds)
            
            stats_text = (
                f"Último Episodio: {int(last_episode)}\n"
                f"Reward: {last_reward:.2f} (avg últimos 10: {avg_reward:.2f})\n"
                f"Balance: ${last_balance:,.0f}\n"
                f"DD: {last_dd:.2f}% (avg últimos 10: {avg_dd:.2f}%)"
            )
            
            self.fig.text(0.02, 0.02, stats_text, fontsize=10, 
                         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            
        except KeyError as e:
            print(f"Error: CSV file missing required column {e}. Delete the CSV file and restart training.")
        except Exception as e:
            print(f"Error updating dashboard: {e}")
    
    def show(self, interval: int = 2000):
        """
        Show dashboard with auto-refresh
        
        Args:
            interval: Refresh interval in milliseconds (default: 2000ms = 2s)
        """
        ani = animation.FuncAnimation(self.fig, self.update, interval=interval, cache_frame_data=False)
        plt.show()
    
    def save_snapshot(self, output_path: str):
        """Save current dashboard as image"""
        self.update(0)  # Update once
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Dashboard snapshot saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description='MARL Training Dashboard')
    parser.add_argument('--log-file', type=str, default='logs/training_metrics.csv',
                       help='Path to training metrics CSV file')
    parser.add_argument('--interval', type=int, default=2000,
                       help='Refresh interval in milliseconds')
    parser.add_argument('--snapshot', type=str, default=None,
                       help='Save snapshot to file and exit')
    
    args = parser.parse_args()
    
    dashboard = TrainingDashboard(args.log_file)
    
    if args.snapshot:
        dashboard.save_snapshot(args.snapshot)
    else:
        print(f"[DASHBOARD] Starting MARL Training Dashboard")
        print(f"[FILE] Monitoring: {args.log_file}")
        print(f"[REFRESH] Interval: {args.interval}ms")
        print(f"Press Ctrl+C to exit")
        dashboard.show(interval=args.interval)


if __name__ == '__main__':
    main()
