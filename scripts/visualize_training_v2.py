"""
UNDERDOG Trading Systems - Professional MARL Training Dashboard

Real-time monitoring dashboard with:
- Portfolio metrics (Balance, DD, Reward)
- Per-agent performance (4x TD3 agents)
- Training dynamics (TD Error, Epsilon)
- CMDP compliance tracking

Usage:
    python scripts/visualize_training_v2.py --log-file logs/training_metrics.csv
"""

import argparse
import time
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.gridspec as gridspec
import pandas as pd
from pathlib import Path
import numpy as np
from datetime import datetime


class EnhancedMARLDashboard:
    """
    Professional MARL training dashboard - Black theme, Consolas font
    
    Layout (3x3 grid):
    +------------------+------------------+------------------+
    |  Portfolio DD    |  Reward Curve    |  Balance Curve   |
    +------------------+------------------+------------------+
    |  CMDP Violations |  Agent Rewards   |  Agent Balances  |
    +------------------+------------------+------------------+
    |  Training Stats  |  TD Error        |  Epsilon Decay   |
    +------------------+------------------+------------------+
    """
    
    def __init__(self, log_file: str):
        """Initialize professional dashboard"""
        self.log_file = Path(log_file)
        
        # Professional styling
        plt.style.use('dark_background')
        plt.rcParams['font.family'] = 'Consolas'
        plt.rcParams['font.size'] = 9
        plt.rcParams['figure.facecolor'] = '#000000'
        plt.rcParams['axes.facecolor'] = '#0a0a0a'
        plt.rcParams['axes.edgecolor'] = '#333333'
        plt.rcParams['axes.labelcolor'] = '#00ff00'
        plt.rcParams['xtick.color'] = '#00ff00'
        plt.rcParams['ytick.color'] = '#00ff00'
        plt.rcParams['grid.color'] = '#1a1a1a'
        plt.rcParams['grid.alpha'] = 0.5
        
        # Create figure with 3x3 grid
        self.fig = plt.figure(figsize=(20, 12))
        
        # Title with timestamp
        title_text = f'UNDERDOG TRADING SYSTEMS | MARL/td3 | {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}'
        self.fig.suptitle(title_text, fontsize=12, fontweight='bold', 
                         color='#00ff00', fontfamily='Consolas')
        
        # Create grid layout
        gs = gridspec.GridSpec(3, 3, figure=self.fig, hspace=0.3, wspace=0.3)
        
        # Row 1: Portfolio Metrics
        self.ax_dd = self.fig.add_subplot(gs[0, 0])
        self.ax_reward = self.fig.add_subplot(gs[0, 1])
        self.ax_balance = self.fig.add_subplot(gs[0, 2])
        
        # Row 2: Agent-level Metrics
        self.ax_violations = self.fig.add_subplot(gs[1, 0])
        self.ax_agent_rewards = self.fig.add_subplot(gs[1, 1])
        self.ax_agent_balance = self.fig.add_subplot(gs[1, 2])
        
        # Row 3: Training Dynamics
        self.ax_stats = self.fig.add_subplot(gs[2, 0])
        self.ax_td_error = self.fig.add_subplot(gs[2, 1])
        self.ax_epsilon = self.fig.add_subplot(gs[2, 2])
        
        self._setup_plots()
        
    def _setup_plots(self):
        """Setup all subplot styles and labels"""
        
        # === ROW 1: Portfolio Metrics ===
        
        # Plot 1: Drawdown
        self.ax_dd.set_title('[DD] Drawdown Maximo (CMDP)', fontweight='bold', color='#ff3333')
        self.ax_dd.set_xlabel('Episode')
        self.ax_dd.set_ylabel('DD (%)')
        self.ax_dd.axhline(y=5.0, color='#ff3333', linestyle='--', linewidth=2, label='Limit 5%')
        self.ax_dd.grid(True, alpha=0.5)
        self.ax_dd.legend(loc='upper left', fontsize=8)
        
        # Plot 2: Reward
        self.ax_reward.set_title('[REWARD] Mean Episode Return', fontweight='bold', color='#33ff33')
        self.ax_reward.set_xlabel('Episode')
        self.ax_reward.set_ylabel('Reward')
        self.ax_reward.axhline(y=0, color='#666666', linestyle='-', linewidth=1, alpha=0.5)
        self.ax_reward.grid(True, alpha=0.5)
        
        # Plot 3: Balance
        self.ax_balance.set_title('[BALANCE] Portfolio Equity', fontweight='bold', color='#3399ff')
        self.ax_balance.set_xlabel('Episode')
        self.ax_balance.set_ylabel('Balance ($)')
        self.ax_balance.axhline(y=100000, color='#666666', linestyle='--', linewidth=1, alpha=0.5, label='Initial')
        self.ax_balance.grid(True, alpha=0.5)
        self.ax_balance.legend(loc='lower left', fontsize=8)
        
        # === ROW 2: Agent-level Metrics ===
        
        # Plot 4: CMDP Violations
        self.ax_violations.set_title('[VIOLATIONS] CMDP Constraint Breaks', fontweight='bold', color='#ffaa33')
        self.ax_violations.set_xlabel('Episode')
        self.ax_violations.set_ylabel('Cumulative')
        self.ax_violations.grid(True, alpha=0.5)
        
        # Plot 5: Agent Rewards (per TD3)
        self.ax_agent_rewards.set_title('[AGENTS] TD3 Rewards', fontweight='bold', color='#aa33ff')
        self.ax_agent_rewards.set_xlabel('Episode')
        self.ax_agent_rewards.set_ylabel('Avg Reward')
        self.ax_agent_rewards.grid(True, alpha=0.5)
        
        # Plot 6: Agent Balance Distribution
        self.ax_agent_balance.set_title('[AGENTS] TD3 Balances', fontweight='bold', color='#33ffaa')
        self.ax_agent_balance.set_xlabel('Episode')
        self.ax_agent_balance.set_ylabel('Balance ($)')
        self.ax_agent_balance.grid(True, alpha=0.5)
        
        # === ROW 3: Training Dynamics ===
        
        # Plot 7: Training Stats (text info)
        self.ax_stats.axis('off')
        self.ax_stats.set_title('[STATS] Training Metrics', fontweight='bold', color='#ffff33')
        
        # Plot 8: TD Error
        self.ax_td_error.set_title('[TD-ERROR] Critic Loss', fontweight='bold', color='#ff33aa')
        self.ax_td_error.set_xlabel('Episode')
        self.ax_td_error.set_ylabel('Mean Loss')
        self.ax_td_error.grid(True, alpha=0.5)
        
        # Plot 9: Epsilon Decay
        self.ax_epsilon.set_title('[EPSILON] Exploration Rate', fontweight='bold', color='#33aaff')
        self.ax_epsilon.set_xlabel('Episode')
        self.ax_epsilon.set_ylabel('Epsilon')
        self.ax_epsilon.grid(True, alpha=0.5)
    
    def update(self, frame):
        """Update all plots with latest data"""
        if not self.log_file.exists():
            return
        
        try:
            # Read data with retry logic
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    df = pd.read_csv(self.log_file)
                    break
                except (pd.errors.EmptyDataError, PermissionError):
                    if attempt < max_retries - 1:
                        time.sleep(0.1)
                        continue
                    else:
                        return
            
            if len(df) == 0:
                return
            
            episodes = df['episode'].values
            rewards = df['reward'].values
            balances = df['final_balance'].values
            dds = df['global_dd'].values * 100  # Convert to percentage
            
            # Per-agent metrics
            agent0_rewards = df['agent0_reward'].values if 'agent0_reward' in df.columns else np.zeros(len(episodes))
            agent1_rewards = df['agent1_reward'].values if 'agent1_reward' in df.columns else np.zeros(len(episodes))
            agent2_rewards = df['agent2_reward'].values if 'agent2_reward' in df.columns else np.zeros(len(episodes))
            agent3_rewards = df['agent3_reward'].values if 'agent3_reward' in df.columns else np.zeros(len(episodes))
            
            agent0_balance = df['agent0_balance'].values if 'agent0_balance' in df.columns else np.ones(len(episodes)) * 25000
            agent1_balance = df['agent1_balance'].values if 'agent1_balance' in df.columns else np.ones(len(episodes)) * 25000
            agent2_balance = df['agent2_balance'].values if 'agent2_balance' in df.columns else np.ones(len(episodes)) * 25000
            agent3_balance = df['agent3_balance'].values if 'agent3_balance' in df.columns else np.ones(len(episodes)) * 25000
            
            # Training dynamics
            avg_epsilon = df['avg_epsilon'].values if 'avg_epsilon' in df.columns else np.ones(len(episodes))
            avg_td_error = df['avg_td_error'].values if 'avg_td_error' in df.columns else np.zeros(len(episodes))
            violations = df['violations'].values if 'violations' in df.columns else (dds > 5.0).astype(int)
            
            # === UPDATE ROW 1: Portfolio Metrics ===
            
            # Clear and re-setup
            self.ax_dd.clear()
            self.ax_reward.clear()
            self.ax_balance.clear()
            
            # Plot 1: Drawdown
            self.ax_dd.set_title('[DD] Drawdown Maximo (CMDP)', fontweight='bold', color='#ff3333')
            self.ax_dd.set_xlabel('Episode')
            self.ax_dd.set_ylabel('DD (%)')
            self.ax_dd.plot(episodes, dds, '-', color='#ff6666', linewidth=2, label='DD')
            self.ax_dd.fill_between(episodes, 0, dds, alpha=0.3, color='#ff3333')
            self.ax_dd.axhline(y=5.0, color='#ff0000', linestyle='--', linewidth=2, label='Limit 5%')
            
            # Highlight violations
            dd_violations = dds > 5.0
            if dd_violations.any():
                self.ax_dd.scatter(episodes[dd_violations], dds[dd_violations], 
                                  color='#ff0000', s=80, marker='x', zorder=5, label='Violation')
            
            self.ax_dd.grid(True, alpha=0.5)
            self.ax_dd.legend(loc='upper left', fontsize=8)
            
            # Plot 2: Reward
            self.ax_reward.set_title('[REWARD] Mean Episode Return', fontweight='bold', color='#33ff33')
            self.ax_reward.set_xlabel('Episode')
            self.ax_reward.set_ylabel('Reward')
            self.ax_reward.plot(episodes, rewards, '-', color='#66ff66', linewidth=2)
            self.ax_reward.axhline(y=0, color='#666666', linestyle='-', linewidth=1, alpha=0.5)
            
            # Color fill by sign
            self.ax_reward.fill_between(episodes, rewards, 0, alpha=0.3, 
                                       where=(rewards >= 0), color='#33ff33', interpolate=True)
            self.ax_reward.fill_between(episodes, rewards, 0, alpha=0.3, 
                                       where=(rewards < 0), color='#ff3333', interpolate=True)
            
            # Add moving average
            if len(rewards) >= 10:
                ma_10 = pd.Series(rewards).rolling(window=10, min_periods=1).mean()
                self.ax_reward.plot(episodes, ma_10, '--', color='#ffaa33', linewidth=2, label='MA(10)')
                self.ax_reward.legend(loc='lower right', fontsize=8)
            
            self.ax_reward.grid(True, alpha=0.5)
            
            # Plot 3: Balance
            self.ax_balance.set_title('[BALANCE] Portfolio Equity', fontweight='bold', color='#3399ff')
            self.ax_balance.set_xlabel('Episode')
            self.ax_balance.set_ylabel('Balance ($)')
            self.ax_balance.plot(episodes, balances, '-', color='#66aaff', linewidth=2)
            self.ax_balance.axhline(y=100000, color='#666666', linestyle='--', linewidth=1, alpha=0.5, label='Initial')
            
            # Color fill
            self.ax_balance.fill_between(episodes, 100000, balances, alpha=0.3, 
                                        where=(balances >= 100000), color='#33ff33', interpolate=True)
            self.ax_balance.fill_between(episodes, 100000, balances, alpha=0.3, 
                                        where=(balances < 100000), color='#ff3333', interpolate=True)
            
            self.ax_balance.grid(True, alpha=0.5)
            self.ax_balance.legend(loc='lower left', fontsize=8)
            
            # === UPDATE ROW 2: Agent Metrics (Placeholder for now) ===
            
            # Plot 4: Violations count
            self.ax_violations.clear()
            self.ax_violations.set_title('[VIOLATIONS] CMDP Constraint Breaks', fontweight='bold', color='#ffaa33')
            self.ax_violations.set_xlabel('Episode')
            self.ax_violations.set_ylabel('Cumulative')
            violation_count = np.cumsum(violations.astype(int))
            self.ax_violations.plot(episodes, violation_count, '-', color='#ffaa33', linewidth=2)
            self.ax_violations.fill_between(episodes, 0, violation_count, alpha=0.3, color='#ff6633')
            self.ax_violations.grid(True, alpha=0.5)
            
            # Plot 5: Agent Rewards
            self.ax_agent_rewards.clear()
            self.ax_agent_rewards.set_title('[AGENTS] TD3 Rewards', fontweight='bold', color='#aa33ff')
            self.ax_agent_rewards.set_xlabel('Episode')
            self.ax_agent_rewards.set_ylabel('Avg Reward')
            self.ax_agent_rewards.plot(episodes, agent0_rewards, '-', color='#ff6666', linewidth=1.5, label='EURUSD', alpha=0.8)
            self.ax_agent_rewards.plot(episodes, agent1_rewards, '-', color='#66ff66', linewidth=1.5, label='GBPUSD', alpha=0.8)
            self.ax_agent_rewards.plot(episodes, agent2_rewards, '-', color='#6666ff', linewidth=1.5, label='USDJPY', alpha=0.8)
            self.ax_agent_rewards.plot(episodes, agent3_rewards, '-', color='#ffff66', linewidth=1.5, label='XAUUSD', alpha=0.8)
            self.ax_agent_rewards.legend(loc='upper right', fontsize=7, ncol=2)
            self.ax_agent_rewards.grid(True, alpha=0.5)
            
            # Plot 6: Agent Balances
            self.ax_agent_balance.clear()
            self.ax_agent_balance.set_title('[AGENTS] TD3 Balances', fontweight='bold', color='#33ffaa')
            self.ax_agent_balance.set_xlabel('Episode')
            self.ax_agent_balance.set_ylabel('Balance ($)')
            self.ax_agent_balance.plot(episodes, agent0_balance, '-', color='#ff6666', linewidth=1.5, label='EURUSD', alpha=0.8)
            self.ax_agent_balance.plot(episodes, agent1_balance, '-', color='#66ff66', linewidth=1.5, label='GBPUSD', alpha=0.8)
            self.ax_agent_balance.plot(episodes, agent2_balance, '-', color='#6666ff', linewidth=1.5, label='USDJPY', alpha=0.8)
            self.ax_agent_balance.plot(episodes, agent3_balance, '-', color='#ffff66', linewidth=1.5, label='XAUUSD', alpha=0.8)
            self.ax_agent_balance.axhline(y=25000, color='#666666', linestyle='--', linewidth=1, alpha=0.5)
            self.ax_agent_balance.legend(loc='upper right', fontsize=7, ncol=2)
            self.ax_agent_balance.grid(True, alpha=0.5)
            
            # === UPDATE ROW 3: Training Dynamics ===
            
            # Plot 7: Stats text
            self.ax_stats.clear()
            self.ax_stats.axis('off')
            
            last_episode = int(episodes[-1])
            last_reward = rewards[-1]
            last_balance = balances[-1]
            last_dd = dds[-1]
            
            # Calculate stats
            avg_reward_10 = np.mean(rewards[-10:]) if len(rewards) >= 10 else np.mean(rewards)
            avg_dd_10 = np.mean(dds[-10:]) if len(dds) >= 10 else np.mean(dds)
            total_violations = int(violation_count[-1]) if len(violation_count) > 0 else 0
            violation_rate = (total_violations / last_episode * 100) if last_episode > 0 else 0
            
            profit_loss = last_balance - 100000
            profit_pct = (profit_loss / 100000) * 100
            
            last_epsilon = avg_epsilon[-1] if len(avg_epsilon) > 0 else 1.0
            last_td_error = avg_td_error[-1] if len(avg_td_error) > 0 else 0.0
            
            stats_text = f"""
[TRAINING STATISTICS]
{'='*40}
Episode: {last_episode}
Total Steps: {last_episode * 400} (approx)

[PERFORMANCE]
Reward (last): {last_reward:.2f}
Reward (MA-10): {avg_reward_10:.2f}
Balance: ${last_balance:,.2f}
P&L: ${profit_loss:+,.2f} ({profit_pct:+.2f}%)

[RISK MANAGEMENT]
DD (last): {last_dd:.2f}%
DD (MA-10): {avg_dd_10:.2f}%
CMDP Violations: {total_violations}
Violation Rate: {violation_rate:.1f}%

[LEARNING DYNAMICS]
Epsilon: {last_epsilon:.4f}
TD Error: {last_td_error:.4f}

[OBJECTIVE]
Maximize return while DD < 5%
            """
            
            self.ax_stats.text(0.05, 0.95, stats_text, 
                             transform=self.ax_stats.transAxes,
                             fontsize=8,
                             verticalalignment='top',
                             fontfamily='Consolas',
                             color='#00ff00',
                             bbox=dict(boxstyle='round', facecolor='#0a0a0a', 
                                     edgecolor='#333333', alpha=0.9))
            
            # Plot 8: TD Error
            self.ax_td_error.clear()
            self.ax_td_error.set_title('[TD-ERROR] Critic Loss', fontweight='bold', color='#ff33aa')
            self.ax_td_error.set_xlabel('Episode')
            self.ax_td_error.set_ylabel('Mean Loss')
            self.ax_td_error.plot(episodes, avg_td_error, '-', color='#ff66aa', linewidth=2)
            self.ax_td_error.fill_between(episodes, 0, avg_td_error, alpha=0.3, color='#ff33aa')
            self.ax_td_error.grid(True, alpha=0.5)
            
            # Plot 9: Epsilon
            self.ax_epsilon.clear()
            self.ax_epsilon.set_title('[EPSILON] Exploration Rate', fontweight='bold', color='#33aaff')
            self.ax_epsilon.set_xlabel('Episode')
            self.ax_epsilon.set_ylabel('Epsilon')
            self.ax_epsilon.plot(episodes, avg_epsilon, '-', color='#66aaff', linewidth=2)
            self.ax_epsilon.fill_between(episodes, 0, avg_epsilon, alpha=0.3, color='#33aaff')
            self.ax_epsilon.axhline(y=0.1, color='#666666', linestyle='--', linewidth=1, alpha=0.5, label='Min')
            self.ax_epsilon.legend(loc='upper right', fontsize=8)
            self.ax_epsilon.grid(True, alpha=0.5)
            
        except KeyError as e:
            print(f"Error: CSV missing column {e}")
        except Exception as e:
            print(f"Error updating dashboard: {e}")
    
    def show(self, interval: int = 2000):
        """Show dashboard with auto-refresh"""
        ani = animation.FuncAnimation(self.fig, self.update, interval=interval, cache_frame_data=False)
        plt.show()


def main():
    parser = argparse.ArgumentParser(description='Enhanced MARL Training Dashboard')
    parser.add_argument('--log-file', type=str, default='logs/training_metrics.csv',
                       help='Path to training metrics CSV file')
    parser.add_argument('--interval', type=int, default=2000,
                       help='Refresh interval in milliseconds')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("UNDERDOG TRADING SYSTEMS")
    print("Professional MARL Training Dashboard")
    print("=" * 60)
    print(f"[FILE] Monitoring: {args.log_file}")
    print(f"[REFRESH] Interval: {args.interval}ms")
    print(f"[THEME] Dark Professional (Consolas)")
    print(f"\nPress Ctrl+C to exit\n")
    
    dashboard = EnhancedMARLDashboard(args.log_file)
    dashboard.show(interval=args.interval)


if __name__ == '__main__':
    main()
