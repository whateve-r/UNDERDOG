"""
POEL vs Baseline Comparison Analysis

Analyzes training metrics from baseline and POEL runs to validate
POEL effectiveness in:
- Reducing DD breach rate (target: 5x reduction)
- Extending episode length (target: 10-20x improvement)
- Improving Calmar Ratio (target: >1.0)
- Discovering valuable skills
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import argparse


def load_metrics(csv_path: str) -> pd.DataFrame:
    """Load training metrics from CSV"""
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} episodes from {csv_path}")
    return df


def calculate_statistics(df: pd.DataFrame, run_name: str) -> dict:
    """Calculate key statistics from training metrics"""
    
    # Episode metrics (using 'episode' as proxy for length since 'steps' not in CSV)
    total_episodes = len(df)
    avg_episode_idx = df['episode'].mean()
    median_episode_idx = df['episode'].median()
    max_episode_idx = df['episode'].max()
    
    # DD breach rate (using global_dd column)
    dd_breaches = (df['global_dd'] >= 0.10).sum()  # 10% total DD limit
    dd_breach_rate = dd_breaches / len(df) * 100
    
    # Balance metrics (using final_balance column)
    final_balance = df['final_balance'].iloc[-1]
    avg_balance = df['final_balance'].mean()
    balance_volatility = df['final_balance'].std()
    
    # Drawdown metrics (using global_dd column)
    avg_total_dd = df['global_dd'].mean()
    max_total_dd = df['global_dd'].max()
    min_total_dd = df['global_dd'].min()
    
    # Reward metrics (using reward column as proxy for PnL)
    total_reward = df['reward'].sum()
    avg_reward_per_episode = df['reward'].mean()
    
    # Violations tracking
    total_violations = df['violations'].sum() if 'violations' in df.columns else 0
    
    # Calmar Ratio estimation (simplified)
    # Calmar = Annualized Return / Max DD
    if max_total_dd > 0:
        total_return = (final_balance - 100000) / 100000
        calmar_ratio = total_return / max_total_dd
    else:
        calmar_ratio = 0.0
    
    stats = {
        'run_name': run_name,
        'total_episodes': total_episodes,
        
        # Episode metrics (using index as proxy)
        'avg_episode_idx': avg_episode_idx,
        'median_episode_idx': median_episode_idx,
        'max_episode_idx': max_episode_idx,
        
        # DD breach
        'dd_breaches': dd_breaches,
        'dd_breach_rate': dd_breach_rate,
        
        # Balance
        'final_balance': final_balance,
        'avg_balance': avg_balance,
        'balance_volatility': balance_volatility,
        
        # Drawdown
        'avg_total_dd': avg_total_dd,
        'max_total_dd': max_total_dd,
        'min_total_dd': min_total_dd,
        
        # Reward (proxy for PnL)
        'total_reward': total_reward,
        'avg_reward_per_episode': avg_reward_per_episode,
        
        # Violations
        'total_violations': total_violations,
        
        # Calmar
        'calmar_ratio': calmar_ratio,
    }
    
    return stats


def print_comparison_table(baseline_stats: dict, poel_stats: dict):
    """Print formatted comparison table"""
    
    print("\n" + "=" * 100)
    print("POEL VS BASELINE COMPARISON")
    print("=" * 100)
    
    print(f"\n{'Metric':<40} {'Baseline':<20} {'POEL':<20} {'Improvement':<20}")
    print("-" * 100)
    
    # Episode count
    print(f"{'Total Episodes':<40} {baseline_stats['total_episodes']:<20} {poel_stats['total_episodes']:<20} {'N/A':<20}")
    
    # DD breach rate
    baseline_breach = baseline_stats['dd_breach_rate']
    poel_breach = poel_stats['dd_breach_rate']
    if poel_breach > 0:
        reduction = baseline_breach / poel_breach
    else:
        reduction = float('inf') if baseline_breach > 0 else 1.0
    print(f"{'DD Breach Rate (%)':<40} {baseline_breach:<20.2f} {poel_breach:<20.2f} {reduction:<20.2f}x reduction")
    
    # Calmar Ratio
    baseline_calmar = baseline_stats['calmar_ratio']
    poel_calmar = poel_stats['calmar_ratio']
    improvement = poel_calmar - baseline_calmar
    print(f"{'Calmar Ratio':<40} {baseline_calmar:<20.2f} {poel_calmar:<20.2f} {improvement:+<20.2f}")
    
    # Average DD
    baseline_avg_dd = baseline_stats['avg_total_dd']
    poel_avg_dd = poel_stats['avg_total_dd']
    reduction_pct = ((baseline_avg_dd - poel_avg_dd) / baseline_avg_dd * 100) if baseline_avg_dd > 0 else 0
    print(f"{'Average Total DD (%)':<40} {baseline_avg_dd*100:<20.2f} {poel_avg_dd*100:<20.2f} {reduction_pct:<20.2f}% reduction")
    
    # Max DD
    baseline_max_dd = baseline_stats['max_total_dd']
    poel_max_dd = poel_stats['max_total_dd']
    reduction_pct = ((baseline_max_dd - poel_max_dd) / baseline_max_dd * 100) if baseline_max_dd > 0 else 0
    print(f"{'Max Total DD (%)':<40} {baseline_max_dd*100:<20.2f} {poel_max_dd*100:<20.2f} {reduction_pct:<20.2f}% reduction")
    
    # Final balance
    baseline_balance = baseline_stats['final_balance']
    poel_balance = poel_stats['final_balance']
    diff = poel_balance - baseline_balance
    print(f"{'Final Balance ($)':<40} {baseline_balance:<20,.2f} {poel_balance:<20,.2f} ${diff:+<20,.2f}")
    
    # Total Reward
    baseline_reward = baseline_stats['total_reward']
    poel_reward = poel_stats['total_reward']
    diff = poel_reward - baseline_reward
    print(f"{'Total Reward':<40} {baseline_reward:<20,.2f} {poel_reward:<20,.2f} {diff:+<20,.2f}")
    
    # Violations
    baseline_violations = baseline_stats['total_violations']
    poel_violations = poel_stats['total_violations']
    diff = baseline_violations - poel_violations
    print(f"{'Total Violations':<40} {baseline_violations:<20} {poel_violations:<20} {diff:+<20}")
    
    print("-" * 100)
    
    # Validation summary
    print("\n" + "=" * 100)
    print("VALIDATION SUMMARY")
    print("=" * 100)
    
    # DD breach reduction validation
    target_reduction = 2.0
    if reduction >= target_reduction:
        status = "✅ PASS"
    elif reduction >= 1.5:
        status = f"⚠️  PARTIAL ({reduction:.1f}x vs {target_reduction}x target)"
    else:
        status = f"❌ FAIL ({reduction:.1f}x vs {target_reduction}x target)"
    print(f"DD Breach Rate (2x reduction target):       {status}")
    
    # Calmar Ratio validation
    target_calmar = 1.0
    if poel_calmar >= target_calmar:
        status = "✅ PASS"
    elif poel_calmar >= 0.5:
        status = f"⚠️  PARTIAL ({poel_calmar:.2f} vs {target_calmar} target)"
    else:
        status = f"❌ NEEDS IMPROVEMENT ({poel_calmar:.2f} vs {target_calmar} target)"
    print(f"Calmar Ratio (>1.0 target):                 {status}")
    
    # Balance improvement
    if diff > 0:
        status = f"✅ POSITIVE (${diff:,.2f} gain)"
    elif diff > -5000:
        status = f"⚠️  MINOR LOSS (${diff:,.2f})"
    else:
        status = f"❌ SIGNIFICANT LOSS (${diff:,.2f})"
    print(f"Final Balance vs Baseline:                  {status}")
    
    print("=" * 100)


def plot_comparison(baseline_df: pd.DataFrame, poel_df: pd.DataFrame, output_path: str):
    """Generate comprehensive comparison plots"""
    
    fig, axes = plt.subplots(3, 3, figsize=(20, 15))
    fig.suptitle('POEL vs Baseline Comprehensive Comparison', fontsize=18, fontweight='bold')
    
    # Plot 1: Total DD over time
    ax = axes[0, 0]
    ax.plot(baseline_df['episode'], baseline_df['global_dd'] * 100, 
            label='Baseline', alpha=0.7, linewidth=2, color='red')
    ax.plot(poel_df['episode'], poel_df['global_dd'] * 100, 
            label='POEL', alpha=0.7, linewidth=2, color='green')
    ax.axhline(y=10, color='darkred', linestyle='--', label='DD Limit (10%)', linewidth=2)
    ax.set_xlabel('Episode', fontsize=11)
    ax.set_ylabel('Total DD (%)', fontsize=11)
    ax.set_title('Total Drawdown Over Time', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Balance evolution
    ax = axes[0, 1]
    ax.plot(baseline_df['episode'], baseline_df['final_balance'], 
            label='Baseline', alpha=0.7, linewidth=2, color='red')
    ax.plot(poel_df['episode'], poel_df['final_balance'], 
            label='POEL', alpha=0.7, linewidth=2, color='green')
    ax.axhline(y=100000, color='gray', linestyle='--', label='Initial Balance', linewidth=1)
    ax.set_xlabel('Episode', fontsize=11)
    ax.set_ylabel('Balance ($)', fontsize=11)
    ax.set_title('Balance Evolution', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
    
    # Plot 3: Cumulative Reward
    ax = axes[0, 2]
    baseline_cumulative = baseline_df['reward'].cumsum()
    poel_cumulative = poel_df['reward'].cumsum()
    ax.plot(baseline_df['episode'], baseline_cumulative, 
            label='Baseline', alpha=0.7, linewidth=2, color='red')
    ax.plot(poel_df['episode'], poel_cumulative, 
            label='POEL', alpha=0.7, linewidth=2, color='green')
    ax.axhline(y=0, color='gray', linestyle='--', linewidth=1)
    ax.set_xlabel('Episode', fontsize=11)
    ax.set_ylabel('Cumulative Reward', fontsize=11)
    ax.set_title('Cumulative Reward', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 4: DD Distribution (Histogram)
    ax = axes[1, 0]
    ax.hist(baseline_df['global_dd'] * 100, bins=20, alpha=0.5, 
            label='Baseline', color='red', edgecolor='black')
    ax.hist(poel_df['global_dd'] * 100, bins=20, alpha=0.5, 
            label='POEL', color='green', edgecolor='black')
    ax.axvline(x=10, color='darkred', linestyle='--', label='DD Limit', linewidth=2)
    ax.set_xlabel('Total DD (%)', fontsize=11)
    ax.set_ylabel('Frequency', fontsize=11)
    ax.set_title('Drawdown Distribution', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Plot 5: Reward per Episode
    ax = axes[1, 1]
    ax.plot(baseline_df['episode'], baseline_df['reward'], 
            label='Baseline', alpha=0.6, linewidth=1.5, color='red')
    ax.plot(poel_df['episode'], poel_df['reward'], 
            label='POEL', alpha=0.6, linewidth=1.5, color='green')
    ax.axhline(y=0, color='gray', linestyle='--', linewidth=1)
    ax.set_xlabel('Episode', fontsize=11)
    ax.set_ylabel('Episode Reward', fontsize=11)
    ax.set_title('Reward per Episode', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 6: Violations over time
    ax = axes[1, 2]
    ax.plot(baseline_df['episode'], baseline_df['violations'], 
            label='Baseline', alpha=0.7, linewidth=2, color='red', marker='o', markersize=3)
    ax.plot(poel_df['episode'], poel_df['violations'], 
            label='POEL', alpha=0.7, linewidth=2, color='green', marker='o', markersize=3)
    ax.set_xlabel('Episode', fontsize=11)
    ax.set_ylabel('Violations', fontsize=11)
    ax.set_title('Safety Violations per Episode', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 7: Agent 0 (EURUSD) Balance
    ax = axes[2, 0]
    ax.plot(baseline_df['episode'], baseline_df['agent0_balance'], 
            label='Baseline', alpha=0.7, linewidth=2, color='red')
    ax.plot(poel_df['episode'], poel_df['agent0_balance'], 
            label='POEL', alpha=0.7, linewidth=2, color='green')
    ax.axhline(y=25000, color='gray', linestyle='--', linewidth=1)
    ax.set_xlabel('Episode', fontsize=11)
    ax.set_ylabel('Balance ($)', fontsize=11)
    ax.set_title('Agent 0 (EURUSD) Balance', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
    
    # Plot 8: Agent 2 (XAUUSD) Balance
    ax = axes[2, 1]
    ax.plot(baseline_df['episode'], baseline_df['agent2_balance'], 
            label='Baseline', alpha=0.7, linewidth=2, color='red')
    ax.plot(poel_df['episode'], poel_df['agent2_balance'], 
            label='POEL', alpha=0.7, linewidth=2, color='green')
    ax.axhline(y=25000, color='gray', linestyle='--', linewidth=1)
    ax.set_xlabel('Episode', fontsize=11)
    ax.set_ylabel('Balance ($)', fontsize=11)
    ax.set_title('Agent 2 (XAUUSD) Balance', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
    
    # Plot 9: Performance Summary (Bar Chart)
    ax = axes[2, 2]
    metrics = ['Final\nBalance', 'Avg\nReward', 'DD\nBreaches']
    baseline_values = [
        baseline_df['final_balance'].iloc[-1] / 1000,  # Scale to thousands
        baseline_df['reward'].mean(),
        (baseline_df['global_dd'] >= 0.10).sum()
    ]
    poel_values = [
        poel_df['final_balance'].iloc[-1] / 1000,
        poel_df['reward'].mean(),
        (poel_df['global_dd'] >= 0.10).sum()
    ]
    
    x = np.arange(len(metrics))
    width = 0.35
    ax.bar(x - width/2, baseline_values, width, label='Baseline', color='red', alpha=0.7)
    ax.bar(x + width/2, poel_values, width, label='POEL', color='green', alpha=0.7)
    ax.set_ylabel('Value', fontsize=11)
    ax.set_title('Performance Summary', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=10)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n📊 Comparison plots saved to: {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Compare POEL vs Baseline training metrics')
    parser.add_argument('--baseline', type=str, required=True, help='Path to baseline metrics CSV')
    parser.add_argument('--poel', type=str, required=True, help='Path to POEL metrics CSV')
    parser.add_argument('--output', type=str, default='docs/poel_comparison.png', help='Output plot path')
    
    args = parser.parse_args()
    
    # Load metrics
    print("Loading metrics...")
    baseline_df = load_metrics(args.baseline)
    poel_df = load_metrics(args.poel)
    
    # Calculate statistics
    print("\nCalculating statistics...")
    baseline_stats = calculate_statistics(baseline_df, "Baseline")
    poel_stats = calculate_statistics(poel_df, "POEL")
    
    # Print comparison
    print_comparison_table(baseline_stats, poel_stats)
    
    # Generate plots
    print("\nGenerating comparison plots...")
    plot_comparison(baseline_df, poel_df, args.output)
    
    print("\n✅ Analysis complete!")


if __name__ == "__main__":
    main()
