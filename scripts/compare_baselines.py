"""
Baseline Comparison Script - TD3 vs TD3+Market Awareness+CBRL

Compares performance between:
1. Baseline TD3 (24D, no CBRL, no Market Awareness)
2. Enhanced TD3 (29D, CBRL, Market Awareness features)

Metrics:
- Sharpe Ratio
- Max Drawdown
- Win Rate
- Average Reward
- DD Violation Rate (CMDP compliance)

Usage:
    # Compare baseline vs enhanced (100 episodes each)
    poetry run python scripts/compare_baselines.py --episodes 100
    
    # Quick comparison (20 episodes each)
    poetry run python scripts/compare_baselines.py --episodes 20 --quick
"""

import os
import sys
import logging
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root to path
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

from underdog.rl.agents import TD3Agent, TD3Config, ReplayBuffer
from underdog.rl.environments import ForexTradingEnv, TradingEnvConfig
from underdog.database.timescale.timescale_connector import TimescaleDBConnector

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


class BaselineComparator:
    """Compare baseline TD3 vs enhanced TD3+CBRL+Market Awareness"""
    
    def __init__(
        self,
        symbol: str = 'EURUSD',
        start_date: str = '2024-01-01',
        end_date: str = '2024-08-31'
    ):
        """
        Initialize comparator
        
        Args:
            symbol: Trading symbol
            start_date: Test period start
            end_date: Test period end
        """
        self.symbol = symbol
        self.start_date = start_date
        self.end_date = end_date
        
        # Load database configuration
        config_path = PROJECT_ROOT / 'config' / 'data_providers.yaml'
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        db_config = config.get('timescaledb', {})
        
        # Initialize database connector
        self.db = TimescaleDBConnector(
            host=db_config.get('host', 'localhost'),
            port=db_config.get('port', 5432),
            database=db_config.get('database', 'underdog_trading'),
            user=db_config.get('user', 'underdog'),
            password=db_config.get('password', 'underdog_trading_2024_secure')
        )
        
        # Load historical data once
        self.historical_data = self._load_data()
        
        logger.info("="*80)
        logger.info("BASELINE COMPARATOR INITIALIZED")
        logger.info("="*80)
        logger.info(f"Symbol: {symbol}")
        logger.info(f"Test period: {start_date} to {end_date}")
        logger.info(f"Data points: {len(self.historical_data):,}")
        logger.info("="*80)
    
    def _load_data(self) -> pd.DataFrame:
        """Load OHLCV data from TimescaleDB"""
        query = """
            SELECT time, symbol, open, high, low, close, volume, spread
            FROM ohlcv
            WHERE symbol = %s
              AND timeframe = 'M1'
              AND time >= %s
              AND time <= %s
            ORDER BY time ASC
        """
        
        with self.db.get_connection() as conn:
            df = pd.read_sql_query(
                query,
                conn,
                params=(self.symbol, self.start_date, self.end_date)
            )
        
        if df.empty:
            raise ValueError(f"No data found for {self.symbol}")
        
        return df
    
    def _run_episodes(
        self,
        agent: TD3Agent,
        env: ForexTradingEnv,
        replay_buffer: ReplayBuffer,
        num_episodes: int,
        label: str
    ) -> dict:
        """
        Train and evaluate agent over multiple episodes
        
        Args:
            agent: TD3 agent to train
            env: Trading environment
            replay_buffer: Replay buffer for training
            num_episodes: Number of episodes to run
            label: Label for logging (e.g., "Baseline", "Enhanced")
        
        Returns:
            Dictionary with aggregated metrics
        """
        logger.info(f"\n{'='*80}")
        logger.info(f"TRAINING {label.upper()}: {num_episodes} episodes")
        logger.info(f"{'='*80}")
        
        rewards = []
        sharpe_ratios = []
        drawdowns = []
        win_rates = []
        dd_violations = []
        total_trades = []
        
        for ep in range(1, num_episodes + 1):
            state, _ = env.reset()
            episode_reward = 0
            done = False
            truncated = False
            
            while not (done or truncated):
                # Select action with exploration during training
                action = agent.select_action(state, explore=True)
                next_state, reward, done, truncated, info = env.step(action)
                
                # Store transition in replay buffer
                replay_buffer.add(state, action, reward, next_state, done)
                
                # Train agent if buffer has enough samples
                if len(replay_buffer) >= 256:  # Batch size
                    agent.train(replay_buffer, batch_size=256)
                
                state = next_state
                episode_reward += reward
            
            # Collect metrics
            rewards.append(episode_reward)
            sharpe_ratios.append(info.get('sharpe_ratio', 0.0))
            drawdowns.append(info.get('max_drawdown', 0.0))
            win_rates.append(info.get('win_rate', 0.0))
            dd_violations.append(1 if info.get('dd_violation', False) else 0)
            total_trades.append(info.get('total_trades', 0))
            
            # Log progress every 10 episodes
            if ep % 10 == 0:
                recent_sharpe = np.mean(sharpe_ratios[-10:])
                recent_dd = np.mean(drawdowns[-10:])
                recent_wr = np.mean(win_rates[-10:])
                logger.info(
                    f"[{label}] Episode {ep}/{num_episodes} | "
                    f"Sharpe: {recent_sharpe:.2f} | DD: {recent_dd:.2%} | WR: {recent_wr:.2%}"
                )
        
        # Aggregate results
        results = {
            'label': label,
            'avg_reward': np.mean(rewards),
            'std_reward': np.std(rewards),
            'avg_sharpe': np.mean(sharpe_ratios),
            'std_sharpe': np.std(sharpe_ratios),
            'avg_drawdown': np.mean(drawdowns),
            'std_drawdown': np.std(drawdowns),
            'avg_win_rate': np.mean(win_rates),
            'std_win_rate': np.std(win_rates),
            'dd_violation_rate': np.mean(dd_violations),
            'avg_trades': np.mean(total_trades),
            'raw_rewards': rewards,
            'raw_sharpe': sharpe_ratios,
            'raw_drawdowns': drawdowns,
            'raw_win_rates': win_rates
        }
        
        logger.info(f"\n{label.upper()} RESULTS:")
        logger.info(f"  Sharpe Ratio: {results['avg_sharpe']:.2f} ± {results['std_sharpe']:.2f}")
        logger.info(f"  Max Drawdown: {results['avg_drawdown']:.2%} ± {results['std_drawdown']:.2%}")
        logger.info(f"  Win Rate: {results['avg_win_rate']:.2%} ± {results['std_win_rate']:.2%}")
        logger.info(f"  DD Violations: {results['dd_violation_rate']:.2%}")
        logger.info(f"  Avg Trades: {results['avg_trades']:.0f}")
        
        return results
    
    def compare(self, num_episodes: int = 100) -> dict:
        """
        Compare baseline vs enhanced agent
        
        Args:
            num_episodes: Number of episodes per configuration
        
        Returns:
            Comparison results dictionary
        """
        logger.info("\n" + "="*80)
        logger.info("STARTING COMPARISON")
        logger.info("="*80)
        
        # Create environment config
        env_config = TradingEnvConfig(
            symbol=self.symbol,
            timeframe='M1',
            initial_balance=100_000,
            max_position_size=1.0,
            commission_pct=0.0002,
            use_safety_shield=True,
            max_daily_dd_pct=0.05,
            max_total_dd_pct=0.10,
            max_steps=1000,
            lookback_bars=200
        )
        
        # ========================================
        # 1. BASELINE: TD3 (24D, no CBRL)
        # ========================================
        logger.info("\n📊 CONFIGURATION 1: BASELINE TD3")
        logger.info("  - State Dim: 24D (original features)")
        logger.info("  - CBRL: Disabled")
        logger.info("  - Market Awareness: None")
        
        baseline_agent = TD3Agent(
            config=TD3Config(
                state_dim=24,  # Original state space
                action_dim=1,
                hidden_dim=256,
                use_cbrl=False  # Disable CBRL
            )
        )
        
        baseline_replay = ReplayBuffer(state_dim=24, action_dim=1, max_size=1_000_000)
        
        baseline_env = ForexTradingEnv(
            config=env_config,
            db_connector=self.db,
            historical_data=self.historical_data
        )
        
        baseline_results = self._run_episodes(
            agent=baseline_agent,
            env=baseline_env,
            replay_buffer=baseline_replay,
            num_episodes=num_episodes,
            label="Baseline"
        )
        
        # ========================================
        # 2. ENHANCED: TD3 (29D, CBRL, Market Awareness)
        # ========================================
        logger.info("\n🧠 CONFIGURATION 2: ENHANCED TD3+CBRL+MARKET AWARENESS")
        logger.info("  - State Dim: 29D (24 + 5 cognitive features)")
        logger.info("  - CBRL: Enabled (chaotic exploration)")
        logger.info("  - Market Awareness: VVR, Wick Ratio, Liquidity Phase, Opportunity, Confidence")
        
        enhanced_agent = TD3Agent(
            config=TD3Config(
                state_dim=29,  # Enhanced state space
                action_dim=1,
                hidden_dim=256,
                use_cbrl=True  # Enable CBRL
            )
        )
        
        enhanced_replay = ReplayBuffer(state_dim=29, action_dim=1, max_size=1_000_000)
        
        enhanced_env = ForexTradingEnv(
            config=env_config,
            db_connector=self.db,
            historical_data=self.historical_data
        )
        
        enhanced_results = self._run_episodes(
            agent=enhanced_agent,
            env=enhanced_env,
            replay_buffer=enhanced_replay,
            num_episodes=num_episodes,
            label="Enhanced"
        )
        
        # ========================================
        # 3. COMPARISON ANALYSIS
        # ========================================
        comparison = {
            'baseline': baseline_results,
            'enhanced': enhanced_results,
            'improvements': {
                'sharpe_improvement': (
                    (enhanced_results['avg_sharpe'] - baseline_results['avg_sharpe'])
                    / abs(baseline_results['avg_sharpe'] + 1e-6) * 100
                ),
                'dd_improvement': (
                    (baseline_results['avg_drawdown'] - enhanced_results['avg_drawdown'])
                    / abs(baseline_results['avg_drawdown'] + 1e-6) * 100
                ),
                'wr_improvement': (
                    (enhanced_results['avg_win_rate'] - baseline_results['avg_win_rate'])
                    / abs(baseline_results['avg_win_rate'] + 1e-6) * 100
                ),
                'dd_violation_reduction': (
                    (baseline_results['dd_violation_rate'] - enhanced_results['dd_violation_rate'])
                    * 100
                )
            }
        }
        
        logger.info("\n" + "="*80)
        logger.info("COMPARISON RESULTS")
        logger.info("="*80)
        logger.info(f"Sharpe Improvement: {comparison['improvements']['sharpe_improvement']:+.1f}%")
        logger.info(f"Drawdown Improvement: {comparison['improvements']['dd_improvement']:+.1f}%")
        logger.info(f"Win Rate Improvement: {comparison['improvements']['wr_improvement']:+.1f}%")
        logger.info(f"DD Violation Reduction: {comparison['improvements']['dd_violation_reduction']:+.1f}%")
        logger.info("="*80)
        
        # Determine verdict
        sharpe_better = comparison['improvements']['sharpe_improvement'] > 20
        dd_better = comparison['improvements']['dd_improvement'] > 10
        
        if sharpe_better and dd_better:
            verdict = "✅ ENHANCED VERSION SIGNIFICANTLY BETTER - PROCEED TO MARL"
        elif sharpe_better or dd_better:
            verdict = "⚠️ MIXED RESULTS - CONSIDER HYPERPARAMETER TUNING"
        else:
            verdict = "❌ NO SIGNIFICANT IMPROVEMENT - REVIEW FEATURE ENGINEERING"
        
        logger.info(f"\n🎯 VERDICT: {verdict}\n")
        
        comparison['verdict'] = verdict
        
        return comparison
    
    def plot_comparison(self, comparison: dict, save_path: str = None):
        """
        Plot comparison results
        
        Args:
            comparison: Comparison results from compare()
            save_path: Optional path to save plot
        """
        baseline = comparison['baseline']
        enhanced = comparison['enhanced']
        
        # Create figure with 4 subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Baseline vs Enhanced TD3 Comparison', fontsize=16, fontweight='bold')
        
        # Sharpe Ratio comparison
        axes[0, 0].bar(['Baseline', 'Enhanced'], 
                      [baseline['avg_sharpe'], enhanced['avg_sharpe']],
                      yerr=[baseline['std_sharpe'], enhanced['std_sharpe']],
                      color=['#3498db', '#2ecc71'], alpha=0.7)
        axes[0, 0].set_title('Sharpe Ratio')
        axes[0, 0].set_ylabel('Sharpe Ratio')
        axes[0, 0].grid(axis='y', alpha=0.3)
        
        # Drawdown comparison
        axes[0, 1].bar(['Baseline', 'Enhanced'], 
                      [baseline['avg_drawdown'], enhanced['avg_drawdown']],
                      yerr=[baseline['std_drawdown'], enhanced['std_drawdown']],
                      color=['#e74c3c', '#f39c12'], alpha=0.7)
        axes[0, 1].set_title('Max Drawdown')
        axes[0, 1].set_ylabel('Drawdown (%)')
        axes[0, 1].yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.1%}'))
        axes[0, 1].grid(axis='y', alpha=0.3)
        
        # Win Rate comparison
        axes[1, 0].bar(['Baseline', 'Enhanced'], 
                      [baseline['avg_win_rate'], enhanced['avg_win_rate']],
                      yerr=[baseline['std_win_rate'], enhanced['std_win_rate']],
                      color=['#9b59b6', '#1abc9c'], alpha=0.7)
        axes[1, 0].set_title('Win Rate')
        axes[1, 0].set_ylabel('Win Rate (%)')
        axes[1, 0].yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.1%}'))
        axes[1, 0].grid(axis='y', alpha=0.3)
        
        # DD Violation Rate comparison
        axes[1, 1].bar(['Baseline', 'Enhanced'], 
                      [baseline['dd_violation_rate'], enhanced['dd_violation_rate']],
                      color=['#e74c3c', '#2ecc71'], alpha=0.7)
        axes[1, 1].set_title('DD Violation Rate')
        axes[1, 1].set_ylabel('Violation Rate (%)')
        axes[1, 1].yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.1%}'))
        axes[1, 1].grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Plot saved: {save_path}")
        
        plt.show()


def main():
    """Main execution"""
    parser = argparse.ArgumentParser(
        description='Compare Baseline TD3 vs Enhanced TD3+CBRL+Market Awareness'
    )
    
    parser.add_argument(
        '--episodes',
        type=int,
        default=100,
        help='Number of episodes per configuration (default: 100)'
    )
    
    parser.add_argument(
        '--symbol',
        type=str,
        default='EURUSD',
        help='Trading symbol (default: EURUSD)'
    )
    
    parser.add_argument(
        '--start-date',
        type=str,
        default='2024-01-01',
        help='Test period start (default: 2024-01-01)'
    )
    
    parser.add_argument(
        '--end-date',
        type=str,
        default='2024-08-31',
        help='Test period end (default: 2024-08-31)'
    )
    
    parser.add_argument(
        '--quick',
        action='store_true',
        help='Quick test mode (20 episodes)'
    )
    
    parser.add_argument(
        '--plot',
        action='store_true',
        help='Generate comparison plots'
    )
    
    args = parser.parse_args()
    
    # Override episodes for quick mode
    if args.quick:
        args.episodes = 20
        logger.info("🚀 QUICK MODE: 20 episodes per configuration")
    
    # Initialize comparator
    comparator = BaselineComparator(
        symbol=args.symbol,
        start_date=args.start_date,
        end_date=args.end_date
    )
    
    # Run comparison
    try:
        results = comparator.compare(num_episodes=args.episodes)
        
        # Generate plots if requested
        if args.plot:
            save_path = PROJECT_ROOT / 'data' / 'test_results' / 'baseline_comparison.png'
            save_path.parent.mkdir(exist_ok=True, parents=True)
            comparator.plot_comparison(results, save_path=str(save_path))
        
        # Save results to JSON
        import json
        results_path = PROJECT_ROOT / 'data' / 'test_results' / 'baseline_comparison.json'
        results_path.parent.mkdir(exist_ok=True, parents=True)
        
        # Remove raw data (too large for JSON)
        results_clean = {
            'baseline': {k: v for k, v in results['baseline'].items() if not k.startswith('raw_')},
            'enhanced': {k: v for k, v in results['enhanced'].items() if not k.startswith('raw_')},
            'improvements': results['improvements'],
            'verdict': results['verdict']
        }
        
        with open(results_path, 'w') as f:
            json.dump(results_clean, f, indent=2)
        
        logger.info(f"\n✅ Results saved: {results_path}")
    
    except KeyboardInterrupt:
        logger.warning("\nComparison interrupted by user")
    
    except Exception as e:
        logger.error(f"Comparison failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
