"""
Quick Comparison Script - Baseline TD3 vs Enhanced TD3 (LIGHTWEIGHT VERSION)

Compares training convergence between:
1. Baseline TD3 (24D, no CBRL)
2. Enhanced TD3 (29D, CBRL, Market Awareness)

NO external plotting dependencies (matplotlib/seaborn removed to avoid scipy conflicts)

Usage:
    poetry run python scripts/quick_compare.py --episodes 20
"""

import os
import sys
import logging
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import yaml

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


def train_agent(
    agent: TD3Agent,
    env: ForexTradingEnv,
    replay_buffer: ReplayBuffer,
    num_episodes: int,
    label: str
) -> dict:
    """
    Train agent and collect metrics
    
    Args:
        agent: TD3 agent
        env: Trading environment
        replay_buffer: Replay buffer
        num_episodes: Number of training episodes
        label: Configuration label
    
    Returns:
        Training metrics dictionary
    """
    logger.info(f"\n{'='*80}")
    logger.info(f"TRAINING {label.upper()}: {num_episodes} episodes")
    logger.info(f"{'='*80}")
    
    rewards = []
    sharpe_ratios = []
    drawdowns = []
    win_rates = []
    dd_violations = []
    
    for ep in range(1, num_episodes + 1):
        state, _ = env.reset()
        episode_reward = 0
        done = False
        truncated = False
        steps = 0
        
        while not (done or truncated):
            # Select action with exploration
            action = agent.select_action(state, explore=True)
            next_state, reward, done, truncated, info = env.step(action)
            
            # Store transition
            replay_buffer.add(state, action, reward, next_state, done)
            
            # Train if enough samples
            if len(replay_buffer) >= 256:
                agent.train(replay_buffer, batch_size=256)
            
            state = next_state
            episode_reward += reward
            steps += 1
        
        # Collect metrics
        rewards.append(episode_reward)
        sharpe_ratios.append(info.get('sharpe_ratio', 0.0))
        drawdowns.append(info.get('max_drawdown', 0.0))
        win_rates.append(info.get('win_rate', 0.0))
        dd_violations.append(1 if info.get('dd_violation', False) else 0)
        
        # Log progress
        if ep % 5 == 0:
            recent_sharpe = np.mean(sharpe_ratios[-5:])
            recent_dd = np.mean(drawdowns[-5:])
            recent_wr = np.mean(win_rates[-5:])
            logger.info(
                f"[{label}] Episode {ep}/{num_episodes} | "
                f"Sharpe: {recent_sharpe:.2f} | DD: {recent_dd:.2%} | WR: {recent_wr:.2%}"
            )
    
    # Aggregate results
    results = {
        'label': label,
        'avg_reward': float(np.mean(rewards)),
        'std_reward': float(np.std(rewards)),
        'avg_sharpe': float(np.mean(sharpe_ratios)),
        'std_sharpe': float(np.std(sharpe_ratios)),
        'avg_drawdown': float(np.mean(drawdowns)),
        'std_drawdown': float(np.std(drawdowns)),
        'avg_win_rate': float(np.mean(win_rates)),
        'std_win_rate': float(np.std(win_rates)),
        'dd_violation_rate': float(np.mean(dd_violations)),
        'rewards_history': [float(r) for r in rewards],
        'sharpe_history': [float(s) for s in sharpe_ratios]
    }
    
    logger.info(f"\n{label.upper()} RESULTS:")
    logger.info(f"  Sharpe Ratio: {results['avg_sharpe']:.2f} ± {results['std_sharpe']:.2f}")
    logger.info(f"  Max Drawdown: {results['avg_drawdown']:.2%} ± {results['std_drawdown']:.2%}")
    logger.info(f"  Win Rate: {results['avg_win_rate']:.2%} ± {results['std_win_rate']:.2%}")
    logger.info(f"  DD Violations: {results['dd_violation_rate']:.2%}")
    
    return results


def main():
    """Main execution"""
    parser = argparse.ArgumentParser(description='Quick TD3 comparison')
    parser.add_argument('--episodes', type=int, default=20, help='Episodes per config')
    parser.add_argument('--symbol', type=str, default='EURUSD', help='Trading symbol')
    parser.add_argument('--start-date', type=str, default='2024-01-01', help='Start date')
    parser.add_argument('--end-date', type=str, default='2024-08-31', help='End date')
    
    args = parser.parse_args()
    
    logger.info("="*80)
    logger.info("QUICK COMPARISON: BASELINE vs ENHANCED TD3")
    logger.info("="*80)
    logger.info(f"Episodes per config: {args.episodes}")
    logger.info(f"Symbol: {args.symbol}")
    logger.info(f"Period: {args.start_date} to {args.end_date}")
    logger.info("="*80)
    
    # Load database config
    config_path = PROJECT_ROOT / 'config' / 'data_providers.yaml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    db_config = config.get('timescaledb', {})
    
    # Initialize database
    db = TimescaleDBConnector(
        host=db_config.get('host', 'localhost'),
        port=db_config.get('port', 5432),
        database=db_config.get('database', 'underdog_trading'),
        user=db_config.get('user', 'underdog'),
        password=db_config.get('password', 'underdog_trading_2024_secure')
    )
    
    # Load historical data
    logger.info("Loading historical data...")
    query = """
        SELECT time, symbol, open, high, low, close, volume, spread
        FROM ohlcv
        WHERE symbol = %s AND timeframe = 'M1'
          AND time >= %s AND time <= %s
        ORDER BY time ASC
    """
    
    with db.get_connection() as conn:
        historical_data = pd.read_sql_query(
            query, conn,
            params=(args.symbol, args.start_date, args.end_date)
        )
    
    logger.info(f"Loaded {len(historical_data):,} bars")
    
    if historical_data.empty:
        logger.error("No data found!")
        return
    
    # Environment config
    env_config = TradingEnvConfig(
        symbol=args.symbol,
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
    # 1. BASELINE: TD3 (29D, no CBRL)
    # ========================================
    logger.info("\n📊 CONFIG 1: BASELINE TD3")
    logger.info("  State: 29D | CBRL: OFF | Market Awareness: ON (using same features)")
    logger.info("  Note: Environment always outputs 29D. Baseline differs only by CBRL=OFF")
    
    baseline_agent = TD3Agent(
        config=TD3Config(
            state_dim=29,  # ✅ MUST match environment output (29D)
            action_dim=1,
            hidden_dim=256,
            use_cbrl=False  # 🎯 KEY DIFFERENCE: No chaos-based exploration
        )
    )
    
    baseline_replay = ReplayBuffer(state_dim=29, action_dim=1, max_size=1_000_000)
    baseline_env = ForexTradingEnv(
        config=env_config,
        db_connector=db,
        historical_data=historical_data
    )
    
    baseline_results = train_agent(
        baseline_agent, baseline_env, baseline_replay,
        args.episodes, "Baseline"
    )
    
    # ========================================
    # 2. ENHANCED: TD3 (29D, CBRL enabled)
    # ========================================
    logger.info("\n🧠 CONFIG 2: ENHANCED TD3")
    logger.info("  State: 29D | CBRL: ON | Market Awareness: ON")
    logger.info("  Note: Uses CBRL (chaotic noise) for intelligent exploration")
    
    enhanced_agent = TD3Agent(
        config=TD3Config(
            state_dim=29,  # ✅ Same state space as baseline
            action_dim=1,
            hidden_dim=256,
            use_cbrl=True  # 🎯 KEY DIFFERENCE: Chaos-based exploration enabled
        )
    )
    
    enhanced_replay = ReplayBuffer(state_dim=29, action_dim=1, max_size=1_000_000)
    enhanced_env = ForexTradingEnv(
        config=env_config,
        db_connector=db,
        historical_data=historical_data
    )
    
    enhanced_results = train_agent(
        enhanced_agent, enhanced_env, enhanced_replay,
        args.episodes, "Enhanced"
    )
    
    # ========================================
    # 3. COMPARISON ANALYSIS
    # ========================================
    sharpe_improvement = (
        (enhanced_results['avg_sharpe'] - baseline_results['avg_sharpe'])
        / (abs(baseline_results['avg_sharpe']) + 1e-6) * 100
    )
    
    dd_improvement = (
        (baseline_results['avg_drawdown'] - enhanced_results['avg_drawdown'])
        / (abs(baseline_results['avg_drawdown']) + 1e-6) * 100
    )
    
    wr_improvement = (
        (enhanced_results['avg_win_rate'] - baseline_results['avg_win_rate'])
        / (abs(baseline_results['avg_win_rate']) + 1e-6) * 100
    )
    
    dd_violation_reduction = (
        (baseline_results['dd_violation_rate'] - enhanced_results['dd_violation_rate']) * 100
    )
    
    logger.info("\n" + "="*80)
    logger.info("COMPARISON RESULTS")
    logger.info("="*80)
    logger.info(f"Sharpe Improvement:       {sharpe_improvement:+.1f}%")
    logger.info(f"Drawdown Improvement:     {dd_improvement:+.1f}%")
    logger.info(f"Win Rate Improvement:     {wr_improvement:+.1f}%")
    logger.info(f"DD Violation Reduction:   {dd_violation_reduction:+.1f}%")
    logger.info("="*80)
    
    # Verdict
    if sharpe_improvement > 20 and dd_improvement > 10:
        verdict = "✅ ENHANCED VERSION SIGNIFICANTLY BETTER - PROCEED TO MARL"
    elif sharpe_improvement > 10 or dd_improvement > 5:
        verdict = "⚠️ MODERATE IMPROVEMENT - CONSIDER MORE EPISODES OR TUNING"
    else:
        verdict = "❌ NO SIGNIFICANT IMPROVEMENT - REVIEW FEATURES"
    
    logger.info(f"\n🎯 VERDICT: {verdict}\n")
    
    # Save results
    results_path = PROJECT_ROOT / 'data' / 'test_results' / 'quick_comparison.json'
    results_path.parent.mkdir(exist_ok=True, parents=True)
    
    comparison = {
        'baseline': baseline_results,
        'enhanced': enhanced_results,
        'improvements': {
            'sharpe': sharpe_improvement,
            'drawdown': dd_improvement,
            'win_rate': wr_improvement,
            'dd_violations': dd_violation_reduction
        },
        'verdict': verdict
    }
    
    with open(results_path, 'w') as f:
        json.dump(comparison, f, indent=2)
    
    logger.info(f"✅ Results saved: {results_path}")


if __name__ == '__main__':
    main()
