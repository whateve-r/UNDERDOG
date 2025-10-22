"""
DRL Agent Training Script - TD3 for Forex Trading

Trains TD3 (Twin Delayed Deep Deterministic Policy Gradient) agent on historical
forex data with technical indicators and regime features.

Based on research papers:
- arXiv:2510.09247v1: TD3 for S&P 500 Options Hedging
- arXiv:2510.10526v1: TD3 for Sentiment-Driven Quantitative Trading
- arXiv:2510.04952v2: PPO for Order Execution with Constraints

Hyperparameters optimized from literature:
- Learning Rate: 1e-4 (Adam optimizer)
- Batch Size: 256
- Hidden Layers: 2 layers × 256 neurons
- Episodes: 2000 (sufficient for convergence)
- Replay Buffer: 1M transitions

Usage:
    # Full training run (2000 episodes)
    poetry run python scripts/train_drl_agent.py --episodes 2000 --eval-freq 100
    
    # Quick validation (100 episodes)
    poetry run python scripts/train_drl_agent.py --episodes 100 --eval-freq 10
    
    # Resume from checkpoint
    poetry run python scripts/train_drl_agent.py --load models/td3_forex_checkpoint.pth --episodes 1000

Features:
- Real-time training metrics (Sharpe, DD, Win Rate)
- Periodic evaluation on validation set
- Checkpoint saving every N episodes
- TensorBoard logging (optional)
- PropFirmSafetyShield integration
"""

import os
import sys
import logging
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
import torch

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
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


class TrainingMetrics:
    """Track training metrics"""
    
    def __init__(self):
        self.episode_rewards: List[float] = []
        self.episode_lengths: List[int] = []
        self.sharpe_ratios: List[float] = []
        self.max_drawdowns: List[float] = []
        self.win_rates: List[float] = []
        self.total_trades: List[int] = []
        
    def add_episode(
        self,
        reward: float,
        length: int,
        sharpe: float,
        drawdown: float,
        win_rate: float,
        trades: int
    ):
        """Add episode metrics"""
        self.episode_rewards.append(reward)
        self.episode_lengths.append(length)
        self.sharpe_ratios.append(sharpe)
        self.max_drawdowns.append(drawdown)
        self.win_rates.append(win_rate)
        self.total_trades.append(trades)
    
    def get_recent_stats(self, window: int = 100) -> Dict:
        """Get statistics for last N episodes"""
        if not self.episode_rewards:
            return {}
        
        recent_rewards = self.episode_rewards[-window:]
        recent_sharpe = self.sharpe_ratios[-window:]
        recent_dd = self.max_drawdowns[-window:]
        recent_wr = self.win_rates[-window:]
        
        return {
            'avg_reward': np.mean(recent_rewards),
            'avg_sharpe': np.mean(recent_sharpe),
            'avg_drawdown': np.mean(recent_dd),
            'avg_win_rate': np.mean(recent_wr),
            'total_episodes': len(self.episode_rewards)
        }


class DRLTrainer:
    """
    TD3 Agent Trainer for Forex Trading
    """
    
    def __init__(
        self,
        symbols: List[str] = None,
        start_date: str = '2022-01-01',
        end_date: str = '2024-08-31',  # Reserve last 2 months for validation
        agent_config: Optional[TD3Config] = None,
        env_config: Optional[TradingEnvConfig] = None
    ):
        """
        Initialize trainer
        
        Args:
            symbols: Trading symbols (default: EURUSD, GBPUSD, USDJPY, XAUUSD)
            start_date: Training start date
            end_date: Training end date (reserve last period for validation)
            agent_config: TD3 agent configuration
            env_config: Environment configuration
        """
        self.symbols = symbols or ['EURUSD', 'GBPUSD', 'USDJPY', 'XAUUSD']
        self.start_date = start_date
        self.end_date = end_date
        
        # Initialize database connector
        self.db = TimescaleDBConnector()
        # Note: Don't call connect() here - it's async and only for async contexts
        # We'll use get_connection() context manager for sync queries
        
        # Agent configuration (optimized from papers)
        if agent_config is None:
            agent_config = TD3Config(
                state_dim=24,  # Expanded observation space (CMDP + risk features)
                action_dim=2,  # [position_size, entry/exit]
                hidden_dim=256,  # Paper recommendation
                lr_actor=1e-4,  # Optimized from arXiv:2510.10526v1
                lr_critic=1e-4,  # Same as actor for stability
                gamma=0.99,
                tau=0.005,
                batch_size=256,  # Paper recommendation
                buffer_size=1_000_000,
                policy_freq=2,
                expl_noise=0.1
            )
        
        self.agent_config = agent_config
        
        # Environment configuration
        if env_config is None:
            env_config = TradingEnvConfig(
                symbol=self.symbols[0],  # Primary symbol
                timeframe='1H',  # 1-hour bars for training (better signal/noise)
                initial_balance=100_000,
                max_position_size=1.0,
                commission_pct=0.0002,  # 2 pips spread
                use_safety_shield=True,
                # 🛡️ CRITICAL: CMDP Safety Configuration
                max_daily_dd_pct=0.05,  # 5% daily DD limit
                max_total_dd_pct=0.10,  # 10% total DD limit
                penalty_daily_dd=-1000.0,  # Catastrophic penalty
                penalty_total_dd=-10000.0,  # Absolute catastrophic penalty
                terminate_on_dd_breach=True,  # Terminate on violation
                max_steps=1000,  # Max 1000 bars per episode
                lookback_bars=200  # 200 bars of history for state
            )
        
        self.env_config = env_config
        
        # Initialize agent
        self.agent = TD3Agent(config=agent_config)
        
        # Initialize replay buffer
        self.replay_buffer = ReplayBuffer(
            state_dim=agent_config.state_dim,
            action_dim=agent_config.action_dim,
            max_size=agent_config.buffer_size
        )
        
        # Metrics tracking
        self.metrics = TrainingMetrics()
        
        logger.info("="*80)
        logger.info("DRL TRAINER INITIALIZED")
        logger.info("="*80)
        logger.info(f"Symbols: {', '.join(self.symbols)}")
        logger.info(f"Training period: {start_date} to {end_date}")
        logger.info(f"Agent: TD3 (state_dim={agent_config.state_dim}, action_dim={agent_config.action_dim})")
        logger.info(f"Hyperparameters:")
        logger.info(f"  - Learning Rate: {agent_config.lr_actor:.0e}")
        logger.info(f"  - Batch Size: {agent_config.batch_size}")
        logger.info(f"  - Hidden Layers: 2x{agent_config.hidden_dim}")
        logger.info(f"  - Replay Buffer: {agent_config.buffer_size:,}")
        logger.info(f"  - Device: {agent_config.device}")
        logger.info("="*80)
    
    def _load_training_data(self) -> pd.DataFrame:
        """
        Load OHLCV data from TimescaleDB for training period
        
        Returns:
            DataFrame with OHLCV data
        """
        logger.info(f"Loading training data: {self.start_date} to {self.end_date}")
        
        query = """
            SELECT time, symbol, open, high, low, close, volume, spread
            FROM ohlcv
            WHERE symbol = %s
              AND timeframe = %s
              AND time >= %s
              AND time <= %s
            ORDER BY time ASC
        """
        
        with self.db.get_connection() as conn:
            df = pd.read_sql_query(
                query,
                conn,
                params=(
                    self.env_config.symbol,
                    self.env_config.timeframe,
                    self.start_date,
                    self.end_date
                )
            )
        
        logger.info(f"Loaded {len(df):,} bars for {self.env_config.symbol}")
        
        if df.empty:
            raise ValueError(f"No data found for {self.env_config.symbol} in period {self.start_date} to {self.end_date}")
        
        return df
    
    def _load_validation_data(self) -> pd.DataFrame:
        """
        Load OHLCV data for validation period (last 2 months)
        
        Returns:
            DataFrame with OHLCV data
        """
        logger.info("Loading validation data: 2024-09-01 to 2024-10-31")
        
        query = """
            SELECT time, symbol, open, high, low, close, volume, spread
            FROM ohlcv
            WHERE symbol = %s
              AND timeframe = %s
              AND time >= '2024-09-01'
              AND time <= '2024-10-31'
            ORDER BY time ASC
        """
        
        with self.db.get_connection() as conn:
            df = pd.read_sql_query(
                query,
                conn,
                params=(
                    self.env_config.symbol,
                    self.env_config.timeframe
                )
            )
        
        logger.info(f"Loaded {len(df):,} validation bars")
        
        return df
    
    def train_episode(self, env: ForexTradingEnv, episode: int) -> Dict:
        """
        Train for one episode
        
        Args:
            env: Trading environment
            episode: Episode number
        
        Returns:
            Episode statistics dict
        """
        state = env.reset()
        episode_reward = 0
        episode_length = 0
        done = False
        
        while not done:
            # Select action with exploration noise
            action = self.agent.select_action(state, explore=True)
            
            # Take step in environment
            next_state, reward, done, info = env.step(action)
            
            # Store transition in replay buffer
            self.replay_buffer.add(state, action, reward, next_state, done)
            
            # Train agent if buffer has enough samples
            if len(self.replay_buffer) >= self.agent_config.batch_size:
                batch = self.replay_buffer.sample(self.agent_config.batch_size)
                self.agent.train(batch)
            
            # Update state
            state = next_state
            episode_reward += reward
            episode_length += 1
        
        # Extract episode statistics from environment
        stats = {
            'reward': episode_reward,
            'length': episode_length,
            'sharpe': info.get('sharpe_ratio', 0.0),
            'drawdown': info.get('max_drawdown', 0.0),
            'win_rate': info.get('win_rate', 0.0),
            'trades': info.get('total_trades', 0)
        }
        
        return stats
    
    def evaluate(self, eval_episodes: int = 10) -> Dict:
        """
        Evaluate agent on validation set
        
        Args:
            eval_episodes: Number of evaluation episodes
        
        Returns:
            Evaluation metrics dict
        """
        logger.info("Running evaluation...")
        
        # Load validation data
        val_data = self._load_validation_data()
        
        # Create validation environment
        val_env = ForexTradingEnv(
            config=self.env_config,
            db_connector=self.db,
            historical_data=val_data
        )
        
        eval_rewards = []
        eval_sharpe = []
        eval_dd = []
        eval_wr = []
        
        for ep in range(eval_episodes):
            state = val_env.reset()
            done = False
            episode_reward = 0
            
            while not done:
                # Select action without exploration noise
                action = self.agent.select_action(state, explore=False)
                next_state, reward, done, info = val_env.step(action)
                state = next_state
                episode_reward += reward
            
            eval_rewards.append(episode_reward)
            eval_sharpe.append(info.get('sharpe_ratio', 0.0))
            eval_dd.append(info.get('max_drawdown', 0.0))
            eval_wr.append(info.get('win_rate', 0.0))
        
        eval_stats = {
            'eval_reward': np.mean(eval_rewards),
            'eval_sharpe': np.mean(eval_sharpe),
            'eval_drawdown': np.mean(eval_dd),
            'eval_win_rate': np.mean(eval_wr)
        }
        
        logger.info(
            f"Evaluation Results: Reward={eval_stats['eval_reward']:.2f}, "
            f"Sharpe={eval_stats['eval_sharpe']:.2f}, DD={eval_stats['eval_drawdown']:.2%}, "
            f"WR={eval_stats['eval_win_rate']:.2%}"
        )
        
        return eval_stats
    
    def train(
        self,
        num_episodes: int = 2000,
        eval_freq: int = 100,
        save_freq: int = 500,
        checkpoint_dir: str = 'models'
    ):
        """
        Main training loop
        
        Args:
            num_episodes: Total episodes to train
            eval_freq: Evaluate every N episodes
            save_freq: Save checkpoint every N episodes
            checkpoint_dir: Directory to save checkpoints
        """
        logger.info("="*80)
        logger.info("STARTING TRAINING")
        logger.info("="*80)
        logger.info(f"Total episodes: {num_episodes}")
        logger.info(f"Evaluation frequency: every {eval_freq} episodes")
        logger.info(f"Checkpoint frequency: every {save_freq} episodes")
        logger.info("="*80)
        
        # Create checkpoint directory
        checkpoint_path = Path(checkpoint_dir)
        checkpoint_path.mkdir(exist_ok=True)
        
        # Load training data
        train_data = self._load_training_data()
        
        # Create training environment
        train_env = ForexTradingEnv(
            config=self.env_config,
            db_connector=self.db,
            historical_data=train_data
        )
        
        # Training loop
        best_eval_sharpe = -np.inf
        
        for episode in range(1, num_episodes + 1):
            # Train one episode
            stats = self.train_episode(train_env, episode)
            
            # Update metrics
            self.metrics.add_episode(
                reward=stats['reward'],
                length=stats['length'],
                sharpe=stats['sharpe'],
                drawdown=stats['drawdown'],
                win_rate=stats['win_rate'],
                trades=stats['trades']
            )
            
            # Log progress
            if episode % 10 == 0:
                recent_stats = self.metrics.get_recent_stats(window=10)
                logger.info(
                    f"Episode {episode}/{num_episodes} | "
                    f"Reward: {recent_stats['avg_reward']:.2f} | "
                    f"Sharpe: {recent_stats['avg_sharpe']:.2f} | "
                    f"DD: {recent_stats['avg_drawdown']:.2%} | "
                    f"WR: {recent_stats['avg_win_rate']:.2%}"
                )
            
            # Periodic evaluation
            if episode % eval_freq == 0:
                eval_stats = self.evaluate(eval_episodes=10)
                
                # Save best model
                if eval_stats['eval_sharpe'] > best_eval_sharpe:
                    best_eval_sharpe = eval_stats['eval_sharpe']
                    best_model_path = checkpoint_path / 'td3_forex_best.pth'
                    self.agent.save(str(best_model_path))
                    logger.info(f"✓ New best model saved (Sharpe: {best_eval_sharpe:.2f})")
            
            # Periodic checkpoint
            if episode % save_freq == 0:
                checkpoint_file = checkpoint_path / f'td3_forex_ep{episode}.pth'
                self.agent.save(str(checkpoint_file))
                logger.info(f"✓ Checkpoint saved: {checkpoint_file}")
        
        # Final save
        final_model_path = checkpoint_path / 'td3_forex_final.pth'
        self.agent.save(str(final_model_path))
        
        logger.info("="*80)
        logger.info("TRAINING COMPLETE")
        logger.info("="*80)
        logger.info(f"Final model saved: {final_model_path}")
        logger.info(f"Best validation Sharpe: {best_eval_sharpe:.2f}")
        
        # Final evaluation
        logger.info("\nRunning final evaluation (50 episodes)...")
        final_eval = self.evaluate(eval_episodes=50)
        logger.info(f"Final Evaluation:")
        logger.info(f"  Sharpe Ratio: {final_eval['eval_sharpe']:.2f}")
        logger.info(f"  Max Drawdown: {final_eval['eval_drawdown']:.2%}")
        logger.info(f"  Win Rate: {final_eval['eval_win_rate']:.2%}")
        logger.info("="*80)


def main():
    """Main execution"""
    parser = argparse.ArgumentParser(
        description='Train TD3 agent for forex trading'
    )
    
    parser.add_argument(
        '--episodes',
        type=int,
        default=2000,
        help='Number of training episodes (default: 2000)'
    )
    
    parser.add_argument(
        '--eval-freq',
        type=int,
        default=100,
        help='Evaluate every N episodes (default: 100)'
    )
    
    parser.add_argument(
        '--save-freq',
        type=int,
        default=500,
        help='Save checkpoint every N episodes (default: 500)'
    )
    
    parser.add_argument(
        '--symbols',
        nargs='+',
        default=['EURUSD', 'GBPUSD', 'USDJPY', 'XAUUSD'],
        help='Trading symbols (default: EURUSD GBPUSD USDJPY XAUUSD)'
    )
    
    parser.add_argument(
        '--start-date',
        type=str,
        default='2022-01-01',
        help='Training start date (default: 2022-01-01)'
    )
    
    parser.add_argument(
        '--end-date',
        type=str,
        default='2024-08-31',
        help='Training end date (default: 2024-08-31, reserves Sept-Oct for validation)'
    )
    
    parser.add_argument(
        '--load',
        type=str,
        default=None,
        help='Load checkpoint to resume training'
    )
    
    parser.add_argument(
        '--lr',
        type=float,
        default=1e-4,
        help='Learning rate (default: 1e-4, optimized from papers)'
    )
    
    parser.add_argument(
        '--batch-size',
        type=int,
        default=256,
        help='Batch size (default: 256, paper recommendation)'
    )
    
    parser.add_argument(
        '--hidden-dim',
        type=int,
        default=256,
        help='Hidden layer dimension (default: 256)'
    )
    
    args = parser.parse_args()
    
    # Create agent config with CLI arguments
    agent_config = TD3Config(
        state_dim=14,
        action_dim=2,
        hidden_dim=args.hidden_dim,
        lr_actor=args.lr,
        lr_critic=args.lr,
        batch_size=args.batch_size
    )
    
    # Initialize trainer
    trainer = DRLTrainer(
        symbols=args.symbols,
        start_date=args.start_date,
        end_date=args.end_date,
        agent_config=agent_config
    )
    
    # Load checkpoint if specified
    if args.load:
        logger.info(f"Loading checkpoint: {args.load}")
        trainer.agent.load(args.load)
    
    # Start training
    try:
        trainer.train(
            num_episodes=args.episodes,
            eval_freq=args.eval_freq,
            save_freq=args.save_freq
        )
    
    except KeyboardInterrupt:
        logger.warning("\nTraining interrupted by user")
        logger.info("Saving current model...")
        trainer.agent.save('models/td3_forex_interrupted.pth')
        logger.info("Model saved: models/td3_forex_interrupted.pth")
    
    except Exception as e:
        logger.error(f"Training failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
