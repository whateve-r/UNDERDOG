"""
Training Script for MARL (Multi-Agent Reinforcement Learning)

This script trains the MTF-MARL architecture:
    - LEVEL 1: 4× TD3 Agents (Local) - EURUSD, GBPUSD, USDJPY, USDCHF
    - LEVEL 2: 1× A3C Meta-Agent (Global) - Portfolio Coordinator

Architecture:
    - CTDE: Centralized Training, Decentralized Execution
    - Meta-Agent: A3C (coordinates risk allocation)
    - Local Agents: TD3 (execute trades on each pair)

References:
    - 2405.19982v1.pdf: A3C for multi-currency Forex
    - ALA2017_Gupta.pdf: CTDE architecture
"""

import argparse
import logging
import sys
import os
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional
import numpy as np
import torch
import csv

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from underdog.rl.multi_asset_env import MultiAssetEnv, MultiAssetConfig
from underdog.rl.meta_agent import A3CMetaAgent, A3CConfig
from underdog.rl.agents import TD3Agent, TD3Config, ReplayBuffer, PrioritizedReplayBuffer
from underdog.rl.multi_agents import PPOAgent, SACAgent, DDPGAgent
from config.agent_configs import get_agent_config, get_agent_metadata, AGENT_MAPPING

# POEL Integration
from underdog.rl.poel import (
    POELMetaAgent,
    POELRewardShaper,
    DistanceMetric,
    TrainingMode,
)

logger = logging.getLogger(__name__)


class MARLTrainer:
    """
    MARL Training Coordinator
    
    Orchestrates training of:
        1. Meta-Agent (A3C) - Learns risk allocation policy
        2. 4× Local Agents (TD3) - Learn trading policies
    
    Training Loop:
        - Episode starts with reset()
        - Each step:
            1. Meta-Agent selects risk limits (Meta-Action)
            2. Apply risk limits to local agents
            3. Each TD3 agent selects trading action
            4. Environment executes all actions
            5. Collect experience (states, actions, rewards)
            6. Update Meta-Agent every N steps (A3C)
            7. Update TD3 agents every M steps
        - Episode ends on DD breach or max steps
    """
    
    def __init__(
        self,
        env: MultiAssetEnv,
        meta_agent: A3CMetaAgent,
        local_agents: List,  # Changed from List[TD3Agent] to support heterogeneous agents
        config: Dict[str, Any],
        agent_names: Optional[List[str]] = None,  # NEW: Agent type names for logging
        poel_enabled: bool = True,  # POEL Integration
        log_name: Optional[str] = None,  # Custom log filename
    ):
        """
        Initialize MARL Trainer
        
        Args:
            env: MultiAssetEnv instance
            meta_agent: A3C Meta-Agent
            local_agents: List of heterogeneous DRL agents (TD3, PPO, SAC, DDPG)
            config: Training configuration
            agent_names: List of agent names/types for logging (e.g., ["EURUSD:TD3+LSTM", ...])
            poel_enabled: Enable POEL (Purpose-Driven Open-Ended Learning)
        """
        self.env = env
        self.meta_agent = meta_agent
        self.local_agents = local_agents
        self.config = config
        self.agent_names = agent_names or [f"Agent{i}" for i in range(len(local_agents))]
        
        # Training tracking
        self.episode = 0
        self.total_steps = 0
        
        # ===== POEL INTEGRATION =====
        self.poel_enabled = poel_enabled
        
        if self.poel_enabled:
            logger.info("=" * 80)
            logger.info("POEL (Purpose-Driven Open-Ended Learning) ENABLED")
            logger.info("=" * 80)
            
            # 1. Initialize POEL Meta-Agent
            self.poel_meta_agent = POELMetaAgent(
                initial_balance=config.get('initial_balance', 100000.0),
                symbols=env.config.symbols,
                max_daily_dd=config.get('max_daily_dd', 0.05),
                max_total_dd=config.get('max_total_dd', 0.10),
                nrf_enabled=config.get('nrf_enabled', True),
                nrf_cycle_frequency=config.get('nrf_cycle_frequency', 10),
                state_dim=31,  # Fixed state_dim for all agents
            )
            
            # 2. Initialize POEL Reward Shapers (one per local agent)
            self.poel_shapers = {}
            for i, agent_name in enumerate(self.agent_names):
                symbol = env.config.symbols[i]
                self.poel_shapers[symbol] = POELRewardShaper(
                    state_dim=31,  # Fixed state_dim for all agents
                    action_dim=1,
                    alpha=config.get('poel_alpha', 0.7),  # 70% PnL, 30% exploration
                    beta=config.get('poel_beta', 1.0),
                    novelty_metric=DistanceMetric.L2,
                    max_local_dd_pct=config.get('poel_max_local_dd', 0.15),
                    initial_balance=config.get('initial_balance', 100000.0) / len(local_agents),
                )
            
            logger.info(f"POEL Meta-Agent initialized:")
            logger.info(f"  - Initial Balance: ${config.get('initial_balance', 100000.0):,.0f}")
            logger.info(f"  - Max Daily DD: {config.get('max_daily_dd', 0.05):.1%}")
            logger.info(f"  - Max Total DD: {config.get('max_total_dd', 0.10):.1%}")
            logger.info(f"  - NRF Enabled: {config.get('nrf_enabled', True)}")
            logger.info(f"  - NRF Cycle Frequency: {config.get('nrf_cycle_frequency', 10)} episodes")
            logger.info(f"\nPOEL Reward Shapers created for {len(self.poel_shapers)} agents")
            logger.info(f"  - Alpha (PnL weight): {config.get('poel_alpha', 0.7):.1%}")
            logger.info(f"  - Beta (Stability weight): {config.get('poel_beta', 1.0):.1f}")
            logger.info(f"  - Max Local DD: {config.get('poel_max_local_dd', 0.15):.1%}")
            logger.info("=" * 80)
        else:
            self.poel_meta_agent = None
            self.poel_shapers = None
            logger.info("POEL disabled - using standard reward shaping")
        # ===== END POEL INTEGRATION =====
        
        # 🚨 CMDP Emergency Tracking (persists across steps)
        self.emergency_mode_active = False
        self.emergency_trigger_dd = 0.0
        
        # Experience buffers
        self.meta_states = []
        self.meta_actions = []
        self.meta_rewards = []
        self.meta_dones = []
        
        # TD3 Replay Buffers (one per agent) - USE PRIORITIZED for CMDP
        # Use environment's observation shapes to get correct buffer dimensions (handles sequences)
        obs_shapes = list(self.env.get_observation_shapes().values())  # Get shapes in order
        self.replay_buffers = [
            PrioritizedReplayBuffer(
                state_dim=obs_shapes[i],  # (60,31), (15,31), (120,31), or (31,)
                action_dim=local_agents[i].config.action_dim,
                max_size=config.get('replay_buffer_size', 100000),
                alpha=0.6,  # Priority exponent
                beta=0.4,   # Importance sampling
                beta_increment=0.001
            )
            for i in range(len(local_agents))
        ]
        
        # Metrics
        self.episode_rewards = []
        self.episode_dds = []
        self.episode_balances = []
        
        # CSV logging for dashboard
        self.log_dir = Path("logs")
        self.log_dir.mkdir(exist_ok=True)
        
        # Use custom filename or default based on POEL vs baseline
        if log_name:
            self.metrics_csv = self.log_dir / log_name
        elif self.poel_enabled:
            self.metrics_csv = self.log_dir / "poel_metrics.csv"
        else:
            self.metrics_csv = self.log_dir / "baseline_metrics.csv"
        
        # Enhanced CSV with per-agent metrics and training dynamics
        self.csv_fieldnames = [
            'episode', 'reward', 'final_balance', 'global_dd',
            # Per-agent rewards
            'agent0_reward', 'agent1_reward', 'agent2_reward', 'agent3_reward',
            # Per-agent balances
            'agent0_balance', 'agent1_balance', 'agent2_balance', 'agent3_balance',
            # Training dynamics
            'avg_epsilon', 'avg_td_error', 'violations',
            # Agent types (for heterogeneous MARL)
            'agent0_type', 'agent1_type', 'agent2_type', 'agent3_type'
        ]
        
        # Store agent names for CSV logging
        self.agent_names = agent_names if 'agent_names' in locals() else [
            f"Agent{i}" for i in range(len(local_agents))
        ]
        
        # Always recreate CSV with headers (fresh start for each training run)
        with open(self.metrics_csv, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.csv_fieldnames)
            writer.writeheader()
        
        # Tracking for metrics
        self.agent_episode_rewards = [[] for _ in range(len(local_agents))]
        self.td_errors = []
        
        logger.info(
            f"MARLTrainer initialized: {len(local_agents)} local agents, "
            f"meta_agent={meta_agent.__class__.__name__}"
        )
        logger.info(f"Metrics logging to: {self.metrics_csv}")
    
    def train_episode(self) -> Dict[str, float]:
        """
        Train one episode
        
        Returns:
            metrics: Episode metrics dict
        """
        self.episode += 1
        
        # 🚨 Reset CMDP emergency tracking for new episode
        self.emergency_mode_active = False
        self.emergency_trigger_dd = 0.0
        
        # ===== POEL EPISODE START =====
        poel_episode_config = None
        if self.poel_enabled:
            poel_episode_config = self.poel_meta_agent.start_episode()
            
            logger.info("=" * 80)
            logger.info(f"POEL Episode {poel_episode_config['episode']}")
            logger.info("=" * 80)
            logger.info(f"  Mode: {poel_episode_config['mode']}")
            logger.info(f"  Curriculum Injection: {poel_episode_config['inject_failure']}")
            
            if poel_episode_config['mode'] == 'nrf':
                # Replace ψ with 'psi' for Windows compatibility
                nrf_msg = poel_episode_config['nrf_config']['message'].replace('ψ', 'psi')
                logger.info(f"  NRF Mode Active: {nrf_msg}")
            
            logger.info("=" * 80)
        # ===== END POEL EPISODE START =====
        
        # Reset environment
        meta_state, info = self.env.reset()
        
        # ===== POEL CURRICULUM INJECTION =====
        # TODO: If failure state injection is requested, modify environment initial state
        # This requires environment support for state initialization
        # if poel_episode_config and poel_episode_config['inject_failure']:
        #     failure_state = poel_episode_config['failure_state']
        #     self.env.set_initial_state(failure_state)
        # ===== END POEL CURRICULUM INJECTION =====
        
        # CRITICAL: Shape validation at episode 1 (BEFORE any training)
        if self.episode == 1:
            logger.info("\n" + "="*70)
            logger.info("CRITICAL: SHAPE VALIDATION (Episode 1, Step 0)")
            logger.info("="*70)
            logger.info("\nOBSERVATION SHAPES (HMARL - Heterogeneous Sequences):")
            
            # Get local observations from buffer manager
            local_obs = self.env.get_local_observations()
            expected_shapes = self.env.get_observation_shapes()
            
            for i, (symbol, obs) in enumerate(zip(self.env.config.symbols, local_obs)):
                expected_shape = expected_shapes[symbol]
                match_str = "OK" if obs.shape == expected_shape else "MISMATCH"
                
                agent_type = self.local_agents[i].__class__.__name__
                architecture = getattr(self.local_agents[i].config, 'architecture', 'Unknown')
                
                logger.info(
                    f"  Agent {i} ({symbol:6s}) | {agent_type:10s} | {architecture:11s} | "
                    f"Expected: {str(expected_shape):15s} | Actual: {str(obs.shape):15s} | {match_str}"
                )
                
                # Critical assertion: halt training if shapes mismatch
                assert obs.shape == expected_shape, (
                    f"CRITICAL: Shape mismatch for {symbol}! "
                    f"Expected {expected_shape}, got {obs.shape}. "
                    f"Training ABORTED to prevent network corruption."
                )
            
            logger.info("\nAll observation shapes VALIDATED - Safe to proceed with training")
            logger.info("="*70 + "\n")
        
        # Reset episode buffers
        self.meta_states = []
        self.meta_actions = []
        self.meta_rewards = []
        self.meta_dones = []
        
        episode_reward = 0.0
        step = 0
        
        done = False
        final_balance = self.env.initial_balance  # Initialize with starting balance
        final_dd = 0.0  # Initialize with zero DD
        
        while not done:
            step += 1
            self.total_steps += 1
            
            # 1. Meta-Agent selects risk limits
            meta_action, log_prob, entropy = self.meta_agent.select_meta_action(
                meta_state,
                deterministic=False
            )
            
            # 2. Apply risk limits to local agents
            self.env._apply_meta_action(meta_action)
            
            # 3. Each TD3 agent selects trading action and execute
            local_prev_states = []
            local_actions = []
            local_states = []
            local_rewards = []
            local_dones = []
            
            # CRITICAL: Validate first step with actual agent forward passes
            if self.episode == 1 and step == 1:
                logger.info("\n" + "="*70)
                logger.info("CRITICAL: AGENT FORWARD PASS VALIDATION (Episode 1, Step 1)")
                logger.info("="*70)
            
            for i, (agent, env) in enumerate(zip(self.local_agents, self.env.local_envs)):
                # Get current state from BUFFER (temporal sequences)
                local_obs_sequences = self.env.get_local_observations()
                prev_state = local_obs_sequences[i]
                local_prev_states.append(prev_state)
                
                # CRITICAL: Log tensor shapes on first forward pass
                if self.episode == 1 and step == 1:
                    agent_type = agent.__class__.__name__
                    architecture = getattr(agent.config, 'architecture', 'Unknown')
                    symbol = self.env.config.symbols[i]
                    
                    logger.info(f"\n  Agent {i} ({symbol}) - {agent_type} with {architecture}:")
                    logger.info(f"    Input shape: {prev_state.shape}")
                
                # 🚨 CMDP EMERGENCY OVERRIDE: Force HOLD action if emergency mode active
                if self.poel_enabled and self.emergency_mode_active:
                    # Force HOLD (neutral action = 0.0)
                    local_action = np.array([0.0])
                    if step % 10 == 0:  # Log every 10 steps to avoid spam
                        logger.warning(f"[EMERGENCY] Forcing HOLD for {self.env.config.symbols[i]}")
                else:
                    # Select action (this triggers policy network forward pass)
                    local_action = agent.select_action(prev_state, explore=True)
                
                local_actions.append(local_action)
                
                # 🔥 CRITICAL: Validate action output shape
                if self.episode == 1 and step == 1:
                    expected_action_dim = agent.config.action_dim
                    actual_action_shape = np.atleast_1d(local_action).shape
                    match_str = "OK" if actual_action_shape[0] == expected_action_dim else "MISMATCH"
                    
                    logger.info(
                        f"    Action shape: {actual_action_shape} "
                        f"(expected {expected_action_dim}D) {match_str}"
                    )
                    
                    # Critical assertion
                    assert actual_action_shape[0] == expected_action_dim, (
                        f"Action dimension mismatch for {symbol}! "
                        f"Expected {expected_action_dim}, got {actual_action_shape[0]}"
                    )
                
                # Execute action
                next_state, reward, done_local, truncated, info = env.step(local_action)
                local_states.append(next_state)
                local_rewards.append(reward)
                local_dones.append(done_local or truncated)
                
                # Track agent rewards for dashboard
                self.agent_episode_rewards[i].append(reward)
            
            # CRITICAL: Close validation logging after first step
            if self.episode == 1 and step == 1:
                logger.info("\nAll agent forward passes VALIDATED - Networks functioning correctly")
                logger.info("="*70 + "\n")
            
            # ===== POEL REWARD ENRICHMENT =====
            # Compute enriched rewards BEFORE storing in buffer
            enriched_rewards = []
            poel_infos = []
            
            if self.poel_enabled:
                for i in range(len(self.local_agents)):
                    symbol = self.env.config.symbols[i]
                    raw_pnl = local_rewards[i]
                    
                    # Convert reward to scalar
                    if isinstance(raw_pnl, (list, tuple, np.ndarray)):
                        raw_pnl_scalar = float(raw_pnl[0]) if len(raw_pnl) > 0 else 0.0
                    else:
                        raw_pnl_scalar = float(raw_pnl)
                    
                    # Get current balance for this agent
                    agent_balance = self.env.local_envs[i].equity
                    
                    # Compute POEL enriched reward
                    enriched_reward, poel_info = self.poel_shapers[symbol].compute_reward(
                        state=local_prev_states[i],
                        action=local_actions[i],
                        raw_pnl=raw_pnl_scalar,
                        new_balance=agent_balance,
                        is_new_day=False,  # TODO: Track day changes if needed
                    )
                    
                    enriched_rewards.append(enriched_reward)
                    poel_infos.append(poel_info)
                    
                    # Update POEL Meta-Agent performance tracking
                    self.poel_meta_agent.update_agent_performance(
                        agent_id=f"agent_{symbol}",
                        symbol=symbol,
                        pnl=raw_pnl_scalar,
                        balance=agent_balance,
                    )
                    
                    # NRF step (if in NRF mode)
                    if self.poel_meta_agent.current_mode == TrainingMode.NRF:
                        nrf_metrics = self.poel_meta_agent.nrf_step(
                            state=local_prev_states[i],
                            training_step=self.total_steps,
                        )
                        if nrf_metrics and step % 100 == 0:
                            logger.debug(f"NRF Update ({symbol}): {nrf_metrics}")
            else:
                # No POEL: use raw rewards
                enriched_rewards = local_rewards
                poel_infos = [None] * len(self.local_agents)
            
            # 🚨 CMDP GLOBAL FILTER: Override all rewards if global DD critical
            # This creates "risk phobia" at the portfolio level
            global_dd_limit = 0.10  # 10% global DD limit
            cmdp_global_threshold = 0.80 * global_dd_limit  # 8% threshold (80% of limit)
            
            # Update emergency tracking
            if final_dd >= cmdp_global_threshold:
                if not self.emergency_mode_active:
                    # First activation
                    self.emergency_mode_active = True
                    self.emergency_trigger_dd = final_dd
                    logger.critical(
                        f"\n{'='*80}\n"
                        f"[CMDP EMERGENCY MODE ACTIVATED]\n"
                        f"{'='*80}\n"
                        f"Global DD: {final_dd:.2%} >= Threshold: {cmdp_global_threshold:.2%}\n"
                        f"ALL FUTURE REWARDS SET TO -1000 UNTIL DD RECOVERS\n"
                        f"{'='*80}\n"
                    )
                
                # DRASTIC PENALTY: Override ALL agent rewards to -1000
                enriched_rewards = [-1000.0] * len(self.local_agents)
                
                # Mark CMDP violation in poel_infos
                for info in poel_infos:
                    if info is not None:
                        info['cmdp_global_violation'] = True
                        info['cmdp_global_penalty'] = -1000.0
            elif self.emergency_mode_active and final_dd < cmdp_global_threshold * 0.90:
                # Recovery with hysteresis (10%)
                self.emergency_mode_active = False
                logger.info(
                    f"\n[OK] CMDP Emergency Mode DEACTIVATED - DD recovered to {final_dd:.2%}\n"
                )
            
            # ===== END POEL REWARD ENRICHMENT =====
            
            # Store transitions
            for i, agent in enumerate(self.local_agents):
                prev_state = local_prev_states[i]
                local_action = local_actions[i]
                next_state = local_states[i]
                reward = enriched_rewards[i]  # USE ENRICHED REWARD
                done_local = local_dones[i]
                
                # Store transition based on agent type
                agent_class = agent.__class__.__name__
                
                if agent_class == 'PPOAgent':
                    # PPO uses on-policy buffer (internal to agent)
                    # Convert reward to scalar
                    if isinstance(reward, (list, tuple, np.ndarray)):
                        reward_scalar = float(reward[0]) if len(reward) > 0 else 0.0
                    else:
                        reward_scalar = float(reward)
                    
                    # Store transition (preserve sequential shape for CNN1D)
                    agent.store_transition(
                        prev_state,  # Keep original shape: (15, 31) for CNN1D
                        np.atleast_1d(local_action).flatten(),
                        reward_scalar,
                        done_local
                    )
                else:
                    # Off-policy agents (TD3, SAC, DDPG) use replay buffer
                    # Convert reward to scalar
                    if isinstance(reward, (list, tuple, np.ndarray)):
                        reward_scalar = float(reward[0]) if len(reward) > 0 else 0.0
                    else:
                        reward_scalar = float(reward)
                    
                    # Ensure action is 1D array
                    action_array = np.atleast_1d(local_action).flatten()
                    
                    # Add to replay buffer (preserve sequential shape for LSTM/CNN1D/Transformer)
                    self.replay_buffers[i].add(
                        prev_state,  # Keep original shape: (60,31), (15,31), (120,31), or (31,)
                        action_array,
                        reward_scalar,  # ENRICHED REWARD
                        next_state,  # Keep original shape
                        float(done_local)
                    )
            
            # 4. Update portfolio balance after all local envs have stepped
            # CRITICAL: MultiAssetEnv.step() is NOT called in this trainer loop,
            # so we must manually update current_balance by summing local equities
            self.env.current_balance = sum(env.equity for env in self.env.local_envs)
            self.env.peak_balance = max(self.env.peak_balance, self.env.current_balance)
            
            # 5. Aggregate for Meta-Agent
            meta_reward = sum(local_rewards)
            next_meta_state = self.env._build_meta_state(local_states)
            done = any(local_dones) or self.env._calculate_global_dd() > self.env.config.max_global_dd_pct
            
            # 6. Store Meta-Agent experience
            self.meta_states.append(meta_state)
            self.meta_actions.append(meta_action)
            self.meta_rewards.append(meta_reward)
            self.meta_dones.append(done)
            
            episode_reward += meta_reward
            
            # 7. Update Meta-Agent every N steps (A3C)
            if len(self.meta_states) >= self.meta_agent.config.n_steps or done:
                self._update_meta_agent(next_meta_state, done)
            
            # 8. Update agents (heterogeneous training)
            for i, agent in enumerate(self.local_agents):
                # Get agent class name
                agent_class = agent.__class__.__name__
                
                if agent_class == 'PPOAgent':
                    # PPO is on-policy: store transitions in agent buffer
                    # Already stored via store_transition() after action selection
                    # Train at episode end or when buffer is full
                    pass
                else:
                    # Off-policy agents (TD3, SAC, DDPG): update from replay buffer
                    if self.total_steps % self.config.get('update_freq', 2) == 0:
                        if len(self.replay_buffers[i]) >= self.config.get('batch_size', 256):
                            train_metrics = agent.train(self.replay_buffers[i])
                            # Track TD error (critic loss) for dashboard
                            if train_metrics and 'critic_loss' in train_metrics:
                                self.td_errors.append(train_metrics['critic_loss'])
            
            # Move to next state
            meta_state = next_meta_state
            
            # Log progress
            if step % 100 == 0:
                logger.info(
                    f"Episode {self.episode} Step {step}: "
                    f"reward={meta_reward:.2f}, global_dd={self.env._calculate_global_dd():.2%}"
                )
        
        # Episode completed - Capture FINAL values NOW (before any reset)
        final_balance = self.env.current_balance
        final_dd = self.env._calculate_global_dd()
        
        # Train PPO agents at episode end (on-policy learning)
        for i, agent in enumerate(self.local_agents):
            if agent.__class__.__name__ == 'PPOAgent':
                train_metrics = agent.train()
                # Track metrics for dashboard
                if train_metrics and 'critic_loss' in train_metrics:
                    self.td_errors.append(train_metrics['critic_loss'])
        
        # DEBUG: Log individual env equities
        env_equities = [env.equity for env in self.env.local_envs]
        env_balances = [env.balance for env in self.env.local_envs]
        logger.info(
            f"Episode {self.episode} completed: "
            f"total_balance=${final_balance:.2f}, DD={final_dd:.2%}, "
            f"env_equities={[f'${e:.2f}' for e in env_equities]}, "
            f"env_balances={[f'${b:.2f}' for b in env_balances]}"
        )
        
        # Calculate per-agent metrics
        agent_rewards = [np.mean(self.agent_episode_rewards[i]) if len(self.agent_episode_rewards[i]) > 0 else 0.0 
                        for i in range(len(self.local_agents))]
        agent_balances = [env.equity for env in self.env.local_envs]
        
        # Calculate training dynamics
        # Epsilon: Only TD3/DDPG have epsilon, PPO/SAC use entropy
        epsilons = []
        for agent in self.local_agents:
            if hasattr(agent, 'epsilon'):
                epsilons.append(agent.epsilon)
            elif hasattr(agent, 'noise_scale'):  # DDPG
                epsilons.append(agent.noise_scale)
        avg_epsilon = np.mean(epsilons) if len(epsilons) > 0 else 0.0
        
        avg_td_error = np.mean(self.td_errors) if len(self.td_errors) > 0 else 0.0
        violations = 1 if final_dd > self.env.config.max_global_dd_pct else 0
        
        # Store metrics
        self.episode_rewards.append(episode_reward)
        self.episode_dds.append(final_dd)
        self.episode_balances.append(final_balance)
        
        metrics = {
            'episode': self.episode,
            'steps': step,
            'reward': episode_reward,
            'final_balance': final_balance,
            'global_dd': final_dd,
            'avg_reward_100': np.mean(self.episode_rewards[-100:]) if len(self.episode_rewards) >= 100 else np.mean(self.episode_rewards),
        }
        
        # Log to CSV for dashboard (unbuffered mode for real-time updates)
        with open(self.metrics_csv, 'a', newline='', buffering=1) as f:
            writer = csv.DictWriter(f, fieldnames=self.csv_fieldnames)
            writer.writerow({
                'episode': self.episode,
                'reward': episode_reward,
                'final_balance': final_balance,
                'global_dd': final_dd,
                'agent0_reward': agent_rewards[0],
                'agent1_reward': agent_rewards[1],
                'agent2_reward': agent_rewards[2],
                'agent3_reward': agent_rewards[3],
                'agent0_balance': agent_balances[0],
                'agent1_balance': agent_balances[1],
                'agent2_balance': agent_balances[2],
                'agent3_balance': agent_balances[3],
                'avg_epsilon': avg_epsilon,
                'avg_td_error': avg_td_error,
                'violations': violations,
                # Agent types for heterogeneous MARL
                'agent0_type': self.agent_names[0] if len(self.agent_names) > 0 else 'N/A',
                'agent1_type': self.agent_names[1] if len(self.agent_names) > 1 else 'N/A',
                'agent2_type': self.agent_names[2] if len(self.agent_names) > 2 else 'N/A',
                'agent3_type': self.agent_names[3] if len(self.agent_names) > 3 else 'N/A',
            })
        
        # Reset per-agent tracking for next episode
        self.agent_episode_rewards = [[] for _ in range(len(self.local_agents))]
        self.td_errors = []
        
        # ===== POEL META-AGENT COORDINATION (END OF EPISODE) =====
        if self.poel_enabled:
            # 1. Compute Purpose and Allocate Capital
            portfolio_pnl = sum(local_rewards)  # Total PnL for this episode
            daily_dd = 0.0  # TODO: Track intra-day DD if needed
            
            meta_result = self.poel_meta_agent.compute_purpose_and_allocate(
                current_balance=final_balance,
                daily_dd_pct=daily_dd,
                total_dd_pct=final_dd,
                portfolio_pnl=portfolio_pnl,
            )
            
            logger.info("\n" + "=" * 80)
            logger.info("POEL META-AGENT COORDINATION")
            logger.info("=" * 80)
            logger.info(f"  Purpose Score: {meta_result['purpose']['purpose']:.2f}")
            logger.info(f"  Global DD: {final_dd:.2%} (Daily: {daily_dd:.2%})")
            logger.info(f"  Emergency Mode: {meta_result['emergency']['emergency_mode']}")
            
            logger.info("\n  Capital Allocation:")
            for symbol, weight in meta_result['weights'].items():
                perf = meta_result['agent_performances'].get(symbol, {})
                calmar = perf.get('calmar_ratio', 0.0)
                logger.info(f"    {symbol:8s}: {weight:6.2%} (Calmar: {calmar:6.2f})")
            
            # 2. Record Failure if DD Breach
            if final_dd > self.config.get('max_total_dd', 0.10):
                dd_breach_size = final_dd - self.config.get('max_total_dd', 0.10)
                
                logger.warning(f"\n  [!] DD BREACH: {final_dd:.2%} > {self.config.get('max_total_dd', 0.10):.1%}")
                logger.warning(f"  Recording failure to Failure Bank...")
                
                # Get agent weights (simplified - just use first agent's state as representative)
                agent_weights = {
                    symbol: np.random.randn(100)  # TODO: Extract actual model weights
                    for symbol in self.env.config.symbols
                }
                
                # Record to Failure Bank
                self.poel_meta_agent.record_failure(
                    state=local_prev_states[0],  # Use last state from first agent
                    dd_breach_size=dd_breach_size,
                    agent_weights=agent_weights,
                    symbol=self.env.config.symbols[0],
                    episode_id=self.episode,
                    step=step,
                    balance=final_balance,
                    pnl=portfolio_pnl,
                    failure_type='total_dd',
                )
                
                logger.info(f"  [OK] Failure recorded (breach size: {dd_breach_size:.2%})")
            
            # 3. Checkpoint Skills (if high Calmar Ratio)
            for symbol, perf in meta_result['agent_performances'].items():
                calmar = perf['calmar_ratio']
                
                if calmar > 2.0:  # High Calmar threshold
                    logger.info(f"\n  [SKILL] High Calmar detected: {symbol} = {calmar:.2f}")
                    
                    # Get novelty score from POEL shaper statistics
                    shaper_stats = self.poel_shapers[symbol].get_statistics()
                    novelty_score = shaper_stats.get('avg_novelty', 0.0)
                    
                    # Checkpoint skill
                    self.poel_meta_agent.checkpoint_skill(
                        agent_id=f"agent_{symbol}",
                        symbol=symbol,
                        calmar_ratio=calmar,
                        novelty_score=novelty_score,
                        model_weights={'placeholder': np.random.randn(100)},  # TODO: Extract actual weights
                        episode_id=self.episode,
                        steps_trained=self.total_steps,
                        skill_name=f"{symbol} Episode {self.episode} Strategy",
                    )
                    
                    logger.info(f"  [SKILL] Skill checkpointed (Calmar={calmar:.2f}, Novelty={novelty_score:.3f})")
            
            # 4. End Episode
            episode_summary = self.poel_meta_agent.end_episode()
            
            logger.info(f"\n  Episode Summary:")
            logger.info(f"    Failure Bank Size: {episode_summary['failure_bank_size']}")
            logger.info(f"    Skill Bank Size: {episode_summary['skill_bank_size']}")
            logger.info("=" * 80 + "\n")
        # ===== END POEL META-AGENT COORDINATION =====
        
        logger.info(
            f"Episode {self.episode} finished: {step} steps, "
            f"reward={episode_reward:.2f}, balance=${final_balance:,.0f}, dd={final_dd:.2%}"
        )
        
        return metrics
    
    def _update_meta_agent(self, next_state: np.ndarray, done: bool):
        """
        Update Meta-Agent with collected experience
        
        Args:
            next_state: Next meta-state for bootstrapping
            done: Episode done flag
        """
        # Convert to tensors
        states = torch.FloatTensor(np.array(self.meta_states))
        actions = torch.FloatTensor(np.array(self.meta_actions))
        
        # Compute returns
        if done:
            next_value = 0.0
        else:
            with torch.no_grad():
                _, _, value = self.meta_agent.forward(
                    torch.FloatTensor(next_state).unsqueeze(0)
                )
                next_value = value.item()
        
        returns = self.meta_agent.compute_returns(
            self.meta_rewards,
            self.meta_dones,
            next_value
        )
        
        # Update
        metrics = self.meta_agent.update(states, actions, returns)
        
        # Clear buffers
        self.meta_states = []
        self.meta_actions = []
        self.meta_rewards = []
        self.meta_dones = []
        
        logger.debug(f"Meta-Agent updated: actor_loss={metrics['actor_loss']:.4f}")
    
    def evaluate(self, num_episodes: int = 10) -> Dict[str, float]:
        """
        Evaluate trained MARL system
        
        Args:
            num_episodes: Number of evaluation episodes
        
        Returns:
            metrics: Evaluation metrics
        """
        logger.info(f"Evaluating MARL system for {num_episodes} episodes...")
        
        # Set agents to eval mode (disables BatchNorm/Dropout)
        for agent in self.local_agents:
            if hasattr(agent, 'actor'):
                agent.actor.eval()
            if hasattr(agent, 'critic'):
                if isinstance(agent.critic, torch.nn.Module):
                    agent.critic.eval()
        
        eval_rewards = []
        eval_dds = []
        eval_balances = []
        
        try:
            for ep in range(num_episodes):
                meta_state, _ = self.env.reset()
                
                episode_reward = 0.0
                done = False
                steps = 0
                
                while not done:
                    # Deterministic Meta-Action
                    meta_action, _, _ = self.meta_agent.select_meta_action(
                        meta_state,
                        deterministic=True
                    )
                
                    # Apply risk limits
                    self.env._apply_meta_action(meta_action)
                    
                    # Deterministic local actions
                    local_actions = []
                    for i, agent in enumerate(self.local_agents):
                        local_state = self.env.local_envs[i]._get_observation()
                        local_action = agent.select_action(local_state, explore=False)
                        local_actions.append(local_action)
                    
                    # Execute
                    local_states = []
                    local_rewards = []
                    local_dones = []
                    
                    for i, (env, action) in enumerate(zip(self.env.local_envs, local_actions)):
                        state, reward, done_local, truncated, info = env.step(action)
                        local_states.append(state)
                        local_rewards.append(reward)
                        local_dones.append(done_local or truncated)
                    
                    meta_reward = sum(local_rewards)
                    next_meta_state = self.env._build_meta_state(local_states)
                    done = any(local_dones) or self.env._calculate_global_dd() > self.env.config.max_global_dd_pct
                    
                    episode_reward += meta_reward
                    meta_state = next_meta_state
                    steps += 1
                
                eval_rewards.append(episode_reward)
                eval_dds.append(self.env._calculate_global_dd())
                eval_balances.append(self.env.current_balance)
                
                # Log evaluation progress every 10 episodes
                if (ep + 1) % 10 == 0 or (ep + 1) == num_episodes:
                    logger.info(
                        f"Evaluation progress: {ep + 1}/{num_episodes} episodes | "
                        f"Last: reward={episode_reward:.2f}, balance=${self.env.current_balance:,.0f}, "
                        f"dd={self.env._calculate_global_dd():.2%}, steps={steps}"
                    )
        
        finally:
            # Restore training mode
            for agent in self.local_agents:
                if hasattr(agent, 'actor'):
                    agent.actor.train()
                if hasattr(agent, 'critic'):
                    if isinstance(agent.critic, torch.nn.Module):
                        agent.critic.train()
        
        metrics = {
            'eval_reward_mean': np.mean(eval_rewards),
            'eval_reward_std': np.std(eval_rewards),
            'eval_dd_mean': np.mean(eval_dds),
            'eval_dd_max': np.max(eval_dds),
            'eval_balance_mean': np.mean(eval_balances),
        }
        
        logger.info("============================================================")
        logger.info("EVALUATION COMPLETE")
        logger.info("============================================================")
        logger.info(
            f"Evaluation complete: reward={metrics['eval_reward_mean']:.2f}±{metrics['eval_reward_std']:.2f}, "
            f"dd={metrics['eval_dd_mean']:.2%} (max={metrics['eval_dd_max']:.2%}), "
            f"balance_mean=${metrics['eval_balance_mean']:,.0f}"
        )
        logger.info("============================================================")
        
        return metrics
    
    def save_checkpoint(self, path: str):
        """Save training checkpoint"""
        checkpoint = {
            'episode': self.episode,
            'total_steps': self.total_steps,
            'meta_agent': self.meta_agent.state_dict(),
            'local_agents': [agent.state_dict() for agent in self.local_agents],
            'episode_rewards': self.episode_rewards,
            'episode_dds': self.episode_dds,
        }
        torch.save(checkpoint, path)
        logger.info(f"Checkpoint saved to {path}")


def main():
    """Main training function"""
    
    # Parse arguments
    parser = argparse.ArgumentParser(description='Train MARL system')
    parser.add_argument('--episodes', type=int, default=1000, help='Number of training episodes')
    parser.add_argument('--eval-freq', type=int, default=100, help='Evaluation frequency (episodes)')
    parser.add_argument('--checkpoint-freq', type=int, default=500, help='Checkpoint frequency (episodes)')
    parser.add_argument('--symbols', nargs='+', default=['EURUSD', 'GBPUSD', 'USDJPY', 'USDCHF'], help='Currency pairs')
    parser.add_argument('--balance', type=float, default=100000.0, help='Initial portfolio balance')
    
    # POEL arguments
    parser.add_argument('--poel', action='store_true', help='Enable POEL (Purpose-Driven Open-Ended Learning)')
    parser.add_argument('--poel-alpha', type=float, default=0.85, help='POEL alpha (PnL weight): 0.85 = 85%% PnL, 15%% exploration')
    parser.add_argument('--poel-beta', type=float, default=3.0, help='POEL beta (stability penalty weight): 3.0 = extreme risk aversion')
    parser.add_argument('--nrf', action='store_true', help='Enable NRF (Neural Reward Functions) for skill discovery')
    parser.add_argument('--nrf-cycle', type=int, default=20, help='NRF cycle frequency (1 in N episodes)')
    parser.add_argument('--log-name', type=str, default=None, help='Custom metrics CSV filename (e.g., poel_risk_focused_metrics.csv)')
    
    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        handlers=[
            logging.FileHandler(f'data/test_results/marl_training_{datetime.now():%Y%m%d_%H%M%S}.log'),
            logging.StreamHandler()
        ]
    )
    
    logger.info("="*60)
    logger.info("MARL TRAINING START")
    logger.info("="*60)
    logger.info(f"Episodes: {args.episodes}")
    logger.info(f"Symbols: {args.symbols}")
    logger.info(f"Initial Balance: ${args.balance:,.0f}")
    logger.info(f"POEL Enabled: {args.poel}")
    if args.poel:
        logger.info(f"  - POEL Alpha: {args.poel_alpha} (PnL weight)")
        logger.info(f"  - POEL Beta: {args.poel_beta} (Stability weight)")
        logger.info(f"  - NRF Enabled: {args.nrf}")
        if args.nrf:
            logger.info(f"  - NRF Cycle: 1 in {args.nrf_cycle} episodes")
    logger.info("="*60)
    
    # Create environment
    env_config = MultiAssetConfig(
        symbols=args.symbols,
        initial_balance=args.balance,
        data_source='historical'
    )
    env = MultiAssetEnv(config=env_config)
    
    # Create Meta-Agent
    meta_agent = A3CMetaAgent(
        config=A3CConfig(
            meta_state_dim=23,  # 🧠 Expanded: 15 (base) + 4 (correlations) + 1 (macro) + 3 (opportunity)
            meta_action_dim=len(args.symbols),
            hidden_dim=128
        )
    )
    
    # Create Local Agents (HETEROGENEOUS ARCHITECTURE)
    # Each symbol gets an optimized algorithm + architecture
    local_agents = []
    agent_names = []  # Track agent types for logging
    
    for symbol in args.symbols:
        # Get asset-specific metadata
        metadata = get_agent_metadata(symbol)
        agent_config = metadata['config']
        algorithm = metadata['algorithm']
        architecture = metadata['architecture']
        
        # Override base config with training-specific settings
        agent_config.state_dim = 31  # Market Awareness + Financial Intelligence (29 + 2 staleness features)
        agent_config.action_dim = 1  # Position size only (simplified)
        
        # Create agent based on algorithm type
        if algorithm == 'TD3':
            agent = TD3Agent(config=agent_config)
        elif algorithm == 'PPO':
            agent = PPOAgent(config=agent_config)
        elif algorithm == 'SAC':
            agent = SACAgent(config=agent_config)
        elif algorithm == 'DDPG':
            agent = DDPGAgent(config=agent_config)
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")
        
        local_agents.append(agent)
        agent_names.append(f"{symbol}:{algorithm}+{architecture}")
        
        logger.info(f"  [{symbol}] {algorithm} + {architecture} - {metadata['description']}")
    
    logger.info(f"\nHeterogeneous MARL System:")
    for name in agent_names:
        logger.info(f"  {name}")
    
    # Create trainer
    training_config = {
        'td3_update_freq': 2,  # Update TD3 every 2 steps
        'initial_balance': args.balance,
        'max_daily_dd': 0.05,
        'max_total_dd': 0.10,
        # POEL configuration
        'poel_alpha': args.poel_alpha,
        'poel_beta': args.poel_beta,
        'poel_max_local_dd': 0.15,  # 15% local DD limit
        'nrf_enabled': args.nrf,
        'nrf_cycle_frequency': args.nrf_cycle,
    }
    
    trainer = MARLTrainer(
        env=env,
        meta_agent=meta_agent,
        local_agents=local_agents,
        config=training_config,
        agent_names=agent_names,  # NEW: Pass agent names for CSV logging
        poel_enabled=args.poel,  # POEL Integration
        log_name=args.log_name,  # Custom metrics filename
    )
    
    # Training loop
    for episode in range(1, args.episodes + 1):
        metrics = trainer.train_episode()
        
        # Evaluation
        if episode % args.eval_freq == 0:
            eval_metrics = trainer.evaluate(num_episodes=10)
            logger.info(f"Evaluation at episode {episode}: {eval_metrics}")
        
        # Checkpoint
        if episode % args.checkpoint_freq == 0:
            checkpoint_path = f"models/marl_checkpoint_ep{episode}.pth"
            trainer.save_checkpoint(checkpoint_path)
    
    # Final evaluation
    logger.info("\n" + "="*60)
    logger.info("TRAINING COMPLETE - Final Evaluation")
    logger.info("="*60)
    final_metrics = trainer.evaluate(num_episodes=50)
    logger.info(f"Final Metrics: {final_metrics}")
    
    # Save final model
    logger.info("\nSaving final models...")
    meta_agent.save("models/marl_meta_agent_final.pth")
    logger.info("  - Saved: models/marl_meta_agent_final.pth")
    
    for i, agent in enumerate(local_agents):
        model_path = f"models/marl_td3_{args.symbols[i]}_final.pth"
        agent.save(model_path)
        logger.info(f"  - Saved: {model_path}")
    
    logger.info("\n" + "="*60)
    logger.info("ALL PROCESSES COMPLETE - SCRIPT FINISHED")
    logger.info("="*60)
    logger.info("MARL training finished successfully!")
    logger.info(f"Training episodes: {args.episodes}")
    logger.info(f"Evaluation episodes: 50")
    logger.info(f"Final balance mean: ${final_metrics['eval_balance_mean']:,.0f}")
    
    # Log appropriate filename based on POEL usage
    log_file = "logs/poel_metrics.csv" if args.poel else "logs/baseline_metrics.csv"
    logger.info(f"Logs saved to: {log_file}")
    logger.info("="*60)


if __name__ == "__main__":
    main()
