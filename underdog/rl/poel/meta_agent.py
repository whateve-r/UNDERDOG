"""
POEL Meta-Agent Coordinator.

Integrates all POEL modules for large-scale MARL coordination:
- Module 1: Purpose-Driven Risk Control (local agents)
- Module 2.1: Dynamic Capital Allocation (meta-level)
- Module 2.2: Failure Bank & Curriculum (learning)
- NRF: Neural Reward Functions (skill discovery)

This is the central orchestrator for the full POEL system.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)

from underdog.rl.poel.purpose import PurposeFunction, BusinessPurpose
from underdog.rl.poel.capital_allocator import (
    CapitalAllocator,
    CalmarCalculator,
    AgentPerformance,
    create_agent_performance,
)
from underdog.rl.poel.failure_bank import (
    FailureBank,
    SkillRepository,
    CurriculumManager,
)
from underdog.rl.poel.neural_reward import (
    NeuralRewardFunction,
    NRFSkillDiscovery,
    NRFConfig,
    create_nrf_system,
)


class TrainingMode(Enum):
    """Training mode for POEL system."""
    NORMAL = "normal"  # Standard PnL-focused training
    NRF = "nrf"  # Neural Reward Function skill discovery
    CURRICULUM = "curriculum"  # Failure recovery training
    EMERGENCY = "emergency"  # Zero-risk emergency mode


@dataclass
class POELMetaState:
    """State of the POEL Meta-Agent."""
    mode: TrainingMode
    current_balance: float
    total_dd: float
    daily_dd: float
    emergency_active: bool
    nrf_cycle: int
    curriculum_injected: bool
    agent_weights: Dict[str, float]  # Capital allocation
    purpose_score: float


class POELMetaAgent:
    """
    Meta-Agent for POEL coordination.
    
    Responsibilities:
    1. Monitor global Purpose (DD limits)
    2. Allocate capital via Calmar Ratio
    3. Trigger emergency protocol
    4. Manage NRF skill discovery cycles
    5. Inject failure states for curriculum learning
    6. Checkpoint successful skills
    """
    
    def __init__(
        self,
        initial_balance: float = 100000.0,
        symbols: List[str] = None,
        max_daily_dd: float = 0.05,
        max_total_dd: float = 0.10,
        nrf_enabled: bool = True,
        nrf_cycle_frequency: int = 20,  # 1 in 20 training runs (REDUCED from 10 - less frequent exploration)
        state_dim: int = 31,
    ):
        """
        Args:
            initial_balance: Starting capital
            symbols: List of trading symbols
            max_daily_dd: Maximum daily drawdown (5% default)
            max_total_dd: Maximum total drawdown (10% default)
            nrf_enabled: Enable Neural Reward Function skill discovery
            nrf_cycle_frequency: Run NRF every N training cycles
            state_dim: Dimension of state space
        """
        self.initial_balance = initial_balance
        self.symbols = symbols or ['EURUSD', 'USDJPY', 'XAUUSD', 'GBPUSD']
        self.state_dim = state_dim
        
        # Module 1: Purpose Function
        self.purpose_fn = PurposeFunction(
            BusinessPurpose(
                lambda_daily_dd=10.0,
                lambda_total_dd=20.0,
                max_daily_dd_pct=max_daily_dd,
                max_total_dd_pct=max_total_dd,
                emergency_threshold_pct=0.80,
            )
        )
        self.purpose_fn.reset(initial_balance)
        
        # Module 2.1: Capital Allocator
        self.capital_allocator = CapitalAllocator(
            emergency_threshold=0.80,
            max_total_dd=max_total_dd,
            epsilon=1e-6,
            min_allocation=0.05,
        )
        
        # Module 2.2: Failure Bank & Skills
        self.failure_bank = FailureBank(max_size=1000)
        self.skill_repository = SkillRepository(max_size=100)
        self.curriculum_manager = CurriculumManager(
            failure_bank=self.failure_bank,
            injection_rate=0.15,
            warmup_episodes=10,
        )
        
        # NRF: Neural Reward Functions
        self.nrf_enabled = nrf_enabled
        self.nrf_cycle_frequency = nrf_cycle_frequency
        self.nrf_system: Optional[NRFSkillDiscovery] = None
        
        if nrf_enabled:
            self.nrf_system = create_nrf_system(
                state_dim=state_dim,
                hidden_dims=[128, 64],
                learning_rate=3e-4,
                buffer_size=50000,
                reward_scale=10.0,
            )
        
        # State tracking
        self.current_mode = TrainingMode.NORMAL
        self.episode_count = 0
        self.nrf_cycle_count = 0
        self.agent_performances: Dict[str, AgentPerformance] = {}
        
        # History for Calmar calculation
        self.agent_pnl_history: Dict[str, List[float]] = {s: [] for s in self.symbols}
        self.agent_balance_history: Dict[str, List[float]] = {s: [] for s in self.symbols}
    
    def start_episode(self) -> Dict[str, any]:
        """
        Initialize new episode.
        
        Returns:
            Dict with episode config and initial state
        """
        self.episode_count += 1
        
        # Determine training mode for this episode
        mode = self._determine_training_mode()
        self.current_mode = mode
        
        # Check if curriculum injection should occur
        inject_failure = False
        failure_state = None
        
        if mode == TrainingMode.CURRICULUM or self.curriculum_manager.should_inject_failure():
            failure_state = self.curriculum_manager.get_failure_state()
            inject_failure = failure_state is not None
        
        # NRF phase setup
        nrf_config = None
        if mode == TrainingMode.NRF and self.nrf_system:
            nrf_config = self.nrf_system.start_skill_generation()
        
        return {
            'episode': self.episode_count,
            'mode': mode.value,
            'inject_failure': inject_failure,
            'failure_state': failure_state,
            'nrf_config': nrf_config,
            'agent_weights': self._get_current_weights(),
        }
    
    def update_agent_performance(
        self,
        agent_id: str,
        symbol: str,
        pnl: float,
        balance: float,
    ):
        """
        Update agent performance metrics.
        
        Call this after each step for each agent.
        
        Args:
            agent_id: Agent identifier
            symbol: Trading symbol
            pnl: PnL for this step
            balance: Current balance
        """
        if symbol not in self.agent_pnl_history:
            self.agent_pnl_history[symbol] = []
            self.agent_balance_history[symbol] = []
        
        self.agent_pnl_history[symbol].append(pnl)
        self.agent_balance_history[symbol].append(balance)
    
    def compute_purpose_and_allocate(
        self,
        current_balance: float,
        daily_dd_pct: float,
        total_dd_pct: float,
        portfolio_pnl: float,
    ) -> Dict[str, any]:
        """
        Core meta-agent step: evaluate purpose and allocate capital.
        
        Args:
            current_balance: Total portfolio balance
            daily_dd_pct: Current daily DD percentage
            total_dd_pct: Current total DD percentage
            portfolio_pnl: Total portfolio PnL
        
        Returns:
            Dict with purpose metrics, capital allocation, emergency status
        """
        # 1. Compute Purpose
        purpose_metrics = self.purpose_fn.compute_purpose(
            current_balance=current_balance,
            daily_dd_pct=daily_dd_pct,
            total_dd_pct=total_dd_pct,
            pnl=portfolio_pnl,
        )
        
        # 2. Update agent performances
        agent_perfs = []
        for symbol in self.symbols:
            if symbol in self.agent_pnl_history:
                perf = create_agent_performance(
                    agent_id=f"agent_{symbol}",
                    symbol=symbol,
                    pnl_history=self.agent_pnl_history[symbol],
                    balance_history=self.agent_balance_history[symbol],
                    initial_balance=self.initial_balance / len(self.symbols),
                )
                agent_perfs.append(perf)
                self.agent_performances[symbol] = perf
        
        # 3. Allocate capital
        if agent_perfs:
            weights = self.capital_allocator.allocate_weights(
                agent_performances=agent_perfs,
                current_total_dd=total_dd_pct,
            )
        else:
            # No performance data yet: equal weights
            weights = {s: 1.0 / len(self.symbols) for s in self.symbols}
        
        # 4. Check emergency protocol
        emergency_signal = self.capital_allocator.get_emergency_signal()
        
        # 5. Update mode if emergency
        if emergency_signal['emergency_mode']:
            self.current_mode = TrainingMode.EMERGENCY
            logger.critical(
                f"[EMERGENCY] MODE ACTIVATED - Total DD: {total_dd_pct:.2%}, "
                f"Action: {emergency_signal['action']}, "
                f"Message: {emergency_signal.get('message', 'N/A')}"
            )
        
        return {
            'purpose': purpose_metrics,
            'weights': weights,
            'emergency': emergency_signal,
            'agent_performances': {
                symbol: {
                    'calmar_ratio': perf.calmar_ratio,
                    'max_dd': perf.max_dd,
                    'recent_pnl': perf.recent_pnl,
                }
                for symbol, perf in self.agent_performances.items()
            },
        }
    
    def record_failure(
        self,
        state: np.ndarray,
        dd_breach_size: float,
        agent_weights: Dict[str, np.ndarray],
        symbol: str,
        episode_id: int,
        step: int,
        balance: float,
        pnl: float,
        failure_type: str = 'total_dd',
    ):
        """
        Record failure event to Failure Bank.
        
        Args:
            state: State at failure point
            dd_breach_size: Magnitude of DD breach
            agent_weights: Model weights at failure
            symbol: Trading symbol
            episode_id: Episode number
            step: Step within episode
            balance: Balance at failure
            pnl: PnL at failure
            failure_type: Type of failure
        """
        self.failure_bank.add_failure(
            state=state,
            dd_breach_size=dd_breach_size,
            agent_weights=agent_weights,
            symbol=symbol,
            episode_id=episode_id,
            step=step,
            balance=balance,
            pnl=pnl,
            failure_type=failure_type,
        )
    
    def checkpoint_skill(
        self,
        agent_id: str,
        symbol: str,
        calmar_ratio: float,
        novelty_score: float,
        model_weights: Dict[str, np.ndarray],
        episode_id: int,
        steps_trained: int,
        skill_name: str = "Discovered Skill",
    ):
        """
        Save successful skill to repository.
        
        Args:
            agent_id: Agent identifier
            symbol: Trading symbol
            calmar_ratio: Calmar Ratio at checkpoint
            novelty_score: Novelty score
            model_weights: Model parameters
            episode_id: Episode number
            steps_trained: Total training steps
            skill_name: Skill description
        """
        perf = self.agent_performances.get(symbol)
        if not perf:
            return
        
        self.skill_repository.add_skill(
            agent_id=agent_id,
            symbol=symbol,
            calmar_ratio=calmar_ratio,
            novelty_score=novelty_score,
            avg_pnl=perf.recent_pnl,
            max_dd=perf.max_dd,
            sharpe_ratio=0.0,  # TODO: Calculate from history
            episode_id=episode_id,
            steps_trained=steps_trained,
            model_weights=model_weights,
            skill_name=skill_name,
        )
    
    def nrf_step(self, state: np.ndarray, training_step: int) -> Optional[Dict[str, any]]:
        """
        Execute NRF step (if in NRF mode).
        
        Args:
            state: Current state
            training_step: Current training step
        
        Returns:
            NRF metrics if update occurred, else None
        """
        if not self.nrf_enabled or not self.nrf_system:
            return None
        
        if self.current_mode == TrainingMode.NRF:
            # Add state to buffer
            self.nrf_system.nrf.add_state(state)
            
            # Update R_ψ if needed
            return self.nrf_system.step(training_step)
        
        return None
    
    def get_nrf_reward(self, state: np.ndarray) -> float:
        """
        Get NRF reward for state (use during NRF mode).
        
        Args:
            state: State vector
        
        Returns:
            NRF reward value
        """
        if not self.nrf_enabled or not self.nrf_system:
            return 0.0
        
        return self.nrf_system.nrf.compute_reward(state)
    
    def end_episode(self) -> Dict[str, any]:
        """
        Finalize episode and prepare for next.
        
        Returns:
            Episode summary
        """
        summary = {
            'episode': self.episode_count,
            'mode': self.current_mode.value,
            'failure_bank_size': len(self.failure_bank.failures),
            'skill_bank_size': len(self.skill_repository.skills),
        }
        
        # NRF cycle management
        if self.current_mode == TrainingMode.NRF and self.nrf_system:
            cycle_summary = self.nrf_system.end_cycle()
            summary['nrf_cycle'] = cycle_summary
        
        return summary
    
    def get_meta_state(self) -> POELMetaState:
        """Get current meta-agent state."""
        purpose_metrics = self.purpose_fn.compute_purpose(
            current_balance=self.initial_balance,  # Placeholder
            daily_dd_pct=0.0,
            total_dd_pct=0.0,
            pnl=0.0,
        )
        
        return POELMetaState(
            mode=self.current_mode,
            current_balance=self.initial_balance,
            total_dd=0.0,
            daily_dd=0.0,
            emergency_active=self.capital_allocator.emergency_mode,
            nrf_cycle=self.nrf_cycle_count,
            curriculum_injected=False,  # Updated per episode
            agent_weights=self._get_current_weights(),
            purpose_score=purpose_metrics['purpose'],
        )
    
    def get_statistics(self) -> Dict[str, any]:
        """Get comprehensive POEL statistics."""
        stats = {
            'episode_count': self.episode_count,
            'current_mode': self.current_mode.value,
            'nrf_cycle_count': self.nrf_cycle_count,
            'failure_bank': self.failure_bank.get_statistics(),
            'skill_bank': self.skill_repository.get_statistics(),
            'curriculum': self.curriculum_manager.get_statistics(),
            'emergency_mode': self.capital_allocator.emergency_mode,
            'agent_performances': {
                symbol: {
                    'calmar': perf.calmar_ratio,
                    'max_dd': perf.max_dd,
                    'pnl': perf.recent_pnl,
                    'weight': perf.weight,
                }
                for symbol, perf in self.agent_performances.items()
            },
        }
        
        if self.nrf_enabled and self.nrf_system:
            stats['nrf'] = self.nrf_system.nrf.get_statistics()
        
        return stats
    
    def _determine_training_mode(self) -> TrainingMode:
        """
        Determine training mode for current episode.
        
        Priority:
        1. Emergency (if DD threshold exceeded)
        2. NRF (1 in N episodes)
        3. Curriculum (15% of episodes after warmup)
        4. Normal (default)
        """
        # Emergency mode persists
        if self.capital_allocator.emergency_mode:
            return TrainingMode.EMERGENCY
        
        # NRF cycle (1 in nrf_cycle_frequency episodes)
        if self.nrf_enabled and (self.episode_count % self.nrf_cycle_frequency == 0):
            self.nrf_cycle_count += 1
            return TrainingMode.NRF
        
        # Curriculum injection (handled by CurriculumManager)
        # Just use normal mode, injection happens transparently
        
        return TrainingMode.NORMAL
    
    def _get_current_weights(self) -> Dict[str, float]:
        """Get current capital allocation weights."""
        if not self.agent_performances:
            return {s: 1.0 / len(self.symbols) for s in self.symbols}
        
        return {
            symbol: perf.weight
            for symbol, perf in self.agent_performances.items()
        }
    
    def reset(self):
        """Reset meta-agent for new training run."""
        self.episode_count = 0
        self.nrf_cycle_count = 0
        self.current_mode = TrainingMode.NORMAL
        self.agent_performances.clear()
        self.agent_pnl_history = {s: [] for s in self.symbols}
        self.agent_balance_history = {s: [] for s in self.symbols}
        self.capital_allocator.reset()
        self.purpose_fn.reset(self.initial_balance)
        self.curriculum_manager.reset_episode_count()
