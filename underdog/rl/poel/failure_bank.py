"""
Failure Bank & Skill Repository for Curriculum Learning.

Stores failure states (DD breaches) and successful skills (high Calmar + Novelty).
Enables curriculum retraining by injecting failure states into episode initialization.
"""

from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple
import numpy as np
from datetime import datetime
import json


@dataclass
class FailureRecord:
    """Record of a failure event (DD breach or crash)."""
    timestamp: datetime
    symbol: str
    state_vector: np.ndarray  # Full state at failure point
    dd_breach_size: float  # How much DD limit was exceeded
    agent_weights: Dict[str, float]  # Model weights at failure
    episode_id: int
    step: int
    balance: float
    pnl: float
    failure_type: str  # 'daily_dd', 'total_dd', 'crash'
    

@dataclass
class SkillCheckpoint:
    """Checkpoint of a successful skill (high performance)."""
    timestamp: datetime
    agent_id: str
    symbol: str
    calmar_ratio: float
    novelty_score: float
    avg_pnl: float
    max_dd: float
    sharpe_ratio: float
    episode_id: int
    steps_trained: int
    model_weights: Dict[str, float]  # Serialized model parameters
    skill_name: str  # E.g., "High-Volatility Scalping", "Trend Following"


class FailureBank:
    """
    In-memory failure bank for curriculum learning.
    
    Stores failure states and provides sampling for episode initialization.
    Can be persisted to TimescaleDB for long-term storage.
    """
    
    def __init__(self, max_size: int = 1000):
        """
        Args:
            max_size: Maximum number of failure records to keep in memory
        """
        self.max_size = max_size
        self.failures: List[FailureRecord] = []
        self.failure_counts: Dict[str, int] = {}  # Count by failure_type
    
    def add_failure(
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
        Add a failure record to the bank.
        
        Args:
            state: Full state vector at failure point
            dd_breach_size: Magnitude of DD breach (e.g., 0.03 for 3% over limit)
            agent_weights: Dict of agent_id -> model weights (numpy arrays)
            symbol: Trading symbol
            episode_id: Episode number
            step: Step within episode
            balance: Balance at failure
            pnl: PnL at failure
            failure_type: Type of failure ('daily_dd', 'total_dd', 'crash')
        """
        # Serialize agent weights (convert numpy to list for storage)
        serialized_weights = {
            agent_id: weights.tolist() if isinstance(weights, np.ndarray) else weights
            for agent_id, weights in agent_weights.items()
        }
        
        record = FailureRecord(
            timestamp=datetime.now(),
            symbol=symbol,
            state_vector=state.copy(),
            dd_breach_size=dd_breach_size,
            agent_weights=serialized_weights,
            episode_id=episode_id,
            step=step,
            balance=balance,
            pnl=pnl,
            failure_type=failure_type,
        )
        
        self.failures.append(record)
        
        # Update counts
        self.failure_counts[failure_type] = self.failure_counts.get(failure_type, 0) + 1
        
        # Evict oldest if over capacity
        if len(self.failures) > self.max_size:
            removed = self.failures.pop(0)
            self.failure_counts[removed.failure_type] -= 1
    
    def sample_failure_state(
        self,
        n_samples: int = 1,
        failure_type: Optional[str] = None,
    ) -> List[np.ndarray]:
        """
        Sample failure states for curriculum injection.
        
        Args:
            n_samples: Number of states to sample
            failure_type: Filter by failure type (None = all types)
        
        Returns:
            List of state vectors
        """
        if not self.failures:
            return []
        
        # Filter by type if specified
        candidates = self.failures
        if failure_type:
            candidates = [f for f in self.failures if f.failure_type == failure_type]
        
        if not candidates:
            return []
        
        # Sample randomly
        n_samples = min(n_samples, len(candidates))
        sampled = np.random.choice(candidates, size=n_samples, replace=False)
        
        return [f.state_vector for f in sampled]
    
    def get_most_severe_failures(self, top_k: int = 10) -> List[FailureRecord]:
        """
        Get the k most severe failures (by DD breach size).
        
        Args:
            top_k: Number of failures to return
        
        Returns:
            List of FailureRecord objects, sorted by severity
        """
        sorted_failures = sorted(
            self.failures,
            key=lambda f: f.dd_breach_size,
            reverse=True,
        )
        return sorted_failures[:top_k]
    
    def get_statistics(self) -> Dict[str, any]:
        """Get failure bank statistics."""
        if not self.failures:
            return {
                'total_failures': 0,
                'by_type': {},
                'avg_breach_size': 0.0,
                'max_breach_size': 0.0,
            }
        
        breach_sizes = [f.dd_breach_size for f in self.failures]
        
        return {
            'total_failures': len(self.failures),
            'by_type': self.failure_counts.copy(),
            'avg_breach_size': np.mean(breach_sizes),
            'max_breach_size': np.max(breach_sizes),
            'oldest_failure': self.failures[0].timestamp,
            'newest_failure': self.failures[-1].timestamp,
        }
    
    def clear(self):
        """Clear all failure records."""
        self.failures.clear()
        self.failure_counts.clear()


class SkillRepository:
    """
    Repository for successful skill checkpoints.
    
    Stores high-performing agent configurations for ensemble building
    and transfer learning.
    """
    
    def __init__(self, max_size: int = 100):
        """
        Args:
            max_size: Maximum number of skills to keep in memory
        """
        self.max_size = max_size
        self.skills: List[SkillCheckpoint] = []
        self.skills_by_agent: Dict[str, List[SkillCheckpoint]] = {}
    
    def add_skill(
        self,
        agent_id: str,
        symbol: str,
        calmar_ratio: float,
        novelty_score: float,
        avg_pnl: float,
        max_dd: float,
        sharpe_ratio: float,
        episode_id: int,
        steps_trained: int,
        model_weights: Dict[str, np.ndarray],
        skill_name: str = "Discovered Skill",
    ):
        """
        Add a skill checkpoint to the repository.
        
        Args:
            agent_id: Agent identifier
            symbol: Trading symbol
            calmar_ratio: Calmar Ratio at checkpoint
            novelty_score: Novelty score at checkpoint
            avg_pnl: Average PnL over recent window
            max_dd: Max DD over recent window
            sharpe_ratio: Sharpe Ratio estimate
            episode_id: Episode number
            steps_trained: Total training steps
            model_weights: Model parameters (numpy arrays)
            skill_name: Human-readable skill description
        """
        # Serialize weights
        serialized_weights = {
            key: weights.tolist() if isinstance(weights, np.ndarray) else weights
            for key, weights in model_weights.items()
        }
        
        checkpoint = SkillCheckpoint(
            timestamp=datetime.now(),
            agent_id=agent_id,
            symbol=symbol,
            calmar_ratio=calmar_ratio,
            novelty_score=novelty_score,
            avg_pnl=avg_pnl,
            max_dd=max_dd,
            sharpe_ratio=sharpe_ratio,
            episode_id=episode_id,
            steps_trained=steps_trained,
            model_weights=serialized_weights,
            skill_name=skill_name,
        )
        
        self.skills.append(checkpoint)
        
        # Index by agent
        if agent_id not in self.skills_by_agent:
            self.skills_by_agent[agent_id] = []
        self.skills_by_agent[agent_id].append(checkpoint)
        
        # Evict oldest if over capacity
        if len(self.skills) > self.max_size:
            removed = self.skills.pop(0)
            self.skills_by_agent[removed.agent_id].remove(removed)
    
    def get_best_skills(
        self,
        top_k: int = 5,
        metric: str = 'calmar_ratio',
    ) -> List[SkillCheckpoint]:
        """
        Get top-k skills by specified metric.
        
        Args:
            top_k: Number of skills to return
            metric: Metric to sort by ('calmar_ratio', 'sharpe_ratio', 'novelty_score')
        
        Returns:
            List of SkillCheckpoint objects
        """
        if not self.skills:
            return []
        
        sorted_skills = sorted(
            self.skills,
            key=lambda s: getattr(s, metric),
            reverse=True,
        )
        return sorted_skills[:top_k]
    
    def get_agent_skills(self, agent_id: str) -> List[SkillCheckpoint]:
        """Get all skills for a specific agent."""
        return self.skills_by_agent.get(agent_id, [])
    
    def get_statistics(self) -> Dict[str, any]:
        """Get repository statistics."""
        if not self.skills:
            return {
                'total_skills': 0,
                'by_agent': {},
                'avg_calmar': 0.0,
                'avg_novelty': 0.0,
            }
        
        return {
            'total_skills': len(self.skills),
            'by_agent': {aid: len(skills) for aid, skills in self.skills_by_agent.items()},
            'avg_calmar': np.mean([s.calmar_ratio for s in self.skills]),
            'avg_novelty': np.mean([s.novelty_score for s in self.skills]),
            'best_calmar': max(s.calmar_ratio for s in self.skills),
            'best_sharpe': max(s.sharpe_ratio for s in self.skills),
        }
    
    def clear(self):
        """Clear all skill checkpoints."""
        self.skills.clear()
        self.skills_by_agent.clear()


class CurriculumManager:
    """
    Manages curriculum learning by injecting failure states into episodes.
    
    Coordinates between FailureBank and episode initialization.
    """
    
    def __init__(
        self,
        failure_bank: FailureBank,
        injection_rate: float = 0.15,  # 15% of episodes start from failures
        warmup_episodes: int = 10,  # Don't inject until after warmup
    ):
        """
        Args:
            failure_bank: FailureBank instance
            injection_rate: Fraction of episodes to start from failure states
            warmup_episodes: Number of episodes before injection starts
        """
        self.failure_bank = failure_bank
        self.injection_rate = injection_rate
        self.warmup_episodes = warmup_episodes
        
        self.episode_count = 0
        self.injected_count = 0
    
    def should_inject_failure(self) -> bool:
        """
        Determine if current episode should start from failure state.
        
        Returns:
            True if injection should occur
        """
        self.episode_count += 1
        
        # Don't inject during warmup
        if self.episode_count <= self.warmup_episodes:
            return False
        
        # Random injection based on rate
        if np.random.random() < self.injection_rate:
            self.injected_count += 1
            return True
        
        return False
    
    def get_failure_state(self, failure_type: Optional[str] = None) -> Optional[np.ndarray]:
        """
        Sample a failure state from the bank.
        
        Args:
            failure_type: Filter by failure type (None = random)
        
        Returns:
            State vector or None if bank is empty
        """
        states = self.failure_bank.sample_failure_state(
            n_samples=1,
            failure_type=failure_type,
        )
        return states[0] if states else None
    
    def get_statistics(self) -> Dict[str, any]:
        """Get curriculum injection statistics."""
        injection_rate_actual = (
            self.injected_count / max(1, self.episode_count - self.warmup_episodes)
        )
        
        return {
            'total_episodes': self.episode_count,
            'injected_episodes': self.injected_count,
            'target_injection_rate': self.injection_rate,
            'actual_injection_rate': injection_rate_actual,
            'warmup_episodes': self.warmup_episodes,
            'failure_bank_size': len(self.failure_bank.failures),
        }
    
    def reset_episode_count(self):
        """Reset episode counting (for new training run)."""
        self.episode_count = 0
        self.injected_count = 0


# TimescaleDB Schema SQL (for reference)
FAILURE_BANK_SCHEMA = """
CREATE TABLE IF NOT EXISTS failure_bank (
    timestamp TIMESTAMPTZ NOT NULL,
    symbol VARCHAR(20) NOT NULL,
    state_vector JSONB NOT NULL,
    dd_breach_size DOUBLE PRECISION NOT NULL,
    agent_weights JSONB NOT NULL,
    episode_id INTEGER NOT NULL,
    step INTEGER NOT NULL,
    balance DOUBLE PRECISION NOT NULL,
    pnl DOUBLE PRECISION NOT NULL,
    failure_type VARCHAR(50) NOT NULL
);

SELECT create_hypertable('failure_bank', 'timestamp', if_not_exists => TRUE);

CREATE INDEX IF NOT EXISTS idx_failure_type ON failure_bank (failure_type);
CREATE INDEX IF NOT EXISTS idx_symbol ON failure_bank (symbol);
CREATE INDEX IF NOT EXISTS idx_dd_breach ON failure_bank (dd_breach_size DESC);
"""

SKILL_BANK_SCHEMA = """
CREATE TABLE IF NOT EXISTS skill_bank (
    timestamp TIMESTAMPTZ NOT NULL,
    agent_id VARCHAR(50) NOT NULL,
    symbol VARCHAR(20) NOT NULL,
    calmar_ratio DOUBLE PRECISION NOT NULL,
    novelty_score DOUBLE PRECISION NOT NULL,
    avg_pnl DOUBLE PRECISION NOT NULL,
    max_dd DOUBLE PRECISION NOT NULL,
    sharpe_ratio DOUBLE PRECISION NOT NULL,
    episode_id INTEGER NOT NULL,
    steps_trained INTEGER NOT NULL,
    model_weights JSONB NOT NULL,
    skill_name TEXT NOT NULL
);

SELECT create_hypertable('skill_bank', 'timestamp', if_not_exists => TRUE);

CREATE INDEX IF NOT EXISTS idx_calmar ON skill_bank (calmar_ratio DESC);
CREATE INDEX IF NOT EXISTS idx_agent_id ON skill_bank (agent_id);
CREATE INDEX IF NOT EXISTS idx_novelty ON skill_bank (novelty_score DESC);
"""
