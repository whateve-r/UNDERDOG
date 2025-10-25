"""
Dynamic Capital Allocation for Large-Scale MARL Coordination.

Implements Calmar Ratio-based weighting to allocate capital across heterogeneous agents.
Emergency protocol triggers at 80% of total DD limit.
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import numpy as np
from collections import deque
import logging

logger = logging.getLogger(__name__)


@dataclass
class AgentPerformance:
    """Performance metrics for a single agent."""
    agent_id: str
    symbol: str
    recent_pnl: float  # Sum of PnL over window
    max_dd: float  # Max drawdown over window
    steps: int  # Number of steps in window
    calmar_ratio: float = 0.0
    weight: float = 0.0


class CalmarCalculator:
    """
    Calculates Calmar Ratio for agent performance evaluation.
    
    Calmar Ratio = CAGR / Max Drawdown
    For short windows: Approximate CAGR as (Total Return / Days) * 252
    """
    
    def __init__(self, window_size: int = 500, min_steps: int = 50):
        """
        Args:
            window_size: Number of recent steps to consider for metrics
            min_steps: Minimum steps required for reliable Calmar calculation
        """
        self.window_size = window_size
        self.min_steps = min_steps
    
    def calculate_calmar_ratio(
        self,
        pnl_history: List[float],
        balance_history: List[float],
        initial_balance: float,
    ) -> float:
        """
        Calculate Calmar Ratio from agent history.
        
        Args:
            pnl_history: Recent PnL values (can be negative)
            balance_history: Recent balance values
            initial_balance: Starting balance for this window
        
        Returns:
            Calmar Ratio (higher is better, negative if DD > gains)
        """
        if len(pnl_history) < self.min_steps:
            return 0.0  # Insufficient data
        
        # Use most recent window
        pnl_window = pnl_history[-self.window_size:]
        balance_window = balance_history[-self.window_size:]
        
        # Calculate CAGR approximation
        total_return = sum(pnl_window)
        days_approx = len(pnl_window) / (24 * 12)  # Assume 5-min bars → ~12 bars/hour
        
        if days_approx < 0.1:  # Less than 2.4 hours
            days_approx = 0.1
        
        # Annualized return rate
        annualized_return = (total_return / initial_balance) * (252 / days_approx)
        
        # Calculate Max Drawdown
        max_dd = self._calculate_max_dd(balance_window)
        
        if max_dd < 1e-6:  # No drawdown (perfect performance)
            return annualized_return * 100  # Very high Calmar
        
        calmar = annualized_return / max_dd
        return calmar
    
    def _calculate_max_dd(self, balance_history: List[float]) -> float:
        """
        Calculate maximum drawdown percentage.
        
        Returns:
            Max DD as decimal (e.g., 0.15 for 15% DD)
        """
        if not balance_history:
            return 0.0
        
        peak = balance_history[0]
        max_dd = 0.0
        
        for balance in balance_history:
            if balance > peak:
                peak = balance
            
            dd = (peak - balance) / peak if peak > 0 else 0.0
            max_dd = max(max_dd, dd)
        
        return max_dd


class CapitalAllocator:
    """
    Allocates capital across agents based on Calmar Ratio.
    
    Key Features:
    - ReLU(Calmar) ensures only profitable agents get capital
    - Epsilon prevents division by zero
    - Normalization ensures weights sum to 1.0
    - Emergency protocol at 80% DD threshold
    """
    
    def __init__(
        self,
        emergency_threshold: float = 0.80,  # 80% of max DD limit
        max_total_dd: float = 0.10,  # 10% total DD limit
        epsilon: float = 1e-6,  # Prevent zero division
        min_allocation: float = 0.05,  # Minimum 5% per agent (diversification)
    ):
        """
        Args:
            emergency_threshold: Fraction of max_total_dd to trigger emergency
            max_total_dd: Maximum permitted total DD (as decimal)
            epsilon: Small value added to prevent zero weights
            min_allocation: Minimum weight per agent (if active)
        """
        self.emergency_threshold = emergency_threshold
        self.max_total_dd = max_total_dd
        self.epsilon = epsilon
        self.min_allocation = min_allocation
        self.calmar_calculator = CalmarCalculator()
        
        # State tracking
        self.emergency_mode = False
        self.best_agent_id: Optional[str] = None
    
    def allocate_weights(
        self,
        agent_performances: List[AgentPerformance],
        current_total_dd: float,
    ) -> Dict[str, float]:
        """
        Allocate capital weights across agents.
        
        Args:
            agent_performances: List of AgentPerformance objects with Calmar ratios
            current_total_dd: Current global DD percentage (decimal)
        
        Returns:
            Dict mapping agent_id -> weight (summing to 1.0)
        """
        # 🚨 CMDP CRITICAL: Check for emergency protocol FIRST
        emergency_dd_threshold = self.emergency_threshold * self.max_total_dd
        
        if current_total_dd >= emergency_dd_threshold:
            # ACTIVATE EMERGENCY MODE IMMEDIATELY
            self.emergency_mode = True
            logger.critical(
                f"[EMERGENCY] PROTOCOL ACTIVATED: Total DD {current_total_dd:.2%} "
                f">= {emergency_dd_threshold:.2%} ({self.emergency_threshold:.0%} of {self.max_total_dd:.2%} limit)"
            )
            return self._emergency_allocation(agent_performances)
        
        # If we were in emergency mode but DD recovered, reset
        if self.emergency_mode and current_total_dd < emergency_dd_threshold * 0.90:  # 10% hysteresis
            logger.info(f"[OK] Emergency mode DEACTIVATED - DD recovered to {current_total_dd:.2%}")
            self.emergency_mode = False
        
        # 🛑 CMDP HARD CONSTRAINT: Block agents with high historical DD
        historical_dd_threshold = 0.08  # 8% MaxDD limit
        blocked_agents = []
        active_performances = []
        
        for perf in agent_performances:
            if perf.max_dd > historical_dd_threshold:
                blocked_agents.append(perf.agent_id)
                logger.warning(
                    f"[BLOCKED] {perf.agent_id}: Historical MaxDD {perf.max_dd:.2%} "
                    f"> {historical_dd_threshold:.2%} limit"
                )
            else:
                active_performances.append(perf)
        
        # If all agents blocked, use emergency allocation
        if not active_performances:
            logger.error(
                "❌ ALL AGENTS BLOCKED - Using emergency allocation"
            )
            return self._emergency_allocation(agent_performances)
        
        # Log blocking activity
        if blocked_agents:
            logger.info(
                f"[CAPITAL] Capital Allocator: {len(blocked_agents)}/{len(agent_performances)} "
                f"agents blocked for high historical risk"
            )
        
        # Normal allocation: ReLU(Calmar) + epsilon (only for non-blocked agents)
        raw_weights = []
        agent_ids = []
        
        for perf in active_performances:
            agent_ids.append(perf.agent_id)
            # ReLU: only positive Calmar ratios contribute
            relu_calmar = max(0.0, perf.calmar_ratio)
            raw_weights.append(relu_calmar + self.epsilon)
        
        # Normalize to sum = 1.0
        total_weight = sum(raw_weights)
        normalized_weights = [w / total_weight for w in raw_weights]
        
        # Apply minimum allocation constraint (diversification)
        adjusted_weights = self._apply_min_allocation(normalized_weights, agent_ids)
        
        # Create dict with active agents
        allocation = {aid: w for aid, w in zip(agent_ids, adjusted_weights)}
        
        # Explicitly set blocked agents to zero weight
        for blocked_id in blocked_agents:
            allocation[blocked_id] = 0.0
        
        # Track best agent for emergency protocol
        best_idx = np.argmax([p.calmar_ratio for p in agent_performances])
        self.best_agent_id = agent_performances[best_idx].agent_id
        
        return allocation
    
    def _emergency_allocation(
        self,
        agent_performances: List[AgentPerformance],
    ) -> Dict[str, float]:
        """
        Emergency protocol: Allocate 100% to best Calmar Ratio agent.
        
        Signal: CLOSE_ALL positions, move to single best performer.
        
        Note: emergency_mode should be set BEFORE calling this method.
        """
        # Find agent with highest positive Calmar Ratio
        positive_agents = [p for p in agent_performances if p.calmar_ratio > 0]
        
        if not positive_agents:
            # No profitable agents: distribute equally (holding mode)
            n = len(agent_performances)
            logger.warning("[EMERGENCY] No profitable agents - equal distribution")
            return {p.agent_id: 1.0 / n for p in agent_performances}
        
        # Best agent gets 100%
        best_agent = max(positive_agents, key=lambda p: p.calmar_ratio)
        self.best_agent_id = best_agent.agent_id
        
        logger.critical(f"[EMERGENCY] 100% allocation to {best_agent.agent_id} (Calmar: {best_agent.calmar_ratio:.2f})")
        
        allocation = {p.agent_id: 0.0 for p in agent_performances}
        allocation[best_agent.agent_id] = 1.0
        
        return allocation
    
    def _apply_min_allocation(
        self,
        weights: List[float],
        agent_ids: List[str],
    ) -> List[float]:
        """
        Ensure each agent gets at least min_allocation (diversification).
        
        Redistribute excess from high-weight agents.
        """
        n_agents = len(weights)
        adjusted = weights.copy()
        
        # Check if all weights are already above minimum
        if all(w >= self.min_allocation for w in weights):
            return adjusted
        
        # Boost low weights to minimum
        deficit = 0.0
        for i in range(n_agents):
            if adjusted[i] < self.min_allocation:
                deficit += (self.min_allocation - adjusted[i])
                adjusted[i] = self.min_allocation
        
        # Redistribute deficit from high-weight agents
        high_weight_agents = [i for i in range(n_agents) if adjusted[i] > self.min_allocation]
        
        if not high_weight_agents:
            # All agents at minimum: normalize
            total = sum(adjusted)
            return [w / total for w in adjusted]
        
        # Proportionally reduce high weights
        excess = sum(adjusted[i] - self.min_allocation for i in high_weight_agents)
        if excess > 0:
            for i in high_weight_agents:
                reduction_ratio = deficit / excess
                adjusted[i] -= (adjusted[i] - self.min_allocation) * reduction_ratio
        
        # Final normalization (ensure sum = 1.0)
        total = sum(adjusted)
        return [w / total for w in adjusted]
    
    def check_emergency_mode(self, current_total_dd: float) -> bool:
        """
        Check if emergency protocol should be activated.
        
        Args:
            current_total_dd: Current global DD percentage (decimal)
        
        Returns:
            True if emergency mode should activate
        """
        threshold = self.emergency_threshold * self.max_total_dd
        return current_total_dd >= threshold
    
    def get_emergency_signal(self) -> Dict[str, any]:
        """
        Get emergency protocol signal for Meta-Agent.
        
        Returns:
            Dict with emergency status and instructions
        """
        return {
            'emergency_mode': self.emergency_mode,
            'action': 'CLOSE_ALL' if self.emergency_mode else 'NORMAL',
            'best_agent_id': self.best_agent_id,
            'message': (
                f"Emergency Protocol Activated: Allocating 100% to {self.best_agent_id}"
                if self.emergency_mode
                else "Normal allocation mode"
            ),
        }
    
    def reset(self):
        """Reset emergency state."""
        self.emergency_mode = False
        self.best_agent_id = None


def create_agent_performance(
    agent_id: str,
    symbol: str,
    pnl_history: List[float],
    balance_history: List[float],
    initial_balance: float,
) -> AgentPerformance:
    """
    Factory function to create AgentPerformance with calculated Calmar Ratio.
    
    Args:
        agent_id: Unique agent identifier
        symbol: Trading symbol (e.g., 'EURUSD')
        pnl_history: Recent PnL values
        balance_history: Recent balance values
        initial_balance: Starting balance for window
    
    Returns:
        AgentPerformance object with Calmar Ratio calculated
    """
    calculator = CalmarCalculator()
    
    calmar = calculator.calculate_calmar_ratio(
        pnl_history=pnl_history,
        balance_history=balance_history,
        initial_balance=initial_balance,
    )
    
    max_dd = calculator._calculate_max_dd(balance_history)
    recent_pnl = sum(pnl_history[-500:]) if pnl_history else 0.0
    
    return AgentPerformance(
        agent_id=agent_id,
        symbol=symbol,
        recent_pnl=recent_pnl,
        max_dd=max_dd,
        steps=len(pnl_history),
        calmar_ratio=calmar,
    )
