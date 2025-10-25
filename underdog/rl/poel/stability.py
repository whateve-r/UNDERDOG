"""
Stability Penalty - Local Drawdown Management

Implements local risk tracking and penalties for individual agents.
Forces agents to learn risk management before Meta-Agent intervention.
"""

from typing import Dict, Optional
from collections import deque
import numpy as np


class LocalDrawdownTracker:
    """Tracks drawdown metrics for individual agent"""
    
    def __init__(
        self,
        initial_balance: float,
        window_size: int = 100,
    ):
        """
        Args:
            initial_balance: Starting balance for agent
            window_size: Number of steps to track for rolling metrics
        """
        self.initial_balance = initial_balance
        self.current_balance = initial_balance
        self.peak_balance = initial_balance
        self.window_size = window_size
        
        # Rolling PnL history
        self.pnl_history = deque(maxlen=window_size)
        
        # Drawdown tracking
        self.current_dd = 0.0
        self.max_dd = 0.0
        self.daily_dd = 0.0
        self.daily_peak = initial_balance
        
    def update(self, new_balance: float, is_new_day: bool = False) -> Dict[str, float]:
        """
        Update balance and compute drawdown metrics.
        
        Args:
            new_balance: Updated balance after step
            is_new_day: Whether this marks start of new trading day
            
        Returns:
            Dictionary with drawdown metrics
        """
        pnl = new_balance - self.current_balance
        self.pnl_history.append(pnl)
        self.current_balance = new_balance
        
        # Update peak tracking
        if new_balance > self.peak_balance:
            self.peak_balance = new_balance
            self.current_dd = 0.0
        else:
            self.current_dd = (self.peak_balance - new_balance) / self.peak_balance
            
        # Update max drawdown
        if self.current_dd > self.max_dd:
            self.max_dd = self.current_dd
            
        # Daily drawdown tracking
        if is_new_day:
            self.daily_peak = new_balance
            self.daily_dd = 0.0
        else:
            if new_balance > self.daily_peak:
                self.daily_peak = new_balance
                self.daily_dd = 0.0
            else:
                self.daily_dd = (self.daily_peak - new_balance) / self.daily_peak
                
        return {
            'balance': self.current_balance,
            'current_dd': self.current_dd,
            'max_dd': self.max_dd,
            'daily_dd': self.daily_dd,
            'pnl': pnl,
        }
        
    def get_volatility(self) -> float:
        """Calculate PnL volatility (std dev of returns)"""
        if len(self.pnl_history) < 2:
            return 0.0
        return float(np.std(self.pnl_history))
        
    def get_sharpe_estimate(self, risk_free_rate: float = 0.0) -> float:
        """Estimate Sharpe ratio from recent history"""
        if len(self.pnl_history) < 2:
            return 0.0
            
        returns = np.array(self.pnl_history) / self.initial_balance
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        
        if std_return < 1e-8:
            return 0.0
            
        return float((mean_return - risk_free_rate) / std_return)
        
    def reset(self, new_initial_balance: Optional[float] = None):
        """Reset tracker for new episode"""
        if new_initial_balance is not None:
            self.initial_balance = new_initial_balance
            
        self.current_balance = self.initial_balance
        self.peak_balance = self.initial_balance
        self.daily_peak = self.initial_balance
        self.current_dd = 0.0
        self.max_dd = 0.0
        self.daily_dd = 0.0
        self.pnl_history.clear()


class StabilityPenalty:
    """
    Calculates stability penalty for local agents.
    
    Penalty = β * (LocalDD / MaxPermittedDD)
    
    This teaches agents to manage risk individually before Meta-Agent
    needs to intervene with capital allocation.
    """
    
    def __init__(
        self,
        max_local_dd_pct: float = 0.15,  # 15% local DD limit
        beta: float = 1.0,  # Penalty weight
        volatility_penalty: bool = True,
        volatility_threshold: float = 0.02,  # 2% volatility threshold
    ):
        """
        Args:
            max_local_dd_pct: Maximum permitted local drawdown
            beta: Weight for stability penalty
            volatility_penalty: Whether to penalize high volatility
            volatility_threshold: Volatility threshold for penalty
        """
        self.max_local_dd_pct = max_local_dd_pct
        self.beta = beta
        self.volatility_penalty = volatility_penalty
        self.volatility_threshold = volatility_threshold
        
    def compute_penalty(
        self,
        tracker: LocalDrawdownTracker,
        current_dd: float,
    ) -> Dict[str, float]:
        """
        Compute stability penalty for current state.
        
        Args:
            tracker: LocalDrawdownTracker instance
            current_dd: Current drawdown percentage
            
        Returns:
            Dictionary with penalty components
        """
        # Base DD penalty (scaled by max permitted)
        dd_penalty = self.beta * (current_dd / self.max_local_dd_pct)
        
        # Volatility penalty (if enabled)
        vol_penalty = 0.0
        if self.volatility_penalty:
            volatility = tracker.get_volatility()
            if volatility > self.volatility_threshold:
                vol_penalty = 0.5 * (volatility / self.volatility_threshold - 1.0)
                
        # Total stability penalty
        total_penalty = dd_penalty + vol_penalty
        
        # Critical penalty if exceeding local DD limit
        critical = current_dd > self.max_local_dd_pct
        if critical:
            total_penalty += 10.0  # Severe penalty
            
        return {
            'stability_penalty': total_penalty,
            'dd_penalty': dd_penalty,
            'volatility_penalty': vol_penalty,
            'critical_dd': critical,
            'current_dd': current_dd,
            'volatility': tracker.get_volatility(),
        }
        
    def get_risk_factor(self, current_dd: float) -> float:
        """
        Get risk factor in [0, 1] for position sizing.
        
        Returns:
            0.0 = at DD limit (no more risk)
            1.0 = no drawdown (full risk)
        """
        risk_used = current_dd / self.max_local_dd_pct
        return max(0.0, 1.0 - risk_used)
