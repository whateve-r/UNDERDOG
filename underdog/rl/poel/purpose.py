"""
Purpose Function - Business Objective Definition

Implements the global business purpose that guides exploration:
Purpose = PnL - λ1*DailyDD - λ2*TotalDD

This ensures agents learn to maximize returns while respecting hard risk constraints.
"""

from dataclasses import dataclass
from typing import Dict, Optional
import numpy as np


@dataclass
class BusinessPurpose:
    """Configuration for business purpose optimization"""
    
    # Risk penalties (INCREASED from 10/20 to 25/50 - extreme risk aversion)
    lambda_daily_dd: float = 25.0  # Penalty for daily drawdown (was 10.0)
    lambda_total_dd: float = 50.0  # Penalty for total drawdown (was 20.0)
    
    # Hard limits from funding requirements
    max_daily_dd_pct: float = 0.049  # 4.9% daily DD limit
    max_total_dd_pct: float = 0.09  # 9% total DD limit
    
    # Emergency risk mode threshold
    emergency_threshold_pct: float = 0.80  # Activate at 80% of total DD limit


class PurposeFunction:
    """
    Calculates the global purpose metric that Meta-Agent optimizes.
    
    This is the core objective function that translates funding requirements
    into a differentiable optimization target.
    """
    
    def __init__(self, config: BusinessPurpose):
        self.config = config
        self._initial_balance: Optional[float] = None
        self._peak_balance: Optional[float] = None
        
    def reset(self, initial_balance: float):
        """Reset tracking for new episode"""
        self._initial_balance = initial_balance
        self._peak_balance = initial_balance
        
    def compute_purpose(
        self,
        current_balance: float,
        daily_dd_pct: float,
        total_dd_pct: float,
        pnl: float
    ) -> Dict[str, float]:
        """
        Compute purpose score and components.
        
        Args:
            current_balance: Current account balance
            daily_dd_pct: Daily drawdown percentage (0.0 - 1.0)
            total_dd_pct: Total drawdown percentage (0.0 - 1.0)
            pnl: Profit/Loss for current step
            
        Returns:
            Dictionary with:
                - purpose: Overall purpose score
                - pnl: Raw PnL
                - daily_dd_penalty: Daily DD penalty
                - total_dd_penalty: Total DD penalty
                - emergency_mode: Whether emergency risk mode triggered
        """
        # Update peak tracking
        if self._peak_balance is None or current_balance > self._peak_balance:
            self._peak_balance = current_balance
            
        # Calculate penalties
        daily_dd_penalty = self.config.lambda_daily_dd * daily_dd_pct
        total_dd_penalty = self.config.lambda_total_dd * total_dd_pct
        
        # Base purpose: reward PnL, penalize DD
        purpose = pnl - daily_dd_penalty - total_dd_penalty
        
        # Emergency mode detection
        emergency_mode = total_dd_pct > (
            self.config.max_total_dd_pct * self.config.emergency_threshold_pct
        )
        
        # Catastrophic penalty if hard limits breached
        if daily_dd_pct > self.config.max_daily_dd_pct:
            purpose -= 1000.0  # CMDP constraint violation
            
        if total_dd_pct > self.config.max_total_dd_pct:
            purpose -= 2000.0  # Critical violation
            
        return {
            'purpose': purpose,
            'pnl': pnl,
            'daily_dd_penalty': daily_dd_penalty,
            'total_dd_penalty': total_dd_penalty,
            'daily_dd_pct': daily_dd_pct,
            'total_dd_pct': total_dd_pct,
            'emergency_mode': emergency_mode,
        }
        
    def get_risk_budget(self, total_dd_pct: float) -> float:
        """
        Calculate remaining risk budget as percentage.
        
        Returns value in [0, 1] where:
        - 1.0 = full risk budget available
        - 0.0 = at risk limit
        """
        remaining = self.config.max_total_dd_pct - total_dd_pct
        budget = remaining / self.config.max_total_dd_pct
        return np.clip(budget, 0.0, 1.0)
        
    def should_enter_emergency_mode(self, total_dd_pct: float) -> bool:
        """Check if emergency risk mode should activate"""
        threshold = self.config.max_total_dd_pct * self.config.emergency_threshold_pct
        return total_dd_pct >= threshold
