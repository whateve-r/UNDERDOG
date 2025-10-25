"""
POEL Reward Shaper

Integrates Purpose, Novelty, and Stability into enriched local rewards:
R' = α*PnL + (1-α)*NoveltyBonus - β*StabilityPenalty
"""

from typing import Dict, Tuple
import numpy as np
from .novelty import NoveltyDetector, DistanceMetric
from .stability import StabilityPenalty, LocalDrawdownTracker


class POELRewardShaper:
    """
    Shapes local agent rewards with novelty bonuses and stability penalties.
    
    This implements the POEL reward formula that balances:
    - Exploitation (PnL maximization)
    - Exploration (Novelty seeking)
    - Risk Control (Stability management)
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        alpha: float = 0.85,  # PnL weight (INCREASED from 0.7 - more exploitation)
        beta: float = 3.0,  # Stability penalty weight (INCREASED from 1.0 - risk aversion)
        novelty_scale: float = 100.0,  # Scale novelty to match PnL magnitude
        novelty_metric: DistanceMetric = DistanceMetric.L2,
        max_local_dd_pct: float = 0.15,
        initial_balance: float = 25000.0,
    ):
        """
        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            alpha: Weight for PnL vs Novelty (higher = more PnL focus)
            beta: Weight for stability penalty
            novelty_scale: Scaling factor for novelty bonus
            novelty_metric: Distance metric for novelty calculation
            max_local_dd_pct: Maximum local drawdown percentage
            initial_balance: Initial balance for DD tracking
        """
        self.alpha = alpha
        self.beta = beta
        self.novelty_scale = novelty_scale
        
        # Novelty detector
        self.novelty_detector = NoveltyDetector(
            state_dim=state_dim,
            action_dim=action_dim,
            metric=novelty_metric,
            normalization=True,
        )
        
        # Stability tracker and penalty
        self.dd_tracker = LocalDrawdownTracker(
            initial_balance=initial_balance,
            window_size=100,
        )
        
        self.stability_penalty = StabilityPenalty(
            max_local_dd_pct=max_local_dd_pct,
            beta=beta,
            volatility_penalty=True,
        )
        
        # Tracking
        self.total_rewards = 0.0
        self.total_novelty_bonus = 0.0
        self.total_stability_penalty = 0.0
        
    def compute_reward(
        self,
        state: np.ndarray,
        action: np.ndarray,
        raw_pnl: float,
        new_balance: float,
        is_new_day: bool = False,
    ) -> Tuple[float, Dict[str, float]]:
        """
        Compute enriched reward with novelty and stability.
        
        Args:
            state: Current state
            action: Action taken
            raw_pnl: Raw PnL from environment
            new_balance: Balance after action
            is_new_day: Whether new trading day started
            
        Returns:
            Tuple of (enriched_reward, info_dict)
        """
        # Flatten state/action if needed (handle sequential states)
        if state.ndim > 1:
            state_flat = state.flatten()
        else:
            state_flat = state
            
        if action.ndim > 1:
            action_flat = action.flatten()
        else:
            action_flat = action
            
        # 1. Compute Novelty Bonus
        novelty_score = self.novelty_detector.compute_novelty(
            state_flat,
            action_flat,
            k_neighbors=5
        )
        novelty_bonus = novelty_score * self.novelty_scale
        
        # Add experience to novelty buffer
        self.novelty_detector.add_experience(state_flat, action_flat)
        
        # 2. Compute Stability Penalty
        dd_metrics = self.dd_tracker.update(new_balance, is_new_day)
        penalty_info = self.stability_penalty.compute_penalty(
            self.dd_tracker,
            dd_metrics['current_dd']
        )
        stability_penalty = penalty_info['stability_penalty']
        
        # 🛑 CMDP HARD CONSTRAINT: Drastic Penalty Filter
        # If local DD exceeds 80% of limit (9.6% of 12%), override entire reward
        local_dd_limit = 0.12  # 12% local DD limit per agent
        cmdp_threshold = 0.80 * local_dd_limit  # 9.6% threshold
        current_local_dd = dd_metrics['current_dd']
        
        cmdp_violation = current_local_dd >= cmdp_threshold
        
        if cmdp_violation:
            # DRASTIC PENALTY: Override all rewards, ignore PnL and Novelty
            # This creates extreme risk aversion - agent learns this is worst outcome
            enriched_reward = -1000.0
            cmdp_active = True
        else:
            # 3. Normal POEL formula: R' = α*PnL + (1-α)*Novelty - β*Stability
            enriched_reward = (
                self.alpha * raw_pnl +
                (1 - self.alpha) * novelty_bonus -
                self.beta * stability_penalty
            )
            cmdp_active = False
        
        # Track cumulative components
        self.total_rewards += enriched_reward
        self.total_novelty_bonus += novelty_bonus
        self.total_stability_penalty += stability_penalty
        
        # Build info dictionary
        info = {
            'raw_pnl': raw_pnl,
            'novelty_bonus': novelty_bonus,
            'novelty_score': novelty_score,
            'stability_penalty': stability_penalty,
            'enriched_reward': enriched_reward,
            'alpha': self.alpha,
            'beta': self.beta,
            # CMDP hard constraint tracking
            'cmdp_violation': cmdp_violation,
            'cmdp_active': cmdp_active,
            'cmdp_threshold': cmdp_threshold,
            'cmdp_penalty': -1000.0 if cmdp_active else 0.0,
            **dd_metrics,
            **penalty_info,
        }
        
        return enriched_reward, info
        
    def get_statistics(self) -> Dict[str, float]:
        """Get cumulative statistics"""
        return {
            'total_rewards': self.total_rewards,
            'total_novelty_bonus': self.total_novelty_bonus,
            'total_stability_penalty': self.total_stability_penalty,
            'avg_novelty': (
                self.total_novelty_bonus / max(1, len(self.dd_tracker.pnl_history))
            ),
            'current_dd': self.dd_tracker.current_dd,
            'max_dd': self.dd_tracker.max_dd,
            'sharpe_estimate': self.dd_tracker.get_sharpe_estimate(),
        }
        
    def reset(self, new_initial_balance: float = None):
        """Reset for new episode"""
        self.novelty_detector.reset()
        self.dd_tracker.reset(new_initial_balance)
        self.total_rewards = 0.0
        self.total_novelty_bonus = 0.0
        self.total_stability_penalty = 0.0
        
    def adjust_exploration(self, alpha: float):
        """
        Dynamically adjust exploration vs exploitation.
        
        Args:
            alpha: New PnL weight (higher = less exploration)
        """
        self.alpha = np.clip(alpha, 0.0, 1.0)
        
    def get_risk_factor(self) -> float:
        """Get current risk factor for position sizing"""
        return self.stability_penalty.get_risk_factor(
            self.dd_tracker.current_dd
        )
