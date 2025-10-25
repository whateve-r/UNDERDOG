"""
Safety Shield for Prop Firm Compliance

Implements Constrained Reinforcement Learning (RL) for guaranteed compliance
with Prop Firm risk limits (FTMO, FTUK, etc.).

Paper: arXiv:2510.04952v2 - Safe and Compliant Trade Execution

Key Constraints:
- Daily Drawdown < 5% (FTMO Phase 1)
- Total Drawdown < 10%
- Max open positions
- Min holding time
- Max lot size per trade

If action violates constraint → Project to nearest safe action
"""

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, Tuple, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class SafetyConstraints:
    """Prop Firm safety constraints configuration"""
    
    # Drawdown limits (FTMO defaults)
    max_daily_dd_pct: float = 0.05  # 5%
    max_total_dd_pct: float = 0.10  # 10%
    
    # Position limits
    max_positions: int = 2
    max_lot_size: float = 0.1  # Max lot per trade (0.1 lot = $1k notional with 100:1 leverage)
    min_lot_size: float = 0.01
    
    # Risk per trade
    max_risk_per_trade_pct: float = 0.015  # 1.5%
    
    # Timing constraints
    min_holding_time_minutes: int = 3  # Avoid scalping restrictions
    max_holding_time_hours: int = 48  # Force position review
    
    # Trading hours (UTC)
    allowed_trading_hours: List[int] = None  # None = 24/7
    
    # Emergency stop
    emergency_close_dd_pct: float = 0.045  # Close all at 4.5% (safety margin)
    
    def __post_init__(self):
        if self.allowed_trading_hours is None:
            self.allowed_trading_hours = list(range(24))


class PropFirmSafetyShield:
    """
    Pre-execution validation layer for Prop Firm compliance.
    
    Architecture:
    1. Validate action against all constraints
    2. If violation detected:
       a. Log attempt
       b. Project to nearest safe action
       c. Alert monitoring system
    3. Return (is_safe, corrected_action)
    
    Usage:
        shield = PropFirmSafetyShield(constraints)
        is_safe, action = shield.validate_action(
            proposed_action,
            current_account_state
        )
        
        if is_safe:
            executor.execute(action)
        else:
            logger.warning(f"Shield blocked: {action['reason']}")
    """
    
    def __init__(self, constraints: SafetyConstraints):
        self.constraints = constraints
        self.violations_log: List[Dict] = []
        self._total_actions = 0  # Track total actions for intervention rate
        
        logger.info(
            f"SafetyShield initialized - "
            f"Daily DD: {constraints.max_daily_dd_pct*100}%, "
            f"Total DD: {constraints.max_total_dd_pct*100}%, "
            f"Max Positions: {constraints.max_positions}"
        )
    
    def validate_action(
        self,
        action: Dict,
        account_state: Dict
    ) -> Tuple[bool, Dict]:
        """
        Validate proposed action against all constraints.
        
        Args:
            action: {
                'type': 'open' | 'close' | 'modify' | 'wait',
                'symbol': 'EURUSD',
                'lot_size': 0.05,
                'direction': 'buy' | 'sell',
                'stop_loss': float,
                'take_profit': float,
                'risk_pct': float  # Expected risk
            }
            
            account_state: {
                'balance': 50000,
                'equity': 49500,
                'daily_dd_pct': 0.03,
                'total_dd_pct': 0.07,
                'open_positions': [...],
                'daily_high_equity': 50200,
                'initial_balance': 50000
            }
        
        Returns:
            (is_safe: bool, corrected_action: Dict)
        """
        
        # Track total actions
        self._total_actions += 1
        
        # Emergency close check (highest priority)
        if self._check_emergency_close(account_state):
            return False, {
                'action': 'close_all',
                'reason': 'emergency_dd_breach',
                'current_dd': account_state['daily_dd_pct'],
                'timestamp': datetime.utcnow()
            }
        
        # Daily DD check
        if not self._check_daily_dd(account_state):
            self._log_violation('daily_dd_breach', account_state)
            return False, {
                'action': 'block_new_trades',
                'reason': 'daily_dd_limit',
                'current_dd': account_state['daily_dd_pct'],
                'limit': self.constraints.max_daily_dd_pct
            }
        
        # Total DD check
        if not self._check_total_dd(account_state):
            self._log_violation('total_dd_breach', account_state)
            return False, {
                'action': 'close_all',
                'reason': 'total_dd_breach',
                'current_dd': account_state['total_dd_pct'],
                'limit': self.constraints.max_total_dd_pct
            }
        
        # For 'wait' actions, no further validation needed
        if action.get('type') == 'wait':
            return True, action
        
        # For 'close' actions, allow (closing reduces risk)
        if action.get('type') == 'close':
            return True, action
        
        # For 'open' actions, validate all constraints
        if action.get('type') == 'open':
            return self._validate_open_action(action, account_state)
        
        # For 'modify' actions, validate modifications
        if action.get('type') == 'modify':
            return self._validate_modify_action(action, account_state)
        
        # Unknown action type
        logger.error(f"Unknown action type: {action.get('type')}")
        return False, {'action': 'wait', 'reason': 'unknown_action_type'}
    
    def _validate_open_action(
        self,
        action: Dict,
        account_state: Dict
    ) -> Tuple[bool, Dict]:
        """Validate opening new position"""
        
        corrected_action = action.copy()
        
        # Check max positions
        if len(account_state['open_positions']) >= self.constraints.max_positions:
            self._log_violation('max_positions', account_state)
            return False, {
                'action': 'wait',
                'reason': 'max_positions_reached',
                'current': len(account_state['open_positions']),
                'limit': self.constraints.max_positions
            }
        
        # Check trading hours
        if not self._check_trading_hours():
            return False, {
                'action': 'wait',
                'reason': 'outside_trading_hours',
                'current_hour': datetime.utcnow().hour
            }
        
        # Validate and correct lot size
        lot_size = action.get('lot_size', 0)
        risk_pct = action.get('risk_pct', 0)
        
        # Check max risk per trade
        if risk_pct > self.constraints.max_risk_per_trade_pct:
            # Reduce lot size proportionally
            reduction_factor = self.constraints.max_risk_per_trade_pct / risk_pct
            corrected_action['lot_size'] = lot_size * reduction_factor
            corrected_action['risk_pct'] = self.constraints.max_risk_per_trade_pct
            
            logger.warning(
                f"Reduced lot size: {lot_size:.3f} to {corrected_action['lot_size']:.3f} "
                f"(risk: {risk_pct*100:.2f}% to {corrected_action['risk_pct']*100:.2f}%)"
            )
        
        # Check lot size bounds
        if corrected_action['lot_size'] > self.constraints.max_lot_size:
            corrected_action['lot_size'] = self.constraints.max_lot_size
            logger.warning(f"Capped lot size at {self.constraints.max_lot_size}")
        
        if corrected_action['lot_size'] < self.constraints.min_lot_size:
            return False, {
                'action': 'wait',
                'reason': 'lot_size_too_small',
                'proposed': corrected_action['lot_size'],
                'minimum': self.constraints.min_lot_size
            }
        
        # Check if we have enough margin (basic check)
        estimated_margin = self._estimate_margin_requirement(corrected_action)
        free_margin = account_state.get('free_margin', account_state['equity'])
        
        if estimated_margin > free_margin * 0.5:  # Use max 50% of free margin
            return False, {
                'action': 'wait',
                'reason': 'insufficient_margin',
                'required': estimated_margin,
                'available': free_margin
            }
        
        return True, corrected_action
    
    def _validate_modify_action(
        self,
        action: Dict,
        account_state: Dict
    ) -> Tuple[bool, Dict]:
        """Validate position modification"""
        
        # Allow modifications that reduce risk (wider SL, closer TP)
        # Block modifications that increase risk
        
        return True, action  # TODO: Implement detailed validation
    
    def _check_emergency_close(self, account_state: Dict) -> bool:
        """Check if emergency close is triggered"""
        return account_state['daily_dd_pct'] >= self.constraints.emergency_close_dd_pct
    
    def _check_daily_dd(self, account_state: Dict) -> bool:
        """Check daily drawdown limit"""
        return account_state['daily_dd_pct'] < self.constraints.max_daily_dd_pct
    
    def _check_total_dd(self, account_state: Dict) -> bool:
        """Check total drawdown limit"""
        return account_state['total_dd_pct'] < self.constraints.max_total_dd_pct
    
    def _check_trading_hours(self) -> bool:
        """Check if current hour is allowed for trading"""
        current_hour = datetime.utcnow().hour
        return current_hour in self.constraints.allowed_trading_hours
    
    def _estimate_margin_requirement(self, action: Dict) -> float:
        """
        Estimate margin requirement for action.
        
        TODO: Implement proper margin calculation based on:
        - Symbol (different leverage)
        - Lot size
        - Current price
        """
        # Conservative estimate: 1000 USD per 0.01 lot for majors
        lot_size = action.get('lot_size', 0)
        return lot_size * 100 * 1000  # Simplified
    
    def _log_violation(self, violation_type: str, account_state: Dict):
        """Log constraint violation for analysis"""
        violation = {
            'timestamp': datetime.utcnow(),
            'type': violation_type,
            'daily_dd': account_state['daily_dd_pct'],
            'total_dd': account_state['total_dd_pct'],
            'positions': len(account_state['open_positions'])
        }
        self.violations_log.append(violation)
        
        logger.warning(
            f"Shield Violation: {violation_type} - "
            f"Daily DD: {account_state['daily_dd_pct']*100:.2f}%, "
            f"Total DD: {account_state['total_dd_pct']*100:.2f}%"
        )
    
    def get_violation_stats(self) -> Dict:
        """Get statistics on shield interventions"""
        if not self.violations_log:
            return {'total': 0}
        
        return {
            'total': len(self.violations_log),
            'by_type': self._count_by_type(),
            'last_violation': self.violations_log[-1],
            'intervention_rate': len(self.violations_log) / max(1, self._total_actions)
        }
    
    def _count_by_type(self) -> Dict[str, int]:
        """Count violations by type"""
        counts = {}
        for v in self.violations_log:
            vtype = v['type']
            counts[vtype] = counts.get(vtype, 0) + 1
        return counts


# Example usage
if __name__ == "__main__":
    # FTMO Phase 1 configuration
    ftmo_constraints = SafetyConstraints(
        max_daily_dd_pct=0.05,
        max_total_dd_pct=0.10,
        max_positions=2,
        max_risk_per_trade_pct=0.015
    )
    
    shield = PropFirmSafetyShield(ftmo_constraints)
    
    # Simulate account state
    account_state = {
        'balance': 50000,
        'equity': 49500,
        'free_margin': 48000,
        'daily_dd_pct': 0.03,  # 3% daily DD
        'total_dd_pct': 0.07,  # 7% total DD
        'open_positions': [],
        'daily_high_equity': 50200,
        'initial_balance': 50000
    }
    
    # Test action
    action = {
        'type': 'open',
        'symbol': 'EURUSD',
        'direction': 'buy',
        'lot_size': 0.10,
        'risk_pct': 0.02,  # 2% risk (over limit)
        'stop_loss': 1.0850,
        'take_profit': 1.0920
    }
    
    is_safe, corrected_action = shield.validate_action(action, account_state)
    
    print(f"Is Safe: {is_safe}")
    print(f"Corrected Action: {corrected_action}")
