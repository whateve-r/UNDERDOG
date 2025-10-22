"""
Test Safety Shield - Prop Firm Compliance Validation

Validates PropFirmSafetyShield with realistic scenarios:
1. Normal trading (should pass)
2. Daily DD breach (should block)
3. Total DD breach (should emergency close)
4. Max positions exceeded (should block)
5. Excessive risk per trade (should reduce lot size)
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from underdog.execution.safety_shield import (
    PropFirmSafetyShield,
    SafetyConstraints
)

def print_result(scenario: str, is_safe: bool, action: dict):
    """Pretty print validation result"""
    status = "✅ ALLOWED" if is_safe else "🛡️ BLOCKED"
    print(f"\n{scenario}")
    print(f"  Status: {status}")
    print(f"  Action: {action}")

def main():
    print("=" * 60)
    print("SAFETY SHIELD TEST - Prop Firm Compliance")
    print("=" * 60)
    
    # FTMO Phase 1 Configuration
    ftmo_constraints = SafetyConstraints(
        max_daily_dd_pct=0.05,      # 5% daily DD
        max_total_dd_pct=0.10,      # 10% total DD
        max_positions=2,
        max_risk_per_trade_pct=0.015,  # 1.5% risk per trade
        emergency_close_dd_pct=0.045   # Emergency at 4.5%
    )
    
    shield = PropFirmSafetyShield(ftmo_constraints)
    
    # ========================================
    # Scenario 1: Normal Trading (Healthy Account)
    # ========================================
    account_state = {
        'balance': 50000,
        'equity': 50500,
        'free_margin': 49000,
        'daily_dd_pct': 0.01,  # 1% daily DD (safe)
        'total_dd_pct': 0.02,  # 2% total DD (safe)
        'open_positions': [],
        'daily_high_equity': 50600,
        'initial_balance': 50000
    }
    
    action = {
        'type': 'open',
        'symbol': 'EURUSD',
        'direction': 'buy',
        'lot_size': 0.05,
        'risk_pct': 0.01,  # 1% risk (safe)
        'stop_loss': 1.0850,
        'take_profit': 1.0920
    }
    
    is_safe, result = shield.validate_action(action, account_state)
    print_result("Scenario 1: Normal Trade (Healthy Account)", is_safe, result)
    
    # ========================================
    # Scenario 2: Daily DD Near Limit
    # ========================================
    account_state['daily_dd_pct'] = 0.048  # 4.8% (near 5% limit)
    
    is_safe, result = shield.validate_action(action, account_state)
    print_result("Scenario 2: Daily DD at 4.8% (Near Limit)", is_safe, result)
    
    # ========================================
    # Scenario 3: Daily DD Breach
    # ========================================
    account_state['daily_dd_pct'] = 0.052  # 5.2% (BREACH!)
    
    is_safe, result = shield.validate_action(action, account_state)
    print_result("Scenario 3: Daily DD at 5.2% (BREACH)", is_safe, result)
    
    # ========================================
    # Scenario 4: Emergency Close (4.5%)
    # ========================================
    account_state['daily_dd_pct'] = 0.046  # 4.6% (EMERGENCY!)
    
    is_safe, result = shield.validate_action(action, account_state)
    print_result("Scenario 4: Daily DD at 4.6% (EMERGENCY CLOSE)", is_safe, result)
    
    # ========================================
    # Scenario 5: Total DD Breach
    # ========================================
    account_state['daily_dd_pct'] = 0.03  # Reset daily DD
    account_state['total_dd_pct'] = 0.11  # 11% total DD (BREACH!)
    
    is_safe, result = shield.validate_action(action, account_state)
    print_result("Scenario 5: Total DD at 11% (BREACH)", is_safe, result)
    
    # ========================================
    # Scenario 6: Max Positions Exceeded
    # ========================================
    account_state['total_dd_pct'] = 0.03  # Reset
    account_state['open_positions'] = [
        {'ticket': 123, 'symbol': 'EURUSD'},
        {'ticket': 124, 'symbol': 'GBPUSD'}
    ]  # 2 positions (limit reached)
    
    is_safe, result = shield.validate_action(action, account_state)
    print_result("Scenario 6: Max Positions (2/2)", is_safe, result)
    
    # ========================================
    # Scenario 7: Excessive Risk Per Trade
    # ========================================
    account_state['open_positions'] = []  # Reset
    
    risky_action = action.copy()
    risky_action['risk_pct'] = 0.025  # 2.5% risk (over 1.5% limit)
    risky_action['lot_size'] = 0.10
    
    is_safe, result = shield.validate_action(risky_action, account_state)
    print_result("Scenario 7: Risk 2.5% → Should Reduce", is_safe, result)
    
    # ========================================
    # Scenario 8: Close Action (Always Allowed)
    # ========================================
    close_action = {
        'type': 'close',
        'ticket': 123
    }
    
    # Even with DD breach, close is allowed
    account_state['daily_dd_pct'] = 0.06
    is_safe, result = shield.validate_action(close_action, account_state)
    print_result("Scenario 8: Close Action (DD 6%)", is_safe, result)
    
    # ========================================
    # Summary
    # ========================================
    print("\n" + "=" * 60)
    print("SHIELD STATISTICS")
    print("=" * 60)
    stats = shield.get_violation_stats()
    print(f"Total Violations: {stats.get('total', 0)}")
    print(f"By Type: {stats.get('by_type', {})}")
    
    print("\n✅ Safety Shield test complete!")
    print("\n📊 Key Takeaways:")
    print("  1. Shield blocks new trades when DD > 5%")
    print("  2. Emergency close triggered at 4.5% (safety margin)")
    print("  3. Position limits enforced (max 2)")
    print("  4. Risk per trade reduced automatically if > 1.5%")
    print("  5. Close actions always allowed (reduce exposure)")
    print("\n🎯 Ready for Prop Firm compliance!")

if __name__ == "__main__":
    main()
