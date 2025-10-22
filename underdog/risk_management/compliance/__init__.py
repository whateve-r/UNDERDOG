"""
Compliance Module - Prop Firm Constraints

Implements Constrained RL (CMDP) for guaranteed Prop Firm compliance.

Based on paper:
- arXiv:2510.04952v2: Safe and Compliant Trade Execution (Shield Module)

Components:
- compliance_shield.py: PropFirmSafetyShield (copied from execution/)
- prop_firm_constraints.py: FTMO/FTUK rule definitions
- cmdp_monitor.py: Constrained Markov Decision Process monitoring

Enforces:
- Daily Drawdown < 5% (FTMO Phase 1)
- Total Drawdown < 10%
- Max positions limit
- Risk per trade limit

Integration: Pre-execution validation in Mt5Executor
"""

from underdog.risk_management.compliance.compliance_shield import PropFirmSafetyShield, SafetyConstraints

__all__ = [
    'PropFirmSafetyShield',
    'SafetyConstraints',
    'PropFirmConstraints',
    'CMDPMonitor',
]

