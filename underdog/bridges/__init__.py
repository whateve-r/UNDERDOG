"""
Bridges module - Signal translation between systems

Contains:
- bt_to_mt5: Backtrader to MT5 bridge for live trading
"""

from .bt_to_mt5 import BacktraderMT5Bridge, LiveStrategy

__all__ = [
    'BacktraderMT5Bridge',
    'LiveStrategy',
]
