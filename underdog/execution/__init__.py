"""
Execution module - Live trading execution components

Contains:
- MT5Executor: MetaTrader 5 order execution with PropFirm risk management
- Bridge modules: Signal translation from backtesting to live execution
"""

from .mt5_executor import MT5Executor, OrderType, OrderStatus, OrderResult, Position

__all__ = [
    'MT5Executor',
    'OrderType',
    'OrderStatus',
    'OrderResult',
    'Position',
]
