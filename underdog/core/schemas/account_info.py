from dataclasses import dataclass
from typing import Optional

@dataclass
class AccountInfo:
    """
    Account information from MT5 broker.
    
    Attributes:
        balance: Account balance in currency
        equity: Account equity (balance + floating P&L)
        margin: Used margin
        margin_free: Free margin available
        margin_level: Margin level percentage (equity/margin * 100)
        leverage: Account leverage (e.g., 100, 500)
        name: Account holder name
        server: MT5 server name
        broker: Broker name
        currency: Account currency (e.g., "USD")
        trading_allowed: Whether trading is allowed
        bot_trading: Whether Expert Advisors are allowed
    """
    balance: float
    equity: float
    margin: float = 0.0
    margin_free: float = 0.0
    margin_level: float = 0.0
    leverage: int = 1
    name: str = ""
    server: str = ""
    broker: str = ""
    currency: str = "USD"
    trading_allowed: bool = True
    bot_trading: bool = True
