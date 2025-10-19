from dataclasses import dataclass

@dataclass
class AccountInfo:
    balance: float
    equity: float
    margin_free: float
    margin_level: float
