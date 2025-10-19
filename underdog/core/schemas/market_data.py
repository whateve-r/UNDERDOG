from dataclasses import dataclass

@dataclass
class Tick:
    symbol: str
    bid: float
    ask: float
    timestamp: float

@dataclass
class OHLCV:
    symbol: str
    timestamp: float
    open: float
    high: float
    low: float
    close: float
    volume: float
