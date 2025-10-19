from dataclasses import dataclass

@dataclass
class Order:
    symbol: str
    volume: float
    price: float
    sl: float
    tp: float
    comment: str
    order_id: int
