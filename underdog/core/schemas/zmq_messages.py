"""
ZeroMQ Message Schemas for MT5 <-> Python Communication
Includes JSON Schema validators for message validation.
"""
from dataclasses import dataclass, field, asdict
from typing import Dict, Any, Optional, List, Literal, Union
from datetime import datetime
from enum import Enum
import json


# ========================================
# Enums for Type Safety
# ========================================

class MessageType(str, Enum):
    """Message types for ZeroMQ communication"""
    TICK = "tick"
    OHLCV = "ohlcv"
    ORDER = "order"
    ACK = "ack"
    ACCOUNT_UPDATE = "account_update"
    POSITION_CLOSE = "position_close"
    POSITION_UPDATE = "position_update"
    HEARTBEAT = "heartbeat"
    ERROR = "error"


class OrderAction(str, Enum):
    """Order action types"""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    CLOSE = "close"
    MODIFY = "modify"
    CANCEL = "cancel"


class OrderSide(str, Enum):
    """Order side (direction)"""
    BUY = "buy"
    SELL = "sell"


class OrderStatus(str, Enum):
    """Order execution status"""
    FILLED = "filled"
    PARTIAL = "partial"
    REJECTED = "rejected"
    QUEUED = "queued"
    CANCELLED = "cancelled"
    PENDING = "pending"


# ========================================
# Market Data Messages (MT5 -> Python)
# ========================================

@dataclass
class TickMessage:
    """Real-time tick data message"""
    type: str = "tick"
    symbol: str = ""
    ts: str = ""  # ISO 8601 timestamp
    bid: float = 0.0
    ask: float = 0.0
    last: Optional[float] = None
    volume: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TickMessage':
        return cls(**{k: v for k, v in data.items() if k in cls.__annotations__})


@dataclass
class OHLCVMessage:
    """OHLCV bar data message"""
    type: str = "ohlcv"
    symbol: str = ""
    granularity: str = ""  # M1, M5, M15, H1, H4, D1, etc.
    ts: str = ""  # ISO 8601 timestamp (bar opening time)
    open: float = 0.0
    high: float = 0.0
    low: float = 0.0
    close: float = 0.0
    volume: float = 0.0
    tick_volume: Optional[int] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'OHLCVMessage':
        return cls(**{k: v for k, v in data.items() if k in cls.__annotations__})


# ========================================
# Trading Messages (Python -> MT5)
# ========================================

@dataclass
class OrderRequest:
    """Order request message from Python to MT5"""
    type: str = "order"
    id: str = ""  # UUID v4 for idempotency
    action: str = "market"  # market, limit, stop, close, modify
    symbol: str = ""
    side: str = "buy"  # buy, sell
    size: float = 0.0  # lot size
    price: Optional[float] = None  # for limit/stop orders
    sl: Optional[float] = None  # stop loss
    tp: Optional[float] = None  # take profit
    slippage: Optional[float] = None
    magic: Optional[int] = None  # magic number
    comment: Optional[str] = None
    meta: Optional[Dict[str, Any]] = field(default_factory=dict)  # strategy metadata
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    def to_json(self) -> str:
        return json.dumps(self.to_dict())
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'OrderRequest':
        return cls(**{k: v for k, v in data.items() if k in cls.__annotations__})


@dataclass
class OrderModifyRequest:
    """Modify existing order/position"""
    type: str = "modify"
    id: str = ""  # request UUID
    order_id: Optional[int] = None  # MT5 order ticket
    position_id: Optional[int] = None  # MT5 position ticket
    sl: Optional[float] = None
    tp: Optional[float] = None
    price: Optional[float] = None  # for pending orders
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    def to_json(self) -> str:
        return json.dumps(self.to_dict())


@dataclass
class ClosePositionRequest:
    """Close position request"""
    type: str = "close"
    id: str = ""
    symbol: str = ""
    position_id: Optional[int] = None
    size: Optional[float] = None  # partial close
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    def to_json(self) -> str:
        return json.dumps(self.to_dict())


# ========================================
# Execution & Account Messages (MT5 -> Python)
# ========================================

@dataclass
class ExecutionACK:
    """Execution acknowledgment from MT5"""
    type: str = "ack"
    order_id: str = ""  # original request UUID
    status: str = "filled"  # filled, rejected, partial, queued
    mt5_order: Optional[int] = None  # MT5 order ticket
    mt5_deal: Optional[int] = None  # MT5 deal ticket
    filled_price: Optional[float] = None
    filled_size: Optional[float] = None
    remaining_size: Optional[float] = None
    ts: str = ""  # execution timestamp
    reason: Optional[str] = None  # rejection/error reason
    retcode: Optional[int] = None  # MT5 return code
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ExecutionACK':
        return cls(**{k: v for k, v in data.items() if k in cls.__annotations__})


@dataclass
class AccountUpdateMessage:
    """Account state update message"""
    type: str = "account_update"
    ts: str = ""
    balance: float = 0.0
    equity: float = 0.0
    margin: float = 0.0
    free_margin: float = 0.0
    margin_level: Optional[float] = None
    leverage: int = 1
    profit: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AccountUpdateMessage':
        return cls(**{k: v for k, v in data.items() if k in cls.__annotations__})


@dataclass
class PositionCloseMessage:
    """Position close notification"""
    type: str = "position_close"
    position_id: int = 0
    symbol: str = ""
    side: str = ""
    size: float = 0.0
    open_price: float = 0.0
    close_price: float = 0.0
    profit: float = 0.0
    commission: float = 0.0
    swap: float = 0.0
    open_ts: str = ""
    close_ts: str = ""
    comment: Optional[str] = None
    magic: Optional[int] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PositionCloseMessage':
        return cls(**{k: v for k, v in data.items() if k in cls.__annotations__})


@dataclass
class PositionUpdateMessage:
    """Open position update"""
    type: str = "position_update"
    position_id: int = 0
    symbol: str = ""
    side: str = ""
    size: float = 0.0
    open_price: float = 0.0
    current_price: float = 0.0
    sl: Optional[float] = None
    tp: Optional[float] = None
    profit: float = 0.0
    swap: float = 0.0
    open_ts: str = ""
    magic: Optional[int] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PositionUpdateMessage':
        return cls(**{k: v for k, v in data.items() if k in cls.__annotations__})


# ========================================
# System Messages
# ========================================

@dataclass
class HeartbeatMessage:
    """Heartbeat/ping message"""
    type: str = "heartbeat"
    ts: str = ""
    source: str = ""  # "mt5" or "python"
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'HeartbeatMessage':
        return cls(**{k: v for k, v in data.items() if k in cls.__annotations__})


@dataclass
class ErrorMessage:
    """Error message"""
    type: str = "error"
    ts: str = ""
    code: Optional[int] = None
    message: str = ""
    context: Optional[Dict[str, Any]] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ErrorMessage':
        return cls(**{k: v for k, v in data.items() if k in cls.__annotations__})


# ========================================
# JSON Schema Validators
# ========================================

TICK_SCHEMA = {
    "type": "object",
    "required": ["type", "symbol", "ts", "bid", "ask"],
    "properties": {
        "type": {"type": "string", "const": "tick"},
        "symbol": {"type": "string", "minLength": 1},
        "ts": {"type": "string", "format": "date-time"},
        "bid": {"type": "number"},
        "ask": {"type": "number"},
        "last": {"type": ["number", "null"]},
        "volume": {"type": ["number", "null"]}
    }
}

OHLCV_SCHEMA = {
    "type": "object",
    "required": ["type", "symbol", "granularity", "ts", "open", "high", "low", "close", "volume"],
    "properties": {
        "type": {"type": "string", "const": "ohlcv"},
        "symbol": {"type": "string", "minLength": 1},
        "granularity": {"type": "string", "pattern": "^(M[0-9]+|H[0-9]+|D1|W1|MN1)$"},
        "ts": {"type": "string", "format": "date-time"},
        "open": {"type": "number"},
        "high": {"type": "number"},
        "low": {"type": "number"},
        "close": {"type": "number"},
        "volume": {"type": "number", "minimum": 0}
    }
}

ORDER_REQUEST_SCHEMA = {
    "type": "object",
    "required": ["type", "id", "action", "symbol", "side", "size"],
    "properties": {
        "type": {"type": "string", "const": "order"},
        "id": {"type": "string", "format": "uuid"},
        "action": {"type": "string", "enum": ["market", "limit", "stop", "close", "modify"]},
        "symbol": {"type": "string", "minLength": 1},
        "side": {"type": "string", "enum": ["buy", "sell"]},
        "size": {"type": "number", "exclusiveMinimum": 0},
        "price": {"type": ["number", "null"]},
        "sl": {"type": ["number", "null"]},
        "tp": {"type": ["number", "null"]},
        "slippage": {"type": ["number", "null"]},
        "magic": {"type": ["integer", "null"]},
        "comment": {"type": ["string", "null"]},
        "meta": {"type": "object"}
    }
}

EXECUTION_ACK_SCHEMA = {
    "type": "object",
    "required": ["type", "order_id", "status", "ts"],
    "properties": {
        "type": {"type": "string", "const": "ack"},
        "order_id": {"type": "string", "format": "uuid"},
        "status": {"type": "string", "enum": ["filled", "rejected", "partial", "queued", "cancelled", "pending"]},
        "mt5_order": {"type": ["integer", "null"]},
        "mt5_deal": {"type": ["integer", "null"]},
        "filled_price": {"type": ["number", "null"]},
        "filled_size": {"type": ["number", "null"]},
        "ts": {"type": "string", "format": "date-time"},
        "reason": {"type": ["string", "null"]},
        "retcode": {"type": ["integer", "null"]}
    }
}


# ========================================
# Schema Validation Helper
# ========================================

def validate_message(message: Dict[str, Any], schema: Dict[str, Any]) -> tuple[bool, Optional[str]]:
    """
    Validate message against JSON schema.
    
    Args:
        message: Message dict to validate
        schema: JSON schema dict
    
    Returns:
        Tuple of (is_valid, error_message)
    """
    try:
        import jsonschema
        jsonschema.validate(instance=message, schema=schema)
        return True, None
    except ImportError:
        # jsonschema not installed, skip validation
        return True, "jsonschema not installed, skipping validation"
    except jsonschema.ValidationError as e:
        return False, str(e)
    except Exception as e:
        return False, f"Validation error: {str(e)}"


# ========================================
# Message Factory & Parser
# ========================================

class MessageFactory:
    """Factory for creating and parsing ZeroMQ messages"""
    
    @staticmethod
    def parse(data: Union[str, bytes, Dict[str, Any]]) -> Optional[Any]:
        """
        Parse raw message data into appropriate message object.
        
        Args:
            data: Raw message (JSON string, bytes, or dict)
        
        Returns:
            Parsed message object or None if parsing fails
        """
        try:
            if isinstance(data, bytes):
                data = data.decode('utf-8')
            if isinstance(data, str):
                data = json.loads(data)
            
            msg_type = data.get('type')
            
            if msg_type == 'tick':
                return TickMessage.from_dict(data)
            elif msg_type == 'ohlcv':
                return OHLCVMessage.from_dict(data)
            elif msg_type == 'ack':
                return ExecutionACK.from_dict(data)
            elif msg_type == 'account_update':
                return AccountUpdateMessage.from_dict(data)
            elif msg_type == 'position_close':
                return PositionCloseMessage.from_dict(data)
            elif msg_type == 'position_update':
                return PositionUpdateMessage.from_dict(data)
            elif msg_type == 'heartbeat':
                return HeartbeatMessage.from_dict(data)
            elif msg_type == 'error':
                return ErrorMessage.from_dict(data)
            else:
                return data  # Return raw dict for unknown types
                
        except Exception as e:
            print(f"Error parsing message: {e}")
            return None
    
    @staticmethod
    def create_order(symbol: str, side: str, size: float, 
                    action: str = "market",
                    price: Optional[float] = None,
                    sl: Optional[float] = None,
                    tp: Optional[float] = None,
                    strategy: Optional[str] = None,
                    confidence: Optional[float] = None) -> OrderRequest:
        """
        Create an order request message.
        
        Args:
            symbol: Trading symbol
            side: "buy" or "sell"
            size: Position size in lots
            action: Order type (market, limit, stop)
            price: Limit/stop price
            sl: Stop loss price
            tp: Take profit price
            strategy: Strategy identifier
            confidence: Confidence score [0-1]
        
        Returns:
            OrderRequest object
        """
        import uuid
        
        meta = {}
        if strategy:
            meta['strategy'] = strategy
        if confidence is not None:
            meta['confidence'] = confidence
        
        return OrderRequest(
            type="order",
            id=str(uuid.uuid4()),
            action=action,
            symbol=symbol,
            side=side,
            size=size,
            price=price,
            sl=sl,
            tp=tp,
            meta=meta
        )
