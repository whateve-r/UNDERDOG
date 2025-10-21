"""
MT5 Executor - Live Order Execution with MetaTrader 5

This module handles real order execution in MetaTrader 5 with:
- Connection management with auto-reconnect
- Pre-execution drawdown validation (PropFirm compliance)
- Position tracking and monitoring
- Emergency stop functionality
- Comprehensive logging for audit trail

Critical for: Paper Trading → FTMO Challenge → Live Trading

Author: Underdog Trading System
Business Goal: Pass Prop Firm challenges (€2,000-4,000/month revenue target)
"""

import MetaTrader5 as mt5
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from enum import Enum
import logging
from dataclasses import dataclass
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class OrderType(Enum):
    """Order types supported"""
    BUY = mt5.ORDER_TYPE_BUY
    SELL = mt5.ORDER_TYPE_SELL


class OrderStatus(Enum):
    """Order execution status"""
    SUCCESS = "success"
    REJECTED_DD = "rejected_dd_limit"
    REJECTED_MT5 = "rejected_mt5_error"
    REJECTED_CONNECTION = "rejected_connection_lost"


@dataclass
class OrderResult:
    """Result of order execution"""
    status: OrderStatus
    ticket: Optional[int] = None
    price: Optional[float] = None
    volume: Optional[float] = None
    error_code: Optional[int] = None
    error_message: Optional[str] = None
    dd_at_execution: Optional[float] = None
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


@dataclass
class Position:
    """Open position representation"""
    ticket: int
    symbol: str
    type: str  # 'buy' or 'sell'
    volume: float
    open_price: float
    current_price: float
    profit: float
    open_time: datetime
    sl: float
    tp: float
    
    @property
    def duration_hours(self) -> float:
        """Position duration in hours"""
        return (datetime.now() - self.open_time).total_seconds() / 3600


class MT5Executor:
    """
    MetaTrader 5 Order Executor with PropFirm Risk Management
    
    Critical Features:
    - Pre-execution DD validation (MUST be <5% daily, <10% total)
    - Auto-reconnect on connection loss
    - Emergency close all positions
    - Comprehensive audit logging
    
    Usage:
        executor = MT5Executor(
            account=12345678,
            password="password",
            server="ICMarkets-Demo",
            max_daily_dd=5.0,
            max_total_dd=10.0
        )
        
        if executor.initialize():
            result = executor.execute_order(
                symbol="EURUSD",
                order_type=OrderType.BUY,
                volume=0.1,
                sl_pips=20,
                tp_pips=40
            )
    """
    
    def __init__(
        self,
        account: int,
        password: str,
        server: str,
        max_daily_dd: float = 5.0,  # PropFirm standard: 5%
        max_total_dd: float = 10.0,  # PropFirm standard: 10%
        starting_balance: Optional[float] = None,
        reconnect_attempts: int = 5,
        reconnect_delay: int = 10
    ):
        """
        Initialize MT5 Executor
        
        Args:
            account: MT5 account number
            password: MT5 account password
            server: MT5 server name (e.g., "ICMarkets-Demo")
            max_daily_dd: Max daily drawdown % (default 5.0 for PropFirms)
            max_total_dd: Max total drawdown % (default 10.0 for PropFirms)
            starting_balance: Initial balance (if None, fetched from MT5)
            reconnect_attempts: Number of reconnection attempts
            reconnect_delay: Seconds between reconnect attempts
        """
        self.account = account
        self.password = password
        self.server = server
        self.max_daily_dd = max_daily_dd
        self.max_total_dd = max_total_dd
        self.starting_balance = starting_balance
        self.reconnect_attempts = reconnect_attempts
        self.reconnect_delay = reconnect_delay
        
        self.is_connected = False
        self.daily_high_balance = None
        self.session_start_time = datetime.now()
        
        logger.info(f"MT5Executor initialized - Account: {account}, Server: {server}")
        logger.info(f"Risk Limits: Daily DD {max_daily_dd}%, Total DD {max_total_dd}%")
    
    def initialize(self) -> bool:
        """
        Initialize MT5 connection and login
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Initialize MT5
            if not mt5.initialize():
                logger.error(f"MT5 initialize() failed, error code: {mt5.last_error()}")
                return False
            
            # Login to account
            if not mt5.login(self.account, password=self.password, server=self.server):
                error = mt5.last_error()
                logger.error(f"MT5 login failed - Account: {self.account}, Error: {error}")
                mt5.shutdown()
                return False
            
            # Get account info
            account_info = mt5.account_info()
            if account_info is None:
                logger.error("Failed to get account info")
                mt5.shutdown()
                return False
            
            # Set starting balance if not provided
            if self.starting_balance is None:
                self.starting_balance = account_info.balance
            
            # Initialize daily high balance
            self.daily_high_balance = account_info.balance
            
            self.is_connected = True
            logger.info(f"✅ MT5 connected successfully")
            logger.info(f"Account Balance: ${account_info.balance:.2f}")
            logger.info(f"Account Equity: ${account_info.equity:.2f}")
            logger.info(f"Starting Balance: ${self.starting_balance:.2f}")
            
            return True
            
        except Exception as e:
            logger.error(f"Exception during MT5 initialization: {e}")
            return False
    
    def _reconnect(self) -> bool:
        """
        Attempt to reconnect to MT5
        
        Returns:
            bool: True if reconnected, False otherwise
        """
        logger.warning("Connection lost, attempting reconnect...")
        
        for attempt in range(1, self.reconnect_attempts + 1):
            logger.info(f"Reconnect attempt {attempt}/{self.reconnect_attempts}")
            
            mt5.shutdown()
            time.sleep(self.reconnect_delay)
            
            if self.initialize():
                logger.info(f"✅ Reconnected successfully on attempt {attempt}")
                return True
        
        logger.error(f"❌ Failed to reconnect after {self.reconnect_attempts} attempts")
        return False
    
    def calculate_drawdown(self) -> Tuple[float, float]:
        """
        Calculate current daily and total drawdown
        
        Returns:
            Tuple[float, float]: (daily_dd_pct, total_dd_pct)
        """
        if not self.is_connected:
            logger.error("Cannot calculate DD - not connected to MT5")
            return 0.0, 0.0
        
        account_info = mt5.account_info()
        if account_info is None:
            logger.error("Failed to get account info for DD calculation")
            if not self._reconnect():
                return 0.0, 0.0
            account_info = mt5.account_info()
        
        current_equity = account_info.equity
        
        # Update daily high balance
        if current_equity > self.daily_high_balance:
            self.daily_high_balance = current_equity
        
        # Daily DD: from today's high
        daily_dd_pct = ((self.daily_high_balance - current_equity) / self.daily_high_balance) * 100
        
        # Total DD: from starting balance
        total_dd_pct = ((self.starting_balance - current_equity) / self.starting_balance) * 100
        
        return daily_dd_pct, total_dd_pct
    
    def _validate_drawdown_limits(self) -> Tuple[bool, str]:
        """
        Validate if current DD is within PropFirm limits
        
        Returns:
            Tuple[bool, str]: (is_valid, reason_if_invalid)
        """
        daily_dd, total_dd = self.calculate_drawdown()
        
        if daily_dd >= self.max_daily_dd:
            reason = f"Daily DD {daily_dd:.2f}% >= limit {self.max_daily_dd}%"
            logger.error(f"❌ DD LIMIT BREACH: {reason}")
            return False, reason
        
        if total_dd >= self.max_total_dd:
            reason = f"Total DD {total_dd:.2f}% >= limit {self.max_total_dd}%"
            logger.error(f"❌ DD LIMIT BREACH: {reason}")
            return False, reason
        
        logger.info(f"✅ DD Check Passed - Daily: {daily_dd:.2f}%, Total: {total_dd:.2f}%")
        return True, ""
    
    def execute_order(
        self,
        symbol: str,
        order_type: OrderType,
        volume: float,
        sl_pips: Optional[float] = None,
        tp_pips: Optional[float] = None,
        comment: str = "Underdog_Bot"
    ) -> OrderResult:
        """
        Execute market order with pre-execution DD validation
        
        Args:
            symbol: Trading symbol (e.g., "EURUSD")
            order_type: OrderType.BUY or OrderType.SELL
            volume: Lot size (e.g., 0.1)
            sl_pips: Stop Loss in pips (optional)
            tp_pips: Take Profit in pips (optional)
            comment: Order comment for tracking
        
        Returns:
            OrderResult: Execution result with status and details
        """
        logger.info(f"📤 ORDER REQUEST: {symbol} {order_type.name} {volume} lots")
        
        # Step 1: Check connection
        if not self.is_connected:
            logger.error("Not connected to MT5")
            if not self._reconnect():
                return OrderResult(
                    status=OrderStatus.REJECTED_CONNECTION,
                    error_message="MT5 connection lost and reconnect failed"
                )
        
        # Step 2: Pre-execution DD validation (CRITICAL for PropFirms)
        is_valid, reason = self._validate_drawdown_limits()
        if not is_valid:
            daily_dd, total_dd = self.calculate_drawdown()
            return OrderResult(
                status=OrderStatus.REJECTED_DD,
                error_message=reason,
                dd_at_execution=max(daily_dd, total_dd)
            )
        
        # Step 3: Get symbol info
        symbol_info = mt5.symbol_info(symbol)
        if symbol_info is None:
            logger.error(f"Symbol {symbol} not found")
            return OrderResult(
                status=OrderStatus.REJECTED_MT5,
                error_message=f"Symbol {symbol} not found"
            )
        
        if not symbol_info.visible:
            if not mt5.symbol_select(symbol, True):
                logger.error(f"Failed to select symbol {symbol}")
                return OrderResult(
                    status=OrderStatus.REJECTED_MT5,
                    error_message=f"Failed to select symbol {symbol}"
                )
        
        # Step 4: Calculate price and SL/TP
        point = symbol_info.point
        
        if order_type == OrderType.BUY:
            price = mt5.symbol_info_tick(symbol).ask
            sl = price - sl_pips * 10 * point if sl_pips else 0
            tp = price + tp_pips * 10 * point if tp_pips else 0
        else:  # SELL
            price = mt5.symbol_info_tick(symbol).bid
            sl = price + sl_pips * 10 * point if sl_pips else 0
            tp = price - tp_pips * 10 * point if tp_pips else 0
        
        # Step 5: Prepare order request
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": volume,
            "type": order_type.value,
            "price": price,
            "sl": sl,
            "tp": tp,
            "deviation": 20,
            "magic": 234000,
            "comment": comment,
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        
        # Step 6: Send order
        logger.info(f"Sending order: {request}")
        result = mt5.order_send(request)
        
        if result is None:
            error = mt5.last_error()
            logger.error(f"order_send failed, error: {error}")
            return OrderResult(
                status=OrderStatus.REJECTED_MT5,
                error_code=error[0] if error else None,
                error_message=f"MT5 error: {error}"
            )
        
        # Step 7: Check result
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            logger.error(f"Order rejected - retcode: {result.retcode}, comment: {result.comment}")
            return OrderResult(
                status=OrderStatus.REJECTED_MT5,
                error_code=result.retcode,
                error_message=result.comment
            )
        
        # SUCCESS
        daily_dd, total_dd = self.calculate_drawdown()
        logger.info(f"✅ ORDER EXECUTED - Ticket: {result.order}, Price: {result.price}, Volume: {result.volume}")
        
        return OrderResult(
            status=OrderStatus.SUCCESS,
            ticket=result.order,
            price=result.price,
            volume=result.volume,
            dd_at_execution=max(daily_dd, total_dd)
        )
    
    def get_open_positions(self) -> List[Position]:
        """
        Get all open positions
        
        Returns:
            List[Position]: List of open positions
        """
        if not self.is_connected:
            logger.error("Not connected to MT5")
            return []
        
        positions = mt5.positions_get()
        if positions is None or len(positions) == 0:
            return []
        
        position_list = []
        for pos in positions:
            position_list.append(Position(
                ticket=pos.ticket,
                symbol=pos.symbol,
                type='buy' if pos.type == mt5.ORDER_TYPE_BUY else 'sell',
                volume=pos.volume,
                open_price=pos.price_open,
                current_price=pos.price_current,
                profit=pos.profit,
                open_time=datetime.fromtimestamp(pos.time),
                sl=pos.sl,
                tp=pos.tp
            ))
        
        return position_list
    
    def close_position(self, ticket: int, reason: str = "Manual close") -> bool:
        """
        Close specific position
        
        Args:
            ticket: Position ticket number
            reason: Reason for closing (for logging)
        
        Returns:
            bool: True if closed successfully
        """
        positions = mt5.positions_get(ticket=ticket)
        if positions is None or len(positions) == 0:
            logger.error(f"Position {ticket} not found")
            return False
        
        position = positions[0]
        
        # Determine close type
        close_type = mt5.ORDER_TYPE_SELL if position.type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY
        
        # Get current price
        tick = mt5.symbol_info_tick(position.symbol)
        price = tick.bid if close_type == mt5.ORDER_TYPE_SELL else tick.ask
        
        # Close request
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": position.symbol,
            "volume": position.volume,
            "type": close_type,
            "position": ticket,
            "price": price,
            "deviation": 20,
            "magic": 234000,
            "comment": f"Close: {reason}",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        
        result = mt5.order_send(request)
        
        if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
            error = mt5.last_error() if result is None else result.comment
            logger.error(f"Failed to close position {ticket}: {error}")
            return False
        
        logger.info(f"✅ Position {ticket} closed - Reason: {reason}")
        return True
    
    def emergency_close_all(self, reason: str = "Emergency stop") -> int:
        """
        Emergency close ALL open positions
        
        Critical for DD breach situations
        
        Args:
            reason: Reason for emergency close
        
        Returns:
            int: Number of positions closed
        """
        logger.warning(f"🚨 EMERGENCY CLOSE ALL TRIGGERED - Reason: {reason}")
        
        positions = self.get_open_positions()
        closed_count = 0
        
        for pos in positions:
            if self.close_position(pos.ticket, reason=reason):
                closed_count += 1
        
        logger.warning(f"Emergency close completed - Closed {closed_count}/{len(positions)} positions")
        return closed_count
    
    def reset_daily_tracking(self):
        """
        Reset daily tracking (call at start of new trading day)
        """
        account_info = mt5.account_info()
        if account_info:
            self.daily_high_balance = account_info.balance
            self.session_start_time = datetime.now()
            logger.info(f"Daily tracking reset - New high balance: ${self.daily_high_balance:.2f}")
    
    def shutdown(self):
        """
        Shutdown MT5 connection cleanly
        """
        if self.is_connected:
            mt5.shutdown()
            self.is_connected = False
            logger.info("MT5 connection shutdown")
    
    def __enter__(self):
        """Context manager entry"""
        self.initialize()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.shutdown()
