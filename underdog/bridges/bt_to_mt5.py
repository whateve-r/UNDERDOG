"""
Backtrader to MT5 Bridge - Signal Translation for Live Trading

This module bridges Backtrader strategy signals to MT5 live execution:
- Intercepts self.buy() and self.sell() calls from Backtrader strategies
- Translates to mt5.order_send() with proper risk management
- Maintains audit trail of all signals and executions
- Handles position sizing and SL/TP calculation

Critical for: Paper Trading → Live Trading transition

Author: Underdog Trading System
Business Goal: Execute Backtrader strategies in MT5 for Prop Firm challenges
"""

import backtrader as bt
from typing import Optional, Dict, Any
import logging
from datetime import datetime
import pandas as pd

from underdog.execution.mt5_executor import MT5Executor, OrderType, OrderStatus

logger = logging.getLogger(__name__)


class BacktraderMT5Bridge:
    """
    Bridge between Backtrader strategy signals and MT5 execution
    
    Usage:
        # In your live trading script
        executor = MT5Executor(account=..., password=..., server=...)
        bridge = BacktraderMT5Bridge(executor=executor)
        
        # Attach to Cerebro
        cerebro = bt.Cerebro()
        cerebro.addstrategy(YourStrategy, mt5_bridge=bridge)
        cerebro.run()
    
    Strategy Integration:
        # In your Backtrader strategy __init__:
        self.mt5_bridge = kwargs.get('mt5_bridge', None)
        
        # In your strategy next():
        if buy_signal:
            if self.mt5_bridge:
                self.mt5_bridge.execute_buy(self, symbol="EURUSD", sl_pips=20, tp_pips=40)
            else:
                self.buy()  # Backtest mode
    """
    
    def __init__(
        self,
        executor: MT5Executor,
        default_volume: float = 0.1,
        default_sl_pips: float = 20,
        default_tp_pips: float = 40,
        enable_logging: bool = True
    ):
        """
        Initialize bridge
        
        Args:
            executor: MT5Executor instance (must be initialized)
            default_volume: Default lot size if not specified
            default_sl_pips: Default SL in pips if not specified
            default_tp_pips: Default TP in pips if not specified
            enable_logging: Log all signals and executions
        """
        self.executor = executor
        self.default_volume = default_volume
        self.default_sl_pips = default_sl_pips
        self.default_tp_pips = default_tp_pips
        self.enable_logging = enable_logging
        
        # Audit trail
        self.signal_log: list = []
        self.execution_log: list = []
        
        logger.info("BacktraderMT5Bridge initialized")
        logger.info(f"Default volume: {default_volume}, SL: {default_sl_pips} pips, TP: {default_tp_pips} pips")
    
    def execute_buy(
        self,
        strategy: bt.Strategy,
        symbol: str,
        volume: Optional[float] = None,
        sl_pips: Optional[float] = None,
        tp_pips: Optional[float] = None,
        comment: Optional[str] = None
    ) -> OrderStatus:
        """
        Execute BUY order in MT5
        
        Args:
            strategy: Backtrader strategy instance (for context)
            symbol: Trading symbol (e.g., "EURUSD")
            volume: Lot size (uses default if None)
            sl_pips: Stop Loss in pips (uses default if None)
            tp_pips: Take Profit in pips (uses default if None)
            comment: Order comment
        
        Returns:
            OrderStatus: Execution status
        """
        volume = volume or self.default_volume
        sl_pips = sl_pips or self.default_sl_pips
        tp_pips = tp_pips or self.default_tp_pips
        comment = comment or f"Underdog_{strategy.__class__.__name__}"
        
        # Log signal
        signal = {
            'timestamp': datetime.now(),
            'strategy': strategy.__class__.__name__,
            'signal': 'BUY',
            'symbol': symbol,
            'volume': volume,
            'sl_pips': sl_pips,
            'tp_pips': tp_pips
        }
        self.signal_log.append(signal)
        
        if self.enable_logging:
            logger.info(f"📊 SIGNAL: {signal}")
        
        # Execute in MT5
        result = self.executor.execute_order(
            symbol=symbol,
            order_type=OrderType.BUY,
            volume=volume,
            sl_pips=sl_pips,
            tp_pips=tp_pips,
            comment=comment
        )
        
        # Log execution
        execution = {
            'timestamp': result.timestamp,
            'strategy': strategy.__class__.__name__,
            'signal': 'BUY',
            'symbol': symbol,
            'status': result.status.value,
            'ticket': result.ticket,
            'price': result.price,
            'volume': result.volume,
            'dd_at_execution': result.dd_at_execution,
            'error_message': result.error_message
        }
        self.execution_log.append(execution)
        
        if self.enable_logging:
            if result.status == OrderStatus.SUCCESS:
                logger.info(f"✅ EXECUTED: Ticket {result.ticket}, Price {result.price}")
            else:
                logger.warning(f"❌ REJECTED: {result.status.value} - {result.error_message}")
        
        return result.status
    
    def execute_sell(
        self,
        strategy: bt.Strategy,
        symbol: str,
        volume: Optional[float] = None,
        sl_pips: Optional[float] = None,
        tp_pips: Optional[float] = None,
        comment: Optional[str] = None
    ) -> OrderStatus:
        """
        Execute SELL order in MT5
        
        Args:
            strategy: Backtrader strategy instance (for context)
            symbol: Trading symbol (e.g., "EURUSD")
            volume: Lot size (uses default if None)
            sl_pips: Stop Loss in pips (uses default if None)
            tp_pips: Take Profit in pips (uses default if None)
            comment: Order comment
        
        Returns:
            OrderStatus: Execution status
        """
        volume = volume or self.default_volume
        sl_pips = sl_pips or self.default_sl_pips
        tp_pips = tp_pips or self.default_tp_pips
        comment = comment or f"Underdog_{strategy.__class__.__name__}"
        
        # Log signal
        signal = {
            'timestamp': datetime.now(),
            'strategy': strategy.__class__.__name__,
            'signal': 'SELL',
            'symbol': symbol,
            'volume': volume,
            'sl_pips': sl_pips,
            'tp_pips': tp_pips
        }
        self.signal_log.append(signal)
        
        if self.enable_logging:
            logger.info(f"📊 SIGNAL: {signal}")
        
        # Execute in MT5
        result = self.executor.execute_order(
            symbol=symbol,
            order_type=OrderType.SELL,
            volume=volume,
            sl_pips=sl_pips,
            tp_pips=tp_pips,
            comment=comment
        )
        
        # Log execution
        execution = {
            'timestamp': result.timestamp,
            'strategy': strategy.__class__.__name__,
            'signal': 'SELL',
            'symbol': symbol,
            'status': result.status.value,
            'ticket': result.ticket,
            'price': result.price,
            'volume': result.volume,
            'dd_at_execution': result.dd_at_execution,
            'error_message': result.error_message
        }
        self.execution_log.append(execution)
        
        if self.enable_logging:
            if result.status == OrderStatus.SUCCESS:
                logger.info(f"✅ EXECUTED: Ticket {result.ticket}, Price {result.price}")
            else:
                logger.warning(f"❌ REJECTED: {result.status.value} - {result.error_message}")
        
        return result.status
    
    def close_all_positions(self, reason: str = "Strategy exit") -> int:
        """
        Close all open positions
        
        Args:
            reason: Reason for closing
        
        Returns:
            int: Number of positions closed
        """
        logger.info(f"Closing all positions - Reason: {reason}")
        return self.executor.emergency_close_all(reason=reason)
    
    def get_signal_log(self) -> pd.DataFrame:
        """
        Get signal log as DataFrame
        
        Returns:
            pd.DataFrame: All signals generated
        """
        if not self.signal_log:
            return pd.DataFrame()
        return pd.DataFrame(self.signal_log)
    
    def get_execution_log(self) -> pd.DataFrame:
        """
        Get execution log as DataFrame
        
        Returns:
            pd.DataFrame: All executions attempted
        """
        if not self.execution_log:
            return pd.DataFrame()
        return pd.DataFrame(self.execution_log)
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get bridge statistics
        
        Returns:
            Dict with signal/execution stats
        """
        if not self.execution_log:
            return {
                'total_signals': 0,
                'total_executions': 0,
                'success_rate': 0.0,
                'dd_rejections': 0,
                'mt5_rejections': 0,
                'connection_rejections': 0
            }
        
        executions = pd.DataFrame(self.execution_log)
        
        total_executions = len(executions)
        successful = len(executions[executions['status'] == 'success'])
        dd_rejected = len(executions[executions['status'] == 'rejected_dd_limit'])
        mt5_rejected = len(executions[executions['status'] == 'rejected_mt5_error'])
        conn_rejected = len(executions[executions['status'] == 'rejected_connection_lost'])
        
        return {
            'total_signals': len(self.signal_log),
            'total_executions': total_executions,
            'success_rate': (successful / total_executions * 100) if total_executions > 0 else 0.0,
            'successful_orders': successful,
            'dd_rejections': dd_rejected,
            'mt5_rejections': mt5_rejected,
            'connection_rejections': conn_rejected
        }
    
    def export_logs(self, filepath: str):
        """
        Export signal and execution logs to CSV
        
        Args:
            filepath: Base filepath (will create _signals.csv and _executions.csv)
        """
        signal_df = self.get_signal_log()
        execution_df = self.get_execution_log()
        
        if not signal_df.empty:
            signal_path = filepath.replace('.csv', '_signals.csv')
            signal_df.to_csv(signal_path, index=False)
            logger.info(f"Signals exported to {signal_path}")
        
        if not execution_df.empty:
            execution_path = filepath.replace('.csv', '_executions.csv')
            execution_df.to_csv(execution_path, index=False)
            logger.info(f"Executions exported to {execution_path}")


class LiveStrategy(bt.Strategy):
    """
    Base class for live Backtrader strategies with MT5 integration
    
    Inherit from this to automatically handle MT5 bridge integration:
    
    Example:
        class MyLiveStrategy(LiveStrategy):
            def __init__(self):
                super().__init__()
                self.sma = bt.indicators.SMA(self.data.close, period=20)
            
            def next(self):
                if self.data.close[0] > self.sma[0]:
                    self.execute_buy(symbol="EURUSD", sl_pips=20, tp_pips=40)
                elif self.data.close[0] < self.sma[0]:
                    self.execute_sell(symbol="EURUSD", sl_pips=20, tp_pips=40)
    """
    
    params = (
        ('mt5_bridge', None),  # BacktraderMT5Bridge instance
        ('symbol', 'EURUSD'),  # Default symbol
        ('volume', 0.1),       # Default volume
        ('sl_pips', 20),       # Default SL
        ('tp_pips', 40),       # Default TP
    )
    
    def __init__(self):
        super().__init__()
        self.mt5_bridge: Optional[BacktraderMT5Bridge] = self.params.mt5_bridge
        self.is_live = self.mt5_bridge is not None
        
        if self.is_live:
            logger.info(f"Strategy {self.__class__.__name__} running in LIVE mode with MT5")
        else:
            logger.info(f"Strategy {self.__class__.__name__} running in BACKTEST mode")
    
    def execute_buy(
        self,
        symbol: Optional[str] = None,
        volume: Optional[float] = None,
        sl_pips: Optional[float] = None,
        tp_pips: Optional[float] = None
    ):
        """
        Execute BUY - works in both backtest and live mode
        """
        if self.is_live:
            self.mt5_bridge.execute_buy(
                strategy=self,
                symbol=symbol or self.params.symbol,
                volume=volume or self.params.volume,
                sl_pips=sl_pips or self.params.sl_pips,
                tp_pips=tp_pips or self.params.tp_pips
            )
        else:
            self.buy()  # Backtest mode
    
    def execute_sell(
        self,
        symbol: Optional[str] = None,
        volume: Optional[float] = None,
        sl_pips: Optional[float] = None,
        tp_pips: Optional[float] = None
    ):
        """
        Execute SELL - works in both backtest and live mode
        """
        if self.is_live:
            self.mt5_bridge.execute_sell(
                strategy=self,
                symbol=symbol or self.params.symbol,
                volume=volume or self.params.volume,
                sl_pips=sl_pips or self.params.sl_pips,
                tp_pips=tp_pips or self.params.tp_pips
            )
        else:
            self.sell()  # Backtest mode
