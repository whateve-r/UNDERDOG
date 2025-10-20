"""
Event-Driven Backtesting Engine
Simulates tick-by-tick execution with realistic market conditions.

Implements the event-driven architecture recommended by Barry Johnson
and Marcos López de Prado for realistic backtesting.

Flow: TICK → SIGNAL → ORDER → FILL

Features:
- Bid/Ask spread modeling
- Dynamic slippage by time of day
- Commission tracking
- Latency simulation
- Position tracking
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable, Any
from datetime import datetime, timedelta
from queue import Queue
from enum import Enum
import pandas as pd
import numpy as np
from collections import defaultdict


class EventType(Enum):
    """Event types in the backtesting system"""
    TICK = "tick"
    SIGNAL = "signal"
    ORDER = "order"
    FILL = "fill"


@dataclass
class TickEvent:
    """Market tick event"""
    timestamp: datetime
    symbol: str
    bid: float
    ask: float
    volume: float = 0.0
    
    def mid_price(self) -> float:
        """Calculate mid price"""
        return (self.bid + self.ask) / 2
    
    def spread(self) -> float:
        """Calculate spread in price units"""
        return self.ask - self.bid


@dataclass
class SignalEvent:
    """Strategy signal event"""
    timestamp: datetime
    strategy_id: str
    symbol: str
    side: str  # 'buy' or 'sell'
    confidence: float
    size_suggestion: float = 0.0
    
    def __repr__(self) -> str:
        return (f"SignalEvent({self.strategy_id}, {self.symbol}, "
                f"{self.side.upper()}, conf={self.confidence:.2f})")


@dataclass
class OrderEvent:
    """Order event"""
    timestamp: datetime
    order_id: str
    symbol: str
    side: str  # 'buy' or 'sell'
    size: float
    order_type: str  # 'market' or 'limit'
    limit_price: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    strategy_id: str = "unknown"
    
    def __repr__(self) -> str:
        return (f"OrderEvent({self.order_id}, {self.symbol}, "
                f"{self.side.upper()}, size={self.size:.2f})")


@dataclass
class FillEvent:
    """Order fill event"""
    timestamp: datetime
    order_id: str
    symbol: str
    side: str
    size: float
    fill_price: float
    commission: float
    slippage: float
    strategy_id: str = "unknown"
    
    def total_cost(self) -> float:
        """Calculate total cost including commission and slippage"""
        return self.size * self.fill_price + self.commission + self.slippage
    
    def __repr__(self) -> str:
        return (f"FillEvent({self.order_id}, {self.symbol}, "
                f"price={self.fill_price:.5f}, slip={self.slippage:.5f})")


@dataclass
class Position:
    """Position tracker"""
    symbol: str
    side: str  # 'long' or 'short'
    size: float
    entry_price: float
    entry_time: datetime
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    strategy_id: str = "unknown"
    
    def unrealized_pnl(self, current_price: float) -> float:
        """Calculate unrealized P&L"""
        if self.side == 'long':
            return (current_price - self.entry_price) * self.size
        else:  # short
            return (self.entry_price - current_price) * self.size
    
    def bars_held(self, current_time: datetime, bar_duration_minutes: int) -> int:
        """Calculate number of bars held"""
        time_held_minutes = (current_time - self.entry_time).total_seconds() / 60
        return int(time_held_minutes / bar_duration_minutes)


@dataclass
class BacktestConfig:
    """Event-driven backtest configuration"""
    initial_capital: float = 100000.0
    commission_pct: float = 0.0  # As percentage (e.g., 0.0005 = 0.05%)
    base_slippage_pips: float = 0.5  # Base slippage in pips
    use_dynamic_spread: bool = True
    use_dynamic_slippage: bool = True
    
    # Slippage multipliers by hour (GMT)
    slippage_by_hour: Dict[int, float] = field(default_factory=lambda: {
        # London open (8-9 AM): Low slippage
        8: 0.5, 9: 0.6,
        # NY open (13-14 PM): Low slippage
        13: 0.6, 14: 0.7,
        # Asian session (2-6 AM): High slippage
        2: 1.5, 3: 1.5, 4: 1.5, 5: 1.5, 6: 1.5,
        # Default
        'default': 1.0
    })


class EventDrivenBacktest:
    """
    Event-driven backtesting engine.
    
    Simulates realistic market execution with:
    - Tick-by-tick processing
    - Bid/Ask spread
    - Dynamic slippage
    - Commission costs
    - Position tracking
    
    Example:
        >>> config = BacktestConfig(initial_capital=100000)
        >>> backtest = EventDrivenBacktest(config)
        >>> results = backtest.run(tick_data, strategy)
        >>> print(f"Final Capital: ${results['final_capital']:.2f}")
    """
    
    def __init__(self, config: Optional[BacktestConfig] = None):
        """
        Initialize event-driven backtest.
        
        Args:
            config: Backtest configuration
        """
        self.config = config or BacktestConfig()
        
        # Event queue
        self.event_queue: Queue = Queue()
        
        # State
        self.current_capital = self.config.initial_capital
        self.initial_capital = self.config.initial_capital
        self.peak_capital = self.config.initial_capital
        
        # Positions
        self.positions: Dict[str, Position] = {}
        
        # Trade history
        self.trades: List[Dict] = []
        self.fills: List[FillEvent] = []
        
        # Market data cache
        self.current_ticks: Dict[str, TickEvent] = {}
        
        # Equity curve
        self.equity_curve: List[Dict] = []
        
        # Order tracking
        self.order_counter = 0
    
    def run(self, 
            tick_data: pd.DataFrame,
            strategy: Callable,
            bar_duration_minutes: int = 15) -> Dict[str, Any]:
        """
        Run event-driven backtest.
        
        Args:
            tick_data: DataFrame with columns [timestamp, symbol, bid, ask, volume]
            strategy: Strategy function that processes ticks and returns signals
            bar_duration_minutes: Timeframe in minutes for position tracking
        
        Returns:
            Dict with backtest results
        """
        print(f"[EVENT-DRIVEN] Starting backtest")
        print(f"[EVENT-DRIVEN] Initial capital: ${self.current_capital:,.2f}")
        print(f"[EVENT-DRIVEN] Total ticks: {len(tick_data):,}")
        
        # Ensure datetime index
        if not isinstance(tick_data.index, pd.DatetimeIndex):
            tick_data = tick_data.set_index('timestamp')
        
        # Process each tick
        for idx, row in tick_data.iterrows():
            # Create TICK event
            tick = TickEvent(
                timestamp=idx,
                symbol=row['symbol'],
                bid=row['bid'],
                ask=row['ask'],
                volume=row.get('volume', 0.0)
            )
            
            # Cache current tick
            self.current_ticks[tick.symbol] = tick
            
            # Add to event queue
            self.event_queue.put(tick)
            
            # Process all events in queue
            while not self.event_queue.empty():
                event = self.event_queue.get()
                
                if isinstance(event, TickEvent):
                    self._handle_tick(event, strategy)
                elif isinstance(event, SignalEvent):
                    self._handle_signal(event)
                elif isinstance(event, OrderEvent):
                    self._handle_order(event)
                elif isinstance(event, FillEvent):
                    self._handle_fill(event)
            
            # Update equity curve
            self._update_equity(tick.timestamp)
        
        # Calculate final results
        results = self._calculate_results()
        self._print_summary(results)
        
        return results
    
    def _handle_tick(self, tick: TickEvent, strategy: Callable):
        """
        Process tick event and generate signals.
        
        Args:
            tick: Tick event
            strategy: Strategy function
        """
        # Let strategy process tick
        signal = strategy(tick, self.positions, self.current_capital)
        
        if signal:
            self.event_queue.put(signal)
    
    def _handle_signal(self, signal: SignalEvent):
        """
        Convert signal to order with risk management.
        
        Args:
            signal: Signal event
        """
        # Simple conversion: signal → order
        # In production, this would include:
        # - Position sizing
        # - Risk checks (Risk Master)
        # - Portfolio exposure limits
        
        if signal.size_suggestion <= 0:
            return
        
        # Create order
        self.order_counter += 1
        order = OrderEvent(
            timestamp=signal.timestamp,
            order_id=f"ORD_{self.order_counter:06d}",
            symbol=signal.symbol,
            side=signal.side,
            size=signal.size_suggestion,
            order_type='market',
            strategy_id=signal.strategy_id
        )
        
        self.event_queue.put(order)
    
    def _handle_order(self, order: OrderEvent):
        """
        Simulate order execution with slippage.
        
        Args:
            order: Order event
        """
        # Get current tick
        tick = self.current_ticks.get(order.symbol)
        if not tick:
            print(f"[EVENT-DRIVEN] WARNING: No tick data for {order.symbol}")
            return
        
        # Calculate fill price with spread
        if order.side == 'buy':
            fill_price = tick.ask  # Buy at ask
        else:  # sell
            fill_price = tick.bid  # Sell at bid
        
        # Add slippage
        slippage_pips = self._calculate_slippage(order.timestamp, order.symbol)
        slippage_price = slippage_pips * 0.0001  # Convert pips to price (for EURUSD)
        
        if order.side == 'buy':
            fill_price += slippage_price
        else:
            fill_price -= slippage_price
        
        # Calculate commission
        commission = order.size * fill_price * self.config.commission_pct
        
        # Create FILL event
        fill = FillEvent(
            timestamp=order.timestamp,
            order_id=order.order_id,
            symbol=order.symbol,
            side=order.side,
            size=order.size,
            fill_price=fill_price,
            commission=commission,
            slippage=slippage_pips,
            strategy_id=order.strategy_id
        )
        
        self.event_queue.put(fill)
    
    def _handle_fill(self, fill: FillEvent):
        """
        Update portfolio with executed trade.
        
        Args:
            fill: Fill event
        """
        # Record fill
        self.fills.append(fill)
        
        # Update positions
        if fill.symbol in self.positions:
            # Close existing position
            position = self.positions[fill.symbol]
            realized_pnl = position.unrealized_pnl(fill.fill_price)
            
            # Record trade
            self.trades.append({
                'timestamp': fill.timestamp,
                'symbol': fill.symbol,
                'side': position.side,
                'entry_price': position.entry_price,
                'exit_price': fill.fill_price,
                'size': position.size,
                'pnl': realized_pnl - fill.commission,
                'commission': fill.commission,
                'slippage': fill.slippage,
                'strategy_id': fill.strategy_id,
                'bars_held': position.bars_held(fill.timestamp, 15)
            })
            
            # Update capital
            self.current_capital += realized_pnl - fill.commission
            
            # Remove position
            del self.positions[fill.symbol]
        else:
            # Open new position
            position = Position(
                symbol=fill.symbol,
                side='long' if fill.side == 'buy' else 'short',
                size=fill.size,
                entry_price=fill.fill_price,
                entry_time=fill.timestamp,
                strategy_id=fill.strategy_id
            )
            self.positions[fill.symbol] = position
            
            # Deduct capital
            self.current_capital -= fill.total_cost()
    
    def _calculate_slippage(self, timestamp: datetime, symbol: str) -> float:
        """
        Calculate dynamic slippage based on time of day.
        
        Args:
            timestamp: Order timestamp
            symbol: Trading symbol
        
        Returns:
            Slippage in pips
        """
        if not self.config.use_dynamic_slippage:
            return self.config.base_slippage_pips
        
        # Get hour of day (GMT)
        hour = timestamp.hour
        
        # Get slippage multiplier
        multiplier = self.config.slippage_by_hour.get(
            hour,
            self.config.slippage_by_hour['default']
        )
        
        return self.config.base_slippage_pips * multiplier
    
    def _update_equity(self, timestamp: datetime):
        """
        Update equity curve with current portfolio value.
        
        Args:
            timestamp: Current timestamp
        """
        # Calculate unrealized P&L
        unrealized_pnl = 0.0
        for symbol, position in self.positions.items():
            tick = self.current_ticks.get(symbol)
            if tick:
                unrealized_pnl += position.unrealized_pnl(tick.mid_price())
        
        # Total equity
        total_equity = self.current_capital + unrealized_pnl
        
        # Update peak
        if total_equity > self.peak_capital:
            self.peak_capital = total_equity
        
        # Record
        self.equity_curve.append({
            'timestamp': timestamp,
            'equity': total_equity,
            'cash': self.current_capital,
            'unrealized_pnl': unrealized_pnl,
            'drawdown_pct': ((self.peak_capital - total_equity) / self.peak_capital) * 100
        })
    
    def _calculate_results(self) -> Dict[str, Any]:
        """Calculate backtest results"""
        if len(self.trades) == 0:
            return {
                'final_capital': self.current_capital,
                'total_return_pct': 0.0,
                'num_trades': 0,
                'sharpe_ratio': 0.0,
                'max_drawdown_pct': 0.0
            }
        
        trades_df = pd.DataFrame(self.trades)
        
        # Returns
        total_return = (self.current_capital - self.initial_capital) / self.initial_capital * 100
        
        # Win rate
        winning_trades = len(trades_df[trades_df['pnl'] > 0])
        win_rate = winning_trades / len(trades_df) * 100
        
        # Sharpe ratio
        returns = trades_df['pnl'] / self.initial_capital
        sharpe = (returns.mean() / returns.std()) * np.sqrt(252) if returns.std() > 0 else 0.0
        
        # Max drawdown
        equity_df = pd.DataFrame(self.equity_curve)
        max_dd = equity_df['drawdown_pct'].max() if len(equity_df) > 0 else 0.0
        
        return {
            'final_capital': self.current_capital,
            'total_return_pct': total_return,
            'num_trades': len(trades_df),
            'win_rate_pct': win_rate,
            'sharpe_ratio': sharpe,
            'max_drawdown_pct': max_dd,
            'avg_pnl': trades_df['pnl'].mean(),
            'total_commission': trades_df['commission'].sum(),
            'avg_slippage_pips': trades_df['slippage'].mean(),
            'trades': trades_df,
            'equity_curve': pd.DataFrame(self.equity_curve)
        }
    
    def _print_summary(self, results: Dict[str, Any]):
        """Print backtest summary"""
        print(f"\n{'='*60}")
        print(f" EVENT-DRIVEN BACKTEST SUMMARY")
        print(f"{'='*60}")
        print(f"Initial Capital:      ${self.initial_capital:,.2f}")
        print(f"Final Capital:        ${results['final_capital']:,.2f}")
        print(f"Total Return:         {results['total_return_pct']:.2f}%")
        print(f"{'─'*60}")
        print(f"Total Trades:         {results['num_trades']}")
        print(f"Win Rate:             {results['win_rate_pct']:.2f}%")
        print(f"Sharpe Ratio:         {results['sharpe_ratio']:.3f}")
        print(f"Max Drawdown:         {results['max_drawdown_pct']:.2f}%")
        print(f"{'─'*60}")
        print(f"Avg P&L per trade:    ${results['avg_pnl']:.2f}")
        print(f"Total Commission:     ${results['total_commission']:.2f}")
        print(f"Avg Slippage:         {results['avg_slippage_pips']:.2f} pips")
        print(f"{'='*60}\n")
