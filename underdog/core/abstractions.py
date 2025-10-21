"""
Core Abstractions for Event-Driven Backtesting Engine
Based on Strategy Pattern for maximum decoupling and ML integration
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Optional
import pandas as pd


# ============================================================================
# DATA STRUCTURES
# ============================================================================

class EventType(Enum):
    """Types of events in the event-driven architecture"""
    MARKET = 'MARKET'  # New market data (tick/bar)
    SIGNAL = 'SIGNAL'  # Trading signal generated
    ORDER = 'ORDER'    # Order to be executed
    FILL = 'FILL'      # Order executed (filled)


class OrderType(Enum):
    """Order types"""
    MARKET = 'MARKET'
    LIMIT = 'LIMIT'
    STOP = 'STOP'
    STOP_LIMIT = 'STOP_LIMIT'


class OrderSide(Enum):
    """Order side"""
    BUY = 'BUY'
    SELL = 'SELL'


class SignalType(Enum):
    """Signal types"""
    LONG = 'LONG'
    SHORT = 'SHORT'
    EXIT_LONG = 'EXIT_LONG'
    EXIT_SHORT = 'EXIT_SHORT'
    NONE = 'NONE'


@dataclass
class MarketEvent:
    """
    Market data event (new bar or tick).
    
    CRITICAL: For Forex rigor, bid/ask MUST be provided (not just mid-price).
    Without bid/ask segregation, slippage and TCA costs are underestimated,
    artificially inflating backtest performance (optimization bias).
    
    Best Practice: Use tick data or 1-minute bid/ask bars from institutional providers.
    """
    timestamp: datetime
    symbol: str
    open: float
    high: float
    low: float
    close: float
    volume: int
    bid: Optional[float] = None  # REQUIRED for realistic Forex modeling
    ask: Optional[float] = None  # REQUIRED for realistic Forex modeling
    
    @property
    def spread(self) -> Optional[float]:
        """
        Calculate spread if bid/ask available.
        
        Typical Forex spreads:
        - EURUSD: 0.1-0.5 pips (0.00001-0.00005)
        - XAUUSD: 20-50 pips (0.20-0.50)
        
        Dynamic spread modeling: Spreads widen during news/low liquidity.
        """
        if self.bid and self.ask:
            return self.ask - self.bid
        return None
    
    @property
    def mid_price(self) -> float:
        """
        Calculate mid-price (average of bid/ask).
        Use ONLY for reference, NOT for backtest execution.
        """
        if self.bid and self.ask:
            return (self.bid + self.ask) / 2.0
        return self.close  # Fallback to close if bid/ask unavailable


@dataclass
class SignalEvent:
    """Trading signal event"""
    timestamp: datetime
    symbol: str
    signal_type: SignalType
    strength: float = 1.0  # Signal confidence (0-1)
    metadata: dict = None  # ML predictions, indicators, etc.


@dataclass
class OrderEvent:
    """Order event to be executed"""
    timestamp: datetime
    symbol: str
    order_type: OrderType
    side: OrderSide
    quantity: float
    price: Optional[float] = None  # For limit/stop orders
    stop_price: Optional[float] = None


@dataclass
class FillEvent:
    """Order execution (fill) event"""
    timestamp: datetime
    symbol: str
    side: OrderSide
    quantity: float
    fill_price: float
    commission: float
    slippage: float = 0.0
    order_id: Optional[str] = None


# ============================================================================
# ABSTRACT BASE CLASSES (Strategy Pattern)
# ============================================================================

class DataHandler(ABC):
    """
    Abstract base class for data handling.
    
    Implementations:
    - HistoricalDataHandler: Load historical bars from Parquet/HF/CSV
    - LiveDataHandler: Stream live data from broker API
    - HuggingFaceDataHandler: Load from HF Datasets Hub
    """
    
    @abstractmethod
    def get_latest_bar(self, symbol: str) -> Optional[MarketEvent]:
        """Get the most recent bar for a symbol"""
        pass
    
    @abstractmethod
    def get_latest_bars(self, symbol: str, n: int = 1) -> list[MarketEvent]:
        """Get the N most recent bars for a symbol"""
        pass
    
    @abstractmethod
    def update_bars(self) -> bool:
        """
        Push next bar(s) to the event queue.
        Returns True if more data available, False otherwise.
        """
        pass
    
    @abstractmethod
    def get_available_symbols(self) -> list[str]:
        """Return list of symbols available"""
        pass


class Strategy(ABC):
    """
    Abstract base class for trading strategies.
    
    Key principle: Strategy ONLY generates signals.
    It has NO knowledge of:
    - Portfolio management
    - Position sizing
    - Order execution
    
    This decoupling allows:
    - Easy ML model integration (model = strategy)
    - A/B testing of strategies
    - Same strategy code for backtest → paper → live
    """
    
    def __init__(self, symbols: list[str], **kwargs):
        self.symbols = symbols
        self.kwargs = kwargs
    
    @abstractmethod
    def generate_signal(self, market_event: MarketEvent) -> SignalEvent:
        """
        Core strategy logic.
        
        Takes market data, returns trading signal.
        Must be CAUSAL (no look-ahead bias):
        - Only use data available at market_event.timestamp
        - All indicators must be lagged by 1 period
        
        Args:
            market_event: New market data
            
        Returns:
            SignalEvent with signal type and strength
        """
        pass
    
    def calculate_indicators(self, bars: list[MarketEvent]) -> dict[str, Any]:
        """
        Optional: Calculate technical indicators.
        
        Override this method if strategy uses indicators.
        Ensure all calculations are lagged to avoid look-ahead bias.
        """
        return {}


class Portfolio(ABC):
    """
    Abstract base class for portfolio management.
    
    PROP FIRM DRAWDOWN TRACKING REQUIREMENTS:
    ------------------------------------------
    1. DAILY DRAWDOWN:
       - Track starting equity at start of each trading day
       - Calculate: DD_daily = (Equity_start - Equity_current) / Equity_start
       - HARD LIMIT: Typically 5-6% (panic close if breached)
    
    2. TOTAL DRAWDOWN:
       - Track high-water mark (peak equity across all time)
       - Calculate: DD_total = (HWM - Equity_current) / HWM
       - HARD LIMIT: Typically 10-12%
    
    3. PANIC CLOSE:
       - Emergency liquidation of ALL positions
       - Triggered when DD limits breached
       - Bypass normal order flow (immediate execution)
       - Log incident for review
    
    4. EQUITY CURVE:
       - Track equity at each bar
       - Calculate running P&L
       - Store for performance analysis
    
    Responsibilities:
    - Convert signals → orders (position sizing)
    - Track positions and P&L
    - Track drawdowns (daily and total)
    - Apply risk management rules
    - Generate OrderEvents
    - Trigger panic close when limits breached
    """
    
    @abstractmethod
    def update_signal(self, signal_event: SignalEvent) -> Optional[OrderEvent]:
        """
        Convert signal to order (if risk checks pass).
        
        Args:
            signal_event: Trading signal from strategy
            
        Returns:
            OrderEvent if position should be opened/closed, None otherwise
        """
        pass
    
    @abstractmethod
    def update_fill(self, fill_event: FillEvent) -> None:
        """
        Update portfolio after order execution.
        
        Args:
            fill_event: Executed order details
        """
        pass
    
    @abstractmethod
    def get_current_positions(self) -> dict[str, float]:
        """Return current positions (symbol → quantity)"""
        pass
    
    @abstractmethod
    def get_total_equity(self) -> float:
        """Return total portfolio value (cash + positions)"""
        pass
    
    @abstractmethod
    def get_holdings(self) -> dict[str, dict]:
        """Return detailed holdings (positions, cash, P&L)"""
        pass
    
    @abstractmethod
    def get_daily_drawdown(self) -> float:
        """
        Calculate daily drawdown from start of trading day.
        
        FORMULA:
        DD_daily = (Equity_start_of_day - Equity_current) / Equity_start_of_day
        
        PROP FIRM COMPLIANCE:
        - Track starting equity at market open each day
        - Reset daily_start_equity at 00:00 UTC or session start
        - Typical limit: 5-6%
        
        Returns:
            Daily drawdown as decimal (e.g., 0.05 = 5%)
            Positive value = drawdown, 0 = no drawdown, negative = profit
        """
        pass
    
    @abstractmethod
    def get_total_drawdown(self) -> float:
        """
        Calculate total drawdown from high-water mark.
        
        FORMULA:
        DD_total = (High_Water_Mark - Equity_current) / High_Water_Mark
        
        HIGH-WATER MARK:
        - Peak equity across all time
        - Update when current equity > HWM
        - Never decreases (only increases with new peaks)
        
        PROP FIRM COMPLIANCE:
        - Typical limit: 10-12%
        - More lenient than daily DD (allows recovery time)
        
        Returns:
            Total drawdown as decimal (e.g., 0.10 = 10%)
            Positive value = drawdown, 0 = at peak equity
        """
        pass
    
    @abstractmethod
    def get_high_water_mark(self) -> float:
        """
        Get current high-water mark (peak equity).
        
        Returns:
            Peak equity value
        """
        pass
    
    @abstractmethod
    def reset_daily_start_equity(self) -> None:
        """
        Reset daily starting equity (call at start of each trading day).
        
        IMPLEMENTATION:
        - Call at 00:00 UTC or market open (09:00 London, 14:00 NY)
        - Set daily_start_equity = current_equity
        - Used for daily DD calculation
        """
        pass
    
    @abstractmethod
    def trigger_panic_close(self, reason: str) -> list[OrderEvent]:
        """
        Emergency liquidation of ALL positions.
        
        TRIGGERS:
        - Daily DD limit breached (5-6%)
        - Total DD limit breached (10-12%)
        - Risk event detected (e.g., flash crash, broker margin call)
        
        EXECUTION:
        - Close ALL open positions immediately
        - Use market orders (no limit orders)
        - Bypass normal order validation
        - Log incident with reason
        - Send notification (email/SMS)
        
        CRITICAL: Must execute IMMEDIATELY to prevent further losses.
        
        Args:
            reason: Why panic close triggered (for logging)
            
        Returns:
            List of OrderEvents to close all positions
        """
        pass


class ExecutionHandler(ABC):
    """
    Abstract base class for order execution.
    
    TCA (Transaction Cost Analysis) MODELING REQUIREMENTS:
    --------------------------------------------------------
    1. SPREAD: Bid/Ask segregated execution
       - BUY orders: Execute at ASK price
       - SELL orders: Execute at BID price
       - NEVER use mid-price (underestimates costs by 30-50%)
    
    2. SLIPPAGE: Function of volatility and order size
       - Increases with ATR (market volatility)
       - Increases with order size (market impact)
       - Increases during news events (spread widening)
       - Typical: 0.01-0.1% for liquid pairs, 0.5-2% during news
    
    3. COMMISSION: Fixed or percentage-based
       - Prop Firms: Usually built into spread
       - Retail Brokers: $3-7 per lot
    
    4. MARKET IMPACT: Larger orders → higher slippage
       - Small retail orders (<1 lot): Minimal impact
       - Large orders (>10 lots): Significant impact
    
    Implementations:
    - SimulatedExecution: Backtesting (models slippage/spread/commission)
    - LiveExecution: Real broker API (MT5/IBKR/Alpaca)
    
    CRITICAL: Realistic TCA modeling is the difference between
    profitable backtests and profitable live trading.
    """
    
    @abstractmethod
    def execute_order(self, order_event: OrderEvent) -> FillEvent:
        """
        Execute an order.
        
        For backtesting: Simulate fill with slippage/spread model
        For live trading: Send order to broker API
        
        Args:
            order_event: Order to execute
            
        Returns:
            FillEvent with actual fill price and costs
        """
        pass
    
    @abstractmethod
    def get_commission(self, fill_event: FillEvent) -> float:
        """Calculate commission for a fill"""
        pass
    
    @abstractmethod
    def calculate_slippage(
        self,
        order: OrderEvent,
        market_event: MarketEvent,
        atr: float,
        volatility_multiplier: float = 1.0,
        size_multiplier: float = 1.0
    ) -> float:
        """
        Calculate realistic slippage for order execution.
        
        Slippage increases with:
        1. Volatility (ATR) - More volatile markets = harder to fill
        2. Order size - Larger orders = market impact
        3. News events - Spread widening during high-impact news
        4. Liquidity - Low liquidity sessions = higher slippage
        
        Formula (example):
        slippage_pips = base_slippage + (ATR * vol_mult) + (size * size_mult)
        
        Args:
            order: Order to execute
            market_event: Current market data
            atr: Average True Range (volatility measure)
            volatility_multiplier: Scale ATR contribution (default 1.0)
            size_multiplier: Scale size contribution (default 1.0)
            
        Returns:
            Slippage in price units (e.g., 0.0002 = 2 pips EURUSD)
        
        Typical values:
        - EURUSD liquid: 0.1-0.5 pips (0.00001-0.00005)
        - EURUSD news: 2-5 pips (0.0002-0.0005)
        - XAUUSD liquid: 20-50 pips (0.20-0.50)
        - XAUUSD news: 100-300 pips (1.0-3.0)
        """
        pass
    
    @abstractmethod
    def get_realistic_fill_price(
        self,
        order: OrderEvent,
        market_event: MarketEvent,
        slippage: float = 0.0
    ) -> float:
        """
        Calculate realistic fill price with bid/ask and slippage.
        
        EXECUTION RULES:
        - BUY orders: Fill at ASK + slippage (higher cost)
        - SELL orders: Fill at BID - slippage (lower revenue)
        
        NEVER:
        - Use mid-price for execution (inflates results)
        - Ignore slippage (unrealistic)
        - Use same price for buys/sells (violates microstructure)
        
        Args:
            order: Order to execute
            market_event: Current market data (with bid/ask)
            slippage: Additional slippage from calculate_slippage()
            
        Returns:
            Realistic fill price
            
        Example:
            Market: bid=1.1000, ask=1.1002 (2 pip spread)
            BUY: fill at 1.1002 + slippage
            SELL: fill at 1.1000 - slippage
        """
        pass


class RiskManager(ABC):
    """
    Abstract base class for risk management.
    
    PROP FIRM COMPLIANCE REQUIREMENTS:
    - Daily Drawdown Limit: Typically 5-6% (HARD STOP)
    - Total Drawdown Limit: Typically 10-12%
    - Minimum Trading Days: 4-5 days before payout
    - Consistency: No single day > 40% of total profit
    
    POSITION SIZING HIERARCHY:
    1. Fixed Fractional Risk (1-2% per trade) - Universal Best Practice
    2. Kelly Criterion (fractional) - Long-term growth optimization
    3. Confidence Weighted - ML signal quality modulation
    
    DISCIPLINE RULES (MUST ENFORCE):
    - Never average down (no adding to losers)
    - Panic close if daily DD limit breached
    - No trading during high-impact news (unless strategy designed for it)
    """
    
    @abstractmethod
    def check_order(self, order_event: OrderEvent, portfolio: Portfolio) -> bool:
        """
        Validate order against risk rules.
        
        CRITICAL CHECKS:
        1. Daily drawdown < limit (e.g., 5%)
        2. Total drawdown < limit (e.g., 10%)
        3. Position size within limits
        4. Not adding to losing position (no averaging down)
        5. Margin requirements met
        
        Args:
            order_event: Order to validate
            portfolio: Current portfolio state
            
        Returns:
            True if order passes risk checks, False otherwise
        """
        pass
    
    @abstractmethod
    def check_daily_drawdown_limit(self, portfolio: Portfolio, limit: float = 0.05) -> bool:
        """
        Check if daily drawdown limit breached.
        
        PROP FIRM CRITICAL: If breached, PANIC CLOSE all positions immediately.
        
        Args:
            portfolio: Current portfolio state
            limit: Daily DD limit (default 5% = 0.05)
            
        Returns:
            True if within limit, False if breached (STOP TRADING)
        """
        pass
    
    @abstractmethod
    def calculate_position_size(
        self,
        signal: SignalEvent,
        portfolio: Portfolio,
        risk_per_trade: float = 0.02,  # 2% default
        stop_loss_distance: Optional[float] = None,
        confidence_multiplier: float = 1.0  # ML confidence weighting
    ) -> float:
        """
        Calculate optimal position size.
        
        FORMULA (Fixed Fractional Risk):
        Position Size = (Account Equity × Risk%) / Stop Loss Distance
        
        ENHANCEMENTS:
        - Kelly Criterion: Use Half-Kelly (0.5 × f*) for conservative growth
        - Confidence Weighted: Multiply by ML signal confidence (0.0-1.0)
        - ATR-based SL: Stop loss distance = ATR × multiplier (e.g., 2.0)
        
        Args:
            signal: Trading signal
            portfolio: Current portfolio state
            risk_per_trade: Risk percentage per trade (default 2%)
            stop_loss_distance: SL distance in price points (if None, use ATR)
            confidence_multiplier: ML confidence score (0.0-1.0)
            
        Returns:
            Position size (in lots/units)
        """
        pass
    
    @abstractmethod
    def calculate_kelly_fraction(
        self,
        win_rate: float,
        avg_win_loss_ratio: float,
        fraction: float = 0.5  # Half-Kelly default
    ) -> float:
        """
        Calculate Kelly Criterion position size (fractional).
        
        FORMULA: f* = (b × p - q) / b
        Where:
        - p = win rate
        - q = 1 - p (loss rate)
        - b = avg_win / avg_loss
        
        CRITICAL: Always use fractional Kelly (0.25-0.5) in production.
        Full Kelly is too volatile and can lead to ruin.
        
        Args:
            win_rate: Historical win rate (0.0-1.0)
            avg_win_loss_ratio: Average win / average loss
            fraction: Kelly fraction (0.5 = Half-Kelly, 0.25 = Quarter-Kelly)
            
        Returns:
            Optimal position fraction (0.0-1.0)
        """
        pass


# ============================================================================
# EVENT QUEUE INTERFACE
# ============================================================================

class EventQueue(ABC):
    """
    Abstract base class for event queue.
    
    The event queue is the "heartbeat" of the event-driven engine.
    All events flow through this queue in chronological order.
    """
    
    @abstractmethod
    def put(self, event: Any) -> None:
        """Add event to queue"""
        pass
    
    @abstractmethod
    def get(self) -> Any:
        """Get next event from queue"""
        pass
    
    @abstractmethod
    def empty(self) -> bool:
        """Check if queue is empty"""
        pass


# ============================================================================
# EXAMPLE: Concrete Implementation Signatures
# ============================================================================

"""
Example implementations (to be created in respective modules):

class SMACrossoverStrategy(Strategy):
    def __init__(self, symbols, fast_period=10, slow_period=50):
        super().__init__(symbols)
        self.fast = fast_period
        self.slow = slow_period
    
    def generate_signal(self, market_event):
        bars = self.data_handler.get_latest_bars(market_event.symbol, self.slow)
        if len(bars) < self.slow:
            return SignalEvent(market_event.timestamp, market_event.symbol, SignalType.NONE)
        
        closes = [b.close for b in bars]
        sma_fast = sum(closes[-self.fast:]) / self.fast
        sma_slow = sum(closes[-self.slow:]) / self.slow
        
        if sma_fast > sma_slow:
            return SignalEvent(market_event.timestamp, market_event.symbol, SignalType.LONG)
        elif sma_fast < sma_slow:
            return SignalEvent(market_event.timestamp, market_event.symbol, SignalType.SHORT)
        else:
            return SignalEvent(market_event.timestamp, market_event.symbol, SignalType.NONE)


class MLPredictorStrategy(Strategy):
    def __init__(self, symbols, model_path):
        super().__init__(symbols)
        self.model = load_model(model_path)  # Keras/PyTorch model
    
    def generate_signal(self, market_event):
        # Get historical bars for feature engineering
        bars = self.data_handler.get_latest_bars(market_event.symbol, 100)
        
        # Calculate features (lagged returns, ATR, RSI)
        features = self.calculate_indicators(bars)
        
        # Predict direction (+1 = long, -1 = short, 0 = neutral)
        prediction = self.model.predict(features)
        confidence = abs(prediction)
        
        if prediction > 0.6:
            return SignalEvent(
                market_event.timestamp,
                market_event.symbol,
                SignalType.LONG,
                strength=confidence,
                metadata={'prediction': prediction}
            )
        elif prediction < -0.6:
            return SignalEvent(
                market_event.timestamp,
                market_event.symbol,
                SignalType.SHORT,
                strength=confidence,
                metadata={'prediction': prediction}
            )
        else:
            return SignalEvent(
                market_event.timestamp,
                market_event.symbol,
                SignalType.NONE,
                strength=0.0
            )
"""
