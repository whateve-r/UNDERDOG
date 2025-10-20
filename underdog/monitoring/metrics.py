"""
Prometheus Metrics Collector
Tracks trading performance, execution latency, and system health metrics.
"""
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Optional
from prometheus_client import (
    Counter, Gauge, Histogram, Summary,
    CollectorRegistry, push_to_gateway, generate_latest
)


@dataclass
class TradingMetrics:
    """Container for trading performance metrics"""
    # Cumulative metrics
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    
    # Financial metrics
    current_capital: float = 0.0
    initial_capital: float = 100000.0
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0
    
    # Risk metrics
    current_drawdown: float = 0.0
    max_drawdown: float = 0.0
    daily_drawdown: float = 0.0
    
    # Position metrics
    open_positions: int = 0
    total_exposure: float = 0.0
    
    # Execution metrics
    last_execution_time_ms: float = 0.0
    avg_execution_time_ms: float = 0.0
    
    # Strategy metrics
    active_strategies: int = 0
    signals_generated: int = 0
    signals_rejected: int = 0
    
    # Timestamp
    last_update: datetime = field(default_factory=datetime.utcnow)
    
    def win_rate(self) -> float:
        """Calculate win rate percentage"""
        if self.total_trades == 0:
            return 0.0
        return (self.winning_trades / self.total_trades) * 100.0
    
    def total_return(self) -> float:
        """Calculate total return percentage"""
        if self.initial_capital == 0:
            return 0.0
        return ((self.current_capital - self.initial_capital) / self.initial_capital) * 100.0
    
    def sharpe_ratio(self, risk_free_rate: float = 0.0) -> float:
        """Calculate Sharpe ratio (simplified)"""
        # Note: Proper Sharpe requires returns history
        # This is a placeholder for real-time estimation
        return 0.0


class MetricsCollector:
    """
    Prometheus Metrics Collector for UNDERDOG Trading System
    
    Tracks:
    - Trade execution metrics (total, wins, losses)
    - Financial metrics (capital, PnL, returns)
    - Risk metrics (DD, exposure, leverage)
    - System metrics (latency, health)
    """
    
    def __init__(self, 
                 registry: Optional[CollectorRegistry] = None,
                 pushgateway_url: Optional[str] = None):
        """
        Initialize metrics collector
        
        Args:
            registry: Prometheus registry (creates new if None)
            pushgateway_url: URL for Prometheus pushgateway (optional)
        """
        self.registry = registry or CollectorRegistry()
        self.pushgateway_url = pushgateway_url
        
        # Initialize metrics storage
        self.metrics = TradingMetrics()
        
        # === Trade Counters ===
        self.trades_total = Counter(
            'underdog_trades_total',
            'Total number of trades executed',
            ['symbol', 'side', 'result'],
            registry=self.registry
        )
        
        self.signals_total = Counter(
            'underdog_signals_total',
            'Total number of signals generated',
            ['strategy', 'action'],
            registry=self.registry
        )
        
        self.rejections_total = Counter(
            'underdog_rejections_total',
            'Total number of signals rejected by risk management',
            ['reason'],
            registry=self.registry
        )
        
        # === Financial Gauges ===
        self.current_capital = Gauge(
            'underdog_capital_usd',
            'Current account capital in USD',
            registry=self.registry
        )
        
        self.realized_pnl = Gauge(
            'underdog_realized_pnl_usd',
            'Realized profit/loss in USD',
            registry=self.registry
        )
        
        self.unrealized_pnl = Gauge(
            'underdog_unrealized_pnl_usd',
            'Unrealized profit/loss in USD',
            registry=self.registry
        )
        
        self.total_return = Gauge(
            'underdog_total_return_pct',
            'Total return percentage',
            registry=self.registry
        )
        
        # === Risk Gauges ===
        self.drawdown_pct = Gauge(
            'underdog_drawdown_pct',
            'Current drawdown percentage',
            ['timeframe'],  # daily, weekly, monthly
            registry=self.registry
        )
        
        self.max_drawdown_pct = Gauge(
            'underdog_max_drawdown_pct',
            'Maximum drawdown percentage',
            registry=self.registry
        )
        
        self.total_exposure = Gauge(
            'underdog_exposure_usd',
            'Total position exposure in USD',
            registry=self.registry
        )
        
        self.leverage_ratio = Gauge(
            'underdog_leverage_ratio',
            'Current leverage ratio',
            registry=self.registry
        )
        
        # === Position Gauges ===
        self.open_positions = Gauge(
            'underdog_open_positions',
            'Number of open positions',
            registry=self.registry
        )
        
        self.position_size = Gauge(
            'underdog_position_size_units',
            'Position size in units',
            ['symbol', 'side'],
            registry=self.registry
        )
        
        # === Strategy Gauges ===
        self.active_strategies = Gauge(
            'underdog_active_strategies',
            'Number of active strategies',
            registry=self.registry
        )
        
        self.strategy_confidence = Gauge(
            'underdog_strategy_confidence',
            'Strategy confidence score (0-1)',
            ['strategy'],
            registry=self.registry
        )
        
        self.regime_state = Gauge(
            'underdog_regime_state',
            'Current market regime (encoded)',
            registry=self.registry
        )
        
        # === Performance Metrics ===
        self.win_rate = Gauge(
            'underdog_win_rate_pct',
            'Win rate percentage',
            registry=self.registry
        )
        
        self.profit_factor = Gauge(
            'underdog_profit_factor',
            'Profit factor (gross profit / gross loss)',
            registry=self.registry
        )
        
        # === Execution Latency Histograms ===
        self.execution_latency = Histogram(
            'underdog_execution_latency_ms',
            'Order execution latency in milliseconds',
            buckets=[1, 5, 10, 25, 50, 100, 250, 500, 1000, 2500, 5000],
            registry=self.registry
        )
        
        self.signal_processing_time = Histogram(
            'underdog_signal_processing_ms',
            'Signal processing time in milliseconds',
            buckets=[0.1, 0.5, 1, 5, 10, 25, 50, 100],
            registry=self.registry
        )
        
        # === System Health ===
        self.system_health = Gauge(
            'underdog_system_health',
            'System health status (1=healthy, 0=unhealthy)',
            ['component'],
            registry=self.registry
        )
        
        self.kill_switch_active = Gauge(
            'underdog_kill_switch_active',
            'Kill switch status (1=active, 0=inactive)',
            registry=self.registry
        )
        
        # === Connection Status ===
        self.mt5_connected = Gauge(
            'underdog_mt5_connected',
            'MT5 connection status (1=connected, 0=disconnected)',
            registry=self.registry
        )
        
        self.zmq_connections = Gauge(
            'underdog_zmq_connections',
            'Number of active ZeroMQ connections',
            registry=self.registry
        )
    
    # === Trade Recording ===
    
    def record_trade(self, symbol: str, side: str, result: str, pnl: float):
        """
        Record a completed trade
        
        Args:
            symbol: Trading symbol (e.g., 'EURUSD')
            side: 'long' or 'short'
            result: 'win' or 'loss'
            pnl: Profit/loss in USD
        """
        self.trades_total.labels(symbol=symbol, side=side, result=result).inc()
        
        self.metrics.total_trades += 1
        if result == 'win':
            self.metrics.winning_trades += 1
        else:
            self.metrics.losing_trades += 1
        
        self.metrics.realized_pnl += pnl
        self.realized_pnl.set(self.metrics.realized_pnl)
        
        # Update win rate
        self.win_rate.set(self.metrics.win_rate())
    
    def record_signal(self, strategy: str, action: str):
        """Record a signal generation"""
        self.signals_total.labels(strategy=strategy, action=action).inc()
        self.metrics.signals_generated += 1
    
    def record_rejection(self, reason: str):
        """Record a signal rejection"""
        self.rejections_total.labels(reason=reason).inc()
        self.metrics.signals_rejected += 1
    
    # === Financial Updates ===
    
    def update_capital(self, capital: float):
        """Update current account capital"""
        self.metrics.current_capital = capital
        self.current_capital.set(capital)
        
        # Update total return
        total_return_pct = self.metrics.total_return()
        self.total_return.set(total_return_pct)
    
    def update_pnl(self, realized: float, unrealized: float):
        """Update PnL metrics"""
        self.metrics.realized_pnl = realized
        self.metrics.unrealized_pnl = unrealized
        
        self.realized_pnl.set(realized)
        self.unrealized_pnl.set(unrealized)
    
    # === Risk Updates ===
    
    def update_drawdown(self, current_dd: float, max_dd: float, daily_dd: float):
        """
        Update drawdown metrics
        
        Args:
            current_dd: Current DD percentage
            max_dd: Maximum DD percentage
            daily_dd: Daily DD percentage
        """
        self.metrics.current_drawdown = current_dd
        self.metrics.max_drawdown = max_dd
        self.metrics.daily_drawdown = daily_dd
        
        self.drawdown_pct.labels(timeframe='current').set(current_dd)
        self.drawdown_pct.labels(timeframe='daily').set(daily_dd)
        self.max_drawdown_pct.set(max_dd)
    
    def update_exposure(self, total_exposure: float, leverage: float):
        """Update exposure and leverage"""
        self.metrics.total_exposure = total_exposure
        self.total_exposure.set(total_exposure)
        self.leverage_ratio.set(leverage)
    
    # === Position Updates ===
    
    def update_positions(self, open_count: int):
        """Update open positions count"""
        self.metrics.open_positions = open_count
        self.open_positions.set(open_count)
    
    def update_position_size(self, symbol: str, side: str, size: float):
        """Update individual position size"""
        self.position_size.labels(symbol=symbol, side=side).set(size)
    
    # === Strategy Updates ===
    
    def update_active_strategies(self, count: int):
        """Update active strategies count"""
        self.metrics.active_strategies = count
        self.active_strategies.set(count)
    
    def update_strategy_confidence(self, strategy: str, confidence: float):
        """Update strategy confidence score"""
        self.strategy_confidence.labels(strategy=strategy).set(confidence)
    
    def update_regime(self, regime_code: int):
        """Update market regime (encoded as int)"""
        self.regime_state.set(regime_code)
    
    # === Execution Latency ===
    
    def record_execution_latency(self, latency_ms: float):
        """Record execution latency"""
        self.execution_latency.observe(latency_ms)
        self.metrics.last_execution_time_ms = latency_ms
    
    def record_signal_processing_time(self, time_ms: float):
        """Record signal processing time"""
        self.signal_processing_time.observe(time_ms)
    
    # === System Health ===
    
    def update_component_health(self, component: str, healthy: bool):
        """
        Update component health status
        
        Args:
            component: Component name (e.g., 'mt5', 'risk_master', 'model')
            healthy: True if healthy, False otherwise
        """
        self.system_health.labels(component=component).set(1 if healthy else 0)
    
    def update_kill_switch(self, active: bool):
        """Update kill switch status"""
        self.kill_switch_active.set(1 if active else 0)
    
    def update_mt5_connection(self, connected: bool):
        """Update MT5 connection status"""
        self.mt5_connected.set(1 if connected else 0)
    
    def update_zmq_connections(self, count: int):
        """Update ZeroMQ connections count"""
        self.zmq_connections.set(count)
    
    # === Export ===
    
    def export_metrics(self) -> str:
        """Export metrics in Prometheus format"""
        return generate_latest(self.registry).decode('utf-8')
    
    def push_metrics(self, job_name: str = 'underdog_trading'):
        """Push metrics to Prometheus pushgateway"""
        if self.pushgateway_url:
            push_to_gateway(self.pushgateway_url, job=job_name, registry=self.registry)
    
    def get_snapshot(self) -> TradingMetrics:
        """Get current metrics snapshot"""
        self.metrics.last_update = datetime.utcnow()
        return self.metrics


# === Context Manager for Timing ===

class Timer:
    """Context manager for timing operations"""
    
    def __init__(self, metrics_collector: MetricsCollector, metric_name: str):
        self.collector = metrics_collector
        self.metric_name = metric_name
        self.start_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed_ms = (time.time() - self.start_time) * 1000
        
        if self.metric_name == 'execution':
            self.collector.record_execution_latency(elapsed_ms)
        elif self.metric_name == 'signal_processing':
            self.collector.record_signal_processing_time(elapsed_ms)
