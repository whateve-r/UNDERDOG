"""
Complete Integration Example - UNDERDOG Trading System
Demonstrates how all components work together.
"""
import asyncio
from datetime import datetime
from typing import Dict, Any

# Core connectivity
from underdog.core.connectors.mt5_connector import Mt5Connector
from underdog.core.schemas.zmq_messages import MessageFactory, TickMessage, OHLCVMessage

# Risk management
from underdog.risk_management.risk_master import RiskMaster, DrawdownLimits, ExposureLimits
from underdog.risk_management.position_sizing import PositionSizer, SizingConfig

# Strategy coordination
from underdog.strategies.strategy_matrix import StrategyMatrix, StrategySignal

# Fuzzy logic
from underdog.strategies.fuzzy_logic.mamdani_inference import ConfidenceScorer


class UnderdogTradingSystem:
    """
    Main trading system that integrates all UNDERDOG components.
    """
    
    def __init__(self, initial_capital: float = 100000):
        """
        Initialize the complete trading system.
        
        Args:
            initial_capital: Starting account balance
        """
        # MT5 Connector
        self.connector = Mt5Connector()
        
        # Risk Management Configuration
        dd_limits = DrawdownLimits(
            max_daily_dd_pct=5.0,      # Prop firm typical: 5%
            max_weekly_dd_pct=10.0,
            max_monthly_dd_pct=15.0,
            max_absolute_dd_pct=20.0,
            soft_limit_pct=0.8         # Start scaling at 80% of limit
        )
        
        exposure_limits = ExposureLimits(
            max_total_exposure_pct=100.0,
            max_per_symbol_pct=10.0,
            max_per_strategy_pct=30.0,
            max_correlated_exposure_pct=40.0,
            max_leverage=2.0
        )
        
        # Initialize Risk Master
        self.risk_master = RiskMaster(
            initial_capital=initial_capital,
            dd_limits=dd_limits,
            exposure_limits=exposure_limits,
            correlation_window=21
        )
        
        # Position Sizing Configuration
        sizing_config = SizingConfig(
            fixed_risk_pct=1.5,
            kelly_fraction=0.2,
            kelly_cap=0.25,
            min_confidence=0.6,
            confidence_exponent=1.0,
            use_confidence_scaling=True,
            use_kelly=True
        )
        
        self.position_sizer = PositionSizer(sizing_config)
        
        # Fuzzy Logic Confidence Scorer
        fuzzy_rules_path = "config/strategies/fuzzy_confidence_rules.yaml"
        self.confidence_scorer = ConfidenceScorer(rules_path=fuzzy_rules_path)
        
        # Strategy Matrix
        self.strategy_matrix = StrategyMatrix(
            risk_master=self.risk_master,
            position_sizer=self.position_sizer,
            confidence_scorer=self.confidence_scorer
        )
        
        # Register strategies
        self._register_strategies()
        
        # Market data buffer
        self.market_data: Dict[str, Any] = {}
        
        print("[UNDERDOG] Trading system initialized")
    
    def _register_strategies(self):
        """Register all active trading strategies"""
        self.strategy_matrix.register_strategy(
            strategy_id="keltner_breakout",
            allocation_pct=20.0,
            priority=1,
            meta={"type": "momentum", "timeframe": "M15"}
        )
        
        self.strategy_matrix.register_strategy(
            strategy_id="pairs_trading",
            allocation_pct=15.0,
            priority=2,
            meta={"type": "mean_reversion", "timeframe": "M5"}
        )
        
        self.strategy_matrix.register_strategy(
            strategy_id="ml_lstm",
            allocation_pct=25.0,
            priority=1,
            meta={"type": "ml", "model": "lstm"}
        )
        
        print("[UNDERDOG] Strategies registered: 3 active")
    
    async def start(self):
        """Start the trading system"""
        # Connect to MT5
        if not await self.connector.connect():
            print("[UNDERDOG] Failed to connect to MT5. Exiting.")
            return
        
        print("[UNDERDOG] Connected to MT5")
        
        # Get initial account info
        account_info = await self.connector.get_account_info()
        if account_info:
            self.risk_master.update_capital(account_info.equity)
            print(f"[UNDERDOG] Account: {account_info.name} | Equity: ${account_info.equity:,.2f}")
        
        # Start market data listeners and strategy loop
        await asyncio.gather(
            self.connector.listen_live(self.handle_live_data),
            self.connector.listen_stream(self.handle_stream_data),
            self.strategy_loop()
        )
    
    async def handle_live_data(self, data: Dict[str, Any]):
        """Handle account and position updates"""
        msg = MessageFactory.parse(data)
        
        if hasattr(msg, 'type'):
            if msg.type == 'account_update':
                self.risk_master.update_capital(msg.equity)
                print(f"[LIVE] Account update: Equity=${msg.equity:,.2f}, DD={self.risk_master.get_current_drawdown_pct():.2f}%")
            
            elif msg.type == 'position_close':
                # Update strategy P&L
                strategy_id = msg.meta.get('strategy') if hasattr(msg, 'meta') else None
                if strategy_id:
                    self.risk_master.update_daily_pnl(msg.profit, strategy_id)
                print(f"[LIVE] Position closed: {msg.symbol} P&L=${msg.profit:,.2f}")
    
    async def handle_stream_data(self, data: Dict[str, Any]):
        """Handle market data stream"""
        msg = MessageFactory.parse(data)
        
        if hasattr(msg, 'type'):
            if msg.type == 'tick':
                self.market_data[msg.symbol] = {
                    'bid': msg.bid,
                    'ask': msg.ask,
                    'timestamp': msg.ts
                }
            
            elif msg.type == 'ohlcv':
                # Store OHLCV for strategy calculations
                if msg.symbol not in self.market_data:
                    self.market_data[msg.symbol] = {}
                
                self.market_data[msg.symbol]['ohlcv'] = {
                    'open': msg.open,
                    'high': msg.high,
                    'low': msg.low,
                    'close': msg.close,
                    'volume': msg.volume,
                    'timestamp': msg.ts
                }
    
    async def strategy_loop(self):
        """Main strategy execution loop"""
        print("[UNDERDOG] Strategy loop started")
        
        while True:
            try:
                # Example: Generate signals from strategies
                # In production, strategies would analyze market_data and generate signals
                
                # Simulate a signal from keltner_breakout strategy
                if "EURUSD" in self.market_data and 'bid' in self.market_data["EURUSD"]:
                    current_price = self.market_data["EURUSD"]['bid']
                    
                    # Mock ML model output and indicators
                    ml_probability = 0.78  # From your ML model
                    atr_ratio = 1.1        # Current ATR / Historical ATR
                    momentum = 0.5         # Normalized momentum
                    
                    # Calculate confidence using fuzzy logic
                    confidence = self.confidence_scorer.score(
                        ml_probability=ml_probability,
                        atr_ratio=atr_ratio,
                        momentum=momentum
                    )
                    
                    # Create signal
                    signal = StrategySignal(
                        strategy_id="keltner_breakout",
                        symbol="EURUSD",
                        side="buy",
                        entry_price=current_price,
                        stop_loss=current_price - 0.0030,  # 30 pips
                        take_profit=current_price + 0.0060,  # 60 pips (2:1 R:R)
                        confidence_score=confidence,
                        size_suggestion=0.1,
                        timestamp=datetime.now()
                    )
                    
                    # Submit to strategy matrix
                    self.strategy_matrix.submit_signal(signal)
                    
                    # Process signals
                    aggregated_signals = self.strategy_matrix.process_signals()
                    
                    # Execute approved signals
                    for agg_sig in aggregated_signals:
                        if agg_sig.approved:
                            await self.execute_order(agg_sig)
                        else:
                            print(f"[STRATEGY] Signal rejected: {agg_sig.rejection_reason}")
                
                # Portfolio maintenance
                await self.perform_maintenance()
                
                # Sleep before next iteration
                await asyncio.sleep(10)
                
            except Exception as e:
                print(f"[ERROR] Strategy loop exception: {e}")
                await asyncio.sleep(5)
    
    async def execute_order(self, signal):
        """Execute an approved aggregated signal"""
        print(f"[EXECUTE] {signal.side.upper()} {signal.symbol} "
              f"Size={signal.final_size:.2f} "
              f"Entry={signal.entry_price:.5f} "
              f"SL={signal.stop_loss:.5f} "
              f"Confidence={signal.combined_confidence:.2f}")
        
        # Create order using MessageFactory
        order = MessageFactory.create_order(
            symbol=signal.symbol,
            side=signal.side,
            size=signal.final_size,
            action="market",
            sl=signal.stop_loss,
            tp=signal.take_profit,
            strategy=",".join(signal.participating_strategies),
            confidence=signal.combined_confidence
        )
        
        # Send order to MT5
        response = await self.connector.sys_request(order.to_dict())
        
        if response:
            print(f"[EXECUTE] Order sent. Response: {response}")
        else:
            print(f"[EXECUTE] Order failed to send")
    
    async def perform_maintenance(self):
        """Perform periodic portfolio maintenance"""
        # Check if we need to reset daily metrics (new trading day)
        current_date = datetime.now().date()
        if current_date > self.risk_master.last_daily_reset:
            self.risk_master.reset_daily_metrics()
            print(f"[MAINTENANCE] Daily metrics reset for {current_date}")
        
        # Update correlation matrix
        self.risk_master.update_correlation_matrix()
        
        # Rebalance strategy allocations (e.g., weekly)
        # self.strategy_matrix.rebalance_allocations()
        
        # Get portfolio status
        status = self.strategy_matrix.get_portfolio_status()
        
        if status['risk_metrics']['is_trading_halted']:
            print(f"[WARNING] Trading is HALTED: {status['risk_metrics']['halt_reason']}")
    
    async def shutdown(self):
        """Graceful shutdown"""
        print("[UNDERDOG] Shutting down...")
        await self.connector.disconnect()
        print("[UNDERDOG] Shutdown complete")


# ========================================
# Main Entry Point
# ========================================

async def main():
    """Main entry point"""
    system = UnderdogTradingSystem(initial_capital=100000)
    
    try:
        await system.start()
    except KeyboardInterrupt:
        print("\n[UNDERDOG] Keyboard interrupt received")
    finally:
        await system.shutdown()


if __name__ == '__main__':
    import os
    
    # Windows event loop policy fix
    if os.name == 'nt':
        try:
            from asyncio import WindowsSelectorEventLoopPolicy
            asyncio.set_event_loop_policy(WindowsSelectorEventLoopPolicy())
        except ImportError:
            pass
    
    print("=" * 60)
    print("UNDERDOG - Algorithmic Trading System")
    print("=" * 60)
    print()
    
    asyncio.run(main())
