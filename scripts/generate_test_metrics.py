"""
Generate Test Metrics for Prometheus Dashboard Testing
========================================================

This script simulates 7 EAs generating trading signals and metrics
to test Grafana dashboards WITHOUT waiting for real trading activity.

Features:
- Simulates 7 EAs with different characteristics
- Generates realistic metrics (signals, P&L, execution times)
- Updates Prometheus every 1 second
- Runs indefinitely until Ctrl+C

Usage:
    poetry run python scripts/generate_test_metrics.py

Then check:
    - Prometheus metrics: http://localhost:8000/metrics
    - Grafana dashboards: http://localhost:3000
"""

import time
import random
import sys
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from underdog.monitoring.prometheus_metrics import (
    start_metrics_server,
    update_account_metrics,
    update_drawdown,
    update_broker_connection,
    ea_status,
    ea_signals_total,
    ea_execution_time_ms,
    ea_positions_open,
    ea_pnl_unrealized,
    ea_pnl_realized,
    ea_pnl_daily,
    ea_win_rate,
    ea_sharpe_ratio,
    ea_profit_factor,
    set_ea_info
)

print("=" * 80)
print("🧪 UNDERDOG - Test Metrics Generator")
print("=" * 80)
print("This script generates synthetic Prometheus metrics for dashboard testing")
print("Press Ctrl+C to stop")
print("=" * 80 + "\n")

# =============================================================================
# EA CONFIGURATIONS
# =============================================================================

EAS_CONFIG = [
    {
        "name": "SuperTrendRSI",
        "symbol": "EURUSD",
        "timeframe": "M15",
        "confidence": 1.0,
        "signal_rate": 0.15,  # 15% probability per update
        "avg_execution_ms": 5.2,
        "win_rate": 0.68,
        "sharpe": 1.85,
        "profit_factor": 1.92
    },
    {
        "name": "ParabolicEMA",
        "symbol": "GBPUSD",
        "timeframe": "M15",
        "confidence": 0.95,
        "signal_rate": 0.12,
        "avg_execution_ms": 4.8,
        "win_rate": 0.64,
        "sharpe": 1.72,
        "profit_factor": 1.78
    },
    {
        "name": "KeltnerBreakout",
        "symbol": "USDJPY",
        "timeframe": "M15",
        "confidence": 0.90,
        "signal_rate": 0.08,
        "avg_execution_ms": 6.1,
        "win_rate": 0.61,
        "sharpe": 1.55,
        "profit_factor": 1.65
    },
    {
        "name": "EmaScalper",
        "symbol": "EURJPY",
        "timeframe": "M5",
        "confidence": 0.85,
        "signal_rate": 0.25,  # Scalper = more signals
        "avg_execution_ms": 3.5,
        "win_rate": 0.58,
        "sharpe": 1.42,
        "profit_factor": 1.52
    },
    {
        "name": "BollingerCCI",
        "symbol": "AUDUSD",
        "timeframe": "M15",
        "confidence": 0.88,
        "signal_rate": 0.10,
        "avg_execution_ms": 5.5,
        "win_rate": 0.62,
        "sharpe": 1.68,
        "profit_factor": 1.72
    },
    {
        "name": "ATRBreakout",
        "symbol": "USDCAD",
        "timeframe": "M15",
        "confidence": 0.87,
        "signal_rate": 0.09,
        "avg_execution_ms": 7.2,
        "win_rate": 0.60,
        "sharpe": 1.50,
        "profit_factor": 1.58
    },
    {
        "name": "PairArbitrage",
        "symbol": "EURUSD_GBPUSD",
        "timeframe": "M15",
        "confidence": 0.92,
        "signal_rate": 0.05,  # Arbitrage = rare but high quality
        "avg_execution_ms": 8.5,
        "win_rate": 0.72,
        "sharpe": 2.05,
        "profit_factor": 2.15
    }
]

# =============================================================================
# SIMULATION STATE
# =============================================================================

class SimulationState:
    def __init__(self):
        self.account_balance = 100000.0
        self.account_equity = 100000.0
        self.peak_equity = 100000.0
        self.daily_start_equity = 100000.0
        
        # Per-EA state
        self.ea_state = {}
        for ea in EAS_CONFIG:
            self.ea_state[ea["name"]] = {
                "total_signals": 0,
                "open_positions": 0,
                "unrealized_pnl": 0.0,
                "realized_pnl": 0.0,
                "daily_pnl": 0.0,
                "last_signal_time": None
            }
    
    def update_account(self, pnl_change: float):
        """Update account with P&L change"""
        self.account_balance += pnl_change
        
        # Update equity (balance + unrealized P&L)
        total_unrealized = sum(s["unrealized_pnl"] for s in self.ea_state.values())
        self.account_equity = self.account_balance + total_unrealized
        
        # Update peak
        if self.account_equity > self.peak_equity:
            self.peak_equity = self.account_equity
    
    def get_drawdown(self):
        """Calculate current drawdown"""
        # Daily DD
        daily_dd_pct = ((self.account_equity - self.daily_start_equity) / self.daily_start_equity) * 100
        daily_dd_usd = self.account_equity - self.daily_start_equity
        
        # Total DD
        total_dd_pct = ((self.account_equity - self.peak_equity) / self.peak_equity) * 100
        total_dd_usd = self.account_equity - self.peak_equity
        
        return daily_dd_pct, total_dd_pct, daily_dd_usd, total_dd_usd

# =============================================================================
# MAIN SIMULATION LOOP
# =============================================================================

def main():
    # Start Prometheus metrics server
    print("🚀 Starting Prometheus metrics server on port 8000...")
    start_metrics_server(port=8000)
    time.sleep(2)
    print("✅ Metrics server running: http://localhost:8000/metrics\n")
    
    # Initialize simulation state
    state = SimulationState()
    
    # Initialize EAs
    print("📊 Initializing 7 EAs...")
    for ea_config in EAS_CONFIG:
        # Set initial status to active (only ea_name label)
        ea_status.labels(ea_name=ea_config["name"]).set(1)
        
        print(f"  ✓ {ea_config['name']} ({ea_config['symbol']} {ea_config['timeframe']}) - Confidence: {ea_config['confidence']}")
    
    print(f"\n{'='*80}")
    print("▶️  Simulation started - Generating metrics every 1 second")
    print(f"{'='*80}\n")
    
    iteration = 0
    
    try:
        while True:
            iteration += 1
            current_time = datetime.now().strftime("%H:%M:%S")
            
            # Update broker connection
            update_broker_connection("MetaQuotes-Demo", "10007888612", True)
            
            # Update each EA
            for ea_config in EAS_CONFIG:
                ea_name = ea_config["name"]
                ea_info = state.ea_state[ea_name]
                
                # Generate signal randomly
                if random.random() < ea_config["signal_rate"]:
                    signal_type = random.choice(["BUY", "SELL"])
                    
                    # Record signal (ea_name, signal_type, symbol)
                    ea_signals_total.labels(
                        ea_name=ea_name,
                        signal_type=signal_type,
                        symbol=ea_config["symbol"]
                    ).inc()
                    
                    ea_info["total_signals"] += 1
                    
                    # Maybe open position (50% chance)
                    if random.random() < 0.5 and ea_info["open_positions"] < 2:
                        ea_info["open_positions"] += 1
                        
                        # Generate unrealized P&L
                        ea_info["unrealized_pnl"] = random.uniform(-100, 200)
                        
                        print(f"[{current_time}] {ea_name}: {signal_type} signal → Position opened")
                
                # Maybe close position
                if ea_info["open_positions"] > 0 and random.random() < 0.1:
                    # Realize P&L
                    win = random.random() < ea_config["win_rate"]
                    
                    if win:
                        pnl = random.uniform(50, 150)
                    else:
                        pnl = random.uniform(-80, -20)
                    
                    ea_info["realized_pnl"] += pnl
                    ea_info["daily_pnl"] += pnl
                    ea_info["open_positions"] -= 1
                    ea_info["unrealized_pnl"] = 0.0
                    
                    state.update_account(pnl)
                    
                    print(f"[{current_time}] {ea_name}: Position closed → P&L: ${pnl:.2f}")
                
                # Update metrics (confidence is set via set_ea_info at startup)
                
                # Execution time with some variance
                exec_time = ea_config["avg_execution_ms"] + random.uniform(-1.0, 1.0)
                ea_execution_time_ms.labels(ea_name=ea_name).observe(exec_time)
                
                # Positions (ea_name, symbol)
                ea_positions_open.labels(
                    ea_name=ea_name,
                    symbol=ea_config["symbol"]
                ).set(ea_info["open_positions"])
                
                # P&L metrics (only ea_name)
                ea_pnl_unrealized.labels(ea_name=ea_name).set(ea_info["unrealized_pnl"])
                
                ea_pnl_realized.labels(ea_name=ea_name).set(ea_info["realized_pnl"])
                
                ea_pnl_daily.labels(ea_name=ea_name).set(ea_info["daily_pnl"])
                
                # Performance metrics (only ea_name)
                ea_win_rate.labels(ea_name=ea_name).set(ea_config["win_rate"])
                
                ea_sharpe_ratio.labels(ea_name=ea_name).set(ea_config["sharpe"])
                
                ea_profit_factor.labels(ea_name=ea_name).set(ea_config["profit_factor"])
            
            # Update account metrics
            update_account_metrics(
                broker="MetaQuotes-Demo",
                account_id="10007888612",
                balance=state.account_balance,
                equity=state.account_equity,
                margin_used=state.account_equity * 0.02,  # 2% margin used
                margin_free=state.account_equity * 0.98
            )
            
            # Update drawdown
            daily_dd_pct, total_dd_pct, daily_dd_usd, total_dd_usd = state.get_drawdown()
            update_drawdown(daily_dd_pct, total_dd_pct, daily_dd_usd, total_dd_usd)
            
            # Print summary every 10 iterations
            if iteration % 10 == 0:
                total_signals = sum(s["total_signals"] for s in state.ea_state.values())
                total_positions = sum(s["open_positions"] for s in state.ea_state.values())
                total_daily_pnl = sum(s["daily_pnl"] for s in state.ea_state.values())
                
                print(f"\n{'─'*80}")
                print(f"[{current_time}] Iteration {iteration}")
                print(f"  Account: ${state.account_equity:,.2f} | DD: {total_dd_pct:.2f}%")
                print(f"  Signals: {total_signals} | Positions: {total_positions} | Daily P&L: ${total_daily_pnl:.2f}")
                print(f"{'─'*80}\n")
            
            # Wait 1 second
            time.sleep(1)
    
    except KeyboardInterrupt:
        print(f"\n\n{'='*80}")
        print("⏹️  Simulation stopped by user")
        print(f"{'='*80}")
        print(f"Final Statistics:")
        print(f"  Iterations: {iteration}")
        print(f"  Final Equity: ${state.account_equity:,.2f}")
        print(f"  Total Signals: {sum(s['total_signals'] for s in state.ea_state.values())}")
        print(f"  Final P&L: ${state.account_equity - 100000:.2f}")
        print(f"{'='*80}\n")

if __name__ == "__main__":
    main()
