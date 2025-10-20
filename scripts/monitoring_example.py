"""
Example: Monitoring System Integration
Demonstrates how to use Prometheus metrics, health checks, and alerting.
"""
import time
from datetime import datetime, timedelta
from underdog.monitoring import (
    MetricsCollector, HealthChecker, AlertManager,
    AlertSeverity, AlertChannel
)
from underdog.monitoring.dashboard import MonitoringDashboard


def main():
    """
    Example monitoring integration
    """
    print("=" * 80)
    print(" UNDERDOG - MONITORING SYSTEM EXAMPLE")
    print("=" * 80)
    print()
    
    # ===================================================================
    # STEP 1: Initialize Metrics Collector
    # ===================================================================
    print("STEP 1: Initialize Metrics Collector")
    print("-" * 80)
    
    metrics = MetricsCollector()
    
    # Set initial capital
    metrics.update_capital(100000.0)
    print(f"✓ Initial capital set: $100,000")
    
    # ===================================================================
    # STEP 2: Record Some Trades
    # ===================================================================
    print("\nSTEP 2: Record Trades")
    print("-" * 80)
    
    # Winning trade
    metrics.record_trade(
        symbol='EURUSD',
        side='long',
        result='win',
        pnl=150.0
    )
    print(f"✓ Trade #1: EURUSD LONG +$150 (WIN)")
    
    # Losing trade
    metrics.record_trade(
        symbol='GBPUSD',
        side='short',
        result='loss',
        pnl=-75.0
    )
    print(f"✓ Trade #2: GBPUSD SHORT -$75 (LOSS)")
    
    # Update capital
    new_capital = 100000 + 150 - 75
    metrics.update_capital(new_capital)
    print(f"✓ Updated capital: ${new_capital:,.2f}")
    
    # ===================================================================
    # STEP 3: Update Risk Metrics
    # ===================================================================
    print("\nSTEP 3: Update Risk Metrics")
    print("-" * 80)
    
    metrics.update_drawdown(
        current_dd=1.2,
        max_dd=2.5,
        daily_dd=0.8
    )
    print(f"✓ Drawdown: Current=1.2%, Max=2.5%, Daily=0.8%")
    
    metrics.update_exposure(
        total_exposure=15000.0,
        leverage=1.5
    )
    print(f"✓ Exposure: $15,000 (1.5x leverage)")
    
    # ===================================================================
    # STEP 4: Record Signals and Rejections
    # ===================================================================
    print("\nSTEP 4: Record Signals")
    print("-" * 80)
    
    metrics.record_signal(strategy='keltner_breakout', action='long')
    metrics.record_signal(strategy='fuzzy_confidence', action='short')
    metrics.record_rejection(reason='high_drawdown')
    
    print(f"✓ Signals generated: 2")
    print(f"✓ Signals rejected: 1")
    
    # ===================================================================
    # STEP 5: Record Execution Latency
    # ===================================================================
    print("\nSTEP 5: Record Execution Latency")
    print("-" * 80)
    
    from underdog.monitoring.metrics import Timer
    
    with Timer(metrics, 'execution'):
        time.sleep(0.015)  # Simulate 15ms execution
    
    print(f"✓ Execution latency: 15ms (simulated)")
    
    # ===================================================================
    # STEP 6: Export Prometheus Metrics
    # ===================================================================
    print("\nSTEP 6: Export Prometheus Metrics")
    print("-" * 80)
    
    prometheus_output = metrics.export_metrics()
    
    print("Sample Prometheus metrics:")
    print(prometheus_output[:500] + "...\n")
    
    # ===================================================================
    # STEP 7: Initialize Health Checker
    # ===================================================================
    print("\nSTEP 7: Health Checker")
    print("-" * 80)
    
    health_checker = HealthChecker(
        check_interval=30.0,
        model_staleness_threshold=86400.0  # 24 hours
    )
    
    # Register custom health check
    def custom_strategy_check():
        """Custom health check for strategy matrix"""
        return (
            'healthy',
            'All strategies operational',
            {'active_strategies': 3}
        )
    
    health_checker.register_check('strategy_matrix', custom_strategy_check)
    print(f"✓ Health checker initialized")
    print(f"✓ Custom check registered: strategy_matrix")
    
    # Mock components for health check
    class MockMT5:
        connected = True
        def get_account_info(self):
            class Info:
                balance = 100075.0
                equity = 100100.0
                server = "BrokerDemo-Server"
            return Info()
    
    class MockRiskMaster:
        kill_switch_active = False
        current_drawdown_pct = 1.2
        daily_drawdown_pct = 0.8
        max_drawdown_pct = 10.0
        positions = []
    
    # Check health
    system_health = health_checker.check_all(
        mt5_connector=MockMT5(),
        risk_master=MockRiskMaster(),
        model_last_updated=datetime.utcnow() - timedelta(hours=2)
    )
    
    print(f"\nSystem Health: {system_health.status.value.upper()}")
    print(f"Uptime: {system_health.uptime_seconds:.1f}s")
    print(f"\nComponent Status:")
    for comp in system_health.components:
        status_symbol = "✓" if comp.status.value == "healthy" else "⚠"
        print(f"  {status_symbol} {comp.name}: {comp.status.value} - {comp.message}")
    
    # ===================================================================
    # STEP 8: Initialize Alert Manager
    # ===================================================================
    print("\n\nSTEP 8: Alert Manager")
    print("-" * 80)
    
    # Configure alerts (Email/Slack/Telegram - use LOG for demo)
    alert_manager = AlertManager(
        cooldown_minutes=5.0
    )
    
    print(f"✓ Alert manager initialized")
    
    # Send test alerts
    alert_manager.send_alert(
        severity=AlertSeverity.INFO,
        title="System Started",
        message="UNDERDOG trading system initialized successfully",
        channels=[AlertChannel.LOG]
    )
    print(f"✓ INFO alert sent: System Started")
    
    # Simulate drawdown alert
    alert_manager.alert_drawdown_breach(
        dd_pct=3.5,
        limit_pct=3.0,
        timeframe='daily'
    )
    print(f"✓ CRITICAL alert sent: Drawdown Breach")
    
    # Get alert stats
    stats = alert_manager.get_alert_stats()
    print(f"\nAlert Statistics:")
    print(f"  Total: {stats['total_alerts']}")
    print(f"  By Severity:")
    for severity, count in stats['by_severity'].items():
        if count > 0:
            print(f"    {severity}: {count}")
    
    # ===================================================================
    # STEP 9: Trading Metrics Summary
    # ===================================================================
    print("\n\nSTEP 9: Trading Metrics Summary")
    print("-" * 80)
    
    snapshot = metrics.get_snapshot()
    
    print(f"\nTrade Performance:")
    print(f"  Total Trades: {snapshot.total_trades}")
    print(f"  Wins: {snapshot.winning_trades}")
    print(f"  Losses: {snapshot.losing_trades}")
    print(f"  Win Rate: {snapshot.win_rate():.1f}%")
    
    print(f"\nFinancial Metrics:")
    print(f"  Current Capital: ${snapshot.current_capital:,.2f}")
    print(f"  Realized P&L: ${snapshot.realized_pnl:,.2f}")
    print(f"  Total Return: {snapshot.total_return():.2f}%")
    
    print(f"\nRisk Metrics:")
    print(f"  Current DD: {snapshot.current_drawdown:.2f}%")
    print(f"  Max DD: {snapshot.max_drawdown:.2f}%")
    print(f"  Exposure: ${snapshot.total_exposure:,.2f}")
    
    print(f"\nSignal Metrics:")
    print(f"  Signals Generated: {snapshot.signals_generated}")
    print(f"  Signals Rejected: {snapshot.signals_rejected}")
    rejection_rate = (snapshot.signals_rejected / max(snapshot.signals_generated, 1)) * 100
    print(f"  Rejection Rate: {rejection_rate:.1f}%")
    
    # ===================================================================
    # STEP 10: FastAPI Dashboard (Optional)
    # ===================================================================
    print("\n\nSTEP 10: FastAPI Dashboard")
    print("-" * 80)
    
    print(f"\nTo start the monitoring dashboard:")
    print(f"  1. Create dashboard: dashboard = MonitoringDashboard(metrics, health_checker, alert_manager)")
    print(f"  2. Run server: dashboard.run(host='0.0.0.0', port=8000)")
    print(f"\nEndpoints:")
    print(f"  GET  http://localhost:8000/health  - Health check")
    print(f"  GET  http://localhost:8000/metrics - Prometheus metrics")
    print(f"  GET  http://localhost:8000/stats   - Trading statistics")
    print(f"  POST http://localhost:8000/alerts/test - Test alerts")
    
    print("\n" + "=" * 80)
    print(" MONITORING SYSTEM DEMO COMPLETE")
    print("=" * 80)
    print()
    print("Next steps:")
    print("  1. Configure email/Slack/Telegram in AlertManager")
    print("  2. Set up Prometheus scraping (scrape_interval: 15s)")
    print("  3. Import Grafana dashboards for visualization")
    print("  4. Deploy monitoring dashboard to production VPS")
    print()


if __name__ == '__main__':
    main()
