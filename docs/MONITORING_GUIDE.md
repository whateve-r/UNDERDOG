# Monitoring & Telemetry Guide

## Overview

The UNDERDOG monitoring system provides comprehensive observability through:
- **Prometheus metrics** for quantitative performance tracking
- **Health checks** for system component validation
- **Alerting** for critical events (Email, Slack, Telegram)
- **FastAPI dashboard** for real-time monitoring

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    UNDERDOG Trading System                   │
│                                                               │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │ Risk Master  │  │ MT5 Connector│  │Strategy Matrix│      │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘      │
│         │                  │                  │               │
│         └──────────────────┴──────────────────┘              │
│                            │                                  │
│                    ┌───────▼──────┐                          │
│                    │  Metrics     │                          │
│                    │  Collector   │                          │
│                    └───────┬──────┘                          │
└────────────────────────────┼─────────────────────────────────┘
                             │
            ┌────────────────┼────────────────┐
            │                │                │
     ┌──────▼──────┐  ┌──────▼──────┐ ┌──────▼──────┐
     │ Prometheus  │  │  FastAPI    │ │   Alert     │
     │   Server    │  │  Dashboard  │ │  Manager    │
     └──────┬──────┘  └──────┬──────┘ └──────┬──────┘
            │                │                │
     ┌──────▼──────┐  ┌──────▼──────┐ ┌──────▼──────┐
     │   Grafana   │  │   Browser   │ │Email/Slack/ │
     │  Dashboards │  │ /health API │ │  Telegram   │
     └─────────────┘  └─────────────┘ └─────────────┘
```

---

## Components

### 1. **MetricsCollector** (`underdog/monitoring/metrics.py`)

Tracks trading performance metrics using Prometheus format.

#### Key Metrics:

**Trade Counters:**
- `underdog_trades_total{symbol, side, result}` - Total trades
- `underdog_signals_total{strategy, action}` - Signals generated
- `underdog_rejections_total{reason}` - Signals rejected

**Financial Gauges:**
- `underdog_capital_usd` - Current capital
- `underdog_realized_pnl_usd` - Realized P&L
- `underdog_unrealized_pnl_usd` - Unrealized P&L
- `underdog_total_return_pct` - Total return percentage

**Risk Gauges:**
- `underdog_drawdown_pct{timeframe}` - Drawdown (daily/weekly/monthly)
- `underdog_max_drawdown_pct` - Maximum drawdown
- `underdog_exposure_usd` - Total position exposure
- `underdog_leverage_ratio` - Current leverage

**Execution Latency Histograms:**
- `underdog_execution_latency_ms` - Order execution time
- `underdog_signal_processing_ms` - Signal processing time

**System Health:**
- `underdog_system_health{component}` - Component health (1=healthy, 0=unhealthy)
- `underdog_kill_switch_active` - Kill switch status
- `underdog_mt5_connected` - MT5 connection status

#### Usage Example:

```python
from underdog.monitoring import MetricsCollector

# Initialize
metrics = MetricsCollector()

# Record trade
metrics.record_trade(
    symbol='EURUSD',
    side='long',
    result='win',
    pnl=150.0
)

# Update capital
metrics.update_capital(100150.0)

# Update drawdown
metrics.update_drawdown(
    current_dd=1.2,
    max_dd=2.5,
    daily_dd=0.8
)

# Record execution latency
from underdog.monitoring.metrics import Timer

with Timer(metrics, 'execution'):
    # Execute order
    place_order()
```

---

### 2. **HealthChecker** (`underdog/monitoring/health_check.py`)

Monitors system components and provides health status.

#### Checked Components:
- **MT5 Connection** - Latency, account info availability
- **ZeroMQ** - Publisher/subscriber health
- **Risk Master** - Kill switch, drawdown levels
- **ML Model** - Freshness (staleness check)
- **Database** - Connection latency
- **Custom checks** - User-defined health checks

#### Health Statuses:
- `HEALTHY` - Component operating normally
- `DEGRADED` - Component operational but degraded (high latency, warnings)
- `UNHEALTHY` - Component failure
- `UNKNOWN` - Unable to determine status

#### Usage Example:

```python
from underdog.monitoring import HealthChecker

# Initialize
health_checker = HealthChecker(
    check_interval=30.0,  # Check every 30 seconds
    model_staleness_threshold=86400.0  # 24 hours
)

# Register custom check
def check_strategy_matrix():
    return ('healthy', 'All strategies active', {'active': 3})

health_checker.register_check('strategy_matrix', check_strategy_matrix)

# Run all checks
system_health = health_checker.check_all(
    mt5_connector=mt5_conn,
    risk_master=risk_mgr,
    model_last_updated=model_timestamp
)

# Check if healthy
if not system_health.status == HealthStatus.HEALTHY:
    logger.warning("System degraded")
```

---

### 3. **AlertManager** (`underdog/monitoring/alerts.py`)

Sends notifications for critical events.

#### Alert Severities:
- `INFO` - Informational (system started, trade completed)
- `WARNING` - Potential issues (high rejection rate, model stale)
- `ERROR` - Serious problems (connection loss, execution errors)
- `CRITICAL` - System failures (kill switch, drawdown breach)

#### Alert Channels:
- **EMAIL** - SMTP email notifications
- **SLACK** - Slack webhook messages
- **TELEGRAM** - Telegram bot messages
- **WEBHOOK** - Custom webhooks
- **LOG** - Python logging

#### Usage Example:

```python
from underdog.monitoring import AlertManager, AlertSeverity

# Configure channels
alert_manager = AlertManager(
    email_config={
        'smtp_server': 'smtp.gmail.com',
        'port': 587,
        'username': 'your-email@gmail.com',
        'password': 'app-password',
        'to_address': 'alerts@your-domain.com'
    },
    slack_webhook='https://hooks.slack.com/services/YOUR/WEBHOOK/URL',
    cooldown_minutes=5.0  # Prevent spam
)

# Send custom alert
alert_manager.send_alert(
    severity=AlertSeverity.ERROR,
    title="Connection Lost",
    message="Lost connection to MT5 broker",
    metadata={'component': 'mt5'}
)

# Use convenience methods
alert_manager.alert_drawdown_breach(dd_pct=3.5, limit_pct=3.0, timeframe='daily')
alert_manager.alert_kill_switch_activated(reason="Daily DD limit exceeded")
alert_manager.alert_connection_loss(component='mt5')
```

---

### 4. **FastAPI Dashboard** (`underdog/monitoring/dashboard.py`)

HTTP API for monitoring and health checks.

#### Endpoints:

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | API information |
| GET | `/health` | System health check (503 if unhealthy) |
| GET | `/health?component=mt5` | Specific component health |
| GET | `/metrics` | Prometheus metrics (text format) |
| GET | `/stats` | Trading statistics (JSON) |
| POST | `/alerts/test` | Test alert system |
| GET | `/alerts/stats` | Alert statistics |

#### Usage Example:

```python
from underdog.monitoring.dashboard import MonitoringDashboard

# Create dashboard
dashboard = MonitoringDashboard(
    metrics_collector=metrics,
    health_checker=health_checker,
    alert_manager=alert_manager,
    host='0.0.0.0',
    port=8000
)

# Run server (blocking)
dashboard.run()
```

#### Testing Endpoints:

```bash
# Health check
curl http://localhost:8000/health

# Prometheus metrics
curl http://localhost:8000/metrics

# Trading stats
curl http://localhost:8000/stats

# Test alert
curl -X POST "http://localhost:8000/alerts/test?severity=warning&title=Test"
```

---

## Configuration

### 1. Email Alerts (Gmail Example)

```python
email_config = {
    'smtp_server': 'smtp.gmail.com',
    'port': 587,
    'username': 'your-email@gmail.com',
    'password': 'your-app-password',  # Generate at google.com/settings/security
    'to_address': 'alerts@your-domain.com'
}
```

### 2. Slack Alerts

1. Create Slack app: https://api.slack.com/apps
2. Enable "Incoming Webhooks"
3. Add webhook to workspace
4. Copy webhook URL

```python
# Replace with your actual Slack webhook URL
slack_webhook = 'https://hooks.slack.com/services/YOUR_WORKSPACE_ID/YOUR_CHANNEL_ID/YOUR_SECRET_TOKEN'
```

### 3. Telegram Alerts

1. Create bot: Talk to @BotFather on Telegram
2. Get bot token
3. Get chat ID: Send message to bot, then visit:
   `https://api.telegram.org/bot<TOKEN>/getUpdates`

```python
telegram_config = {
    'bot_token': '1234567890:ABCdefGHIjklMNOpqrsTUVwxyz',
    'chat_id': '123456789'
}
```

---

## Prometheus Integration

### 1. Install Prometheus

**Windows:**
```powershell
# Download from https://prometheus.io/download/
# Extract and run
.\prometheus.exe --config.file=prometheus.yml
```

**Linux/macOS:**
```bash
docker run -d -p 9090:9090 \
  -v $(pwd)/prometheus.yml:/etc/prometheus/prometheus.yml \
  prom/prometheus
```

### 2. Configure Prometheus (`prometheus.yml`)

```yaml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: 'underdog_trading'
    static_configs:
      - targets: ['localhost:8000']
    metrics_path: '/metrics'
```

### 3. Start UNDERDOG Dashboard

```python
# In your trading script
dashboard.run(host='0.0.0.0', port=8000)
```

### 4. Verify Scraping

Visit: `http://localhost:9090/targets`

---

## Grafana Dashboards

### 1. Install Grafana

```bash
docker run -d -p 3000:3000 \
  --name grafana \
  grafana/grafana
```

### 2. Add Prometheus Data Source

1. Login: `http://localhost:3000` (admin/admin)
2. Configuration → Data Sources → Add Prometheus
3. URL: `http://prometheus:9090` (or `http://host.docker.internal:9090` on Windows/Mac)

### 3. Import Dashboard

Create dashboard with panels:

**Trade Performance:**
- Win rate: `(underdog_trades_total{result="win"} / underdog_trades_total) * 100`
- Total trades: `sum(underdog_trades_total)`
- P&L: `underdog_realized_pnl_usd`

**Risk Metrics:**
- Current DD: `underdog_drawdown_pct{timeframe="current"}`
- Max DD: `underdog_max_drawdown_pct`
- Exposure: `underdog_exposure_usd`

**Execution Performance:**
- Execution latency (P95): `histogram_quantile(0.95, underdog_execution_latency_ms_bucket)`
- Avg latency: `rate(underdog_execution_latency_ms_sum[5m]) / rate(underdog_execution_latency_ms_count[5m])`

**System Health:**
- Components: `underdog_system_health`
- Kill switch: `underdog_kill_switch_active`
- MT5 connection: `underdog_mt5_connected`

---

## VPS Recommendations

### ❌ **MetaTrader VPS** - NOT Recommended for UNDERDOG

**Pros:**
- Ultra-low latency to broker (<1ms)
- Automatic MT5 terminal sync

**Cons:**
- Windows-only (limited Python support)
- Limited disk space (~10GB)
- No root/admin access
- Cannot run custom Python services (FastAPI, Prometheus)
- No Docker support
- Expensive for specs ($10-30/month for limited resources)

### ✅ **Recommended VPS Providers**

#### 1. **DigitalOcean** (Best for beginners)
- **Plan**: Basic Droplet
- **Specs**: 2 vCPU, 2GB RAM, 50GB SSD
- **Price**: $12/month
- **Location**: Choose closest to your broker (e.g., London for EU brokers)
- **OS**: Ubuntu 22.04 LTS

**Setup:**
```bash
# Create droplet
doctl compute droplet create underdog-prod \
  --image ubuntu-22-04-x64 \
  --size s-2vcpu-2gb \
  --region lon1

# SSH and install Docker
ssh root@your-droplet-ip
curl -fsSL https://get.docker.com | sh
```

#### 2. **Vultr** (Best latency)
- **Plan**: High Frequency
- **Specs**: 1 vCPU, 2GB RAM, 32GB SSD
- **Price**: $12/month
- **Locations**: 25+ worldwide (choose near broker)

#### 3. **Contabo** (Best price/performance)
- **Plan**: Cloud VPS S
- **Specs**: 4 vCPU, 8GB RAM, 200GB SSD
- **Price**: €5.99/month (~$6.50)
- **Location**: EU/US

#### 4. **AWS Lightsail** (Enterprise-grade)
- **Plan**: 2GB instance
- **Specs**: 1 vCPU, 2GB RAM, 60GB SSD
- **Price**: $10/month
- **Bonus**: Free backup snapshots

### Latency Comparison

| Provider | London → EU Broker | New York → US Broker |
|----------|-------------------|----------------------|
| DigitalOcean | 5-15ms | 1-5ms |
| Vultr | 3-10ms | 1-3ms |
| Contabo | 10-20ms | 15-25ms |
| AWS Lightsail | 5-15ms | 1-5ms |
| **MT5 VPS** | **<1ms** | **<1ms** |

**Note:** For algo trading, <50ms is acceptable. MT5 VPS advantage only matters for HFT (<10ms).

---

## Deployment Checklist

### Pre-Production

- [ ] Configure Prometheus scraping (15s interval)
- [ ] Set up Grafana dashboards
- [ ] Configure alert channels (Email/Slack/Telegram)
- [ ] Test health checks for all components
- [ ] Verify metrics are exported correctly
- [ ] Test alert cooldown logic

### Production

- [ ] Deploy to VPS (DigitalOcean/Vultr recommended)
- [ ] Set up Docker containers (see Docker guide)
- [ ] Configure firewall (allow 8000, 9090, 3000)
- [ ] Set up SSL/TLS for FastAPI (Let's Encrypt)
- [ ] Configure log rotation
- [ ] Set up backup cron jobs
- [ ] Test failover scenarios

### Monitoring

- [ ] Monitor Prometheus disk usage (retention: 15d)
- [ ] Set up Grafana alert rules
- [ ] Configure Uptime Kuma for dashboard monitoring
- [ ] Set up PagerDuty/Opsgenie for critical alerts

---

## Troubleshooting

### Issue: Prometheus not scraping metrics

**Solution:**
```bash
# Check if dashboard is accessible
curl http://localhost:8000/metrics

# Check Prometheus targets
curl http://localhost:9090/api/v1/targets

# Restart Prometheus
docker restart prometheus
```

### Issue: Alerts not sending

**Solution:**
```python
# Test email configuration
import smtplib
server = smtplib.SMTP('smtp.gmail.com', 587)
server.starttls()
server.login('user', 'password')  # Should not raise error

# Test Slack webhook
import requests
requests.post(webhook_url, json={'text': 'test'})

# Check alert cooldown
print(alert_manager.alert_history)  # Should be empty after cooldown
```

### Issue: Health checks failing

**Solution:**
```python
# Check each component individually
health = health_checker.check_mt5_connection(mt5_conn)
print(health.to_dict())

# Enable debug logging
import logging
logging.basicConfig(level=logging.DEBUG)
```

---

## Example Integration

See `scripts/monitoring_example.py` for complete integration example.

---

## Next Steps

1. **Complete Docker Setup** - Containerize all services
2. **Database Integration** - Store metrics in TimescaleDB
3. **Advanced Grafana** - Create custom dashboards
4. **CI/CD Pipeline** - Automated deployment

---

## Support

For issues or questions:
- Check logs: `underdog/logs/monitoring.log`
- GitHub Issues: https://github.com/whateve-r/UNDERDOG/issues
- Documentation: `docs/MONITORING_GUIDE.md`
