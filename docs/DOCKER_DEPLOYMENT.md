# =============================================================================
# UNDERDOG Trading System - Docker Deployment Guide
# Complete instructions for production deployment
# =============================================================================

## Prerequisites

### 1. Install Docker & Docker Compose

**Windows:**
```powershell
# Install Docker Desktop
# Download from: https://www.docker.com/products/docker-desktop

# Verify installation
docker --version
docker-compose --version
```

**Linux (Ubuntu/Debian):**
```bash
# Install Docker
curl -fsSL https://get.docker.com | sh

# Install Docker Compose
sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose

# Add user to docker group
sudo usermod -aG docker $USER
newgrp docker

# Verify
docker --version
docker-compose --version
```

**macOS:**
```bash
# Install Docker Desktop
# Download from: https://www.docker.com/products/docker-desktop

# Or use Homebrew
brew install --cask docker
```

---

## Quick Start

### 1. Clone and Configure

```bash
cd c:\Users\manud\OneDrive\Escritorio\tfg\UNDERDOG

# Copy environment template
cp docker/.env.template docker/.env

# Edit .env with your credentials
notepad docker/.env
```

**Required Variables:**
- `MT5_LOGIN` - Your MT5 account number
- `MT5_PASSWORD` - Your MT5 password
- `MT5_SERVER` - Your broker's server (e.g., "ICMarkets-Demo02")
- `DB_PASSWORD` - Strong password for PostgreSQL
- `GRAFANA_PASSWORD` - Strong password for Grafana

### 2. Build and Start

```bash
# Navigate to docker directory
cd docker

# Build images
docker-compose build

# Start all services
docker-compose up -d

# Check status
docker-compose ps
```

**Expected Output:**
```
NAME                    STATUS              PORTS
underdog-trading        Up (healthy)        0.0.0.0:8000->8000/tcp
underdog-timescaledb    Up (healthy)        0.0.0.0:5432->5432/tcp
underdog-prometheus     Up                  0.0.0.0:9090->9090/tcp
underdog-grafana        Up                  0.0.0.0:3000->3000/tcp
```

### 3. Verify Services

**UNDERDOG Dashboard:**
```bash
curl http://localhost:8000/health
# Expected: {"status": "healthy", ...}
```

**Prometheus:**
- URL: http://localhost:9090
- Check Targets: http://localhost:9090/targets
- Should show `underdog` target as UP

**Grafana:**
- URL: http://localhost:3000
- Username: `admin` (or from GRAFANA_USER)
- Password: (from GRAFANA_PASSWORD in .env)

**TimescaleDB:**
```bash
docker exec -it underdog-timescaledb psql -U underdog -d underdog_trading -c "\dt"
# Expected: List of tables (ohlcv, trades, positions, metrics, account_snapshots)
```

---

## Service Details

### 1. UNDERDOG Trading (`underdog:8000`)

**Purpose:** Main trading application with ML/risk management/execution

**Endpoints:**
- `GET /` - API info
- `GET /health` - Health check
- `GET /metrics` - Prometheus metrics
- `GET /stats` - Trading statistics

**Logs:**
```bash
docker-compose logs -f underdog
```

**Restart:**
```bash
docker-compose restart underdog
```

### 2. TimescaleDB (`timescaledb:5432`)

**Purpose:** Time-series database for OHLCV, trades, metrics

**Connect:**
```bash
docker exec -it underdog-timescaledb psql -U underdog -d underdog_trading
```

**Queries:**
```sql
-- View recent trades
SELECT * FROM trades ORDER BY time DESC LIMIT 10;

-- Calculate daily P&L
SELECT 
    DATE(time) as date,
    SUM(realized_pnl) as daily_pnl,
    COUNT(*) as num_trades
FROM trades
GROUP BY DATE(time)
ORDER BY date DESC;

-- Current positions
SELECT * FROM positions;
```

**Backup:**
```bash
docker exec underdog-timescaledb pg_dump -U underdog underdog_trading > backup.sql
```

**Restore:**
```bash
cat backup.sql | docker exec -i underdog-timescaledb psql -U underdog -d underdog_trading
```

### 3. Prometheus (`prometheus:9090`)

**Purpose:** Metrics collection and storage

**Query Examples:**

```promql
# Win rate
(sum(underdog_trades_total{result="win"}) / sum(underdog_trades_total)) * 100

# Current capital
underdog_capital_usd

# Drawdown
underdog_drawdown_pct{timeframe="current"}

# Execution latency P95
histogram_quantile(0.95, rate(underdog_execution_latency_ms_bucket[5m]))
```

**Data Retention:** 15 days (configurable in prometheus.yml)

### 4. Grafana (`grafana:3000`)

**Purpose:** Visualization dashboards

**Default Dashboards:**
- Trading Performance (win rate, P&L, Sharpe)
- Risk Metrics (DD, exposure, leverage)
- Execution Performance (latency, rejections)
- System Health (components, uptime)

**Import Custom Dashboard:**
1. Login to Grafana
2. Configuration → Data Sources → Verify Prometheus
3. Dashboards → Import → Upload JSON
4. Set Prometheus as data source

---

## Configuration

### Environment Variables (.env)

```bash
# === MetaTrader 5 ===
MT5_LOGIN=12345678
MT5_PASSWORD=your_password
MT5_SERVER=BrokerName-Demo

# === Database ===
DB_PASSWORD=strong_password_here

# === Grafana ===
GRAFANA_PASSWORD=another_strong_password

# === Alerts ===
SMTP_SERVER=smtp.gmail.com
SMTP_PORT=587
SMTP_USERNAME=your-email@gmail.com
SMTP_PASSWORD=your_app_password
ALERT_EMAIL=alerts@your-domain.com

SLACK_WEBHOOK_URL=https://hooks.slack.com/services/...
TELEGRAM_BOT_TOKEN=123456789:ABC...
TELEGRAM_CHAT_ID=123456789

# === Trading ===
INITIAL_CAPITAL=100000.0
MAX_DRAWDOWN_PCT=10.0
DAILY_DD_LIMIT_PCT=2.0
```

### Volume Mounts

| Container Path | Host Path | Purpose |
|----------------|-----------|---------|
| `/app/data` | `../data` | Historical data, processed data |
| `/app/logs` | `../logs` | Application logs |
| `/app/mlruns` | `../mlruns` | MLflow experiments |
| `/app/models` | `../models` | Trained ML models |
| `/app/config` | `../config` | Strategy configurations (read-only) |

---

## Production Deployment

### 1. Deploy to VPS

**Recommended:** DigitalOcean Droplet (2GB RAM, 2 vCPU, $12/month)

```bash
# 1. Create droplet (Ubuntu 22.04)
doctl compute droplet create underdog-prod \
  --image ubuntu-22-04-x64 \
  --size s-2vcpu-2gb \
  --region lon1

# 2. SSH to droplet
ssh root@your-droplet-ip

# 3. Install Docker
curl -fsSL https://get.docker.com | sh

# 4. Clone repository
git clone https://github.com/whateve-r/UNDERDOG.git
cd UNDERDOG/docker

# 5. Configure
cp .env.template .env
nano .env  # Fill in credentials

# 6. Start services
docker-compose up -d

# 7. Check logs
docker-compose logs -f
```

### 2. SSL/TLS (HTTPS)

**Option A: Nginx Reverse Proxy with Let's Encrypt**

```bash
# Install Nginx
sudo apt install nginx certbot python3-certbot-nginx

# Configure Nginx
sudo nano /etc/nginx/sites-available/underdog
```

```nginx
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://localhost:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }

    location /prometheus/ {
        proxy_pass http://localhost:9090/;
    }

    location /grafana/ {
        proxy_pass http://localhost:3000/;
    }
}
```

```bash
# Enable site
sudo ln -s /etc/nginx/sites-available/underdog /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl restart nginx

# Get SSL certificate
sudo certbot --nginx -d your-domain.com
```

**Option B: Cloudflare Tunnel (No port forwarding needed)**

```bash
# Install cloudflared
wget https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64.deb
sudo dpkg -i cloudflared-linux-amd64.deb

# Login
cloudflared tunnel login

# Create tunnel
cloudflared tunnel create underdog

# Configure tunnel
nano ~/.cloudflared/config.yml
```

```yaml
tunnel: <TUNNEL_ID>
credentials-file: /root/.cloudflared/<TUNNEL_ID>.json

ingress:
  - hostname: underdog.your-domain.com
    service: http://localhost:8000
  - hostname: prometheus.your-domain.com
    service: http://localhost:9090
  - hostname: grafana.your-domain.com
    service: http://localhost:3000
  - service: http_status:404
```

```bash
# Run tunnel
cloudflared tunnel run underdog
```

### 3. Firewall Configuration

```bash
# Allow SSH (22), HTTP (80), HTTPS (443)
sudo ufw allow 22/tcp
sudo ufw allow 80/tcp
sudo ufw allow 443/tcp

# Block direct access to internal ports
sudo ufw deny 5432/tcp  # PostgreSQL
sudo ufw deny 9090/tcp  # Prometheus
sudo ufw deny 3000/tcp  # Grafana

# Enable firewall
sudo ufw enable
```

### 4. Monitoring and Alerts

**Uptime Monitoring:**
- Use Uptime Kuma or UptimeRobot
- Monitor: `https://your-domain.com/health`
- Alert on 503 status

**Log Aggregation:**
```bash
# Forward logs to external service (optional)
docker-compose logs --follow | tee /var/log/underdog.log
```

---

## Maintenance

### Backup Strategy

**1. Database Backups (Daily)**
```bash
# Automated backup script
cat > /root/backup-db.sh <<'EOF'
#!/bin/bash
DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_DIR="/backup/database"
mkdir -p $BACKUP_DIR

docker exec underdog-timescaledb pg_dump -U underdog underdog_trading | \
  gzip > $BACKUP_DIR/underdog_${DATE}.sql.gz

# Keep only last 30 days
find $BACKUP_DIR -name "*.sql.gz" -mtime +30 -delete
EOF

chmod +x /root/backup-db.sh

# Add to crontab (daily at 2 AM)
crontab -e
# Add: 0 2 * * * /root/backup-db.sh
```

**2. Volume Backups (Weekly)**
```bash
# Backup Docker volumes
docker run --rm -v underdog_timescaledb-data:/data -v $(pwd):/backup \
  ubuntu tar czf /backup/timescaledb_backup.tar.gz /data
```

### Updates

```bash
cd UNDERDOG/docker

# Pull latest code
git pull origin main

# Rebuild images
docker-compose build

# Restart services (zero-downtime with rolling update)
docker-compose up -d --no-deps --build underdog

# Check logs
docker-compose logs -f underdog
```

### Scaling

**Horizontal Scaling (Multiple Instances):**
```yaml
# docker-compose.yml
services:
  underdog:
    deploy:
      replicas: 3
    ports:
      - "8000-8002:8000"
```

**Load Balancer (Nginx):**
```nginx
upstream underdog_backend {
    least_conn;
    server localhost:8000;
    server localhost:8001;
    server localhost:8002;
}

server {
    location / {
        proxy_pass http://underdog_backend;
    }
}
```

---

## Troubleshooting

### Issue: Container won't start

```bash
# Check logs
docker-compose logs underdog

# Common issues:
# 1. Missing .env file → Copy .env.template
# 2. Port already in use → Change port in docker-compose.yml
# 3. Permission denied → Run with sudo or add user to docker group
```

### Issue: Can't connect to MT5

```bash
# Check MT5 credentials
docker-compose exec underdog python -c "
import MetaTrader5 as mt5
mt5.initialize()
print(f'Connected: {mt5.terminal_info()}')
"

# Verify server name
# Common: "ICMarkets-Demo02", "XM-Demo 3", "FTMO-Server"
```

### Issue: Database connection failed

```bash
# Check if TimescaleDB is healthy
docker-compose ps timescaledb

# Test connection
docker exec -it underdog-timescaledb psql -U underdog -d underdog_trading -c "SELECT version();"

# Reset database (CAUTION: Deletes all data)
docker-compose down -v
docker-compose up -d timescaledb
```

### Issue: Prometheus not scraping

```bash
# Check if UNDERDOG dashboard is accessible
curl http://localhost:8000/metrics

# Check Prometheus configuration
docker-compose exec prometheus cat /etc/prometheus/prometheus.yml

# Restart Prometheus
docker-compose restart prometheus
```

---

## Security Best Practices

1. **Change default passwords** in `.env`
2. **Use secrets management** (Docker Secrets or Vault)
3. **Enable SSL/TLS** for all endpoints
4. **Restrict database access** (firewall rules)
5. **Regularly update** Docker images
6. **Monitor logs** for suspicious activity
7. **Backup credentials** securely (encrypted storage)
8. **Use read-only volumes** for config files
9. **Implement rate limiting** on API endpoints
10. **Enable 2FA** for Grafana

---

## Performance Tuning

### Database

```sql
-- Vacuum and analyze
VACUUM ANALYZE trades;

-- Reindex
REINDEX TABLE trades;

-- Check compression
SELECT * FROM timescaledb_information.compressed_chunk_stats;
```

### Prometheus

```yaml
# prometheus.yml - Increase retention
global:
  scrape_interval: 30s  # Reduce frequency
  
storage:
  tsdb:
    retention.time: 30d  # Increase retention
    retention.size: 50GB
```

### Application

```python
# Optimize Python code
# - Use async/await for I/O operations
# - Cache ML model predictions
# - Batch database writes
# - Use connection pooling
```

---

## Cost Estimation (Monthly)

| Resource | Provider | Specs | Cost |
|----------|----------|-------|------|
| VPS | DigitalOcean | 2GB RAM, 2 vCPU | $12 |
| Domain | Namecheap | .com domain | $1 |
| SSL | Let's Encrypt | Free | $0 |
| Monitoring | UptimeRobot | Free tier | $0 |
| **Total** | | | **$13/month** |

**Enterprise Setup (+$50/month):**
- VPS: AWS EC2 t3.medium ($30)
- RDS PostgreSQL ($20)
- CloudWatch monitoring ($5)
- Total: ~$68/month

---

## Support

- **Documentation**: `docs/`
- **GitHub Issues**: https://github.com/whateve-r/UNDERDOG/issues
- **Docker Logs**: `docker-compose logs -f`
- **Health Check**: `curl http://localhost:8000/health`

---

## License

MIT License - See LICENSE file for details
