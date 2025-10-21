# 🚀 UNDERDOG - PRODUCTION ROADMAP (REAL BUSINESS)

**Mission:** Sistema de trading algorítmico autónomo para pasar pruebas de Prop Firms y generar rentabilidad mensual sostenible.

**Deployment Target:** OVHCloud / TradingVPS (24/7 uptime)  
**Market:** Forex (tipos de cambio)  
**Risk Profile:** Prop Firm compliant (DD limits strict)

---

## 🎯 OBJETIVOS REALES DEL NEGOCIO

### Objetivo Principal
**Generar rentabilidad mensual consistente mediante trading algorítmico en múltiples Prop Firms.**

### Métricas de Éxito (KPIs)
1. **Pasar fase 1 de Prop Firm:** ≥8% profit, <5% daily DD, <10% total DD
2. **Pasar fase 2 de Prop Firm:** ≥5% profit, <5% daily DD, <10% total DD
3. **Mantener cuenta funded:** Rentabilidad >0% mensual, DD <8%
4. **Escalabilidad:** Operar en 3-5 Prop Firms simultáneamente
5. **ROI sistema:** >300% anual (contando fees de Prop Firms)

### Anti-Objetivos (NO es un TFG)
- ❌ NO necesitamos "demostrar metodología científica"
- ❌ NO necesitamos "diversidad de estrategias por experimentación"
- ❌ NO necesitamos "análisis comparativo para paper académico"
- ✅ **SÍ necesitamos:** Código que FUNCIONA, hace dinero y NO PIERDE cuentas

---

## 📊 ESTADO ACTUAL VS NECESIDADES PRODUCCIÓN

| **Componente** | **Estado Actual** | **Necesidad Producción** | **Gap** |
|----------------|-------------------|--------------------------|---------|
| **Backtesting** | ✅ 95% | ✅ Suficiente | Ninguno |
| **Risk Management** | ✅ 85% | ✅ PropFirmRiskManager OK | Ninguno |
| **Estrategias rentables** | 🔴 3 sin validar | ✅ 2-3 validadas (Sharpe >1.5) | **CRÍTICO** |
| **MT5 Integration** | 🔴 0% | ✅ Live execution | **CRÍTICO** |
| **Monitoring 24/7** | 🟡 50% | ✅ Alertas + Dashboards | **ALTO** |
| **VPS Deployment** | 🔴 0% | ✅ Docker + systemd | **ALTO** |
| **ML que funciona** | 🟡 50% | 🟡 Nice-to-have | MEDIO |
| **Database histórico** | 🟡 70% | 🟡 Nice-to-have | BAJO |

---

## 🔥 PLAN DE ACCIÓN (NEXT 60 DAYS TO LIVE TRADING)

### FASE 1: VALIDATION SPRINT (Semana 1-2) 🎯

**Objetivo:** Tener 2 estrategias PROBADAS que generan alpha en backtesting

#### Week 1: Backtest Intensivo
**Días 1-3: Validar estrategias existentes con datos REALES**
```bash
# Setup HuggingFace token
poetry run python scripts/setup_hf_token.py

# Backtest exhaustivo (2 años de datos)
# ATRBreakout
poetry run python scripts/test_end_to_end.py --strategy ATRBreakout --use-hf-data

# SuperTrendRSI
poetry run python scripts/test_end_to_end.py --strategy SuperTrendRSI --use-hf-data

# BollingerCCI
poetry run python scripts/test_end_to_end.py --strategy BollingerCCI --use-hf-data
```

**Criterio de Selección:**
- ✅ Sharpe Ratio >1.5
- ✅ Calmar Ratio >2.0
- ✅ Max DD <8%
- ✅ Win Rate >48%
- ✅ Profit Factor >1.4
- ✅ Monte Carlo ROBUST (p-value >0.05)

**Resultado esperado:** Seleccionar TOP 2 estrategias

**Días 4-5: Optimización de parámetros (TOP 2 estrategias)**
```python
# Usar bt_engine.run_parameter_sweep()
# Optimizar para Calmar Ratio (no Sharpe - queremos preservar capital)
```

**Días 6-7: Forward Testing (Out-of-Sample)**
```python
# Walk-Forward Optimization
# Train: 2023-2024
# Test: Q1 2025
# Validar que estrategias mantienen performance OOS
```

**ENTREGABLE SEMANA 1:** 
- 📊 Report con 2 estrategias validadas
- 📈 Parámetros óptimos confirmados
- ✅ Confianza >80% de que funcionarán en live

---

### FASE 2: LIVE EXECUTION ENGINE (Semana 3-4) 🔌

**Objetivo:** Conectar MT5 y ejecutar órdenes automáticamente

#### Week 3: MT5 Integration

**Días 8-10: Implementar MT5Executor**
```python
# underdog/execution/mt5_executor.py
class MT5Executor:
    """
    CRITICAL: Este es el módulo que ejecuta órdenes REALES
    """
    def __init__(self, account: int, password: str, server: str):
        import MetaTrader5 as mt5
        if not mt5.initialize():
            raise RuntimeError("MT5 initialization failed")
        
        if not mt5.login(account, password, server):
            raise RuntimeError(f"MT5 login failed: {mt5.last_error()}")
    
    def execute_order(self, signal: dict) -> dict:
        """
        Ejecutar orden con validaciones de riesgo PRE-ejecución
        """
        # 1. Validar DD limits ANTES de enviar
        if self._check_dd_breach():
            self.emergency_close_all()
            raise RiskLimitBreached("Daily DD exceeded")
        
        # 2. Calcular position size con Kelly
        size = self._calculate_position_size(signal)
        
        # 3. Enviar orden a MT5
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": signal['symbol'],
            "volume": size,
            "type": mt5.ORDER_TYPE_BUY if signal['side'] == 'buy' else mt5.ORDER_TYPE_SELL,
            "price": mt5.symbol_info_tick(signal['symbol']).ask,
            "sl": signal['stop_loss'],
            "tp": signal['take_profit'],
            "magic": 234000,
            "comment": f"UNDERDOG_{signal['strategy']}"
        }
        
        result = mt5.order_send(request)
        
        # 4. Log EVERYTHING (para auditoría)
        self._log_trade(request, result)
        
        return result
```

**Días 11-12: Signal Bridge (Backtrader → MT5)**
```python
# underdog/bridges/bt_to_mt5.py
class BacktraderMT5Bridge:
    """
    Convierte señales de Backtrader a órdenes MT5
    """
    def __init__(self, strategy_bt: bt.Strategy, executor: MT5Executor):
        self.strategy = strategy_bt
        self.executor = executor
    
    def on_signal(self):
        """
        Callback cuando Backtrader genera buy/sell
        """
        if len(self.strategy) > 0:  # Has position
            # Ya tenemos posición, skip
            return
        
        # Extraer señal de Backtrader
        signal = {
            'strategy': self.strategy.__class__.__name__,
            'symbol': self.strategy.data._name,
            'side': 'buy' if self.strategy.signal == 1 else 'sell',
            'entry_price': self.strategy.data.close[0],
            'stop_loss': self.strategy.sl,
            'take_profit': self.strategy.tp
        }
        
        # Ejecutar en MT5
        self.executor.execute_order(signal)
```

**Días 13-14: Paper Trading Test (Cuenta DEMO)**
```bash
# Configure .env with DEMO account
MT5_ACCOUNT=12345678
MT5_PASSWORD=demo_pass
MT5_SERVER=ICMarketsSC-Demo

# Run live with demo account
poetry run python scripts/start_live.py \
    --strategy ATRBreakout \
    --symbol EURUSD \
    --mode paper
```

**ENTREGABLE SEMANA 3:**
- ✅ MT5Executor funcional
- ✅ 10+ operaciones ejecutadas en DEMO
- ✅ Verificar que DD limits funcionan
- ✅ Logs completos de ejecución

---

### FASE 3: MONITORING & RELIABILITY (Semana 5) 📡

**Objetivo:** Sistema que NO FALLA nunca (uptime 99.9%)

#### Week 5: Production Hardening

**Días 15-16: Monitoring Stack**
```yaml
# docker-compose.yml
version: '3.8'
services:
  underdog:
    build: .
    restart: always  # CRITICAL: Auto-restart si falla
    environment:
      - MT5_ACCOUNT=${MT5_ACCOUNT}
      - MT5_PASSWORD=${MT5_PASSWORD}
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
  
  prometheus:
    image: prom/prometheus:latest
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
    ports:
      - "9090:9090"
  
  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=${GRAFANA_PASSWORD}
  
  alertmanager:
    image: prom/alertmanager:latest
    volumes:
      - ./alertmanager.yml:/etc/alertmanager/alertmanager.yml
    ports:
      - "9093:9093"
```

**Días 17-18: Alerting (CRITICAL para Prop Firms)**
```yaml
# alertmanager.yml
route:
  receiver: 'telegram'
  group_by: ['alertname']
  group_wait: 10s
  group_interval: 10s
  repeat_interval: 1h

receivers:
  - name: 'telegram'
    telegram_configs:
      - bot_token: '${TELEGRAM_BOT_TOKEN}'
        chat_id: ${TELEGRAM_CHAT_ID}
        message: |
          🚨 UNDERDOG ALERT 🚨
          {{ range .Alerts }}
          Alert: {{ .Labels.alertname }}
          Status: {{ .Status }}
          Details: {{ .Annotations.description }}
          {{ end }}

# Alertas CRÍTICAS
alerts:
  - name: daily_dd_breach
    expr: underdog_daily_drawdown_pct > 4.5  # Alert ANTES de breach (5%)
    for: 1m
    labels:
      severity: critical
    annotations:
      description: "Daily DD at {{ $value }}% - CLOSE TO LIMIT!"
  
  - name: mt5_connection_lost
    expr: up{job="underdog"} == 0
    for: 2m
    labels:
      severity: critical
    annotations:
      description: "MT5 connection lost! Check VPS."
  
  - name: position_stuck
    expr: underdog_position_duration_hours > 48
    for: 1h
    labels:
      severity: warning
    annotations:
      description: "Position open >48h - check for stale trades"
```

**Días 19-21: Failure Recovery**
```python
# underdog/execution/recovery.py
class FailureRecoveryManager:
    """
    Maneja fallos de conexión, VPS reboot, etc.
    """
    def __init__(self, executor: MT5Executor):
        self.executor = executor
    
    def on_startup(self):
        """
        Ejecutar SIEMPRE al iniciar el bot
        """
        # 1. Sincronizar posiciones abiertas
        mt5_positions = self.executor.get_open_positions()
        local_positions = self.load_positions_from_db()
        
        if mt5_positions != local_positions:
            self._reconcile_positions(mt5_positions, local_positions)
        
        # 2. Verificar órdenes pendientes
        pending_orders = self.executor.get_pending_orders()
        for order in pending_orders:
            if order['age_hours'] > 24:
                self.executor.cancel_order(order['ticket'])
        
        # 3. Calcular DD actual
        current_dd = self.executor.calculate_drawdown()
        if current_dd > 0.048:  # 4.8% (cerca de 5% limit)
            logger.critical(f"STARTUP: DD at {current_dd:.2%} - RISK MODE")
            self.enable_defensive_mode()
    
    def on_connection_lost(self):
        """
        MT5 connection dropped
        """
        logger.error("MT5 connection lost - attempting reconnect")
        
        for attempt in range(5):
            time.sleep(10)
            if self.executor.reconnect():
                logger.info(f"Reconnected on attempt {attempt+1}")
                self.on_startup()  # Re-sync state
                return
        
        # Failed to reconnect - EMERGENCY
        self._send_emergency_alert("MT5 reconnection failed after 5 attempts")
```

**ENTREGABLE SEMANA 5:**
- ✅ Docker stack completo
- ✅ Grafana dashboards configurados
- ✅ Alertas Telegram funcionando
- ✅ Recovery manager testeado (simular desconexión)

---

### FASE 4: VPS DEPLOYMENT (Semana 6-7) ☁️

**Objetivo:** Bot corriendo 24/7 en VPS

#### Week 6: VPS Setup

**Provider Recomendado:** OVHCloud VPS (€6/mes)
```
Specs:
- 1 vCPU (suficiente para 2-3 estrategias)
- 2 GB RAM
- 40 GB SSD
- Ubuntu 22.04 LTS
- IP dedicada (para MT5 whitelist)
```

**Días 22-24: VPS Configuration**
```bash
# SSH into VPS
ssh root@your_vps_ip

# Install dependencies
apt update && apt upgrade -y
apt install -y python3.13 python3-pip docker.io docker-compose git

# Install Poetry
curl -sSL https://install.python-poetry.org | python3 -

# Clone repo
git clone https://github.com/whateve-r/UNDERDOG.git
cd UNDERDOG

# Configure environment
cp .env.template .env
nano .env  # Fill MT5 credentials, Telegram tokens, etc.

# Build Docker image
docker-compose build

# Start services
docker-compose up -d

# Verify
docker-compose ps
curl http://localhost:8000/health
```

**Días 25-26: Systemd Service (Auto-start on reboot)**
```bash
# /etc/systemd/system/underdog.service
[Unit]
Description=UNDERDOG Trading Bot
After=docker.service
Requires=docker.service

[Service]
Type=simple
WorkingDirectory=/root/UNDERDOG
ExecStart=/usr/bin/docker-compose up
ExecStop=/usr/bin/docker-compose down
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

```bash
# Enable service
systemctl enable underdog
systemctl start underdog

# Check logs
journalctl -u underdog -f
```

**Día 27: Firewall & Security**
```bash
# ufw (firewall)
ufw default deny incoming
ufw default allow outgoing
ufw allow 22/tcp      # SSH
ufw allow 443/tcp     # HTTPS (Grafana)
ufw enable

# Fail2ban (SSH brute-force protection)
apt install fail2ban -y

# SSH key-only (disable password)
nano /etc/ssh/sshd_config
# PasswordAuthentication no
systemctl restart sshd
```

**ENTREGABLE SEMANA 6:**
- ✅ Bot corriendo en VPS
- ✅ Auto-restart configurado
- ✅ Grafana accesible vía HTTPS
- ✅ Logs persistentes

---

### FASE 5: LIVE PAPER TRADING (Semana 8-9) 📈

**Objetivo:** 30 días de paper trading sin intervención

#### Week 8-9: Automated Trading (DEMO Account)

**Setup:**
```bash
# Start bot on VPS with DEMO account
# NO INTERVENTION por 30 días
# Solo monitorear métricas
```

**Daily Checklist:**
- ✅ Check Grafana dashboard (5 min/día)
- ✅ Verify no alerts en Telegram
- ✅ Review trade log (1x semana)

**Success Criteria (para pasar a LIVE):**
- ✅ 30 días sin crashes
- ✅ DD máximo <7% en cualquier día
- ✅ Rentabilidad >0% (positivo cualquier valor)
- ✅ No alertas críticas
- ✅ Todas las órdenes ejecutadas correctamente

**ENTREGABLE SEMANA 9:**
- ✅ 30 días de logs completos
- ✅ Report de performance
- ✅ Decisión GO/NO-GO para cuenta real

---

### FASE 6: PROP FIRM CHALLENGE (Semana 10+) 💰

**Objetivo:** Pasar fase 1 de Prop Firm

#### Prop Firm Selection

**Recomendados (por orden):**
1. **FTMO** (€155 for 50k account challenge)
   - Fase 1: 8% profit, 5% daily DD, 10% total DD (30 días)
   - Fase 2: 5% profit, 5% daily DD, 10% total DD (60 días)
   
2. **The5ers** ($230 for 50k account)
   - Aggressive scaling (50k → 500k)
   - 6% profit target per step
   
3. **MyForexFunds** (€99 for 25k account)
   - Más fácil de pasar
   - Lower payouts (60% profit split)

**Strategy:**
```python
# Configuración conservadora para challenge
PROP_FIRM_CONFIG = {
    'initial_capital': 50000,
    'risk_per_trade': 0.015,  # 1.5% (más conservador que 2%)
    'daily_dd_limit': 0.045,  # 4.5% (buffer antes de 5%)
    'total_dd_limit': 0.09,   # 9% (buffer antes de 10%)
    'max_positions': 2,       # Máximo 2 posiciones simultáneas
    'min_rr_ratio': 1.5,      # Risk/Reward mínimo 1:1.5
}
```

**Week 10-12: FTMO Phase 1 (30 days)**
- Day 1-10: Operar conservative (1-2 trades/day max)
- Day 11-20: Si rentabilidad >4%, mantener
- Day 21-30: Si rentabilidad >7%, STOP trading (objetivo cumplido)

**Week 13-21: FTMO Phase 2 (60 days)**
- Mismo approach pero más relajado (solo 5% profit needed)

**Week 22+: FUNDED ACCOUNT**
- 🎉 GENERANDO INCOME REAL
- Profit split: 80% tú, 20% FTMO
- Target: €2,000-4,000/mes con 50k account

---

## 🛠️ ARQUITECTURA PRODUCCIÓN (REAL)

### Stack Tecnológico Final

```
┌─────────────────────────────────────────────────────────┐
│                    VPS (OVHCloud)                       │
│  ┌──────────────────────────────────────────────────┐  │
│  │  Docker Container: underdog_trading              │  │
│  │  ┌────────────────────────────────────────────┐  │  │
│  │  │  Python 3.13 + Poetry                      │  │  │
│  │  │  ├── MT5 Connector (live execution)        │  │  │
│  │  │  ├── Strategy Engine (ATRBreakout, etc)    │  │  │
│  │  │  ├── PropFirmRiskManager (DD guardian)     │  │  │
│  │  │  ├── FailureRecoveryManager                │  │  │
│  │  │  └── Prometheus Exporter (metrics)         │  │  │
│  │  └────────────────────────────────────────────┘  │  │
│  └──────────────────────────────────────────────────┘  │
│  ┌──────────────────────────────────────────────────┐  │
│  │  TimescaleDB (trade history)                     │  │
│  └──────────────────────────────────────────────────┘  │
│  ┌──────────────────────────────────────────────────┐  │
│  │  Prometheus (metrics scraping)                    │  │
│  └──────────────────────────────────────────────────┘  │
│  ┌──────────────────────────────────────────────────┐  │
│  │  Grafana (dashboards + alerting)                 │  │
│  └──────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────┘
           │                          │
           ▼                          ▼
    ┌─────────────┐          ┌──────────────┐
    │  MT5 Broker │          │   Telegram   │
    │  (FTMO)     │          │   (Alerts)   │
    └─────────────┘          └──────────────┘
```

### Módulos CRÍTICOS vs Nice-to-Have

**🔴 CRÍTICOS (sin esto NO FUNCIONA):**
1. ✅ `bt_engine.py` - Backtesting
2. ✅ `prop_firm_rme.py` - Risk management
3. 🔴 `mt5_executor.py` - Live execution (TO DO)
4. 🔴 `failure_recovery.py` - Reliability (TO DO)
5. 🟡 `prometheus_metrics.py` - Monitoring (50% done)

**🟡 IMPORTANTES (mejoran performance):**
1. 🟡 ML strategies (50% done) - Puede outperform tradicionales
2. ✅ Monte Carlo validation - Evita lucky backtests
3. ✅ WFO - Evita overfitting

**🔵 NICE-TO-HAVE (pueden esperar):**
1. Event-Driven architecture - Backtrader es suficiente
2. HuggingFace data - Synthetic data funciona para validación
3. News sentiment - No impacta significativamente
4. Multiple ML models - Logistic Regression suficiente para empezar

---

## 📊 MÉTRICAS DE NEGOCIO (KPIs REALES)

### Métricas Operacionales

```python
# Dashboard Grafana
OPERATIONAL_KPIS = {
    # Uptime
    'bot_uptime_pct': 99.9,  # Target: 99.9%
    'mt5_connection_uptime': 99.5,
    
    # Performance
    'daily_return_avg': 0.15,  # 0.15% diario = 4.5%/mes
    'sharpe_ratio_live': 1.8,
    'max_dd_ever': 7.2,  # NUNCA superar 8%
    
    # Risk
    'daily_dd_breaches': 0,  # ZERO tolerance
    'total_dd_breaches': 0,
    'trades_rejected_by_risk': 12,  # Cuantos más, mejor (RME working)
    
    # Execution
    'avg_slippage_pips': 0.3,
    'order_execution_time_ms': 150,
    'failed_orders_pct': 0.1,  # <0.1%
    
    # Business
    'monthly_profit_eur': 2500,  # Target con 50k account
    'prop_firm_accounts_active': 3,  # FTMO, The5ers, MFF
    'total_aum': 150000,  # 50k x 3
}
```

### Revenue Model

**Año 1 (Conservative):**
```
FTMO 50k:      €2,000/mes x 80% split = €1,600/mes
The5ers 50k:   €1,800/mes x 75% split = €1,350/mes
MFF 25k:       €800/mes x 60% split   = €480/mes
────────────────────────────────────────────────
TOTAL:                                 €3,430/mes
                                       €41,160/año
```

**Año 2 (Scaling):**
```
FTMO 100k:     €4,000/mes x 80% = €3,200/mes
The5ers 100k:  €3,600/mes x 75% = €2,700/mes
FTMO 50k (2nd):€2,000/mes x 80% = €1,600/mes
MFF 50k:       €1,600/mes x 60% = €960/mes
────────────────────────────────────────────
TOTAL:                            €8,460/mes
                                  €101,520/año
```

**Costos:**
```
VPS OVHCloud:           €6/mes
Prop Firm fees:         €350/año (challenges iniciales)
Data feeds (opcional):  €0 (HuggingFace gratis)
────────────────────────────────────────────
Total costs:            €422/año

Net profit Year 2:      €101,098/año
```

---

## ✅ CHECKLIST PRODUCCIÓN (GO-LIVE)

### Pre-Live Validation

- [ ] **Backtesting riguroso**
  - [ ] 2+ años de datos reales (HuggingFace)
  - [ ] Sharpe >1.5 on top 2 strategies
  - [ ] Monte Carlo ROBUST (p>0.05)
  - [ ] WFO performance positiva OOS

- [ ] **Risk Management**
  - [ ] PropFirmRiskManager testeado con edge cases
  - [ ] Daily DD limit enforcement verified
  - [ ] Emergency stop funciona (simulated breach)
  - [ ] Position sizing con Kelly validated

- [ ] **MT5 Integration**
  - [ ] 100+ orders ejecutadas en DEMO sin fallos
  - [ ] Slippage promedio <0.5 pips
  - [ ] Order rejection rate <0.1%
  - [ ] Reconnection funciona (simulate disconnection)

- [ ] **Monitoring**
  - [ ] Grafana dashboards deployed
  - [ ] Prometheus scraping cada 15s
  - [ ] Telegram alerts funcionando
  - [ ] Email backup alerts configured

- [ ] **VPS Reliability**
  - [ ] 30 días uptime >99.9%
  - [ ] Auto-restart after reboot verified
  - [ ] Failure recovery tested (kill process, reboot VPS)
  - [ ] Logs rotating correctly

- [ ] **Paper Trading**
  - [ ] 30 días DEMO account sin intervención
  - [ ] Rentabilidad >0%
  - [ ] DD máximo <7%
  - [ ] Zero critical alerts

### GO-LIVE Decision

**Criteria (TODOS deben cumplirse):**
1. ✅ Backtesting Sharpe >1.5
2. ✅ 30 días paper trading positivos
3. ✅ Zero DD breaches en paper trading
4. ✅ VPS uptime >99.9%
5. ✅ Monitoring & alerts funcionando
6. ✅ €350 disponibles para FTMO challenge

**Si 1 criterio falla → NO GO LIVE**

---

## 🎯 PRÓXIMOS PASOS INMEDIATOS

### HOY (Siguiente 2 horas):

1. **Fix test_end_to_end.py error**
```bash
# Error: monte_carlo_runs not expected
# Fix en bt_engine.py signature
```

2. **Ejecutar backtest con datos REALES**
```bash
poetry run python scripts/setup_hf_token.py
poetry run python scripts/test_end_to_end.py --use-hf-data --quick
```

3. **Revisar resultados y decidir estrategias TOP 2**

### ESTA SEMANA:

1. **Días 1-2:** Backtesting exhaustivo (2 años datos reales)
2. **Día 3:** Optimización parámetros
3. **Día 4-5:** WFO validation
4. **Día 6-7:** Documentar resultados + decidir TOP 2 strategies

### MES 1:

- Semana 1-2: Validation Sprint ✅
- Semana 3-4: MT5 Integration 🔌
- Semana 5: Monitoring & Reliability 📡
- Semana 6: VPS Deployment ☁️

### MES 2-3:

- 30 días paper trading
- Ajustes finales
- GO-LIVE decision

---

**Este es un NEGOCIO, no un TFG. Cada línea de código debe justificarse en términos de ROI y risk management.**

¿Empezamos arreglando el test y validando con datos reales?
