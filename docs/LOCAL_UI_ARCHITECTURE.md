# 🖥️ ARQUITECTURA UI LOCAL (NO WEB DEPLOYMENT)

**Fecha**: Octubre 20, 2025  
**Contexto**: UI para monitoreo local en PC (no deployment público)  
**Objetivo**: Dashboard profesional con mínima complejidad

---

## 🎯 DECISIÓN: GRAFANA + STREAMLIT (Híbrido)

### Por qué NO FastAPI + Dash + WebSocket:

❌ **Over-engineering** para uso local  
❌ Requiere infraestructura compleja (Redis Pub/Sub, WSS)  
❌ Diseñado para multi-usuario (no necesario aquí)  
❌ Más código de mantenimiento  

### Por qué SÍ Grafana + Streamlit:

✅ **Grafana**: Monitoreo en tiempo real (PROFESIONAL)  
✅ **Streamlit**: Backtesting UI (SIMPLE)  
✅ Ya tienes Grafana en `docker-compose.yml`  
✅ 90% menos código que FastAPI+Dash  
✅ Usado por empresas reales (Netflix, Uber)  

---

## 📊 ARQUITECTURA PROPUESTA

```
┌─────────────────────────────────────────────────────────────┐
│                    MONITORING (Real-Time)                    │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  7 EAs (Python TA-Lib)                                      │
│         ↓                                                    │
│  Prometheus Client (push metrics cada 1 seg)               │
│         ↓                                                    │
│  Prometheus Server (localhost:9090)                         │
│         ↓                                                    │
│  Grafana (localhost:3000)                                   │
│         ↓                                                    │
│  Browser → Dashboard profesional                            │
│                                                              │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                    BACKTESTING UI                            │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Streamlit App (localhost:8501)                             │
│         ↓                                                    │
│  • Parameter sliders (RSI period, ATR mult, etc.)           │
│  • Run Backtest button                                      │
│  • Heatmap de sensibilidad (anti-overfitting)               │
│  • Walk-forward analysis chart                              │
│  • Equity curve con DD shading                              │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## 🚀 IMPLEMENTACIÓN

### 1. GRAFANA SETUP (Ya tienes Docker Compose)

**Paso 1**: Verificar que `docker-compose.yml` tiene:

```yaml
services:
  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./docker/prometheus.yml:/etc/prometheus/prometheus.yml

  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
```

**Paso 2**: Iniciar Docker:

```powershell
cd C:\Users\manud\OneDrive\Escritorio\tfg\UNDERDOG
docker-compose up -d
```

**Paso 3**: Acceder a Grafana:

- URL: `http://localhost:3000`
- User: `admin`
- Pass: `admin` (cambiar en primer login)

**Paso 4**: Añadir Prometheus como Data Source:

1. Settings → Data Sources → Add data source
2. Seleccionar "Prometheus"
3. URL: `http://prometheus:9090`
4. Save & Test

---

### 2. INSTRUMENTAR LAS 7 EAs CON PROMETHEUS

**Archivo**: `underdog/monitoring/prometheus_metrics.py` (nuevo)

```python
"""
Prometheus metrics for 7 EAs
"""
from prometheus_client import Counter, Gauge, Histogram, start_http_server
import time

# Métricas por EA
ea_signals_total = Counter(
    'ea_signals_total', 
    'Total signals generated',
    ['ea_name', 'signal_type']
)

ea_positions_open = Gauge(
    'ea_positions_open',
    'Current open positions',
    ['ea_name', 'symbol']
)

ea_pnl_usd = Gauge(
    'ea_pnl_usd',
    'Current P&L in USD',
    ['ea_name']
)

ea_balance = Gauge(
    'ea_balance',
    'Account balance',
    []
)

ea_equity = Gauge(
    'ea_equity',
    'Account equity',
    []
)

ea_drawdown_pct = Gauge(
    'ea_drawdown_pct',
    'Current drawdown percentage',
    ['drawdown_type']  # daily, total
)

ea_execution_time = Histogram(
    'ea_execution_time_ms',
    'EA execution time in milliseconds',
    ['ea_name'],
    buckets=[0.1, 0.5, 1, 5, 10, 50, 100]
)

ea_confidence = Gauge(
    'ea_confidence',
    'Signal confidence score',
    ['ea_name']
)

# Iniciar servidor HTTP para Prometheus scraping
def start_prometheus_server(port=8000):
    """Start Prometheus metrics server"""
    start_http_server(port)
    print(f"✅ Prometheus metrics server started on port {port}")
```

**Modificar cada EA** para instrumentar:

```python
# En ea_supertrend_rsi_v4.py (y las demás 6 EAs)
from underdog.monitoring.prometheus_metrics import (
    ea_signals_total,
    ea_confidence,
    ea_execution_time
)

async def generate_signal(self, df: pd.DataFrame) -> Optional[Signal]:
    start_time = time.time()
    
    # ... lógica existente ...
    
    if signal:
        # Incrementar contador
        ea_signals_total.labels(
            ea_name="SuperTrendRSI",
            signal_type="BUY" if signal.type == SignalType.BUY else "SELL"
        ).inc()
        
        # Actualizar confidence
        ea_confidence.labels(ea_name="SuperTrendRSI").set(signal.confidence)
    
    # Medir tiempo de ejecución
    elapsed_ms = (time.time() - start_time) * 1000
    ea_execution_time.labels(ea_name="SuperTrendRSI").observe(elapsed_ms)
    
    return signal
```

---

### 3. GRAFANA DASHBOARD (Template JSON)

**Archivo**: `docker/grafana-dashboards/trading-monitor.json` (nuevo)

Dashboards a crear:

#### **Dashboard 1: Portfolio Overview**

```
┌────────────────────────────────────────────────────────────┐
│  PORTFOLIO OVERVIEW                           🔴 LIVE      │
├────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐  │
│  │ Balance  │  │  Equity  │  │ Daily DD │  │ Total DD │  │
│  │ $10,250  │  │  $10,180 │  │  -1.2%   │  │  -3.5%   │  │
│  │  +2.5%   │  │  +2.1%   │  │  ⚠️      │  │   ✅     │  │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘  │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐  │
│  │  EQUITY CURVE (Last 24h)                           │  │
│  │                                                     │  │
│  │    10.5K ┤                         ╭─────          │  │
│  │         │                     ╭────╯               │  │
│  │    10.0K ┤              ╭─────╯                    │  │
│  │         │        ╭──────╯                          │  │
│  │     9.5K ┤───────╯                                 │  │
│  │         └─────────────────────────────────────────│  │
│  │          00:00  06:00  12:00  18:00  24:00        │  │
│  └─────────────────────────────────────────────────────┘  │
│                                                             │
└────────────────────────────────────────────────────────────┘
```

#### **Dashboard 2: EA Performance**

```
┌────────────────────────────────────────────────────────────┐
│  EA PERFORMANCE MATRIX                        🔴 LIVE      │
├────────────────────────────────────────────────────────────┤
│                                                             │
│  EA Name         │ Signals │ Win Rate │ P&L   │ Status    │
│  ────────────────┼─────────┼──────────┼───────┼─────────  │
│  SuperTrendRSI   │   15    │  73.3%   │ +$250 │ 🟢 ACTIVE │
│  ParabolicEMA    │   12    │  66.7%   │ +$180 │ 🟢 ACTIVE │
│  KeltnerBreakout │    8    │  75.0%   │ +$120 │ 🟢 ACTIVE │
│  EmaScalper      │   42    │  61.9%   │ +$95  │ 🟢 ACTIVE │
│  BollingerCCI    │    5    │  80.0%   │ +$65  │ 🟢 ACTIVE │
│  ATRBreakout     │    6    │  66.7%   │ +$45  │ 🟢 ACTIVE │
│  PairArbitrage   │    3    │ 100.0%   │ +$30  │ 🟢 ACTIVE │
│                                                             │
└────────────────────────────────────────────────────────────┘
```

#### **Dashboard 3: Open Positions**

```
┌────────────────────────────────────────────────────────────┐
│  OPEN POSITIONS (7 active)                    🔴 LIVE      │
├────────────────────────────────────────────────────────────┤
│                                                             │
│  Symbol  │ Type │ Volume │ Entry   │ Current │ P&L       │
│  ────────┼──────┼────────┼─────────┼─────────┼─────────  │
│  EURUSD  │ BUY  │ 0.10   │ 1.0850  │ 1.0872  │ +$22 ✅  │
│  GBPUSD  │ SELL │ 0.08   │ 1.2650  │ 1.2638  │ +$9.6 ✅ │
│  USDJPY  │ BUY  │ 0.12   │ 149.50  │ 149.68  │ +$14 ✅  │
│  EURJPY  │ BUY  │ 0.05   │ 162.80  │ 162.65  │ -$7.5 ⚠️ │
│                                                             │
└────────────────────────────────────────────────────────────┘
```

---

### 4. STREAMLIT BACKTESTING UI

**Archivo**: `underdog/ui/streamlit_backtest.py` (nuevo, ~300 líneas)

```python
"""
Streamlit UI for backtesting and parameter optimization
"""
import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np

st.set_page_config(
    page_title="UNDERDOG Backtesting",
    page_icon="📊",
    layout="wide"
)

st.title("📊 UNDERDOG Backtesting & Optimization")

# Sidebar: EA Selection
st.sidebar.header("⚙️ Configuration")
ea_name = st.sidebar.selectbox(
    "Select EA",
    ["SuperTrendRSI", "ParabolicEMA", "KeltnerBreakout", 
     "EmaScalper", "BollingerCCI", "ATRBreakout", "PairArbitrage"]
)

symbol = st.sidebar.selectbox("Symbol", ["EURUSD", "GBPUSD", "USDJPY"])
timeframe = st.sidebar.selectbox("Timeframe", ["M5", "M15", "H1", "H4"])

# Parameters (dynamic based on EA)
st.sidebar.subheader("📐 Parameters")

if ea_name == "SuperTrendRSI":
    rsi_period = st.sidebar.slider("RSI Period", 5, 30, 14)
    rsi_overbought = st.sidebar.slider("RSI Overbought", 60, 80, 65)
    rsi_oversold = st.sidebar.slider("RSI Oversold", 20, 40, 35)
    atr_period = st.sidebar.slider("ATR Period", 5, 30, 14)
    atr_multiplier = st.sidebar.slider("ATR Multiplier", 1.0, 5.0, 2.0)

# Run Backtest Button
if st.sidebar.button("🚀 Run Backtest", type="primary"):
    with st.spinner("Running backtest..."):
        # TODO: Llamar a backtesting engine
        st.success("✅ Backtest completed!")

# Main Area: Results
tab1, tab2, tab3, tab4 = st.tabs([
    "📈 Equity Curve", 
    "🔥 Heatmap", 
    "📊 Metrics", 
    "📋 Trades"
])

with tab1:
    st.subheader("Equity Curve")
    # TODO: Plotly equity curve con DD shading

with tab2:
    st.subheader("Parameter Sensitivity Heatmap")
    # TODO: Heatmap de Sharpe vs RSI period vs ATR mult

with tab3:
    st.subheader("Performance Metrics")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Return", "25.3%", "+2.1%")
    col2.metric("Sharpe Ratio", "1.85", "+0.12")
    col3.metric("Max DD", "-8.5%", "-1.2%")

with tab4:
    st.subheader("Trade History")
    # TODO: DataFrame con todos los trades
```

**Ejecutar**:

```powershell
cd C:\Users\manud\OneDrive\Escritorio\tfg\UNDERDOG
poetry run streamlit run underdog/ui/streamlit_backtest.py
```

Abre automáticamente `http://localhost:8501`

---

## 📦 ACTUALIZAR DEPENDENCIAS

```toml
[tool.poetry.dependencies]
# MANTENER:
prometheus-client = "^0.21.0"  # ✅ Ya tienes
plotly = "^5.24.0"              # ✅ Ya tienes

# AÑADIR:
streamlit = "^1.40.0"           # 🆕 Backtesting UI

# ELIMINAR (ya no necesarios):
# dash = "^2.18.0"              # ❌ REMOVER
# dash-bootstrap-components     # ❌ REMOVER
# fastapi = "^0.115.0"          # ❌ REMOVER (o mantener solo si usas health checks)
# uvicorn = "^0.32.0"           # ❌ REMOVER
# websockets = "^14.1"          # ❌ REMOVER
# aioredis = "^2.0.1"           # ❌ REMOVER (Redis opcional)
# datashader = "^0.16.3"        # ❌ REMOVER (overkill)
# dask = "^2024.11.2"           # ❌ REMOVER (overkill)
# panel = "^1.5.4"              # ❌ REMOVER
# grafana-client = "^4.0.0"     # ❌ REMOVER (no necesario)
```

---

## ✅ RESUMEN

| Componente | Propósito | URL | Complejidad |
|------------|-----------|-----|-------------|
| **Grafana** | Monitoreo real-time | `localhost:3000` | ⭐⭐ (config inicial) |
| **Prometheus** | Metrics storage | `localhost:9090` | ⭐ (auto-config) |
| **Streamlit** | Backtesting UI | `localhost:8501` | ⭐ (muy simple) |

**Tiempo de implementación**:
- Grafana dashboards: 2-3 horas
- Instrumentar 7 EAs: 1-2 horas
- Streamlit UI: 3-4 horas
- **Total: 6-9 horas** (vs 4-7 días con FastAPI+Dash)

**Código a escribir**:
- `prometheus_metrics.py`: ~150 líneas
- Modificar 7 EAs: ~50 líneas cada uno (350 total)
- `streamlit_backtest.py`: ~300 líneas
- **Total: ~800 líneas** (vs 1,400 con FastAPI+Dash+Backend)

---

## 🎯 DECISIÓN FINAL

**¿Qué eliminamos de `pyproject.toml`?**

Elimino las dependencias excesivas y mantengo solo lo esencial. ¿Procedo? 🚀
