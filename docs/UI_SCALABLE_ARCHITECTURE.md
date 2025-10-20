# ARQUITECTURA UI ESCALABLE - UNDERDOG TRADING SYSTEM
**Documento Técnico de Diseño**  
**Fecha**: Octubre 20, 2025  
**Versión**: 4.0.0

---

## 🎯 OBJETIVOS DEL SISTEMA

### Requisitos Críticos:
1. **Baja Latencia**: <10ms para actualizaciones de streaming (WebSocket)
2. **Alta Escalabilidad**: Soportar 100+ estrategias simultáneas, múltiples Prop Firms, múltiples brokers
3. **Desacoplamiento Total**: Motor de trading independiente de la UI
4. **Robustez Operacional**: Monitoreo en tiempo real sin bloquear ejecución
5. **Interactividad Analítica**: Backtesting interactivo con visualización de sensibilidad paramétrica

---

## 📐 ARQUITECTURA DE MICROSERVICIOS

```
┌─────────────────────────────────────────────────────────────────────┐
│                         UNDERDOG SYSTEM                             │
│              Multi-Strategy Trading with Scalable UI                │
└─────────────────────────────────────────────────────────────────────┘
                                │
                ┌───────────────┴───────────────┐
                │                               │
         ┌──────▼──────┐                 ┌─────▼──────┐
         │   LAYER 1   │                 │  LAYER 2   │
         │ TRADING     │◄──── ZeroMQ ───►│  EXECUTION │
         │ STRATEGIES  │     (Signals)   │   (MT5)    │
         │  (Python)   │                 │   (MQL5)   │
         └──────┬──────┘                 └─────┬──────┘
                │                               │
                │  Publish Events               │
                │  (Orders, P&L, etc.)          │
                │                               │
         ┌──────▼───────────────────────────────▼──────┐
         │          REDIS PUB/SUB BACKBONE              │
         │  (High-Availability Message Bus)             │
         └──────┬───────────────────────────────────────┘
                │  Subscribe to Channels
                │  (order:new, pnl:update, signal:generated)
                │
         ┌──────▼──────┐
         │   LAYER 3   │
         │  FASTAPI    │  ◄──── REST API (Config)
         │  GATEWAY    │  ◄──── WebSocket (Streaming)
         └──────┬──────┘
                │  WSS (Full-Duplex)
                │  Bidirectional Control
                │
         ┌──────▼──────────────────────────────────────┐
         │              LAYER 4 - FRONTEND              │
         │  ┌─────────────────────────────────────┐    │
         │  │  DASH (PLOTLY) - PRODUCTION UI      │    │
         │  │  ────────────────────────────────   │    │
         │  │  ✅ TradingView Lightweight Charts  │    │
         │  │  ✅ WebGL-accelerated rendering     │    │
         │  │  ✅ Heatmaps for parameter analysis │    │
         │  │  ✅ Real-time P&L monitoring        │    │
         │  └─────────────────────────────────────┘    │
         │                                              │
         │  ┌─────────────────────────────────────┐    │
         │  │  GRAFANA - OBSERVABILITY (Keep)     │    │
         │  │  ────────────────────────────────   │    │
         │  │  ✅ Prometheus metrics (CPU, RAM)   │    │
         │  │  ✅ Infrastructure health           │    │
         │  │  ✅ RED/USE methodology             │    │
         │  └─────────────────────────────────────┘    │
         └──────────────────────────────────────────────┘
                │
         ┌──────▼──────┐
         │   LAYER 5   │
         │  ANALYTICS  │
         │   STORAGE   │
         │  ─────────  │
         │  DuckDB     │  ◄──── Historical Backtesting
         │  Redis      │  ◄──── Live State Cache
         │  PostgreSQL │  ◄──── Transactional Data
         └─────────────┘
```

---

## 🔧 STACK TECNOLÓGICO

### Core Components:

| Layer | Component | Technology | Justification |
|-------|-----------|------------|---------------|
| **Strategy Engine** | Trading Logic | Python 3.11+ + **TA-Lib** | Industry-standard indicators (C-based, 10-50x faster) |
| **Message Bus** | Pub/Sub | **Redis Pub/Sub** | High-throughput, HA backbone, <1ms latency |
| **API Gateway** | Backend | **FastAPI + Uvicorn** | ASGI async, native WebSockets, 3x faster than Flask |
| **Frontend** | Production UI | **Dash (Plotly)** | Scalable, WebGL support, production-grade |
| **Charting** | Financial Viz | **Dash TradingView LWC** | Optimized for real-time tick data, GPU-accelerated |
| **Big Data** | Analytics | **Datashader + Dask** | Handle millions of data points without lag |
| **Storage** | Time-Series | **DuckDB** | 100x faster queries than Pandas |
| **Cache** | State | **Redis** | In-memory, <1ms read/write |
| **Observability** | Monitoring | **Prometheus + Grafana** | Keep existing infrastructure health monitoring |

---

## 🚀 PROTOCOLO DE COMUNICACIÓN

### WebSocket (WSS) - Bidirectional Control

**Ventajas**:
- ✅ Full-duplex: UI puede enviar comandos al motor (stop EA, adjust TP/SL, etc.)
- ✅ Baja latencia: <5ms para actualizaciones críticas
- ✅ Conexión persistente: Sin overhead de HTTP polling

**Flujo de Datos**:
```python
# Trading Engine → Redis Pub/Sub → FastAPI → WebSocket → Dash UI

# Example: Order Execution Event
{
    "event": "order:executed",
    "timestamp": 1729468800000,
    "ea_name": "SuperTrendRSI",
    "symbol": "EURUSD",
    "ticket": 123456789,
    "type": "BUY",
    "volume": 0.05,
    "entry_price": 1.10250,
    "sl": 1.10100,
    "tp": 1.10400,
    "confidence": 0.95
}

# Example: P&L Update
{
    "event": "pnl:update",
    "timestamp": 1729468801000,
    "account_balance": 100500.00,
    "equity": 100520.50,
    "daily_dd": -0.5,
    "total_dd": -2.3,
    "floating_dd": -0.02,
    "sharpe_ratio": 1.85
}
```

---

## 📊 DISEÑO DE LA UI (DASH + TVLWC)

### Layout Modular (Single-Page Dashboard):

```
┌─────────────────────────────────────────────────────────────────┐
│  HEADER: UNDERDOG Trading System | Account: $100,520 (+0.52%)  │
└─────────────────────────────────────────────────────────────────┘
┌──────────────────────┬──────────────────────────────────────────┐
│  LEFT SIDEBAR        │  MAIN CONTENT AREA                       │
│  (Control Panel)     │                                          │
│  ──────────────      │  ┌────────────────────────────────────┐ │
│  ✅ EA Status        │  │ TradingView Lightweight Chart      │ │
│  SuperTrendRSI: ON   │  │ (Real-time EURUSD M5)              │ │
│  ParabolicEMA: ON    │  │ ────────────────────────────────   │ │
│  PairArbitrage: OFF  │  │ WebGL-accelerated rendering        │ │
│  ...                 │  │ Overlays: Signals, SL/TP markers   │ │
│                      │  └────────────────────────────────────┘ │
│  📊 Risk Metrics     │                                          │
│  Daily DD: -0.5%     │  ┌─────────┬─────────┬─────────┬──────┐│
│  Total DD: -2.3%     │  │ P&L     │ Win Rate│ Sharpe  │ Calmar│
│  Floating DD: -0.02% │  │ +$520   │ 58%     │ 1.85    │ 2.10  │
│                      │  └─────────┴─────────┴─────────┴──────┘│
│  🎯 Position Control │                                          │
│  Open Positions: 3   │  ┌────────────────────────────────────┐ │
│  Total Exposure: 0.15│  │ ACTIVE POSITIONS TABLE             │ │
│  Max Risk: 1.5%      │  │ Ticket | EA | Symbol | Type | P&L  │ │
│                      │  │ ────────────────────────────────── │ │
│  🔧 Quick Actions    │  │ 123456 | SuperTrend | EURUSD| +$20 │ │
│  [Close All]         │  └────────────────────────────────────┘ │
│  [Pause Trading]     │                                          │
│  [Emergency Stop]    │  ┌────────────────────────────────────┐ │
│                      │  │ BACKTESTING HEATMAP                │ │
│                      │  │ (Parameter Sensitivity Analysis)   │ │
│                      │  │ ────────────────────────────────   │ │
│                      │  │ RSI_Period vs SuperTrend_Mult      │ │
│                      │  │ Color: Sharpe Ratio (0.0 - 3.0)    │ │
│                      │  └────────────────────────────────────┘ │
└──────────────────────┴──────────────────────────────────────────┘
```

### Key Components:

1. **TradingView Lightweight Charts** (`dash-tvlwc`)
   - Real-time candlestick rendering
   - Signal overlays (BUY/SELL markers)
   - SL/TP levels visualization
   - Volume histogram
   - Timeframe toggles (M5, M15, H1, H4, D1)

2. **KPI Cards** (Bootstrap Components)
   - Live metrics: Balance, Equity, P&L
   - Color-coded alerts (Green/Yellow/Red)
   - Gauge charts for DD thresholds

3. **Heatmap for Parameter Optimization** (Plotly)
   - 2D grid: Parameter1 vs Parameter2
   - Color intensity: Sharpe Ratio
   - Interactive drill-down to individual backtest results
   - **Purpose**: Identify robust parameter plateaus (not peaks)

4. **Active Positions Table** (Dash DataTable)
   - Filterable, sortable, paginated
   - One-click position closure
   - Trailing stop modification

---

## 🔌 FASTAPI BACKEND ARCHITECTURE

### Endpoints:

#### REST API (Configuration):
```python
# GET /api/v1/strategies - List all EAs
# POST /api/v1/strategies/{ea_name}/start - Start EA
# POST /api/v1/strategies/{ea_name}/stop - Stop EA
# PUT /api/v1/strategies/{ea_name}/params - Update parameters

# GET /api/v1/backtest/{ea_name} - Get backtest results
# POST /api/v1/backtest/{ea_name}/run - Run new backtest with params

# GET /api/v1/positions - Get all open positions
# DELETE /api/v1/positions/{ticket} - Close specific position
```

#### WebSocket (Streaming):
```python
# WS /ws/live - Real-time streaming of:
#   - Price updates (tick data)
#   - P&L updates (every 1s)
#   - Order executions
#   - Signal generation
#   - DD alerts

# WS /ws/control - Bidirectional control:
#   - Client → Server: Commands (stop EA, modify SL/TP)
#   - Server → Client: Acknowledgements
```

### Implementation Pattern:

```python
# fastapi_gateway.py
from fastapi import FastAPI, WebSocket
from aioredis import Redis
import asyncio

app = FastAPI()
redis = Redis.from_url("redis://localhost:6379")

@app.websocket("/ws/live")
async def websocket_live(websocket: WebSocket):
    await websocket.accept()
    
    # Subscribe to Redis channels
    pubsub = redis.pubsub()
    await pubsub.subscribe("order:*", "pnl:*", "signal:*")
    
    try:
        async for message in pubsub.listen():
            if message["type"] == "message":
                # Forward to WebSocket client
                await websocket.send_json(message["data"])
    except WebSocketDisconnect:
        await pubsub.unsubscribe()

@app.post("/api/v1/strategies/{ea_name}/stop")
async def stop_ea(ea_name: str):
    # Publish command to Redis
    await redis.publish(f"command:stop", {"ea_name": ea_name})
    return {"status": "command_sent", "ea": ea_name}
```

---

## 📈 INTEGRACIÓN TA-LIB EN ESTRATEGIAS

### Ventajas de TA-Lib:
- **Rendimiento**: Implementado en C, 10-50x más rápido que NumPy
- **Precisión**: Cálculos validados por la industria (Wilder's smoothing, etc.)
- **Estandarización**: Mismo código usado por Bloomberg, Interactive Brokers

### Ejemplo de Uso:

```python
import talib
import numpy as np

# Before (NumPy - Custom Implementation):
def calculate_rsi_numpy(close, period=14):
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = np.convolve(gain, np.ones(period)/period, mode='valid')
    avg_loss = np.convolve(loss, np.ones(period)/period, mode='valid')
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi  # ~5ms for 1000 bars

# After (TA-Lib - Industry Standard):
def calculate_rsi_talib(close, period=14):
    rsi = talib.RSI(close, timeperiod=period)
    return rsi  # ~0.1ms for 1000 bars (50x faster)

# Same for all indicators:
atr = talib.ATR(high, low, close, timeperiod=14)
adx = talib.ADX(high, low, close, timeperiod=14)
sar = talib.SAR(high, low, acceleration=0.02, maximum=0.2)
ema = talib.EMA(close, timeperiod=50)
bbands = talib.BBANDS(close, timeperiod=20, nbdevup=2, nbdevdn=2)
cci = talib.CCI(high, low, close, timeperiod=14)
```

---

## 🎯 PREVENCIÓN DE SESGOS EN BACKTESTING

### UI Design for Robustness:

1. **Heatmap de Sensibilidad Paramétrica**
   - **Objetivo**: Identificar mesetas robustas (no picos aislados)
   - **Visualización**: Plotly Heatmap con color = Sharpe Ratio
   - **Interpretación**: Región amplia de alto Sharpe = Estrategia robusta

2. **Walk-Forward Analysis Visualization**
   - **Train Window**: Rolling window de 6 meses
   - **Test Window**: Out-of-sample de 3 meses
   - **Overlay**: Performance train vs test
   - **Red Flag**: Gran divergencia = Overfitting

3. **Costos Reales Visibles**
   - **Commission**: $7 per lot (FTMO standard)
   - **Slippage**: 2 pips (average)
   - **Spread**: Dynamic (real broker data)
   - **Comparison**: P&L con/sin costos side-by-side

4. **Look-Ahead Bias Prevention**
   - **Rule**: Usar precio de apertura de la SIGUIENTE barra
   - **UI Enforcement**: Backtesting engine valida point-in-time
   - **Alert**: Warning si strategy usa `close[-1]` en decisión actual

---

## 🔍 MÉTRICAS CRÍTICAS (KPIs)

### Visualización Recomendada:

| Métrica | Tipo | Visualización | Threshold Alert |
|---------|------|---------------|-----------------|
| **Sharpe Ratio** | Running | Line Chart (1min rolling) | < 1.0 (Yellow), < 0.5 (Red) |
| **Calmar Ratio** | Running | Line Chart | < 1.5 (Yellow), < 1.0 (Red) |
| **Daily DD** | Real-time | Gauge (0-5%) | > 3% (Yellow), > 4.5% (Red) |
| **Total DD** | Real-time | Gauge (0-10%) | > 7% (Yellow), > 9% (Red) |
| **Floating DD** | Real-time | Gauge (0-1%) | > 0.7% (Yellow), > 0.9% (Red) |
| **Win Rate** | Rolling | Bar Chart (per EA) | < 50% (Yellow), < 40% (Red) |
| **Profit Factor** | Rolling | Bar Chart | < 1.3 (Yellow), < 1.1 (Red) |
| **Avg Trade Duration** | Histogram | Histogram | Detect outliers |
| **Slippage** | Real-time | Line Chart | > 3 pips (Yellow), > 5 pips (Red) |

---

## 🛠️ ROADMAP DE IMPLEMENTACIÓN

### Fase 1: Backend Escalable (Semana 1)
- [x] Redis Pub/Sub setup
- [ ] FastAPI Gateway con WebSocket
- [ ] Event publishing desde EAs
- [ ] Health check endpoints

### Fase 2: Frontend Básico (Semana 2)
- [ ] Dash app structure
- [ ] TradingView Lightweight Charts integration
- [ ] KPI cards (Balance, DD, Sharpe)
- [ ] WebSocket client connection

### Fase 3: Estrategias con TA-Lib (Semana 3-4)
- [ ] Reimplementar las 7 EAs con TA-Lib
- [ ] Benchmarking (TA-Lib vs NumPy)
- [ ] Backtesting engine integration
- [ ] Parameter optimization visualizer

### Fase 4: Observabilidad Completa (Semana 5)
- [ ] Prometheus metrics export
- [ ] Grafana dashboards (infrastructure)
- [ ] Alerting rules (DD, slippage, etc.)
- [ ] Logging pipeline (structured logs)

### Fase 5: Multi-Strategy Management (Semana 6)
- [ ] Portfolio view (all EAs)
- [ ] Correlation matrix heatmap
- [ ] Dynamic allocation (rebalance EAs)
- [ ] Multi-Prop-Firm support

---

## 📚 REFERENCIAS TÉCNICAS

1. **FastAPI + WebSockets**: https://fastapi.tiangolo.com/advanced/websockets/
2. **Dash TradingView LWC**: https://github.com/pip-install-python/dash-tvlwc
3. **Redis Pub/Sub Pattern**: https://redis.io/docs/manual/pubsub/
4. **TA-Lib Documentation**: https://ta-lib.org/function.html
5. **Datashader for Big Data**: https://datashader.org/
6. **WebGL Performance**: https://plotly.com/python/webgl-vs-svg/

---

**PRÓXIMO PASO**: Reimplementar las 7 estrategias con TA-Lib + Event Publishing a Redis

---

*Actualizado: Octubre 20, 2025*  
*Autor: UNDERDOG Development Team*  
*Versión: 4.0.0 - Arquitectura UI Escalable*
