# 🎯 Sistema UNDERDOG - Guía de Ejecución Completa

**Fecha**: 20 de Octubre 2025  
**Estado**: Listo para testing  
**Archivos creados**: 5 archivos nuevos (~1,500 líneas)

---

## ✅ Archivos Creados

### 1. **`docker/.env`** (30 líneas)
```bash
# Credenciales para Docker Compose

```

### 2. **`scripts/generate_test_metrics.py`** (~400 líneas)
- Simula 7 EAs generando métricas Prometheus
- Actualiza cada 1 segundo
- Para testing de dashboards sin esperar trading real
- **Características**:
  - SuperTrendRSI (EURUSD M15) - Win rate 68%, Sharpe 1.85
  - ParabolicEMA (GBPUSD M15) - Win rate 64%, Sharpe 1.72
  - KeltnerBreakout (USDJPY M15) - Win rate 61%, Sharpe 1.55
  - EmaScalper (EURJPY M5) - Win rate 58%, Sharpe 1.42 (scalper)
  - BollingerCCI (AUDUSD M15) - Win rate 62%, Sharpe 1.68
  - ATRBreakout (USDCAD M15) - Win rate 60%, Sharpe 1.50
  - PairArbitrage (EURUSD/GBPUSD M15) - Win rate 72%, Sharpe 2.05

### 3. **`scripts/generate_synthetic_data.py`** (~400 líneas)
- Genera datos históricos OHLCV sintéticos realistas
- **Modelo estadístico**:
  - Geometric Brownian Motion (tendencia + volatilidad)
  - GARCH-like volatility clustering
  - Intraday seasonality (Londres/NY = alta vol, Asia = baja vol)
  - Bid/Ask spread realista por símbolo
- **Parámetros por símbolo**:
  - EURUSD: spread 0.8 pips, vol diaria 0.6%
  - GBPUSD: spread 1.2 pips, vol diaria 0.8%
  - USDJPY: spread 0.9 pips, vol diaria 0.7%

### 4. **`underdog/backtesting/simple_runner.py`** (~350 líneas)
- Wrapper simple para EventDrivenBacktest
- **Interfaz simplificada**:
  ```python
  results = run_simple_backtest(
      ea_class=FxSuperTrendRSI,
      ea_config=config,
      ohlcv_data=df,
      initial_capital=100000
  )
  # Returns: equity_curve, trades, metrics, final_capital
  ```
- Convierte OHLCV → ticks (bid/ask)
- Maneja loop asíncrono automáticamente
- Output formateado para Streamlit

### 5. **`scripts/start_trading_with_monitoring.py`** (actualizado)
- **ANTES**: Solo 1 EA (SuperTrendRSI)
- **AHORA**: 7 EAs configurados
  - Diferentes símbolos: EURUSD, GBPUSD, USDJPY, EURJPY, AUDUSD, USDCAD, pair
  - Diferentes timeframes: M5 (scalper), M15 (resto)
  - Magic numbers: 101-107

---

## 🚀 Workflow de Ejecución

### **PASO 1: Generar Datos Históricos** (5 minutos)

```powershell
# Opción A: Generar 1 símbolo
poetry run python scripts/generate_synthetic_data.py `
    --symbol EURUSD `
    --timeframe M15 `
    --start-date 2023-01-01 `
    --end-date 2024-10-20 `
    --output data/historical/

# Opción B (RECOMENDADO): Generar 3 símbolos (EURUSD, GBPUSD, USDJPY)
poetry run python scripts/generate_synthetic_data.py --multiple
```

**Salida esperada**:
```
✓ Generated 24,000+ bars
✓ Saved to: data/historical/EURUSD_M15.csv
File size: ~2.5 MB
```

---

### **PASO 2: Test Dashboards de Grafana** (10 minutos)

#### 2.1 Generar métricas de prueba (background)
```powershell
# Lanzar en background
poetry run python scripts/generate_test_metrics.py
# Dejarlo corriendo (Ctrl+C para detener)
```

**Salida esperada**:
```
🚀 Starting Prometheus metrics server on port 8000...
✅ Metrics server running: http://localhost:8000/metrics
📊 Initializing 7 EAs...
▶️  Simulation started - Generating metrics every 1 second
[14:32:15] SuperTrendRSI: BUY signal → Position opened
[14:32:18] ParabolicEMA: SELL signal → Position opened
[14:32:22] KeltnerBreakout: Position closed → P&L: $85.23
```

#### 2.2 Verificar métricas en Prometheus
```powershell
# En PowerShell, abrir navegador:
Start-Process "http://localhost:8000/metrics"
```

Deberías ver algo como:
```
# HELP ea_status EA status (1=active, 0=inactive)
ea_status{ea_name="SuperTrendRSI",broker="MetaQuotes-Demo",account_id="10007888612"} 1.0
ea_status{ea_name="ParabolicEMA",...} 1.0
...

# HELP ea_signals_total Total signals generated
ea_signals_total{ea_name="SuperTrendRSI",signal_type="BUY",...} 15.0
...
```

#### 2.3 Lanzar Docker Compose
```powershell
cd docker
docker-compose up -d prometheus grafana
```

**Verificar servicios**:
```powershell
docker ps  # Deberías ver: underdog-prometheus, underdog-grafana
```

#### 2.4 Acceder a Grafana
```powershell
Start-Process "http://localhost:3000"
```

**Login**:
- Usuario: `admin`
- Password: `admin123`

**Dashboards a verificar**:
1. **Portfolio Overview** (`/d/underdog-portfolio`)
   - Account Balance, Equity
   - Daily/Total Drawdown %
   - Equity Curve
   - EA Status

2. **EA Performance Matrix** (`/d/underdog-ea-performance`)
   - Performance Table (7 EAs)
   - Signal Generation Rate
   - Execution Time Percentiles
   - Daily P&L por EA

3. **Open Positions** (`/d/underdog-positions`)
   - Positions by EA
   - Unrealized P&L Evolution
   - P&L Contribution

---

### **PASO 3: Test Streamlit Backtesting** (15 minutos)

#### 3.1 Actualizar Streamlit (PENDIENTE)
```python
# TODO: Reemplazar dummy data en streamlit_backtest.py
# Líneas ~230-280: Generar equity curve con simple_runner.py
# Botón "Run Backtest" debe llamar a run_simple_backtest()
```

#### 3.2 Lanzar Streamlit
```powershell
poetry run streamlit run underdog/ui/streamlit_backtest.py
```

#### 3.3 Verificar funcionalidad
- ✅ Seleccionar EA (ej: SuperTrendRSI)
- ✅ Ajustar parámetros (RSI period, ATR multiplier)
- ✅ Click "Run Backtest"
- ✅ Ver equity curve REAL (no dummy)
- ✅ Ver trades table con datos reales
- ✅ Heatmap de sensibilidad de parámetros

---

### **PASO 4: Sistema Completo con MT5** (OPCIONAL, 20 minutos)

#### 4.1 Conectar a MT5
```powershell
# Asegúrate de que MT5 está abierto y logueado con:
# Login: 10007888612
# Password: @bNeV5Gk
# Server: MetaQuotes-Demo
```

#### 4.2 Lanzar trading system con 7 EAs
```powershell
poetry run python scripts/start_trading_with_monitoring.py
```

**Salida esperada**:
```
🚀 UNDERDOG TRADING SYSTEM WITH PROMETHEUS MONITORING
✅ MT5 initialized
📊 Started 7 EAs:
  - SuperTrendRSI (EURUSD M15)
  - ParabolicEMA (GBPUSD M15)
  - KeltnerBreakout (USDJPY M15)
  - EmaScalper (EURJPY M5)
  - BollingerCCI (AUDUSD M15)
  - ATRBreakout (USDCAD M15)
  - PairArbitrage (EURUSD/GBPUSD M15)

✅ Prometheus metrics: http://localhost:8000/metrics
📈 Grafana dashboard: http://localhost:3000
```

#### 4.3 Monitorear en Grafana
- Ver dashboards actualizándose con datos REALES de MT5
- Verificar señales de trading
- Monitorear drawdown en tiempo real

---

## 📊 Resumen de URLs

| Servicio | URL | Credenciales |
|----------|-----|--------------|
| **Prometheus** | http://localhost:9090 | N/A |
| **Grafana** | http://localhost:3000 | admin / admin123 |
| **Prometheus Metrics** | http://localhost:8000/metrics | N/A |
| **Streamlit** | http://localhost:8501 | N/A |

---

## 🐛 Troubleshooting

### Prometheus no scrapeando métricas
```powershell
# Verificar que generate_test_metrics.py está corriendo
curl http://localhost:8000/metrics

# Verificar target en Prometheus
Start-Process "http://localhost:9090/targets"
# Debe mostrar: underdog-trading (UP)
```

### Grafana dashboards vacíos
```powershell
# 1. Verificar datasource
# Grafana → Configuration → Data sources → Prometheus
# URL debe ser: http://prometheus:9090 (dentro de Docker)

# 2. Verificar que hay métricas
# En Prometheus: http://localhost:9090/graph
# Query: ea_status
# Debe mostrar datos
```

### Docker Compose falla
```powershell
# Verificar logs
docker-compose logs prometheus
docker-compose logs grafana

# Recrear contenedores
docker-compose down
docker-compose up -d
```

### Streamlit error al cargar datos
```powershell
# Verificar que existen datos históricos
ls data/historical/

# Si está vacío, generar datos:
poetry run python scripts/generate_synthetic_data.py --multiple
```

---

## ✅ Checklist de Testing

### Fase 1: Dashboards
- [ ] `generate_test_metrics.py` corriendo
- [ ] http://localhost:8000/metrics muestra métricas
- [ ] Docker Prometheus + Grafana levantados
- [ ] http://localhost:3000 accesible
- [ ] Dashboard "Portfolio Overview" con datos
- [ ] Dashboard "EA Performance Matrix" con 7 EAs
- [ ] Dashboard "Open Positions" con posiciones simuladas

### Fase 2: Streamlit Backtesting
- [ ] Datos históricos generados en `data/historical/`
- [ ] `streamlit_backtest.py` actualizado con `simple_runner.py`
- [ ] http://localhost:8501 accesible
- [ ] Botón "Run Backtest" funciona
- [ ] Equity curve muestra datos reales
- [ ] Trades table con datos reales
- [ ] Heatmap de sensibilidad generado

### Fase 3: Sistema Completo (Opcional)
- [ ] MT5 conectado con cuenta demo
- [ ] `start_trading_with_monitoring.py` corriendo con 7 EAs
- [ ] Grafana muestra métricas REALES de MT5
- [ ] Señales de trading generándose
- [ ] Drawdown monitoreándose en tiempo real

---

## 🎯 Próximos Pasos

### Inmediato (HOY)
1. ✅ Ejecutar Workflow Paso 1: Generar datos
2. ✅ Ejecutar Workflow Paso 2: Test dashboards
3. ⏳ **PENDIENTE**: Actualizar `streamlit_backtest.py` (Paso 3)

### Corto Plazo (ESTA SEMANA)
- Implementar cálculo real de drawdown (no dummy)
- Añadir alertas en Grafana (DD > 5%, DD > 10%)
- Crear 4º dashboard: Trade Analytics (duration, P&L distribution)

### Medio Plazo (PRÓXIMA SEMANA)
- Conectar a cuenta FTMO demo real
- Backtesting walk-forward con 7 EAs
- Optimización de parámetros con Optuna
- Reportes automáticos diarios (PDF)

---

## 📝 Notas Finales

**Total de líneas agregadas**: ~1,500 líneas  
**Archivos modificados**: 6 archivos  
**Tiempo estimado de testing**: 30-45 minutos  

**Recomendación**: Empezar con Workflow Paso 1 + Paso 2 (dashboards) ya que es lo más rápido de verificar. El Paso 3 (Streamlit) requiere actualizar el código primero.

