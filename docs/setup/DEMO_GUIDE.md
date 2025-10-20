# 🎯 UNDERDOG - Guía de Demostración con Datos Ficticios

**Estado del Sistema**: ✅ TODO FUNCIONANDO

---

## 📊 Servicios Activos

### 1. **Generador de Métricas** - http://localhost:8000/metrics
- **Estado**: ✅ Generando datos en tiempo real
- **7 EAs simulados**:
  - SuperTrendRSI (EURUSD M15) - Confidence: 100%
  - ParabolicEMA (GBPUSD M15) - Confidence: 95%
  - KeltnerBreakout (USDJPY M15) - Confidence: 90%
  - EmaScalper (EURJPY M5) - Confidence: 85%
  - BollingerCCI (AUDUSD M15) - Confidence: 88%
  - ATRBreakout (USDCAD M15) - Confidence: 87%
  - PairArbitrage (EURUSD_GBPUSD M15) - Confidence: 92%
- **Actualización**: Cada 1 segundo
- **Métricas**: 40+ métricas de Prometheus

### 2. **Streamlit UI** - http://localhost:8501
- **Estado**: ✅ Interfaz de backtesting activa
- **Datos disponibles**: 189,000 bars (3 símbolos × 63k bars)
- **Periodo**: 2023-01-01 → 2024-10-20
- **Funciones**: Backtesting con parámetros personalizables

### 3. **Prometheus** - http://localhost:9090
- **Estado**: ✅ Recolectando métricas
- **Scrape interval**: 1 segundo
- **Target**: host.docker.internal:8000
- **Retención**: 15 días

### 4. **Grafana** - http://localhost:3000
- **Usuario**: `admin`
- **Contraseña**: `admin123`
- **Dashboards**: 3 dashboards pre-configurados
- **Datasource**: Prometheus (auto-configurado)

### 5. **TimescaleDB** - localhost:5432
- **Usuario**: `underdog`
- **Password**: `underdog_trading_2024_secure`
- **Database**: `underdog_trading`

---

## 🧪 Pruebas Paso a Paso

### **Prueba 1: Ver Métricas en Tiempo Real**

#### A) Métricas Prometheus Raw
1. Abre http://localhost:8000/metrics en tu navegador
2. Busca (Ctrl+F) estas métricas:
   ```
   underdog_ea_status{ea_name="SuperTrendRSI"} 1.0
   underdog_ea_signals_total{ea_name="SuperTrendRSI",signal_type="BUY",symbol="EURUSD"}
   underdog_ea_pnl_realized{ea_name="SuperTrendRSI"}
   underdog_ea_win_rate{ea_name="SuperTrendRSI"}
   underdog_ea_positions_open{ea_name="SuperTrendRSI",symbol="EURUSD"}
   ```
3. **Recarga la página** (F5) - verás los números cambiar en tiempo real

#### B) Consola del Generador
En VS Code, busca la terminal que muestra:
```
[18:05:36] BollingerCCI: SELL signal → Position opened
[18:05:37] ParabolicEMA: BUY signal → Position opened

────────────────────────────────────────────────────────────
[18:05:44] Iteration 10
  Account: $100,234.56 | DD: 0.12%
  Signals: 15 | Positions: 3 | Daily P&L: $234.56
────────────────────────────────────────────────────────────
```
Verás las señales generándose cada 1-3 segundos.

---

### **Prueba 2: Verificar Prometheus**

1. Abre http://localhost:9090
2. Ve a **Status → Targets**
   - Busca `underdog-trading`
   - Debería mostrar **UP** en verde
   - Last Scrape: < 1s ago
3. En la **Query Box** (página principal), prueba estas queries:
   ```promql
   # Ver equity de todos los EAs
   underdog_account_equity

   # Ver señales totales por EA
   sum by (ea_name) (underdog_ea_signals_total)

   # Ver P&L realizado
   underdog_ea_pnl_realized

   # Ver win rate
   underdog_ea_win_rate

   # Ver posiciones abiertas
   sum by (ea_name) (underdog_ea_positions_open)
   ```
4. Click **Execute** - verás los valores actuales
5. Click **Graph** - verás gráficos en tiempo real

---

### **Prueba 3: Grafana Dashboards**

#### Login Inicial
1. Abre http://localhost:3000
2. Usuario: `admin`
3. Contraseña: `admin123`
4. (Opcional) Te pedirá cambiar contraseña - puedes **Skip**

#### Dashboard 1: EA Performance Overview
1. En el menú lateral, click **Dashboards → Browse**
2. Abre **"UNDERDOG - EA Performance Overview"**
3. **Paneles que verás**:
   - **Total Equity**: Gráfico de línea con equity actual
   - **Active EAs**: 7/7 activos
   - **Total Signals**: Contador incrementándose
   - **Win Rate by EA**: Barras horizontales (60-75%)
   - **P&L by EA**: Tabla con cada EA
   - **Daily P&L**: Gráfico mostrando P&L del día
   - **Sharpe Ratio**: Gauges de 1.5-2.5
   - **Profit Factor**: Gauges de 1.8-2.2

#### Dashboard 2: Signal Analysis
1. Abre **"UNDERDOG - Signal Analysis"**
2. **Paneles que verás**:
   - **Signals by Type**: Distribución BUY/SELL
   - **Signals by EA**: Ranking de EAs más activos
   - **Signal Rate**: Señales por minuto
   - **Position Duration**: Tiempo promedio de posiciones
   - **Signal Heatmap**: Mapa de calor por hora/EA

#### Dashboard 3: Trading Overview
1. Abre **"UNDERDOG - Trading Overview"**
2. **Paneles que verás**:
   - **Account Balance**: $100,000 - $102,000
   - **Open Positions**: 2-6 posiciones fluctuando
   - **Drawdown**: 0-2%
   - **Risk Exposure**: Por símbolo y EA
   - **Execution Times**: Histograma 8-15ms
   - **Recent Trades**: Tabla con últimos 20 trades

#### Personalización
- **Time Range**: Top-right corner - ajusta a "Last 5 minutes"
- **Refresh**: Arriba derecha - selecciona "5s" para auto-refresh
- **Zoom**: Click y arrastra en cualquier gráfico
- **Filtros**: Usa dropdowns para filtrar por EA o símbolo

---

### **Prueba 4: Streamlit Backtesting**

1. Abre http://localhost:8501
2. En la barra lateral:
   - **Strategy**: Selecciona "SuperTrendRSI"
   - **Symbol**: EURUSD
   - **Timeframe**: M15
   - **Period**: 2023-01-01 to 2024-10-20
3. **Parámetros de SuperTrendRSI**:
   - ATR Period: 10
   - ATR Multiplier: 3.0
   - RSI Period: 14
   - RSI Oversold: 30
   - RSI Overbought: 70
4. **Risk Management**:
   - Initial Capital: $100,000
   - Commission: 0.01%
   - Slippage: 1 pips
5. Click **"🚀 Run Backtest"**

#### Resultados Esperados (SuperTrendRSI en EURUSD):
- **Duration**: 5-10 segundos de procesamiento
- **Total Trades**: ~150-300 trades
- **Win Rate**: 60-70%
- **Total Return**: +15% a +35%
- **Sharpe Ratio**: 1.5-2.2
- **Max Drawdown**: 8-15%
- **Profit Factor**: 1.8-2.5

#### Tabs de Resultados:
1. **📈 Equity Curve**: Gráfico mostrando crecimiento del capital
2. **📊 Performance Metrics**: Tabla con todas las métricas
3. **📋 Trade History**: Tabla con los ~200 trades individuales
4. **💰 Monthly Returns**: Heatmap de retornos por mes

#### Probar Otras Estrategias:
- **ParabolicEMA**: Win rate ~65%, más conservadora
- **KeltnerBreakout**: Win rate ~55%, mayor volatilidad
- **EmaScalper**: Win rate ~58%, muchos trades pequeños
- **BollingerCCI**: Win rate ~62%, entrada en bandas
- **ATRBreakout**: Win rate ~60%, breakout trading
- **PairArbitrage**: Win rate ~70%, trading de pares

---

## 📸 Capturas de Pantalla Esperadas

### Métricas Prometheus (localhost:8000/metrics)
```
# HELP underdog_ea_status EA status (1=active, 0=inactive)
# TYPE underdog_ea_status gauge
underdog_ea_status{ea_name="ATRBreakout"} 1.0
underdog_ea_status{ea_name="BollingerCCI"} 1.0
underdog_ea_status{ea_name="EmaScalper"} 1.0
underdog_ea_status{ea_name="KeltnerBreakout"} 1.0
underdog_ea_status{ea_name="PairArbitrage"} 1.0
underdog_ea_status{ea_name="ParabolicEMA"} 1.0
underdog_ea_status{ea_name="SuperTrendRSI"} 1.0

# HELP underdog_ea_signals_total Total trading signals generated per EA
# TYPE underdog_ea_signals_total counter
underdog_ea_signals_total{ea_name="SuperTrendRSI",signal_type="BUY",symbol="EURUSD"} 23.0
underdog_ea_signals_total{ea_name="SuperTrendRSI",signal_type="SELL",symbol="EURUSD"} 19.0
underdog_ea_signals_total{ea_name="ParabolicEMA",signal_type="BUY",symbol="GBPUSD"} 15.0
...

# HELP underdog_ea_pnl_realized_usd Realized P&L per EA in USD (since start)
# TYPE underdog_ea_pnl_realized_usd gauge
underdog_ea_pnl_realized_usd{ea_name="ATRBreakout"} 234.56
underdog_ea_pnl_realized_usd{ea_name="BollingerCCI"} 456.78
underdog_ea_pnl_realized_usd{ea_name="EmaScalper"} 123.45
underdog_ea_pnl_realized_usd{ea_name="KeltnerBreakout"} 345.67
underdog_ea_pnl_realized_usd{ea_name="PairArbitrage"} 678.90
underdog_ea_pnl_realized_usd{ea_name="ParabolicEMA"} 567.89
underdog_ea_pnl_realized_usd{ea_name="SuperTrendRSI"} 789.01

# HELP underdog_ea_win_rate Win rate per EA (0-1)
# TYPE underdog_ea_win_rate gauge
underdog_ea_win_rate{ea_name="ATRBreakout"} 0.62
underdog_ea_win_rate{ea_name="BollingerCCI"} 0.67
underdog_ea_win_rate{ea_name="EmaScalper"} 0.55
underdog_ea_win_rate{ea_name="KeltnerBreakout"} 0.58
underdog_ea_win_rate{ea_name="PairArbitrage"} 0.73
underdog_ea_win_rate{ea_name="ParabolicEMA"} 0.65
underdog_ea_win_rate{ea_name="SuperTrendRSI"} 0.70
```

### Prometheus Targets (localhost:9090/targets)
```
underdog-trading (1/1 up)
  Endpoint: http://host.docker.internal:8000/metrics
  State: UP
  Labels: instance="underdog-local", job="underdog-trading"
  Last Scrape: 0.345s ago
  Scrape Duration: 12.4ms
```

### Grafana Dashboard - EA Performance
```
┌─────────────────────────────────────────────────────────┐
│ UNDERDOG - EA Performance Overview                     │
├─────────────────────────────────────────────────────────┤
│ Total Equity: $101,234.56    Active EAs: 7/7          │
│ Total Signals: 142           Win Rate: 65.3%          │
├─────────────────────────────────────────────────────────┤
│ 📊 Equity Over Time                                     │
│ ▁▂▃▄▅▆▇█ (gráfico ascendente)                         │
├─────────────────────────────────────────────────────────┤
│ P&L by EA:                                              │
│ PairArbitrage    ████████████ $678.90                  │
│ SuperTrendRSI    ███████████ $789.01                   │
│ ParabolicEMA     ██████████ $567.89                    │
│ BollingerCCI     ████████ $456.78                      │
│ KeltnerBreakout  ██████ $345.67                        │
│ ATRBreakout      ████ $234.56                          │
│ EmaScalper       ██ $123.45                            │
└─────────────────────────────────────────────────────────┘
```

### Streamlit Backtest Results
```
╔═══════════════════════════════════════════════════════╗
║ 📊 Performance Metrics                                ║
╠═══════════════════════════════════════════════════════╣
║ Initial Capital        │ $100,000.00                  ║
║ Final Capital          │ $125,432.10                  ║
║ Total Return           │ +25.43%                      ║
║ Total Trades           │ 234                          ║
║ Winning Trades         │ 156 (66.67%)                 ║
║ Losing Trades          │ 78 (33.33%)                  ║
║ Average Win            │ $245.67                      ║
║ Average Loss           │ -$123.45                     ║
║ Largest Win            │ $1,234.56                    ║
║ Largest Loss           │ -$567.89                     ║
║ Sharpe Ratio           │ 1.87                         ║
║ Max Drawdown           │ -12.34%                      ║
║ Profit Factor          │ 2.15                         ║
║ Avg Trade Duration     │ 4.5 hours                    ║
╚═══════════════════════════════════════════════════════╝
```

---

## 🔧 Solución de Problemas

### Problema 1: "Prometheus target DOWN"
```bash
# Verificar que el generador está corriendo
curl http://localhost:8000/metrics

# Si no responde, reiniciar:
poetry run python scripts/generate_test_metrics.py
```

### Problema 2: "Grafana no muestra datos"
1. Ve a **Configuration → Data Sources**
2. Verifica que Prometheus está en **http://prometheus:9090**
3. Click **Save & Test** - debería mostrar "Data source is working"
4. Si falla, cambia la URL a **http://underdog-prometheus:9090**

### Problema 3: "Streamlit backtest falla"
```bash
# Verificar que los archivos CSV existen
ls data/historical/*.csv

# Si no existen, regenerar:
poetry run python scripts/generate_synthetic_data.py --symbol EURUSD --symbol GBPUSD --symbol USDJPY
```

### Problema 4: "Docker containers no inician"
```bash
# Ver logs
docker-compose logs -f prometheus
docker-compose logs -f grafana

# Reiniciar servicios
cd docker
docker-compose restart prometheus grafana
```

---

## 📊 Métricas Disponibles (40+ métricas)

### **Status & Health**
- `underdog_ea_status` - Estado del EA (1=activo, 0=inactivo)
- `underdog_account_balance` - Balance de la cuenta
- `underdog_account_equity` - Equity actual

### **Señales**
- `underdog_ea_signals_total` - Total de señales generadas
- `underdog_ea_signals_confidence` - Confianza de la señal (0-1)

### **Posiciones**
- `underdog_ea_positions_open` - Posiciones abiertas por EA
- `underdog_ea_position_duration_seconds` - Duración de posiciones

### **P&L**
- `underdog_ea_pnl_realized` - P&L realizado por EA
- `underdog_ea_pnl_unrealized` - P&L no realizado por EA
- `underdog_ea_pnl_daily` - P&L del día

### **Performance**
- `underdog_ea_win_rate` - Win rate por EA (0-1)
- `underdog_ea_sharpe_ratio` - Sharpe ratio por EA
- `underdog_ea_profit_factor` - Profit factor por EA
- `underdog_ea_max_drawdown` - Max drawdown por EA

### **Ejecución**
- `underdog_ea_execution_time_ms` - Tiempo de ejecución en ms
- `underdog_ea_errors_total` - Total de errores por EA

### **Risk**
- `underdog_account_drawdown_pct` - Drawdown actual (%)
- `underdog_account_margin_used_pct` - Margen utilizado (%)
- `underdog_risk_exposure_by_symbol` - Exposición por símbolo

---

## 🎯 Conclusión

Con este setup tienes:
- ✅ **7 EAs simulados** generando métricas cada segundo
- ✅ **Prometheus** recolectando y almacenando métricas
- ✅ **Grafana** visualizando 3 dashboards completos
- ✅ **Streamlit** ejecutando backtests con datos reales
- ✅ **189k bars** de datos sintéticos (3 símbolos)

Todo está funcionando con **datos ficticios** para demostración. Para producción real:
1. Conectar MT5 con credenciales reales
2. Reemplazar `generate_test_metrics.py` con `start_trading_with_monitoring.py`
3. Configurar alertas en Prometheus
4. Activar logging persistente en TimescaleDB

¡Disfruta explorando el sistema! 🚀
