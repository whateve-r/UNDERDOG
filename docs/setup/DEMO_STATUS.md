# 🎯 UNDERDOG - Estado del Sistema de Demostración

**Fecha**: 20 de Octubre de 2025, 18:14 PM

---

## ✅ SERVICIOS FUNCIONANDO

### 1. **Generador de Métricas** ✅
- **URL**: http://localhost:8000/metrics
- **Estado**: ✅ FUNCIONANDO
- **Descripción**: 7 EAs simulados generando métricas cada 1 segundo
- **Terminal activo**: Ver salida en VS Code terminal
- **Métricas actuales**:
  - Account: ~$101,000
  - Signals: ~67 señales generadas
  - Positions: 1-9 posiciones abiertas
  - Daily P&L: +$1,008

### 2. **Prometheus** ✅
- **URL**: http://localhost:9090
- **Estado**: ✅ FUNCIONANDO
- **Targets**:
  - ✅ prometheus (localhost:9090) - UP
  - ✅ underdog-trading (host.docker.internal:8000) - Configurado
- **Scrape interval**: 1 segundo
- **Retención**: 15 días

### 3. **Grafana** ✅
- **URL**: http://localhost:3000
- **Estado**: ✅ FUNCIONANDO
- **Credenciales**:
  - Usuario: `admin`
  - Contraseña: `admin123`
- **Dashboards disponibles**: 3 dashboards (EA Performance, Signal Analysis, Trading Overview)

### 4. **TimescaleDB** ✅
- **Puerto**: localhost:5432
- **Estado**: ✅ FUNCIONANDO (healthy)
- **Credenciales**:
  - Usuario: `underdog`
  - Password: `underdog_trading_2024_secure`
  - Database: `underdog_trading`

---

## ❌ SERVICIOS CON PROBLEMAS

### 5. **Streamlit** ❌
- **URL**: http://localhost:8501
- **Estado**: ❌ SE DETIENE INMEDIATAMENTE
- **Problema**: El proceso se inicia pero se cierra automáticamente
- **Posible causa**: Conflicto de dependencias o error en el código
- **Solución temporal**: Usar Grafana para visualización en su lugar

---

## 🧪 CÓMO PROBAR EL SISTEMA

### **Opción 1: Ver Métricas en Tiempo Real** (MÁS FÁCIL)

#### A) Métricas Raw de Prometheus
1. Abre tu navegador Chrome/Edge
2. Ve a: **http://localhost:8000/metrics**
3. Busca (Ctrl+F): `underdog_ea_pnl_realized`
4. Recarga (F5) cada 2-3 segundos
5. **Verás** los números cambiando en tiempo real

**Ejemplo de lo que verás**:
```
underdog_ea_pnl_realized{ea_name="SuperTrendRSI"} 234.56
underdog_ea_pnl_realized{ea_name="ParabolicEMA"} 456.78
underdog_ea_pnl_realized{ea_name="KeltnerBreakout"} 123.45
```

#### B) Consola del Generador
En VS Code, busca la terminal que muestra:
```
[18:14:06] ParabolicEMA: BUY signal → Position opened
[18:14:06] EmaScalper: Position closed → P&L: $83.04

────────────────────────────────────────────────────────────
[18:14:05] Iteration 80
  Account: $100,979.37 | DD: -0.25%
  Signals: 67 | Positions: 1 | Daily P&L: $1,008.86
────────────────────────────────────────────────────────────
```

---

### **Opción 2: Prometheus UI** (INTERMEDIO)

1. Abre: **http://localhost:9090**
2. Ve a **Status → Targets** (menú superior)
3. Verifica que `underdog-trading` esté **UP**
   - Si dice "DOWN", espera 10 segundos y recarga
   - El target es: `host.docker.internal:8000`
4. Vuelve a la página principal (click en el logo Prometheus)
5. En la **Query Box**, prueba estas queries:

```promql
# Ver P&L realizado por EA
underdog_ea_pnl_realized

# Ver posiciones abiertas
underdog_ea_positions_open

# Ver señales totales
sum by (ea_name) (underdog_ea_signals_total)

# Ver win rate
underdog_ea_win_rate

# Ver equity de cuenta
underdog_account_equity
```

6. Click **Execute** para ver valores
7. Click **Graph** para ver gráficos en tiempo real
8. Ajusta time range a "5m" (5 minutos)

---

### **Opción 3: Grafana Dashboards** (MÁS VISUAL)

#### Login:
1. Abre: **http://localhost:3000**
2. Usuario: `admin`
3. Contraseña: `admin123`
4. Si pide cambiar contraseña, click **Skip**

#### Configurar Datasource (SOLO PRIMERA VEZ):
1. Menú lateral → **Connections → Data sources**
2. Click **Add data source**
3. Selecciona **Prometheus**
4. URL: `http://underdog-prometheus:9090`
   - (O prueba: `http://prometheus:9090`)
5. Scroll abajo, click **Save & Test**
6. Debería decir: ✅ "Data source is working"

#### Abrir Dashboards:
1. Menú lateral → **Dashboards → Browse**
2. Verás 3 dashboards:
   - **UNDERDOG - EA Performance Overview**
   - **UNDERDOG - Signal Analysis**
   - **UNDERDOG - Trading Overview**
3. Click en cualquiera para abrir

#### Si no hay datos en los dashboards:
1. Arriba derecha, ajusta **Time Range** a "Last 5 minutes"
2. Click el ícono de **Refresh** (🔄) y selecciona "5s"
3. Si aún no hay datos:
   - Ve a cada panel
   - Click en el título → **Edit**
   - Verifica que la query esté correcta
   - Ejemplo: `underdog_ea_pnl_realized`

---

## 📊 MÉTRICAS DISPONIBLES (40+ métricas)

### Status & Health
```
underdog_ea_status{ea_name="..."} - 1=activo, 0=inactivo
underdog_account_balance - Balance de cuenta
underdog_account_equity - Equity actual
```

### Señales
```
underdog_ea_signals_total{ea_name="...", signal_type="BUY/SELL", symbol="..."} - Total señales
```

### Posiciones
```
underdog_ea_positions_open{ea_name="...", symbol="..."} - Posiciones abiertas
```

### P&L
```
underdog_ea_pnl_realized{ea_name="..."} - P&L realizado
underdog_ea_pnl_unrealized{ea_name="..."} - P&L no realizado
underdog_ea_pnl_daily{ea_name="..."} - P&L del día
```

### Performance
```
underdog_ea_win_rate{ea_name="..."} - Win rate (0-1)
underdog_ea_sharpe_ratio{ea_name="..."} - Sharpe ratio
underdog_ea_profit_factor{ea_name="..."} - Profit factor
```

### Ejecución
```
underdog_ea_execution_time_ms{ea_name="..."} - Tiempo de ejecución (ms)
```

---

## 🐛 SOLUCIÓN DE PROBLEMAS

### Problema 1: "Prometheus target DOWN"
```bash
# En PowerShell, verifica que el generador está corriendo:
netstat -ano | findstr ":8000"

# Debería mostrar: TCP 0.0.0.0:8000 LISTENING

# Si no aparece, relanza:
poetry run python scripts/generate_test_metrics.py
```

### Problema 2: "Grafana no muestra datos"
1. Ve a **Configuration → Data Sources**
2. Click en Prometheus
3. Cambia URL a: `http://underdog-prometheus:9090`
4. Click **Save & Test**
5. Si falla, prueba: `http://localhost:9090` (pero Grafana está en Docker, puede no funcionar)

### Problema 3: "Localhost:8000 no carga en navegador"
Esto es NORMAL si intentas acceder mientras curl está consultando. El servidor solo acepta 1 conexión a la vez. Espera 2 segundos y recarga.

### Problema 4: "Métricas no actualizan"
```bash
# Verifica que el generador está corriendo:
# En VS Code, busca la terminal con salida como:
# [18:14:06] EmaScalper: Position closed → P&L: $83.04

# Si no aparece, el generador se detuvo. Relanza:
poetry run python scripts/generate_test_metrics.py
```

---

## 📈 RESULTADOS ESPERADOS

### Generador de Métricas (después de 5 minutos):
- **Signals**: 200-400 señales generadas
- **Account**: $100,000 - $105,000
- **Daily P&L**: +$2,000 - +$5,000
- **Positions**: 0-12 posiciones fluctuando
- **Win Rate**: 60-75% por EA

### Prometheus Queries:
- `underdog_ea_pnl_realized`: Verás 7 valores (uno por EA), entre $-500 y +$1,500
- `underdog_ea_positions_open`: Verás 0-3 posiciones por EA
- `underdog_ea_signals_total`: Verás contadores incrementándose

### Grafana Dashboards:
- **EA Performance**: Gráficos de equity ascendente, P&L por EA, win rates
- **Signal Analysis**: Distribución BUY/SELL, señales por EA, heatmaps
- **Trading Overview**: Balance, posiciones, drawdown, trades recientes

---

## 🎯 CONCLUSIÓN

**FUNCIONANDO AHORA**:
- ✅ Generador de métricas (7 EAs simulados)
- ✅ Prometheus (recolectando métricas)
- ✅ Grafana (dashboards visuales)
- ✅ TimescaleDB (base de datos)

**NO FUNCIONANDO**:
- ❌ Streamlit (se detiene inmediatamente)

**RECOMENDACIÓN**:
Por ahora, **usa Prometheus y Grafana** para ver los datos ficticios. Son más estables y profesionales que Streamlit para monitorización en tiempo real.

**PARA STREAMLIT**:
Si necesitas absolutamente Streamlit, podemos debuggear el problema, pero Grafana es mejor para demostración de sistemas de trading profesionales.

---

## 📞 SIGUIENTE PASO

¿Qué quieres hacer?

**A)** Abrir Prometheus y hacer queries → http://localhost:9090
**B)** Configurar Grafana y ver dashboards → http://localhost:3000
**C)** Ver métricas raw en tiempo real → http://localhost:8000/metrics
**D)** Debuggear Streamlit para hacerlo funcionar
**E)** Exportar un video/captura de Grafana mostrando los datos

**Dime la letra y te guío paso a paso.** 🚀
