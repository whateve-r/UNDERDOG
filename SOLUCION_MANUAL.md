# 🚀 SOLUCIÓN PASO A PASO - Grafana + Prometheus + Streamlit

**PROBLEMA ACTUAL**: Los servicios se detienen automáticamente cuando VS Code ejecuta comandos.

**SOLUCIÓN**: Ejecutar manualmente en terminales separadas.

---

## 📋 PASOS A SEGUIR

### **PASO 1: Abrir PowerShell dedicado para Métricas** ⭐ CRÍTICO

1. Presiona `Windows + R`
2. Escribe: `powershell`
3. Enter
4. En la nueva ventana de PowerShell, ejecuta:

```powershell
cd C:\Users\manud\OneDrive\Escritorio\tfg\UNDERDOG
poetry run python scripts/generate_test_metrics.py
```

5. **DÉJALA CORRIENDO**. Verás algo como:
```
================================================================================
🧪 UNDERDOG - Test Metrics Generator
================================================================================
🚀 Starting Prometheus metrics server on port 8000...
✅ Metrics server running: http://localhost:8000/metrics

📊 Initializing 7 EAs...
  ✓ SuperTrendRSI (EURUSD M15) - Confidence: 1.0
  ...

▶️  Simulation started - Generating metrics every 1 second
[18:24:04] EmaScalper: BUY signal → Position opened
```

6. **NO CIERRES ESTA VENTANA**. Minimízala.

---

### **PASO 2: Verificar que Prometheus está capturando datos**

1. Abre tu navegador (Chrome/Edge)
2. Ve a: **http://localhost:9090**
3. Click en **"Status"** → **"Targets"** (menú superior)
4. Busca `underdog-trading`
5. Debe decir: **UP** (en verde)
   - Si dice "DOWN", espera 10 segundos y recarga (F5)
6. Vuelve a la página principal (click en logo Prometheus)
7. En la caja de texto grande, escribe:
```
underdog_ea_pnl_realized
```
8. Click **"Execute"**
9. Deberías ver 7 valores (uno por cada EA)
10. Click **"Graph"** para ver gráfico en tiempo real

---

### **PASO 3: Configurar Grafana** (SOLO UNA VEZ)

1. Abre: **http://localhost:3000**
2. Login:
   - Usuario: `admin`
   - Contraseña: `admin123`
3. Si pide cambiar contraseña, click **"Skip"**

#### Configurar Datasource:
4. Menú lateral izquierdo → **"Connections"** → **"Data sources"**
5. Click **"Add data source"**
6. Selecciona **"Prometheus"**
7. En **"URL"**, escribe: `http://underdog-prometheus:9090`
8. Scroll hasta abajo
9. Click **"Save & Test"**
10. Debe decir: ✅ **"Data source is working"**

---

### **PASO 4: Ver Dashboards en Grafana**

1. Menú lateral izquierdo → **"Dashboards"** → **"Browse"**
2. Verás 3 dashboards:
   - UNDERDOG - EA Performance Overview
   - UNDERDOG - Signal Analysis
   - UNDERDOG - Trading Overview
3. Click en cualquiera para abrir
4. **Arriba a la derecha**:
   - Click en el reloj 🕐 → Selecciona **"Last 5 minutes"**
   - Click en 🔄 → Selecciona **"5s"** (auto-refresh cada 5 segundos)
5. Deberías ver:
   - Gráficos de equity subiendo/bajando
   - P&L por EA
   - Señales generadas
   - Posiciones abiertas
   - Win rates

**Si no ves datos**:
- Verifica que el PowerShell del PASO 1 sigue corriendo
- Ve a Prometheus (http://localhost:9090) y verifica que `underdog-trading` está **UP**
- En Grafana, click en el título de un panel → **"Edit"** → Verifica la query

---

### **PASO 5: Abrir Streamlit** (OPCIONAL)

1. Abre **OTRA** ventana de PowerShell (Windows + R → powershell)
2. Ejecuta:
```powershell
cd C:\Users\manud\OneDrive\Escritorio\tfg\UNDERDOG
poetry run streamlit run underdog/ui/streamlit_backtest.py --server.port 8501 --server.headless true
```
3. Espera a que aparezca:
```
You can now view your Streamlit app in your browser.
Local URL: http://localhost:8501
```
4. Abre: **http://localhost:8501**
5. Selecciona:
   - Strategy: **SuperTrendRSI**
   - Symbol: **EURUSD**
   - Timeframe: **M15**
6. Click **"Run Backtest"**
7. Espera 10-20 segundos
8. Verás resultados con ~200 trades

---

## ✅ VERIFICACIÓN FINAL

Deberías tener:

1. ✅ PowerShell corriendo con métricas (no cerrar)
2. ✅ Prometheus: http://localhost:9090 - Target "UP", queries funcionando
3. ✅ Grafana: http://localhost:3000 - Dashboards mostrando datos en tiempo real
4. ✅ Streamlit (opcional): http://localhost:8501 - Backtesting funcional

---

## 🐛 SOLUCIÓN DE PROBLEMAS

### Problema: "Prometheus target DOWN"
**Solución**:
1. Verifica que el PowerShell del PASO 1 esté corriendo
2. En ese PowerShell, busca: `✅ Metrics server running: http://localhost:8000/metrics`
3. Si no aparece, cierra y vuelve a ejecutar el PASO 1

### Problema: "Grafana no muestra datos"
**Solución**:
1. Ve a http://localhost:9090
2. Ejecuta query: `underdog_ea_status`
3. Si NO ves resultados, el problema está en Prometheus (ver problema anterior)
4. Si VES resultados, el problema está en Grafana:
   - Ve a **Connections → Data sources → Prometheus**
   - Click **"Test"**
   - Si falla, cambia URL a: `http://prometheus:9090` o `http://192.168.1.36:9090`

### Problema: "Streamlit no inicia"
**Solución**:
1. Cierra la ventana de PowerShell de Streamlit
2. Ve a VS Code terminal
3. Ejecuta:
```powershell
cd C:\Users\manud\OneDrive\Escritorio\tfg\UNDERDOG
poetry run streamlit run underdog/ui/streamlit_backtest.py
```
4. NO uses `--server.headless true`

### Problema: "Puerto 8000 ya en uso"
**Solución**:
```powershell
# Encuentra el proceso
netstat -ano | findstr ":8000"

# Anota el PID (última columna)
Stop-Process -Id <PID> -Force

# Relanza el PASO 1
```

---

## 📊 LO QUE DEBERÍAS VER

### En Grafana - Dashboard "EA Performance":
```
┌─────────────────────────────────────────┐
│ Total Equity: $100,500   Active EAs: 7 │
│ Total Signals: 150       Win Rate: 65% │
├─────────────────────────────────────────┤
│ 📈 Equity Over Time                     │
│ (gráfico ascendente de $100k a $101k)  │
├─────────────────────────────────────────┤
│ P&L by EA:                              │
│ SuperTrendRSI    ████████ $350.00      │
│ ParabolicEMA     ███████ $280.00       │
│ KeltnerBreakout  ██████ $220.00        │
│ ...                                     │
└─────────────────────────────────────────┘
```

### En Prometheus - Query `underdog_ea_pnl_realized`:
```
underdog_ea_pnl_realized{ea_name="ATRBreakout"} → 123.45
underdog_ea_pnl_realized{ea_name="BollingerCCI"} → 234.56
underdog_ea_pnl_realized{ea_name="EmaScalper"} → 345.67
...
```

### En Streamlit - Backtest Results:
```
╔════════════════════════════════════╗
║ 📊 Performance Metrics             ║
╠════════════════════════════════════╣
║ Total Return      │ +23.45%        ║
║ Total Trades      │ 234            ║
║ Win Rate          │ 65.81%         ║
║ Sharpe Ratio      │ 1.85           ║
║ Max Drawdown      │ -12.34%        ║
╚════════════════════════════════════╝
```

---

## 🎯 ORDEN RECOMENDADO DE PRUEBA

1. **PRIMERO**: Ejecutar PASO 1 (generador de métricas)
2. **SEGUNDO**: Verificar PASO 2 (Prometheus funcionando)
3. **TERCERO**: Configurar PASO 3 (Grafana datasource)
4. **CUARTO**: Ver PASO 4 (Dashboards de Grafana)
5. **QUINTO** (Opcional): PASO 5 (Streamlit backtesting)

---

## 📞 SI NADA FUNCIONA

Ejecuta esto y mándame el output:

```powershell
# En PowerShell
docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
netstat -ano | findstr ":8000 :9090 :3000 :8501" | findstr "LISTENING"
curl http://localhost:9090/api/v1/targets 2>$null | ConvertFrom-Json | Select-Object -ExpandProperty data | Select-Object -ExpandProperty activeTargets | Select-Object labels,health
```

---

**¡IMPORTANTE!**: El generador de métricas (PASO 1) DEBE estar corriendo TODO EL TIEMPO. Es el corazón del sistema. Sin él, Prometheus no tendrá datos y Grafana estará vacío.

🚀 **¡Adelante! Ejecuta el PASO 1 AHORA y déjalo corriendo.** Luego verifica Prometheus.
