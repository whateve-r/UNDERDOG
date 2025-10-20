# ✅ CHECKLIST INTERACTIVA - PONER EN MARCHA EL SISTEMA

**Usa este documento para verificar cada paso**  
Marca ✅ cuando completes cada tarea

---

## 📋 PRE-REQUISITOS

- [x] Docker Desktop instalado y corriendo
- [x] Firewall rules creadas (4 reglas UNDERDOG)
- [x] Docker containers UP (Prometheus, Grafana, TimescaleDB)
- [ ] **Metrics generator NO corriendo aún** (esto es lo que vamos a hacer)

---

## 🚀 PASO 1: INICIAR METRICS GENERATOR (3 min)

### 1.1. Abrir PowerShell Nativo
- [ ] Presiona `Windows + R`
- [ ] Escribe: `powershell`
- [ ] Presiona `Enter`
- [ ] Se abre ventana nueva de PowerShell (NO en VS Code)

### 1.2. Navegar al Proyecto
```powershell
cd C:\Users\manud\OneDrive\Escritorio\tfg\UNDERDOG
```
- [ ] Comando ejecutado sin errores

### 1.3. Iniciar Generador
```powershell
poetry run python scripts\generate_test_metrics.py
```

### 1.4. Verificar Salida
Debes ver:
```
================================================================================
🧪 UNDERDOG - Test Metrics Generator
================================================================================
🚀 Starting Prometheus metrics server on port 8000...
✅ Metrics server running: http://localhost:8000/metrics

📊 Initializing 7 EAs...
  ✓ SuperTrendRSI (EURUSD M15) - Confidence: 1.0
  ✓ ParabolicEMA (GBPUSD M15) - Confidence: 0.95
  ✓ KeltnerBreakout (EURUSD M5) - Confidence: 0.90
  ✓ EmaScalper (EURUSD M5) - Confidence: 0.85
  ✓ BollingerCCI (GBPUSD M15) - Confidence: 0.88
  ✓ ATRBreakout (USDJPY M15) - Confidence: 0.87
  ✓ PairArbitrage (EURUSD/GBPUSD H1) - Confidence: 0.92

▶️  Simulation started - Generating metrics every 1 second
```

Checklist:
- [ ] ✅ Mensaje "Metrics server running" apareció
- [ ] ✅ 7 EAs inicializados con checkmarks ✓
- [ ] ✅ "Simulation started" apareció
- [ ] ✅ Ves líneas de signals aparecer cada 1-2 segundos

**⚠️ IMPORTANTE**: 
- [ ] **NO CIERRES ESTA VENTANA**
- [ ] **MINIMÍZALA** (no cerrar)

---

## 🔍 PASO 2: VERIFICAR MÉTRICAS (2 min)

### 2.1. Abrir SEGUNDA Ventana de PowerShell
- [ ] `Windows + R` → `powershell` → `Enter`

### 2.2. Test Localhost
```powershell
curl http://localhost:8000/metrics | Select-Object -First 10
```

Debes ver:
```
# HELP underdog_ea_status EA status (1 = active, 0 = inactive)
# TYPE underdog_ea_status gauge
underdog_ea_status{ea_name="SuperTrendRSI"} 1.0
underdog_ea_status{ea_name="ParabolicEMA"} 1.0
...
```

- [ ] ✅ Métricas aparecen (no error "connection refused")

### 2.3. Test IP Externa (Docker)
```powershell
curl http://192.168.1.36:8000/metrics | Select-Object -First 10
```

- [ ] ✅ Métricas aparecen desde IP externa

### 2.4. Verificar Puerto Listening
```powershell
netstat -ano | findstr ":8000" | findstr "LISTENING"
```

Debes ver:
```
TCP    0.0.0.0:8000    0.0.0.0:0    LISTENING    12345
```

- [ ] ✅ Puerto 8000 está LISTENING en 0.0.0.0

---

## 📊 PASO 3: VERIFICAR PROMETHEUS (2 min)

### 3.1. Abrir Prometheus Targets
- [ ] Abre navegador
- [ ] Ve a: `http://localhost:9090/targets`

### 3.2. Verificar Target Estado
Busca: `underdog-trading (192.168.1.36:8000)`

Debe mostrar:
- **State**: **UP** (verde)
- **Last Scrape**: < 5 segundos ago
- **Scrape Duration**: ~0.05s

Checklist:
- [ ] ✅ Target aparece en la lista
- [ ] ✅ Estado es "UP" (texto verde)
- [ ] ✅ "Last Scrape" es reciente (< 10s)

**Si está DOWN**:
```powershell
# Espera 10 segundos y recarga la página
# Si sigue DOWN, revisa que el generador sigue corriendo (ventana PowerShell #1)
```

### 3.3. Ejecutar Query
- [ ] Click en logo "Prometheus" (arriba izquierda) para volver a home
- [ ] En la caja de texto grande, escribe: `underdog_ea_status`
- [ ] Click botón azul **"Execute"**

Debes ver:
```
underdog_ea_status{ea_name="ATRBreakout", instance="underdog-local", job="underdog-trading"} → 1
underdog_ea_status{ea_name="BollingerCCI", instance="underdog-local", job="underdog-trading"} → 1
underdog_ea_status{ea_name="EmaScalper", instance="underdog-local", job="underdog-trading"} → 1
underdog_ea_status{ea_name="KeltnerBreakout", instance="underdog-local", job="underdog-trading"} → 1
underdog_ea_status{ea_name="PairArbitrage", instance="underdog-local", job="underdog-trading"} → 1
underdog_ea_status{ea_name="ParabolicEMA", instance="underdog-local", job="underdog-trading"} → 1
underdog_ea_status{ea_name="SuperTrendRSI", instance="underdog-local", job="underdog-trading"} → 1
```

Checklist:
- [ ] ✅ 7 resultados aparecen
- [ ] ✅ Todos los valores son "1"
- [ ] ✅ Nombres de EAs correctos

---

## 🎨 PASO 4: ABRIR GRAFANA DASHBOARDS (3 min)

### 4.1. Acceder a Grafana
- [ ] Abre navegador (nueva pestaña)
- [ ] Ve a: `http://localhost:3000`
- [ ] Login: `admin`
- [ ] Password: `admin123`
- [ ] Si pide cambiar contraseña, click **"Skip"**

### 4.2. Navegar a Dashboard
- [ ] Menu lateral izquierdo → **"Dashboards"**
- [ ] Click **"Browse"**
- [ ] Click **"UNDERDOG - EA Performance Overview"**

### 4.3. Configurar Refresh y Time Range
**Arriba a la derecha**:
- [ ] Click en reloj 🕐 → Selecciona **"Last 5 minutes"**
- [ ] Click en 🔄 → Selecciona **"5s"** (auto-refresh cada 5 segundos)

### 4.4. Verificar Datos Aparecer
Debes ver en los paneles:

**Panel 1: Total Equity**
- [ ] ✅ Valor: ~$100,000 (puede variar ligeramente)

**Panel 2: Active EAs**
- [ ] ✅ Valor: 7

**Panel 3: Total Signals**
- [ ] ✅ Valor incrementando cada 5 segundos (ej: 10 → 15 → 20)

**Panel 4: Win Rate**
- [ ] ✅ Valor entre 50-75%

**Panel 5: Equity Curve (gráfico línea)**
- [ ] ✅ Línea aparece (ascendente o zigzag)
- [ ] ✅ Línea se actualiza cada 5 segundos

**Panel 6: P&L by EA (gráfico barras)**
- [ ] ✅ 7 barras aparecen (una por EA)
- [ ] ✅ Algunas barras verdes (positivas), otras rojas (negativas)

**Panel 7: Signals by Type (pie chart)**
- [ ] ✅ Chart aparece con segmentos BUY/SELL

**Si NO ves datos**:
- [ ] Verifica que metrics generator sigue corriendo (ventana PowerShell #1)
- [ ] Verifica Prometheus target UP (paso 3.2)
- [ ] Espera 10-15 segundos más y recarga Grafana (F5)

---

## 🎯 PASO 5: EXPLORAR OTROS DASHBOARDS (OPCIONAL - 5 min)

### 5.1. Dashboard: Signal Analysis
- [ ] Dashboards → Browse → **"UNDERDOG - Signal Analysis"**
- [ ] Configurar refresh 5s y Last 5 minutes
- [ ] Verificar datos aparecen en:
  - [ ] Signal Distribution (bar chart)
  - [ ] Signal Rate Over Time (line chart)
  - [ ] Confidence Heatmap

### 5.2. Dashboard: Trading Overview
- [ ] Dashboards → Browse → **"UNDERDOG - Trading Overview"**
- [ ] Configurar refresh 5s y Last 5 minutes
- [ ] Verificar datos aparecen en:
  - [ ] Account Balance
  - [ ] Open Positions
  - [ ] Daily P&L
  - [ ] Drawdown %

---

## 🌐 PASO 6: INICIAR STREAMLIT (OPCIONAL - 5 min)

### 6.1. Abrir TERCERA Ventana PowerShell
- [ ] `Windows + R` → `powershell` → `Enter`

### 6.2. Navegar y Ejecutar
```powershell
cd C:\Users\manud\OneDrive\Escritorio\tfg\UNDERDOG
poetry run streamlit run underdog\ui\streamlit_backtest.py --server.port 8501
```

### 6.3. Esperar Inicio
Debes ver:
```
You can now view your Streamlit app in your browser.

  Local URL: http://localhost:8501
  Network URL: http://192.168.1.36:8501
```

- [ ] ✅ Mensaje "You can now view" apareció
- [ ] ✅ URL local mostrada

### 6.4. Abrir en Navegador
- [ ] Abre: `http://localhost:8501`
- [ ] Dashboard de Streamlit carga

### 6.5. Ejecutar Backtest de Prueba
- [ ] Sidebar → Select EA: **"SuperTrendRSI"**
- [ ] Symbol: **EURUSD**
- [ ] Timeframe: **M15**
- [ ] Date range: último año
- [ ] Click **"🚀 Run Backtest"** (botón azul abajo)

Espera 10-20 segundos:
- [ ] ✅ "Running backtest..." mensaje aparece
- [ ] ✅ Backtest completa sin errores
- [ ] ✅ Equity curve aparece
- [ ] ✅ Performance metrics mostradas
- [ ] ✅ Trade history table aparece

---

## ✅ VERIFICACIÓN FINAL

### Sistema Completo Operativo
- [ ] ✅ Metrics generator corriendo (ventana PowerShell #1 activa)
- [ ] ✅ Puerto 8000 LISTENING
- [ ] ✅ Prometheus target UP
- [ ] ✅ Prometheus queries devuelven 7 EAs
- [ ] ✅ Grafana dashboards muestran datos actualizándose
- [ ] ✅ Streamlit dashboard accesible (opcional)

### Ventanas PowerShell Abiertas
- [ ] Ventana #1: Metrics generator (NO CERRAR - minimizada)
- [ ] Ventana #2: Tests y comandos (puede cerrarse)
- [ ] Ventana #3: Streamlit (NO CERRAR si quieres usar - minimizada)

---

## 🎉 ¡ÉXITO!

Si todos los checkboxes están marcados:

✅ **Sistema 100% operativo**  
✅ **Demo lista para mostrar**  
✅ **Todas las integraciones funcionando**

---

## 📸 CAPTURAS RECOMENDADAS (Para documentación)

Toma screenshots de:
- [ ] Prometheus targets (todos UP)
- [ ] Prometheus query `underdog_ea_status` (7 resultados)
- [ ] Grafana "EA Performance Overview" (con datos)
- [ ] Streamlit backtest results
- [ ] Terminal con metrics generator corriendo

---

## 🔄 PARA DETENER EL SISTEMA

Cuando termines:
1. [ ] Ventana metrics generator: `Ctrl + C`
2. [ ] Ventana Streamlit: `Ctrl + C`
3. [ ] Docker containers (opcional): `docker-compose down`

---

## 📝 NOTAS IMPORTANTES

⚠️ **NO CIERRES** las ventanas de PowerShell con los servicios corriendo  
⚠️ **NO EJECUTES OTROS COMANDOS** en la ventana del metrics generator  
⚠️ Si algo falla, volver a **FIREWALL_SETUP_COMPLETE.md** → Sección Troubleshooting

---

**Última actualización**: 2025-10-20  
**Tiempo total estimado**: 15-20 minutos  
**Dificultad**: ⭐⭐ (Baja-Media)

---

*Fin de la checklist - ¡Suerte!* 🚀
