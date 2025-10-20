# ✅ FIREWALL CONFIGURADO - ESTADO ACTUAL

**Fecha**: 2025-10-20  
**Última actualización**: Reglas de firewall creadas exitosamente

---

## 🔥 REGLAS DE FIREWALL CREADAS

| Regla                             | Puerto | Dirección | Acción | Estado |
|-----------------------------------|--------|-----------|--------|--------|
| UNDERDOG Metrics Server (TCP-In)  | 8000   | Inbound   | Allow  | ✅ Activa |
| UNDERDOG Streamlit UI (TCP-In)    | 8501   | Inbound   | Allow  | ✅ Activa |
| UNDERDOG Prometheus (TCP-In)      | 9090   | Inbound   | Allow  | ✅ Activa |
| UNDERDOG Grafana (TCP-In)         | 3000   | Inbound   | Allow  | ✅ Activa |

---

## 📊 ESTADO DE SERVICIOS

| Puerto | Servicio          | Estado Firewall | Estado Proceso | Acción Requerida |
|--------|-------------------|-----------------|----------------|------------------|
| 8000   | Metrics Generator | ✅ Permitido    | ⏳ NO ACTIVO   | Iniciar manualmente |
| 8501   | Streamlit         | ✅ Permitido    | ⏳ NO ACTIVO   | Iniciar manualmente |
| 9090   | Prometheus        | ✅ Permitido    | ✅ LISTENING   | Ninguna |
| 3000   | Grafana           | ✅ Permitido    | ✅ LISTENING   | Ninguna |

---

## 🚀 PRÓXIMOS PASOS

### 1. Iniciar Metrics Generator (CRÍTICO)

**Abrir PowerShell NUEVO (fuera de VS Code)**:
```powershell
# Presiona Windows + R → escribe "powershell" → Enter
cd C:\Users\manud\OneDrive\Escritorio\tfg\UNDERDOG
poetry run python scripts\generate_test_metrics.py
```

**Salida esperada**:
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

⚠️ **IMPORTANTE**: NO CIERRES ESTA VENTANA. Déjala corriendo en segundo plano.

---

### 2. Verificar Métricas Accesibles

**En otra ventana de PowerShell**:
```powershell
# Test localhost
curl http://localhost:8000/metrics | Select-Object -First 10

# Test IP externa (desde Docker)
curl http://192.168.1.36:8000/metrics | Select-Object -First 10
```

**Salida esperada** (primeras líneas):
```
# HELP underdog_ea_status EA status (1 = active, 0 = inactive)
# TYPE underdog_ea_status gauge
underdog_ea_status{ea_name="SuperTrendRSI"} 1.0
underdog_ea_status{ea_name="ParabolicEMA"} 1.0
underdog_ea_status{ea_name="KeltnerBreakout"} 1.0
...
```

---

### 3. Verificar Prometheus Scraping

1. Abrir navegador: **http://localhost:9090/targets**
2. Buscar target: `underdog-trading (192.168.1.36:8000)`
3. Estado esperado: **UP** (verde) con "Last Scrape" < 5s ago

**Si target está DOWN**:
```powershell
# Verificar puerto 8000 escuchando
netstat -ano | findstr ":8000" | findstr "LISTENING"

# Debe mostrar algo como:
# TCP    0.0.0.0:8000    0.0.0.0:0    LISTENING    12345
```

---

### 4. Verificar Prometheus Query

1. En **http://localhost:9090**
2. Ejecutar query: `underdog_ea_status`
3. Click **"Execute"**
4. Resultado esperado: 7 series (una por EA)

**Ejemplo de resultado**:
```
underdog_ea_status{ea_name="SuperTrendRSI", instance="underdog-local", job="underdog-trading"} → 1
underdog_ea_status{ea_name="ParabolicEMA", instance="underdog-local", job="underdog-trading"} → 1
underdog_ea_status{ea_name="KeltnerBreakout", instance="underdog-local", job="underdog-trading"} → 1
...
```

---

### 5. Abrir Grafana Dashboards

1. Abrir navegador: **http://localhost:3000**
2. Login: `admin` / `admin123`
3. Navegar: **Dashboards → Browse**
4. Seleccionar: **"UNDERDOG - EA Performance Overview"**
5. Configurar refresh: **Top-right → 🔄 → 5s**
6. Configurar time range: **Top-right → 🕐 → Last 5 minutes**

**Paneles esperados con datos**:
- Total Equity (valor ~$100,000)
- Active EAs (valor 7)
- Total Signals (incrementando cada 1-2s)
- Win Rate (valores entre 60-75%)
- Equity curve (línea ascendente)
- P&L by EA (barras con valores positivos/negativos)

---

### 6. Iniciar Streamlit (OPCIONAL)

**En PowerShell NUEVO #3**:
```powershell
cd C:\Users\manud\OneDrive\Escritorio\tfg\UNDERDOG
poetry run streamlit run underdog\ui\streamlit_backtest.py --server.port 8501
```

**Abrir**: http://localhost:8501

---

## 🔍 TROUBLESHOOTING

### Problema: Firewall sigue bloqueando
**Solución**: Re-crear reglas con perfil Public también
```powershell
# Ejecutar como Administrador
Get-NetFirewallRule | Where-Object {$_.DisplayName -like "*UNDERDOG*"} | 
    Set-NetFirewallRule -Profile Private,Domain,Public
```

### Problema: Puerto 8000 no accesible desde 192.168.1.36
**Verificar**: `prometheus_metrics.py` línea 322
```python
start_http_server(port, addr='0.0.0.0')  # ← Debe tener addr='0.0.0.0'
```

### Problema: Prometheus target DOWN después de 1 minuto
**Causa**: Timeout muy corto en prometheus.yml
```yaml
# docker/prometheus.yml - línea 38
scrape_timeout: 2s  # Cambiar de 500ms a 2s
```

Después de cambiar:
```powershell
docker restart underdog-prometheus
```

### Problema: Metrics generator se detiene al ejecutar comandos
**Causa**: VS Code terminal multiplexer
**Solución**: Ejecutar en PowerShell NATIVO (fuera de VS Code)

---

## ✅ CHECKLIST FINAL

- [x] Firewall rules creadas (puertos 8000, 8501, 9090, 3000)
- [x] Docker containers running (Prometheus, Grafana)
- [ ] Metrics generator corriendo (ventana PowerShell dedicada)
- [ ] Puerto 8000 LISTENING
- [ ] Prometheus target UP
- [ ] Grafana dashboards mostrando datos
- [ ] Streamlit accesible (opcional)

---

## 📝 RESUMEN EJECUTIVO

✅ **Problema de firewall RESUELTO**  
✅ **Reglas creadas correctamente**  
✅ **Servicios Docker operativos**  
⏳ **Pendiente**: Iniciar metrics generator manualmente  

**Tiempo estimado para completar**: 5 minutos  
**Dificultad**: Baja (solo seguir pasos 1-5)

---

**Siguiente acción**: Abrir PowerShell nativo y ejecutar paso 1 (metrics generator)
