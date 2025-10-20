# 🔥 FIREWALL AUDIT - UNDERDOG Trading System

**Fecha**: 2025-10-20  
**Objetivo**: Diagnosticar problemas de conectividad entre Prometheus, Grafana, Streamlit y el generador de métricas

---

## 📊 ESTADO ACTUAL

### Perfiles de Firewall
```
Name    Enabled   DefaultInboundAction   DefaultOutboundAction
----    -------   --------------------   ---------------------
Domain  True      NotConfigured          NotConfigured
Private True      NotConfigured          NotConfigured
Public  True      NotConfigured          NotConfigured
```

✅ **Firewall activo** en todos los perfiles  
⚠️ **Acciones no configuradas** - usando defaults del sistema

---

## 🔌 PUERTOS EN USO

### Servicios Docker (OK)
| Puerto | Servicio   | Estado         | PID  | Bind Address |
|--------|-----------|----------------|------|--------------|
| 3000   | Grafana   | ✅ LISTENING   | 3740 | 0.0.0.0      |
| 9090   | Prometheus| ✅ LISTENING   | 3740 | 0.0.0.0      |

### Servicios Python (NO ACTIVOS)
| Puerto | Servicio          | Estado         | Razón                          |
|--------|-------------------|----------------|--------------------------------|
| 8000   | Metrics Generator | ❌ NO LISTENING| Proceso no iniciado            |
| 8501   | Streamlit         | ❌ NO LISTENING| Proceso no iniciado            |

---

## 🚨 PROBLEMAS IDENTIFICADOS

### 1. **Puerto 8000 - Metrics Generator**
- **Test localhost**: ❌ FAILED
- **Test 192.168.1.36**: ❌ FAILED  
- **Causa**: Proceso no corriendo (debe ejecutarse manualmente en PowerShell separado)

### 2. **Puerto 8501 - Streamlit**  
- **Estado**: No verificado (asumido igual que 8000)
- **Causa**: Proceso no corriendo

### 3. **Reglas de Firewall**
- **Reglas UNDERDOG**: ❌ NO EXISTEN
- **Reglas puertos específicos**: ❌ NO EXISTEN
- **Impacto**: Cuando los servicios se inicien, el firewall puede bloquear conexiones entrantes

---

## 🔧 SOLUCIÓN PROPUESTA

### Fase 1: Crear Reglas de Firewall (PREVENTIVO)

Crear reglas **ANTES** de iniciar servicios para evitar bloqueos:

```powershell
# 1. Metrics Generator (Puerto 8000) - Entrada
New-NetFirewallRule -DisplayName "UNDERDOG Metrics Server (TCP-In)" `
    -Direction Inbound `
    -Protocol TCP `
    -LocalPort 8000 `
    -Action Allow `
    -Profile Private,Domain `
    -Description "Allow inbound connections to UNDERDOG metrics server on port 8000"

# 2. Streamlit (Puerto 8501) - Entrada
New-NetFirewallRule -DisplayName "UNDERDOG Streamlit UI (TCP-In)" `
    -Direction Inbound `
    -Protocol TCP `
    -LocalPort 8501 `
    -Action Allow `
    -Profile Private,Domain `
    -Description "Allow inbound connections to UNDERDOG Streamlit dashboard on port 8501"

# 3. Prometheus (Puerto 9090) - Entrada (redundante pero explícito)
New-NetFirewallRule -DisplayName "UNDERDOG Prometheus (TCP-In)" `
    -Direction Inbound `
    -Protocol TCP `
    -LocalPort 9090 `
    -Action Allow `
    -Profile Private,Domain `
    -Description "Allow inbound connections to Prometheus monitoring on port 9090"

# 4. Grafana (Puerto 3000) - Entrada (redundante pero explícito)
New-NetFirewallRule -DisplayName "UNDERDOG Grafana (TCP-In)" `
    -Direction Inbound `
    -Protocol TCP `
    -LocalPort 3000 `
    -Action Allow `
    -Profile Private,Domain `
    -Description "Allow inbound connections to Grafana dashboards on port 3000"

# 5. Python.exe - Salida (para scrapers y APIs)
New-NetFirewallRule -DisplayName "UNDERDOG Python Outbound" `
    -Direction Outbound `
    -Program "C:\Users\manud\AppData\Local\Programs\Python\Python*\python.exe" `
    -Action Allow `
    -Profile Private,Domain `
    -Description "Allow Python applications to make outbound connections"
```

### Fase 2: Verificar Reglas Creadas

```powershell
Get-NetFirewallRule | Where-Object {$_.DisplayName -like "*UNDERDOG*"} | 
    Select-Object DisplayName, Enabled, Direction, Action | 
    Format-Table -AutoSize
```

### Fase 3: Iniciar Servicios en Orden

#### 3.1. Metrics Generator (PowerShell dedicado)
```powershell
cd C:\Users\manud\OneDrive\Escritorio\tfg\UNDERDOG
poetry run python scripts/generate_test_metrics.py
```

**Salida esperada**:
```
🚀 Starting Prometheus metrics server on port 8000...
✅ Metrics server running: http://localhost:8000/metrics
📊 Initializing 7 EAs...
```

#### 3.2. Verificar Métricas Accesibles
```powershell
# Desde localhost
curl http://localhost:8000/metrics | Select-Object -First 10

# Desde IP externa
curl http://192.168.1.36:8000/metrics | Select-Object -First 10
```

#### 3.3. Streamlit (PowerShell dedicado #2)
```powershell
cd C:\Users\manud\OneDrive\Escritorio\tfg\UNDERDOG
poetry run streamlit run underdog/ui/streamlit_backtest.py --server.port 8501
```

### Fase 4: Verificar Prometheus Target

1. Abrir: http://localhost:9090/targets
2. Buscar: `underdog-trading (192.168.1.36:8000)`
3. Estado esperado: **UP** (verde)

---

## 🎯 CHECKLIST DE VERIFICACIÓN

### Pre-Inicio
- [ ] Firewall rules creadas (ejecutar comandos Fase 1)
- [ ] Docker containers running (Prometheus, Grafana, TimescaleDB)
- [ ] Verificar `docker ps` muestra 3 contenedores UP

### Durante Inicio
- [ ] Metrics generator corriendo sin interrupciones (ventana PowerShell activa)
- [ ] Puerto 8000 LISTENING (`netstat -ano | findstr ":8000"`)
- [ ] Curl localhost:8000/metrics devuelve métricas (no error 404/refused)
- [ ] Curl 192.168.1.36:8000/metrics también accesible

### Post-Inicio
- [ ] Prometheus target UP (http://localhost:9090/targets)
- [ ] Prometheus query `underdog_ea_status` devuelve 7 resultados
- [ ] Grafana dashboards muestran datos actualizándose
- [ ] Streamlit accesible en http://localhost:8501

---

## 🔍 DIAGNÓSTICO DE ERRORES COMUNES

### Error: "Connection refused" en 192.168.1.36:8000
**Causa**: Servidor no binding a 0.0.0.0  
**Solución**: Verificar que `prometheus_metrics.py` tiene:
```python
start_http_server(port, addr='0.0.0.0')  # ← Debe tener addr='0.0.0.0'
```

### Error: "Context deadline exceeded" en Prometheus
**Causa**: Scrape timeout (500ms puede ser muy corto)  
**Solución**: Aumentar `scrape_timeout` en `docker/prometheus.yml`:
```yaml
scrape_timeout: 2s  # ← Cambiar de 500ms a 2s
```

### Error: "Target DOWN" después de 5 minutos
**Causa**: Proceso Python se detuvo (VS Code terminal interrupt)  
**Solución**: Ejecutar en PowerShell nativo (fuera de VS Code)

### Error: "Firewall blocking" en logs de Docker
**Causa**: Reglas de firewall no creadas  
**Solución**: Ejecutar comandos de Fase 1 con permisos de Administrador

---

## 📝 REGISTRO DE ACCIONES

### 2025-10-20 - Auditoría Inicial
- ✅ Verificado firewall activo (3 perfiles)
- ✅ Identificados servicios Docker corriendo (Grafana, Prometheus)
- ❌ Confirmado ausencia de reglas UNDERDOG
- ❌ Confirmado puertos 8000 y 8501 no listening
- 📋 Creado plan de acción de 4 fases

### Próximas Acciones
1. Ejecutar comandos Fase 1 (requiere Admin)
2. Iniciar metrics generator (PowerShell dedicado)
3. Verificar conectividad puerto 8000
4. Validar Prometheus scraping

---

## 🚀 COMANDO RÁPIDO (TODO EN UNO)

Para usuarios avanzados, ejecutar en PowerShell como **Administrador**:

```powershell
# Crear todas las reglas
@(8000, 8501, 9090, 3000) | ForEach-Object {
    $port = $_
    $name = switch($port) {
        8000 { "Metrics Server" }
        8501 { "Streamlit UI" }
        9090 { "Prometheus" }
        3000 { "Grafana" }
    }
    
    New-NetFirewallRule -DisplayName "UNDERDOG $name (TCP-In)" `
        -Direction Inbound `
        -Protocol TCP `
        -LocalPort $port `
        -Action Allow `
        -Profile Private,Domain `
        -Description "Allow inbound to UNDERDOG $name on port $port" `
        -ErrorAction SilentlyContinue
}

Write-Host "✅ Firewall rules created for ports: 8000, 8501, 9090, 3000" -ForegroundColor Green
```

---

**Fin del Audit Report**  
Para ejecutar: Seguir SOLUCION_MANUAL.md + este documento  
Última actualización: 2025-10-20
