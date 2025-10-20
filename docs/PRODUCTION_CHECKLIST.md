# 📋 UNDERDOG - PRODUCTION CHECKLIST

**Fecha inicio**: 2025-10-20  
**Objetivo**: Sistema en producción con MT5 en servidor  
**Tiempo estimado total**: 3-5 días

---

## 🎯 PROGRESO GENERAL

```
[██████░░░░] 60% - Fase 1 completada ✅
```

- [x] Fase 1: Limpieza (7/7) ✅ **COMPLETADA**
- [ ] Fase 2: Datos (0/15)
- [ ] Fase 3: MT5 (0/20)
- [ ] Fase 4: Config (0/5)
- [ ] Fase 5: Deploy (0/8)
- [ ] Fase 6: Mobile (0/4)

---

## 📝 FASE 1: LIMPIEZA (1.5-2.5h) ✅ **COMPLETADA**

### 1.1 EAs No Utilizados ✅
- [x] Listar archivos en `underdog/strategies/`
- [x] Crear carpeta `_archived/`
- [x] Mover EAs no usados (sin v4)
- [x] Actualizar imports
- [x] Ejecutar tests
- [x] Commit cambios

**Archivados**:
- ✅ `ea_parabolic_ema.py` → `_archived/` (tenemos `ea_parabolic_ema_v4.py`)
- ✅ `ea_supertrend_rsi.py` → `_archived/` (tenemos `ea_supertrend_rsi_v4.py`)

### 1.2 Documentación ✅
- [x] Mover `.md` a `docs/setup/`, `docs/troubleshooting/`
- [x] Crear `docs/README.md` con índice
- [x] Actualizar enlaces
- [x] Commit cambios

**Reorganizados**:
- ✅ 3 archivos → `docs/setup/` (CHECKLIST_STARTUP, DEMO_GUIDE, DEMO_STATUS)
- ✅ 7 archivos → `docs/troubleshooting/` (FIREWALL_*, GRAFANA_*, SOLUCION_MANUAL)
- ✅ 4 archivos → `docs/` (ESTADO_ACTUAL, TESTING_COMPLETE, PRODUCTION_CHECKLIST, ROADMAP_PRODUCTION)
- ✅ Creado `docs/README.md` con índice completo

**Commit**: `5328ac9` - refactor: Phase 1 cleanup

**Comandos rápidos**:
```bash
# Listar EAs
ls underdog/expert_advisors/*.py

# Mover docs
git mv CHECKLIST_STARTUP.md docs/setup/
git mv FIREWALL_*.md docs/troubleshooting/
git mv GRAFANA_*.md docs/monitoring/

# Commit
git commit -m "refactor: Clean up EAs and reorganize documentation"
```

---

## 💾 FASE 2: DATOS (4-6h)

### 2.1 FX-1-Minute-Data
- [ ] Clonar repo: `git clone https://github.com/philipperemy/FX-1-Minute-Data.git data/fx-repo`
- [ ] Analizar estructura de datos
- [ ] Crear `scripts/import_fx_minute_data.py`
- [ ] Importar EURUSD, GBPUSD, USDJPY (2020-2023)
- [ ] Verificar en TimescaleDB

### 2.2 Parquet
- [ ] Instalar: `poetry add pyarrow fastparquet`
- [ ] Crear `scripts/convert_to_parquet.py`
- [ ] Convertir datos a Parquet con compresión Snappy
- [ ] Benchmark: CSV vs Parquet
- [ ] Actualizar pipelines para usar Parquet

### 2.3 DuckDB
- [ ] Instalar: `poetry add duckdb`
- [ ] Crear `underdog/data/duckdb_client.py`
- [ ] Actualizar Streamlit para usar DuckDB
- [ ] Crear queries optimizadas
- [ ] Tests de performance

**Comandos rápidos**:
```bash
# Instalar deps
poetry add pyarrow fastparquet duckdb

# Importar datos
poetry run python scripts/import_fx_minute_data.py --pairs EURUSD,GBPUSD --years 2020-2023

# Convertir a Parquet
poetry run python scripts/convert_to_parquet.py --input data/raw --output data/raw/parquet

# Benchmark
poetry run python scripts/benchmark_data_formats.py
```

---

## 🔌 FASE 3: METATRADER 5 (8-12h)

### 3.1 Setup Conexión
- [ ] Instalar: `poetry add MetaTrader5`
- [ ] Crear `underdog/broker/mt5_connector.py`
- [ ] Agregar credenciales a `.env`: `MT5_LOGIN`, `MT5_PASSWORD`, `MT5_SERVER`
- [ ] Test: `poetry run python scripts/test_mt5_connection.py`
- [ ] Verificar info de cuenta

### 3.2 Streaming Tiempo Real
- [ ] Crear `underdog/broker/mt5_streamer.py`
- [ ] Implementar `MT5Streamer` con threading
- [ ] Crear `TickBuffer` para indicadores
- [ ] Integrar con EAs
- [ ] Test de latencia (<100ms)

### 3.3 Ejecución de Órdenes
- [ ] Crear `underdog/broker/order_manager.py`
- [ ] Implementar `OrderManager.process_signal()`
- [ ] Calcular position sizing (1% risk)
- [ ] Calcular SL/TP automático
- [ ] Logging de órdenes
- [ ] Guardar en TimescaleDB

### 3.4 Monitoreo Posiciones
- [ ] Crear `underdog/broker/position_monitor.py`
- [ ] Implementar trailing stop
- [ ] Auto-close en señales de salida
- [ ] Integrar con Prometheus

### 3.5 Grafana Live
- [ ] Agregar métricas MT5 a Prometheus
- [ ] Crear dashboard "Live Trading MT5"
- [ ] Configurar alertas (drawdown, conexión)
- [ ] Test visualización en tiempo real

**Comandos rápidos**:
```bash
# Instalar MT5
poetry add MetaTrader5

# Test conexión
poetry run python scripts/test_mt5_connection.py

# Iniciar live trading (DEMO)
poetry run python scripts/start_live_trading.py --demo

# Ver logs en tiempo real
tail -f logs/live_trading.log
```

---

## ⚙️ FASE 4: CONFIGURACIONES (2-3h)

- [ ] Consolidar variables en `.env`
- [ ] Crear `.env.example`
- [ ] Configurar logging JSON estructurado
- [ ] Logs rotatorios (10MB, 5 backups)
- [ ] Health check endpoint `/health`

**Comandos rápidos**:
```bash
# Validar .env
poetry run python -c "from dotenv import load_dotenv; load_dotenv(); import os; print('OK' if os.getenv('MT5_LOGIN') else 'MISSING')"

# Test health
curl http://localhost:8000/health
```

---

## 🚀 FASE 5: DEPLOYMENT (4-6h)

### 5.1 Servidor
- [ ] Crear servidor (DigitalOcean, AWS, Hetzner)
- [ ] Ubuntu 22.04, 4GB RAM, 20GB SSD
- [ ] Instalar Docker + Docker Compose
- [ ] Configurar usuario `underdog`
- [ ] Setup firewall (SSH, HTTP, HTTPS)

### 5.2 Docker Production
- [ ] Crear `docker-compose.prod.yml`
- [ ] Configurar Nginx reverse proxy
- [ ] Setup SSL con Let's Encrypt
- [ ] Configurar backups automáticos
- [ ] Deploy inicial

### 5.3 CI/CD
- [ ] Crear `.github/workflows/deploy.yml`
- [ ] Configurar secrets en GitHub
- [ ] Test auto-deploy en push a main

**Comandos rápidos**:
```bash
# SSH al servidor
ssh underdog@tu_servidor_ip

# Clonar repo
git clone https://github.com/whateve-r/UNDERDOG.git
cd UNDERDOG

# Deploy
docker-compose -f docker-compose.prod.yml up -d

# Ver logs
docker-compose -f docker-compose.prod.yml logs -f

# SSL
certbot --nginx -d tu_dominio.com
```

---

## 📱 FASE 6: MOBILE (2-3h)

- [ ] Obtener IP pública: `curl ifconfig.me`
- [ ] Configurar dominio (opcional)
- [ ] Actualizar `GF_SERVER_ROOT_URL`
- [ ] Instalar Grafana app en móvil
- [ ] Configurar conexión mobile
- [ ] Test dashboards en móvil

**Comandos rápidos**:
```bash
# IP pública
curl ifconfig.me

# Test acceso remoto
curl http://tu_ip:3000/api/health
```

---

## ✅ CHECKLIST PRE-PRODUCCIÓN

Antes de pasar de DEMO a LIVE:

- [ ] Sistema corriendo 1 semana en DEMO sin errores
- [ ] Winrate > 50% en DEMO
- [ ] Drawdown máximo < 10% en DEMO
- [ ] Todas las órdenes loggeadas correctamente
- [ ] Reconnect automático funcionando
- [ ] Alertas de Grafana probadas
- [ ] Backups funcionando y testeados
- [ ] Servidor con uptime > 99.9%
- [ ] Cuenta LIVE con balance mínimo ($500-1000)
- [ ] Risk management validado (1% por trade)

---

## 🔥 COMANDOS IMPORTANTES

### Inicio Rápido Local
```bash
# Iniciar Docker
docker-compose up -d

# Iniciar metrics generator
poetry run python scripts/generate_test_metrics.py

# Ver Grafana
start http://localhost:3000
```

### Inicio Live Trading
```bash
# DEMO
poetry run python scripts/start_live_trading.py --demo --pairs EURUSD,GBPUSD

# LIVE (después de validar DEMO)
poetry run python scripts/start_live_trading.py --live --pairs EURUSD
```

### Monitoreo
```bash
# Logs en tiempo real
tail -f logs/live_trading.log

# Estado de contenedores
docker ps

# Métricas Prometheus
curl http://localhost:9090/api/v1/query?query=underdog_mt5_balance

# Ver posiciones abiertas
poetry run python scripts/show_mt5_positions.py
```

### Deployment
```bash
# Build imagen
docker build -t underdog-trading:latest .

# Push a registry
docker push underdog-trading:latest

# Deploy en servidor
ssh underdog@servidor "cd UNDERDOG && docker-compose -f docker-compose.prod.yml pull && docker-compose -f docker-compose.prod.yml up -d"
```

---

## 📊 KPIs A MONITOREAR

| Métrica | Objetivo | Alerta |
|---------|----------|--------|
| Uptime | >99.9% | <99% |
| Latencia órdenes | <200ms | >500ms |
| Win rate | >55% | <50% |
| Drawdown diario | <3% | >5% |
| Profit factor | >1.5 | <1.2 |
| Sharpe ratio | >1.5 | <1.0 |

---

## 🆘 TROUBLESHOOTING RÁPIDO

**Problema**: MT5 no conecta
```bash
# Verificar credenciales
echo $MT5_LOGIN
# Verificar que MT5 terminal esté abierto
# Check firewall
```

**Problema**: Grafana sin datos
```bash
# Verificar Prometheus targets
curl http://localhost:9090/targets
# Reiniciar Grafana
docker-compose restart grafana
```

**Problema**: Servidor sin responder
```bash
# SSH al servidor
ssh underdog@servidor
# Ver logs de Docker
docker-compose logs --tail 100
# Reiniciar servicios
docker-compose restart
```

---

**Documentación completa**: Ver `ROADMAP_PRODUCTION.md`  
**Troubleshooting**: Ver `docs/troubleshooting/`  
**Setup inicial**: Ver `docs/setup/CHECKLIST_STARTUP.md`

