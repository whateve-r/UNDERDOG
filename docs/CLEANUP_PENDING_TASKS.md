# Tareas Pendientes Post-Limpieza

## ✅ Completado
- [x] Eliminar 46 archivos obsoletos (scripts + módulos)
- [x] Actualizar `underdog/data/__init__.py` → MT5HistoricalDataLoader
- [x] Comentar lógica HuggingFace en `bt_engine.py`
- [x] Git commit: d4bd4ba

## 🔴 Pendiente - Manual

### 1. Consolidar Carpetas Risk (ALTA PRIORIDAD)
**Problema:** Existen 2 carpetas con contenido diferente:
- `underdog/risk/` → prop_firm_rme.py
- `underdog/risk_management/` → cvar.py, position_sizing.py, risk_master.py, rules_engine.py

**Acción Requerida:**
1. Revisar contenido de ambas carpetas
2. Decidir estructura final (¿fusionar en `underdog/risk/`?)
3. Mover archivos necesarios
4. Actualizar imports en todo el proyecto
5. Eliminar carpeta vacía

### 2. Limpieza Opcional de Carpetas Data (BAJA PRIORIDAD)
**ADVERTENCIA:** Hacer backup primero

Carpetas candidatas a eliminar:
- `data/raw/` → CSVs antiguos de Dukascopy/HistData
- `data/historical/` → Cache HuggingFace
- `data/parquet/` → Datos obsoletos en Parquet

Carpetas a **MANTENER**:
- `data/mt5_historical/` → Datos de MT5 (fuente actual)
- `data/processed/` → Resultados de backtests
- `data/test_results/` → Logs de pruebas

**Comando sugerido:**
```powershell
# BACKUP PRIMERO
tar -czf data_backup_$(Get-Date -Format 'yyyyMMdd').tar.gz data/

# Verificar contenido antes de eliminar
ls data/raw/
ls data/historical/
ls data/parquet/

# Eliminar si están vacías o no son críticas
Remove-Item -Recurse data/raw/
Remove-Item -Recurse data/historical/
Remove-Item -Recurse data/parquet/
```

## 📋 Próximos Pasos - Nueva Arquitectura

### 3. Setup TimescaleDB (SIGUIENTE)
**Referencia:** `docs/AUDIT_CLEANUP_2025_10_22.md` sección "Nueva Arquitectura"

Crear `scripts/setup_timescaledb.py`:
```python
# 1. Docker Compose con TimescaleDB
# 2. Crear hypertables:
#    - ohlcv_data (time, symbol, open, high, low, close, volume)
#    - reddit_sentiment (time, subreddit, symbol, score, compound_sentiment)
#    - fred_indicators (time, indicator_name, value)
#    - alpha_vantage_fundamentals (time, symbol, metric, value)
# 3. Configurar retention policies (mantener 5 años)
# 4. Crear continuous aggregates (1min → 5min → 15min → 1h → 1d)
```

### 4. Backfill yfinance (SEMANA 1)
Crear `scripts/backfill_yfinance.py`:
- Download 5 años de OHLCV para majors (EURUSD, GBPUSD, etc)
- Insertar en TimescaleDB hypertable `ohlcv_data`
- Verificar performance (objetivo: >10k bars/segundo)

### 5. Backfill FRED (SEMANA 1)
Crear `scripts/backfill_fred.py`:
- Indicadores macro: DFF (Fed Funds Rate), T10Y2Y (Yield Curve), VIX, etc
- Insertar en hypertable `fred_indicators`

### 6. Reddit Scraper (SEMANA 2)
Crear `underdog/data/reddit_scraper.py`:
- Reddit API → r/wallstreetbets, r/forex
- Sentiment analysis con VADER/FinBERT
- Insertar en hypertable `reddit_sentiment`

### 7. Alpha Vantage Fundamentals (SEMANA 2)
Crear `underdog/data/alpha_vantage_loader.py`:
- Fundamentals para acciones correlacionadas (SPY, etc)
- Insertar en hypertable `alpha_vantage_fundamentals`

### 8. Fix MT5 Async Issue (SEMANA 3)
**Problema:** `test_mt5_historical_download.py` timeout en async ZMQ
**Root cause:** Multiple PULL sockets → round-robin message loss

**Soluciones posibles:**
1. Singleton `Mt5Connector` (evitar múltiples sockets)
2. REQ/REP pattern en vez de PULL/PUSH
3. Usar `download_sync()` directamente (ya funciona)

## 📊 Métricas de Éxito

- [x] 46 archivos eliminados (de 47 scripts → 8 esenciales = **83% reducción**)
- [x] 11,643 líneas eliminadas
- [ ] TimescaleDB funcionando con 4 fuentes de datos
- [ ] Backtests 10x más rápidos (TimescaleDB vs PostgreSQL)
- [ ] Alternative data integrada (Reddit + FRED)
- [ ] MT5 async funcionando sin timeouts

## 📅 Timeline Sugerido

| Semana | Tarea | Horas | Prioridad |
|--------|-------|-------|-----------|
| 1 | Consolidar risk/ | 2h | Alta |
| 1 | Setup TimescaleDB | 4h | Crítica |
| 1 | Backfill yfinance | 3h | Alta |
| 1 | Backfill FRED | 2h | Media |
| 2 | Reddit scraper | 5h | Media |
| 2 | Alpha Vantage | 3h | Baja |
| 3 | Fix MT5 async | 3h | Media |
| 3 | Tests integración | 3h | Alta |

**Total:** ~25 horas → **3 semanas part-time**

---

**Última actualización:** 2025-01-22  
**Commit limpieza:** d4bd4ba  
**Checkpoint pre-limpieza:** 33fde7f (rollback disponible)
