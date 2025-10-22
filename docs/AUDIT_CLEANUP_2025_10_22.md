# 🔍 Auditoría y Limpieza del Proyecto UNDERDOG
**Fecha:** 22 de Octubre 2025  
**Objetivo:** Identificar y eliminar componentes obsoletos antes de migrar a TimescaleDB

---

## 📋 Executive Summary

El proyecto UNDERDOG ha evolucionado a través de múltiples fases experimentales, dejando:
- **47 scripts** en `/scripts` (muchos de prueba/diagnóstico)
- **Múltiples fuentes de datos obsoletas** (HuggingFace, Dukascopy, HistData)
- **Carpetas duplicadas** (`risk/` y `risk_management/`)
- **Configuraciones antiguas** (Lean, Streamlit, múltiples DBs)

**Estrategia Nueva:** Migrar a TimescaleDB con fuentes: yfinance, Alpha Vantage, Reddit, FRED (inspirado en AlphaQuanter)

---

## 🗑️ COMPONENTES A ELIMINAR

### 1️⃣ **Scripts Obsoletos de Testing/Diagnóstico** (17 archivos)

#### Scripts de Diagnóstico ZMQ/MT5 (ya resueltos)
```
✗ scripts/test_zmq_raw.py                  # Diagnóstico ZMQ - YA RESUELTO
✗ scripts/test_zmq_dual.py                 # Test dual socket - YA RESUELTO
✗ scripts/test_history_raw.py              # Test HISTORY raw - YA RESUELTO
✗ scripts/test_history_count.py            # Test con COUNT - YA RESUELTO
✗ scripts/test_history_full_output.py      # Test output completo - YA RESUELTO
✗ scripts/test_history_data_first.py       # Test orden sockets - YA RESUELTO
✗ scripts/diagnose_zmq_connection.py       # Diagnóstico completo ZMQ - YA RESUELTO
✗ scripts/test_mt5_connection.py           # Test conexión MT5 - DUPLICADO de test_mt5_simple.py
```
**Razón:** Problemas ZMQ ya resueltos. `test_mt5_simple.py` y `test_mt5_historical_download.py` son suficientes.

#### Scripts de Testing de Librerías Obsoletas
```
✗ scripts/test_lean_install.py             # QuantConnect Lean - NO SE USA
✗ scripts/test_lean_simple.py              # QuantConnect Lean - NO SE USA
✗ scripts/test_hf_loader.py                # HuggingFace loader - SE REEMPLAZARÁ
✗ scripts/explore_hf_forex_datasets.py     # HuggingFace datasets - SE REEMPLAZARÁ
```
**Razón:** Lean nunca se implementó. HuggingFace se reemplaza con yfinance/Alpha Vantage.

#### Scripts de Testing de Backtrader Duplicados
```
✗ scripts/test_backtrader.py               # Test básico - DUPLICADO
✗ scripts/test_backtrader_simple.py        # Test simple - DUPLICADO
✗ scripts/test_bt_minimal.py               # Test mínimo - DUPLICADO
```
**Mantener:** `scripts/test_end_to_end.py` (test completo e2e con PropFirm rules)

#### Scripts de Testing de TA-Lib
```
✗ scripts/test_talib_performance.py        # Performance test TA-Lib
```
**Razón:** TA-Lib validado y en producción. No necesitamos benchmark continuo.

#### Scripts de Testing Científicos
```
✗ scripts/test_scientific_improvements.py  # Test mejoras científicas de Phase 3
```
**Razón:** Mejoras científicas validadas e integradas en `bt_engine.py`.

---

### 2️⃣ **Fuentes de Datos Obsoletas** (6 archivos + carpetas)

#### Loaders de Datos Antiguos
```
✗ underdog/data/hf_loader.py               # HuggingFace loader - SE REEMPLAZA
✗ underdog/data/dukascopy_loader.py        # Dukascopy loader - SE REEMPLAZA
✗ underdog/database/histdata_ingestion.py  # HistData ingestion - SE REEMPLAZA
```

#### Scripts de Descarga de Datos Antiguos
```
✗ scripts/download_all_histdata.py         # HistData download - OBSOLETO
✗ scripts/download_histdata_1min_only.py   # HistData M1 only - OBSOLETO
✗ scripts/backfill_histdata_parquet.py     # HistData → Parquet - OBSOLETO
✗ scripts/backfill_all_data.py             # Backfill genérico - OBSOLETO
✗ scripts/setup_hf_token.py                # HuggingFace token setup - OBSOLETO
```

#### Carpetas de Datos Antiguos
```
✗ data/raw/                                # HistData CSVs antiguos
✗ data/historical/                         # Dukascopy/HF data cache
✗ data/parquet/                            # Parquet cache antiguo (pre-TimescaleDB)
```

**Mantener:**
```
✓ data/mt5_historical/                     # Cache de MT5 (datos reales broker)
✓ data/processed/                          # Features procesados para ML
✓ data/test_results/                       # Resultados de backtests
```

---

### 3️⃣ **Scripts de Configuración/Setup Antiguos** (5 archivos)

```
✗ scripts/setup_db_simple.ps1              # Setup PostgreSQL antiguo
✗ scripts/setup_indicators_db.ps1          # Setup DB indicadores
✗ scripts/setup_firewall_rules.ps1         # Firewall rules (específico VPS anterior)
✗ scripts/insert_indicators_to_db.py       # Insert indicadores → DB antigua
✗ scripts/resample_and_calculate.py        # Resample + calc indicadores → DB antigua
```
**Razón:** Migración a TimescaleDB requiere nuevos scripts de setup.

---

### 4️⃣ **Scripts de Demo/Integración Incompletos** (5 archivos)

```
✗ scripts/demo_paper_trading.py            # Demo paper trading - INCOMPLETO (bloqueado por MT5Executor refactor)
✗ scripts/complete_trading_workflow.py     # Workflow completo - INCOMPLETO
✗ scripts/integrated_trading_system.py     # Sistema integrado - INCOMPLETO
✗ scripts/integration_test.py              # Test integración - SUPERSEDED por test_end_to_end.py
✗ scripts/start_live.py                    # Start live trading - INCOMPLETO
```
**Razón:** Workflows incompletos reemplazados por `start_trading_with_monitoring.py` (más completo).

---

### 5️⃣ **Scripts de Monitoring/Prometheus Antiguos** (3 archivos)

```
✗ scripts/instrument_eas_prometheus.py     # Instrumentación EAs - OBSOLETO (JsonAPI EA no tiene esto)
✗ scripts/generate_test_metrics.py         # Generar métricas test - OBSOLETO
✗ scripts/start_metrics.ps1                # Start Prometheus - REEMPLAZAR con docker-compose
```
**Mantener:** `scripts/monitoring_example.py` (ejemplo funcional)

---

### 6️⃣ **Scripts de ML/Preprocessing Antiguos** (3 archivos)

```
✗ scripts/ml_preprocessing_example.py      # Ejemplo preprocessing - INCOMPLETO
✗ scripts/retrain_models.py                # Retrain ML models - INCOMPLETO
✗ scripts/generate_synthetic_data.py       # Generate synthetic data - NO SE USA (backtest usa synthetic inline)
```
**Razón:** ML pipeline incompleto. Con nueva arquitectura TimescaleDB + AlphaQuanter, necesitaremos reescribir.

---

### 7️⃣ **Scripts de Descarga Índices/Pares** (2 archivos)

```
✗ scripts/download_indices.py              # Download índices económicos - SE REEMPLAZA con FRED
✗ scripts/download_liquid_pairs.py         # Download pares líquidos - SE REEMPLAZA con yfinance
```

---

### 8️⃣ **Carpetas/Módulos Duplicados**

```
✗ underdog/risk/                           # DUPLICADO de underdog/risk_management/
```
**Acción:** Consolidar en `underdog/risk_management/` (más específico).

---

### 9️⃣ **UI Streamlit Obsoleto**

```
✗ scripts/streamlit_dashboard.py          # Streamlit UI - NO SE USA (preferimos Grafana)
✗ underdog/ui/                             # Módulo UI incompleto
```
**Razón:** Grafana es mejor para monitoreo 24/7. Streamlit es para exploración, no producción.

---

### 🔟 **Scripts de Start Incompletos**

```
✗ scripts/start_backtest.py               # Start backtest - SUPERSEDED por test_end_to_end.py
```
**Mantener:** `scripts/start_trading_with_monitoring.py` (único script de start completo).

---

## ✅ COMPONENTES A MANTENER (Core del Sistema)

### Scripts Esenciales (8 archivos)
```
✓ scripts/test_mt5_simple.py               # Test conexión MT5 ZMQ (producción)
✓ scripts/test_mt5_historical_download.py  # Test descarga datos MT5 (producción)
✓ scripts/test_end_to_end.py               # Test e2e backtest con PropFirm rules
✓ scripts/start_trading_with_monitoring.py # Sistema completo live + monitoring
✓ scripts/monitoring_example.py            # Ejemplo Prometheus/Grafana
✓ scripts/desc.py                          # Utilidad descripción (?)
```

### Core del Sistema (`underdog/`)
```
✓ underdog/backtesting/                    # Backtrader engine + adaptadores
  ✓ bt_engine.py                           # Backtesting engine (PRODUCTION READY)
  ✓ bt_adapter.py                          # Backtrader → internal adapter
  
✓ underdog/bridges/                        # Bridges estrategias
  ✓ bt_to_mt5.py                           # Backtrader → MT5 bridge (PRODUCTION READY)
  
✓ underdog/core/                           # Core abstracciones
  ✓ connectors/mt5_connector.py            # MT5 ZMQ connector (PRODUCTION READY)
  ✓ schemas/                               # Schemas Pydantic
  ✓ abstractions.py                        # Base classes
  ✓ logger.py                              # Logging system
  
✓ underdog/data/                           # Data handlers
  ✓ mt5_historical_loader.py               # MT5 historical data (PRODUCTION READY)
  
✓ underdog/execution/                      # Execution layer
  ✓ mt5_executor.py                        # MT5 order executor (NEEDS REFACTOR to use mt5_connector.py)
  
✓ underdog/risk_management/                # Risk management
  ✓ propfirm_risk.py                       # PropFirm rules (FTMO/FTUK) (PRODUCTION READY)
  
✓ underdog/strategies/                     # Trading strategies
  ✓ atr_breakout.py                        # ATR Breakout strategy (PRODUCTION READY)
  
✓ underdog/monitoring/                     # Monitoring
  ✓ prometheus_metrics.py                  # Prometheus instrumentation
  
✓ underdog/validation/                     # Validation
  ✓ monte_carlo.py                         # Monte Carlo simulation (PRODUCTION READY)
```

---

## 📦 PLAN DE MIGRACIÓN A TimescaleDB

### Nuevas Fuentes de Datos (inspirado en AlphaQuanter)
```
1. yfinance          → Precios históricos (OHLCV)
2. Alpha Vantage     → Datos fundamentales + económicos
3. Reddit API        → Sentiment analysis (r/wallstreetbets, r/stocks)
4. FRED API          → Indicadores macroeconómicos (Fed Reserve)
```

### Nueva Arquitectura de Datos
```
┌─────────────────────────────────────────────────────────────┐
│                       TimescaleDB                            │
├─────────────────────────────────────────────────────────────┤
│  Hypertable: ohlcv_data (time, symbol, open, high, low...)  │
│  Hypertable: reddit_sentiment (time, subreddit, score...)   │
│  Hypertable: fred_indicators (time, indicator, value...)    │
│  Table: alpha_vantage_fundamentals (company, pe, ...)       │
└─────────────────────────────────────────────────────────────┘
         ↑                    ↑                    ↑
         │                    │                    │
    yfinance           Reddit API           FRED API
    backfill           scraper              backfill
```

### Scripts Nuevos a Crear
```
□ scripts/setup_timescaledb.py              # Setup TimescaleDB hypertables
□ scripts/backfill_yfinance.py              # Backfill desde yfinance
□ scripts/backfill_fred.py                  # Backfill FRED indicators
□ scripts/backfill_reddit.py                # Backfill Reddit sentiment
□ underdog/data/timescale_loader.py         # TimescaleDB data loader
□ underdog/data/yfinance_ingestion.py       # yfinance → TimescaleDB
□ underdog/data/fred_ingestion.py           # FRED → TimescaleDB
□ underdog/data/reddit_ingestion.py         # Reddit → TimescaleDB
```

---

## 🎯 ACCIÓN RECOMENDADA

### Fase 1: Limpieza Inmediata (10 min)
```bash
# Eliminar scripts de diagnóstico ZMQ resueltos
rm scripts/test_zmq_*.py
rm scripts/test_history_*.py
rm scripts/diagnose_zmq_connection.py
rm scripts/test_mt5_connection.py

# Eliminar scripts de testing duplicados
rm scripts/test_backtrader*.py
rm scripts/test_bt_minimal.py
rm scripts/test_talib_performance.py
rm scripts/test_scientific_improvements.py

# Eliminar loaders antiguos de datos
rm scripts/download_all_histdata.py
rm scripts/download_histdata_1min_only.py
rm scripts/backfill_*.py
rm scripts/setup_hf_token.py
rm scripts/test_hf_loader.py
rm scripts/explore_hf_forex_datasets.py

# Eliminar scripts de setup antiguos
rm scripts/setup_db_simple.ps1
rm scripts/setup_indicators_db.ps1
rm scripts/setup_firewall_rules.ps1
rm scripts/insert_indicators_to_db.py
rm scripts/resample_and_calculate.py

# Eliminar demos incompletos
rm scripts/demo_paper_trading.py
rm scripts/complete_trading_workflow.py
rm scripts/integrated_trading_system.py
rm scripts/integration_test.py
rm scripts/start_live.py
rm scripts/start_backtest.py

# Eliminar monitoring antiguo
rm scripts/instrument_eas_prometheus.py
rm scripts/generate_test_metrics.py
rm scripts/start_metrics.ps1

# Eliminar ML incompleto
rm scripts/ml_preprocessing_example.py
rm scripts/retrain_models.py
rm scripts/generate_synthetic_data.py

# Eliminar descarga índices antiguos
rm scripts/download_indices.py
rm scripts/download_liquid_pairs.py

# Eliminar UI Streamlit
rm scripts/streamlit_dashboard.py

# Eliminar tests de Lean
rm scripts/test_lean_*.py
```

### Fase 2: Limpieza de Módulos (5 min)
```bash
# Eliminar loaders de datos antiguos
rm underdog/data/hf_loader.py
rm underdog/data/dukascopy_loader.py
rm underdog/database/histdata_ingestion.py

# Consolidar risk management
mv underdog/risk_management/* underdog/risk/
rmdir underdog/risk_management

# Eliminar UI incompleto
rm -rf underdog/ui/
```

### Fase 3: Limpieza de Datos (5 min)
```bash
# Eliminar carpetas de datos antiguos (CUIDADO: backup primero si hay algo importante)
rm -rf data/raw/
rm -rf data/historical/
rm -rf data/parquet/

# Mantener:
# - data/mt5_historical/ (cache MT5 real broker)
# - data/processed/ (features ML)
# - data/test_results/ (resultados backtests)
```

### Fase 4: Actualizar Imports (5 min)
```python
# underdog/data/__init__.py
# ANTES:
from underdog.data.hf_loader import HuggingFaceDataHandler, ForexNewsHandler
__all__ = ['HuggingFaceDataHandler', 'ForexNewsHandler']

# DESPUÉS:
from underdog.data.mt5_historical_loader import MT5HistoricalDataLoader
__all__ = ['MT5HistoricalDataLoader']
```

```python
# underdog/backtesting/bt_engine.py
# ELIMINAR imports de HuggingFace:
# from underdog.data.hf_loader import HuggingFaceDataHandler

# MANTENER:
# - Synthetic data generation (inline)
# - MT5HistoricalDataLoader para datos reales
```

---

## 📊 RESUMEN DE LIMPIEZA

| Categoría | Archivos a Eliminar | Razón |
|-----------|---------------------|-------|
| Scripts diagnóstico ZMQ | 8 | Problemas resueltos |
| Scripts testing duplicados | 6 | Test e2e suficiente |
| Loaders datos antiguos | 9 | Migración a TimescaleDB |
| Scripts setup antiguos | 5 | Nueva arquitectura DB |
| Demos incompletos | 6 | Reemplazados |
| Monitoring antiguo | 3 | Docker + Grafana |
| ML incompleto | 3 | Pipeline nuevo |
| Descarga datos antiguo | 2 | yfinance/FRED |
| UI Streamlit | 2 | Grafana preferido |
| Tests Lean | 2 | Nunca usado |
| **TOTAL** | **46 archivos** | |

### Reducción de Scripts: 47 → **8 esenciales** (83% reducción)

---

## 🚀 PRÓXIMOS PASOS

1. ✅ **Limpieza** (este documento)
2. □ **Setup TimescaleDB** local (Docker)
3. □ **Crear backfill yfinance** (OHLCV histórico)
4. □ **Crear backfill FRED** (indicadores macro)
5. □ **Crear backfill Reddit** (sentiment)
6. □ **Integrar Alpha Vantage** (fundamentals)
7. □ **Actualizar bt_engine.py** para usar TimescaleDB
8. □ **Refactor MT5Executor** para usar mt5_connector.py
9. □ **Implementar continuous trading loop** (Task 2 en todo list)
10. □ **Deploy VPS** con TimescaleDB + Grafana

---

## 📚 Referencias
- AlphaQuanter: https://github.com/AlphaQuanter/AlphaQuanter
- TimescaleDB Docs: https://docs.timescale.com/
- yfinance: https://github.com/ranaroussi/yfinance
- FRED API: https://fred.stlouisfed.org/docs/api/
- Alpha Vantage: https://www.alphavantage.co/documentation/

---

**Autor:** GitHub Copilot  
**Reviewed by:** User (manud)  
**Status:** READY FOR EXECUTION
