# 🎯 Resumen Ejecutivo: Pivote Arquitectónico UNDERDOG

**Fecha:** 21 de Octubre de 2025  
**Decisión:** Migración de arquitectura HistData + Indicadores Precalculados → Event-Driven Framework Robusto

---

## 📊 Contexto y Motivación

### Situación Previa
- ✅ Phase 1 completa: Cleanup y reorganización
- ⏳ Phase 2 en progreso: Descarga manual de HistData
  - **Problemas encontrados:** 6 bugs críticos en API `histdata` library
  - **Tiempo estimado:** ~7 horas de descarga + 2-3 horas de resampling
  - **Almacenamiento:** ~105 GB de Parquet files
  - **Error recurrente:** "I/O operation on closed file" durante resampling (no resuelto después de 5 intentos)

### Conclusión de Investigación
El usuario realizó una investigación exhaustiva sobre arquitecturas de backtesting robustas para Forex y Machine Learning, identificando las siguientes limitaciones del enfoque actual:

1. **Vectorized backtesting insuficiente:**
   - No modela microestructura de mercado (spreads dinámicos, slippage)
   - Infla resultados artificialmente
   - No preparado para Prop Firms (límites MDD estrictos)

2. **Indicadores precalculados contraproducentes:**
   - Event-Driven engines calculan indicadores on-the-fly
   - Precálculo masivo desperdicia 7+ horas de compute
   - Dificulta Walk-Forward Optimization

3. **Falta de separación de responsabilidades:**
   - Estrategias MQL5 monolíticas
   - Dificulta integración de modelos ML
   - Código no reutilizable entre backtest → paper → live

---

## 🏗️ Nueva Arquitectura: Event-Driven con Strategy Pattern

### Principios Fundamentales

```
╔══════════════════════════════════════════════════════════╗
║         EVENT-DRIVEN BACKTESTING ENGINE                  ║
║  (Procesa eventos cronológicamente: Market → Signal      ║
║   → Order → Fill, como en trading real)                  ║
╚══════════════════════════════════════════════════════════╝
                          │
        ┌─────────────────┼─────────────────┐
        ▼                 ▼                 ▼
  ┌───────────┐     ┌──────────┐     ┌───────────┐
  │  Strategy │     │Portfolio │     │ Execution │
  │  (ABC)    │────→│ Manager  │────→│  Handler  │
  └───────────┘     └──────────┘     └───────────┘
   • generate_      • Position       • Simulated
     signal()         sizing          (slippage)
   • ML models      • Risk mgmt      • Live Broker
   • Indicators     • P&L calc         (MT5/IBKR)
```

### Componentes Creados

1. **`underdog/core/abstractions.py`** (✅ Completado)
   - `Strategy(ABC)`: Interface para generar señales
   - `DataHandler(ABC)`: Interface para proveer datos
   - `Portfolio(ABC)`: Gestión de posiciones y P&L
   - `ExecutionHandler(ABC)`: Interface para ejecución
   - `RiskManager(ABC)`: Validación de riesgo
   - Data structures: `MarketEvent`, `SignalEvent`, `OrderEvent`, `FillEvent`

2. **`docs/ARCHITECTURE_REFACTOR.md`** (✅ Completado)
   - Comparativa Backtrader vs Lean Engine
   - Pipeline de preprocesamiento ML (Log Returns, ADF Test, Feature Engineering)
   - Walk-Forward Optimization framework
   - Risk Management (CVaR, Calmar Ratio, Kelly Criterion)
   - Roadmap de implementación (4 sprints)

3. **`docs/FRAMEWORK_EVALUATION.md`** (✅ Completado)
   - Template de evaluación práctica
   - 6 tests comparativos (instalación, SMA crossover, microestructura, ML, WFO, HF data)
   - Matriz de decisión con scoring

---

## 📋 Decisiones Arquitectónicas

### 1. Framework de Backtesting

**Opciones evaluadas:**
- ✅ **Backtrader:** Event-Driven, Pythonic, control total, flexible
- ✅ **Lean Engine:** Event-Driven, C# core con Python API, datos curados, producción seamless
- ❌ **Zipline:** Descartado (legacy, no mantenido, centrado en equities)

**Decisión pendiente:** Ejecutar tests prácticos (ver `FRAMEWORK_EVALUATION.md`) antes de comprometer.

**Recomendación actual:**
- **Backtrader** para prototipado inicial (menor curva de aprendizaje, control local)
- Evaluar **Lean Engine** si se requieren datos curados profesionales o cloud deployment

### 2. Fuente de Datos

**Antes:**
- ❌ Descarga manual de HistData.com (histdata library)
- ❌ Almacenamiento local masivo (105 GB)
- ❌ Proceso lento (7+ horas)

**Después:**
- ✅ **Hugging Face Datasets Hub**
  - Forex OHLCV (1M/5M): EURUSD, GBPUSD, USDJPY, XAUUSD, etc.
  - Forex Factory News (calendario económico)
  - Carga instantánea con `datasets.load_dataset()`
  - Sin almacenamiento local necesario

### 3. Preprocesamiento de Datos para ML

**Pipeline estándar:**
```python
1. Log Returns: ln(P_t / P_{t-1})        # Estacionariedad
2. ADF Test: p-value < 0.05              # Validación
3. Lagged Features: return_{t-1}, ...    # Autocorrelación
4. Technical Features: ATR, RSI, SMA     # TA-Lib
5. Standardization: μ=0, σ=1             # StandardScaler
```

**Crítico:** Nunca usar precios raw en modelos ML → regresión espuria garantizada.

### 4. Validación Rigurosa

**Walk-Forward Optimization (WFO):**
```
┌────────────────────────────────────────────┐
│ Year 1-5: Train → Year 6: Test (OOS)      │
│ Year 2-6: Train → Year 7: Test (OOS)      │
│ Year 3-7: Train → Year 8: Test (OOS)      │
│ ...                                        │
│ Concatenate all OOS → True Performance    │
└────────────────────────────────────────────┘
```

**Monte Carlo Validation:**
- Shuffle trades 10,000 veces
- Si equity curve original < 5th percentile → Lucky backtest (no robusto)

### 5. Risk Management Avanzado

**Métricas prioritarias:**
1. **Calmar Ratio** (función objetivo): `Return / |Max Drawdown|`
   - Target: > 2.0 para Prop Firms
2. **CVaR (Expected Shortfall)**: Pérdida promedio en 5% peores escenarios
   - Monitoreo rodante para tail risk
3. **Kelly Criterion** (position sizing): Fracción óptima del capital
   - Usar Half-Kelly (50%) para reducir volatilidad

---

## 🗂️ Nueva Estructura de Archivos

```
underdog/
├── core/
│   ├── abstractions.py       ✅ (Creado)
│   ├── event_engine.py       ⏳ (Pendiente)
│   └── portfolio.py          ⏳ (Pendiente)
├── data/
│   ├── hf_loader.py          ⏳ (Pendiente - Hugging Face)
│   └── feeds/                ⏳ (Custom DataFeeds)
├── strategies/
│   ├── base_strategy.py      ⏳ (Strategy ABC impl)
│   ├── sma_crossover.py      ⏳ (Migrar de MQL5)
│   └── ml_predictor.py       ⏳ (ML strategy)
├── ml/
│   ├── preprocessing.py      ⏳ (Log returns, ADF, features)
│   ├── models/               ⏳ (Keras/PyTorch)
│   └── feature_store.py      ⏳ (Cached features)
├── execution/
│   ├── simulated.py          ⏳ (Backtesting execution)
│   └── live_broker.py        ⏳ (MT5 connector)
├── risk/
│   ├── rme.py                ⏳ (Risk Management Engine)
│   ├── metrics.py            ⏳ (CVaR, Calmar, etc.)
│   └── position_sizing.py    ⏳ (Kelly Criterion)
├── validation/
│   ├── wfo.py                ⏳ (Walk-Forward Optimization)
│   └── monte_carlo.py        ⏳ (Trade shuffling)
└── backtesting/
    ├── backtrader_engine.py  ⏳ (Backtrader integration)
    └── lean_engine.py        ⏳ (Lean integration - opcional)
```

---

## 🗑️ Archivos a Eliminar (Obsoletos)

### Scripts de HistData (No necesarios)
- ❌ `scripts/download_histdata_1min_only.py`
- ❌ `scripts/download_liquid_pairs.py`
- ❌ `scripts/download_all_histdata.py`
- ❌ `scripts/resample_and_calculate.py`
- ❌ `scripts/insert_indicators_to_db.py`
- ❌ `scripts/setup_db_simple.ps1`

### Datos Locales
- ❌ `data/parquet/` (Parquet files descargados)
- ❌ `data/raw/` (ZIPs de HistData)

### Schemas DB
- ⚠️ `docker/init-indicators-db.sql` (mantener solo para live trading, no para backtesting)

**Razón:** Event-Driven frameworks calculan indicadores on-the-fly durante el event loop. No se requiere precálculo masivo.

---

## 📅 Roadmap de Implementación

### Sprint 1: Framework Setup (1-2 días) - **EN PROGRESO**
- [x] Crear `docs/ARCHITECTURE_REFACTOR.md`
- [x] Crear `underdog/core/abstractions.py` con ABCs
- [x] Crear `docs/FRAMEWORK_EVALUATION.md`
- [ ] **Ejecutar tests prácticos (Backtrader vs Lean)**
- [ ] Seleccionar framework definitivo
- [ ] `poetry add [framework]`

### Sprint 2: Integración de Datos (1 día)
- [ ] Explorar Hugging Face Datasets (Forex OHLCV)
- [ ] Crear `underdog/data/hf_loader.py`
- [ ] Implementar `HuggingFaceDataHandler(DataHandler)`
- [ ] Test: Cargar EURUSD 2020-2024 y validar formato

### Sprint 3: Migración de Primera Estrategia (1-2 días)
- [ ] Migrar SMA Crossover de MQL5 → Python
- [ ] Crear `underdog/strategies/sma_crossover.py(Strategy)`
- [ ] Implementar `generate_signal()` method
- [ ] Test: Backtest EURUSD 2020-2024
- [ ] Comparar resultados vs MQL5 backtest

### Sprint 4: ML Pipeline (2-3 días)
- [ ] Crear `underdog/ml/preprocessing.py`
- [ ] Implementar Log Returns + ADF Test
- [ ] Feature engineering (lagged, ATR, RSI)
- [ ] StandardScaler integration
- [ ] Test: Entrenar Logistic Regression simple

### Sprint 5: WFO + Validation (2-3 días)
- [ ] Crear `underdog/validation/wfo.py`
- [ ] Implementar rolling IS/OOS windows
- [ ] Implementar Monte Carlo shuffling
- [ ] Test: WFO sobre SMA Crossover (5 años)

### Sprint 6: Risk Management (1-2 días)
- [ ] Crear `underdog/risk/rme.py`
- [ ] Implementar CVaR calculation
- [ ] Implementar Calmar Ratio
- [ ] Implementar Kelly Criterion
- [ ] Test: Backtest con position sizing dinámico

### Sprint 7: Limpieza (0.5 días)
- [ ] Eliminar scripts obsoletos de HistData
- [ ] Eliminar Parquet files descargados
- [ ] Actualizar `README.md` con nueva arquitectura
- [ ] Git commit: "feat: Event-Driven architecture migration"

---

## ⚠️ Risks y Mitigaciones

| **Riesgo**                           | **Probabilidad** | **Impacto** | **Mitigación**                                  |
|--------------------------------------|------------------|-------------|-------------------------------------------------|
| Curva de aprendizaje Lean Engine     | Media            | Alto        | Empezar con Backtrader (más simple)             |
| Datasets HF incompletos/baja calidad | Baja             | Medio       | Validar coverage temporal antes de comprometer  |
| Performance (Python vs MQL5 C++)     | Media            | Medio       | Usar NumPy/Pandas vectorizado, Numba JIT        |
| WFO computacionalmente costoso       | Alta             | Bajo        | Paralelizar con joblib/multiprocessing          |

---

## ✅ Criterios de Éxito

### Fase 1: Arquitectura (Sprint 1-3)
- [x] ABCs definidas y documentadas
- [ ] Framework seleccionado e instalado
- [ ] Primer backtest Event-Driven ejecutado
- [ ] Resultados comparables con MQL5

### Fase 2: ML Integration (Sprint 4)
- [ ] Pipeline de preprocesamiento funcional
- [ ] ADF test passing (p-value < 0.05)
- [ ] Modelo simple entrenado y validado

### Fase 3: Validación Rigurosa (Sprint 5)
- [ ] WFO implementado (3+ ventanas)
- [ ] Monte Carlo validation passing (>5th percentile)
- [ ] Out-of-sample results positivos

### Fase 4: Production-Ready (Sprint 6-7)
- [ ] Risk Management Engine operacional
- [ ] MDD < 6% (Prop Firm compliant)
- [ ] Calmar Ratio > 2.0
- [ ] Código limpio y documentado

---

## 📚 Referencias

### Papers Académicos
- **Lopez de Prado (2018)**: *Advances in Financial Machine Learning*
- **Pardo (2008)**: *The Evaluation and Optimization of Trading Strategies*

### Documentación
- [Backtrader Docs](https://www.backtrader.com/docu/)
- [QuantConnect Lean Docs](https://www.quantconnect.com/docs/v2/)
- [Hugging Face Datasets](https://huggingface.co/docs/datasets/)

### Python Libraries
```toml
[tool.poetry.dependencies]
backtrader = "^1.9.78"              # Event-Driven backtesting
# lean-cli = "^1.0.0"               # QuantConnect (opcional)
huggingface-hub = "^0.20.0"         # HF Datasets
datasets = "^2.16.0"                # Data loading
ta-lib = "^0.4.28"                  # Technical indicators
statsmodels = "^0.14.0"             # ADF Test
scikit-learn = "^1.3.0"             # ML models, StandardScaler
```

---

## 🎯 Próxima Acción Inmediata

**Task:** Ejecutar evaluación práctica de frameworks  
**File:** `docs/FRAMEWORK_EVALUATION.md`  
**Owner:** @user  
**Estimado:** 2-3 horas  

**Steps:**
1. Instalar Backtrader: `poetry add backtrader`
2. Ejecutar Test 2 (SMA Crossover simple)
3. Ejecutar Test 3 (Spread/Slippage modeling)
4. Completar matriz de decisión
5. Seleccionar framework definitivo

---

**Status:** 🟢 Arquitectura definida, evaluación en progreso  
**Version:** 1.0  
**Last Updated:** 2025-10-21
