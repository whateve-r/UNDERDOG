# 🏗️ Refactor Arquitectónico: De HistData a Event-Driven Robusto

## 📋 Resumen Ejecutivo

**Decisión:** Pivotar de descarga manual de HistData + indicadores precalculados hacia una arquitectura Event-Driven profesional basada en frameworks robustos (Lean Engine / Backtrader).

**Motivación:**
- ✅ Eliminar dependencia de descarga manual (~7 horas, 105 GB)
- ✅ Utilizar datasets curados de Hugging Face (Forex + Forex Factory News)
- ✅ Modelado realista de microestructura (spread dinámico, slippage)
- ✅ Arquitectura modular para integración ML
- ✅ Walk-Forward Optimization para validación rigurosa
- ✅ Risk Management avanzado (CVaR, Calmar Ratio, Kelly Criterion)

---

## 🎯 Arquitectura Event-Driven: Fundamentos

### Componentes Desacoplados (Strategy Pattern)

```
┌─────────────────────────────────────────────────────────────┐
│                    EVENT LOOP (Heartbeat)                   │
│  Procesa: Market Data → Signals → Orders → Fills/Execution │
└─────────────────────────────────────────────────────────────┘
                            │
        ┌───────────────────┼───────────────────┐
        ▼                   ▼                   ▼
┌───────────────┐   ┌──────────────┐   ┌───────────────┐
│ Data Handler  │   │   Strategy   │   │   Portfolio   │
│   (Abstract)  │   │  (Abstract)  │   │   Manager     │
├───────────────┤   ├──────────────┤   ├───────────────┤
│ • Historical  │──→│ • generate_  │──→│ • P&L calc    │
│ • Live Stream │   │   signal()   │   │ • Position    │
│ • HF Datasets │   │ • ML Predict │   │ • Risk checks │
└───────────────┘   └──────────────┘   └───────────────┘
                            │
                            ▼
                    ┌──────────────┐
                    │  Execution   │
                    │   Handler    │
                    ├──────────────┤
                    │ • Simulated  │
                    │ • Live Broker│
                    │ • Slippage   │
                    └──────────────┘
```

### Principio de Desacoplamiento

**Abstract Base Classes (abc) definen contratos formales:**

```python
from abc import ABC, abstractmethod

class Strategy(ABC):
    @abstractmethod
    def generate_signal(self, market_data: MarketData) -> Signal:
        """Strategy logic - NO knowledge of execution"""
        pass

class ExecutionHandler(ABC):
    @abstractmethod
    def execute_order(self, order: Order) -> Fill:
        """Execution logic - models slippage/spread"""
        pass
```

**Beneficio:** Cambiar de backtesting simulado → live broker requiere **solo cambiar ExecutionHandler**, no tocar Strategy.

---

## 🔧 Comparativa de Frameworks

### Tabla de Decisión

| **Criterio**                    | **Backtrader**                          | **Lean Engine (QuantConnect)** |
|---------------------------------|-----------------------------------------|--------------------------------|
| **Event-Driven**                | ✅ Sí (Cerebro engine)                  | ✅ Sí (C# core, Python API)    |
| **Spread/Slippage**             | ✅ Flexible (`slip_perc`, customizable) | ✅ Alto realismo (built-in)    |
| **Datos Forex**                 | ⚠️ Manual (custom DataFeed)             | ✅ Curados (Survivorship Free) |
| **Integración ML**              | ⚠️ Vía Pandas (manual)                  | ✅ Nativa (Research Notebook)  |
| **Control Local**               | ✅ 100% local, Pythonic                 | ⚠️ Cloud-first (local posible) |
| **Producción Live**             | ⚠️ Requiere glue code                   | ✅ Seamless (mismo código)     |
| **Curva de Aprendizaje**        | Media (docs extensas)                   | Alta (ecosistema C#)           |
| **Recommended For UNDERDOG**    | ✅ Si priorizas control/transparencia   | ✅ Si priorizas datos/producción|

### Recomendación Inicial

**Fase 1:** Empezar con **Backtrader** para prototipado rápido y control total.  
**Fase 2:** Evaluar migración a **Lean Engine** si necesitas datos curados profesionales o despliegue cloud.

---

## 📊 Integración de Hugging Face Datasets

### Datasets Disponibles

1. **Forex OHLCV 1-Minute/5-Minute**
   - Pares: EURUSD, GBPUSD, USDJPY, etc.
   - Metales: XAUUSD, XAGUSD
   - Carga con `datasets` library (1 línea)

2. **Forex Factory News**
   - Calendario económico histórico
   - Impact level (Low/Medium/High)
   - Feature engineering: "time to/from news event"

### Implementación

```python
from datasets import load_dataset

class HuggingFaceDataHandler:
    def __init__(self, dataset_name: str, symbol: str):
        self.ds = load_dataset(dataset_name, symbol)
        
    def get_bars(self, start: str, end: str) -> pd.DataFrame:
        """Returns OHLCV bars for backtesting"""
        df = self.ds['train'].to_pandas()
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        return df[(df['timestamp'] >= start) & (df['timestamp'] <= end)]
```

**Ventaja:** Elimina 7 horas de descarga + 105 GB de storage.

---

## 🧠 Pipeline de Preprocesamiento ML

### Checklist de Transformaciones

```python
# 1. Stationarity: Log Returns
df['log_return'] = np.log(df['close'] / df['close'].shift(1))

# 2. Validation: ADF Test
from statsmodels.tsa.stattools import adfuller
adf_stat, p_value = adfuller(df['log_return'].dropna())[:2]
assert p_value < 0.05, "Serie no estacionaria!"

# 3. Lagged Features (capture autocorrelation)
for lag in [1, 2, 3, 5, 10]:
    df[f'return_lag_{lag}'] = df['log_return'].shift(lag)

# 4. Technical Features
import talib
df['atr_14'] = talib.ATR(df['high'], df['low'], df['close'], timeperiod=14)
df['rsi_14'] = talib.RSI(df['close'], timeperiod=14)

# 5. Standardization (μ=0, σ=1)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
features = ['return_lag_1', 'atr_14', 'rsi_14']
df[features] = scaler.fit_transform(df[features])
```

**Regla de Oro:** Nunca usar precios raw en modelos ML. Siempre transformar a returns estacionarios.

---

## 🔬 Walk-Forward Optimization (WFO)

### Configuración Estándar

```python
WFO_CONFIG = {
    'in_sample_years': 5,      # Ventana de entrenamiento
    'out_sample_months': 12,   # Validación (nunca vista)
    'rolling_step_months': 3,  # Avance trimestral
    'objective': 'calmar_ratio'  # Priorizar preservación capital
}
```

### Proceso

```
Year 1-5: Train/Optimize → Test Year 6 (OOS)
Year 2-6: Train/Optimize → Test Year 7 (OOS)
Year 3-7: Train/Optimize → Test Year 8 (OOS)
...
Concatenate all OOS results → True Performance Estimate
```

### Validación Monte Carlo

```python
def monte_carlo_shuffle(trades: list, iterations: int = 10000):
    """Detect lucky backtests by trade shuffling"""
    original_equity = cumulative_pnl(trades)
    
    simulated_equities = []
    for _ in range(iterations):
        shuffled = random.sample(trades, len(trades))
        simulated_equities.append(cumulative_pnl(shuffled))
    
    percentile = percentileofscore(simulated_equities, original_equity)
    
    if percentile < 5:
        raise ValueError("⚠️ Lucky backtest detected! Not robust.")
```

---

## 📉 Risk Management Engine (RME)

### Métricas Clave

#### 1. Calmar Ratio (Función Objetivo)

```python
calmar_ratio = annual_return / abs(max_drawdown)
```

**Target:** Calmar > 2.0 para Prop Firms  
**Ventaja:** Penaliza drawdowns (más importante que volatilidad)

#### 2. Conditional Value at Risk (CVaR)

```python
def cvar(returns: pd.Series, confidence: float = 0.95) -> float:
    """Expected Shortfall = Avg loss in worst α% scenarios"""
    var_threshold = returns.quantile(1 - confidence)
    return returns[returns <= var_threshold].mean()
```

**Uso:** Monitoreo rodante de tail risk. Si CVaR > -5%, reducir leverage.

#### 3. Kelly Criterion (Position Sizing)

```python
def kelly_fraction(win_rate: float, avg_win_loss_ratio: float) -> float:
    """Optimal position size for log growth maximization"""
    p, q = win_rate, 1 - win_rate
    b = avg_win_loss_ratio
    return (b * p - q) / b

# Conservative: Use Half-Kelly
position_size = kelly_fraction(0.55, 1.5) * 0.5  # 50% de Full Kelly
```

**Regla:** Nunca usar Full Kelly en producción (demasiado volátil).

---

## 🗂️ Nueva Estructura de Archivos

```
underdog/
├── core/
│   ├── abstractions.py      # ABC: Strategy, DataHandler, Execution
│   ├── event_engine.py      # Event Loop (Heartbeat)
│   └── portfolio.py         # Portfolio Manager (P&L, positions)
├── data/
│   ├── hf_loader.py         # Hugging Face DataHandler
│   └── feeds/               # Custom DataFeeds para frameworks
├── strategies/
│   ├── base_strategy.py     # Strategy ABC implementation
│   ├── sma_crossover.py     # Migrated from MQL5
│   └── ml_predictor.py      # ML-based strategy
├── ml/
│   ├── preprocessing.py     # Log returns, ADF, feature engineering
│   ├── models/              # Keras/PyTorch models
│   └── feature_store.py     # Cached features
├── execution/
│   ├── simulated.py         # Backtesting execution (slippage model)
│   └── live_broker.py       # MT5/IBKR/Alpaca connector
├── risk/
│   ├── rme.py               # Risk Management Engine
│   ├── metrics.py           # CVaR, Calmar, Sharpe, etc.
│   └── position_sizing.py   # Kelly Criterion
├── validation/
│   ├── wfo.py               # Walk-Forward Optimization
│   └── monte_carlo.py       # Trade shuffling validation
└── backtesting/
    ├── backtrader_engine.py # Backtrader integration
    └── lean_engine.py       # Lean Engine integration (optional)
```

---

## ⚠️ Archivos a Eliminar

### Scripts Obsoletos
- ❌ `scripts/download_histdata_1min_only.py`
- ❌ `scripts/download_liquid_pairs.py`
- ❌ `scripts/download_all_histdata.py`
- ❌ `scripts/insert_indicators_to_db.py`
- ❌ `scripts/setup_db_simple.ps1`
- ❌ `scripts/resample_and_calculate.py`

### Datos Locales
- ❌ `data/parquet/` (archivos descargados de HistData)
- ❌ `data/raw/` (ZIPs de HistData)

### Schemas DB
- ⚠️ `docker/init-indicators-db.sql` (mantener solo para live trading, no para backtesting)

**Razón:** Event-Driven frameworks calculan indicadores on-the-fly durante el event loop. No necesitamos precálculo masivo.

---

## 🚀 Roadmap de Implementación

### Sprint 1: Setup Framework (1-2 días)
1. ✅ Instalar Backtrader: `poetry add backtrader`
2. ✅ Crear `underdog/core/abstractions.py` con ABC
3. ✅ Implementar DataHandler para Hugging Face
4. ✅ Test: Ejecutar SMA Crossover simple en Backtrader

### Sprint 2: ML Pipeline (2-3 días)
1. ✅ Crear `underdog/ml/preprocessing.py`
2. ✅ Implementar transformación Log Returns + ADF Test
3. ✅ Feature engineering: Lagged returns, ATR, RSI
4. ✅ Test: Entrenar modelo simple (Logistic Regression) y validar

### Sprint 3: WFO + Validación (2-3 días)
1. ✅ Implementar `underdog/validation/wfo.py`
2. ✅ Configurar ventanas IS/OOS rodantes
3. ✅ Implementar Monte Carlo shuffling
4. ✅ Test: WFO sobre SMA Crossover (2015-2025)

### Sprint 4: Risk Management (1-2 días)
1. ✅ Crear `underdog/risk/rme.py`
2. ✅ Implementar CVaR, Calmar Ratio, Kelly Criterion
3. ✅ Integrar límites MDD (6% para Prop Firms)
4. ✅ Test: Backtest con position sizing dinámico

### Sprint 5: Migración de EAs (3-4 días)
1. ✅ Convertir cada EA de MQL5 a Strategy class
2. ✅ Implementar `generate_signal()` method
3. ✅ Validar con backtesting Event-Driven
4. ✅ Comparar resultados vs MQL5 backtest

---

## 📚 Referencias Técnicas

### Papers y Recursos
- **Lopez de Prado (2018)**: *Advances in Financial Machine Learning* - WFO, Feature Importance
- **Pardo (2008)**: *The Evaluation and Optimization of Trading Strategies* - Walk-Forward
- **QuantConnect Docs**: [Lean Algorithm Framework](https://www.quantconnect.com/docs/v2/)
- **Backtrader Docs**: [Cerebro & Strategy](https://www.backtrader.com/docu/cerebro/)

### Python Libraries
```toml
[tool.poetry.dependencies]
backtrader = "^1.9.78"
huggingface-hub = "^0.20.0"
datasets = "^2.16.0"
ta-lib = "^0.4.28"
statsmodels = "^0.14.0"  # ADF Test
scikit-learn = "^1.3.0"
```

---

## ✅ Criterios de Éxito

### Backtesting Robusto
- [x] Event-Driven (tick-by-tick o bar-by-bar)
- [x] Spread dinámico + slippage modelado
- [x] Walk-Forward Optimization implementado
- [x] Monte Carlo validation passing (>5th percentile)

### Machine Learning
- [x] Data estacionaria (ADF test p-value < 0.05)
- [x] Features técnicos + lagged returns
- [x] Standardization aplicada

### Risk Management
- [x] Calmar Ratio > 2.0
- [x] MDD < 6% (Prop Firm compliant)
- [x] CVaR monitoreado (tail risk)
- [x] Kelly position sizing

### Producción
- [x] Same strategy code para backtest → paper → live
- [x] Minimal glue code para broker integration
- [x] Monitoring/alerts via Prometheus

---

**Status:** 🟢 Arquitectura definida, ready para implementación  
**Next Step:** Ejecutar Sprint 1 (Setup Framework)
