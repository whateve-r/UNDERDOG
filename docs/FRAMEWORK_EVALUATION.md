# 🔬 Evaluación de Frameworks: Backtrader vs Lean Engine

## 📋 Objetivo

Decidir el framework de backtesting óptimo para UNDERDOG basado en pruebas prácticas con un EA simple (SMA Crossover).

---

## 🏗️ Criterios de Evaluación

| **Criterio**                  | **Peso** | **Backtrader** | **Lean Engine** |
|-------------------------------|----------|----------------|-----------------|
| Event-Driven Architecture     | ⭐⭐⭐     | ?              | ?               |
| Modelado de Spread/Slippage   | ⭐⭐⭐     | ?              | ?               |
| Integración de Datos Forex    | ⭐⭐⭐     | ?              | ?               |
| Facilidad de Integración ML   | ⭐⭐⭐     | ?              | ?               |
| Control Local (No Cloud)      | ⭐⭐       | ?              | ?               |
| Curva de Aprendizaje          | ⭐⭐       | ?              | ?               |
| Path to Production (Live)     | ⭐⭐       | ?              | ?               |
| Documentación y Comunidad     | ⭐         | ?              | ?               |

---

## 🧪 Test 1: Instalación y Setup

### Backtrader

```bash
# Instalación
poetry add backtrader

# Test de importación
python -c "import backtrader as bt; print(bt.__version__)"
```

**Resultado:**
- ✅/❌ Instalación exitosa
- ✅/❌ Compatible con Python 3.13
- Tiempo: ___ minutos
- Problemas encontrados: ___

### Lean Engine

```bash
# Instalación (Lean CLI)
dotnet tool install -g QuantConnect.Lean.Cli
lean cloud login

# O instalación local
git clone https://github.com/QuantConnect/Lean.git
cd Lean
pip install -r requirements.txt
```

**Resultado:**
- ✅/❌ Instalación exitosa
- ✅/❌ Compatible con entorno Windows/WSL
- Tiempo: ___ minutos
- Problemas encontrados: ___

---

## 🧪 Test 2: SMA Crossover Simple (Sin Microestructura)

### Objetivo
Implementar estrategia básica de cruce de medias móviles (SMA 10/50) y ejecutar backtest sobre EURUSD 2020-2024.

### Backtrader Implementation

```python
import backtrader as bt
import pandas as pd

class SMACrossover(bt.Strategy):
    params = (
        ('fast', 10),
        ('slow', 50),
    )
    
    def __init__(self):
        self.sma_fast = bt.indicators.SMA(self.data.close, period=self.params.fast)
        self.sma_slow = bt.indicators.SMA(self.data.close, period=self.params.slow)
        self.crossover = bt.indicators.CrossOver(self.sma_fast, self.sma_slow)
    
    def next(self):
        if not self.position:
            if self.crossover > 0:  # Fast crosses above slow
                self.buy()
        else:
            if self.crossover < 0:  # Fast crosses below slow
                self.sell()

# Setup
cerebro = bt.Cerebro()
cerebro.addstrategy(SMACrossover)

# Load data (CSV/Parquet)
data = bt.feeds.PandasData(dataname=pd.read_parquet('eurusd_5m.parquet'))
cerebro.adddata(data)

# Run
cerebro.broker.setcash(10000.0)
cerebro.broker.setcommission(commission=0.001)  # 0.1%
results = cerebro.run()

# Results
print(f"Final Portfolio Value: {cerebro.broker.getvalue():.2f}")
```

**Resultado:**
- ✅/❌ Código funciona sin errores
- Return: ___% 
- Max Drawdown: ___%
- Trades: ___
- Tiempo de ejecución: ___ segundos
- Observaciones: ___

### Lean Engine Implementation

```python
from AlgorithmImports import *

class SMACrossoverAlgorithm(QCAlgorithm):
    def Initialize(self):
        self.SetStartDate(2020, 1, 1)
        self.SetEndDate(2024, 12, 31)
        self.SetCash(10000)
        
        # Add Forex pair
        self.eurusd = self.AddForex("EURUSD", Resolution.Minute).Symbol
        
        # Indicators
        self.fast = self.SMA(self.eurusd, 10)
        self.slow = self.SMA(self.eurusd, 50)
    
    def OnData(self, data):
        if not self.fast.IsReady or not self.slow.IsReady:
            return
        
        if not self.Portfolio.Invested:
            if self.fast.Current.Value > self.slow.Current.Value:
                self.SetHoldings(self.eurusd, 1.0)
        else:
            if self.fast.Current.Value < self.slow.Current.Value:
                self.Liquidate(self.eurusd)
```

**Resultado:**
- ✅/❌ Código funciona sin errores
- Return: ___% 
- Max Drawdown: ___%
- Trades: ___
- Tiempo de ejecución: ___ segundos
- Observaciones: ___

---

## 🧪 Test 3: Modelado de Microestructura (Spread/Slippage)

### Objetivo
Validar qué framework modela mejor los costos de transacción realistas en Forex.

### Backtrader: Custom Spread Model

```python
class ForexCommission(bt.CommInfoBase):
    params = (
        ('commission', 0.0),  # No commission, only spread
        ('spread', 0.0002),   # 2 pips spread
        ('leverage', 100),
    )
    
    def _getcommission(self, size, price, pseudoexec):
        # Commission is embedded in spread
        return abs(size) * self.p.spread * price

# Apply to cerebro
cerebro.broker.addcommissioninfo(ForexCommission())

# Add slippage
cerebro.broker.set_slippage_perc(0.0001)  # 0.01% slippage
```

**Test:** Re-run SMA Crossover con spread de 2 pips + slippage 0.01%

**Resultado:**
- Return con costos: ___%
- Diferencia vs sin costos: ___%
- Trades rentables: ___/__
- Observaciones: ___

### Lean Engine: Built-in Spread Model

```python
class SMACrossoverWithCosts(QCAlgorithm):
    def Initialize(self):
        # ... (same as before)
        
        # Set realistic Forex costs
        self.eurusd = self.AddForex("EURUSD", Resolution.Minute, Market.Oanda)
        self.eurusd.SetLeverage(100)
        
        # Lean automatically uses bid/ask spread from data
        # Can customize slippage model:
        self.SetSlippageModel(ConstantSlippageModel(0.0001))
```

**Resultado:**
- Return con costos: ___%
- Diferencia vs sin costos: ___%
- Trades rentables: ___/___
- Observaciones: ___

**Comparación:**
¿Qué framework modela los costos de forma más realista?
- Backtrader: ___
- Lean: ___

---

## 🧪 Test 4: Integración de Machine Learning

### Objetivo
Evaluar facilidad de integración de un modelo ML (simple Logistic Regression) para predicción direccional.

### Backtrader: ML Strategy

```python
import joblib
from sklearn.linear_model import LogisticRegression

class MLStrategy(bt.Strategy):
    def __init__(self):
        self.model = joblib.load('lr_model.pkl')
        self.lookback = 20
    
    def next(self):
        if len(self.data) < self.lookback:
            return
        
        # Feature engineering (manual)
        closes = [self.data.close[-i] for i in range(self.lookback)]
        returns = [(closes[i] - closes[i+1]) / closes[i+1] for i in range(len(closes)-1)]
        
        features = np.array(returns).reshape(1, -1)
        prediction = self.model.predict(features)[0]
        
        if prediction == 1 and not self.position:
            self.buy()
        elif prediction == 0 and self.position:
            self.sell()
```

**Resultado:**
- ✅/❌ Integración exitosa
- Código adicional requerido: ___ líneas
- Complejidad: Baja/Media/Alta
- Observaciones: ___

### Lean Engine: Research + Algorithm

```python
# In Research Notebook (Jupyter)
from sklearn.linear_model import LogisticRegression
import joblib

# Train model
qb = QuantBook()
history = qb.History(qb.Securities.Keys, 365, Resolution.Daily)
# ... feature engineering, training ...
model = LogisticRegression()
model.fit(X_train, y_train)
joblib.dump(model, 'models/lr_model.pkl')

# In Algorithm
class MLAlgorithm(QCAlgorithm):
    def Initialize(self):
        # ... setup ...
        self.model = joblib.load('models/lr_model.pkl')
        self.SetWarmUp(20)
    
    def OnData(self, data):
        if self.IsWarmingUp:
            return
        
        # Feature engineering (use QuantConnect indicators)
        features = self.CalculateFeatures()
        prediction = self.model.predict(features)[0]
        
        if prediction == 1:
            self.SetHoldings(self.eurusd, 1.0)
        else:
            self.Liquidate()
```

**Resultado:**
- ✅/❌ Integración exitosa
- Código adicional requerido: ___ líneas
- Complejidad: Baja/Media/Alta
- Observaciones: ___

---

## 🧪 Test 5: Walk-Forward Optimization

### Objetivo
Implementar WFO básico (3 ventanas: 2020-2022 train, 2023 test, 2024 test).

### Backtrader: Manual WFO

```python
def walk_forward_optimization(
    strategy_class,
    data_df: pd.DataFrame,
    train_years: int = 3,
    test_months: int = 12
):
    """Manual WFO implementation"""
    results_oos = []
    
    start_year = 2020
    end_year = 2024
    
    for year in range(start_year + train_years, end_year + 1):
        # In-sample
        train_start = year - train_years
        train_end = year - 1
        train_data = data_df[(data_df.index.year >= train_start) & 
                              (data_df.index.year <= train_end)]
        
        # Optimize parameters on train_data
        best_params = optimize_strategy(strategy_class, train_data)
        
        # Out-of-sample
        test_data = data_df[data_df.index.year == year]
        result = backtest_strategy(strategy_class, test_data, best_params)
        results_oos.append(result)
    
    return pd.DataFrame(results_oos)
```

**Resultado:**
- ✅/❌ WFO implementado
- Líneas de código: ___
- Complejidad: Baja/Media/Alta
- Observaciones: ___

### Lean Engine: Built-in Optimization

```python
# Lean CLI
lean optimize --strategy-name SMACrossover \
    --parameter fast 5-20 \
    --parameter slow 30-100 \
    --target "Sharpe Ratio" \
    --target-direction max
```

**Resultado:**
- ✅/❌ Optimización funcional
- WFO nativa: ✅/❌
- Líneas de código: ___
- Complejidad: Baja/Media/Alta
- Observaciones: ___

---

## 🧪 Test 6: Datos de Hugging Face

### Objetivo
Cargar dataset de Forex desde Hugging Face y usarlo en el framework.

### Backtrader: Custom DataFeed

```python
from datasets import load_dataset

# Load from HF
ds = load_dataset('financial_datasets/forex_ohlcv', 'EURUSD')
df = ds['train'].to_pandas()

# Convert to Backtrader format
data = bt.feeds.PandasData(
    dataname=df,
    datetime='timestamp',
    open='open',
    high='high',
    low='low',
    close='close',
    volume='volume',
    openinterest=-1
)

cerebro.adddata(data)
```

**Resultado:**
- ✅/❌ Integración exitosa
- Código requerido: ___ líneas
- Problemas: ___

### Lean Engine: Custom Data Source

```python
class HuggingFaceData(PythonData):
    def GetSource(self, config, date, isLiveMode):
        from datasets import load_dataset
        ds = load_dataset('financial_datasets/forex_ohlcv', 'EURUSD')
        return SubscriptionDataSource(ds, SubscriptionTransportMedium.RemoteFile)
    
    def Reader(self, config, line, date, isLiveMode):
        # Parse HF data format
        bar = HuggingFaceData()
        # ... populate bar ...
        return bar

# In Algorithm
self.AddData(HuggingFaceData, "EURUSD", Resolution.Minute)
```

**Resultado:**
- ✅/❌ Integración exitosa
- Código requerido: ___ líneas
- Problemas: ___

---

## 📊 Matriz de Decisión Final

| **Criterio**                  | **Backtrader** | **Lean Engine** | **Ganador**     |
|-------------------------------|----------------|-----------------|-----------------|
| Event-Driven                  | ___/10         | ___/10          | ___             |
| Spread/Slippage               | ___/10         | ___/10          | ___             |
| Datos Forex                   | ___/10         | ___/10          | ___             |
| Integración ML                | ___/10         | ___/10          | ___             |
| Control Local                 | ___/10         | ___/10          | ___             |
| Curva Aprendizaje             | ___/10         | ___/10          | ___             |
| Path to Production            | ___/10         | ___/10          | ___             |
| Docs/Comunidad                | ___/10         | ___/10          | ___             |
| **TOTAL**                     | **___/80**     | **___/80**      | **___**         |

---

## 🎯 Recomendación Final

**Framework seleccionado:** ___

**Justificación:**
1. ___
2. ___
3. ___

**Trade-offs aceptados:**
- ___
- ___

**Próximos pasos:**
1. Instalar `poetry add ___`
2. Crear `underdog/backtesting/____engine.py`
3. Migrar primer EA (SMA Crossover) a nueva arquitectura
4. Validar resultados vs MQL5 backtest

---

**Status:** 🟡 En evaluación  
**Owner:** @user  
**Deadline:** ___
