# Arquitectura DRL para UNDERDOG - Prop Firm Trading

## 📋 Executive Summary

Basado en el análisis de papers científicos (arXiv:2510.04952v2, arXiv:2510.10526v1, arXiv:2510.03236v1), implementaremos una arquitectura de **Deep Reinforcement Learning (DRL)** con gestión de riesgo explícita para cumplir con restricciones de Prop Firms (FTMO).

**Objetivo:** Maximizar retorno ajustado al riesgo mientras **garantizamos** cumplimiento de límites (Daily DD <5%, Total DD <10%).

---

## 🏗️ Arquitectura de 4 Capas

```
┌─────────────────────────────────────────────────────────────┐
│ CAPA 1: INGESTA DE DATOS                                    │
│ ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐    │
│ │  OHLCV   │  │Technical │  │ LLM/NLP  │  │  FRED    │    │
│ │  MT5/    │  │Indicators│  │Sentiment │  │  Macro   │    │
│ │yfinance  │  │ (TA-Lib) │  │ (FinGPT) │  │Indicators│    │
│ └──────────┘  └──────────┘  └──────────┘  └──────────┘    │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ CAPA 2: REGIME-SWITCHING (Clasificación de Mercado)         │
│ ┌─────────────────────────────────────────────────────┐    │
│ │ GMM + XGBoost Classifier                            │    │
│ │ Output: Régimen actual (Trend/Range/Transition)    │    │
│ │ Features: Volatilidad, Volumen, Indicadores        │    │
│ └─────────────────────────────────────────────────────┘    │
│ Paper: arXiv:2510.03236v1                                  │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ CAPA 3: DRL AGENT (Decisión de Trading)                     │
│ ┌─────────────────────────────────────────────────────┐    │
│ │ TD3 / PPO Agent (stable-baselines3)                 │    │
│ │ State: [Price, Indicators, Sentiment, Regime]       │    │
│ │ Action: [Position_Size, Entry/Exit]                 │    │
│ │ Reward: Sharpe Ratio (risk-adjusted return)        │    │
│ └─────────────────────────────────────────────────────┘    │
│ Paper: arXiv:2510.10526v1 (LLM + RL Integration)          │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ CAPA 4: CONSTRAINED EXECUTION (Shield Module)               │
│ ┌─────────────────────────────────────────────────────┐    │
│ │ Safety Shield (Pre-execution Validation)            │    │
│ │ Checks: Daily DD <5%, Total DD <10%, Max Positions  │    │
│ │ Action: Project unsafe actions → safe actions       │    │
│ └─────────────────────────────────────────────────────┘    │
│ Paper: arXiv:2510.04952v2 (Safe Trade Execution)          │
└─────────────────────────────────────────────────────────────┘
                            ↓
                    [MT5 Broker Execution]
```

---

## 📦 Componentes Python

### 1. Regime-Switching Module

**Path:** `underdog/ml/regime_classifier.py`

```python
# Clasificador de Régimen de Mercado
# Paper: arXiv:2510.03236v1

from sklearn.mixture import GaussianMixture
from xgboost import XGBClassifier

class RegimeSwitchingModel:
    """
    Classifies market regime: Trend/Range/Transition
    
    Features:
    - ATR (volatility)
    - Volume normalized
    - ADX (trend strength)
    - Bollinger Band width
    """
    
    def __init__(self, n_regimes=3):
        self.gmm = GaussianMixture(n_components=n_regimes)
        self.classifier = XGBClassifier()
    
    def fit(self, features: np.ndarray):
        # GMM para clustering suave
        # XGBoost para clasificación final
        pass
    
    def predict(self, features: np.ndarray) -> str:
        # Returns: 'trend' / 'range' / 'transition'
        pass
```

**Uso:**
- Entrena offline con 5 años de datos históricos
- Inference en tiempo real (cada tick/barra)
- Output usado como feature para DRL Agent

---

### 2. DRL Agent (TD3 / PPO)

**Path:** `underdog/ml/drl_agent.py`

```python
# Deep Reinforcement Learning Agent
# Paper: arXiv:2510.10526v1 (LLM + RL)

from stable_baselines3 import TD3, PPO
from stable_baselines3.common.vec_env import DummyVecEnv

class TradingDRLAgent:
    """
    TD3 Agent for adaptive signal integration
    
    State Space:
    - Price features (OHLC, indicators)
    - LLM sentiment score (-1 to 1)
    - Regime classification (one-hot)
    - Position state (size, PnL, duration)
    
    Action Space:
    - Continuous: position_size (-1 to 1)
    - Discrete: entry/exit decision
    
    Reward:
    - Sharpe Ratio (rolling 30 periods)
    - Penalized by drawdown violations
    """
    
    def __init__(self, env, algorithm='td3'):
        if algorithm == 'td3':
            self.model = TD3('MlpPolicy', env)
        elif algorithm == 'ppo':
            self.model = PPO('MlpPolicy', env)
    
    def train(self, total_timesteps=100_000):
        self.model.learn(total_timesteps=total_timesteps)
    
    def predict(self, state):
        action, _states = self.model.predict(state)
        return action
```

**Librerías:**
- `stable-baselines3` (PPO, TD3)
- `gymnasium` (trading environment)

---

### 3. Safety Shield (Constrained RL)

**Path:** `underdog/execution/safety_shield.py`

```python
# Safety Shield for Prop Firm Compliance
# Paper: arXiv:2510.04952v2 (Safe Execution)

class PropFirmSafetyShield:
    """
    Pre-execution validation layer
    
    Enforces:
    - Daily Drawdown < 5% (FTMO Phase 1)
    - Total Drawdown < 10%
    - Max open positions <= 2
    - Min holding time >= 3 minutes
    
    If action violates constraint:
    → Project to nearest safe action
    → Log violation attempt
    → Alert monitoring
    """
    
    def __init__(self, config: dict):
        self.max_daily_dd = config.get('max_daily_dd', 0.05)
        self.max_total_dd = config.get('max_total_dd', 0.10)
        self.max_positions = config.get('max_positions', 2)
    
    def validate_action(self, action: dict, account_state: dict) -> tuple[bool, dict]:
        """
        Returns: (is_safe, corrected_action)
        """
        # Check Daily DD
        if account_state['daily_dd_pct'] >= self.max_daily_dd:
            return False, {'action': 'close_all', 'reason': 'daily_dd_breach'}
        
        # Check Total DD
        if account_state['total_dd_pct'] >= self.max_total_dd:
            return False, {'action': 'close_all', 'reason': 'total_dd_breach'}
        
        # Check Max Positions
        if len(account_state['open_positions']) >= self.max_positions:
            if action['type'] == 'open':
                return False, {'action': 'wait', 'reason': 'max_positions'}
        
        # Reduce position size if necessary
        max_risk_per_trade = 0.015  # 1.5%
        if action.get('lot_size', 0) * action.get('risk_pct', 0) > max_risk_per_trade:
            action['lot_size'] = max_risk_per_trade / action['risk_pct']
        
        return True, action
```

**Integración con Mt5Connector:**
```python
# En underdog/execution/mt5_executor.py
shield = PropFirmSafetyShield(config)

def execute_order(self, order: dict):
    is_safe, corrected_order = shield.validate_action(order, self.get_account_state())
    
    if not is_safe:
        logger.warning(f"Shield blocked order: {corrected_order['reason']}")
        return corrected_order
    
    # Proceed with execution
    return self.connector.place_order(corrected_order)
```

---

### 4. LLM Sentiment Integration

**Path:** `underdog/ml/sentiment_analyzer.py`

```python
# LLM Sentiment Analysis
# Paper: arXiv:2510.10526v1

from transformers import pipeline

class FinGPTSentimentAnalyzer:
    """
    Fine-tuned FinGPT for Forex sentiment
    
    Sources:
    - Reddit (r/forex, r/wallstreetbets)
    - Twitter/X financial accounts
    - News APIs (Alpha Vantage, NewsAPI)
    
    Output: Sentiment score [-1, 1]
    """
    
    def __init__(self, model_name='ProsusAI/finbert'):
        self.sentiment_pipeline = pipeline(
            "sentiment-analysis",
            model=model_name,
            device=0  # GPU
        )
    
    def analyze(self, text: str) -> float:
        """
        Returns: score in [-1, 1]
        -1 = bearish, 0 = neutral, 1 = bullish
        """
        result = self.sentiment_pipeline(text)[0]
        
        # Map to [-1, 1]
        if result['label'] == 'positive':
            return result['score']
        elif result['label'] == 'negative':
            return -result['score']
        else:
            return 0.0
    
    def get_market_sentiment(self, symbol: str, window_hours=24) -> float:
        """
        Aggregate sentiment from multiple sources
        """
        # Fetch recent news/reddit posts
        # Average sentiment scores
        pass
```

---

## 🔄 Workflow Completo

### Training Phase (Offline)

```python
# scripts/train_drl_agent.py

from underdog.ml.regime_classifier import RegimeSwitchingModel
from underdog.ml.drl_agent import TradingDRLAgent
from underdog.ml.sentiment_analyzer import FinGPTSentimentAnalyzer

# 1. Prepare historical data (5 years)
df = load_historical_data('EURUSD', '2020-01-01', '2025-01-01')

# 2. Train Regime Classifier
regime_model = RegimeSwitchingModel(n_regimes=3)
regime_model.fit(df[['atr', 'volume', 'adx', 'bb_width']])
regime_model.save('models/regime_classifier.pkl')

# 3. Create Trading Environment
env = TradingEnvironment(
    data=df,
    regime_model=regime_model,
    sentiment_analyzer=FinGPTSentimentAnalyzer(),
    safety_shield=PropFirmSafetyShield(config)
)

# 4. Train DRL Agent
agent = TradingDRLAgent(env, algorithm='td3')
agent.train(total_timesteps=500_000)  # ~1 semana GPU
agent.save('models/drl_agent_td3.zip')

# 5. Backtest with Shield
backtest_results = backtest_with_shield(agent, env, test_data)
print(f"Sharpe: {backtest_results['sharpe']}")
print(f"Max DD: {backtest_results['max_dd']}")
print(f"Shield Interventions: {backtest_results['shield_blocks']}")
```

### Production Phase (Live)

```python
# scripts/start_live_drl_trading.py

from underdog.execution.mt5_executor import Mt5Executor
from underdog.ml.drl_agent import TradingDRLAgent
from underdog.execution.safety_shield import PropFirmSafetyShield

# Load trained models
agent = TradingDRLAgent.load('models/drl_agent_td3.zip')
regime_model = RegimeSwitchingModel.load('models/regime_classifier.pkl')
shield = PropFirmSafetyShield(config)

# Initialize executor
executor = Mt5Executor(shield=shield)

# Main loop
while True:
    # 1. Get current state
    market_data = executor.get_market_data('EURUSD')
    sentiment = sentiment_analyzer.get_market_sentiment('EURUSD')
    regime = regime_model.predict(market_data)
    
    state = create_state_vector(market_data, sentiment, regime)
    
    # 2. DRL Agent decision
    action = agent.predict(state)
    
    # 3. Shield validation
    is_safe, corrected_action = shield.validate_action(
        action, 
        executor.get_account_state()
    )
    
    # 4. Execute (if safe)
    if is_safe:
        executor.execute_order(corrected_action)
    else:
        logger.warning(f"Shield blocked: {corrected_action['reason']}")
    
    time.sleep(60)  # Check every minute
```

---

## 📊 Estructura de Carpetas Actualizada

```
underdog/
├── ml/                          # NEW: Machine Learning components
│   ├── __init__.py
│   ├── regime_classifier.py     # Regime-Switching (GMM+XGBoost)
│   ├── drl_agent.py             # TD3/PPO Agent (stable-baselines3)
│   ├── sentiment_analyzer.py    # FinGPT/FinBERT Sentiment
│   ├── trading_env.py           # Gymnasium Trading Environment
│   └── feature_engineering.py   # Feature extraction
│
├── execution/
│   ├── safety_shield.py         # NEW: Constrained RL Shield
│   ├── mt5_executor.py          # UPDATED: Integrate Shield
│   └── recovery.py
│
├── risk/                        # CONSOLIDATED
│   ├── prop_firm_rme.py         # FTMO limits
│   ├── position_sizing.py
│   └── drawdown_monitor.py
│
├── data/
│   ├── mt5_historical_loader.py
│   ├── reddit_scraper.py        # NEW: Reddit sentiment
│   └── news_aggregator.py       # NEW: News + Twitter
│
└── backtesting/
    ├── bt_engine.py
    └── drl_backtester.py        # NEW: Backtest DRL agents

scripts/
├── train_drl_agent.py           # NEW: Offline training
├── start_live_drl_trading.py    # NEW: Live DRL trading
└── backtest_drl_strategy.py     # NEW: DRL backtest

models/                          # NEW: Trained models
├── regime_classifier.pkl
├── drl_agent_td3.zip
└── sentiment_analyzer/
```

---

## 🚀 Plan de Implementación (3 Semanas)

### Semana 1: Foundation
- [ ] Setup `underdog/ml/` estructura
- [ ] Implementar `RegimeSwitchingModel` (GMM + XGBoost)
- [ ] Crear `TradingEnvironment` (Gymnasium)
- [ ] Configurar `stable-baselines3` + GPU

### Semana 2: DRL Core
- [ ] Implementar `TradingDRLAgent` (TD3)
- [ ] Integrar `FinGPTSentimentAnalyzer`
- [ ] Train offline con 5 años datos (GPU)
- [ ] Backtest con métricas Sharpe/DD

### Semana 3: Safety + Production
- [ ] Implementar `PropFirmSafetyShield`
- [ ] Integrar Shield en `Mt5Executor`
- [ ] Tests end-to-end con DEMO MT5
- [ ] Monitoring + alertas

---

## 📈 Métricas de Éxito

### Training Phase
- **Sharpe Ratio > 1.5** (backtest 2024)
- **Max Drawdown < 8%** (sin shield)
- **Shield Blocks < 5%** (baja interferencia)

### Paper Trading (30 días)
- **Daily DD never > 4.5%** (margen de seguridad)
- **Total DD never > 9%**
- **Uptime > 99.9%**
- **Sharpe Ratio > 1.2** (live)

### FTMO Phase 1 (30 días)
- **Profit > 8%** (objetivo FTMO)
- **Daily DD < 5%** (hard limit)
- **Total DD < 10%** (hard limit)
- **Win Rate > 50%**

---

## 🔬 Papers de Referencia

1. **arXiv:2510.04952v2** - Safe and Compliant Trade Execution
   - Constrained RL con Shield Module
   - PPO + Safety Constraints
   
2. **arXiv:2510.10526v1** - Integrating LLM and RL for Trading
   - TD3 para adaptive signal weighting
   - FinGPT sentiment integration
   
3. **arXiv:2510.03236v1** - Regime-Switching Methods
   - GMM + XGBoost para clasificación de régimen
   - Feature engineering para volatilidad

---

## 🐍 Dependencias Nuevas

```toml
# pyproject.toml

[project.dependencies]
# DRL
stable-baselines3 = "^2.2.0"
gymnasium = "^0.29.0"
tensorboard = "^2.15.0"

# ML
xgboost = "^2.0.0"
scikit-learn = "^1.3.0"
hmmlearn = "^0.3.0"

# NLP/LLM
transformers = "^4.35.0"
torch = "^2.1.0"  # GPU support
sentencepiece = "^0.1.99"

# Reddit/News
praw = "^7.7.0"  # Reddit API
tweepy = "^4.14.0"  # Twitter API
newsapi-python = "^0.2.7"
```

---

## ⚠️ Consideraciones Críticas

### GPU Requirements
- **Training:** RTX 3060+ (12GB VRAM) o cloud GPU
- **Inference:** CPU suficiente para live trading
- **Estimación:** 1 semana training offline (500k timesteps)

### Complejidad vs. Simplicidad
- **Fase 1 (Actual):** Usar DRL solo para **position sizing** + **regime detection**
- **Fase 2 (Futuro):** DRL full control (entry/exit signals)
- **Razón:** Reducir superficie de error durante FTMO

### Fallback Strategy
- Si DRL falla en paper trading → **Fallback a estrategias rule-based actuales**
- Shield es **mandatorio** independiente de DRL
- Monitoreo 24/7 con alertas Telegram

---

## 📞 Próximos Pasos Inmediatos

1. **AHORA:** ¿Aprobamos esta arquitectura?
2. **Hoy:** Crear `underdog/ml/` estructura vacía
3. **Mañana:** Implementar `safety_shield.py` (crítico para Prop Firm)
4. **Esta semana:** Integrar Shield en `Mt5Executor`
5. **Semana próxima:** Comenzar training DRL offline

---

**Última actualización:** 2025-01-22  
**Autor:** UNDERDOG Team  
**Status:** PROPUESTA - Pending Approval
