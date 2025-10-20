# UNDERDOG - Algorithmic Trading System

**A sophisticated, modular algorithmic trading platform designed for quantitative analysis, automated trading, and Prop Firm compliance.**

[![Python 3.13](https://img.shields.io/badge/python-3.13-blue.svg)](https://www.python.org/downloads/)
[![Poetry](https://img.shields.io/badge/dependency-Poetry-blue)](https://python-poetry.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Phase 2: 62.5%](https://img.shields.io/badge/Phase%202-62.5%25-yellow.svg)](docs/PHASE2_COMPLETION_GUIDE.md)

---

## 🎯 Current Status: Phase 2 - Advanced Modules (5/8 Complete)

**✅ Completed**:
- Walk-Forward Optimization (WFO) - Automated IS/OOS parameter validation
- Monte Carlo Simulation - Robustness testing with VaR/CVaR
- MLflow Integration - Experiment tracking & model registry
- HMM Regime Classifier - Market regime detection with strategy gating
- Feature Engineering Pipeline - Hash-based versioning for reproducibility

**⏳ In Progress**:
- Unit Tests Suite (pytest)
- Monitoring & Telemetry (Prometheus + Grafana)
- Docker Production Setup (docker-compose with TimescaleDB)

👉 **Quick Start**: Run `python scripts/complete_trading_workflow.py` to see full integration demo

---

## 🚀 Features

### Core Capabilities (Phase 1 ✅)
- **Real-Time Market Connectivity**: Seamless integration with MetaTrader 5 via ZeroMQ for low-latency async communication
- **Multi-Strategy Architecture**: Portfolio-level strategy coordination with correlation tracking and dynamic allocation
- **Advanced Risk Management**: Daily/weekly/monthly DD limits, exposure constraints, correlation-based position scaling
- **Intelligent Position Sizing**: Fixed fractional + fractional Kelly + confidence-weighted + DD-scaled adaptive sizing
- **Fuzzy Logic Confidence Scoring**: Transform ML outputs and indicators into actionable confidence scores (Mamdani inference)
- **High-Performance Data Handling**: Efficient time-series management with pandas and vectorized computations with numpy
- **Technical Analysis**: 150+ indicators with TA-Lib integration (RSI, Bollinger, ATR, SuperTrend)

### Advanced Modules (Phase 2 - 62.5% ✅)
- **Walk-Forward Optimization**: Automated IS/OOS segmentation, grid search, anchored/rolling windows
- **Monte Carlo Simulation**: 5000+ simulations with bootstrap resampling, slippage modeling, percentile risk analysis
- **MLflow Integration**: Experiment tracking, model versioning, staging workflow (None → Staging → Production)
- **HMM Regime Detection**: 3-5 state Hidden Markov Models for market regime classification (BULL/BEAR/SIDEWAYS/HIGH_VOL/LOW_VOL)
- **Feature Engineering Pipeline**: Hash-versioned feature sets with 50+ technical/momentum/volatility features, cyclic temporal encoding
- **Statistical Modeling**: Cointegration tests, ADF stationarity validation, Kalman filters for pairs trading
- **ML Training Pipeline**: LSTM, CNN, Random Forest, XGBoost with reproducible seeds and data hashing
- **Strategy Gating**: Regime-based activation logic (trend active in bull/bear, mean-reversion in sideways)

---

## 📁 Project Structure

```
UNDERDOG/
├── config/
│   ├── runtime/env/
│   │   └── mt5_credentials.yaml      # MT5 connection config
│   └── strategies/
│       ├── keltner_breakout.yaml
│       └── fuzzy_confidence_rules.yaml  # Fuzzy logic rules
│
├── underdog/
│   ├── core/
│   │   ├── connectors/
│   │   │   ├── mt5_connector.py       # ZeroMQ MT5 integration
│   │   │   └── trading_bot_example.py
│   │   ├── schemas/
│   │   │   └── zmq_messages.py        # Message schemas & validation
│   │   ├── comms_bus/                  # Async queues, pub/sub
│   │   └── data_handlers/              # OHLCV processing
│   │
│   ├── risk_management/
│   │   ├── risk_master.py             # Portfolio-level risk manager
│   │   ├── position_sizing.py         # Multi-factor position sizing
│   │   └── rules_engine.py
│   │
│   ├── strategies/
│   │   ├── strategy_matrix.py         # Multi-strategy coordinator
│   │   ├── base_strategy.py
│   │   ├── fuzzy_logic/
│   │   │   └── mamdani_inference.py   # Fuzzy confidence scoring
│   │   ├── keltner_breakout/
│   │   ├── pairs_trading/
│   │   │   └── kalman_hedge.py
│   │   └── ml_strategies/
│   │       └── feature_engineering.py
│   │
│   ├── backtesting/
│   │   ├── engines/event_driven.py
│   │   └── validation/
│   │       ├── wfo.py                  # Walk-Forward Optimization
│   │       ├── monte_carlo.py         # Monte Carlo simulation
│   │       └── metrics.py
│   │
│   ├── ml/
│   │   ├── models/regime_classifier.py
│   │   ├── training/train_pipeline.py
│   │   └── evaluation/metrics_ml.py
│   │
│   ├── execution/
│   │   ├── order_manager.py
│   │   └── retry_logic.py
│   │
│   ├── monitoring/
│   │   ├── dashboard.py
│   │   ├── alerts.py
│   │   └── health_check.py
│   │
│   └── database/
│       ├── data_store.py
│       └── ingestion_pipeline.py
│
├── scripts/
│   ├── integrated_trading_system.py   # Complete integration example
│   ├── start_live.py
│   ├── start_backtest.py
│   └── retrain_models.py
│
├── tests/
│   ├── test_core.py
│   ├── test_strategies.py
│   ├── test_backtesting.py
│   └── test_risk_management.py
│
├── docs/
│   ├── IMPLEMENTATION_SUMMARY.md      # Detailed implementation guide
│   ├── architecture.md
│   └── api_reference.rst
│
├── docker/
│   ├── Dockerfile
│   └── docker-compose.yml
│
├── pyproject.toml                      # Poetry dependencies
└── README.md
```

---

## 🛠 Requirements

- **Python 3.13**
- **Poetry** (for dependency management)
- **MetaTrader 5** with ZeroMQ EA
- **Windows** (for MT5 integration)

### Optional
- **TA-Lib** (precompiled wheel recommended for Windows)
- **TimescaleDB** or **InfluxDB** (for time-series data storage)
- **Docker** (for containerized deployment)

---

## 📦 Installation

### 1. Clone Repository
```powershell
git clone https://github.com/whateve-r/UNDERDOG.git
cd UNDERDOG
```

### 2. Install Poetry
```powershell
pip install poetry
```

### 3. Install Dependencies
```powershell
poetry install
```

### 4. Install TA-Lib (Optional)
Download precompiled wheel from [TA-Lib releases](https://github.com/cgohlke/talib-build/releases) and install:
```powershell
poetry run pip install TA_Lib‑0.4.28‑cp313‑cp313‑win_amd64.whl
```

### 5. Configure MT5 Connection
Edit `config/runtime/env/mt5_credentials.yaml`:
```yaml
mt5_exe_path: "C:\\Your\\Path\\To\\MetaTrader 5\\terminal64.exe"
mql5_script: "JsonAPI.ex5"  # Your ZeroMQ EA
```

---

## 🚀 Quick Start

### Basic Usage

```python
from underdog.risk_management.risk_master import RiskMaster, DrawdownLimits
from underdog.risk_management.position_sizing import PositionSizer
from underdog.strategies.strategy_matrix import StrategyMatrix

# Initialize risk management
risk_master = RiskMaster(
    initial_capital=100000,
    dd_limits=DrawdownLimits(max_daily_dd_pct=5.0)
)

position_sizer = PositionSizer()

# Create strategy matrix
matrix = StrategyMatrix(risk_master, position_sizer)
matrix.register_strategy("keltner_breakout", allocation_pct=20.0)

# Submit and process signals
# ... (see docs/IMPLEMENTATION_SUMMARY.md for full example)
```

### Run Integrated System

```powershell
poetry run python scripts/integrated_trading_system.py
```

### Run Backtest

```powershell
poetry run python scripts/start_backtest.py --strategy keltner_breakout --period 2023-01-01:2024-12-31
```

---

## 🧪 Testing

```powershell
# Run all tests
poetry run pytest tests/ -v

# Run specific test module
poetry run pytest tests/test_risk_management.py -v

# Run with coverage
poetry run pytest --cov=underdog tests/
```

---

## 📊 Architecture Overview

### Data Flow

```
MetaTrader 5 (MQL5 EA)
    ↓ ZeroMQ PUB/SUB
MT5 Connector (Python)
    ↓ Market Data
Feature Engineering → ML Models → Fuzzy Logic
    ↓ Raw Signals
Strategy Matrix (Aggregation & Correlation)
    ↓ Aggregated Signals
Risk Master (DD & Exposure Checks)
    ↓ Risk-Adjusted Signals
Position Sizer (Kelly + Confidence)
    ↓ Final Orders
Execution Engine → MT5
```

### Key Components

1. **MT5 Connector**: Async ZeroMQ-based communication with MetaTrader 5
2. **Strategy Matrix**: Portfolio-level coordinator for multiple strategies
3. **Risk Master**: Portfolio-wide risk management with DD limits and correlation tracking
4. **Position Sizer**: Multi-factor position sizing (Fixed Fractional + Kelly + Confidence)
5. **Fuzzy Logic**: Transform ML outputs into confidence scores
6. **Backtesting Engine**: Event-driven with Walk-Forward Optimization and Monte Carlo

---

## 🔐 Risk Management

### Drawdown Limits (Prop Firm Compliant)
- **Daily DD**: 5% (configurable)
- **Weekly DD**: 10%
- **Monthly DD**: 15%
- **Absolute DD**: 20%
- **Soft Limits**: Position scaling starts at 80% of limit

### Exposure Constraints
- **Total Portfolio**: 100% max exposure
- **Per Symbol**: 10% max
- **Per Strategy**: 30% max
- **Correlated Assets**: 40% max

### Position Sizing Formula
```
BaseSize = (AccountBalance × RiskPct) / (StopDistance × PipValue)
KellyAdjusted = BaseSize × (1 + KellyFraction)
ConfidenceWeighted = KellyAdjusted × (Confidence ^ Exponent)
FinalSize = ConfidenceWeighted × DDScaling × PortfolioScaling
```

---

## 🤖 Machine Learning Pipeline

### Supported Models
- **LSTM**: Sequence-to-sequence for time-series prediction
- **CNN 1D**: Local pattern extraction in price windows
- **Transformers**: Attention-based for multi-asset dependencies
- **HMM**: Market regime detection
- **Kalman Filters**: Dynamic hedge ratio estimation for pairs trading

### Experiment Tracking
- **MLflow** integration for model versioning
- **Optuna** for hyperparameter optimization
- Reproducible experiments with DVC/hash versioning

---

## 📈 Validation & Backtesting

### Walk-Forward Optimization (WFO)
- Automated IS/OS segmentation
- Parameter optimization per fold
- Out-of-sample performance metrics
- Robust strategy validation

### Monte Carlo Simulation
- Trade sequence resampling
- Slippage and latency distributions
- Parameter perturbation testing
- Percentile-based risk assessment (5th, 1st percentile DD)

---

## 🐳 Docker Deployment

```powershell
# Build image
docker-compose build

# Start services
docker-compose up -d

# View logs
docker-compose logs -f underdog
```

---

## 📝 Configuration

### MT5 Credentials (`config/runtime/env/mt5_credentials.yaml`)
```yaml
zmq_host: "127.0.0.1"
sys_port: 25555
mt5_exe_path: "C:\\Program Files\\MetaTrader 5\\terminal64.exe"
mql5_script: "JsonAPI.ex5"
sys_timeout: 3.0
heartbeat: 5
auto_restart: true
```

### Fuzzy Rules (`config/strategies/fuzzy_confidence_rules.yaml`)
```yaml
input_variables:
  - name: ml_prob
    range: [0.0, 1.0]
    sets:
      - label: high
        mf:
          type: trapezoidal
          a: 0.5
          b: 0.6
          c: 1.0
          d: 1.0
rules:
  - if:
      - var: ml_prob
        set: high
      - var: momentum
        set: positive
    then:
      var: confidence
      set: very_high
```

---

## 🔧 Development

### Code Formatting
```powershell
poetry run black underdog/
poetry run isort underdog/
```

### Linting
```powershell
poetry run flake8 underdog/
poetry run mypy underdog/
```

---

## 📚 Documentation

- **Implementation Guide**: `docs/IMPLEMENTATION_SUMMARY.md`
- **Architecture**: `docs/architecture.md`
- **API Reference**: `docs/api_reference.rst`

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## ⚠️ Disclaimer

**This software is for educational and research purposes only. Trading involves substantial risk of loss. The authors are not responsible for any financial losses incurred through the use of this software. Always test thoroughly on demo accounts before live trading.**

---

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## 🙏 Acknowledgments

- MetaTrader 5 platform
- ZeroMQ messaging library
- TA-Lib technical analysis library
- MLflow for experiment tracking
- The quantitative finance community

---

## 📞 Contact

- **Author**: whateve-r
- **Repository**: https://github.com/whateve-r/UNDERDOG
- **Issues**: https://github.com/whateve-r/UNDERDOG/issues

---

**Built with ❤️ for algorithmic traders and quantitative researchers**
