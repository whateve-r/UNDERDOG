# Phase 2 Implementation - Completion Guide

## 🎯 Status: **5 of 8 Modules Complete** (62.5%)

**Completed**: ✅ WFO, ✅ Monte Carlo, ✅ MLflow, ✅ HMM Regimes, ✅ Feature Engineering  
**Remaining**: ⏳ Unit Tests, ⏳ Monitoring, ⏳ Docker

---

## 📦 Completed Implementations

### 1. Walk-Forward Optimization Engine ✅
**File**: `underdog/backtesting/validation/wfo.py` (600+ lines)

**Features**:
- Automated IS/OOS segmentation (anchored & rolling windows)
- Grid search parameter optimization per fold
- Comprehensive metrics (Sharpe, Sortino, Profit Factor, CAGR, MDD)
- Matplotlib visualization with IS/OOS comparison
- Prevents overfitting through rigorous out-of-sample validation

**Scientific Basis**: Addresses *Optimization Bias* (data snooping) through systematic parameter validation on unseen data.

**Key Classes**:
- `WFOConfig`: Configuration (n_folds, train_ratio, anchored)
- `WalkForwardOptimizer`: Main optimization engine
- `WFOResult`: Results with fold-level and aggregated metrics

**Usage**:
```python
from underdog.backtesting.validation.wfo import WalkForwardOptimizer, WFOConfig

config = WFOConfig(n_folds=5, train_ratio=0.7, anchored=False)
optimizer = WalkForwardOptimizer(config)

results = optimizer.run(
    prices=price_series,
    strategy_func=your_strategy,
    param_grid={'period': [10, 20, 30]}
)

print(f"Avg OOS Sharpe: {results.summary['avg_oos_sharpe']:.3f}")
```

---

### 2. Monte Carlo Simulation Engine ✅
**File**: `underdog/backtesting/validation/monte_carlo.py` (550+ lines)

**Features**:
- Bootstrap trade resampling with replacement
- Slippage simulation (normal distribution, µ=0.0001, σ configurable)
- Percentile analysis (5th, 50th, 95th)
- Risk metrics: VaR (5%), CVaR, Worst-case Drawdown
- Probability calculations (P(positive), P(target), P(ruin))

**Scientific Basis**: Quantifies *robustness* under parameter uncertainty and execution friction. Critical for understanding tail risk.

**Key Classes**:
- `MonteCarloConfig`: Configuration (n_simulations, slippage params)
- `MonteCarloSimulator`: Simulation engine
- `MonteCarloResult`: Results with percentiles and risk metrics

**Usage**:
```python
from underdog.backtesting.validation.monte_carlo import MonteCarloSimulator, MonteCarloConfig

config = MonteCarloConfig(
    n_simulations=5000,
    add_slippage=True,
    slippage_std=0.0001
)
simulator = MonteCarloSimulator(config)

results = simulator.run(trades_df)
print(f"VaR (5%): ${results.summary['var_5']:.2f}")
print(f"Max DD Mean: {results.summary['max_dd_mean']:.2%}")
```

---

### 3. MLflow Integration ✅
**File**: `underdog/ml/training/train_pipeline.py` (650+ lines)

**Features**:
- Experiment tracking (`mlflow.log_params`, `mlflow.log_metrics`)
- Model registry with versioning (MLflow Model Registry)
- Staging workflow (None → Staging → Production → Archived)
- Artifact management (model checkpoints, configs)
- Data hash versioning (SHA256) for reproducibility
- Multi-framework support (PyTorch, scikit-learn, XGBoost)

**Scientific Basis**: Ensures *reproducibility* and *governance* through versioned experiments and model lineage tracking.

**Key Classes**:
- `TrainingConfig`: Training configuration
- `MLTrainingPipeline`: Pipeline with MLflow integration
- Methods: `train()`, `promote_model()`, `load_production_model()`

**Model Types Supported**:
- LSTM (PyTorch, 2-layer architecture)
- CNN 1D (Conv layers with adaptive pooling)
- Random Forest (scikit-learn baseline)
- XGBoost (gradient boosting)

**Usage**:
```python
from underdog.ml.training.train_pipeline import MLTrainingPipeline, TrainingConfig

config = TrainingConfig(
    model_name="lstm_predictor",
    model_type="lstm",
    experiment_name="underdog_ml"
)

pipeline = MLTrainingPipeline(config)
results = pipeline.train(X_train, y_train, X_val, y_val, run_name="exp_001")

# Promote to production
pipeline.promote_model("lstm_predictor", version=1, stage="Production")
```

---

### 4. HMM Regime Classifier ✅
**File**: `underdog/ml/models/regime_classifier.py` (700+ lines)

**Features**:
- Multi-state HMM (3-5 states) fitted on returns + volatility
- Regime types: BULL, BEAR, SIDEWAYS, HIGH_VOL, LOW_VOL
- Viterbi algorithm for most likely state sequence
- ADF test for stationarity validation (mean reversion)
- Model gating logic for strategy activation/deactivation
- Regime persistence filtering (minimum duration)
- Dynamic retraining (configurable frequency)

**Scientific Basis**: Captures *latent regime transitions* in market dynamics. Prevents applying trend strategies in sideways markets (and vice versa).

**Key Classes**:
- `RegimeConfig`: Configuration (n_states, min_regime_duration)
- `HMMRegimeClassifier`: Main classifier
- `RegimeState`: Current state with confidence and characteristics

**Strategy Gating Logic**:
- **Trend strategies** (SMA, Momentum): Active in BULL/BEAR, inactive in SIDEWAYS
- **Mean reversion** (Pairs Trading): Active in SIDEWAYS (if stationary), inactive in trending
- **Breakout strategies**: Active in LOW_VOL (compression before breakout)
- **ML predictive**: Active in all regimes except HIGH_VOL

**Usage**:
```python
from underdog.ml.models.regime_classifier import HMMRegimeClassifier, RegimeConfig

config = RegimeConfig(n_states=4, min_regime_duration=5)
classifier = HMMRegimeClassifier(config)

classifier.fit(price_series)
current_state = classifier.predict_current(price_series)

print(f"Regime: {current_state.regime.value}")
print(f"Confidence: {current_state.confidence:.2%}")

# Strategy gating
if classifier.get_strategy_gate('trend'):
    execute_trend_strategy()
```

---

### 5. Feature Engineering Pipeline ✅
**File**: `underdog/strategies/ml_strategies/feature_engineering.py` (800+ lines)

**Features**:
- Hash-based versioning (config_hash + data_hash → deterministic version)
- Technical indicators: SMA, EMA, RSI, Bollinger, ATR
- Momentum features: ROC, momentum, stochastic
- Volatility features: Realized vol, Parkinson vol
- Lagged features (1, 2, 3, 5, 10 periods) for ML
- Cyclic temporal encoding (hour_sin/cos, dow_sin/cos, month_sin/cos)
- Target generation (return, direction, classification)
- Look-ahead bias validation
- Parquet export for efficient storage

**Scientific Basis**: Ensures *reproducibility* through versioned feature sets. Prevents *look-ahead bias* through strict point-in-time calculation.

**Key Classes**:
- `FeatureConfig`: Configuration with all feature parameters
- `FeatureEngineer`: Main transformation engine
- `FeatureMetadata`: Versioning metadata (hashes, timestamps)

**Versioning Workflow**:
1. Features generated → config_hash + data_hash calculated
2. Features saved: `features_{config_hash}_{data_hash}.parquet`
3. Metadata saved: `metadata_{config_hash}_{data_hash}.json`
4. Load by version: `FeatureEngineer.load_features(version)`

**Usage**:
```python
from underdog.strategies.ml_strategies.feature_engineering import FeatureEngineer, FeatureConfig

config = FeatureConfig(
    sma_periods=[20, 50, 200],
    lag_periods=[1, 2, 3],
    target_type="classification"
)

engineer = FeatureEngineer(config)
features = engineer.transform(ohlcv_data)

# Save with versioning
path = engineer.save_features(features, base_path="data/processed")

# Load later
features, metadata = FeatureEngineer.load_features(version="abc123_def456")
```

---

## 🔗 Complete Integration Example

**File**: `scripts/complete_trading_workflow.py` (400+ lines)

Demonstrates end-to-end workflow:
1. **Feature Engineering** → Hash-versioned features
2. **Regime Detection** → HMM-based regime classification
3. **ML Training** → MLflow experiment tracking
4. **Strategy Gating** → Regime-based activation logic
5. **Risk Management** → Multi-layer validation (Risk Master + Position Sizer)
6. **Walk-Forward Optimization** → Parameter validation
7. **Monte Carlo Validation** → Robustness testing

**Run**:
```bash
python scripts/complete_trading_workflow.py
```

---

## 🚀 Acceleration Recommendations (Based on Scientific Methodology)

### Priority 1: Remaining Module Implementation ⏰ **FAST TRACK**

#### A. Unit Tests Suite (⏳ ~2 hours)
**File**: `tests/test_*.py`

**Critical Tests** (aligned with scientific rigor):
1. **`test_schemas.py`**: ZeroMQ message validation (no look-ahead bias in timestamps)
2. **`test_risk_master.py`**: Drawdown limits enforcement, correlation matrix updates
3. **`test_position_sizing.py`**: Kelly fraction calculation, confidence scaling
4. **`test_fuzzy_logic.py`**: Mamdani inference, rule evaluation, confidence scoring
5. **`test_feature_engineering.py`**: Hash reproducibility, no look-ahead bias validation
6. **`test_regime_classifier.py`**: State labeling, strategy gating logic

**Implementation Approach** (pytest):
```python
import pytest
from underdog.risk_management.risk_master import RiskMaster, DrawdownLimits

def test_daily_drawdown_limit():
    """Test that daily DD limit triggers kill switch"""
    limits = DrawdownLimits(daily_max=0.02)
    risk_master = RiskMaster(initial_capital=100000.0, dd_limits=limits)
    
    # Simulate 2.5% loss
    risk_master.update_pnl(-2500.0)
    
    # Should reject new trades
    assert not risk_master.pre_trade_check('EURUSD', 'long', 10000.0)
```

**Estimated Time**: 2 hours (10-15 tests total, pytest fixtures)

---

#### B. Monitoring & Telemetry (⏳ ~3 hours)
**Files**: `underdog/monitoring/dashboard.py`, `health_check.py`, `alerts.py`

**Components**:
1. **Prometheus Metrics** (`prometheus_client`):
   - Counters: `trades_total`, `orders_filled`, `orders_rejected`
   - Gauges: `current_capital`, `drawdown_pct`, `active_positions`
   - Histograms: `trade_pnl`, `execution_latency_ms`

2. **Health Checks** (HTTP endpoint `/health`):
   - MT5 connection status
   - ZeroMQ subscriber alive
   - Risk Master state (kill switch status)
   - Model staleness (last prediction timestamp)

3. **Alerting** (Email/Slack via webhooks):
   - Drawdown threshold breached
   - Connection lost to MT5
   - Anomalous trade rejected (>5 consecutive rejections)

**Implementation Sketch**:
```python
from prometheus_client import Counter, Gauge, Histogram, start_http_server

trades_counter = Counter('trades_total', 'Total trades executed', ['symbol', 'side'])
capital_gauge = Gauge('current_capital', 'Current account capital')
pnl_histogram = Histogram('trade_pnl', 'Trade P&L distribution')

# Start metrics server
start_http_server(8000)  # Accessible at http://localhost:8000/metrics

# In trading loop
trades_counter.labels(symbol='EURUSD', side='long').inc()
capital_gauge.set(risk_master.current_capital)
pnl_histogram.observe(trade_pnl)
```

**Grafana Dashboard** (optional): JSON config for visualizing metrics.

**Estimated Time**: 3 hours (basic Prometheus + health endpoint + Slack alerts)

---

#### C. Docker Production Setup (⏳ ~2 hours)
**Files**: `docker/Dockerfile`, `docker/docker-compose.yml`

**Multi-Service Architecture**:
1. **underdog** (main trading container)
   - Python 3.13 + all dependencies
   - Connects to MT5 via ZeroMQ
   - Mounts `config/` and `data/` volumes

2. **timescaledb** (time-series database)
   - PostgreSQL with TimescaleDB extension
   - Stores OHLCV data, trades, metrics

3. **grafana** (monitoring dashboard)
   - Connects to TimescaleDB + Prometheus
   - Pre-configured dashboards

4. **prometheus** (metrics collection)
   - Scrapes `/metrics` from underdog container

**docker-compose.yml Sketch**:
```yaml
version: '3.8'

services:
  underdog:
    build: .
    container_name: underdog_trading
    volumes:
      - ./config:/app/config
      - ./data:/app/data
    environment:
      - MT5_SERVER=${MT5_SERVER}
      - MT5_LOGIN=${MT5_LOGIN}
      - MT5_PASSWORD=${MT5_PASSWORD}
    depends_on:
      - timescaledb
    restart: unless-stopped

  timescaledb:
    image: timescale/timescaledb:latest-pg14
    container_name: underdog_db
    environment:
      - POSTGRES_PASSWORD=${DB_PASSWORD}
    volumes:
      - timescale_data:/var/lib/postgresql/data
    ports:
      - "5432:5432"

  prometheus:
    image: prom/prometheus:latest
    container_name: underdog_prometheus
    volumes:
      - ./docker/prometheus.yml:/etc/prometheus/prometheus.yml
    ports:
      - "9090:9090"

  grafana:
    image: grafana/grafana:latest
    container_name: underdog_grafana
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=${GRAFANA_PASSWORD}

volumes:
  timescale_data:
```

**Estimated Time**: 2 hours (Dockerfile + docker-compose + .env template)

---

### Priority 2: Scientific Rigor Enhancements (Based on Provided Methodology)

#### A. **Slippage Modeling** (Monte Carlo Enhancement)
**Current**: Basic normal distribution slippage (`µ=0.0001, σ=0.0001`)  
**Enhancement**: Regime-dependent slippage (higher in HIGH_VOL regimes)

```python
# In MonteCarloSimulator
if regime == RegimeType.HIGH_VOL:
    slippage_std = config.slippage_std * 2.0  # 2x slippage in high vol
elif regime == RegimeType.LOW_VOL:
    slippage_std = config.slippage_std * 0.5  # 0.5x in low vol
```

**Scientific Basis**: *Slippage increases with volatility*. Failing to model this overstates strategy performance in volatile regimes.

---

#### B. **Stop Loss Optimization** (ATR-Based Dynamic SL)
**Current**: Fixed fractional SL in Position Sizer  
**Enhancement**: ATR-based dynamic SL (as described in methodology)

```python
# In PositionSizer
def calculate_stop_loss_atr(self, entry_price: float, atr: float, direction: str, multiplier: float = 1.5) -> float:
    """Calculate ATR-based stop loss"""
    if direction == 'long':
        return entry_price - (multiplier * atr)
    else:
        return entry_price + (multiplier * atr)
```

**Scientific Basis**: *Fixed pips fail in high volatility*. ATR adapts SL to market noise, reducing whipsaws.

---

#### C. **Axiom of Last Night's Close** (Psychological Objectivity)
**Implementation**: Add to Risk Master as cognitive bias mitigation

```python
# In RiskMaster
def evaluate_position_objectively(self, position: Dict) -> Dict:
    """
    Evaluate position as if entry was last night's close.
    
    Prevents anchoring to original entry price (cognitive bias).
    """
    current_price = position['current_price']
    last_close = position['last_close']  # Yesterday's close
    
    # Calculate P&L from last close (not original entry)
    objective_pnl = (current_price - last_close) / last_close
    
    return {
        'should_hold': objective_pnl > -0.01,  # Exit if down 1% from yesterday
        'objective_pnl': objective_pnl
    }
```

**Scientific Basis**: *Sunk cost fallacy* causes traders to hold losers. Re-evaluating from last close enforces discipline.

---

### Priority 3: Cloud Deployment (Production Readiness)

#### A. **VPS/Cloud Setup** (DigitalOcean/AWS EC2)
**Requirements**:
- **Latency**: <50ms to broker server (use ping to test)
- **Uptime**: 99.9% SLA (avoid consumer-grade hosting)
- **Security**: SSH key-only access, firewall rules, fail2ban

**Recommended Specs**:
- **CPU**: 2 vCPUs (sufficient for Python async)
- **RAM**: 4GB (for NumPy/Pandas operations)
- **Storage**: 50GB SSD (for historical data)
- **OS**: Ubuntu 22.04 LTS

**Setup Steps**:
```bash
# 1. Create droplet/instance (Ubuntu 22.04)
# 2. SSH into server
ssh -i ~/.ssh/underdog_key root@<server_ip>

# 3. Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sh get-docker.sh

# 4. Clone repo
git clone https://github.com/whateve-r/UNDERDOG.git
cd UNDERDOG

# 5. Configure secrets
cp config/runtime/env/mt5_credentials.yaml.example config/runtime/env/mt5_credentials.yaml
nano config/runtime/env/mt5_credentials.yaml  # Edit with real credentials

# 6. Deploy with Docker Compose
docker-compose up -d

# 7. Monitor logs
docker logs -f underdog_trading
```

**Estimated Time**: 1 hour (provisioning + deployment)

---

#### B. **SSL-Encrypted Jupyter Lab** (Remote Development)
**Purpose**: Secure remote access for research and debugging

```bash
# Install Jupyter Lab in container
pip install jupyterlab

# Generate SSL cert
openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
  -keyout jupyter.key -out jupyter.crt

# Configure Jupyter
jupyter lab --generate-config
# Edit ~/.jupyter/jupyter_lab_config.py:
c.ServerApp.certfile = '/path/to/jupyter.crt'
c.ServerApp.keyfile = '/path/to/jupyter.key'
c.ServerApp.password = 'sha1:...'  # Generate with: jupyter lab password

# Run
jupyter lab --ip=0.0.0.0 --port=8888 --no-browser
```

**Access**: `https://<server_ip>:8888`

**Estimated Time**: 30 minutes

---

## 🎯 Final Checklist (Before Live Trading)

### Pre-Deployment Validation
- [ ] **Backtests**: Run WFO on ≥3 years of data, validate OOS Sharpe ≥1.0
- [ ] **Monte Carlo**: Run 10,000 simulations, ensure P(ruin) <1%
- [ ] **Slippage**: Verify slippage parameters match broker's average (use real fills)
- [ ] **Latency**: Measure round-trip time (signal → execution) <100ms
- [ ] **Drawdown**: Confirm MDD <15% in Monte Carlo worst-case scenario
- [ ] **Regime Detection**: Validate HMM states align with visual market analysis
- [ ] **Strategy Gating**: Test that strategies deactivate in wrong regimes (e.g., trend in sideways)
- [ ] **Risk Limits**: Trigger kill switch manually, verify all trading halts

### Unit Tests
- [ ] All tests pass: `pytest tests/ -v`
- [ ] Coverage ≥80%: `pytest --cov=underdog tests/`

### Monitoring
- [ ] Prometheus metrics accessible: `curl http://localhost:8000/metrics`
- [ ] Health check returns 200: `curl http://localhost:8080/health`
- [ ] Alerting tested: Send test alert to Slack/email

### Docker
- [ ] All containers start: `docker-compose up -d`
- [ ] Logs clean (no errors): `docker logs underdog_trading`
- [ ] Database connection: `psql -h localhost -U postgres -d underdog`

### Compliance (Prop Firm)
- [ ] **Drawdown Limits**: Daily 2%, Weekly 5%, Monthly 10%
- [ ] **Position Sizing**: No single position >5% of capital
- [ ] **Leverage**: ≤2x (configurable in SizingConfig)
- [ ] **Prohibited Strategies**: No martingale, no grid trading (not implemented)

---

## 📚 Key References (Scientific Methodology)

1. **Overfitting/Data Snooping**: Walk-Forward Optimization (WFO) mitigates by validating parameters on unseen data.
2. **Look-Ahead Bias**: Feature engineering validates point-in-time calculation (no future data leakage).
3. **Survivorship Bias**: Use bias-free datasets or limit backtests to recent periods.
4. **Slippage Modeling**: Monte Carlo simulates realistic execution costs (normal distribution).
5. **Kelly Criterion**: Position Sizer uses Half-Kelly for conservative sizing (mitigates estimation error).
6. **ATR-Based SL**: Dynamic stops adapt to volatility (prevents whipsaws).
7. **Axiom of Last Close**: Risk Master evaluates positions objectively (prevents anchoring bias).
8. **Regime Detection**: HMM captures latent market states (prevents applying wrong strategy type).

---

## 🚀 Next Actions (Immediate)

### For Next 4 Hours (Fast Completion):
1. **Hour 1-2**: Implement Unit Tests (`test_schemas.py`, `test_risk_master.py`, `test_position_sizing.py`)
2. **Hour 3**: Implement Monitoring (`prometheus_client` metrics + health check endpoint)
3. **Hour 4**: Implement Docker (`Dockerfile` + `docker-compose.yml` with TimescaleDB)

### After Phase 2 Completion:
1. **Deploy to VPS**: DigitalOcean droplet (2 vCPU, 4GB RAM, $24/month)
2. **Connect MT5**: Test ZeroMQ connection with live account (demo first)
3. **Paper Trade**: Run for 30 days, validate execution and slippage
4. **Go Live**: Start with 1% capital allocation, scale gradually

---

## 📖 Documentation

- **Architecture**: `docs/architecture.md`
- **Implementation Summary**: `docs/IMPLEMENTATION_SUMMARY.md`
- **API Reference**: `docs/api_reference.rst`
- **Complete Workflow**: Run `python scripts/complete_trading_workflow.py`

---

## 🎓 Key Takeaways (Scientific Rigor)

1. **Method Scientific**: Every strategy has hypothesis → test → validation (not data mining)
2. **Overfitting Prevention**: WFO + Monte Carlo ensure robustness
3. **Transaction Costs**: Slippage modeled realistically (critical for HFT)
4. **Risk Management**: Multi-layer (Risk Master → Position Sizer → Strategy Matrix)
5. **Regime Awareness**: HMM prevents applying trend strategies in sideways markets
6. **Reproducibility**: Hash-based versioning (features, models, experiments)
7. **Psychological Discipline**: Axiom of Last Close + predetermined SL/TP
8. **Infrastructure**: Cloud deployment for 24/7 uptime + low latency

---

**Status**: Ready for final 3 modules (Tests, Monitoring, Docker) → Full production deployment

**Estimated Completion**: 4-6 hours

**Next Command**: `python scripts/complete_trading_workflow.py` (verify all components)
