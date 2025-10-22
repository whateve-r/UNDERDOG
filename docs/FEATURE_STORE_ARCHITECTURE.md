# Feature Store Architecture - TimescaleDB + Redis

**Date**: 2025-10-22  
**Status**: IMPLEMENTATION READY  
**Papers**: AlphaQuanter.pdf, arXiv:2510.10526v1

---

## 🎯 Objective

Implementar **Feature Store centralizado** para DRL Agent con 3 categorías de datos según frecuencia y latencia.

---

## 📊 Data Categories (User-Provided Architecture)

### 1. 📈 Microestructura (Alta Frecuencia - Streaming)

| Dato | Fuente | Frecuencia | Latencia | Uso |
|------|--------|------------|----------|-----|
| **Precios Históricos** | MT5 / Dukascopy | Backfill (1 vez) | N/A | Base para features técnicos |
| **Precios Tiempo Real** | MT5 (ZeroMQ) | Streaming (tick/M1) | <100ms | Ejecución algorítmica |
| **Order Book / Bid-Ask** | MT5 / Alpha Vantage | Pooling (1-5s) | <1s | Optimización slippage (Constrained RL) |

**Storage**: TimescaleDB `ohlcv` table (hypertable partitioned by time)

### 2. 📰 Sentiment (Media Frecuencia - Pooling 15min)

| Dato | Fuente | Frecuencia | Latencia | Uso |
|------|--------|------------|----------|-----|
| **Noticias Financieras** | NewsAPI / GNews | Pooling (1-15min) | ~5s | Input para FinGPT |
| **Reddit Posts** | Reddit API (PRAW) | Pooling (15-30min) | ~10s | Sentiment retail |
| **Sentiment Score** | FinGPT (local inference) | Streaming (post-news) | ~5-10s | Feature DRL (TD3) |

**Storage**: TimescaleDB `sentiment_scores` table + Redis cache (1h TTL)

### 3. 🌍 Macro (Baja Frecuencia - EOD)

| Dato | Fuente | Frecuencia | Latencia | Uso |
|------|--------|------------|----------|-----|
| **Datos Macroeconómicos** | FRED API | EOD / después publicación | ~2s | Clasificación régimen macro |
| **SEC / Insider Trading** | Alpha Vantage / Edgar | EOD | ~2s | Fundamentals (si operas acciones) |

**Storage**: TimescaleDB `macro_indicators` table

---

## 🏗️ Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                     DATA SOURCES (3 Categories)                  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
    ┌──────────────┬──────────────┬──────────────┐
    │   MT5 (ZMQ)  │  Reddit API  │  FRED API    │
    │  Streaming   │  Pooling 15m │  Pooling EOD │
    └──────────────┴──────────────┴──────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│           DataIngestionOrchestrator (Coordinator)                │
│  • MT5 Streaming Pipeline (tick-by-tick)                        │
│  • Sentiment Pooling Pipeline (Reddit → FinGPT → Score)         │
│  • Macro Pooling Pipeline (FRED → Indicators)                   │
│  • Health Monitor (stats, errors, last_update)                  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
         ┌─────────────────────────────────────┐
         │      FinGPT Connector (Inference)    │
         │  • Model: ProsusAI/finbert (135M)   │
         │  • Input: News text + Reddit posts  │
         │  • Output: Sentiment [-1, 1]        │
         │  • Latency: ~5-10s (batch)          │
         └─────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                  TimescaleDB (Feature Store)                     │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │ ohlcv                    • Hypertable (1-day chunks)       │ │
│  │ ├── time, symbol, open, high, low, close, volume, spread  │ │
│  │ ├── Compression: 30 days                                  │ │
│  │ └── Retention: 2 years                                    │ │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │ sentiment_scores         • Hypertable (1-day chunks)       │ │
│  │ ├── time, symbol, sentiment, source, confidence           │ │
│  │ ├── Compression: 60 days                                  │ │
│  │ └── Retention: 1 year                                     │ │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │ macro_indicators         • Hypertable (7-day chunks)       │ │
│  │ ├── time, series_id, value, name, units                   │ │
│  │ ├── Compression: 180 days                                 │ │
│  │ └── Retention: 5 years                                    │ │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │ regime_predictions       • Hypertable (1-day chunks)       │ │
│  │ ├── time, symbol, regime, confidence, features            │ │
│  │ ├── Compression: 90 days                                  │ │
│  │ └── Retention: 2 years                                    │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                  │
│  Continuous Aggregates:                                          │
│  • daily_trading_summary (trades, win rate, PnL)                │
│  • hourly_sentiment_summary (avg sentiment, volatility)         │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
         ┌─────────────────────────────────────┐
         │     Redis (Hot Cache - 1h TTL)      │
         │  • Latest sentiment scores          │
         │  • Latest macro indicators          │
         │  • Current regime predictions       │
         │  • Max memory: 512MB (LRU eviction) │
         └─────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│               StateVectorBuilder (14-dim)                        │
│  • Price features (close, returns, ATR)                          │
│  • Technical indicators (MACD, RSI, BB width)                    │
│  • Sentiment score (from TimescaleDB/Redis)                      │
│  • Regime one-hot (trend, range, transition)                     │
│  • Macro indicators (VIX, Fed Funds, Yield Curve)                │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
         ┌─────────────────────────────────────┐
         │        DRL Agent (TD3)              │
         │  • State: 14-dim vector             │
         │  • Action: Position size + Entry    │
         │  • Reward: Sharpe - DD penalty      │
         └─────────────────────────────────────┘
```

---

## 🔧 Implementation Components

### 1. `underdog/database/timescale/timescale_connector.py` (READY ✅)

**Status**: Implemented (300 LOC)

```python
class TimescaleDBConnector:
    async def connect()
    async def create_schema()
    async def insert_ohlcv_batch(bars: List[OHLCVBar])
    async def query_ohlcv(symbol, start_date, end_date) -> pd.DataFrame
    async def insert_sentiment_batch(scores: List[SentimentScore])
    async def get_latest_sentiment(symbol, source) -> float
```

**Features**:
- Async connection pool (asyncpg)
- Batch inserts (>100k bars/sec target)
- Time-series queries optimized

### 2. `underdog/database/timescale/data_orchestrator.py` (READY ✅)

**Status**: Implemented (400 LOC)

```python
class DataIngestionOrchestrator:
    async def start()
    async def _mt5_streaming_pipeline()
    async def _sentiment_pooling_pipeline()
    async def _macro_pooling_pipeline()
    async def _health_monitor()
```

**Features**:
- 3 async pipelines (MT5, Sentiment, Macro)
- FinGPT integration for sentiment inference
- Health monitoring (stats every 5 min)
- Error handling + retries

### 3. `underdog/sentiment/llm_connector.py` (READY ✅)

**Status**: Implemented (200 LOC)

```python
class FinGPTConnector:
    def analyze(text: str) -> float
    def analyze_batch(texts: List[str], aggregate=True) -> float | List[float]
    def analyze_detailed(text: str) -> SentimentResult
```

**Model**: `ProsusAI/finbert` (135M params, FinBERT fine-tuned)

**Key Point** (User Clarification):
- **NO re-training from scratch**
- Use **pre-trained FinGPT** for inference
- Local inference (no Alpha Vantage MCP dependency)
- Fine-tuning optional (if needed for FOREX-specific slang)

### 4. `docker/docker-compose.yml` (UPDATED ✅)

**Services**:
- `timescaledb`: PostgreSQL + TimescaleDB extension (port 5432)
- `redis`: Hot cache (port 6379, 512MB max, LRU eviction)
- `prometheus`: Metrics collection
- `grafana`: Dashboards

**Volumes**:
- `timescaledb-data`: Persistent DB storage
- `redis-data`: Persistent cache (optional)

### 5. `docker/init-db.sql` (EXTENDED ✅)

**New Tables**:
- `sentiment_scores` (time, symbol, sentiment, source, confidence)
- `macro_indicators` (time, series_id, value, name, units)
- `regime_predictions` (time, symbol, regime, confidence, features)

**Continuous Aggregates**:
- `hourly_sentiment_summary` (rolling 24h avg sentiment)
- `daily_trading_summary` (trades, win rate, PnL)

**Policies**:
- Compression: 30-180 days (depending on table)
- Retention: 1-5 years (depending on table)

---

## 📦 Dependencies

Add to `pyproject.toml`:

```toml
[tool.poetry.dependencies]
asyncpg = "^0.29.0"        # TimescaleDB async driver
redis = "^5.0.0"           # Redis client
transformers = "^4.35.0"   # FinGPT/FinBERT
torch = "^2.1.0"           # PyTorch (for transformers)
praw = "^7.8.1"            # Reddit API (already added)
aiohttp = "^3.9.0"         # Async HTTP (already added)
```

Install:
```bash
poetry add asyncpg redis transformers torch
```

---

## 🚀 Deployment Steps

### Step 1: Start Docker Stack (5 min)

```bash
cd docker

# Create .env file
cat > .env << EOF
DB_USER=underdog
DB_PASSWORD=your_secure_password
DB_NAME=underdog_trading
GRAFANA_PASSWORD=admin123
EOF

# Start services
docker-compose up -d timescaledb redis

# Verify
docker ps
docker logs underdog-timescaledb
docker logs underdog-redis
```

### Step 2: Verify TimescaleDB Schema (2 min)

```bash
# Connect to DB
docker exec -it underdog-timescaledb psql -U underdog -d underdog_trading

# Check tables
\dt

# Check hypertables
SELECT hypertable_name, num_chunks 
FROM timescaledb_information.hypertables;

# Exit
\q
```

Expected tables:
- `ohlcv`
- `sentiment_scores`
- `macro_indicators`
- `regime_predictions`
- `trades`, `positions`, `metrics`, `account_snapshots`

### Step 3: Test TimescaleDB Connector (5 min)

```bash
# Set environment variables
export DB_HOST=localhost
export DB_PORT=5432
export DB_USER=underdog
export DB_PASSWORD=your_secure_password
export DB_NAME=underdog_trading

# Test connector
python underdog/database/timescale/timescale_connector.py
```

Expected output:
```
✓ Connected
✓ Inserted OHLCV bar
✓ Queried 1 bars
✓ Disconnected
```

### Step 4: Setup API Credentials (3 min)

```bash
# Add to .env or export
export REDDIT_CLIENT_ID="your_reddit_client_id"
export REDDIT_CLIENT_SECRET="your_reddit_client_secret"
export FRED_API_KEY="your_fred_api_key"
export ALPHAVANTAGE_API_KEY="your_alphavantage_api_key"
```

### Step 5: Test Data Orchestrator (10 min)

```bash
# Test orchestrator (runs for 2 min)
python underdog/database/timescale/data_orchestrator.py
```

Expected output:
```
DataIngestionOrchestrator initialized
✓ MT5 streaming pipeline started
✓ Sentiment pooling pipeline started
✓ Macro pooling pipeline started
✓ Health monitor started

=== Data Ingestion Health ===
OHLCV bars ingested: 120
Sentiment scores ingested: 8
Macro indicators ingested: 8
Errors: 0
============================
```

### Step 6: Download FinGPT Model (10 min, one-time)

```bash
# Test FinGPT (downloads model ~500MB)
python underdog/sentiment/llm_connector.py
```

Expected output:
```
Loading model: ProsusAI/finbert
Downloading (…)lve/main/config.json: 100%|██████████| 878/878 [00:00<00:00, 439kB/s]
Downloading pytorch_model.bin: 100%|██████████| 438M/438M [02:30<00:00, 2.91MB/s]
Model loaded successfully

Single analysis:
  'EUR/USD looking very bullish! Strong momentum!' → 0.876
  'Market crash imminent, sell everything' → -0.912
  'Neutral consolidation continues' → 0.012

Batch analysis:
  Mean sentiment: -0.008

Detailed analysis:
  Score: 0.876, Label: positive, Confidence: 0.958
```

Model cached at: `~/.cache/huggingface/hub/models--ProsusAI--finbert/`

---

## 🧪 Integration Testing

### Test 1: Full Pipeline (Reddit → FinGPT → TimescaleDB)

```python
import asyncio
from underdog.database.timescale.data_orchestrator import DataIngestionOrchestrator, DataSourceConfig

async def test_full_pipeline():
    config = DataSourceConfig(
        mt5_enabled=False,  # Disabled for test
        reddit_enabled=True,
        sentiment_interval_minutes=1,  # Fast test
        fred_enabled=True,
        macro_interval_hours=1,
    )
    
    orchestrator = DataIngestionOrchestrator(config)
    await orchestrator.start()
    await asyncio.sleep(120)  # Run for 2 min
    await orchestrator.stop()
    
    print(f"Sentiment scores ingested: {orchestrator.stats['sentiment_scores_ingested']}")
    print(f"Macro indicators ingested: {orchestrator.stats['macro_indicators_ingested']}")

asyncio.run(test_full_pipeline())
```

### Test 2: Query Latest Sentiment

```python
import asyncio
from underdog.database.timescale.timescale_connector import TimescaleDBConnector

async def test_query_sentiment():
    db = TimescaleDBConnector()
    await db.connect()
    
    sentiment = await db.get_latest_sentiment('EURUSD', 'reddit')
    print(f"Latest Reddit sentiment for EURUSD: {sentiment}")
    
    await db.disconnect()

asyncio.run(test_query_sentiment())
```

---

## 📊 Performance Benchmarks

### Target Metrics

| Metric | Target | Actual (TBD) |
|--------|--------|--------------|
| **OHLCV Insert Rate** | >100k bars/sec | _Test needed_ |
| **Sentiment Latency** | <10s (batch 100 posts) | ~5-8s (FinGPT) |
| **Query Latency** | <50ms (1M bars) | _Test needed_ |
| **Redis Hit Rate** | >90% | _Monitor_ |
| **DB Compression** | 70-80% reduction | TimescaleDB default |

### Load Test Script

```python
import asyncio
from datetime import datetime, timedelta
from underdog.database.timescale.timescale_connector import TimescaleDBConnector, OHLCVBar

async def load_test():
    db = TimescaleDBConnector()
    await db.connect()
    
    # Generate 1M bars
    bars = [
        OHLCVBar(
            timestamp=datetime.now() - timedelta(minutes=i),
            symbol='EURUSD',
            open=1.1000 + i*0.0001,
            high=1.1010 + i*0.0001,
            low=1.0990 + i*0.0001,
            close=1.1005 + i*0.0001,
            volume=1000,
            spread=0.0001
        )
        for i in range(1_000_000)
    ]
    
    # Batch insert (10k bars per batch)
    batch_size = 10_000
    start = datetime.now()
    
    for i in range(0, len(bars), batch_size):
        await db.insert_ohlcv_batch(bars[i:i+batch_size])
    
    elapsed = (datetime.now() - start).total_seconds()
    rate = len(bars) / elapsed
    
    print(f"Inserted {len(bars)} bars in {elapsed:.2f}s")
    print(f"Insert rate: {rate:.0f} bars/sec")
    
    await db.disconnect()

asyncio.run(load_test())
```

---

## 🔍 Monitoring & Debugging

### TimescaleDB Health Check

```sql
-- Check table sizes
SELECT 
    hypertable_name,
    num_chunks,
    table_bytes / 1024 / 1024 AS table_mb,
    index_bytes / 1024 / 1024 AS index_mb,
    total_bytes / 1024 / 1024 AS total_mb
FROM timescaledb_information.hypertables;

-- Check compression status
SELECT 
    hypertable_name,
    compression_enabled,
    compressed_chunks,
    uncompressed_chunks
FROM timescaledb_information.compression_settings;

-- Check latest sentiment scores
SELECT * FROM sentiment_scores 
ORDER BY time DESC 
LIMIT 10;

-- Check macro indicators
SELECT * FROM macro_indicators 
ORDER BY time DESC 
LIMIT 10;
```

### Redis Health Check

```bash
# Connect to Redis
docker exec -it underdog-redis redis-cli

# Check keys
KEYS *

# Check memory usage
INFO memory

# Get cached sentiment
GET sentiment:EURUSD:reddit
```

---

## 📝 Next Steps (Week 1)

### IMMEDIATE (Today)
1. ✅ Docker Compose updated (TimescaleDB + Redis)
2. ✅ init-db.sql extended (sentiment, macro, regime tables)
3. ✅ DataOrchestrator implemented (400 LOC)
4. ⏳ Test deployment (`docker-compose up -d`)
5. ⏳ Verify schema creation
6. ⏳ Download FinGPT model (~500MB)

### SHORT-TERM (Week 1)
1. ⏳ Integrate Mt5Connector (ZeroMQ) with orchestrator
2. ⏳ Add News API collector (NewsAPI / GNews)
3. ⏳ Implement Redis caching layer
4. ⏳ Create StateVectorBuilder (14-dim)
5. ⏳ Load test: Insert 1M bars (target >100k/sec)
6. ⏳ Train RegimeSwitchingModel (5 years EURUSD data)

### MEDIUM-TERM (Week 2)
1. ⏳ DRL Agent (TD3) implementation
2. ⏳ Trading Environment (Gymnasium)
3. ⏳ End-to-end backtest (DRL + Shield + Regime)

---

## 💡 Key Insights (User-Provided)

### FinGPT Usage (CRITICAL)
**User Clarification**: 
> "El paper FinGPT (arXiv:2510.10526v1) no sugiere que debas entrenar FinGPT desde cero, sino que utilices la versión pre-entrenada y la afines (fine-tune) con datos recientes si es necesario."

**Implementation**:
- ✅ Use `ProsusAI/finbert` pre-trained model (Hugging Face)
- ✅ Local inference (no cloud dependency)
- ❌ NO re-training from scratch
- 🟡 Fine-tuning optional (if FOREX slang needed)

### Alpha Vantage MCP (Optional)
**User Mention**: 
> "Alpha Vantage ofreciendo servidores MCP para entrenar FinGPT sin una DB"

**Our Approach**:
- Use Alpha Vantage for **data** (news, fundamentals)
- Use FinGPT for **inference** (local, pre-trained)
- TimescaleDB as **Feature Store** (not for LLM training)

### Data Injection Strategy
**User Architecture**:
1. **Streaming** (MT5): <100ms latency → Real-time execution
2. **Pooling 15min** (Sentiment): ~5-10s latency → DRL feature
3. **Pooling EOD** (Macro): Daily updates → Regime classification

**Result**: Optimal balance between latency and data freshness

---

## ✅ Phase 2 COMPLETE

**Status**: Feature Store architecture implemented  
**Next**: Week 1 testing + StateVectorBuilder  
**Timeline**: 3 weeks to DRL training  
**Business Gate**: 30-day paper trading → FTMO Phase 1

---

**Commit Message**:
```
feat: Feature Store (TimescaleDB + Redis + DataOrchestrator)

- Extended init-db.sql: sentiment_scores, macro_indicators, regime_predictions
- Implemented DataIngestionOrchestrator (400 LOC): 3 async pipelines
- Added Redis to Docker Compose (512MB cache, LRU eviction)
- FinGPT inference (local, pre-trained ProsusAI/finbert)
- Continuous aggregates: hourly_sentiment_summary
- Compression/retention policies optimized

Papers: AlphaQuanter.pdf (Feature Store), arXiv:2510.10526v1 (LLM+RL)
Next: Deploy Docker stack + test ingestion pipelines
```
