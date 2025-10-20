# 📊 Phase 3: Database Backfill & Data Quality Validation

## Objetivo

Poblar TimescaleDB con **2+ años de datos históricos de alta calidad** libres de sesgos (survivorship bias, look-ahead bias) para habilitar:
- WFO con múltiples folds (10+ folds requieren ~2 años)
- Monte Carlo con 5000+ simulaciones estadísticamente significativas
- ML Training con datasets robustos (train/val/test splits)

---

## 🎯 Criterios de Calidad de Datos (Según Literatura Científica)

### 1. **Survivorship Bias Free**
- ❌ **Incorrecto**: Descargar solo símbolos que existen HOY
- ✅ **Correcto**: Incluir símbolos delisted/quebrados en el periodo histórico

**Fuentes Recomendadas**:
- **Quandl/NASDAQ Data Link**: Ofrece datasets survivorship-bias-free
- **AlgoSeek**: Incluye símbolos delisted
- **FirstRate Data**: Forex/CFDs con histórico completo

### 2. **Timestamp Accuracy (Precisión de Timestamps)**
- ❌ **Incorrecto**: Usar timestamps de cierre de vela (introduce look-ahead bias)
- ✅ **Correcto**: Timestamps deben reflejar cuando la información estuvo DISPONIBLE

**Ejemplo**:
```python
# INCORRECTO: Bar de 09:00-09:05 con timestamp 09:05
# (Implica que conoce el precio de cierre a las 09:05)

# CORRECTO: Bar de 09:00-09:05 con timestamp 09:05:01
# (Solo puede actuar DESPUÉS del cierre de la vela)
```

### 3. **Corporate Actions Adjustment**
- Para equities (si se expande más allá de Forex):
  - Dividendos
  - Splits
  - Fusiones

### 4. **Data Gaps Handling**
- **Problema**: Fines de semana, festivos, halts de mercado
- **Solución**: Forward-fill con flag de "missing data"

---

## 📋 Plan de Implementación

### Fase 3.1: Data Ingestion Pipeline (Semana 1)

**Archivo**: `underdog/database/ingestion_pipeline.py` (expandir existente)

**Features**:
```python
class HistoricalDataIngestion:
    """Pipeline para ingestión masiva de datos históricos."""
    
    def __init__(self, db_connection, config):
        self.db = db_connection
        self.config = config
        self.validators = [
            SurvivorshipBiasValidator(),
            TimestampValidator(),
            GapDetector(),
            DuplicateDetector()
        ]
    
    def ingest_from_source(self, source: str, symbols: List[str], 
                           start_date: str, end_date: str):
        """
        Ingesta datos con validación multi-layer.
        
        Args:
            source: 'histdata', 'quandl', 'firstrate', 'mt5'
            symbols: ['EURUSD', 'GBPUSD', ...]
            start_date: '2020-01-01'
            end_date: '2024-12-31'
        """
        for symbol in symbols:
            logger.info(f"Ingesting {symbol} from {start_date} to {end_date}")
            
            # 1. Download
            raw_data = self._download_from_source(source, symbol, start_date, end_date)
            
            # 2. Validate
            validated_data = self._validate_data(raw_data, symbol)
            
            # 3. Transform
            transformed_data = self._transform_to_schema(validated_data)
            
            # 4. Load to TimescaleDB
            self._load_to_db(transformed_data, symbol)
            
            # 5. Log metadata
            self._log_ingestion_metadata(symbol, len(transformed_data))
    
    def _validate_data(self, df, symbol):
        """Ejecuta todos los validadores."""
        for validator in self.validators:
            issues = validator.validate(df, symbol)
            if issues:
                logger.warning(f"{symbol}: {validator.__class__.__name__} found {len(issues)} issues")
                # Decidir: skip, fix, alert
        return df
```

**Validators Críticos**:

```python
class SurvivorshipBiasValidator:
    """Detecta si faltan símbolos delisted en el periodo."""
    
    def validate(self, df, symbol):
        # Consultar fuente externa para verificar si símbolo estaba activo en todo el periodo
        pass

class TimestampValidator:
    """Valida que timestamps sean consistentes y no introduzcan look-ahead bias."""
    
    def validate(self, df, symbol):
        issues = []
        
        # Check 1: Timestamps deben ser monotónicamente crecientes
        if not df.index.is_monotonic_increasing:
            issues.append("Non-monotonic timestamps detected")
        
        # Check 2: No debe haber timestamps futuros
        if df.index.max() > pd.Timestamp.now():
            issues.append("Future timestamps detected")
        
        # Check 3: Intervalos entre bars deben ser consistentes
        intervals = df.index.to_series().diff()
        expected_interval = pd.Timedelta(minutes=5)  # Ejemplo para M5
        outliers = intervals[intervals > expected_interval * 2]
        if len(outliers) > 0:
            issues.append(f"Large gaps detected: {len(outliers)} intervals > {expected_interval * 2}")
        
        return issues

class GapDetector:
    """Detecta y rellena gaps de datos."""
    
    def validate(self, df, symbol):
        # Detectar gaps (ej: falta data de un día completo)
        # Opción 1: Forward-fill
        # Opción 2: Marcar como "missing" con flag
        pass
```

---

### Fase 3.2: Integration con Fuentes de Datos (Semana 1-2)

#### Opción A: HistData.com (Forex Tick Data - GRATIS)

**Ventajas**:
- ✅ Gratis para uso no comercial
- ✅ Tick data de calidad para major pairs (EURUSD, GBPUSD, etc.)
- ✅ Histórico hasta 2003

**Limitaciones**:
- ⚠️ Solo Forex (no equities/crypto)
- ⚠️ Descarga manual (no API oficial)

**Implementación**:
```python
import requests
from zipfile import ZipFile
import io

def download_histdata_month(symbol, year, month):
    """Descarga archivo .zip de HistData para un mes específico."""
    url = f"https://www.histdata.com/download-free-forex-historical-data/?/ascii/tick-data-quotes/{symbol}/{year}/{month}"
    
    response = requests.get(url)
    with ZipFile(io.BytesIO(response.content)) as zip_file:
        csv_name = zip_file.namelist()[0]
        df = pd.read_csv(
            zip_file.open(csv_name),
            names=['timestamp', 'bid', 'ask'],
            parse_dates=['timestamp']
        )
    
    # Convertir tick data a OHLCV (M1, M5, etc.)
    ohlcv = df.resample('5T', on='timestamp').agg({
        'bid': ['first', 'max', 'min', 'last'],
        'ask': ['first', 'max', 'min', 'last']
    })
    ohlcv['volume'] = df.resample('5T', on='timestamp').size()  # Tick count como proxy
    
    return ohlcv
```

#### Opción B: MT5 Historical Data (Si ya tiene cuenta)

**Ventajas**:
- ✅ Ya integrado en su proyecto (mt5_connector.py)
- ✅ Datos directos del broker (mismo que usará en producción)

**Limitaciones**:
- ⚠️ Histórico limitado (típicamente 1-2 años en cuentas demo)
- ⚠️ Puede tener survivorship bias

**Implementación**:
```python
# En mt5_connector.py - Añadir método de descarga masiva
def download_historical_range(self, symbol, timeframe, start_date, end_date):
    """Descarga histórico completo en chunks (MT5 limita a 100k bars por request)."""
    all_bars = []
    current = start_date
    
    while current < end_date:
        # MT5 limita requests, descargar en chunks de 100k bars
        bars = mt5.copy_rates_range(
            symbol,
            timeframe,
            current,
            min(current + pd.Timedelta(days=365), end_date)
        )
        
        if bars is None or len(bars) == 0:
            break
        
        all_bars.extend(bars)
        current = pd.Timestamp(bars[-1]['time'], unit='s') + pd.Timedelta(seconds=1)
        
        # Rate limiting
        time.sleep(0.5)
    
    return pd.DataFrame(all_bars)
```

#### Opción C: Quandl/NASDAQ Data Link (Pago, pero profesional)

**Ventajas**:
- ✅ Survivorship-bias-free datasets
- ✅ Corporate actions adjusted
- ✅ API oficial con rate limits generosos

**Costo**: ~$50-200/mes dependiendo del plan

**Implementación**:
```python
import quandl

quandl.ApiConfig.api_key = os.getenv('QUANDL_API_KEY')

def download_quandl_forex(symbol, start_date, end_date):
    """Descarga Forex data de Quandl."""
    dataset_code = f"FRED/{symbol}"  # Ej: FRED/DEXUSEU para EUR/USD
    df = quandl.get(dataset_code, start_date=start_date, end_date=end_date)
    return df
```

---

### Fase 3.3: Database Schema Validation (Semana 2)

**Verificar que init-db.sql esté optimizado**:

```sql
-- Añadir índices para queries frecuentes
CREATE INDEX idx_ohlcv_symbol_time ON ohlcv (symbol, time DESC);
CREATE INDEX idx_trades_strategy_time ON trades (strategy, time DESC);

-- Verificar que compression policies estén activas
SELECT * FROM timescaledb_information.compression_settings;

-- Verificar que retention policies estén activas
SELECT * FROM timescaledb_information.jobs;

-- Estadísticas de uso de espacio
SELECT 
    hypertable_name,
    pg_size_pretty(hypertable_size(hypertable_name::regclass)) as size,
    pg_size_pretty(total_bytes) as total_size_with_indexes
FROM timescaledb_information.hypertables;
```

---

### Fase 3.4: Data Quality Tests (Semana 2)

**Archivo**: `tests/test_data_quality.py`

```python
import pytest
from underdog.database.data_store import DataStore

def test_no_duplicate_timestamps(db_connection):
    """Verifica que no haya timestamps duplicados."""
    query = """
    SELECT symbol, time, COUNT(*) as cnt
    FROM ohlcv
    GROUP BY symbol, time
    HAVING COUNT(*) > 1
    """
    duplicates = db_connection.execute(query).fetchall()
    assert len(duplicates) == 0, f"Found {len(duplicates)} duplicate timestamps"

def test_no_gaps_in_data(db_connection, symbol='EURUSD', timeframe='5min'):
    """Verifica que no haya gaps mayores a 1 día (excepto fines de semana)."""
    query = f"""
    SELECT 
        time,
        LAG(time) OVER (ORDER BY time) as prev_time,
        time - LAG(time) OVER (ORDER BY time) as gap
    FROM ohlcv
    WHERE symbol = '{symbol}'
    ORDER BY time
    """
    df = pd.read_sql(query, db_connection)
    
    # Filtrar fines de semana
    df = df[df['time'].dt.dayofweek < 5]
    
    # Gaps mayores a 1 día son sospechosos
    large_gaps = df[df['gap'] > pd.Timedelta(days=1)]
    assert len(large_gaps) == 0, f"Found {len(large_gaps)} gaps > 1 day"

def test_price_sanity_checks(db_connection, symbol='EURUSD'):
    """Verifica que precios estén en rangos razonables."""
    query = f"""
    SELECT time, open, high, low, close
    FROM ohlcv
    WHERE symbol = '{symbol}'
      AND (
          high < low OR  -- High debe ser >= Low
          close > high OR  -- Close debe estar dentro del rango
          close < low OR
          open > high OR
          open < low
      )
    """
    invalid_bars = pd.read_sql(query, db_connection)
    assert len(invalid_bars) == 0, f"Found {len(invalid_bars)} bars with invalid OHLC relationships"

def test_sufficient_history(db_connection, min_days=730):
    """Verifica que haya al menos 2 años de datos."""
    query = """
    SELECT 
        symbol,
        MIN(time) as start_date,
        MAX(time) as end_date,
        EXTRACT(DAY FROM (MAX(time) - MIN(time))) as days
    FROM ohlcv
    GROUP BY symbol
    """
    df = pd.read_sql(query, db_connection)
    
    insufficient = df[df['days'] < min_days]
    assert len(insufficient) == 0, (
        f"Symbols with < {min_days} days of data: {insufficient['symbol'].tolist()}"
    )
```

---

### Fase 3.5: Backfill Execution Script (Semana 2)

**Archivo**: `scripts/backfill_historical_data.py`

```python
#!/usr/bin/env python3
"""
Script para poblar TimescaleDB con datos históricos.

Uso:
    python scripts/backfill_historical_data.py --source histdata --symbols EURUSD,GBPUSD --start 2020-01-01 --end 2024-12-31
"""

import argparse
from underdog.database.ingestion_pipeline import HistoricalDataIngestion
from underdog.database.data_store import DataStore
from underdog.utils.logging import setup_logger

logger = setup_logger("backfill")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', required=True, choices=['histdata', 'mt5', 'quandl'])
    parser.add_argument('--symbols', required=True, help='Comma-separated list (EURUSD,GBPUSD)')
    parser.add_argument('--start', required=True, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end', required=True, help='End date (YYYY-MM-DD)')
    parser.add_argument('--timeframe', default='5min', choices=['1min', '5min', '15min', '1h', '1d'])
    parser.add_argument('--validate', action='store_true', help='Run data quality tests after ingestion')
    
    args = parser.parse_args()
    
    symbols = args.symbols.split(',')
    
    logger.info(f"Starting backfill: {args.source} | {symbols} | {args.start} to {args.end}")
    
    # Conectar a DB
    db = DataStore()
    
    # Ejecutar ingestion
    pipeline = HistoricalDataIngestion(db.connection, config={})
    pipeline.ingest_from_source(
        source=args.source,
        symbols=symbols,
        start_date=args.start,
        end_date=args.end
    )
    
    logger.info("Backfill complete!")
    
    # Validar (opcional)
    if args.validate:
        logger.info("Running data quality tests...")
        import subprocess
        result = subprocess.run(['pytest', 'tests/test_data_quality.py', '-v'])
        if result.returncode != 0:
            logger.error("Data quality tests FAILED!")
            return 1
    
    logger.info("✅ Backfill successful and validated!")
    return 0

if __name__ == '__main__':
    exit(main())
```

**Ejecución**:
```bash
# Ejemplo 1: HistData (gratis, Forex)
python scripts/backfill_historical_data.py \
    --source histdata \
    --symbols EURUSD,GBPUSD,USDJPY \
    --start 2020-01-01 \
    --end 2024-12-31 \
    --validate

# Ejemplo 2: MT5 (si ya tiene cuenta)
python scripts/backfill_historical_data.py \
    --source mt5 \
    --symbols EURUSD \
    --start 2023-01-01 \
    --end 2024-12-31 \
    --timeframe 5min \
    --validate
```

---

## 📊 Success Criteria (Criterios de Éxito)

Antes de pasar a la Fase 4 (Production Deployment), validar:

### ✅ Checklist de Calidad de Datos

- [ ] **Volumen**: Al menos 2 años de datos para símbolos principales
- [ ] **Completitud**: < 1% de gaps (excluyendo fines de semana)
- [ ] **Validez**: 100% de bars pasan sanity checks (high >= low, etc.)
- [ ] **No Duplicados**: 0 timestamps duplicados
- [ ] **Survivorship Bias**: Validado que incluye delisted symbols (si aplica)
- [ ] **Timestamp Accuracy**: Timestamps reflejan disponibilidad de info (no look-ahead bias)

### ✅ Validación con WFO

Después del backfill, ejecutar WFO en el dataset completo:

```bash
python scripts/complete_trading_workflow.py --run-wfo --symbols EURUSD --folds 10
```

**Esperado**:
- Avg OOS Sharpe >= 0.5 (positivo y estadísticamente significativo)
- IS vs OOS degradation < 30% (ej: IS Sharpe 1.0, OOS Sharpe >= 0.7)

### ✅ Validación con Monte Carlo

```bash
python scripts/complete_trading_workflow.py --run-montecarlo --simulations 10000
```

**Esperado**:
- VaR (5%) < 10% del capital inicial
- CVaR (5%) < 15% del capital inicial
- P(Max DD > 20%) < 5%

---

## 🚨 Riesgos y Mitigaciones

| Riesgo | Probabilidad | Impacto | Mitigación |
|--------|-------------|---------|------------|
| Datos incompletos (gaps) | Alta | Alto | GapDetector + forward-fill + flag missing data |
| Survivorship bias | Media | Crítico | Usar fuente survivorship-bias-free (Quandl) |
| Look-ahead bias en timestamps | Media | Crítico | TimestampValidator + manual spot checks |
| Límites de rate limiting (APIs) | Alta | Bajo | Rate limiting + retry logic en ingestion_pipeline |
| Espacio en disco insuficiente | Baja | Medio | TimescaleDB compression (reduce 90%+ del espacio) |

---

## 📅 Timeline Estimado

| Fase | Duración | Entregable |
|------|----------|------------|
| 3.1: Ingestion Pipeline | 2-3 días | `ingestion_pipeline.py` con validators |
| 3.2: Integration con Fuentes | 2-3 días | Connectors para HistData/MT5/Quandl |
| 3.3: Schema Validation | 1 día | Queries de validación en init-db.sql |
| 3.4: Data Quality Tests | 1-2 días | `test_data_quality.py` |
| 3.5: Backfill Execution | 1 día | `backfill_historical_data.py` + ejecución |
| **Total** | **7-10 días** | **TimescaleDB poblada y validada** |

---

## 🎯 Siguiente Fase (Después del Backfill)

Una vez completada la Fase 3, el proyecto estará listo para:

### **Fase 4: Large-Scale WFO & Monte Carlo**
- Ejecutar WFO con 10+ folds en dataset completo (2+ años)
- Monte Carlo con 10,000+ simulaciones para estimaciones robustas de riesgo
- Optimización de parámetros a gran escala (grid search o Bayesian)

### **Fase 5: ML Model Training at Scale**
- Train HMM Regime Classifier en 2+ años de datos
- Feature Engineering en dataset completo
- MLflow tracking de múltiples experimentos
- Model selection basado en OOS performance

### **Fase 6: Production Deployment**
- Deploy to VPS (DigitalOcean/Vultr) con Docker
- Demo account testing por 1-2 meses
- Gradual transition to live account (si Prop Firm aprueba)

---

**Status**: 📋 **Phase 3 Plan READY**  
**Next Action**: Implementar `ingestion_pipeline.py` con validators  
**Goal**: TimescaleDB poblada con 2+ años de datos de alta calidad, libres de sesgos
