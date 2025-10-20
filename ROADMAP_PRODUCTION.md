# 🚀 UNDERDOG - ROADMAP TO PRODUCTION

**Fecha de inicio**: 2025-10-20  
**Estado actual**: Sistema funcional en desarrollo local  
**Objetivo**: Sistema completamente automatizado en servidor con cuenta demo MT5

---

## 📋 TABLA DE CONTENIDOS

1. [Resumen Ejecutivo](#resumen-ejecutivo)
2. [Fase 1: Limpieza y Reorganización](#fase-1-limpieza-y-reorganización)
3. [Fase 2: Optimización de Datos](#fase-2-optimización-de-datos)
4. [Fase 3: Integración MetaTrader 5](#fase-3-integración-metatrader-5)
5. [Fase 4: Configuraciones Finales](#fase-4-configuraciones-finales)
6. [Fase 5: Deployment a Servidor](#fase-5-deployment-a-servidor)
7. [Fase 6: Acceso Remoto y Mobile](#fase-6-acceso-remoto-y-mobile)
8. [Cronograma](#cronograma)
9. [Riesgos y Mitigaciones](#riesgos-y-mitigaciones)

---

## 📊 RESUMEN EJECUTIVO

### Objetivos Principales

| # | Objetivo | Prioridad | Complejidad | Tiempo Estimado |
|---|----------|-----------|-------------|-----------------|
| 1 | Limpieza de EAs no utilizados | 🔴 ALTA | ⭐ BAJA | 1-2 horas |
| 2 | Reorganización de documentación | 🟡 MEDIA | ⭐ BAJA | 30 min |
| 3 | Migración a datos Parquet + DuckDB | 🔴 ALTA | ⭐⭐⭐ ALTA | 4-6 horas |
| 4 | Integración con MetaTrader 5 | 🔴 CRÍTICA | ⭐⭐⭐⭐ MUY ALTA | 8-12 horas |
| 5 | Configuraciones finales | 🟡 MEDIA | ⭐⭐ MEDIA | 2-3 horas |
| 6 | Deployment a servidor | 🔴 CRÍTICA | ⭐⭐⭐ ALTA | 4-6 horas |
| 7 | Acceso remoto/mobile | 🟢 BAJA | ⭐⭐ MEDIA | 2-3 horas |

**Tiempo total estimado**: 22-33 horas (~3-5 días de trabajo)

---

## 🧹 FASE 1: LIMPIEZA Y REORGANIZACIÓN

### Prioridad: 🔴 ALTA | Duración: 1.5-2.5 horas

### 1.1 Eliminar EAs No Utilizados

**Objetivo**: Mantener solo los 7 EAs principales que están siendo monitoreados.

#### EAs Actuales en el Sistema

```python
# En: underdog/expert_advisors/__init__.py
ACTIVE_EAS = [
    'SuperTrendRSI',      # ✅ Mantener
    'ParabolicEMA',       # ✅ Mantener
    'KeltnerBreakout',    # ✅ Mantener
    'EmaScalper',         # ✅ Mantener
    'BollingerCCI',       # ✅ Mantener
    'ATRBreakout',        # ✅ Mantener
    'PairArbitrage',      # ✅ Mantener
]
```

#### Acciones

- [ ] **1.1.1** Listar todos los archivos EA en `underdog/expert_advisors/`
  ```bash
  ls underdog/expert_advisors/*.py
  ```

- [ ] **1.1.2** Identificar EAs no utilizados (no en la lista de ACTIVE_EAS)

- [ ] **1.1.3** Crear carpeta de archivo
  ```bash
  mkdir underdog/expert_advisors/_archived
  ```

- [ ] **1.1.4** Mover EAs no utilizados a `_archived/`

- [ ] **1.1.5** Actualizar imports en `__init__.py`

- [ ] **1.1.6** Ejecutar tests para verificar que no se rompió nada
  ```bash
  poetry run pytest tests/test_expert_advisors.py -v
  ```

- [ ] **1.1.7** Commit cambios
  ```bash
  git commit -m "refactor: Archive unused EAs, keep only 7 active strategies"
  ```

#### Criterios de Éxito
- ✅ Solo 7 archivos EA activos en el directorio principal
- ✅ EAs archivados movidos a `_archived/`
- ✅ Todos los tests pasan
- ✅ Sistema funciona con los 7 EAs

---

### 1.2 Reorganizar Documentación

**Objetivo**: Mover todos los archivos `.md` de raíz a `docs/`

#### Archivos a Mover

```bash
# Archivos en la raíz que deben ir a docs/
CHECKLIST_STARTUP.md
DEMO_GUIDE.md
DEMO_STATUS.md
ESTADO_ACTUAL.md
FIREWALL_AUDIT_FINAL_SUMMARY.md
FIREWALL_AUDIT_INDEX.md
FIREWALL_AUDIT_SUMMARY.md
FIREWALL_AUDIT.md
FIREWALL_SETUP_COMPLETE.md
GRAFANA_DASHBOARDS_FIX.md
SOLUCION_MANUAL.md
TESTING_COMPLETE.md
# ... otros archivos .md
```

#### Acciones

- [ ] **1.2.1** Crear subcarpetas en `docs/` si es necesario
  ```bash
  mkdir docs/setup
  mkdir docs/troubleshooting
  mkdir docs/monitoring
  ```

- [ ] **1.2.2** Categorizar archivos por tema
  - Setup: `CHECKLIST_STARTUP.md`, `DEMO_GUIDE.md`
  - Troubleshooting: `FIREWALL_AUDIT*.md`, `SOLUCION_MANUAL.md`, `GRAFANA_DASHBOARDS_FIX.md`
  - Status: `DEMO_STATUS.md`, `ESTADO_ACTUAL.md`, `TESTING_COMPLETE.md`

- [ ] **1.2.3** Mover archivos
  ```bash
  git mv CHECKLIST_STARTUP.md docs/setup/
  git mv FIREWALL_AUDIT*.md docs/troubleshooting/
  # ... etc
  ```

- [ ] **1.2.4** Crear índice principal en `docs/README.md`

- [ ] **1.2.5** Actualizar referencias en otros archivos

- [ ] **1.2.6** Actualizar README.md principal con nueva estructura

- [ ] **1.2.7** Commit cambios
  ```bash
  git commit -m "docs: Reorganize documentation into docs/ subfolders"
  ```

#### Criterios de Éxito
- ✅ Raíz del proyecto limpia (solo README.md, pyproject.toml, etc.)
- ✅ Documentación organizada por categorías en `docs/`
- ✅ Índice actualizado y navegable
- ✅ Enlaces internos funcionando

---

## 💾 FASE 2: OPTIMIZACIÓN DE DATOS

### Prioridad: 🔴 ALTA | Duración: 4-6 horas

### 2.1 Migración a FX-1-Minute-Data (GitHub Repo)

**Objetivo**: Importar datos históricos desde el repositorio de philipperemy en lugar de Histdata.

#### Investigación Previa

**Repositorio**: https://github.com/philipperemy/FX-1-Minute-Data

**Características**:
- Datos de 1996-2023 (28 años)
- 28 pares de divisas
- Formato CSV con compresión
- Datos validados y limpios

#### Acciones

- [ ] **2.1.1** Clonar repositorio localmente
  ```bash
  cd data/
  git clone https://github.com/philipperemy/FX-1-Minute-Data.git
  ```

- [ ] **2.1.2** Analizar estructura de datos del repo
  ```python
  # Script: scripts/analyze_fx_data_structure.py
  import os
  import pandas as pd
  
  def analyze_repo_structure():
      # Explorar estructura de carpetas
      # Identificar formato de archivos
      # Determinar esquema de columnas
      pass
  ```

- [ ] **2.1.3** Crear script de importación `scripts/import_fx_minute_data.py`
  ```python
  """
  Import FX-1-Minute-Data from philipperemy's repo to TimescaleDB
  """
  import pandas as pd
  from pathlib import Path
  from underdog.data.timescale_client import TimescaleClient
  
  def import_pair(pair: str, year_start: int, year_end: int):
      # Leer CSVs del repo
      # Convertir a formato UNDERDOG
      # Insertar en TimescaleDB
      pass
  ```

- [ ] **2.1.4** Implementar importación por lotes (chunked)
  - Evitar cargar todos los datos en RAM
  - Procesar por año o por mes

- [ ] **2.1.5** Agregar barra de progreso
  ```python
  from tqdm import tqdm
  
  for pair in tqdm(PAIRS, desc="Importing pairs"):
      import_pair(pair)
  ```

- [ ] **2.1.6** Ejecutar importación completa
  ```bash
  poetry run python scripts/import_fx_minute_data.py \
    --pairs EURUSD,GBPUSD,USDJPY \
    --years 2020-2023
  ```

- [ ] **2.1.7** Verificar datos importados en TimescaleDB
  ```sql
  SELECT pair, MIN(timestamp), MAX(timestamp), COUNT(*) 
  FROM ohlcv_data 
  GROUP BY pair;
  ```

#### Criterios de Éxito
- ✅ Todos los pares principales importados (EURUSD, GBPUSD, USDJPY, etc.)
- ✅ Datos desde 2020 hasta presente
- ✅ Sin gaps significativos en los datos
- ✅ Queries de prueba funcionan correctamente

---

### 2.2 Conversión a Formato Parquet

**Objetivo**: Almacenar datos en Parquet para reducir tamaño y mejorar velocidad.

#### Comparación de Formatos

| Formato | Tamaño (1 año EURUSD) | Velocidad Lectura | Compresión |
|---------|----------------------|-------------------|------------|
| CSV | ~500 MB | 🐢 Lenta | ❌ No |
| CSV.GZ | ~50 MB | 🐢 Muy lenta | ✅ Sí (pero lenta) |
| Parquet | ~20 MB | ⚡ Rápida | ✅ Sí (nativa) |

#### Acciones

- [ ] **2.2.1** Instalar dependencias
  ```bash
  poetry add pyarrow fastparquet
  ```

- [ ] **2.2.2** Crear script de conversión `scripts/convert_to_parquet.py`
  ```python
  import pandas as pd
  import pyarrow as pa
  import pyarrow.parquet as pq
  
  def csv_to_parquet(csv_path: Path, parquet_path: Path):
      # Leer CSV en chunks
      # Escribir directamente a Parquet con compresión
      df = pd.read_csv(csv_path)
      df.to_parquet(
          parquet_path,
          engine='pyarrow',
          compression='snappy',  # O 'zstd' para mejor compresión
          index=False
      )
  ```

- [ ] **2.2.3** Organizar estructura de carpetas
  ```
  data/
    raw/
      parquet/
        EURUSD/
          2020.parquet
          2021.parquet
          2022.parquet
        GBPUSD/
          ...
  ```

- [ ] **2.2.4** Convertir todos los datos históricos

- [ ] **2.2.5** Crear función de lectura optimizada
  ```python
  def read_parquet_range(pair: str, start_date: str, end_date: str):
      # Leer solo las columnas necesarias
      # Filtrar por rango de fechas
      return pd.read_parquet(
          f"data/raw/parquet/{pair}/",
          columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'],
          filters=[('timestamp', '>=', start_date), ('timestamp', '<=', end_date)]
      )
  ```

- [ ] **2.2.6** Actualizar pipelines de datos para usar Parquet

- [ ] **2.2.7** Benchmarking
  ```python
  # Comparar velocidad CSV vs Parquet
  import time
  
  start = time.time()
  df_csv = pd.read_csv("data.csv")
  print(f"CSV: {time.time() - start:.2f}s")
  
  start = time.time()
  df_parquet = pd.read_parquet("data.parquet")
  print(f"Parquet: {time.time() - start:.2f}s")
  ```

#### Criterios de Éxito
- ✅ Tamaño de datos reducido en ~70-80%
- ✅ Velocidad de lectura 5-10x más rápida
- ✅ Todos los scripts actualizados para usar Parquet
- ✅ Benchmarks documentados

---

### 2.3 Integración con DuckDB

**Objetivo**: Usar DuckDB como capa de query para datos Parquet, eliminando CSVs en Streamlit.

#### Ventajas de DuckDB

- ✅ Queries SQL sobre Parquet directamente (sin cargar en RAM)
- ✅ 10-100x más rápido que Pandas para agregaciones
- ✅ Soporte nativo para Parquet
- ✅ Compatible con Pandas (puede devolver DataFrames)
- ✅ Embebido (no necesita servidor separado)

#### Acciones

- [ ] **2.3.1** Instalar DuckDB
  ```bash
  poetry add duckdb
  ```

- [ ] **2.3.2** Crear módulo `underdog/data/duckdb_client.py`
  ```python
  """
  DuckDB client for fast analytical queries on Parquet data
  """
  import duckdb
  from pathlib import Path
  from typing import Optional
  import pandas as pd
  
  class DuckDBClient:
      def __init__(self, db_path: str = ":memory:"):
          self.conn = duckdb.connect(db_path)
          
      def register_parquet_folder(self, alias: str, folder: Path):
          """Register a folder of Parquet files as a table"""
          self.conn.execute(f"""
              CREATE VIEW {alias} AS 
              SELECT * FROM read_parquet('{folder}/*.parquet')
          """)
      
      def query(self, sql: str) -> pd.DataFrame:
          """Execute SQL query and return DataFrame"""
          return self.conn.execute(sql).fetchdf()
      
      def get_ohlcv(
          self, 
          pair: str, 
          start: str, 
          end: str, 
          timeframe: str = '1m'
      ) -> pd.DataFrame:
          """Get OHLCV data for backtesting"""
          return self.query(f"""
              SELECT * FROM ohlcv_{pair}
              WHERE timestamp >= '{start}'
                AND timestamp <= '{end}'
              ORDER BY timestamp
          """)
  ```

- [ ] **2.3.3** Actualizar Streamlit para usar DuckDB
  ```python
  # En: underdog/ui/streamlit_backtest.py
  
  from underdog.data.duckdb_client import DuckDBClient
  
  # Reemplazar:
  # df = pd.read_csv(f"data/raw/{pair}.csv")
  
  # Por:
  db = DuckDBClient()
  db.register_parquet_folder("ohlcv_EURUSD", Path("data/raw/parquet/EURUSD"))
  df = db.get_ohlcv("EURUSD", start_date, end_date)
  ```

- [ ] **2.3.4** Implementar cache de queries frecuentes
  ```python
  from functools import lru_cache
  
  @lru_cache(maxsize=100)
  def get_cached_ohlcv(pair: str, start: str, end: str):
      return db.get_ohlcv(pair, start, end)
  ```

- [ ] **2.3.5** Crear queries optimizadas para análisis
  ```python
  def get_daily_statistics(pair: str, start: str, end: str):
      """Agregación diaria ultra rápida con DuckDB"""
      return db.query(f"""
          SELECT 
              DATE_TRUNC('day', timestamp) as date,
              FIRST(open) as open,
              MAX(high) as high,
              MIN(low) as low,
              LAST(close) as close,
              SUM(volume) as volume
          FROM ohlcv_{pair}
          WHERE timestamp >= '{start}' AND timestamp <= '{end}'
          GROUP BY date
          ORDER BY date
      """)
  ```

- [ ] **2.3.6** Benchmarking DuckDB vs Pandas
  ```python
  # Comparar velocidad para queries complejas
  # Aggregations, filters, joins, etc.
  ```

- [ ] **2.3.7** Actualizar tests
  ```python
  # tests/test_duckdb_client.py
  def test_query_ohlcv():
      db = DuckDBClient()
      df = db.get_ohlcv("EURUSD", "2023-01-01", "2023-01-31")
      assert len(df) > 0
      assert df['close'].dtype == float
  ```

#### Criterios de Éxito
- ✅ Streamlit carga datos 10x más rápido
- ✅ No hay CSVs cargados en RAM
- ✅ Queries complejas <1 segundo
- ✅ UI responsiva incluso con años de datos

---

## 🔌 FASE 3: INTEGRACIÓN METATRADER 5

### Prioridad: 🔴 CRÍTICA | Duración: 8-12 horas

### 3.1 Setup de Conexión MT5

**Objetivo**: Conectar Python con MetaTrader 5 para trading en tiempo real.

#### Requisitos Previos

- [ ] MetaTrader 5 instalado
- [ ] Cuenta demo configurada
- [ ] Terminal MT5 abierto (debe estar ejecutándose)

#### Acciones

- [ ] **3.1.1** Instalar librería MT5
  ```bash
  poetry add MetaTrader5
  ```

- [ ] **3.1.2** Crear módulo `underdog/broker/mt5_connector.py`
  ```python
  """
  MetaTrader 5 Connector for real-time trading
  """
  import MetaTrader5 as mt5
  from datetime import datetime
  import pandas as pd
  from typing import Optional, Dict, List
  
  class MT5Connector:
      def __init__(self):
          self.connected = False
          
      def connect(self, login: int, password: str, server: str) -> bool:
          """Initialize MT5 connection"""
          if not mt5.initialize():
              print(f"MT5 initialize() failed: {mt5.last_error()}")
              return False
          
          # Login
          if not mt5.login(login, password, server):
              print(f"Login failed: {mt5.last_error()}")
              return False
          
          self.connected = True
          print(f"✅ Connected to MT5: {mt5.account_info()}")
          return True
      
      def disconnect(self):
          """Shutdown MT5 connection"""
          mt5.shutdown()
          self.connected = False
      
      def get_account_info(self) -> Dict:
          """Get account information"""
          info = mt5.account_info()
          return {
              'balance': info.balance,
              'equity': info.equity,
              'margin': info.margin,
              'margin_free': info.margin_free,
              'profit': info.profit
          }
      
      def get_tick(self, symbol: str) -> Dict:
          """Get latest tick for symbol"""
          tick = mt5.symbol_info_tick(symbol)
          return {
              'symbol': symbol,
              'bid': tick.bid,
              'ask': tick.ask,
              'time': datetime.fromtimestamp(tick.time)
          }
      
      def place_order(
          self, 
          symbol: str, 
          order_type: str,  # 'BUY' or 'SELL'
          volume: float,
          sl: Optional[float] = None,
          tp: Optional[float] = None
      ) -> bool:
          """Place a market order"""
          # Preparar request
          request = {
              "action": mt5.TRADE_ACTION_DEAL,
              "symbol": symbol,
              "volume": volume,
              "type": mt5.ORDER_TYPE_BUY if order_type == 'BUY' else mt5.ORDER_TYPE_SELL,
              "deviation": 20,
              "magic": 234000,
              "comment": "UNDERDOG EA",
              "type_time": mt5.ORDER_TIME_GTC,
              "type_filling": mt5.ORDER_FILLING_IOC,
          }
          
          # Agregar SL/TP si están definidos
          if sl:
              request["sl"] = sl
          if tp:
              request["tp"] = tp
          
          # Enviar orden
          result = mt5.order_send(request)
          
          if result.retcode != mt5.TRADE_RETCODE_DONE:
              print(f"❌ Order failed: {result.comment}")
              return False
          
          print(f"✅ Order placed: {result.order}")
          return True
      
      def get_positions(self) -> List[Dict]:
          """Get all open positions"""
          positions = mt5.positions_get()
          return [
              {
                  'ticket': pos.ticket,
                  'symbol': pos.symbol,
                  'type': 'BUY' if pos.type == 0 else 'SELL',
                  'volume': pos.volume,
                  'price_open': pos.price_open,
                  'price_current': pos.price_current,
                  'profit': pos.profit,
                  'sl': pos.sl,
                  'tp': pos.tp
              }
              for pos in positions
          ]
  ```

- [ ] **3.1.3** Crear script de testing de conexión
  ```python
  # scripts/test_mt5_connection.py
  from underdog.broker.mt5_connector import MT5Connector
  import os
  
  def main():
      mt5 = MT5Connector()
      
      # Credenciales desde variables de entorno
      login = int(os.getenv("MT5_LOGIN"))
      password = os.getenv("MT5_PASSWORD")
      server = os.getenv("MT5_SERVER")
      
      if mt5.connect(login, password, server):
          print("\n✅ Conexión exitosa!")
          print(f"Account Info: {mt5.get_account_info()}")
          print(f"EURUSD Tick: {mt5.get_tick('EURUSD')}")
          print(f"Open Positions: {mt5.get_positions()}")
          mt5.disconnect()
      else:
          print("\n❌ Conexión fallida")
  
  if __name__ == "__main__":
      main()
  ```

- [ ] **3.1.4** Agregar credenciales MT5 a `.env`
  ```bash
  # .env
  MT5_LOGIN=12345678
  MT5_PASSWORD=tu_password
  MT5_SERVER=demo.server.com
  ```

- [ ] **3.1.5** Ejecutar test de conexión
  ```bash
  poetry run python scripts/test_mt5_connection.py
  ```

#### Criterios de Éxito
- ✅ Conexión a MT5 establecida
- ✅ Información de cuenta accesible
- ✅ Datos de ticks en tiempo real
- ✅ Posiciones abiertas listables

---

### 3.2 Streaming de Datos en Tiempo Real

**Objetivo**: Recibir ticks de MT5 y alimentar sistema de señales.

#### Acciones

- [ ] **3.2.1** Crear módulo `underdog/broker/mt5_streamer.py`
  ```python
  """
  Real-time tick streaming from MT5
  """
  import time
  from threading import Thread
  from queue import Queue
  from underdog.broker.mt5_connector import MT5Connector
  
  class MT5Streamer:
      def __init__(self, symbols: List[str]):
          self.symbols = symbols
          self.mt5 = MT5Connector()
          self.tick_queue = Queue()
          self.running = False
          
      def start(self):
          """Start streaming ticks"""
          self.running = True
          self.thread = Thread(target=self._stream_loop, daemon=True)
          self.thread.start()
          print(f"✅ Streaming started for {self.symbols}")
      
      def stop(self):
          """Stop streaming"""
          self.running = False
          self.thread.join()
          print("⏹️  Streaming stopped")
      
      def _stream_loop(self):
          """Internal loop to fetch ticks"""
          while self.running:
              for symbol in self.symbols:
                  tick = self.mt5.get_tick(symbol)
                  self.tick_queue.put(tick)
              time.sleep(0.1)  # 10 ticks por segundo
      
      def get_latest_tick(self, symbol: str) -> Optional[Dict]:
          """Get latest tick from queue"""
          # Obtener todos los ticks del queue hasta encontrar el símbolo
          latest = None
          while not self.tick_queue.empty():
              tick = self.tick_queue.get()
              if tick['symbol'] == symbol:
                  latest = tick
          return latest
  ```

- [ ] **3.2.2** Integrar streamer con sistema de señales
  ```python
  # En: underdog/trading/live_trading_engine.py
  
  from underdog.broker.mt5_streamer import MT5Streamer
  from underdog.expert_advisors import get_all_eas
  
  class LiveTradingEngine:
      def __init__(self):
          self.streamer = MT5Streamer(['EURUSD', 'GBPUSD', 'USDJPY'])
          self.eas = get_all_eas()
          
      def run(self):
          """Main trading loop"""
          self.streamer.start()
          
          while True:
              for symbol in self.streamer.symbols:
                  tick = self.streamer.get_latest_tick(symbol)
                  if tick:
                      # Evaluar EAs con el tick
                      for ea in self.eas:
                          signal = ea.generate_signal(tick)
                          if signal:
                              self.process_signal(signal)
              
              time.sleep(1)  # Evaluar cada segundo
  ```

- [ ] **3.2.3** Crear buffer de ticks para indicadores
  ```python
  from collections import deque
  
  class TickBuffer:
      def __init__(self, maxlen: int = 1000):
          self.buffer = deque(maxlen=maxlen)
      
      def add_tick(self, tick: Dict):
          self.buffer.append(tick)
      
      def to_dataframe(self) -> pd.DataFrame:
          """Convert buffer to DataFrame for indicators"""
          return pd.DataFrame(list(self.buffer))
  ```

#### Criterios de Éxito
- ✅ Ticks recibidos en tiempo real
- ✅ Latencia <100ms
- ✅ EAs reciben datos actualizados
- ✅ Sin memory leaks en streaming

---

### 3.3 Ejecución de Órdenes

**Objetivo**: Enviar señales de EAs a MT5 como órdenes reales.

#### Acciones

- [ ] **3.3.1** Crear módulo `underdog/broker/order_manager.py`
  ```python
  """
  Order management and execution
  """
  from underdog.broker.mt5_connector import MT5Connector
  from underdog.core.signal import Signal
  
  class OrderManager:
      def __init__(self, mt5: MT5Connector):
          self.mt5 = mt5
          self.pending_orders = []
          self.executed_orders = []
          
      def process_signal(self, signal: Signal) -> bool:
          """Convert signal to MT5 order"""
          # Validar señal
          if not self.validate_signal(signal):
              return False
          
          # Calcular tamaño de posición
          volume = self.calculate_position_size(signal)
          
          # Calcular SL/TP
          sl = self.calculate_stop_loss(signal)
          tp = self.calculate_take_profit(signal)
          
          # Ejecutar orden
          success = self.mt5.place_order(
              symbol=signal.symbol,
              order_type=signal.direction,  # 'BUY' or 'SELL'
              volume=volume,
              sl=sl,
              tp=tp
          )
          
          if success:
              self.executed_orders.append(signal)
              print(f"✅ Order executed: {signal}")
          else:
              print(f"❌ Order failed: {signal}")
          
          return success
      
      def validate_signal(self, signal: Signal) -> bool:
          """Validate signal before execution"""
          # Check confidence threshold
          if signal.confidence < 0.7:
              return False
          
          # Check if symbol is tradeable
          # Check market hours
          # Check account balance
          # etc.
          
          return True
      
      def calculate_position_size(self, signal: Signal) -> float:
          """Calculate position size based on risk management"""
          account_info = self.mt5.get_account_info()
          balance = account_info['balance']
          
          # Risk 1% per trade
          risk_amount = balance * 0.01
          
          # Calculate volume based on SL distance
          # ...
          
          return 0.01  # Placeholder
  ```

- [ ] **3.3.2** Implementar risk management
  ```python
  class RiskManager:
      def __init__(self, max_risk_per_trade: float = 0.01):
          self.max_risk_per_trade = max_risk_per_trade
      
      def calculate_position_size(
          self, 
          balance: float, 
          entry_price: float, 
          stop_loss: float
      ) -> float:
          """Calculate position size based on risk"""
          risk_amount = balance * self.max_risk_per_trade
          distance = abs(entry_price - stop_loss)
          return risk_amount / distance
  ```

- [ ] **3.3.3** Implementar logging de órdenes
  ```python
  import logging
  
  order_logger = logging.getLogger("underdog.orders")
  order_logger.info(f"Order placed: {order_details}")
  ```

- [ ] **3.3.4** Guardar órdenes en TimescaleDB
  ```python
  def save_order_to_db(order: Dict):
      # INSERT INTO orders (timestamp, symbol, type, volume, ...)
      pass
  ```

#### Criterios de Éxito
- ✅ Señales convertidas a órdenes MT5
- ✅ Risk management funcionando
- ✅ Todas las órdenes loggeadas
- ✅ Órdenes guardadas en base de datos

---

### 3.4 Monitoreo de Posiciones

**Objetivo**: Trackear posiciones abiertas y cerrarlas según lógica de EAs.

#### Acciones

- [ ] **3.4.1** Crear módulo `underdog/broker/position_monitor.py`
  ```python
  """
  Monitor open positions and manage exits
  """
  class PositionMonitor:
      def __init__(self, mt5: MT5Connector):
          self.mt5 = mt5
          
      def monitor_positions(self):
          """Check all open positions"""
          positions = self.mt5.get_positions()
          
          for pos in positions:
              # Check if SL/TP hit
              # Check trailing stop
              # Check time-based exit
              # Check EA signals for exit
              
              if self.should_close_position(pos):
                  self.close_position(pos['ticket'])
      
      def close_position(self, ticket: int):
          """Close a position"""
          # Use MT5 API to close
          pass
  ```

- [ ] **3.4.2** Implementar trailing stop
  ```python
  def update_trailing_stop(position: Dict, current_price: float):
      # Si profit >= X pips, mover SL a breakeven
      # Si profit >= Y pips, mover SL trailing
      pass
  ```

- [ ] **3.4.3** Integrar con Prometheus metrics
  ```python
  from prometheus_client import Gauge
  
  open_positions = Gauge('underdog_open_positions', 'Number of open positions')
  position_profit = Gauge('underdog_position_profit', 'Current position profit')
  ```

#### Criterios de Éxito
- ✅ Posiciones monitoreadas en tiempo real
- ✅ Exits automáticos funcionando
- ✅ Trailing stops activos
- ✅ Métricas en Prometheus

---

### 3.5 Sincronización con Grafana

**Objetivo**: Ver trading en tiempo real en dashboards de Grafana.

#### Acciones

- [ ] **3.5.1** Extender métricas de Prometheus
  ```python
  # En: underdog/monitoring/prometheus_metrics.py
  
  mt5_connection_status = Gauge('underdog_mt5_connection', 'MT5 connection status')
  mt5_account_balance = Gauge('underdog_mt5_balance', 'MT5 account balance')
  mt5_account_equity = Gauge('underdog_mt5_equity', 'MT5 account equity')
  mt5_open_positions_count = Gauge('underdog_mt5_positions', 'Number of open positions')
  mt5_daily_profit = Gauge('underdog_mt5_daily_profit', 'Daily profit/loss')
  ```

- [ ] **3.5.2** Actualizar métricas cada segundo
  ```python
  def update_mt5_metrics():
      while True:
          if mt5.connected:
              info = mt5.get_account_info()
              mt5_account_balance.set(info['balance'])
              mt5_account_equity.set(info['equity'])
              
              positions = mt5.get_positions()
              mt5_open_positions_count.set(len(positions))
          
          time.sleep(1)
  ```

- [ ] **3.5.3** Crear dashboard "Live Trading" en Grafana
  ```json
  {
    "title": "UNDERDOG - Live Trading MT5",
    "panels": [
      {
        "title": "Account Balance",
        "targets": [{
          "expr": "underdog_mt5_balance"
        }]
      },
      {
        "title": "Open Positions",
        "targets": [{
          "expr": "underdog_mt5_positions"
        }]
      }
    ]
  }
  ```

- [ ] **3.5.4** Agregar alertas en Grafana
  ```yaml
  # Alerta si drawdown > 5%
  # Alerta si conexión MT5 se pierde
  # Alerta si equity < balance - X%
  ```

#### Criterios de Éxito
- ✅ Dashboard "Live Trading" funcional
- ✅ Métricas actualizándose cada segundo
- ✅ Alertas configuradas
- ✅ Histórico de trades visible

---

## ⚙️ FASE 4: CONFIGURACIONES FINALES

### Prioridad: 🟡 MEDIA | Duración: 2-3 horas

### 4.1 Variables de Entorno

- [ ] **4.1.1** Consolidar todas las variables en `.env`
  ```bash
  # Database
  DB_HOST=timescaledb
  DB_PORT=5432
  DB_USER=underdog
  DB_PASSWORD=strong_password
  DB_NAME=underdog_trading
  
  # Grafana
  GRAFANA_USER=admin
  GRAFANA_PASSWORD=admin123
  
  # MetaTrader 5
  MT5_LOGIN=12345678
  MT5_PASSWORD=demo_password
  MT5_SERVER=demo.icmarkets.com
  
  # Risk Management
  MAX_RISK_PER_TRADE=0.01
  MAX_DAILY_LOSS=0.05
  MAX_OPEN_POSITIONS=5
  ```

- [ ] **4.1.2** Crear `.env.example` para documentación

- [ ] **4.1.3** Validar que todas las variables se carguen correctamente

---

### 4.2 Logging Centralizado

- [ ] **4.2.1** Configurar logging estructurado
  ```python
  # underdog/core/logging_config.py
  import logging
  import json
  from datetime import datetime
  
  class JSONFormatter(logging.Formatter):
      def format(self, record):
          return json.dumps({
              'timestamp': datetime.utcnow().isoformat(),
              'level': record.levelname,
              'logger': record.name,
              'message': record.getMessage(),
              'module': record.module,
              'function': record.funcName
          })
  ```

- [ ] **4.2.2** Guardar logs en archivos rotatorios
  ```python
  from logging.handlers import RotatingFileHandler
  
  handler = RotatingFileHandler(
      'logs/underdog.log',
      maxBytes=10*1024*1024,  # 10 MB
      backupCount=5
  )
  ```

---

### 4.3 Health Checks

- [ ] **4.3.1** Crear endpoint de health en FastAPI
  ```python
  @app.get("/health")
  def health_check():
      return {
          "status": "healthy",
          "mt5_connected": mt5.connected,
          "db_connected": db.is_connected(),
          "prometheus_up": True,
          "grafana_up": True
      }
  ```

---

## 🚀 FASE 5: DEPLOYMENT A SERVIDOR

### Prioridad: 🔴 CRÍTICA | Duración: 4-6 horas

### 5.1 Preparación del Servidor

- [ ] **5.1.1** Elegir proveedor de servidor
  - DigitalOcean (Droplet $6/mes)
  - AWS EC2 (t3.micro free tier)
  - Hetzner (€4/mes)
  - VPS local

- [ ] **5.1.2** Requisitos del servidor
  - Ubuntu 22.04 LTS
  - 2 GB RAM mínimo (4 GB recomendado)
  - 20 GB SSD
  - Docker instalado
  - Docker Compose instalado

- [ ] **5.1.3** Setup inicial del servidor
  ```bash
  # Conectar via SSH
  ssh root@tu_servidor_ip
  
  # Actualizar sistema
  apt update && apt upgrade -y
  
  # Instalar Docker
  curl -fsSL https://get.docker.com -o get-docker.sh
  sh get-docker.sh
  
  # Instalar Docker Compose
  apt install docker-compose -y
  
  # Crear usuario para la app
  adduser underdog
  usermod -aG docker underdog
  ```

---

### 5.2 Deployment con Docker

- [ ] **5.2.1** Crear `docker-compose.prod.yml`
  ```yaml
  version: '3.8'
  
  services:
    underdog:
      image: underdog-trading:latest
      restart: always
      environment:
        - ENV=production
      volumes:
        - ./data:/app/data
        - ./logs:/app/logs
      networks:
        - underdog-net
    
    # ... resto de servicios
  ```

- [ ] **5.2.2** Configurar Nginx como reverse proxy
  ```nginx
  server {
      listen 80;
      server_name tu_dominio.com;
      
      location /grafana/ {
          proxy_pass http://localhost:3000/;
      }
      
      location /api/ {
          proxy_pass http://localhost:8000/;
      }
  }
  ```

- [ ] **5.2.3** Setup SSL con Let's Encrypt
  ```bash
  apt install certbot python3-certbot-nginx
  certbot --nginx -d tu_dominio.com
  ```

- [ ] **5.2.4** Configurar firewall
  ```bash
  ufw allow 22/tcp   # SSH
  ufw allow 80/tcp   # HTTP
  ufw allow 443/tcp  # HTTPS
  ufw enable
  ```

---

### 5.3 CI/CD Pipeline

- [ ] **5.3.1** Crear GitHub Actions workflow
  ```yaml
  # .github/workflows/deploy.yml
  name: Deploy to Production
  
  on:
    push:
      branches: [main]
  
  jobs:
    deploy:
      runs-on: ubuntu-latest
      steps:
        - uses: actions/checkout@v3
        
        - name: Build Docker image
          run: docker build -t underdog-trading:latest .
        
        - name: Push to registry
          run: |
            echo ${{ secrets.DOCKER_PASSWORD }} | docker login -u ${{ secrets.DOCKER_USERNAME }} --password-stdin
            docker push underdog-trading:latest
        
        - name: Deploy to server
          uses: appleboy/ssh-action@master
          with:
            host: ${{ secrets.SERVER_IP }}
            username: underdog
            key: ${{ secrets.SSH_KEY }}
            script: |
              cd /home/underdog/UNDERDOG
              docker-compose -f docker-compose.prod.yml pull
              docker-compose -f docker-compose.prod.yml up -d
  ```

---

## 📱 FASE 6: ACCESO REMOTO Y MOBILE

### Prioridad: 🟢 BAJA | Duración: 2-3 horas

### 6.1 Configurar IP Pública

- [ ] **6.1.1** Obtener IP pública del servidor
  ```bash
  curl ifconfig.me
  ```

- [ ] **6.1.2** Configurar dominio (opcional)
  - Registrar dominio en Namecheap, GoDaddy, etc.
  - Configurar DNS A record apuntando a la IP

- [ ] **6.1.3** Actualizar configuraciones para usar IP pública
  ```yaml
  # docker-compose.prod.yml
  environment:
    - GF_SERVER_ROOT_URL=http://tu_ip:3000
  ```

---

### 6.2 Acceso Mobile

- [ ] **6.2.1** Instalar app de Grafana en móvil
  - iOS: https://apps.apple.com/app/grafana/id1463211246
  - Android: https://play.google.com/store/apps/details?id=com.grafana.mobile

- [ ] **6.2.2** Configurar conexión en app móvil
  ```
  URL: http://tu_ip_publica:3000
  User: admin
  Password: tu_password
  ```

- [ ] **6.2.3** Crear dashboards optimizados para móvil
  - Paneles más grandes
  - Menos información por pantalla
  - Touch-friendly

---

## 📅 CRONOGRAMA

### Semana 1 (Días 1-2)
- ✅ Fase 1: Limpieza y reorganización
- ✅ Fase 2.1: Importar datos FX-1-Minute

### Semana 1 (Días 3-4)
- ⏳ Fase 2.2: Conversión a Parquet
- ⏳ Fase 2.3: Integración DuckDB

### Semana 2 (Días 5-7)
- ⏳ Fase 3.1-3.2: Setup MT5 + Streaming

### Semana 2 (Días 8-9)
- ⏳ Fase 3.3-3.5: Ejecución + Monitoreo

### Semana 3 (Día 10)
- ⏳ Fase 4: Configuraciones finales

### Semana 3 (Días 11-12)
- ⏳ Fase 5: Deployment a servidor

### Semana 3 (Día 13)
- ⏳ Fase 6: Acceso remoto
- ⏳ Testing final
- ⏳ Documentación

---

## ⚠️ RIESGOS Y MITIGACIONES

| Riesgo | Probabilidad | Impacto | Mitigación |
|--------|--------------|---------|------------|
| Conexión MT5 inestable | MEDIA | ALTO | Implementar reconexión automática + fallback |
| Pérdida de datos en servidor | BAJA | CRÍTICO | Backups automáticos diarios a S3 |
| Bug en ejecución de órdenes | MEDIA | CRÍTICO | Testing exhaustivo en demo antes de live |
| Servidor caído | BAJA | ALTO | Monitoring con UptimeRobot + alertas SMS |
| Límite de API de broker | BAJA | MEDIO | Rate limiting + queue de órdenes |

---

## ✅ CHECKLIST FINAL

- [ ] Sistema funciona 100% en local
- [ ] Todos los tests pasan
- [ ] Documentación completa
- [ ] MT5 conectado y ejecutando órdenes en demo
- [ ] Grafana mostrando datos en tiempo real
- [ ] Servidor configurado y seguro
- [ ] Backups automáticos funcionando
- [ ] Monitoring y alertas activas
- [ ] Acceso mobile funcionando
- [ ] Sistema probado durante 1 semana en demo

---

## 📚 RECURSOS

- **MT5 Python API**: https://www.mql5.com/en/docs/python_metatrader5
- **DuckDB Docs**: https://duckdb.org/docs/
- **Parquet Docs**: https://parquet.apache.org/docs/
- **Docker Deployment**: https://docs.docker.com/compose/production/
- **Grafana Mobile**: https://grafana.com/docs/grafana/latest/mobile/

---

**Última actualización**: 2025-10-20  
**Estado**: 🟡 EN PLANIFICACIÓN  
**Siguiente paso**: Comenzar Fase 1 - Limpieza de EAs

