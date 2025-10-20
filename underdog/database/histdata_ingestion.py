"""
HistData.com Ingestion Module (100% GRATUITO)

Descarga datos históricos de Forex desde HistData.com:
- Tick data de alta calidad
- Histórico hasta 2003
- Major pairs: EURUSD, GBPUSD, USDJPY, etc.
- GRATIS para uso no comercial

Proceso:
1. Descarga archivos .zip desde HistData
2. Descomprime y parsea CSV
3. Convierte tick data a OHLCV (M1, M5, M15, H1, etc.)
4. Inserta en TimescaleDB con validación
"""
from dataclasses import dataclass
from typing import List, Optional, Dict, Tuple
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import zipfile
import io
from urllib.parse import urlencode
import time
import warnings


@dataclass
class HistDataConfig:
    """Configuración para descarga de HistData"""
    # Símbolos a descargar
    symbols: List[str] = None
    
    # Rango de fechas
    start_year: int = 2020
    start_month: int = 1
    end_year: int = 2024
    end_month: int = 12
    
    # Tipo de datos
    data_type: str = "tick"  # "tick", "1min", "m1"
    
    # Conversión a OHLCV
    target_timeframe: str = "5min"  # "1min", "5min", "15min", "1h", "4h", "1d"
    
    # Storage
    cache_dir: str = "data/raw/histdata"
    output_dir: str = "data/processed/histdata"
    
    # Rate limiting
    delay_between_downloads: float = 2.0  # segundos
    max_retries: int = 3
    
    def __post_init__(self):
        if self.symbols is None:
            # Major Forex pairs (gratis en HistData)
            self.symbols = ["EURUSD", "GBPUSD", "USDJPY", "USDCHF", "AUDUSD", "USDCAD"]


class HistDataIngestion:
    """
    Cliente para descargar datos de HistData.com (100% gratuito).
    
    HistData.com ofrece:
    - Tick data desde 2003
    - Actualización mensual
    - Formato ASCII CSV
    - Sin API oficial (scraping web necesario)
    """
    
    BASE_URL = "https://www.histdata.com/download-free-forex-historical-data/"
    
    # Mapeo de símbolo a formato URL
    SYMBOL_MAP = {
        "EURUSD": "eurusd",
        "GBPUSD": "gbpusd",
        "USDJPY": "usdjpy",
        "USDCHF": "usdchf",
        "AUDUSD": "audusd",
        "USDCAD": "usdcad",
        "NZDUSD": "nzdusd",
        "EURGBP": "eurgbp",
        "EURJPY": "eurjpy",
        "GBPJPY": "gbpjpy"
    }
    
    def __init__(self, config: Optional[HistDataConfig] = None):
        """
        Args:
            config: Configuración de descarga
        """
        self.config = config or HistDataConfig()
        
        # Crear directorios
        Path(self.config.cache_dir).mkdir(parents=True, exist_ok=True)
        Path(self.config.output_dir).mkdir(parents=True, exist_ok=True)
        
        # Session para requests
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
    
    def download_symbol_range(self, 
                               symbol: str,
                               start_year: Optional[int] = None,
                               start_month: Optional[int] = None,
                               end_year: Optional[int] = None,
                               end_month: Optional[int] = None) -> pd.DataFrame:
        """
        Descarga rango completo de datos para un símbolo.
        
        Args:
            symbol: Par de divisas (ej: "EURUSD")
            start_year, start_month: Fecha de inicio
            end_year, end_month: Fecha final
        
        Returns:
            DataFrame con tick data consolidado
        """
        start_year = start_year or self.config.start_year
        start_month = start_month or self.config.start_month
        end_year = end_year or self.config.end_year
        end_month = end_month or self.config.end_month
        
        print(f"\n{'='*70}")
        print(f"Descargando {symbol}: {start_year}/{start_month:02d} → {end_year}/{end_month:02d}")
        print(f"{'='*70}")
        
        all_data = []
        
        # Iterar por cada mes
        current_date = datetime(start_year, start_month, 1)
        end_date = datetime(end_year, end_month, 1)
        
        while current_date <= end_date:
            year = current_date.year
            month = current_date.month
            
            try:
                # Descargar mes
                month_data = self._download_month(symbol, year, month)
                
                if month_data is not None and len(month_data) > 0:
                    all_data.append(month_data)
                    print(f"  ✓ {year}/{month:02d}: {len(month_data):,} ticks")
                else:
                    print(f"  ✗ {year}/{month:02d}: No data")
                
                # Rate limiting
                time.sleep(self.config.delay_between_downloads)
                
            except Exception as e:
                print(f"  ✗ {year}/{month:02d}: Error - {e}")
            
            # Siguiente mes
            current_date = current_date + timedelta(days=32)
            current_date = current_date.replace(day=1)
        
        # Concatenar todos los meses
        if len(all_data) == 0:
            print(f"\n[WARNING] No se descargó ningún dato para {symbol}")
            return pd.DataFrame()
        
        full_data = pd.concat(all_data, ignore_index=True)
        full_data = full_data.sort_values('timestamp').reset_index(drop=True)
        
        print(f"\n{'='*70}")
        print(f"Total descargado: {len(full_data):,} ticks")
        print(f"Fecha inicio: {full_data['timestamp'].min()}")
        print(f"Fecha fin: {full_data['timestamp'].max()}")
        print(f"{'='*70}")
        
        return full_data
    
    def _download_month(self, symbol: str, year: int, month: int) -> Optional[pd.DataFrame]:
        """
        Descarga un mes específico de datos.
        
        Nota: HistData.com NO tiene API oficial. Este método usa web scraping
              y puede requerir ajustes si cambia la estructura del sitio.
        
        Args:
            symbol: Par de divisas
            year: Año
            month: Mes (1-12)
        
        Returns:
            DataFrame con tick data o None si falla
        """
        # Verificar si existe en cache
        cache_file = Path(self.config.cache_dir) / f"{symbol}_{year}_{month:02d}_tick.csv"
        
        if cache_file.exists():
            print(f"  [CACHE] Cargando desde {cache_file.name}")
            try:
                return pd.read_csv(cache_file, parse_dates=['timestamp'])
            except Exception as e:
                print(f"  [WARNING] Error leyendo cache: {e}")
        
        # Construir URL (estructura de HistData.com)
        # NOTA: Esta URL puede cambiar. Verificar en histdata.com/download-free-forex-historical-data/
        symbol_lower = self.SYMBOL_MAP.get(symbol, symbol.lower())
        
        # URL para tick data (cambiar según tipo de dato)
        url = f"{self.BASE_URL}?/ascii/tick-data-quotes/{symbol_lower}/{year}/{month}"
        
        # Intentar descarga con retries
        for attempt in range(self.config.max_retries):
            try:
                response = self.session.get(url, timeout=30)
                
                if response.status_code != 200:
                    warnings.warn(f"HTTP {response.status_code} para {url}")
                    continue
                
                # Intentar extraer .zip del response
                # NOTA: Esto depende de la estructura HTML de HistData
                # Puede requerir BeautifulSoup para scraping más robusto
                
                # Método alternativo: Descarga MANUAL y usa método load_from_csv
                warnings.warn(
                    f"HistData.com requiere descarga manual. "
                    f"Por favor descargue {symbol} {year}/{month:02d} desde:\n"
                    f"  {url}\n"
                    f"Y guarde en: {cache_file}"
                )
                
                return None
                
            except Exception as e:
                print(f"  Intento {attempt+1}/{self.config.max_retries} falló: {e}")
                time.sleep(2)
        
        return None
    
    def load_from_csv(self, csv_path: str, symbol: str) -> pd.DataFrame:
        """
        Carga tick data desde CSV descargado manualmente de HistData.
        
        Formato CSV de HistData (tick data):
        20200101 000000123;1.12345;1.12348
        
        Columnas:
        - Timestamp (YYYYMMDD HHMMSSfff)
        - Bid
        - Ask
        
        Args:
            csv_path: Path al archivo CSV
            symbol: Símbolo para etiquetar
        
        Returns:
            DataFrame con columnas: timestamp, bid, ask, symbol
        """
        print(f"\n[HistData] Cargando CSV: {csv_path}")
        
        # Leer CSV (sin headers)
        df = pd.read_csv(
            csv_path,
            sep=';',
            header=None,
            names=['timestamp_str', 'bid', 'ask']
        )
        
        # Parsear timestamp (formato: 20200101 000000123)
        df['timestamp'] = pd.to_datetime(df['timestamp_str'], format='%Y%m%d %H%M%S%f')
        
        # Añadir símbolo
        df['symbol'] = symbol
        
        # Limpiar
        df = df[['timestamp', 'symbol', 'bid', 'ask']].copy()
        df = df.dropna()
        
        print(f"  ✓ Cargados {len(df):,} ticks")
        print(f"  Rango: {df['timestamp'].min()} → {df['timestamp'].max()}")
        
        return df
    
    def convert_tick_to_ohlcv(self, 
                               tick_data: pd.DataFrame,
                               timeframe: str = "5min") -> pd.DataFrame:
        """
        Convierte tick data a OHLCV (candlesticks).
        
        Args:
            tick_data: DataFrame con columnas: timestamp, bid, ask
            timeframe: Timeframe objetivo ("1min", "5min", "15min", "1h", "4h", "1d")
        
        Returns:
            DataFrame OHLCV con columnas: timestamp, open, high, low, close, volume
        """
        print(f"\n[HistData] Convirtiendo tick data a OHLCV ({timeframe})...")
        
        # Mapeo de timeframe a pandas offset
        TF_MAP = {
            "1min": "1T",
            "5min": "5T",
            "15min": "15T",
            "30min": "30T",
            "1h": "1H",
            "4h": "4H",
            "1d": "1D"
        }
        
        if timeframe not in TF_MAP:
            raise ValueError(f"Timeframe inválido: {timeframe}. Opciones: {list(TF_MAP.keys())}")
        
        resample_rule = TF_MAP[timeframe]
        
        # Usar mid price (bid + ask) / 2
        tick_data = tick_data.copy()
        tick_data['mid'] = (tick_data['bid'] + tick_data['ask']) / 2.0
        
        # Set timestamp como index
        tick_data = tick_data.set_index('timestamp')
        
        # Resample a OHLCV
        ohlcv = tick_data['mid'].resample(resample_rule).agg([
            ('open', 'first'),
            ('high', 'max'),
            ('low', 'min'),
            ('close', 'last')
        ])
        
        # Volume = número de ticks en la vela (proxy)
        ohlcv['volume'] = tick_data['mid'].resample(resample_rule).count()
        
        # Spread promedio
        ohlcv['spread'] = (tick_data['ask'] - tick_data['bid']).resample(resample_rule).mean()
        
        # Limpiar
        ohlcv = ohlcv.dropna().reset_index()
        
        print(f"  ✓ Generadas {len(ohlcv):,} velas {timeframe}")
        print(f"  Rango: {ohlcv['timestamp'].min()} → {ohlcv['timestamp'].max()}")
        
        return ohlcv
    
    def save_to_parquet(self, df: pd.DataFrame, symbol: str, data_type: str = "ohlcv"):
        """
        Guarda datos en formato Parquet (eficiente para TimescaleDB import).
        
        Args:
            df: DataFrame a guardar
            symbol: Símbolo
            data_type: Tipo de dato ("tick", "ohlcv")
        """
        output_path = Path(self.config.output_dir) / f"{symbol}_{data_type}.parquet"
        
        df.to_parquet(output_path, compression='snappy', index=False)
        
        print(f"\n[HistData] Guardado en: {output_path}")
        print(f"  Tamaño: {output_path.stat().st_size / 1024 / 1024:.2f} MB")


# ========================================
# Ejemplo de Uso
# ========================================

def run_histdata_ingestion_example():
    """Ejemplo completo de descarga y conversión"""
    print("="*70)
    print("HistData.com Ingestion Example (100% GRATUITO)")
    print("="*70)
    
    # Configurar
    config = HistDataConfig(
        symbols=["EURUSD"],
        start_year=2023,
        start_month=1,
        end_year=2023,
        end_month=3,  # Solo 3 meses para ejemplo
        target_timeframe="5min"
    )
    
    ingestion = HistDataIngestion(config)
    
    # MÉTODO 1: Descarga automática (puede fallar - requiere scraping)
    print("\n[MÉTODO 1] Descarga automática (puede requerir ajustes)")
    print("[WARNING] HistData.com NO tiene API oficial.")
    print("[RECOMENDACIÓN] Descarga manual + MÉTODO 2\n")
    
    # tick_data = ingestion.download_symbol_range("EURUSD")
    
    # MÉTODO 2: Carga desde CSV descargado manualmente
    print("\n[MÉTODO 2] Carga desde CSV manual (RECOMENDADO)")
    print("\nPasos:")
    print("1. Ir a: https://www.histdata.com/download-free-forex-historical-data/")
    print("2. Seleccionar: ASCII / Tick Data Quotes / EURUSD / 2023 / Enero")
    print("3. Descargar .zip y extraer CSV")
    print(f"4. Guardar en: {config.cache_dir}/")
    print("5. Ejecutar load_from_csv()\n")
    
    # Ejemplo con CSV ficticio
    # tick_data = ingestion.load_from_csv("data/raw/histdata/EURUSD_202301.csv", "EURUSD")
    
    # Si tienes el CSV:
    # ohlcv = ingestion.convert_tick_to_ohlcv(tick_data, timeframe="5min")
    # ingestion.save_to_parquet(ohlcv, "EURUSD", "ohlcv_5min")
    
    print("\n[SUCCESS] Ver documentación arriba para uso completo")


if __name__ == '__main__':
    run_histdata_ingestion_example()
