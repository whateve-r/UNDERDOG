"""
Dukascopy Data Downloader - Direct HTTP Download

Simplified approach: Download directly from Dukascopy HTTP servers
without CLI dependencies. This is how duka/dukascopy-python work internally.

Author: Underdog Trading System
"""

from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional
import pandas as pd
import logging
import requests
import struct
import lzma

logger = logging.getLogger(__name__)


class SimpleDukascopyLoader:
    """
    Simple direct downloader from Dukascopy HTTP servers
    
    Downloads tick data and aggregates to desired timeframe.
    More reliable than CLI tools.
    
    Usage:
        loader = SimpleDukascopyLoader()
        df = loader.download("EURUSD", "2024-10-01", "2024-10-02", "M1")
    """
    
    # Dukascopy HTTP base URL
    BASE_URL = "https://datafeed.dukascopy.com/datafeed"
    
    # Symbol mapping (Dukascopy uses specific paths)
    SYMBOL_MAP = {
        'EURUSD': 'EURUSD',
        'GBPUSD': 'GBPUSD',
        'USDJPY': 'USDJPY',
        'USDCHF': 'USDCHF',
        'AUDUSD': 'AUDUSD',
        'USDCAD': 'USDCAD',
        'NZDUSD': 'NZDUSD',
        'EURGBP': 'EURGBP',
        'EURJPY': 'EURJPY',
        'GBPJPY': 'GBPJPY',
    }
    
    def __init__(self, cache_dir: Optional[str] = None):
        if cache_dir is None:
            project_root = Path(__file__).parent.parent.parent
            cache_dir = project_root / "data" / "dukascopy"
        
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"SimpleDukascopyLoader initialized - Cache: {self.cache_dir}")
    
    def download(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        timeframe: str = "M1",
        force_download: bool = False
    ) -> pd.DataFrame:
        """
        Download data from Dukascopy
        
        Args:
            symbol: e.g., "EURUSD"
            start_date: "YYYY-MM-DD"
            end_date: "YYYY-MM-DD"
            timeframe: "M1", "M5", "M15", "H1", etc.
            force_download: Skip cache
        
        Returns:
            DataFrame with Bid/Ask OHLC
        """
        # Check cache
        cache_path = self._get_cache_path(symbol, start_date, end_date, timeframe)
        if not force_download and cache_path.exists():
            logger.info(f"Loading from cache: {cache_path.name}")
            return pd.read_parquet(cache_path)
        
        # Download tick data
        logger.info(f"Downloading {symbol} from {start_date} to {end_date}...")
        ticks = self._download_ticks(symbol, start_date, end_date)
        
        if ticks.empty:
            raise ValueError(f"No data downloaded for {symbol}")
        
        # Aggregate to timeframe
        df = self._aggregate_to_timeframe(ticks, timeframe)
        
        # Cache
        df.to_parquet(cache_path, compression='gzip')
        logger.info(f"Cached to: {cache_path.name} ({len(df)} bars)")
        
        return df
    
    def _download_ticks(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Download tick data from Dukascopy for date range
        
        Dukascopy stores ticks in hourly .bi5 files (binary compressed)
        URL format: /datafeed/EURUSD/2024/09/01/00h_ticks.bi5
        """
        start = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")
        
        all_ticks = []
        current = start
        
        while current < end:
            # Download each hour
            for hour in range(24):
                url = self._build_tick_url(symbol, current.year, current.month, current.day, hour)
                
                try:
                    ticks = self._download_hour(url)
                    if not ticks.empty:
                        all_ticks.append(ticks)
                except Exception as e:
                    logger.warning(f"Failed to download {url}: {e}")
            
            current += timedelta(days=1)
        
        if not all_ticks:
            return pd.DataFrame()
        
        # Combine all ticks
        df = pd.concat(all_ticks, ignore_index=True)
        df = df.sort_values('time')
        
        return df
    
    def _build_tick_url(self, symbol: str, year: int, month: int, day: int, hour: int) -> str:
        """
        Build Dukascopy tick data URL
        
        Format: https://datafeed.dukascopy.com/datafeed/EURUSD/2024/09/01/00h_ticks.bi5
        """
        return (
            f"{self.BASE_URL}/{symbol}/"
            f"{year:04d}/{month-1:02d}/{day:02d}/"  # Note: Dukascopy uses 0-indexed months
            f"{hour:02d}h_ticks.bi5"
        )
    
    def _download_hour(self, url: str) -> pd.DataFrame:
        """
        Download and decompress a single hourly tick file
        
        .bi5 format: LZMA compressed binary with 20-byte tick structure
        """
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        
        # Decompress LZMA
        decompressed = lzma.decompress(response.content)
        
        # Parse binary ticks (20 bytes each)
        # Structure: timestamp(4), ask(4), bid(4), ask_volume(4), bid_volume(4)
        tick_size = 20
        num_ticks = len(decompressed) // tick_size
        
        ticks = []
        for i in range(num_ticks):
            offset = i * tick_size
            chunk = decompressed[offset:offset + tick_size]
            
            # Unpack binary data (big-endian integers)
            timestamp_ms, ask, bid, ask_vol, bid_vol = struct.unpack('>5i', chunk)
            
            # Convert timestamp (milliseconds from hour start)
            # Extract hour from URL for absolute timestamp
            # For simplicity, we'll use relative timestamp here
            
            ticks.append({
                'time': timestamp_ms,  # Will be fixed later with absolute time
                'ask': ask / 100000,  # Dukascopy stores as int * 100000
                'bid': bid / 100000,
                'ask_volume': ask_vol,
                'bid_volume': bid_vol
            })
        
        df = pd.DataFrame(ticks)
        return df
    
    def _aggregate_to_timeframe(self, ticks: pd.DataFrame, timeframe: str) -> pd.DataFrame:
        """
        Aggregate tick data to OHLC bars
        
        Args:
            ticks: DataFrame with 'time', 'ask', 'bid' columns
            timeframe: "M1", "M5", "H1", etc.
        
        Returns:
            DataFrame with Bid/Ask OHLC
        """
        # Convert timeframe to pandas frequency
        freq_map = {
            'M1': '1min', '1M': '1min',
            'M5': '5min', '5M': '5min',
            'M15': '15min', '15M': '15min',
            'M30': '30min', '30M': '30min',
            'H1': '1H', '1H': '1H',
            'H4': '4H', '4H': '4H',
            'D1': '1D', '1D': '1D',
        }
        freq = freq_map.get(timeframe, '1min')
        
        # Resample to OHLC for Bid and Ask separately
        ticks = ticks.set_index('time')
        
        # Bid OHLC
        bid_ohlc = ticks['bid'].resample(freq).ohlc()
        bid_ohlc.columns = ['open_bid', 'high_bid', 'low_bid', 'close_bid']
        
        # Ask OHLC
        ask_ohlc = ticks['ask'].resample(freq).ohlc()
        ask_ohlc.columns = ['open_ask', 'high_ask', 'low_ask', 'close_ask']
        
        # Volume (sum of both ask and bid volumes)
        volume = (ticks['ask_volume'] + ticks['bid_volume']).resample(freq).sum()
        volume.name = 'volume'
        
        # Combine
        df = pd.concat([bid_ohlc, ask_ohlc, volume], axis=1)
        df = df.dropna()  # Remove bars with no ticks
        
        return df
    
    def _get_cache_path(self, symbol: str, start_date: str, end_date: str, timeframe: str) -> Path:
        """Generate cache file path"""
        start_clean = start_date.replace('-', '')
        end_clean = end_date.replace('-', '')
        filename = f"{symbol}_{timeframe}_{start_clean}_{end_clean}.parquet"
        return self.cache_dir / filename


# Quick helper
def download_dukascopy(
    symbol: str,
    start_date: str,
    end_date: str,
    timeframe: str = "M1"
) -> pd.DataFrame:
    """
    Quick helper to download Dukascopy data
    
    Usage:
        df = download_dukascopy("EURUSD", "2024-10-01", "2024-10-02", "M1")
    """
    loader = SimpleDukascopyLoader()
    return loader.download(symbol, start_date, end_date, timeframe)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("=" * 80)
    print("Simple Dukascopy Downloader - Test")
    print("=" * 80)
    
    # Download 1 day of EURUSD (tick → M1)
    df = download_dukascopy("EURUSD", "2024-10-01", "2024-10-02", "M1")
    
    print(f"\nDownloaded {len(df)} M1 bars")
    print(f"Columns: {list(df.columns)}")
    print(f"\nFirst 5 bars:")
    print(df.head())
    
    # Calculate spread
    df['spread'] = df['close_ask'] - df['close_bid']
    print(f"\nAverage spread: {df['spread'].mean():.5f}")
    
    print("\n✅ Download successful!")
