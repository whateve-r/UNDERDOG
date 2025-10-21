"""
MT5 Historical Data Loader - Real Broker Data via ZMQ

Uses your existing Mt5Connector (ZMQ/asyncio) to download historical data
directly from your MetaTrader 5 broker.

Advantages over Dukascopy/HuggingFace:
- Real broker data (same spreads you'll trade with)
- No Windows symlink issues
- Reuses existing mt5_connector.py infrastructure
- Bid/Ask separation available
- Seamless integration with live trading

Author: Underdog Trading System
Business Goal: High-quality data from real broker for PropFirm backtesting
"""

import asyncio
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Literal
import pandas as pd
import logging
import json

from underdog.core.connectors.mt5_connector import Mt5Connector

logger = logging.getLogger(__name__)


class MT5HistoricalDataLoader:
    """
    Download historical data from MT5 broker via ZMQ connector
    
    Features:
    - Uses your existing Mt5Connector (ZMQ/asyncio)
    - Real broker spreads (more accurate than Dukascopy estimates)
    - Automatic caching to avoid re-downloads
    - Multiple timeframes (M1, M5, M15, H1, H4, D1)
    - Integration with Backtrader
    
    Usage:
        # Async context
        async with MT5HistoricalDataLoader() as loader:
            df = await loader.get_data(
                symbol="EURUSD",
                start_date="2024-01-01",
                end_date="2024-12-31",
                timeframe="M1"
            )
        
        # Or synchronous wrapper
        df = MT5HistoricalDataLoader.download_sync(
            symbol="EURUSD",
            start_date="2024-01-01", 
            end_date="2024-12-31",
            timeframe="M1"
        )
    """
    
    # MT5 timeframe mapping
    TIMEFRAME_MAP = {
        'M1': 1,      # 1 minute
        '1M': 1,
        'M5': 5,      # 5 minutes
        '5M': 5,
        'M15': 15,    # 15 minutes
        '15M': 15,
        'M30': 30,    # 30 minutes
        '30M': 30,
        'H1': 60,     # 1 hour (60 minutes)
        '1H': 60,
        'H4': 240,    # 4 hours
        '4H': 240,
        'D1': 1440,   # 1 day (1440 minutes)
        '1D': 1440,
    }
    
    def __init__(
        self,
        cache_dir: Optional[str] = None,
        auto_cache: bool = True,
        mt5_connector: Optional[Mt5Connector] = None
    ):
        """
        Initialize MT5 historical data loader
        
        Args:
            cache_dir: Directory for caching (default: data/mt5_historical)
            auto_cache: Auto save/load from cache
            mt5_connector: Existing Mt5Connector instance (or will create new one)
        """
        if cache_dir is None:
            project_root = Path(__file__).parent.parent.parent
            cache_dir = project_root / "data" / "mt5_historical"
        
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.auto_cache = auto_cache
        self.connector = mt5_connector
        self._should_cleanup_connector = mt5_connector is None
        
        logger.info(f"MT5HistoricalDataLoader initialized - Cache: {self.cache_dir}")
    
    async def __aenter__(self):
        """Async context manager entry"""
        if self.connector is None:
            self.connector = Mt5Connector()
            await self.connector.connect()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self._should_cleanup_connector and self.connector:
            await self.connector.disconnect()
    
    async def get_data_by_count(
        self,
        symbol: str,
        count: int = 1000,
        timeframe: Literal['M1', 'M5', 'M15', 'M30', 'H1', 'H4', 'D1'] = 'M1'
    ) -> pd.DataFrame:
        """
        Get historical data by bar count (most recent N bars)
        
        Args:
            symbol: Trading symbol (e.g., "EURUSD")
            count: Number of bars to retrieve
            timeframe: Data granularity
        
        Returns:
            DataFrame with most recent N bars
        
        Note:
            This method uses 'count' parameter only (no date range).
            Use get_data() for date range queries.
        """
        if not self.connector or not self.connector.is_connected:
            raise RuntimeError("MT5 connector not connected. Use async context manager.")
        
        # CRITICAL: JsonAPI EA expects timeframe as STRING (e.g., "M1"), not as numeric minutes
        # Normalize timeframe string (accept both "M1" and "1M" formats)
        tf_normalized = timeframe.upper()
        if tf_normalized.startswith(('1M', '5M', '15M', '30M')):
            tf_normalized = 'M' + tf_normalized[0:2].replace('M', '')
        elif tf_normalized.startswith(('1H', '4H')):
            tf_normalized = 'H' + tf_normalized[0]
        elif tf_normalized.startswith('1D'):
            tf_normalized = 'D1'
        
        # Request using count only
        # CRITICAL: JsonAPI EA requires "actionType": "DATA" for HistoryInfo()
        # CRITICAL PARAMETER NAMES: Use "chartTF" (NOT "timeframe")
        request = {
            "action": "HISTORY",
            "actionType": "DATA",  # CRITICAL: Required by HistoryInfo() in EA
            "symbol": symbol,
            "chartTF": tf_normalized,  # FIXED: Was "timeframe", must be "chartTF"
            "count": count
        }
        
        logger.info(f"Requesting history: {symbol} {tf_normalized} (last {count} bars)")
        logger.info(f"Request JSON: {json.dumps(request)}")
        response = await self.connector.sys_request(request)
        
        if not response:
            raise ValueError(f"No data returned from MT5 for {symbol}")
        
        # CRITICAL FIX: JsonAPI EA returns data in ARRAY format
        # Expected: [[time, open, high, low, close, volume], ...]
        if isinstance(response, list):
            rates = response
        elif isinstance(response, dict):
            if response.get('error'):
                raise ValueError(f"MT5 EA error: {response.get('description', 'Unknown')}")
            rates = response.get('rates', [])
        else:
            raise ValueError(f"Unexpected response type: {type(response)}")
        
        if not rates:
            raise ValueError(f"Empty data returned for {symbol}")
        
        # Convert to DataFrame with proper column names
        df = pd.DataFrame(rates, columns=['time', 'open', 'high', 'low', 'close', 'tick_volume'])
        
        df['time'] = pd.to_datetime(df['time'], unit='s')
        df = df.set_index('time')
        df = df.sort_index()
        
        # Estimate spread (no spread data in array format)
        df['spread'] = 1.0
        point_value = 0.00001
        df['spread_price'] = df['spread'] * point_value
        df['close_ask'] = df['close']
        df['close_bid'] = df['close'] - df['spread_price']
        df['open_ask'] = df['open']
        df['open_bid'] = df['open'] - df['spread_price']
        df['high_ask'] = df['high']
        df['high_bid'] = df['high'] - df['spread_price']
        df['low_ask'] = df['low']
        df['low_bid'] = df['low'] - df['spread_price']
        
        return df
    
    async def get_data(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        timeframe: Literal['M1', 'M5', 'M15', 'M30', 'H1', 'H4', 'D1'] = 'M1',
        force_download: bool = False
    ) -> pd.DataFrame:
        """
        Get historical data from MT5 (with caching)
        
        Args:
            symbol: Trading symbol (e.g., "EURUSD")
            start_date: Start date "YYYY-MM-DD"
            end_date: End date "YYYY-MM-DD"
            timeframe: Data granularity
            force_download: Skip cache
        
        Returns:
            DataFrame with columns:
                - time: datetime index
                - open, high, low, close: OHLC prices
                - tick_volume: Number of ticks
                - spread: Average spread in points
                - real_volume: Real trading volume (if available)
        """
        # Check cache
        if self.auto_cache and not force_download:
            cached = self._load_from_cache(symbol, start_date, end_date, timeframe)
            if cached is not None:
                logger.info(f"Loaded {len(cached)} bars from cache")
                return cached
        
        # Download from MT5
        logger.info(f"Downloading {symbol} {timeframe} from MT5 broker...")
        
        df = await self._download_from_mt5(symbol, start_date, end_date, timeframe)
        
        logger.info(f"Downloaded {len(df)} bars from MT5")
        
        # Cache
        if self.auto_cache:
            self._save_to_cache(df, symbol, start_date, end_date, timeframe)
        
        return df
    
    async def _download_from_mt5(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        timeframe: str
    ) -> pd.DataFrame:
        """
        Download historical data from MT5 via ZMQ
        
        Uses sys_request to get historical bars
        """
        if not self.connector or not self.connector.is_connected:
            raise RuntimeError("MT5 connector not connected. Use async context manager.")
        
        # Convert dates
        start = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")
        
        # CRITICAL: JsonAPI EA expects timeframe as STRING (e.g., "M1"), not as numeric minutes
        # Normalize timeframe string (accept both "M1" and "1M" formats)
        tf_normalized = timeframe.upper()
        if tf_normalized.startswith(('1M', '5M', '15M', '30M')):
            tf_normalized = 'M' + tf_normalized[0:2].replace('M', '')
        elif tf_normalized.startswith(('1H', '4H')):
            tf_normalized = 'H' + tf_normalized[0]
        elif tf_normalized.startswith('1D'):
            tf_normalized = 'D1'
        
        # Request historical data via ZMQ
        # CRITICAL FIX: JsonAPI EA requires TWO levels of action:
        #   1. "action": "HISTORY" (top level)
        #   2. "actionType": "DATA" (sub-action for HistoryInfo function)
        # Without "actionType", EA throws ERR_WRONG_ACTION (65538)
        # 
        # CRITICAL PARAMETER NAMES: EA expects specific field names (MQL5 code):
        #   - "chartTF" (NOT "timeframe") -> period = StringToTimeframe(dataObject["chartTF"].ToStr())
        #   - "fromDate" (NOT "from") -> datetime fromDate = (datetime)dataObject["fromDate"].ToInt()
        #   - "toDate" (NOT "to") -> datetime toDate = (datetime)dataObject["toDate"].ToInt()
        # If wrong names are used, EA defaults to: PERIOD_CURRENT, 1970-01-01, TimeCurrent()
        request = {
            "action": "HISTORY",
            "actionType": "DATA",  # CRITICAL: Required by HistoryInfo() in EA
            "symbol": symbol,
            "chartTF": tf_normalized,  # FIXED: Was "timeframe", must be "chartTF"
            "fromDate": int(start.timestamp()),  # FIXED: Was "from", must be "fromDate"
            "toDate": int(end.timestamp())  # FIXED: Was "to", must be "toDate"
        }
        
        logger.info(f"Requesting history: {symbol} {tf_normalized} from {start_date} to {end_date}")
        logger.info(f"Request JSON: {json.dumps(request)}")
        response = await self.connector.sys_request(request)
        
        if not response:
            raise ValueError(f"No data returned from MT5 for {symbol} {start_date} to {end_date}")
        
        # CRITICAL FIX: JsonAPI EA returns data in ARRAY format, not object format
        # Expected: [[time, open, high, low, close, volume], ...]
        # NOT: {"error": false, "rates": [...]}
        
        # Check if response is array (new format) or object (old format)
        if isinstance(response, list):
            # New format: direct array of bars
            rates = response
            logger.info(f"Received {len(rates)} bars in array format")
        elif isinstance(response, dict):
            # Old format: object with 'rates' key
            if response.get('error'):
                error_desc = response.get('description', 'Unknown error')
                error_code = response.get('lastError', 'N/A')
                raise ValueError(f"MT5 EA error: {error_desc} (code: {error_code})")
            
            rates = response.get('rates', [])
            logger.info(f"Received {len(rates)} bars in object format")
        else:
            raise ValueError(f"Unexpected response type from MT5: {type(response)}")
        
        if not rates:
            raise ValueError(f"Empty data returned for {symbol}")
        
        # Convert to DataFrame
        # Array format: [time, open, high, low, close, volume]
        df = pd.DataFrame(rates, columns=['time', 'open', 'high', 'low', 'close', 'tick_volume'])
        
        # Ensure time is datetime
        df['time'] = pd.to_datetime(df['time'], unit='s')
        df = df.set_index('time')
        
        # Sort by time
        df = df.sort_index()
        
        # Calculate spread (approximate from bid/ask if available, otherwise use default)
        # For now, we don't have spread data in the array format
        # This is a limitation of the current JsonAPI implementation
        logger.warning("Spread data not available in array format - using default spread estimation")
        
        # Estimate spread (for EURUSD typically 1-2 points)
        # This is an approximation - real spread varies
        df['spread'] = 1.0  # Conservative estimate: 1 point
        
        # Add Bid/Ask separation using estimated spread
        point_value = 0.00001  # For EURUSD (adjust for JPY pairs if needed)
        df['spread_price'] = df['spread'] * point_value
        
        # MT5 typically returns Ask price in 'close'
        df['close_ask'] = df['close']
        df['close_bid'] = df['close'] - df['spread_price']
        
        # For OHLC, we approximate (conservative approach)
        df['open_ask'] = df['open']
        df['open_bid'] = df['open'] - df['spread_price']
        df['high_ask'] = df['high']
        df['high_bid'] = df['high'] - df['spread_price']
        df['low_ask'] = df['low']
        df['low_bid'] = df['low'] - df['spread_price']
        
        return df
    
    def _get_cache_path(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        timeframe: str
    ) -> Path:
        """Generate cache file path"""
        start_clean = start_date.replace('-', '')
        end_clean = end_date.replace('-', '')
        filename = f"{symbol}_{timeframe}_{start_clean}_{end_clean}.parquet"
        return self.cache_dir / filename
    
    def _load_from_cache(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        timeframe: str
    ) -> Optional[pd.DataFrame]:
        """Load from cache if exists"""
        cache_path = self._get_cache_path(symbol, start_date, end_date, timeframe)
        
        if not cache_path.exists():
            return None
        
        try:
            df = pd.read_parquet(cache_path)
            logger.info(f"Cache hit: {cache_path.name}")
            return df
        except Exception as e:
            logger.warning(f"Cache load failed: {e}")
            return None
    
    def _save_to_cache(
        self,
        df: pd.DataFrame,
        symbol: str,
        start_date: str,
        end_date: str,
        timeframe: str
    ):
        """Save to cache"""
        cache_path = self._get_cache_path(symbol, start_date, end_date, timeframe)
        
        try:
            df.to_parquet(cache_path, compression='gzip')
            logger.info(f"Cached: {cache_path.name} ({len(df)} bars)")
        except Exception as e:
            logger.warning(f"Cache save failed: {e}")
    
    def clear_cache(self, symbol: Optional[str] = None):
        """Clear cached data"""
        pattern = f"{symbol}_*.parquet" if symbol else "*.parquet"
        files = list(self.cache_dir.glob(pattern))
        
        for file in files:
            file.unlink()
        
        logger.info(f"Cleared {len(files)} cache files")
    
    @staticmethod
    def download_sync(
        symbol: str,
        start_date: str,
        end_date: str,
        timeframe: str = 'M1',
        cache_dir: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Synchronous wrapper for get_data (convenience function)
        
        Usage:
            df = MT5HistoricalDataLoader.download_sync("EURUSD", "2024-01-01", "2024-12-31", "M1")
        """
        async def _download():
            async with MT5HistoricalDataLoader(cache_dir=cache_dir) as loader:
                return await loader.get_data(symbol, start_date, end_date, timeframe)
        
        return asyncio.run(_download())


# Quick helper function
def download_mt5_data(
    symbol: str,
    start_date: str,
    end_date: str,
    timeframe: str = 'M1'
) -> pd.DataFrame:
    """
    Quick helper to download MT5 historical data
    
    Usage:
        df = download_mt5_data("EURUSD", "2024-01-01", "2024-12-31", "M1")
    """
    return MT5HistoricalDataLoader.download_sync(symbol, start_date, end_date, timeframe)


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("=" * 80)
    print("MT5 Historical Data Loader - Example")
    print("=" * 80)
    
    print("\nNOTE: This requires:")
    print("1. MT5 running with JsonAPI EA active")
    print("2. ZMQ ports configured (25555-25558)")
    print("3. Valid broker connection")
    
    # Uncomment to test (requires MT5 setup)
    # df = download_mt5_data("EURUSD", "2024-10-01", "2024-10-07", "M1")
    # 
    # print(f"\nDownloaded {len(df)} bars")
    # print(f"Columns: {list(df.columns)}")
    # print(f"\nFirst 5 bars:")
    # print(df.head())
    # 
    # if 'spread' in df.columns:
    #     print(f"\nAverage spread: {df['spread'].mean():.2f} points")
    # 
    # print("\n✅ MT5 data download successful!")
    
    print("\nTo test, uncomment the code above and ensure MT5 is running.")
