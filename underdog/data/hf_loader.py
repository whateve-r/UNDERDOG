"""
Hugging Face Dataset Loader for Event-Driven Backtesting.

Datasets:
- elthariel/histdata_fx_1m: 1-minute Forex OHLC data (HistData)
- Ehsanrs2/Forex_Factory_Calendar: Forex Factory news calendar
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Optional, List
from datasets import load_dataset

from underdog.core.abstractions import DataHandler, MarketEvent


class HuggingFaceDataHandler(DataHandler):
    """
    DataHandler for Hugging Face datasets.
    
    Loads historical Forex data from HF Hub and serves it bar-by-bar
    for event-driven backtesting.
    """
    
    def __init__(
        self,
        dataset_id: str = 'elthariel/histdata_fx_1m',
        symbol: str = 'EURUSD',
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ):
        """
        Initialize HF data handler.
        
        Args:
            dataset_id: HF dataset identifier
            symbol: Trading symbol (e.g., 'EURUSD')
            start_date: Start date (YYYY-MM-DD) or None for all
            end_date: End date (YYYY-MM-DD) or None for all
        """
        self.dataset_id = dataset_id
        self.symbol = symbol.upper()
        self.start_date = start_date
        self.end_date = end_date
        
        # Load data
        self.df = self._load_data()
        self.current_idx = 0
        
        print(f"✓ Loaded {len(self.df):,} bars for {symbol}")
        print(f"  Date range: {self.df.index[0]} to {self.df.index[-1]}")
    
    def _load_data(self) -> pd.DataFrame:
        """Load and prepare data from HF Hub."""
        print(f"Loading dataset: {self.dataset_id}")
        
        try:
            # Load dataset (may require authentication)
            ds = load_dataset(self.dataset_id, split='train')
            df = ds.to_pandas()
        except Exception as e:
            print(f"⚠️  Could not load dataset: {e}")
            print(f"   Generating synthetic data for testing...")
            return self._generate_synthetic_data()
        
        # Filter by symbol if needed
        if 'symbol' in df.columns:
            df = df[df['symbol'].str.upper() == self.symbol].copy()
        
        # Rename columns to standard format
        column_mapping = {
            'timestamp': 'timestamp',
            'time': 'timestamp',
            'date': 'timestamp',
            'datetime': 'timestamp',
            'open': 'open',
            'high': 'high',
            'low': 'low',
            'close': 'close',
            'volume': 'volume',
            'vol': 'volume',
            'tick_volume': 'volume',
        }
        
        for old_col, new_col in column_mapping.items():
            if old_col in df.columns and old_col != new_col:
                df.rename(columns={old_col: new_col}, inplace=True)
        
        # Parse timestamp
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)
            df.sort_index(inplace=True)
        
        # Filter by date range
        if self.start_date:
            df = df[df.index >= self.start_date]
        if self.end_date:
            df = df[df.index <= self.end_date]
        
        # Ensure required columns exist
        required = ['open', 'high', 'low', 'close']
        for col in required:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")
        
        # Add volume if missing
        if 'volume' not in df.columns:
            df['volume'] = 0.0
        
        # Calculate bid/ask from close (estimate)
        # For HistData, we don't have bid/ask, so we estimate
        # Typical EURUSD spread: 0.1-0.5 pips (0.00001-0.00005)
        spread = 0.00002  # 0.2 pips for EURUSD
        df['bid'] = df['close'] - (spread / 2)
        df['ask'] = df['close'] + (spread / 2)
        
        return df
    
    def _generate_synthetic_data(self) -> pd.DataFrame:
        """Generate synthetic OHLC data for testing when HF dataset unavailable."""
        print("Generating synthetic data...")
        
        # Date range
        start = pd.to_datetime(self.start_date) if self.start_date else pd.to_datetime('2024-01-01')
        end = pd.to_datetime(self.end_date) if self.end_date else pd.to_datetime('2024-12-31')
        
        # Generate 1-minute bars
        timestamps = pd.date_range(start=start, end=end, freq='1min')
        n = len(timestamps)
        
        # Generate realistic price series (random walk with drift)
        np.random.seed(42)
        returns = np.random.randn(n) * 0.0001 + 0.000001  # 0.01% std, small drift
        price = 1.1000 * np.exp(np.cumsum(returns))
        
        df = pd.DataFrame({
            'timestamp': timestamps,
            'open': price * (1 + np.random.uniform(-0.00005, 0.00005, n)),
            'high': price * (1 + np.random.uniform(0, 0.0001, n)),
            'low': price * (1 - np.random.uniform(0, 0.0001, n)),
            'close': price,
            'volume': np.random.uniform(100, 1000, n)
        })
        
        df.set_index('timestamp', inplace=True)
        
        # Add bid/ask
        spread = 0.00002  # 0.2 pips
        df['bid'] = df['close'] - (spread / 2)
        df['ask'] = df['close'] + (spread / 2)
        
        print(f"✓ Generated {len(df):,} synthetic bars")
        
        return df
    
    def get_latest_bar(self) -> Optional[MarketEvent]:
        """Get current bar as MarketEvent."""
        if self.current_idx >= len(self.df):
            return None
        
        row = self.df.iloc[self.current_idx]
        
        event = MarketEvent(
            timestamp=row.name,
            symbol=self.symbol,
            open=float(row['open']),
            high=float(row['high']),
            low=float(row['low']),
            close=float(row['close']),
            volume=float(row['volume']),
            bid=float(row['bid']),
            ask=float(row['ask'])
        )
        
        return event
    
    def get_latest_bars(self, n: int = 1) -> List[MarketEvent]:
        """Get last N bars as MarketEvents."""
        if self.current_idx < n:
            n = self.current_idx
        
        events = []
        for i in range(self.current_idx - n, self.current_idx):
            row = self.df.iloc[i]
            event = MarketEvent(
                timestamp=row.name,
                symbol=self.symbol,
                open=float(row['open']),
                high=float(row['high']),
                low=float(row['low']),
                close=float(row['close']),
                volume=float(row['volume']),
                bid=float(row['bid']),
                ask=float(row['ask'])
            )
            events.append(event)
        
        return events
    
    def update_bars(self) -> bool:
        """
        Move to next bar.
        
        Returns:
            True if more data available, False if end of data
        """
        self.current_idx += 1
        return self.current_idx < len(self.df)
    
    def get_available_symbols(self) -> List[str]:
        """Return list of available symbols."""
        return [self.symbol]
    
    def reset(self):
        """Reset to beginning of data."""
        self.current_idx = 0


class ForexNewsHandler:
    """
    Handler for Forex Factory news calendar.
    
    Used to filter high-impact news events that widen spreads.
    """
    
    def __init__(self):
        """Load Forex Factory calendar from HF Hub."""
        print("Loading Forex Factory calendar...")
        ds = load_dataset('Ehsanrs2/Forex_Factory_Calendar', split='train')
        self.df = ds.to_pandas()
        
        # Parse timestamp
        if 'date' in self.df.columns:
            self.df['timestamp'] = pd.to_datetime(self.df['date'])
        elif 'datetime' in self.df.columns:
            self.df['timestamp'] = pd.to_datetime(self.df['datetime'])
        
        # Filter high-impact news only
        if 'impact' in self.df.columns:
            self.df = self.df[self.df['impact'].str.lower() == 'high'].copy()
        
        print(f"✓ Loaded {len(self.df):,} high-impact news events")
    
    def is_news_event(self, timestamp: datetime, window_minutes: int = 15) -> bool:
        """
        Check if timestamp is within news event window.
        
        Args:
            timestamp: Time to check
            window_minutes: Minutes before/after news to consider
            
        Returns:
            True if within news window, False otherwise
        """
        window = pd.Timedelta(minutes=window_minutes)
        
        # Check if any news event within window
        mask = (
            (self.df['timestamp'] >= timestamp - window) &
            (self.df['timestamp'] <= timestamp + window)
        )
        
        return mask.any()
    
    def get_news_at(self, timestamp: datetime, window_minutes: int = 15) -> pd.DataFrame:
        """Get news events near timestamp."""
        window = pd.Timedelta(minutes=window_minutes)
        
        mask = (
            (self.df['timestamp'] >= timestamp - window) &
            (self.df['timestamp'] <= timestamp + window)
        )
        
        return self.df[mask]
