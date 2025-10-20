"""
Generate Synthetic Historical OHLCV Data for Backtesting
==========================================================

Creates realistic Forex price data with:
- Trend components (drift)
- Volatility clustering (GARCH-like)
- Intraday seasonality
- Bid/Ask spread

This allows testing backtesting engine without downloading real data.

Usage:
    poetry run python scripts/generate_synthetic_data.py \\
        --symbol EURUSD \\
        --timeframe M15 \\
        --start-date 2023-01-01 \\
        --end-date 2024-10-20 \\
        --output data/historical/

Output:
    CSV files with: timestamp, open, high, low, close, volume, bid, ask
"""

import argparse
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import sys

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

print("=" * 80)
print("📊 Synthetic Historical Data Generator")
print("=" * 80 + "\n")


class SyntheticDataGenerator:
    """
    Generate realistic Forex OHLCV data using statistical models.
    
    Features:
    - Geometric Brownian Motion for base price
    - GARCH-like volatility clustering
    - Intraday seasonality (volatility higher during London/NY open)
    - Realistic bid/ask spreads
    """
    
    def __init__(self, symbol: str, timeframe: str, start_price: float = 1.10):
        self.symbol = symbol
        self.timeframe = timeframe
        self.start_price = start_price
        
        # Timeframe to minutes mapping
        self.tf_minutes = {
            "M1": 1,
            "M5": 5,
            "M15": 15,
            "M30": 30,
            "H1": 60,
            "H4": 240,
            "D1": 1440
        }
        
        # Symbol characteristics
        self.symbol_params = {
            "EURUSD": {"spread_pips": 0.8, "daily_vol": 0.006, "drift": 0.0001},
            "GBPUSD": {"spread_pips": 1.2, "daily_vol": 0.008, "drift": 0.0002},
            "USDJPY": {"spread_pips": 0.9, "daily_vol": 0.007, "drift": -0.0001},
            "EURJPY": {"spread_pips": 1.5, "daily_vol": 0.009, "drift": 0.0001},
            "AUDUSD": {"spread_pips": 1.0, "daily_vol": 0.007, "drift": 0.0},
            "USDCAD": {"spread_pips": 1.1, "daily_vol": 0.006, "drift": 0.0001},
        }
        
        self.params = self.symbol_params.get(symbol, {
            "spread_pips": 1.0,
            "daily_vol": 0.007,
            "drift": 0.0
        })
    
    def generate_price_series(self, n_bars: int, seed: int = 42) -> pd.DataFrame:
        """
        Generate price series using Geometric Brownian Motion.
        
        Args:
            n_bars: Number of bars to generate
            seed: Random seed for reproducibility
        
        Returns:
            DataFrame with mid prices
        """
        np.random.seed(seed)
        
        # Parameters
        S0 = self.start_price  # Initial price
        mu = self.params["drift"]  # Drift (annualized)
        sigma = self.params["daily_vol"]  # Volatility (daily)
        
        # Time step (in years)
        minutes_per_bar = self.tf_minutes[self.timeframe]
        dt = minutes_per_bar / (252 * 24 * 60)  # Fraction of trading year
        
        # Generate returns with volatility clustering
        returns = np.zeros(n_bars)
        volatility = np.ones(n_bars) * sigma
        
        for i in range(1, n_bars):
            # GARCH-like volatility
            volatility[i] = 0.94 * volatility[i-1] + 0.06 * sigma * (1 + np.abs(returns[i-1]) * 5)
            
            # Intraday seasonality (higher vol during London/NY session)
            hour = (i * minutes_per_bar // 60) % 24
            if 8 <= hour <= 16:  # London + NY overlap
                vol_multiplier = 1.3
            elif hour < 6:  # Asian session (quieter)
                vol_multiplier = 0.7
            else:
                vol_multiplier = 1.0
            
            # Generate return
            returns[i] = mu * dt + volatility[i] * vol_multiplier * np.sqrt(dt) * np.random.normal()
        
        # Convert returns to prices
        prices = S0 * np.exp(np.cumsum(returns))
        
        return prices
    
    def generate_ohlcv(
        self,
        start_date: str,
        end_date: str,
        seed: int = 42
    ) -> pd.DataFrame:
        """
        Generate full OHLCV data with bid/ask spread.
        
        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            seed: Random seed
        
        Returns:
            DataFrame with timestamp, open, high, low, close, volume, bid, ask
        """
        print(f"Generating {self.symbol} {self.timeframe} data...")
        print(f"  Date range: {start_date} → {end_date}")
        
        # Calculate number of bars
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(end_date, "%Y-%m-%d")
        
        minutes_per_bar = self.tf_minutes[self.timeframe]
        total_minutes = int((end_dt - start_dt).total_seconds() / 60)
        n_bars = total_minutes // minutes_per_bar
        
        print(f"  Total bars: {n_bars:,}")
        
        # Generate timestamps (skip weekends)
        timestamps = []
        current = start_dt
        
        while len(timestamps) < n_bars:
            # Skip weekends (Saturday=5, Sunday=6)
            if current.weekday() < 5:
                timestamps.append(current)
            
            current += timedelta(minutes=minutes_per_bar)
        
        # Generate mid prices
        mid_prices = self.generate_price_series(len(timestamps), seed=seed)
        
        # Generate OHLC from mid prices
        np.random.seed(seed + 1)
        
        ohlc_data = []
        
        for i, (ts, mid) in enumerate(zip(timestamps, mid_prices)):
            # Generate intrabar volatility
            intrabar_vol = mid * self.params["daily_vol"] * np.sqrt(minutes_per_bar / (24 * 60))
            
            # Open (previous close or mid)
            if i == 0:
                open_price = mid
            else:
                open_price = ohlc_data[-1]["close"]
            
            # Close (mid price with small random walk)
            close_price = mid + np.random.normal(0, intrabar_vol * 0.3)
            
            # High/Low (realistic spread from open-close range)
            price_range = abs(close_price - open_price)
            high_price = max(open_price, close_price) + intrabar_vol * np.random.uniform(0.5, 1.5)
            low_price = min(open_price, close_price) - intrabar_vol * np.random.uniform(0.5, 1.5)
            
            # Volume (higher during London/NY session)
            hour = ts.hour
            if 8 <= hour <= 16:
                base_volume = 1000
            else:
                base_volume = 500
            
            volume = int(base_volume * np.random.uniform(0.7, 1.3))
            
            # Bid/Ask spread
            spread = self.params["spread_pips"] * 0.0001  # Pips to price
            bid = close_price - spread / 2
            ask = close_price + spread / 2
            
            ohlc_data.append({
                "timestamp": ts,
                "open": open_price,
                "high": high_price,
                "low": low_price,
                "close": close_price,
                "volume": volume,
                "bid": bid,
                "ask": ask
            })
        
        df = pd.DataFrame(ohlc_data)
        
        print(f"  ✓ Generated {len(df):,} bars")
        print(f"  Price range: {df['close'].min():.5f} → {df['close'].max():.5f}")
        print(f"  Avg spread: {self.params['spread_pips']:.1f} pips")
        
        return df
    
    def save_to_csv(self, df: pd.DataFrame, output_dir: str) -> str:
        """Save data to CSV file"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        filename = f"{self.symbol}_{self.timeframe}.csv"
        filepath = output_path / filename
        
        df.to_csv(filepath, index=False)
        
        print(f"  ✓ Saved to: {filepath}")
        print(f"  File size: {filepath.stat().st_size / 1024:.1f} KB")
        
        return str(filepath)


def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic historical OHLCV data for backtesting"
    )
    
    parser.add_argument(
        "--symbol",
        type=str,
        default="EURUSD",
        choices=["EURUSD", "GBPUSD", "USDJPY", "EURJPY", "AUDUSD", "USDCAD"],
        help="Trading symbol"
    )
    
    parser.add_argument(
        "--timeframe",
        type=str,
        default="M15",
        choices=["M1", "M5", "M15", "M30", "H1", "H4", "D1"],
        help="Timeframe"
    )
    
    parser.add_argument(
        "--start-date",
        type=str,
        default="2023-01-01",
        help="Start date (YYYY-MM-DD)"
    )
    
    parser.add_argument(
        "--end-date",
        type=str,
        default="2024-10-20",
        help="End date (YYYY-MM-DD)"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default="data/historical",
        help="Output directory"
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    
    args = parser.parse_args()
    
    # Generate data
    generator = SyntheticDataGenerator(
        symbol=args.symbol,
        timeframe=args.timeframe
    )
    
    df = generator.generate_ohlcv(
        start_date=args.start_date,
        end_date=args.end_date,
        seed=args.seed
    )
    
    # Save to CSV
    filepath = generator.save_to_csv(df, args.output)
    
    print("\n" + "=" * 80)
    print("✅ Data generation complete!")
    print("=" * 80)
    print(f"\nYou can now use this data for backtesting:")
    print(f"  File: {filepath}")
    print(f"  Bars: {len(df):,}")
    print(f"  Date range: {df['timestamp'].min()} → {df['timestamp'].max()}")
    print("\n" + "=" * 80 + "\n")


def generate_multiple_symbols():
    """Helper function to generate data for multiple symbols"""
    symbols = ["EURUSD", "GBPUSD", "USDJPY"]
    timeframe = "M15"
    start_date = "2023-01-01"
    end_date = "2024-10-20"
    output_dir = "data/historical"
    
    print("\n📊 Generating data for multiple symbols...\n")
    
    for symbol in symbols:
        print(f"\n{'='*80}")
        print(f"Processing {symbol}")
        print(f"{'='*80}\n")
        
        generator = SyntheticDataGenerator(symbol=symbol, timeframe=timeframe)
        df = generator.generate_ohlcv(start_date, end_date)
        generator.save_to_csv(df, output_dir)
    
    print("\n" + "="*80)
    print("✅ All symbols generated!")
    print("="*80 + "\n")


if __name__ == "__main__":
    # Check if called with --multiple flag
    if len(sys.argv) > 1 and sys.argv[1] == "--multiple":
        generate_multiple_symbols()
    else:
        main()
