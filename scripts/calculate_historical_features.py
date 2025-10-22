"""
Historical Feature Engineering & Regime Detection

Calculates technical indicators and regime predictions on historical OHLCV data.
Populates feature_store_metadata and regime_predictions tables.

Features Calculated:
- Technical Indicators: RSI, ATR, MACD, BB_width, ADX, Volume Ratio
- Price Transformations: Log returns, normalized price
- Regime Classification: Trend/Range/Transition (GMM + XGBoost)

Usage:
    poetry run python scripts/calculate_historical_features.py --symbols EURUSD GBPUSD
    
    # All symbols, all timeframes
    poetry run python scripts/calculate_historical_features.py --all

Business Goal:
    - Populate feature_store_metadata with active features
    - Populate regime_predictions with historical classifications
    - Enable StateVectorBuilder to work with historical data
"""

import asyncio
import argparse
from datetime import datetime
from pathlib import Path
import sys
import logging
from typing import List, Dict
import yaml
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
import joblib

sys.path.insert(0, str(Path(__file__).parent.parent))

from underdog.database.timescale.timescale_connector import TimescaleDBConnector

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger(__name__)


class FeatureEngineer:
    """
    Feature engineering for Forex trading
    
    Calculates technical indicators matching StateVectorBuilder:
    - RSI (14): Relative Strength Index
    - ATR (14): Average True Range
    - MACD (12,26,9): Moving Average Convergence Divergence
    - BB_width: Bollinger Bands width
    - ADX (14): Average Directional Index
    - Volume Ratio: Current volume / 20-period MA
    """
    
    def __init__(self):
        self.scaler = StandardScaler()
    
    def calculate_rsi(self, series: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Relative Strength Index"""
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average True Range"""
        high_low = df['high'] - df['low']
        high_close = (df['high'] - df['close'].shift()).abs()
        low_close = (df['low'] - df['close'].shift()).abs()
        
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = true_range.rolling(window=period).mean()
        return atr
    
    def calculate_macd(
        self,
        series: pd.Series,
        fast: int = 12,
        slow: int = 26,
        signal: int = 9
    ) -> Dict[str, pd.Series]:
        """Calculate MACD"""
        ema_fast = series.ewm(span=fast, adjust=False).mean()
        ema_slow = series.ewm(span=slow, adjust=False).mean()
        
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        histogram = macd_line - signal_line
        
        return {
            'macd': macd_line,
            'signal': signal_line,
            'histogram': histogram
        }
    
    def calculate_bollinger_bands(
        self,
        series: pd.Series,
        period: int = 20,
        std_dev: int = 2
    ) -> Dict[str, pd.Series]:
        """Calculate Bollinger Bands"""
        sma = series.rolling(window=period).mean()
        std = series.rolling(window=period).std()
        
        upper = sma + (std_dev * std)
        lower = sma - (std_dev * std)
        width = (upper - lower) / sma
        
        return {
            'upper': upper,
            'middle': sma,
            'lower': lower,
            'width': width
        }
    
    def calculate_adx(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average Directional Index"""
        # +DM, -DM
        high_diff = df['high'].diff()
        low_diff = -df['low'].diff()
        
        plus_dm = high_diff.where((high_diff > low_diff) & (high_diff > 0), 0)
        minus_dm = low_diff.where((low_diff > high_diff) & (low_diff > 0), 0)
        
        # ATR for normalization
        atr = self.calculate_atr(df, period)
        
        # +DI, -DI
        plus_di = 100 * (plus_dm.rolling(window=period).mean() / atr)
        minus_di = 100 * (minus_dm.rolling(window=period).mean() / atr)
        
        # DX
        dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
        
        # ADX (smoothed DX)
        adx = dx.rolling(window=period).mean()
        return adx
    
    def calculate_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate all technical indicators
        
        Args:
            df: OHLCV DataFrame
        
        Returns:
            DataFrame with all features
        """
        logger.info(f"Calculating features for {len(df)} bars...")
        
        # Make copy to avoid modifying original
        features = df.copy()
        
        # Price transformations
        features['log_return'] = np.log(features['close'] / features['close'].shift(1))
        features['price_norm'] = (features['close'] - features['close'].rolling(50).mean()) / features['close'].rolling(50).std()
        
        # Technical indicators
        features['rsi'] = self.calculate_rsi(features['close'], 14)
        features['atr'] = self.calculate_atr(features, 14)
        
        macd = self.calculate_macd(features['close'])
        features['macd'] = macd['macd']
        features['macd_signal'] = macd['signal']
        features['macd_hist'] = macd['histogram']
        
        bb = self.calculate_bollinger_bands(features['close'])
        features['bb_upper'] = bb['upper']
        features['bb_middle'] = bb['middle']
        features['bb_lower'] = bb['lower']
        features['bb_width'] = bb['width']
        
        features['adx'] = self.calculate_adx(features, 14)
        
        # Volume ratio
        features['volume_ma'] = features['volume'].rolling(window=20).mean()
        features['volume_ratio'] = features['volume'] / features['volume_ma']
        
        # Fill NaN in volume_ratio with 1.0 (neutral ratio) for initial periods
        features['volume_ratio'].fillna(1.0, inplace=True)
        
        # Drop NaN rows only for critical features (keep initial warm-up period data)
        # Only drop rows where technical indicators are NaN
        critical_cols = ['rsi', 'atr', 'adx', 'macd', 'bb_width']
        features = features.dropna(subset=critical_cols)
        
        logger.info(f"✓ Calculated {len(features.columns)} features for {len(features)} bars")
        return features
    
    def prepare_regime_features(self, df: pd.DataFrame) -> np.ndarray:
        """
        Prepare feature matrix for regime detection
        
        Uses:
        - ATR (volatility)
        - ADX (trend strength)
        - RSI (momentum)
        - Volume Ratio (activity)
        - BB Width (volatility)
        - MACD Histogram (momentum)
        """
        feature_cols = ['atr', 'adx', 'rsi', 'volume_ratio', 'bb_width', 'macd_hist']
        
        # Check which columns exist
        missing_cols = [col for col in feature_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Log NaN counts for debugging
        logger.info(f"NaN counts before dropna:")
        for col in feature_cols:
            nan_count = df[col].isna().sum()
            logger.info(f"  {col}: {nan_count} NaN ({nan_count/len(df)*100:.2f}%)")
        
        # Select features and drop any remaining NaN rows
        X_df = df[feature_cols].dropna()
        
        logger.info(f"Rows after dropna: {len(X_df)} (dropped {len(df) - len(X_df)} rows)")
        
        if len(X_df) == 0:
            raise ValueError(f"No valid data after dropping NaN. All {len(df)} rows had NaN in regime features.")
        
        X = X_df.values
        
        # Standardize
        X_scaled = self.scaler.fit_transform(X)
        
        return X_scaled


class RegimeDetector:
    """
    Market regime detection using GMM
    
    Regimes:
    - Trend: High ADX, directional movement
    - Range: Low ADX, mean reversion
    - Transition: Mixed signals
    """
    
    def __init__(self, n_regimes: int = 3):
        self.n_regimes = n_regimes
        self.gmm = GaussianMixture(
            n_components=n_regimes,
            covariance_type='full',
            max_iter=200,
            random_state=42
        )
        self.regime_labels = {
            0: 'trend',
            1: 'range',
            2: 'transition'
        }
    
    def fit(self, X: np.ndarray):
        """Train GMM on feature matrix"""
        logger.info("Training GMM for regime detection...")
        self.gmm.fit(X)
        logger.info("✓ GMM trained")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict regime labels"""
        return self.gmm.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict regime probabilities"""
        return self.gmm.predict_proba(X)
    
    def save(self, path: str):
        """Save trained model"""
        joblib.dump(self.gmm, path)
        logger.info(f"✓ Saved GMM to {path}")
    
    def load(self, path: str):
        """Load trained model"""
        self.gmm = joblib.load(path)
        logger.info(f"✓ Loaded GMM from {path}")


async def process_symbol_timeframe(
    db: TimescaleDBConnector,
    symbol: str,
    timeframe: str,
    engineer: FeatureEngineer,
    detector: RegimeDetector
):
    """
    Process a single symbol-timeframe combination
    
    Steps:
    1. Load OHLCV from TimescaleDB
    2. Calculate technical indicators
    3. Detect regimes
    4. Insert regime predictions
    5. Update feature_store_metadata
    """
    logger.info(f"\n{'='*80}")
    logger.info(f"Processing {symbol} {timeframe}")
    logger.info(f"{'='*80}")
    
    # Load OHLCV data
    query = """
    SELECT time, open, high, low, close, volume, spread
    FROM ohlcv
    WHERE symbol = $1 AND timeframe = $2
    ORDER BY time ASC
    """
    
    rows = await db.pool.fetch(query, symbol, timeframe)
    
    if not rows:
        logger.warning(f"No data found for {symbol} {timeframe}")
        return
    
    logger.info(f"Loaded {len(rows)} bars")
    
    # Convert to DataFrame
    df = pd.DataFrame(rows, columns=['time', 'open', 'high', 'low', 'close', 'volume', 'spread'])
    
    # Calculate features
    features_df = engineer.calculate_all_features(df)
    
    # Prepare regime features
    X = engineer.prepare_regime_features(features_df)
    
    # Detect regimes
    regime_labels = detector.predict(X)
    regime_probs = detector.predict_proba(X)
    
    # Map labels to names
    regime_names = [detector.regime_labels[label] for label in regime_labels]
    
    # Add to DataFrame
    features_df['regime'] = regime_names
    features_df['regime_confidence'] = regime_probs.max(axis=1)
    
    # Insert regime predictions
    logger.info("Inserting regime predictions...")
    
    records = [
        (
            row['time'],
            symbol,
            row['regime'],
            float(row['regime_confidence'])
        )
        for _, row in features_df.iterrows()
    ]
    
    # Batch insert
    batch_size = 10000
    for i in range(0, len(records), batch_size):
        batch = records[i:i+batch_size]
        await db.pool.executemany(
            """
            INSERT INTO regime_predictions (time, symbol, regime, confidence)
            VALUES ($1, $2, $3, $4)
            ON CONFLICT (time, symbol) DO UPDATE
            SET regime = EXCLUDED.regime, confidence = EXCLUDED.confidence
            """,
            batch
        )
    
    logger.info(f"✓ Inserted {len(records)} regime predictions")
    
    # Update feature_store_metadata
    feature_list = [
        'rsi', 'atr', 'macd', 'macd_signal', 'macd_hist',
        'bb_upper', 'bb_middle', 'bb_lower', 'bb_width',
        'adx', 'volume_ratio', 'log_return', 'price_norm'
    ]
    
    for feature_name in feature_list:
        await db.pool.execute(
            """
            INSERT INTO feature_store_metadata (
                feature_id, symbol, timeframe, source_table, 
                calculation_params, is_active, last_updated
            )
            VALUES ($1, $2, $3, $4, $5, $6, NOW())
            ON CONFLICT (feature_id, symbol, timeframe) DO UPDATE
            SET is_active = EXCLUDED.is_active, last_updated = NOW()
            """,
            f"{symbol}_{timeframe}_{feature_name}",
            symbol,
            timeframe,
            'ohlcv',
            {},  # JSONB empty dict
            True
        )
    
    logger.info(f"✓ Updated feature_store_metadata with {len(feature_list)} features")
    
    # Print regime distribution
    regime_counts = features_df['regime'].value_counts()
    logger.info("\nRegime Distribution:")
    for regime, count in regime_counts.items():
        pct = 100 * count / len(features_df)
        logger.info(f"  {regime:12} | {count:8,} bars ({pct:5.1f}%)")


async def main():
    parser = argparse.ArgumentParser(description='Calculate historical features and detect regimes')
    
    parser.add_argument(
        '--symbols',
        nargs='+',
        default=['EURUSD', 'GBPUSD', 'USDJPY', 'XAUUSD', 'SPXUSD', 'USDXUSD'],  # 4 trading + 2 macro
        help='Symbols to process (default: all 6 core symbols)'
    )
    
    parser.add_argument(
        '--timeframes',
        nargs='+',
        default=['M5', 'M15'],
        help='Timeframes to process'
    )
    
    parser.add_argument(
        '--all',
        action='store_true',
        help='Process all symbols and timeframes in database'
    )
    
    args = parser.parse_args()
    
    logger.info("="*80)
    logger.info("HISTORICAL FEATURE ENGINEERING & REGIME DETECTION")
    logger.info("="*80)
    
    # Load config
    with open('config/data_providers.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Initialize DB
    db = TimescaleDBConnector(
        host=config['timescaledb']['host'],
        port=config['timescaledb']['port'],
        database=config['timescaledb']['database'],
        user=config['timescaledb']['user'],
        password=config['timescaledb']['password']
    )
    
    await db.connect()
    
    # Get symbols/timeframes
    if args.all:
        query = """
        SELECT DISTINCT symbol, timeframe
        FROM ohlcv
        ORDER BY symbol, timeframe
        """
        rows = await db.pool.fetch(query)
        pairs = [(row['symbol'], row['timeframe']) for row in rows]
    else:
        pairs = [(s, t) for s in args.symbols for t in args.timeframes]
    
    logger.info(f"Processing {len(pairs)} symbol-timeframe pairs\n")
    
    # Initialize feature engineer and regime detector
    engineer = FeatureEngineer()
    detector = RegimeDetector(n_regimes=3)
    
    # Train regime detector on first symbol (if needed)
    # For production, you'd load a pre-trained model
    first_symbol, first_tf = pairs[0]
    query = """
    SELECT time, open, high, low, close, volume, spread
    FROM ohlcv
    WHERE symbol = $1 AND timeframe = $2
    ORDER BY time ASC
    """
    rows = await db.pool.fetch(query, first_symbol, first_tf)
    df_train = pd.DataFrame(rows, columns=['time', 'open', 'high', 'low', 'close', 'volume', 'spread'])
    features_train = engineer.calculate_all_features(df_train)
    X_train = engineer.prepare_regime_features(features_train)
    detector.fit(X_train)
    
    # Save trained model
    model_path = Path('data/models/regime_gmm.pkl')
    model_path.parent.mkdir(parents=True, exist_ok=True)
    detector.save(str(model_path))
    
    # Process each symbol-timeframe
    for symbol, timeframe in pairs:
        try:
            await process_symbol_timeframe(db, symbol, timeframe, engineer, detector)
        except Exception as e:
            logger.error(f"Error processing {symbol} {timeframe}: {e}")
            continue
    
    await db.disconnect()
    
    logger.info("\n" + "="*80)
    logger.info("FEATURE ENGINEERING COMPLETE")
    logger.info("="*80)
    logger.info("Next steps:")
    logger.info("  1. Backfill FRED: poetry run python scripts/backfill_fred_macro.py")
    logger.info("  2. Backfill Reddit: poetry run python scripts/backfill_reddit_sentiment.py")
    logger.info("  3. Train DRL agent: poetry run python scripts/train_drl_agent.py")
    logger.info("="*80)


if __name__ == "__main__":
    asyncio.run(main())
