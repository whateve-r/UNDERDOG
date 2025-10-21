"""
ML Preprocessing Pipeline - Stationarity and Feature Engineering

CRITICAL PRINCIPLES:
1. Stationarity is MANDATORY for ML (avoid spurious regression)
2. All features must be LAGGED (causality, no look-ahead bias)
3. Standardization required for distance-based models (NN, SVM)

Based on:
- Lopez de Prado (2018): Advances in Financial Machine Learning
- Academic best practices for time series ML
"""

import numpy as np
import pandas as pd
from typing import Optional, List, Tuple
from statsmodels.tsa.stattools import adfuller
from sklearn.preprocessing import StandardScaler
import talib


class MLPreprocessor:
    """
    ML preprocessing pipeline for financial time series.
    
    Pipeline:
    1. Log Returns (stationarity)
    2. ADF Test (validation)
    3. Lagged Features (causality)
    4. Technical Features (ATR, RSI, etc.)
    5. Standardization (μ=0, σ=1)
    """
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.is_fitted = False
    
    # =========================================================================
    # STATIONARITY (Step 1: Mandatory Transformation)
    # =========================================================================
    
    def log_returns(self, df: pd.DataFrame, price_col: str = 'close') -> pd.DataFrame:
        """
        Transform prices to log returns (stationarity).
        
        FORMULA: R_t = ln(P_t / P_{t-1})
        
        WHY: Price series are non-stationary (mean/variance change over time).
        Log returns are stationary and suitable for ML training.
        
        WITHOUT THIS: Model learns spurious relationships → fails OOS.
        
        Args:
            df: DataFrame with price data
            price_col: Column name for price (default 'close')
            
        Returns:
            DataFrame with 'log_return' column added
        """
        df = df.copy()
        df['log_return'] = np.log(df[price_col] / df[price_col].shift(1))
        df['log_return'] = df['log_return'].replace([np.inf, -np.inf], np.nan)
        df['log_return'] = df['log_return'].fillna(0)
        
        return df
    
    def adf_test(self, series: pd.Series, significance: float = 0.05) -> dict:
        """
        Augmented Dickey-Fuller test for stationarity.
        
        NULL HYPOTHESIS: Series has unit root (non-stationary)
        ALTERNATIVE: Series is stationary
        
        DECISION RULE:
        - p-value < 0.05 → Reject H0 → Series IS stationary ✓
        - p-value ≥ 0.05 → Fail to reject H0 → Series NOT stationary ✗
        
        CRITICAL: Always validate stationarity before ML training.
        
        Args:
            series: Time series to test
            significance: Significance level (default 0.05)
            
        Returns:
            dict with test results
        """
        result = adfuller(series.dropna())
        
        output = {
            'adf_statistic': result[0],
            'p_value': result[1],
            'critical_values': result[4],
            'is_stationary': result[1] < significance
        }
        
        if not output['is_stationary']:
            print(f"⚠️ WARNING: Series is NOT stationary (p-value = {result[1]:.4f})")
            print("   Action: Apply differencing or log returns transformation")
        else:
            print(f"✓ Series IS stationary (p-value = {result[1]:.4f})")
        
        return output
    
    # =========================================================================
    # FEATURE ENGINEERING (Step 2: Causal Features)
    # =========================================================================
    
    def create_lagged_features(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        lags: List[int] = [1, 2, 3, 5, 10]
    ) -> pd.DataFrame:
        """
        Create lagged features (prevent look-ahead bias).
        
        CAUSALITY PRINCIPLE: Signal at time t can ONLY use data from t-1 or earlier.
        
        IMPLEMENTATION: Use .shift(n) to create lagged versions.
        
        Example:
        - return_lag_1 = return at t-1 (used to predict t)
        - return_lag_5 = return at t-5 (used to predict t)
        
        Args:
            df: DataFrame with features
            feature_cols: Columns to lag
            lags: List of lag periods
            
        Returns:
            DataFrame with lagged features added
        """
        df = df.copy()
        
        for col in feature_cols:
            if col not in df.columns:
                print(f"⚠️ Warning: Column '{col}' not found, skipping")
                continue
                
            for lag in lags:
                df[f'{col}_lag_{lag}'] = df[col].shift(lag)
        
        return df
    
    def add_technical_features(
        self,
        df: pd.DataFrame,
        atr_period: int = 14,
        rsi_period: int = 14,
        sma_periods: List[int] = [20, 50, 200]
    ) -> pd.DataFrame:
        """
        Add technical indicators as features.
        
        CRITICAL: All indicators are LAGGED by 1 period to prevent look-ahead bias.
        
        Features added:
        - ATR (volatility)
        - RSI (momentum)
        - SMA (trend)
        - Volume features
        - Seasonality (day of week, hour)
        
        Args:
            df: DataFrame with OHLCV data
            atr_period: ATR calculation period
            rsi_period: RSI calculation period
            sma_periods: List of SMA periods
            
        Returns:
            DataFrame with technical features added
        """
        df = df.copy()
        
        # Ensure we have required columns
        required = ['open', 'high', 'low', 'close', 'volume']
        if not all(col in df.columns for col in required):
            raise ValueError(f"DataFrame must contain: {required}")
        
        # ATR (Average True Range - volatility measure)
        df['atr'] = talib.ATR(df['high'], df['low'], df['close'], timeperiod=atr_period)
        df['atr'] = df['atr'].shift(1)  # LAG by 1 period
        
        # RSI (Relative Strength Index - momentum)
        df['rsi'] = talib.RSI(df['close'], timeperiod=rsi_period)
        df['rsi'] = df['rsi'].shift(1)  # LAG by 1 period
        
        # SMA (Simple Moving Averages - trend)
        for period in sma_periods:
            df[f'sma_{period}'] = talib.SMA(df['close'], timeperiod=period)
            df[f'sma_{period}'] = df[f'sma_{period}'].shift(1)  # LAG by 1 period
        
        # Volume features
        df['volume_sma_20'] = df['volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma_20']
        df['volume_ratio'] = df['volume_ratio'].shift(1)  # LAG
        
        # Seasonality features (no lag needed - these are temporal encodings)
        if isinstance(df.index, pd.DatetimeIndex):
            # Day of week (0=Monday, 6=Sunday)
            df['day_of_week'] = df.index.dayofweek
            df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
            df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
            
            # Hour of day (for intraday data)
            if df.index.hour.nunique() > 1:  # Check if we have intraday data
                df['hour'] = df.index.hour
                df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
                df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        
        return df
    
    # =========================================================================
    # STANDARDIZATION (Step 3: Normalize for Distance-Based Models)
    # =========================================================================
    
    def standardize(
        self,
        df: pd.DataFrame,
        feature_cols: Optional[List[str]] = None,
        fit: bool = True
    ) -> pd.DataFrame:
        """
        Standardize features to μ=0, σ=1.
        
        CRITICAL for:
        - Neural Networks (gradient descent stability)
        - SVM (distance-based)
        - K-Nearest Neighbors
        
        NOT needed for:
        - Tree-based models (Random Forest, XGBoost)
        
        Args:
            df: DataFrame with features
            feature_cols: Columns to standardize (if None, auto-detect)
            fit: Whether to fit scaler (True for train, False for test)
            
        Returns:
            DataFrame with standardized features
        """
        df = df.copy()
        
        # Auto-detect numeric columns if not specified
        if feature_cols is None:
            # Exclude OHLCV and timestamp columns
            exclude = ['open', 'high', 'low', 'close', 'volume', 'timestamp']
            feature_cols = [col for col in df.select_dtypes(include=[np.number]).columns 
                           if col not in exclude]
        
        # Remove columns with NaN or Inf
        valid_cols = []
        for col in feature_cols:
            if col not in df.columns:
                continue
            if df[col].isnull().any() or np.isinf(df[col]).any():
                print(f"⚠️ Skipping '{col}' (contains NaN/Inf)")
                continue
            valid_cols.append(col)
        
        if not valid_cols:
            print("⚠️ No valid columns to standardize")
            return df
        
        # Fit and transform
        if fit:
            df[valid_cols] = self.scaler.fit_transform(df[valid_cols])
            self.is_fitted = True
        else:
            if not self.is_fitted:
                raise ValueError("Scaler not fitted. Call with fit=True first.")
            df[valid_cols] = self.scaler.transform(df[valid_cols])
        
        return df
    
    # =========================================================================
    # COMPLETE PIPELINE
    # =========================================================================
    
    def preprocess(
        self,
        df: pd.DataFrame,
        target_col: str = 'close',
        lags: List[int] = [1, 2, 3, 5, 10],
        validate_stationarity: bool = True,
        fit: bool = True
    ) -> Tuple[pd.DataFrame, dict]:
        """
        Complete ML preprocessing pipeline.
        
        Pipeline:
        1. Log returns (stationarity)
        2. ADF test (validation)
        3. Technical features (ATR, RSI, SMA)
        4. Lagged features (causality)
        5. Standardization (μ=0, σ=1)
        6. Drop NaN rows
        
        Args:
            df: Raw OHLCV DataFrame
            target_col: Price column for returns calculation
            lags: Lag periods for features
            validate_stationarity: Whether to run ADF test
            fit: Whether to fit scaler (True for train, False for test)
            
        Returns:
            Tuple of (processed DataFrame, validation results dict)
        """
        print("=" * 80)
        print("ML PREPROCESSING PIPELINE")
        print("=" * 80)
        
        validation = {}
        
        # Step 1: Log Returns
        print("\n[1/5] Transforming to log returns...")
        df = self.log_returns(df, target_col)
        
        # Step 2: ADF Test
        if validate_stationarity:
            print("\n[2/5] Validating stationarity (ADF Test)...")
            validation['adf_test'] = self.adf_test(df['log_return'])
            if not validation['adf_test']['is_stationary']:
                print("⚠️ WARNING: Series not stationary. Model may fail OOS.")
        
        # Step 3: Technical Features
        print("\n[3/5] Calculating technical features...")
        df = self.add_technical_features(df)
        
        # Step 4: Lagged Features
        print("\n[4/5] Creating lagged features...")
        lag_cols = ['log_return', 'atr', 'rsi']
        df = self.create_lagged_features(df, lag_cols, lags)
        
        # Step 5: Standardization
        print("\n[5/5] Standardizing features...")
        df = self.standardize(df, fit=fit)
        
        # Drop NaN rows (from lagging and indicators)
        initial_rows = len(df)
        df = df.dropna()
        dropped_rows = initial_rows - len(df)
        print(f"\nDropped {dropped_rows} rows with NaN (from lagging/indicators)")
        
        # Summary
        print("\n" + "=" * 80)
        print("PREPROCESSING SUMMARY")
        print("=" * 80)
        print(f"Final rows: {len(df):,}")
        print(f"Features: {len(df.columns)}")
        if validate_stationarity:
            print(f"Stationary: {'✓ YES' if validation['adf_test']['is_stationary'] else '✗ NO'}")
        print("=" * 80)
        
        return df, validation


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if __name__ == '__main__':
    # Example: Preprocess EURUSD data for ML training
    
    # Load data (example)
    df = pd.read_parquet('data/parquet/EURUSD/5min/EURUSD_2020_5min.parquet')
    df.set_index('timestamp', inplace=True)
    
    # Preprocess
    preprocessor = MLPreprocessor()
    df_processed, validation = preprocessor.preprocess(
        df,
        target_col='close',
        lags=[1, 2, 3, 5, 10, 20],
        validate_stationarity=True,
        fit=True  # True for training set
    )
    
    # Check stationarity
    if validation['adf_test']['is_stationary']:
        print("✓ Data ready for ML training")
        
        # Features for training
        feature_cols = [col for col in df_processed.columns 
                       if col not in ['open', 'high', 'low', 'close', 'volume', 'log_return']]
        
        X = df_processed[feature_cols]
        y = (df_processed['log_return'].shift(-1) > 0).astype(int)  # Predict next bar direction
        
        print(f"\nX shape: {X.shape}")
        print(f"y shape: {y.shape}")
        print(f"Feature columns: {len(feature_cols)}")
    else:
        print("✗ Data NOT stationary. Cannot train ML model.")
