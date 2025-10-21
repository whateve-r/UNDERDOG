"""
Example: ML Preprocessing Pipeline for Forex Trading

Demonstrates:
1. Loading OHLCV data
2. Preprocessing with stationarity validation
3. Feature engineering
4. Train/test split
5. Model training (Logistic Regression example)
6. Backtest integration

CRITICAL PRINCIPLES:
- Stationarity is MANDATORY
- All features must be LAGGED
- Fit scaler on train, transform on test
- Validate with ADF test
"""

import numpy as np
import pandas as pd
from datetime import datetime
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from underdog.ml.preprocessing import MLPreprocessor


def load_sample_data():
    """
    Load sample OHLCV data.
    
    For production, replace with:
    - Hugging Face dataset: datasets.load_dataset('forex/eurusd')
    - Parquet files: pd.read_parquet('data/parquet/EURUSD/5min/...')
    - Database: pd.read_sql('SELECT * FROM ohlcv WHERE symbol = "EURUSD"')
    """
    print("=" * 80)
    print("LOADING SAMPLE DATA")
    print("=" * 80)
    
    # Generate synthetic OHLCV data for demonstration
    np.random.seed(42)
    
    dates = pd.date_range(start='2020-01-01', end='2024-12-31', freq='1h')
    n = len(dates)
    
    # Generate realistic price series (random walk with drift)
    returns = np.random.randn(n) * 0.001 + 0.0001  # 0.1% std, 0.01% drift
    price = 1.1000 * np.exp(np.cumsum(returns))
    
    df = pd.DataFrame({
        'timestamp': dates,
        'open': price * (1 + np.random.uniform(-0.0005, 0.0005, n)),
        'high': price * (1 + np.random.uniform(0, 0.001, n)),
        'low': price * (1 - np.random.uniform(0, 0.001, n)),
        'close': price,
        'volume': np.random.uniform(100, 1000, n)
    })
    
    df.set_index('timestamp', inplace=True)
    
    print(f"Loaded {len(df):,} bars")
    print(f"Date range: {df.index[0]} to {df.index[-1]}")
    print(f"Symbol: EURUSD (synthetic)")
    print(f"Timeframe: 1H")
    print("\nFirst 3 rows:")
    print(df.head(3))
    
    return df


def preprocess_data(df, fit=True):
    """
    Preprocess data for ML training.
    
    Args:
        df: Raw OHLCV DataFrame
        fit: Whether to fit scaler (True for train, False for test)
        
    Returns:
        Tuple of (processed DataFrame, validation results, preprocessor)
    """
    print("\n" + "=" * 80)
    print("PREPROCESSING DATA")
    print("=" * 80)
    
    preprocessor = MLPreprocessor()
    
    df_processed, validation = preprocessor.preprocess(
        df,
        target_col='close',
        lags=[1, 2, 3, 5, 10, 20],
        validate_stationarity=True,
        fit=fit
    )
    
    # Check stationarity
    if not validation['adf_test']['is_stationary']:
        print("\n⚠️ WARNING: Series NOT stationary!")
        print("   Cannot proceed with ML training.")
        print("   Consider:")
        print("   - Applying differencing")
        print("   - Using fractional differentiation")
        print("   - Checking data quality")
        return None, None, None
    
    print("\n✅ Data ready for ML training")
    
    return df_processed, validation, preprocessor


def create_features_target(df_processed):
    """
    Split processed data into features (X) and target (y).
    
    Target: Predict next bar direction (up/down)
    """
    print("\n" + "=" * 80)
    print("CREATING FEATURES AND TARGET")
    print("=" * 80)
    
    # Exclude OHLCV and log_return from features
    exclude_cols = ['open', 'high', 'low', 'close', 'volume', 'log_return']
    feature_cols = [col for col in df_processed.columns if col not in exclude_cols]
    
    X = df_processed[feature_cols]
    
    # Target: Predict next bar direction (1 = up, 0 = down)
    y = (df_processed['log_return'].shift(-1) > 0).astype(int)
    
    # Drop last row (no target)
    X = X[:-1]
    y = y[:-1]
    
    print(f"Features shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    print(f"\nFeature columns ({len(feature_cols)}):")
    for col in feature_cols[:10]:  # Show first 10
        print(f"  - {col}")
    if len(feature_cols) > 10:
        print(f"  ... and {len(feature_cols) - 10} more")
    
    print(f"\nTarget distribution:")
    print(f"  Up (1): {y.sum():,} ({y.mean():.1%})")
    print(f"  Down (0): {(~y.astype(bool)).sum():,} ({1-y.mean():.1%})")
    
    return X, y, feature_cols


def train_test_split(X, y, train_ratio=0.7):
    """
    Split data into train and test sets.
    
    CRITICAL: Time series split (not random shuffle)
    - Train: First 70% (2020-2023)
    - Test: Last 30% (2023-2024)
    """
    print("\n" + "=" * 80)
    print("TRAIN/TEST SPLIT")
    print("=" * 80)
    
    split_idx = int(len(X) * train_ratio)
    
    X_train = X.iloc[:split_idx]
    y_train = y.iloc[:split_idx]
    
    X_test = X.iloc[split_idx:]
    y_test = y.iloc[split_idx:]
    
    print(f"Train set: {len(X_train):,} samples ({train_ratio:.0%})")
    print(f"  Date range: {X_train.index[0]} to {X_train.index[-1]}")
    
    print(f"\nTest set: {len(X_test):,} samples ({1-train_ratio:.0%})")
    print(f"  Date range: {X_test.index[0]} to {X_test.index[-1]}")
    
    return X_train, X_test, y_train, y_test


def train_model(X_train, y_train):
    """
    Train ML model (Logistic Regression example).
    
    For production, consider:
    - Random Forest (robust, no tuning needed)
    - XGBoost (state-of-art, needs tuning)
    - LSTM (sequential patterns, complex)
    """
    print("\n" + "=" * 80)
    print("TRAINING MODEL")
    print("=" * 80)
    
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    
    print("Model: Logistic Regression")
    print("Solver: lbfgs")
    print("Max iterations: 1000")
    
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)
    
    # Train performance
    y_train_pred = model.predict(X_train)
    train_acc = accuracy_score(y_train, y_train_pred)
    
    print(f"\nTrain accuracy: {train_acc:.2%}")
    
    return model


def evaluate_model(model, X_test, y_test):
    """
    Evaluate model on test set (OOS performance).
    """
    print("\n" + "=" * 80)
    print("EVALUATING MODEL (OUT-OF-SAMPLE)")
    print("=" * 80)
    
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
    
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]  # Probability of UP
    
    # Metrics
    acc = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    print(f"Accuracy: {acc:.2%}")
    print(f"Precision: {precision:.2%}")
    print(f"Recall: {recall:.2%}")
    print(f"F1 Score: {f1:.2%}")
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Down', 'Up']))
    
    # Baseline comparison
    baseline = y_test.mean()
    print(f"\nBaseline (always predict UP): {baseline:.2%}")
    print(f"Model improvement: {(acc - baseline) / baseline:.1%}")
    
    if acc < baseline * 1.05:  # Less than 5% improvement
        print("\n⚠️ WARNING: Model barely beats baseline!")
        print("   Consider:")
        print("   - Adding more features")
        print("   - Using more complex model")
        print("   - Checking data quality")
    else:
        print("\n✅ Model shows meaningful improvement over baseline")
    
    return y_pred, y_proba


def feature_importance(model, feature_cols):
    """
    Show feature importance (for Logistic Regression).
    """
    print("\n" + "=" * 80)
    print("FEATURE IMPORTANCE")
    print("=" * 80)
    
    # Get coefficients
    coef = model.coef_[0]
    
    # Sort by absolute value
    importance = pd.DataFrame({
        'feature': feature_cols,
        'coefficient': coef,
        'abs_coefficient': np.abs(coef)
    }).sort_values('abs_coefficient', ascending=False)
    
    print("\nTop 10 most important features:")
    print(importance.head(10).to_string(index=False))
    
    return importance


def backtest_integration_example(y_test, y_proba):
    """
    Example of integrating ML predictions into backtest.
    
    This is a SIMPLIFIED example. For production, use:
    - underdog.strategies.ml_strategy.MLStrategy(Strategy)
    - underdog.core.abstractions.SignalEvent with ML metadata
    """
    print("\n" + "=" * 80)
    print("BACKTEST INTEGRATION (SIMPLIFIED EXAMPLE)")
    print("=" * 80)
    
    # Create signals
    df_signals = pd.DataFrame({
        'timestamp': y_test.index,
        'actual': y_test.values,
        'predicted': (y_proba > 0.5).astype(int),
        'confidence': y_proba
    })
    
    # Simple strategy: Long when predicted UP with confidence > 0.6
    df_signals['signal'] = ((df_signals['predicted'] == 1) & 
                            (df_signals['confidence'] > 0.6)).astype(int)
    
    # Calculate returns (assuming we can capture 50% of move)
    df_signals['return'] = df_signals['actual'] * df_signals['signal'] * 0.5
    
    # Performance
    total_trades = df_signals['signal'].sum()
    win_trades = ((df_signals['signal'] == 1) & (df_signals['actual'] == 1)).sum()
    win_rate = win_trades / total_trades if total_trades > 0 else 0
    
    cumulative_return = df_signals['return'].sum()
    
    print(f"Total trades: {total_trades:,}")
    print(f"Win trades: {win_trades:,}")
    print(f"Win rate: {win_rate:.2%}")
    print(f"Cumulative return: {cumulative_return:.2%}")
    
    print("\n⚠️ NOTE: This is a SIMPLIFIED example.")
    print("   For production, use:")
    print("   - underdog.strategies.ml_strategy.MLStrategy")
    print("   - Proper risk management (stop loss, position sizing)")
    print("   - Transaction costs (spread, slippage, commission)")
    print("   - Walk-Forward Optimization")
    
    return df_signals


def main():
    """
    Complete ML preprocessing workflow.
    """
    print("\n" + "=" * 80)
    print("ML PREPROCESSING PIPELINE - FOREX TRADING EXAMPLE")
    print("=" * 80)
    
    # Step 1: Load data
    df = load_sample_data()
    
    # Step 2: Train/test split (BEFORE preprocessing)
    split_idx = int(len(df) * 0.7)
    df_train = df.iloc[:split_idx]
    df_test = df.iloc[split_idx:]
    
    # Step 3: Preprocess train data (fit=True)
    df_train_processed, validation_train, preprocessor = preprocess_data(df_train, fit=True)
    
    if df_train_processed is None:
        print("\n❌ Preprocessing failed. Exiting.")
        return
    
    # Step 4: Preprocess test data (fit=False - use train scaler)
    print("\n" + "=" * 80)
    print("PREPROCESSING TEST DATA (Using train scaler)")
    print("=" * 80)
    
    df_test_processed, validation_test, _ = preprocess_data(df_test, fit=False)
    
    # Step 5: Create features and target
    X_train, y_train, feature_cols = create_features_target(df_train_processed)
    X_test, y_test, _ = create_features_target(df_test_processed)
    
    # Step 6: Train model
    model = train_model(X_train, y_train)
    
    # Step 7: Evaluate model
    y_pred, y_proba = evaluate_model(model, X_test, y_test)
    
    # Step 8: Feature importance
    importance = feature_importance(model, feature_cols)
    
    # Step 9: Backtest integration example
    signals = backtest_integration_example(y_test, y_proba)
    
    print("\n" + "=" * 80)
    print("✅ WORKFLOW COMPLETE")
    print("=" * 80)
    print("\nNext steps:")
    print("1. Replace synthetic data with real data (Hugging Face, parquet, DB)")
    print("2. Integrate into underdog.strategies.ml_strategy.MLStrategy")
    print("3. Add proper risk management (RiskManager, position sizing)")
    print("4. Run Walk-Forward Optimization (WFO)")
    print("5. Validate with Monte Carlo shuffling")
    print("=" * 80)


if __name__ == '__main__':
    main()
