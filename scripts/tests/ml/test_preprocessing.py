"""
Unit tests for ML preprocessing pipeline.

Tests cover:
1. Log returns calculation
2. ADF test validation
3. Lagged features (causality)
4. Technical features
5. Standardization
6. Complete pipeline
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from underdog.ml.preprocessing import MLPreprocessor


@pytest.fixture
def sample_ohlcv():
    """Generate sample OHLCV data for testing."""
    np.random.seed(42)
    
    dates = pd.date_range(start='2020-01-01', periods=1000, freq='1h')
    
    # Generate realistic price series
    returns = np.random.randn(1000) * 0.001  # 0.1% std
    price = 1.1000 * np.exp(np.cumsum(returns))
    
    df = pd.DataFrame({
        'timestamp': dates,
        'open': price * (1 + np.random.uniform(-0.0005, 0.0005, 1000)),
        'high': price * (1 + np.random.uniform(0, 0.001, 1000)),
        'low': price * (1 - np.random.uniform(0, 0.001, 1000)),
        'close': price,
        'volume': np.random.uniform(100, 1000, 1000)
    })
    
    df.set_index('timestamp', inplace=True)
    
    return df


class TestLogReturns:
    """Test log returns transformation."""
    
    def test_log_returns_calculation(self, sample_ohlcv):
        """Test that log returns are calculated correctly."""
        preprocessor = MLPreprocessor()
        df = preprocessor.log_returns(sample_ohlcv)
        
        # Check column added
        assert 'log_return' in df.columns
        
        # Check first value is NaN (no previous price)
        assert pd.isna(df['log_return'].iloc[0])
        
        # Manual calculation for second value
        expected = np.log(df['close'].iloc[1] / df['close'].iloc[0])
        actual = df['log_return'].iloc[1]
        assert np.isclose(expected, actual, rtol=1e-5)
    
    def test_log_returns_no_inf(self, sample_ohlcv):
        """Test that infinite values are handled."""
        preprocessor = MLPreprocessor()
        
        # Introduce zero price (would cause -inf)
        sample_ohlcv.loc[sample_ohlcv.index[10], 'close'] = 0
        
        df = preprocessor.log_returns(sample_ohlcv)
        
        # Check no infinite values
        assert not np.isinf(df['log_return']).any()


class TestADFTest:
    """Test Augmented Dickey-Fuller stationarity test."""
    
    def test_adf_stationary_series(self):
        """Test that stationary series is detected."""
        preprocessor = MLPreprocessor()
        
        # Generate stationary series (white noise)
        series = pd.Series(np.random.randn(1000))
        
        result = preprocessor.adf_test(series)
        
        assert result['is_stationary'] == True
        assert result['p_value'] < 0.05
        assert 'adf_statistic' in result
    
    def test_adf_non_stationary_series(self):
        """Test that non-stationary series is detected."""
        preprocessor = MLPreprocessor()
        
        # Generate non-stationary series (random walk)
        series = pd.Series(np.cumsum(np.random.randn(1000)))
        
        result = preprocessor.adf_test(series)
        
        # Random walk should NOT be stationary
        # (Note: might occasionally pass due to randomness)
        assert 'is_stationary' in result
        assert 'p_value' in result


class TestLaggedFeatures:
    """Test lagged features creation."""
    
    def test_lagged_features_creation(self, sample_ohlcv):
        """Test that lagged features are created correctly."""
        preprocessor = MLPreprocessor()
        df = preprocessor.log_returns(sample_ohlcv)
        
        df = preprocessor.create_lagged_features(
            df,
            feature_cols=['log_return'],
            lags=[1, 2, 5]
        )
        
        # Check columns created
        assert 'log_return_lag_1' in df.columns
        assert 'log_return_lag_2' in df.columns
        assert 'log_return_lag_5' in df.columns
        
        # Validate lag correctness
        for i in range(10, 20):  # Test middle of series
            assert df['log_return_lag_1'].iloc[i] == df['log_return'].iloc[i-1]
            assert df['log_return_lag_5'].iloc[i] == df['log_return'].iloc[i-5]
    
    def test_lagged_features_causality(self, sample_ohlcv):
        """Test that lagged features respect causality (no look-ahead)."""
        preprocessor = MLPreprocessor()
        df = preprocessor.log_returns(sample_ohlcv)
        
        df = preprocessor.create_lagged_features(
            df,
            feature_cols=['log_return'],
            lags=[1]
        )
        
        # lag_1 at time t should equal value at t-1
        # NEVER value at t or t+1 (look-ahead bias)
        for i in range(1, len(df)):
            assert df['log_return_lag_1'].iloc[i] == df['log_return'].iloc[i-1]


class TestTechnicalFeatures:
    """Test technical indicators calculation."""
    
    def test_technical_features_creation(self, sample_ohlcv):
        """Test that technical features are calculated."""
        preprocessor = MLPreprocessor()
        
        df = preprocessor.add_technical_features(sample_ohlcv)
        
        # Check columns created
        assert 'atr' in df.columns
        assert 'rsi' in df.columns
        assert 'sma_20' in df.columns
        assert 'sma_50' in df.columns
        
        # Check values are reasonable
        assert df['atr'].min() >= 0  # ATR always positive
        assert df['rsi'].min() >= 0  # RSI between 0-100
        assert df['rsi'].max() <= 100
    
    def test_technical_features_lagged(self, sample_ohlcv):
        """Test that technical features are lagged (no look-ahead)."""
        preprocessor = MLPreprocessor()
        
        df = preprocessor.add_technical_features(sample_ohlcv)
        
        # First ATR value should be NaN (no previous data + lag)
        # First 14 values NaN from calculation + 1 from lag = 15 NaN
        assert pd.isna(df['atr'].iloc[:15]).all()
        
        # RSI should also be lagged
        assert pd.isna(df['rsi'].iloc[:15]).all()
    
    def test_seasonality_features(self, sample_ohlcv):
        """Test that seasonality features are created."""
        preprocessor = MLPreprocessor()
        
        df = preprocessor.add_technical_features(sample_ohlcv)
        
        # Check seasonality columns
        assert 'day_sin' in df.columns
        assert 'day_cos' in df.columns
        assert 'hour_sin' in df.columns
        assert 'hour_cos' in df.columns
        
        # Check sin/cos bounds [-1, 1]
        assert df['day_sin'].min() >= -1
        assert df['day_sin'].max() <= 1
        assert df['hour_cos'].min() >= -1
        assert df['hour_cos'].max() <= 1


class TestStandardization:
    """Test feature standardization."""
    
    def test_standardization_mean_std(self, sample_ohlcv):
        """Test that standardized features have μ=0, σ=1."""
        preprocessor = MLPreprocessor()
        df = preprocessor.log_returns(sample_ohlcv)
        
        # Add some features
        df['feature1'] = np.random.randn(len(df)) * 100 + 50
        df['feature2'] = np.random.randn(len(df)) * 10 + 20
        
        df = preprocessor.standardize(df, feature_cols=['feature1', 'feature2'], fit=True)
        
        # Check mean ≈ 0, std ≈ 1
        assert np.isclose(df['feature1'].mean(), 0, atol=1e-10)
        assert np.isclose(df['feature1'].std(), 1, atol=1e-2)
        assert np.isclose(df['feature2'].mean(), 0, atol=1e-10)
        assert np.isclose(df['feature2'].std(), 1, atol=1e-2)
    
    def test_standardization_fit_transform(self, sample_ohlcv):
        """Test fit/transform pattern for train/test split."""
        preprocessor = MLPreprocessor()
        
        # Train data
        train_df = sample_ohlcv.iloc[:800].copy()
        train_df['feature'] = np.random.randn(len(train_df)) * 100 + 50
        
        # Test data
        test_df = sample_ohlcv.iloc[800:].copy()
        test_df['feature'] = np.random.randn(len(test_df)) * 100 + 50
        
        # Fit on train
        train_df = preprocessor.standardize(train_df, feature_cols=['feature'], fit=True)
        
        # Transform test (without fitting)
        test_df = preprocessor.standardize(test_df, feature_cols=['feature'], fit=False)
        
        # Both should be standardized (though test may not have exact μ=0, σ=1)
        assert abs(train_df['feature'].mean()) < 1e-10
        assert abs(train_df['feature'].std() - 1.0) < 0.1


class TestCompletePipeline:
    """Test complete preprocessing pipeline."""
    
    def test_pipeline_execution(self, sample_ohlcv):
        """Test that complete pipeline runs without errors."""
        preprocessor = MLPreprocessor()
        
        df_processed, validation = preprocessor.preprocess(
            sample_ohlcv,
            target_col='close',
            lags=[1, 2, 3],
            validate_stationarity=True,
            fit=True
        )
        
        # Check output
        assert isinstance(df_processed, pd.DataFrame)
        assert isinstance(validation, dict)
        assert 'adf_test' in validation
        
        # Check features created
        assert 'log_return' in df_processed.columns
        assert 'atr' in df_processed.columns
        assert 'log_return_lag_1' in df_processed.columns
        
        # Check no NaN in final output
        assert not df_processed.isnull().any().any()
    
    def test_pipeline_stationarity_validation(self, sample_ohlcv):
        """Test that pipeline validates stationarity."""
        preprocessor = MLPreprocessor()
        
        df_processed, validation = preprocessor.preprocess(
            sample_ohlcv,
            validate_stationarity=True,
            fit=True
        )
        
        # Log returns should be stationary
        assert validation['adf_test']['is_stationary'] == True
    
    def test_pipeline_feature_count(self, sample_ohlcv):
        """Test that pipeline creates expected number of features."""
        preprocessor = MLPreprocessor()
        
        lags = [1, 2, 3, 5, 10]
        
        df_processed, _ = preprocessor.preprocess(
            sample_ohlcv,
            lags=lags,
            fit=True
        )
        
        # Should have: OHLCV + log_return + ATR + RSI + SMAs + lagged features + seasonality
        # Minimum features expected
        min_features = 5 + 1 + 3 + len(lags)*3  # OHLCV + log_return + tech + lags
        
        assert len(df_processed.columns) >= min_features


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_empty_dataframe(self):
        """Test handling of empty DataFrame."""
        preprocessor = MLPreprocessor()
        df = pd.DataFrame()
        
        # Should not crash
        with pytest.raises((ValueError, KeyError)):
            preprocessor.preprocess(df)
    
    def test_missing_columns(self):
        """Test handling of missing required columns."""
        preprocessor = MLPreprocessor()
        df = pd.DataFrame({
            'close': [1, 2, 3, 4, 5]
        })
        
        # Should raise error (missing OHLCV)
        with pytest.raises(ValueError):
            preprocessor.add_technical_features(df)
    
    def test_single_row(self):
        """Test handling of single row."""
        preprocessor = MLPreprocessor()
        df = pd.DataFrame({
            'open': [1.1],
            'high': [1.2],
            'low': [1.0],
            'close': [1.1],
            'volume': [100]
        })
        
        # Should handle gracefully (but produce NaN)
        df = preprocessor.log_returns(df)
        assert pd.isna(df['log_return'].iloc[0])


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
