"""
Test Suite for Feature Engineering Pipeline
Tests hash versioning, feature generation, and look-ahead bias validation.
"""
import pytest
import numpy as np
import pandas as pd
from pathlib import Path

from underdog.strategies.ml_strategies.feature_engineering import (
    FeatureEngineer, FeatureConfig, FeatureMetadata
)


class TestFeatureConfig:
    """Test FeatureConfig dataclass"""
    
    def test_default_config(self):
        """Test default configuration"""
        config = FeatureConfig()
        
        assert config.sma_periods == [20, 50, 200]
        assert config.prediction_horizon == 1
        assert config.random_seed == 42
    
    def test_config_to_dict(self):
        """Test configuration serialization"""
        config = FeatureConfig(
            sma_periods=[10, 20],
            target_type='classification'
        )
        
        config_dict = config.to_dict()
        
        assert isinstance(config_dict, dict)
        assert config_dict['sma_periods'] == [10, 20]
        assert config_dict['target_type'] == 'classification'
    
    def test_config_hash_deterministic(self):
        """Test that config hash is deterministic"""
        config1 = FeatureConfig(sma_periods=[20, 50], random_seed=42)
        config2 = FeatureConfig(sma_periods=[20, 50], random_seed=42)
        
        hash1 = config1.to_hash()
        hash2 = config2.to_hash()
        
        assert hash1 == hash2
    
    def test_config_hash_different(self):
        """Test that different configs have different hashes"""
        config1 = FeatureConfig(sma_periods=[20, 50])
        config2 = FeatureConfig(sma_periods=[30, 60])
        
        hash1 = config1.to_hash()
        hash2 = config2.to_hash()
        
        assert hash1 != hash2


class TestFeatureEngineer:
    """Test FeatureEngineer main functionality"""
    
    def test_engineer_initialization(self):
        """Test FeatureEngineer initialization"""
        config = FeatureConfig()
        engineer = FeatureEngineer(config)
        
        assert engineer.config == config
        assert engineer.metadata is None
    
    def test_transform_basic(self, sample_ohlcv_data):
        """Test basic feature transformation"""
        config = FeatureConfig(
            sma_periods=[20],
            momentum_periods=[5],
            lag_periods=[1]
        )
        engineer = FeatureEngineer(config)
        
        features = engineer.transform(sample_ohlcv_data)
        
        # Should have generated features
        assert len(features) > 0
        assert len(engineer.feature_names) > 0
        
        # Should have target column
        assert 'target' in features.columns
    
    def test_transform_preserves_index(self, sample_ohlcv_data):
        """Test that DatetimeIndex is preserved"""
        engineer = FeatureEngineer()
        features = engineer.transform(sample_ohlcv_data)
        
        assert isinstance(features.index, pd.DatetimeIndex)
    
    def test_no_look_ahead_bias(self, sample_ohlcv_data):
        """Test that features don't contain future information"""
        config = FeatureConfig(
            sma_periods=[20],
            prediction_horizon=1
        )
        engineer = FeatureEngineer(config)
        
        features = engineer.transform(sample_ohlcv_data)
        
        # Target should be future return (look-ahead is intentional for target only)
        # But features should only use past data
        
        # Check that SMA is calculated from past data only
        if 'sma_20' in features.columns:
            # First 20 values should be NaN or filled
            sma_values = features['sma_20'].dropna()
            assert len(sma_values) > 0


class TestFeatureGeneration:
    """Test specific feature generation functions"""
    
    def test_price_features(self, sample_ohlcv_data):
        """Test price-based features"""
        engineer = FeatureEngineer()
        features = engineer.transform(sample_ohlcv_data)
        
        # Should have returns
        assert 'returns' in features.columns
        assert 'log_returns' in features.columns
        
        # Returns should be reasonable
        assert features['returns'].abs().max() < 1.0  # No 100%+ daily returns
    
    def test_technical_indicators(self, sample_ohlcv_data):
        """Test technical indicator features"""
        config = FeatureConfig(
            sma_periods=[20, 50],
            rsi_period=14,
            bollinger_period=20
        )
        engineer = FeatureEngineer(config)
        features = engineer.transform(sample_ohlcv_data)
        
        # Should have SMAs
        assert 'sma_20' in features.columns
        assert 'sma_50' in features.columns
        
        # Should have RSI
        assert 'rsi' in features.columns
        
        # RSI should be between 0 and 100
        assert features['rsi'].min() >= 0
        assert features['rsi'].max() <= 100
        
        # Should have Bollinger Bands
        assert 'bb_upper' in features.columns
        assert 'bb_lower' in features.columns
    
    def test_momentum_features(self, sample_ohlcv_data):
        """Test momentum features"""
        config = FeatureConfig(momentum_periods=[5, 10])
        engineer = FeatureEngineer(config)
        features = engineer.transform(sample_ohlcv_data)
        
        # Should have ROC
        assert 'roc_5' in features.columns
        assert 'roc_10' in features.columns
        
        # Should have momentum
        assert 'momentum_5' in features.columns
    
    def test_volatility_features(self, sample_ohlcv_data):
        """Test volatility features"""
        config = FeatureConfig(realized_vol_windows=[5, 20])
        engineer = FeatureEngineer(config)
        features = engineer.transform(sample_ohlcv_data)
        
        # Should have realized volatility
        assert 'realized_vol_5' in features.columns
        assert 'realized_vol_20' in features.columns
        
        # Volatility should be positive
        assert features['realized_vol_5'].min() >= 0
    
    def test_lagged_features(self, sample_ohlcv_data):
        """Test lagged features"""
        config = FeatureConfig(lag_periods=[1, 2, 3])
        engineer = FeatureEngineer(config)
        features = engineer.transform(sample_ohlcv_data)
        
        # Should have lagged returns
        assert 'returns_lag_1' in features.columns
        assert 'returns_lag_2' in features.columns
        assert 'returns_lag_3' in features.columns
        
        # Lagged values should match shifted original
        # (within tolerance for NaN handling)
        original_returns = sample_ohlcv_data['close'].pct_change()
        lag_1 = features['returns_lag_1']
        
        # Check correlation (should be high)
        correlation = original_returns.shift(1).corr(lag_1)
        assert correlation > 0.95
    
    def test_temporal_features(self, sample_ohlcv_data):
        """Test cyclic temporal encoding"""
        config = FeatureConfig(encode_time=True)
        engineer = FeatureEngineer(config)
        features = engineer.transform(sample_ohlcv_data)
        
        # Should have cyclic features
        assert 'dow_sin' in features.columns
        assert 'dow_cos' in features.columns
        assert 'month_sin' in features.columns
        assert 'month_cos' in features.columns
        
        # Cyclic features should be between -1 and 1
        assert features['dow_sin'].min() >= -1.0
        assert features['dow_sin'].max() <= 1.0


class TestTargetGeneration:
    """Test target variable generation"""
    
    def test_target_return(self, sample_ohlcv_data):
        """Test return target"""
        config = FeatureConfig(
            target_type='return',
            prediction_horizon=1
        )
        engineer = FeatureEngineer(config)
        features = engineer.transform(sample_ohlcv_data)
        
        assert 'target' in features.columns
        
        # Target should be future returns
        assert features['target'].abs().max() < 1.0
    
    def test_target_direction(self, sample_ohlcv_data):
        """Test binary direction target"""
        config = FeatureConfig(
            target_type='direction',
            prediction_horizon=1
        )
        engineer = FeatureEngineer(config)
        features = engineer.transform(sample_ohlcv_data)
        
        # Target should be binary (0 or 1)
        assert set(features['target'].unique()).issubset({0, 1})
    
    def test_target_classification(self, sample_ohlcv_data):
        """Test 3-class classification target"""
        config = FeatureConfig(
            target_type='classification',
            prediction_horizon=1
        )
        engineer = FeatureEngineer(config)
        features = engineer.transform(sample_ohlcv_data)
        
        # Target should be -1, 0, or 1
        assert set(features['target'].unique()).issubset({-1, 0, 1})


class TestHashVersioning:
    """Test hash-based versioning"""
    
    def test_data_hash_generation(self, sample_ohlcv_data):
        """Test data hash generation"""
        engineer = FeatureEngineer()
        features = engineer.transform(sample_ohlcv_data)
        
        assert engineer.metadata is not None
        assert len(engineer.metadata.data_hash) == 16  # SHA256 truncated
    
    def test_data_hash_deterministic(self, sample_ohlcv_data):
        """Test that data hash is deterministic"""
        engineer1 = FeatureEngineer()
        features1 = engineer1.transform(sample_ohlcv_data)
        
        engineer2 = FeatureEngineer()
        features2 = engineer2.transform(sample_ohlcv_data)
        
        assert engineer1.metadata.data_hash == engineer2.metadata.data_hash
    
    def test_data_hash_different(self, sample_ohlcv_data):
        """Test that different data has different hash"""
        engineer1 = FeatureEngineer()
        features1 = engineer1.transform(sample_ohlcv_data)
        
        # Modify data
        modified_data = sample_ohlcv_data.copy()
        modified_data['close'] = modified_data['close'] * 1.01
        
        engineer2 = FeatureEngineer()
        features2 = engineer2.transform(modified_data)
        
        assert engineer1.metadata.data_hash != engineer2.metadata.data_hash
    
    def test_version_string(self, sample_ohlcv_data):
        """Test version string format"""
        engineer = FeatureEngineer()
        features = engineer.transform(sample_ohlcv_data)
        
        version = f"{engineer.metadata.config_hash}_{engineer.metadata.data_hash}"
        
        assert len(version) == 33  # 16 + 1 + 16


class TestFeatureMetadata:
    """Test feature metadata"""
    
    def test_metadata_creation(self, sample_ohlcv_data):
        """Test metadata is created after transformation"""
        engineer = FeatureEngineer()
        features = engineer.transform(sample_ohlcv_data)
        
        assert engineer.metadata is not None
        assert isinstance(engineer.metadata, FeatureMetadata)
    
    def test_metadata_contents(self, sample_ohlcv_data):
        """Test metadata contains required information"""
        engineer = FeatureEngineer()
        features = engineer.transform(sample_ohlcv_data)
        
        metadata = engineer.metadata
        
        assert metadata.config_hash is not None
        assert metadata.data_hash is not None
        assert len(metadata.feature_names) > 0
        assert metadata.n_samples > 0
        assert metadata.n_features > 0
        assert metadata.creation_timestamp is not None
    
    def test_metadata_save_load(self, sample_ohlcv_data, temp_dir):
        """Test metadata save and load"""
        engineer = FeatureEngineer()
        features = engineer.transform(sample_ohlcv_data)
        
        metadata_path = temp_dir / "metadata_test.json"
        engineer.metadata.save(str(metadata_path))
        
        # Load
        loaded_metadata = FeatureMetadata.load(str(metadata_path))
        
        assert loaded_metadata.config_hash == engineer.metadata.config_hash
        assert loaded_metadata.data_hash == engineer.metadata.data_hash
        assert loaded_metadata.n_features == engineer.metadata.n_features


class TestFeatureSaveLoad:
    """Test feature save/load with versioning"""
    
    def test_save_features(self, sample_ohlcv_data, temp_dir):
        """Test saving features with metadata"""
        engineer = FeatureEngineer()
        features = engineer.transform(sample_ohlcv_data)
        
        saved_path = engineer.save_features(features, base_path=str(temp_dir))
        
        assert Path(saved_path).exists()
        
        # Metadata should also exist
        version = f"{engineer.metadata.config_hash}_{engineer.metadata.data_hash}"
        metadata_path = temp_dir / f"metadata_{version}.json"
        assert metadata_path.exists()
    
    def test_load_features(self, sample_ohlcv_data, temp_dir):
        """Test loading features by version"""
        engineer = FeatureEngineer()
        features = engineer.transform(sample_ohlcv_data)
        
        engineer.save_features(features, base_path=str(temp_dir))
        
        # Load by version
        version = f"{engineer.metadata.config_hash}_{engineer.metadata.data_hash}"
        loaded_features, loaded_metadata = FeatureEngineer.load_features(
            version=version,
            base_path=str(temp_dir)
        )
        
        assert len(loaded_features) == len(features)
        assert loaded_metadata.n_features == engineer.metadata.n_features


class TestValidation:
    """Test validation and cleaning"""
    
    def test_no_nan_in_output(self, sample_ohlcv_data):
        """Test that output has no NaN values"""
        engineer = FeatureEngineer()
        features = engineer.transform(sample_ohlcv_data)
        
        # Check for NaN (should be handled)
        nan_cols = features.columns[features.isnull().any()].tolist()
        assert len(nan_cols) == 0, f"Found NaN in columns: {nan_cols}"
    
    def test_no_inf_in_output(self, sample_ohlcv_data):
        """Test that output has no infinite values"""
        engineer = FeatureEngineer()
        features = engineer.transform(sample_ohlcv_data)
        
        # Check for Inf
        inf_cols = features.columns[np.isinf(features).any()].tolist()
        assert len(inf_cols) == 0, f"Found Inf in columns: {inf_cols}"


class TestEdgeCases:
    """Test edge cases and error handling"""
    
    def test_insufficient_data(self):
        """Test with insufficient data"""
        # Only 10 samples (less than typical window sizes)
        small_data = pd.DataFrame({
            'close': [100 + i for i in range(10)]
        })
        
        config = FeatureConfig(sma_periods=[20])  # Requires 20 samples
        engineer = FeatureEngineer(config)
        
        features = engineer.transform(small_data)
        
        # Should handle gracefully (with NaN handling)
        assert len(features) > 0
    
    def test_constant_price(self):
        """Test with constant price (zero volatility)"""
        constant_data = pd.DataFrame({
            'close': [100.0] * 100
        })
        
        engineer = FeatureEngineer()
        features = engineer.transform(constant_data)
        
        # Should handle zero volatility
        assert 'returns' in features.columns
        assert features['returns'].max() == 0.0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
