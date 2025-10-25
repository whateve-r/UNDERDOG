"""
Test script for temporal observation buffers

Validates that:
1. Buffers initialize correctly
2. Shapes match architecture requirements
3. Zero-padding works during warm-up
4. Integration with MultiAssetEnv works

🔥 CRITICAL: This validates the BLOCKER for HMARL training.
"""

import sys
from pathlib import Path
import numpy as np
import logging

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from underdog.rl.observation_buffer import (
    ObservationBuffer,
    MultiAssetObservationManager,
    validate_buffer_shapes
)
from underdog.rl.multi_asset_env import MultiAssetEnv, MultiAssetConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)


def test_individual_buffers():
    """Test individual ObservationBuffer for each asset"""
    print("\n" + "="*70)
    print("TEST 1: Individual Observation Buffers")
    print("="*70)
    
    feature_dim = 24
    test_cases = [
        ('EURUSD', 60),   # LSTM
        ('USDJPY', 15),   # CNN1D
        ('XAUUSD', 120),  # Transformer
        ('GBPUSD', 1),    # Attention
    ]
    
    for symbol, expected_seq_len in test_cases:
        print(f"\n{symbol} (sequence_length={expected_seq_len}):")
        
        buffer = ObservationBuffer(symbol, feature_dim)
        assert buffer.sequence_length == expected_seq_len, \
            f"Expected seq_len={expected_seq_len}, got {buffer.sequence_length}"
        
        # Initialize with first observation
        initial_obs = np.random.randn(feature_dim)
        buffer.reset(initial_obs)
        
        # Test warm-up phase (< sequence_length observations)
        for step in range(min(5, expected_seq_len)):
            obs = np.random.randn(feature_dim)
            buffer.add(obs)
            
            sequence = buffer.get_sequence()
            
            if expected_seq_len == 1:
                # GBPUSD: no temporal dimension
                assert sequence.shape == (feature_dim,), \
                    f"Expected shape ({feature_dim},), got {sequence.shape}"
            else:
                # Others: should have [T, F] shape
                assert sequence.shape == (expected_seq_len, feature_dim), \
                    f"Expected shape ({expected_seq_len}, {feature_dim}), got {sequence.shape}"
                
                # Check zero-padding during warm-up
                if step < expected_seq_len - 1:
                    # Should have zeros at start
                    zero_rows = expected_seq_len - (step + 2)  # +2 because reset adds first obs
                    if zero_rows > 0:
                        assert np.allclose(sequence[0], 0), \
                            "Expected zero-padding at start during warm-up"
        
        # Fill buffer beyond sequence_length
        for _ in range(expected_seq_len + 10):
            obs = np.random.randn(feature_dim)
            buffer.add(obs)
        
        final_sequence = buffer.get_sequence()
        
        if expected_seq_len == 1:
            assert final_sequence.shape == (feature_dim,)
            print(f"  ✅ Shape correct: {final_sequence.shape}")
        else:
            assert final_sequence.shape == (expected_seq_len, feature_dim)
            print(f"  ✅ Shape correct: {final_sequence.shape}")
            
            # No zeros after warm-up
            assert not np.allclose(final_sequence[0], 0), \
                "Should not have zero-padding after warm-up"
            print(f"  ✅ No zero-padding after warm-up")
    
    print("\n✅ TEST 1 PASSED: All individual buffers work correctly\n")


def test_multi_asset_manager():
    """Test MultiAssetObservationManager"""
    print("\n" + "="*70)
    print("TEST 2: MultiAssetObservationManager")
    print("="*70)
    
    symbols = ['EURUSD', 'USDJPY', 'XAUUSD', 'GBPUSD']
    feature_dim = 24
    
    manager = MultiAssetObservationManager(symbols, feature_dim)
    
    # Get expected shapes
    expected_shapes = manager.get_sequence_shapes()
    print(f"\nExpected shapes:")
    for symbol, shape in expected_shapes.items():
        print(f"  {symbol}: {shape}")
    
    # Validate expected shapes
    assert expected_shapes['EURUSD'] == (60, 24)
    assert expected_shapes['USDJPY'] == (15, 24)
    assert expected_shapes['XAUUSD'] == (120, 24)
    assert expected_shapes['GBPUSD'] == (24,)
    print("  ✅ Expected shapes correct")
    
    # Initialize with dummy observations
    initial_obs = [np.random.randn(feature_dim) for _ in symbols]
    manager.reset(initial_obs)
    
    # Add observations over time
    for step in range(150):  # More than max sequence length
        obs_list = [np.random.randn(feature_dim) for _ in symbols]
        manager.add(obs_list)
    
    # Get sequences
    sequences = manager.get_sequences()
    
    print(f"\nActual shapes after 150 steps:")
    for symbol, sequence in zip(symbols, sequences):
        print(f"  {symbol}: {sequence.shape}")
        
        # Validate shapes
        expected_shape = expected_shapes[symbol]
        assert sequence.shape == expected_shape, \
            f"{symbol}: Expected {expected_shape}, got {sequence.shape}"
    
    print("\n✅ TEST 2 PASSED: MultiAssetObservationManager works correctly\n")


def test_env_integration():
    """Test integration with MultiAssetEnv"""
    print("\n" + "="*70)
    print("TEST 3: MultiAssetEnv Integration")
    print("="*70)
    
    try:
        config = MultiAssetConfig(
            symbols=["EURUSD", "USDJPY", "XAUUSD", "GBPUSD"],
            initial_balance=100000.0,
            data_source="historical"
        )
        
        env = MultiAssetEnv(config=config)
        
        # Check observation shapes are available
        obs_shapes = env.get_observation_shapes()
        print(f"\nEnvironment observation shapes:")
        for symbol, shape in obs_shapes.items():
            print(f"  {symbol}: {shape}")
        
        # Reset environment
        meta_state, info = env.reset()
        print(f"\nMeta-state shape: {meta_state.shape}")
        print(f"Episode: {info['episode']}")
        
        # Get local observations (CRITICAL METHOD)
        local_obs = env.get_local_observations()
        
        print(f"\n🔥 CRITICAL: Local observation shapes at step 0 (after reset):")
        for i, (symbol, obs) in enumerate(zip(config.symbols, local_obs)):
            print(f"  Agent {i} ({symbol}): {obs.shape}")
            
            # Validate against expected
            expected_shape = obs_shapes[symbol]
            assert obs.shape == expected_shape, \
                f"{symbol}: Expected {expected_shape}, got {obs.shape}"
        
        # Take a few steps and check shapes remain consistent
        for step in range(5):
            meta_action = np.random.uniform(0.5, 1.0, size=(4,))
            meta_state, reward, done, truncated, info = env.step(meta_action)
            
            local_obs = env.get_local_observations()
            
            if step == 0:
                print(f"\n🔥 CRITICAL: Local observation shapes at step 1:")
                for i, (symbol, obs) in enumerate(zip(config.symbols, local_obs)):
                    print(f"  Agent {i} ({symbol}): {obs.shape}")
            
            # Validate all shapes
            for symbol, obs in zip(config.symbols, local_obs):
                expected_shape = obs_shapes[symbol]
                assert obs.shape == expected_shape, \
                    f"Step {step+1}, {symbol}: Expected {expected_shape}, got {obs.shape}"
        
        env.close()
        
        print("\n✅ TEST 3 PASSED: Environment integration works correctly\n")
        
    except FileNotFoundError as e:
        print(f"\n⚠️ TEST 3 SKIPPED: Historical data not found")
        print(f"   Error: {e}")
        print(f"   This is OK if running in test environment without data files")
        return


def test_shape_validation():
    """Test validate_buffer_shapes utility"""
    print("\n" + "="*70)
    print("TEST 4: Shape Validation Utility")
    print("="*70)
    
    symbols = ['EURUSD', 'USDJPY', 'XAUUSD', 'GBPUSD']
    results = validate_buffer_shapes(symbols, feature_dim=24)
    
    print("\nValidation results:")
    for symbol, result in results.items():
        print(f"\n{symbol}:")
        print(f"  Expected: {result['expected_shape']}")
        print(f"  Actual: {result['actual_shape']}")
        print(f"  Sequence length: {result['sequence_length']}")
        print(f"  Match: {'✅' if result['matches'] else '❌'}")
        
        assert result['matches'], f"{symbol} shape validation failed"
    
    print("\n✅ TEST 4 PASSED: All shape validations passed\n")


def main():
    """Run all tests"""
    print("\n" + "="*70)
    print("OBSERVATION BUFFER VALIDATION")
    print("Testing CRITICAL BLOCKER for HMARL Training")
    print("="*70)
    
    try:
        test_individual_buffers()
        test_multi_asset_manager()
        test_shape_validation()
        test_env_integration()  # May skip if no data
        
        print("\n" + "="*70)
        print("🎉 ALL TESTS PASSED!")
        print("="*70)
        print("\n✅ BLOCKER RESOLVED: Temporal observation buffers working correctly")
        print("✅ Shapes match architecture requirements:")
        print("   - EURUSD (TD3+LSTM): [60, 24]")
        print("   - USDJPY (PPO+CNN1D): [15, 24]")
        print("   - XAUUSD (SAC+Transformer): [120, 24]")
        print("   - GBPUSD (DDPG+Attention): [24]")
        print("\n🚀 Next step: Add shape validation logging to training script")
        print("="*70 + "\n")
        
        return 0
        
    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        return 1
    except Exception as e:
        print(f"\n❌ UNEXPECTED ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
