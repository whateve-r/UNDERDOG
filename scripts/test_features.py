"""
Test Feature Engineering Improvements

Validates:
    1. Turbulence Index Local calculates correctly
    2. Meta-State has 15D and no NaN/Inf
    3. Features improve state representation quality

Quick test: 10 episodes to verify implementation
"""

import numpy as np
import pandas as pd
from underdog.rl.environments import ForexTradingEnv, TradingEnvConfig
from underdog.rl.multi_asset_env import MultiAssetEnv, MultiAssetConfig


def generate_synthetic_ohlcv(periods=2000):
    """Generate synthetic OHLCV data for testing"""
    dates = pd.date_range('2024-01-01', periods=periods, freq='h')  # lowercase 'h'
    
    # Random walk with drift
    np.random.seed(42)
    returns = np.random.normal(0.0001, 0.002, periods)
    price = 1.1000 * np.exp(np.cumsum(returns))
    
    df = pd.DataFrame({
        'time': dates,  # lowercase to match environment expectations
        'open': price * (1 + np.random.uniform(-0.001, 0.001, periods)),
        'high': price * (1 + np.random.uniform(0.0005, 0.002, periods)),
        'low': price * (1 + np.random.uniform(-0.002, -0.0005, periods)),
        'close': price,
        'volume': np.random.randint(100, 1000, periods)
    })
    
    return df


def test_turbulence_index():
    """Test 1: Turbulence Index Local"""
    print("\n" + "="*60)
    print("TEST 1: TURBULENCE INDEX LOCAL")
    print("="*60)
    
    # Create environment
    df = generate_synthetic_ohlcv(periods=2000)
    config = TradingEnvConfig(
        symbol='EURUSD',
        initial_balance=100000.0,
        max_steps=500
    )
    env = ForexTradingEnv(config=config, historical_data=df)
    
    # Reset and collect turbulence values
    state, _ = env.reset()
    turbulences = []
    
    print("\n[1] Running 100 steps to collect turbulence values...")
    for step in range(100):
        action = np.array([0.0])  # Hold position
        state, reward, done, truncated, info = env.step(action)
        
        # Turbulence is at index 21 in 24D state
        turbulence = state[21]
        turbulences.append(turbulence)
        
        if step % 20 == 0:
            print(f"   Step {step}: turbulence={turbulence:.4f}")
        
        if done or truncated:
            break
    
    # Analyze turbulence values
    print(f"\n[2] Turbulence Statistics:")
    print(f"   Mean: {np.mean(turbulences):.4f}")
    print(f"   Std: {np.std(turbulences):.4f}")
    print(f"   Min: {np.min(turbulences):.4f}")
    print(f"   Max: {np.max(turbulences):.4f}")
    print(f"   Range valid: [{np.min(turbulences) >= 0.0}, {np.max(turbulences) <= 1.0}]")
    
    # Validation
    has_nan = np.any(np.isnan(turbulences))
    has_inf = np.any(np.isinf(turbulences))
    in_range = np.all((np.array(turbulences) >= 0.0) & (np.array(turbulences) <= 1.0))
    
    print(f"\n[3] Validation:")
    print(f"   No NaN: {not has_nan} ✓" if not has_nan else f"   No NaN: {not has_nan} ✗")
    print(f"   No Inf: {not has_inf} ✓" if not has_inf else f"   No Inf: {not has_inf} ✗")
    print(f"   In [0,1]: {in_range} ✓" if in_range else f"   In [0,1]: {in_range} ✗")
    
    env.close()
    return not has_nan and not has_inf and in_range


def test_meta_state():
    """Test 2: Meta-State 15D"""
    print("\n" + "="*60)
    print("TEST 2: META-STATE (15D)")
    print("="*60)
    
    # Create MultiAssetEnv (requires historical data for each symbol)
    print("\n[1] Creating synthetic data for 4 symbols...")
    historical_data = {}
    for symbol in ["EURUSD", "GBPUSD", "USDJPY", "USDCHF"]:
        historical_data[symbol] = generate_synthetic_ohlcv(periods=2000)
    
    print("\n[2] Initializing MultiAssetEnv...")
    config = MultiAssetConfig(
        symbols=["EURUSD", "GBPUSD", "USDJPY", "USDCHF"],
        initial_balance=100000.0
    )
    env = MultiAssetEnv(config=config, historical_data=historical_data)
    
    # Reset and check meta-state
    print("\n[3] Testing meta-state generation...")
    meta_state, _ = env.reset()
    
    print(f"   Meta-state shape: {meta_state.shape}")
    print(f"   Expected shape: (15,)")
    print(f"   Shape correct: {meta_state.shape == (15,)}")
    
    # Display meta-state components
    print(f"\n[4] Meta-State Components:")
    print(f"   [0] Global DD:        {meta_state[0]:.4f}")
    print(f"   [1] Total Balance:    {meta_state[1]:.4f}")
    print(f"   [2] Turbulence Global:{meta_state[2]:.4f}")
    print(f"   [3-6] Local DDs:      {meta_state[3:7]}")
    print(f"   [7-10] Local Positions:{meta_state[7:11]}")
    print(f"   [11-14] Local Balances:{meta_state[11:15]}")
    
    # Validation
    has_nan = np.any(np.isnan(meta_state))
    has_inf = np.any(np.isinf(meta_state))
    correct_shape = meta_state.shape == (15,)
    
    print(f"\n[5] Validation:")
    print(f"   Shape (15D): {correct_shape} ✓" if correct_shape else f"   Shape (15D): {correct_shape} ✗")
    print(f"   No NaN: {not has_nan} ✓" if not has_nan else f"   No NaN: {not has_nan} ✗")
    print(f"   No Inf: {not has_inf} ✓" if not has_inf else f"   No Inf: {not has_inf} ✗")
    
    # Run 10 episodes to ensure stability
    print(f"\n[6] Running 10 episodes for stability check...")
    for ep in range(10):
        meta_state, _ = env.reset()
        for step in range(50):
            # Random meta-action (4D risk limits)
            meta_action = np.random.uniform(0.1, 1.0, size=(4,))
            meta_state, reward, done, truncated, info = env.step(meta_action)
            
            if done or truncated:
                break
        
        # Check final state
        if np.any(np.isnan(meta_state)) or np.any(np.isinf(meta_state)):
            print(f"   Episode {ep+1}: FAILED (NaN/Inf detected)")
            env.close()
            return False
        
        if ep % 3 == 0:
            print(f"   Episode {ep+1}: OK (turbulence_global={meta_state[2]:.4f})")
    
    env.close()
    return correct_shape and not has_nan and not has_inf


def test_state_quality():
    """Test 3: State Quality Improvement"""
    print("\n" + "="*60)
    print("TEST 3: STATE QUALITY METRICS")
    print("="*60)
    
    df = generate_synthetic_ohlcv(periods=2000)
    config = TradingEnvConfig(
        symbol='EURUSD',
        initial_balance=100000.0,
        max_steps=500
    )
    env = ForexTradingEnv(config=config, historical_data=df)
    
    # Collect states over 200 steps
    print("\n[1] Collecting 200 states...")
    state, _ = env.reset()
    states = []
    
    for step in range(200):
        action = np.random.uniform(-1, 1, size=(1,))
        state, reward, done, truncated, info = env.step(action)
        states.append(state)
        
        if done or truncated:
            break
    
    states = np.array(states)
    
    # Analyze state diversity
    print(f"\n[2] State Statistics:")
    print(f"   States collected: {len(states)}")
    print(f"   State dimension: {states.shape[1]}D")
    
    # Feature-wise statistics
    turbulence_idx = 21
    print(f"\n[3] Feature Analysis:")
    print(f"   Turbulence [21]: mean={np.mean(states[:, turbulence_idx]):.4f}, "
          f"std={np.std(states[:, turbulence_idx]):.4f}")
    print(f"   Price [0]: mean={np.mean(states[:, 0]):.4f}, "
          f"std={np.std(states[:, 0]):.4f}")
    print(f"   Returns [1]: mean={np.mean(states[:, 1]):.4f}, "
          f"std={np.std(states[:, 1]):.4f}")
    
    # Check state variability (important for learning)
    state_variance = np.var(states, axis=0)
    low_variance_features = np.sum(state_variance < 0.001)
    
    print(f"\n[4] State Variability:")
    print(f"   Low variance features (<0.001): {low_variance_features}/24")
    print(f"   Good variability: {low_variance_features < 5} ✓" if low_variance_features < 5 
          else f"   Good variability: False ✗")
    
    env.close()
    return low_variance_features < 5


if __name__ == "__main__":
    print("="*60)
    print("FEATURE ENGINEERING VALIDATION TEST")
    print("="*60)
    print("\nTesting new features:")
    print("  - Turbulence Index Local (volatility of log-returns)")
    print("  - Meta-State 15D (global metrics for A3C)")
    
    # Run tests
    test1_pass = test_turbulence_index()
    test2_pass = test_meta_state()
    test3_pass = test_state_quality()
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    print(f"Test 1 - Turbulence Index: {'✅ PASS' if test1_pass else '❌ FAIL'}")
    print(f"Test 2 - Meta-State 15D:   {'✅ PASS' if test2_pass else '❌ FAIL'}")
    print(f"Test 3 - State Quality:    {'✅ PASS' if test3_pass else '❌ FAIL'}")
    
    all_pass = test1_pass and test2_pass and test3_pass
    print(f"\nOverall: {'✅ ALL TESTS PASSED' if all_pass else '❌ SOME TESTS FAILED'}")
    print("="*60)
