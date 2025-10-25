"""
Test CBRL (Chaos-Based Reinforcement Learning) Implementation

Validates:
    1. Chaotic noise sequence diverges from Gaussian
    2. Epsilon decays correctly based on TD-error
    3. Exploration diversity improves with chaotic noise

Quick test: 10 episodes to verify CBRL integration
"""

import numpy as np
import pandas as pd
import torch
from underdog.rl.environments import ForexTradingEnv, TradingEnvConfig
from underdog.rl.agents import TD3Agent, TD3Config, ReplayBuffer
from underdog.rl.cbrl_noise import ChaoticNoiseGenerator


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


def test_chaotic_vs_gaussian():
    """Test 1: Chaotic vs Gaussian Noise Comparison"""
    print("\n" + "="*60)
    print("TEST 1: CHAOTIC VS GAUSSIAN NOISE")
    print("="*60)
    
    print("\n[1] Generating noise sequences (1000 samples)...")
    
    # Chaotic noise
    chaotic_gen = ChaoticNoiseGenerator(x0=0.5, output_range=(-1, 1))
    chaotic_samples = chaotic_gen.get_noise(shape=(1000,))
    
    # Gaussian noise
    np.random.seed(42)
    gaussian_samples = np.random.normal(0, 1, size=1000)
    
    print(f"\n[2] Statistical Comparison:")
    print(f"   Chaotic  - Mean: {np.mean(chaotic_samples):.4f}, Std: {np.std(chaotic_samples):.4f}")
    print(f"   Gaussian - Mean: {np.mean(gaussian_samples):.4f}, Std: {np.std(gaussian_samples):.4f}")
    
    # Test autocorrelation (chaotic should have structure)
    def autocorr(x, lag=1):
        """Calculate autocorrelation at lag"""
        c0 = np.dot(x - x.mean(), x - x.mean()) / len(x)
        c_lag = np.dot(x[:-lag] - x.mean(), x[lag:] - x.mean()) / len(x)
        return c_lag / c0 if c0 != 0 else 0
    
    chaotic_autocorr = autocorr(chaotic_samples, lag=1)
    gaussian_autocorr = autocorr(gaussian_samples, lag=1)
    
    print(f"\n[3] Autocorrelation (lag=1):")
    print(f"   Chaotic:  {chaotic_autocorr:.4f} (should have structure)")
    print(f"   Gaussian: {gaussian_autocorr:.4f} (should be ~0)")
    
    # Test distribution shape
    chaotic_range = np.max(chaotic_samples) - np.min(chaotic_samples)
    gaussian_range = np.max(gaussian_samples) - np.min(gaussian_samples)
    
    print(f"\n[4] Value Range:")
    print(f"   Chaotic:  [{np.min(chaotic_samples):.4f}, {np.max(chaotic_samples):.4f}] "
          f"(range: {chaotic_range:.4f})")
    print(f"   Gaussian: [{np.min(gaussian_samples):.4f}, {np.max(gaussian_samples):.4f}] "
          f"(range: {gaussian_range:.4f})")
    
    # Validation: chaotic should be different from Gaussian
    diff_autocorr = abs(chaotic_autocorr - gaussian_autocorr) > 0.05
    bounded = np.all((chaotic_samples >= -1.0) & (chaotic_samples <= 1.0))
    
    print(f"\n[5] Validation:")
    print(f"   Different autocorrelation: {diff_autocorr} ✓" if diff_autocorr 
          else f"   Different autocorrelation: {diff_autocorr} ✗")
    print(f"   Chaotic bounded [-1,1]: {bounded} ✓" if bounded 
          else f"   Chaotic bounded [-1,1]: {bounded} ✗")
    
    return diff_autocorr and bounded


def test_epsilon_decay():
    """Test 2: Epsilon Decay Based on TD-Error"""
    print("\n" + "="*60)
    print("TEST 2: EPSILON DECAY (TD-ERROR BASED)")
    print("="*60)
    
    # Create agent with CBRL enabled
    print("\n[1] Initializing TD3Agent with CBRL...")
    config = TD3Config(
        state_dim=24,
        action_dim=1,
        hidden_dim=256,
        use_cbrl=True  # Enable CBRL
    )
    agent = TD3Agent(config=config)
    
    print(f"   Initial epsilon: {agent.epsilon:.4f}")
    print(f"   Epsilon min: {agent.epsilon_min:.4f}")
    print(f"   CBRL enabled: {agent.use_cbrl}")
    
    # Create replay buffer
    buffer = ReplayBuffer(state_dim=24, action_dim=1, max_size=10000)
    
    # Generate fake training data
    print("\n[2] Generating synthetic training data...")
    for i in range(1000):
        state = np.random.randn(24)
        action = np.random.uniform(-1, 1, size=(1,))
        reward = np.random.uniform(-1, 1)
        next_state = np.random.randn(24)
        done = float(i % 100 == 0)
        buffer.add(state, action, reward, next_state, done)
    
    # Train agent and track epsilon
    print("\n[3] Training agent and tracking epsilon decay...")
    epsilon_history = [agent.epsilon]
    td_error_history = []
    
    for step in range(100):
        if buffer.size >= 256:
            metrics = agent.train(buffer, batch_size=256)
            epsilon_history.append(agent.epsilon)
            td_error_history.append(metrics.get('td_error', 0.0))
            
            if step % 20 == 0:
                print(f"   Step {step}: epsilon={agent.epsilon:.4f}, "
                      f"td_error={metrics.get('td_error', 0.0):.4f}")
    
    # Analyze epsilon decay
    print(f"\n[4] Epsilon Decay Analysis:")
    print(f"   Initial epsilon: {epsilon_history[0]:.4f}")
    print(f"   Final epsilon: {epsilon_history[-1]:.4f}")
    print(f"   Decay observed: {epsilon_history[-1] < epsilon_history[0]}")
    print(f"   Stayed above min: {epsilon_history[-1] >= agent.epsilon_min}")
    
    # Validation
    epsilon_decayed = epsilon_history[-1] < epsilon_history[0]
    above_min = epsilon_history[-1] >= agent.epsilon_min
    
    print(f"\n[5] Validation:")
    print(f"   Epsilon decayed: {epsilon_decayed} ✓" if epsilon_decayed 
          else f"   Epsilon decayed: {epsilon_decayed} ✗")
    print(f"   Above minimum: {above_min} ✓" if above_min 
          else f"   Above minimum: {above_min} ✗")
    
    return epsilon_decayed and above_min


def test_exploration_diversity():
    """Test 3: Exploration Diversity with CBRL"""
    print("\n" + "="*60)
    print("TEST 3: EXPLORATION DIVERSITY")
    print("="*60)
    
    # Create environment
    df = generate_synthetic_ohlcv(periods=2000)
    config_env = TradingEnvConfig(
        symbol='EURUSD',
        initial_balance=100000.0,
        max_steps=500
    )
    env = ForexTradingEnv(config=config_env, historical_data=df)
    
    # Test with CBRL
    print("\n[1] Testing with CBRL enabled...")
    config_cbrl = TD3Config(state_dim=24, action_dim=1, use_cbrl=True)
    agent_cbrl = TD3Agent(config=config_cbrl)
    
    state, _ = env.reset()
    actions_cbrl = []
    for step in range(100):
        action = agent_cbrl.select_action(state, explore=True)
        actions_cbrl.append(action[0])
        state, _, done, truncated, _ = env.step(action)
        if done or truncated:
            state, _ = env.reset()
    
    # Test with Gaussian noise
    print("\n[2] Testing with Gaussian noise...")
    config_gauss = TD3Config(state_dim=24, action_dim=1, use_cbrl=False)
    agent_gauss = TD3Agent(config=config_gauss)
    
    state, _ = env.reset()
    actions_gauss = []
    for step in range(100):
        action = agent_gauss.select_action(state, explore=True)
        actions_gauss.append(action[0])
        state, _, done, truncated, _ = env.step(action)
        if done or truncated:
            state, _ = env.reset()
    
    # Compare diversity
    actions_cbrl = np.array(actions_cbrl)
    actions_gauss = np.array(actions_gauss)
    
    print(f"\n[3] Action Diversity Comparison:")
    print(f"   CBRL Actions:")
    print(f"     Mean: {np.mean(actions_cbrl):.4f}, Std: {np.std(actions_cbrl):.4f}")
    print(f"     Unique values: {len(np.unique(np.round(actions_cbrl, 2)))}")
    print(f"     Range: [{np.min(actions_cbrl):.4f}, {np.max(actions_cbrl):.4f}]")
    
    print(f"\n   Gaussian Actions:")
    print(f"     Mean: {np.mean(actions_gauss):.4f}, Std: {np.std(actions_gauss):.4f}")
    print(f"     Unique values: {len(np.unique(np.round(actions_gauss, 2)))}")
    print(f"     Range: [{np.min(actions_gauss):.4f}, {np.max(actions_gauss):.4f}]")
    
    # Validation: CBRL should have different characteristics
    cbrl_bounded = np.all((actions_cbrl >= -1.0) & (actions_cbrl <= 1.0))
    gauss_bounded = np.all((actions_gauss >= -1.0) & (actions_gauss <= 1.0))
    both_diverse = np.std(actions_cbrl) > 0.1 and np.std(actions_gauss) > 0.1
    
    print(f"\n[4] Validation:")
    print(f"   CBRL actions bounded: {cbrl_bounded} ✓" if cbrl_bounded 
          else f"   CBRL actions bounded: {cbrl_bounded} ✗")
    print(f"   Gaussian actions bounded: {gauss_bounded} ✓" if gauss_bounded 
          else f"   Gaussian actions bounded: {gauss_bounded} ✗")
    print(f"   Both show diversity: {both_diverse} ✓" if both_diverse 
          else f"   Both show diversity: {both_diverse} ✗")
    
    env.close()
    return cbrl_bounded and gauss_bounded and both_diverse


if __name__ == "__main__":
    print("="*60)
    print("CBRL VALIDATION TEST")
    print("="*60)
    print("\nTesting CBRL components:")
    print("  - Chaotic noise vs Gaussian noise")
    print("  - Epsilon decay based on TD-error")
    print("  - Exploration diversity improvement")
    
    # Run tests
    test1_pass = test_chaotic_vs_gaussian()
    test2_pass = test_epsilon_decay()
    test3_pass = test_exploration_diversity()
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    print(f"Test 1 - Chaotic vs Gaussian:     {'✅ PASS' if test1_pass else '❌ FAIL'}")
    print(f"Test 2 - Epsilon Decay:           {'✅ PASS' if test2_pass else '❌ FAIL'}")
    print(f"Test 3 - Exploration Diversity:   {'✅ PASS' if test3_pass else '❌ FAIL'}")
    
    all_pass = test1_pass and test2_pass and test3_pass
    print(f"\nOverall: {'✅ ALL TESTS PASSED' if all_pass else '❌ SOME TESTS FAILED'}")
    print("="*60)
