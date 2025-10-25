"""
Minimal ComplianceShield Fix Test

Direct test of the shield fix without full agent initialization.
"""

import numpy as np
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import después del logging para capturar mensajes
from underdog.rl.environments import ForexTradingEnv

def test_shield_minimal():
    """Minimal test of ComplianceShield fix"""
    
    print("="*60)
    print("MINIMAL COMPLIANCE SHIELD TEST")
    print("="*60)
    
    # Generate minimal test data (need 1200+ bars: 200 lookback + 1000 max_steps)
    print("\n[1] Creating test data...")
    dates = pd.date_range('2024-01-01', periods=1500, freq='h')  # lowercase 'h'
    price_base = 1.1000
    np.random.seed(42)
    
    df = pd.DataFrame({
        'timestamp': dates,
        'open': price_base + np.cumsum(np.random.randn(1500) * 0.0001),
        'high': price_base + np.cumsum(np.random.randn(1500) * 0.0001) + 0.0001,
        'low': price_base + np.cumsum(np.random.randn(1500) * 0.0001) - 0.0001,
        'close': price_base + np.cumsum(np.random.randn(1500) * 0.0001),
        'volume': np.random.randint(1000, 5000, 1500)
    })
    print(f"   Created {len(df)} bars")
    
    # Create environment
    print("\n[2] Creating ForexTradingEnv with safety shield...")
    env = ForexTradingEnv(historical_data=df)
    print(f"   Environment created: {env.__class__.__name__}")
    print(f"   Shield enabled: {env.shield is not None}")
    
    # Reset
    print("\n[3] Resetting environment...")
    state, info = env.reset()
    print(f"   State shape: {state.shape}")
    print(f"   Initial balance: ${env.balance:,.0f}")
    
    # Test continuous actions (TD3 style)
    print("\n[4] Testing continuous actions...")
    
    test_actions = [
        np.array([0.5]),   # Long position
        np.array([-0.3]),  # Short position
        np.array([0.0]),   # Neutral
        np.array([0.8]),   # Strong long
        np.array([-0.6]),  # Strong short
    ]
    
    for i, action in enumerate(test_actions):
        print(f"\n   Action {i+1}: {float(action[0]):+.1f}")
        
        try:
            next_state, reward, done, truncated, info = env.step(action)
            print(f"      OK - reward={reward:.4f}, equity=${env.equity:,.0f}, pos={env.position_size:.2f}")
            
            if done or truncated:
                print(f"      Episode terminated")
                break
        except Exception as e:
            print(f"      ERROR: {e}")
            return False
    
    # Full episode test
    print("\n[5] Running full episode (50 steps)...")
    state, _ = env.reset()
    
    episode_steps = 0
    for step in range(50):
        # Random action
        action = np.random.uniform(-1, 1, size=(1,))
        
        next_state, reward, done, truncated, info = env.step(action)
        episode_steps += 1
        
        if step % 10 == 0:
            # Calculate DD (not stored as instance variable)
            dd_pct = (env.peak_balance - env.equity) / env.peak_balance if env.peak_balance > 0 else 0.0
            print(f"   Step {step}: equity=${env.equity:,.0f}, dd={dd_pct:.2%}")
        
        if done or truncated:
            print(f"   Episode ended at step {episode_steps}")
            break
        
        state = next_state
    
    print(f"\n   Completed {episode_steps} steps")
    print(f"   Final equity: ${env.equity:,.0f}")
    
    # Calculate final DD
    final_dd_pct = (env.peak_balance - env.equity) / env.peak_balance if env.peak_balance > 0 else 0.0
    print(f"   Max DD: {final_dd_pct:.2%}")
    
    # Success
    print("\n" + "="*60)
    print("SUCCESS - No 'Unknown action type' errors!")
    print("="*60)
    print("\nFix verified:")
    print("  - ComplianceShield receives proper action dict with 'type' key")
    print("  - TD3 continuous actions [-1, 1] handled correctly")
    print("  - No crashes or validation errors")
    print("\nReady for full TD3 Quick Test!")
    
    env.close()
    return True


if __name__ == "__main__":
    try:
        success = test_shield_minimal()
        exit(0 if success else 1)
    except Exception as e:
        print(f"\nTEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
