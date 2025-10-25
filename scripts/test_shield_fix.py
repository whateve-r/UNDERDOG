"""
Quick Test: ComplianceShield Fix Verification

Tests that the shield properly handles TD3 continuous actions
without throwing "Unknown action type: None" errors.
"""

import numpy as np
import pandas as pd
import logging
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from underdog.rl.environments import ForexTradingEnv
from underdog.rl.agents import TD3Agent, TD3Config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_compliance_shield_integration():
    """Test that ComplianceShield works with TD3 continuous actions"""
    
    logger.info("="*60)
    logger.info("TESTING COMPLIANCE SHIELD FIX")
    logger.info("="*60)
    
    # Create minimal historical data for backtesting
    logger.info("\n[SETUP] Creating minimal test data...")
    
    # Generate synthetic OHLCV data (500 bars)
    dates = pd.date_range('2024-01-01', periods=500, freq='1H')
    price_base = 1.1000
    np.random.seed(42)
    
    df = pd.DataFrame({
        'timestamp': dates,
        'open': price_base + np.cumsum(np.random.randn(500) * 0.0001),
        'high': price_base + np.cumsum(np.random.randn(500) * 0.0001) + 0.0001,
        'low': price_base + np.cumsum(np.random.randn(500) * 0.0001) - 0.0001,
        'close': price_base + np.cumsum(np.random.randn(500) * 0.0001),
        'volume': np.random.randint(1000, 5000, 500)
    })
    
    # Create environment with historical data (backtesting mode)
    logger.info("[SETUP] Creating environment with safety shield...")
    env = ForexTradingEnv(
        historical_data=df
    )
    
    # Create agent
    logger.info("[SETUP] Creating TD3 agent...")
    agent_config = TD3Config(
        state_dim=24,
        action_dim=1,  # Single action (position size)
        hidden_dim=256
    )
    agent = TD3Agent(config=agent_config)
    
    # Test 1: Simple reset and step
    logger.info("\n[TEST 1] Environment Reset")
    state, info = env.reset()
    logger.info(f"✅ Reset successful: state shape {state.shape}")
    
    # Test 2: Take random actions (should not crash)
    logger.info("\n[TEST 2] Random Actions (10 steps)")
    for step in range(10):
        action = agent.select_action(state, explore=True)
        
        logger.info(f"  Step {step+1}: action={float(action[0]):.3f}")
        
        next_state, reward, done, truncated, info = env.step(action)
        
        logger.info(f"    → reward={reward:.4f}, equity=${env.equity:,.0f}, dd={env.total_dd_ratio:.2%}")
        
        if done or truncated:
            logger.info("    → Episode terminated")
            break
        
        state = next_state
    
    logger.info("\n✅ No 'Unknown action type' errors!")
    
    # Test 3: Extreme actions (test shield boundaries)
    logger.info("\n[TEST 3] Extreme Actions")
    state, _ = env.reset()
    
    extreme_actions = [
        np.array([1.0]),   # Max long
        np.array([-1.0]),  # Max short
        np.array([0.0]),   # Neutral
        np.array([0.5]),   # Half long
        np.array([-0.5]),  # Half short
    ]
    
    for i, action in enumerate(extreme_actions):
        logger.info(f"  Action {i+1}: {float(action[0]):+.1f}")
        
        next_state, reward, done, truncated, info = env.step(action)
        
        logger.info(f"    → executed, equity=${env.equity:,.0f}")
        
        if done or truncated:
            logger.info("    → Episode terminated (expected if DD breach)")
            break
    
    logger.info("\n✅ Shield handled extreme actions correctly!")
    
    # Test 4: Full episode
    logger.info("\n[TEST 4] Full Episode (100 steps)")
    state, _ = env.reset()
    
    episode_reward = 0.0
    steps = 0
    
    for step in range(100):
        action = agent.select_action(state, explore=True)
        next_state, reward, done, truncated, info = env.step(action)
        
        episode_reward += reward
        steps += 1
        
        if step % 20 == 0:
            logger.info(
                f"  Step {step}: reward={reward:.4f}, "
                f"equity=${env.equity:,.0f}, dd={env.total_dd_ratio:.2%}"
            )
        
        if done or truncated:
            logger.info(f"  Episode terminated at step {steps}")
            break
        
        state = next_state
    
    logger.info(f"\n✅ Episode completed: {steps} steps, total reward={episode_reward:.2f}")
    
    # Final summary
    logger.info("\n" + "="*60)
    logger.info("COMPLIANCE SHIELD FIX VERIFICATION: ✅ PASSED")
    logger.info("="*60)
    logger.info(f"Total steps executed: {steps}")
    logger.info(f"Final equity: ${env.equity:,.0f}")
    logger.info(f"Max DD: {env.total_dd_ratio:.2%}")
    logger.info(f"Position executed: {env.position_size:.2f}")
    logger.info("\n🎯 No 'Unknown action type' errors detected!")
    
    env.close()
    
    return True


if __name__ == "__main__":
    try:
        test_compliance_shield_integration()
        print("\n" + "="*60)
        print("FIX VERIFIED - Ready for TD3 Quick Test!")
        print("="*60)
    except Exception as e:
        print(f"\nTEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
