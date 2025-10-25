"""
POEL Integration Example

Demonstrates how to integrate Purpose-Driven Open-Ended Learning
with existing TD3/PPO/SAC agents.
"""

import numpy as np
from underdog.rl.poel import (
    POELRewardShaper,
    PurposeFunction,
    BusinessPurpose,
    DistanceMetric,
)


def integrate_poel_with_agent(
    agent,
    state_dim: int,
    action_dim: int,
    initial_balance: float = 25000.0,
):
    """
    Integrate POEL reward shaping with existing agent.
    
    Args:
        agent: TD3Agent, PPOAgent, or SACAgent instance
        state_dim: Flattened state dimension
        action_dim: Action dimension
        initial_balance: Starting balance for agent
        
    Returns:
        POELRewardShaper instance configured for agent
    """
    
    # Create POEL reward shaper
    poel = POELRewardShaper(
        state_dim=state_dim,
        action_dim=action_dim,
        alpha=0.7,  # 70% PnL focus, 30% exploration
        beta=1.0,   # Moderate stability penalty
        novelty_scale=100.0,  # Scale to match PnL magnitude
        novelty_metric=DistanceMetric.L2,  # Euclidean distance
        max_local_dd_pct=0.15,  # 15% local DD limit
        initial_balance=initial_balance,
    )
    
    return poel


def training_step_with_poel(
    agent,
    poel: POELRewardShaper,
    state: np.ndarray,
    action: np.ndarray,
    next_state: np.ndarray,
    raw_reward: float,
    done: bool,
    new_balance: float,
    is_new_day: bool = False,
):
    """
    Execute training step with POEL-enriched rewards.
    
    Args:
        agent: RL agent instance
        poel: POELRewardShaper instance
        state: Current state
        action: Action taken
        next_state: Next state
        raw_reward: Raw reward from environment (PnL)
        done: Episode termination flag
        new_balance: Balance after action
        is_new_day: Trading day boundary flag
        
    Returns:
        Training step info dictionary
    """
    
    # 1. Compute enriched reward with POEL
    enriched_reward, poel_info = poel.compute_reward(
        state=state,
        action=action,
        raw_pnl=raw_reward,
        new_balance=new_balance,
        is_new_day=is_new_day,
    )
    
    # 2. Store transition with enriched reward
    agent.replay_buffer.add(
        state=state,
        action=action,
        reward=enriched_reward,  # Use POEL reward instead of raw
        next_state=next_state,
        done=done,
    )
    
    # 3. Train agent (if buffer ready)
    train_info = {}
    if len(agent.replay_buffer) >= agent.batch_size:
        train_info = agent.train()
        
    # 4. Combine info
    info = {
        **poel_info,
        **train_info,
        'poel_risk_factor': poel.get_risk_factor(),
    }
    
    return info


def create_meta_purpose_tracker(
    num_agents: int = 4,
    initial_balance: float = 100000.0,
):
    """
    Create Purpose Function for Meta-Agent coordination.
    
    Args:
        num_agents: Number of local agents
        initial_balance: Total portfolio balance
        
    Returns:
        Tuple of (PurposeFunction, BusinessPurpose config)
    """
    
    # Define business purpose (funding requirements)
    purpose_config = BusinessPurpose(
        lambda_daily_dd=10.0,   # Heavy daily DD penalty
        lambda_total_dd=20.0,   # Severe total DD penalty
        max_daily_dd_pct=0.05,  # 5% daily limit
        max_total_dd_pct=0.10,  # 10% total limit
        emergency_threshold_pct=0.80,  # Emergency at 8% DD
    )
    
    # Create purpose tracker
    purpose_fn = PurposeFunction(purpose_config)
    purpose_fn.reset(initial_balance)
    
    return purpose_fn, purpose_config


def meta_agent_step_with_purpose(
    purpose_fn: PurposeFunction,
    current_balance: float,
    daily_dd_pct: float,
    total_dd_pct: float,
    portfolio_pnl: float,
):
    """
    Compute Meta-Agent purpose score and check emergency mode.
    
    Args:
        purpose_fn: PurposeFunction instance
        current_balance: Current total balance
        daily_dd_pct: Daily drawdown percentage
        total_dd_pct: Total drawdown percentage
        portfolio_pnl: Portfolio PnL for step
        
    Returns:
        Purpose metrics dictionary
    """
    
    # Compute purpose
    metrics = purpose_fn.compute_purpose(
        current_balance=current_balance,
        daily_dd_pct=daily_dd_pct,
        total_dd_pct=total_dd_pct,
        pnl=portfolio_pnl,
    )
    
    # Check emergency mode
    if metrics['emergency_mode']:
        print(f"⚠️  EMERGENCY MODE ACTIVATED at {total_dd_pct:.2%} DD")
        print("    Triggering Zero Risk Protocol...")
        
    # Check constraint violations
    if daily_dd_pct > purpose_fn.config.max_daily_dd_pct:
        print(f"🚨 DAILY DD BREACH: {daily_dd_pct:.2%} > {purpose_fn.config.max_daily_dd_pct:.2%}")
        
    if total_dd_pct > purpose_fn.config.max_total_dd_pct:
        print(f"🚨 TOTAL DD BREACH: {total_dd_pct:.2%} > {purpose_fn.config.max_total_dd_pct:.2%}")
        
    return metrics


# Example usage
if __name__ == "__main__":
    print("="*60)
    print("POEL Integration Example")
    print("="*60)
    
    # Simulated agent config
    state_dim = 31  # Financial Intelligence features
    action_dim = 1  # Position size
    
    print("\n1. Creating POEL Reward Shaper...")
    poel = integrate_poel_with_agent(
        agent=None,  # Would be actual TD3/PPO/SAC agent
        state_dim=state_dim,
        action_dim=action_dim,
        initial_balance=25000.0,
    )
    print(f"   ✓ Alpha (PnL weight): {poel.alpha}")
    print(f"   ✓ Beta (Stability weight): {poel.beta}")
    print(f"   ✓ Novelty metric: {poel.novelty_detector.metric}")
    
    print("\n2. Creating Meta-Agent Purpose Function...")
    purpose_fn, purpose_config = create_meta_purpose_tracker()
    print(f"   ✓ Daily DD limit: {purpose_config.max_daily_dd_pct:.1%}")
    print(f"   ✓ Total DD limit: {purpose_config.max_total_dd_pct:.1%}")
    print(f"   ✓ Emergency threshold: {purpose_config.emergency_threshold_pct:.1%}")
    
    print("\n3. Simulating trading step...")
    # Dummy data
    state = np.random.randn(state_dim)
    action = np.array([0.05])  # 5% position
    raw_pnl = 150.0
    new_balance = 25150.0
    
    enriched_reward, info = poel.compute_reward(
        state=state,
        action=action,
        raw_pnl=raw_pnl,
        new_balance=new_balance,
    )
    
    print(f"   Raw PnL: ${raw_pnl:.2f}")
    print(f"   Novelty Bonus: ${info['novelty_bonus']:.2f}")
    print(f"   Stability Penalty: ${info['stability_penalty']:.2f}")
    print(f"   Enriched Reward: ${enriched_reward:.2f}")
    print(f"   Current DD: {info['current_dd']:.2%}")
    
    print("\n4. Meta-Agent Purpose Check...")
    purpose_metrics = meta_agent_step_with_purpose(
        purpose_fn=purpose_fn,
        current_balance=100500.0,
        daily_dd_pct=0.03,  # 3% daily DD
        total_dd_pct=0.05,  # 5% total DD
        portfolio_pnl=500.0,
    )
    
    print(f"   Purpose Score: {purpose_metrics['purpose']:.2f}")
    print(f"   Risk Budget Remaining: {purpose_fn.get_risk_budget(0.05):.1%}")
    print(f"   Emergency Mode: {purpose_metrics['emergency_mode']}")
    
    print("\n" + "="*60)
    print("✅ POEL Integration Complete!")
    print("="*60)
