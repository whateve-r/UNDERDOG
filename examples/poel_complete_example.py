"""
Complete POEL Integration Example.

Demonstrates full integration of POEL Meta-Agent with HARL system:
- Module 1: Purpose-Driven Risk Control (local agents)
- Module 2.1: Dynamic Capital Allocation
- Module 2.2: Failure Bank & Curriculum
- NRF: Neural Reward Function skill discovery

This example shows the complete workflow from episode initialization to
capital allocation, failure recording, and skill checkpointing.
"""

import numpy as np
from typing import Dict, List

from underdog.rl.poel import (
    POELMetaAgent,
    POELRewardShaper,
    DistanceMetric,
    TrainingMode,
)


def create_complete_poel_system(
    initial_balance: float = 100000.0,
    symbols: List[str] = None,
    state_dim: int = 31,
) -> Dict:
    """
    Create complete POEL system with Meta-Agent and local reward shapers.
    
    Args:
        initial_balance: Starting capital
        symbols: Trading symbols
        state_dim: Dimension of state space
    
    Returns:
        Dict with meta_agent and local_shapers
    """
    if symbols is None:
        symbols = ['EURUSD', 'USDJPY', 'XAUUSD', 'GBPUSD']
    
    # Create Meta-Agent
    meta_agent = POELMetaAgent(
        initial_balance=initial_balance,
        symbols=symbols,
        max_daily_dd=0.05,
        max_total_dd=0.10,
        nrf_enabled=True,
        nrf_cycle_frequency=10,
        state_dim=state_dim,
    )
    
    # Create local POEL shapers for each agent
    local_shapers = {}
    for symbol in symbols:
        shaper = POELRewardShaper(
            state_dim=state_dim,
            action_dim=1,
            alpha=0.7,  # 70% PnL, 30% exploration
            beta=1.0,
            novelty_metric=DistanceMetric.L2,
            max_local_dd_pct=0.15,
            initial_balance=initial_balance / len(symbols),
        )
        local_shapers[symbol] = shaper
    
    return {
        'meta_agent': meta_agent,
        'local_shapers': local_shapers,
        'symbols': symbols,
    }


def episode_workflow_example():
    """
    Complete episode workflow with POEL Meta-Agent.
    
    Shows:
    1. Episode initialization
    2. Training step with enriched rewards
    3. Meta-Agent purpose evaluation
    4. Capital allocation
    5. Failure recording
    6. Skill checkpointing
    """
    print("=" * 80)
    print("POEL COMPLETE INTEGRATION EXAMPLE")
    print("=" * 80)
    
    # 1. Create POEL system
    print("\n1. Creating POEL System...")
    poel_system = create_complete_poel_system(
        initial_balance=100000.0,
        symbols=['EURUSD', 'USDJPY', 'XAUUSD', 'GBPUSD'],
        state_dim=31,
    )
    
    meta_agent = poel_system['meta_agent']
    local_shapers = poel_system['local_shapers']
    symbols = poel_system['symbols']
    
    print(f"   ✓ Meta-Agent created")
    print(f"   ✓ Local shapers: {list(local_shapers.keys())}")
    
    # 2. Start Episode
    print("\n2. Starting Episode...")
    episode_config = meta_agent.start_episode()
    
    print(f"   Episode: {episode_config['episode']}")
    print(f"   Mode: {episode_config['mode']}")
    print(f"   Inject Failure: {episode_config['inject_failure']}")
    
    # 3. Simulate Training Step
    print("\n3. Simulating Training Step...")
    
    # Simulate state and action for each agent
    states = {
        symbol: np.random.randn(31) for symbol in symbols
    }
    actions = {
        symbol: np.random.randn(1) for symbol in symbols
    }
    
    # Simulate environment rewards (PnL)
    raw_pnls = {
        'EURUSD': 120.0,
        'USDJPY': -50.0,
        'XAUUSD': 300.0,
        'GBPUSD': 80.0,
    }
    
    # Simulate balances
    balances = {
        'EURUSD': 25120.0,
        'USDJPY': 24950.0,
        'XAUUSD': 25300.0,
        'GBPUSD': 25080.0,
    }
    
    # Compute enriched rewards for each agent
    enriched_rewards = {}
    for symbol in symbols:
        reward, info = local_shapers[symbol].compute_reward(
            state=states[symbol],
            action=actions[symbol],
            raw_pnl=raw_pnls[symbol],
            new_balance=balances[symbol],
        )
        enriched_rewards[symbol] = (reward, info)
        
        # Update meta-agent with agent performance
        meta_agent.update_agent_performance(
            agent_id=f"agent_{symbol}",
            symbol=symbol,
            pnl=raw_pnls[symbol],
            balance=balances[symbol],
        )
    
    # Print enriched rewards
    print("\n   Enriched Rewards:")
    for symbol, (reward, info) in enriched_rewards.items():
        print(f"   {symbol:8s}: Raw=${info['raw_pnl']:7.2f} → Enriched=${reward:7.2f} "
              f"(Novelty=${info['novelty_bonus']:6.2f}, Stability=${info['stability_penalty']:6.2f})")
    
    # 4. Meta-Agent Purpose Evaluation & Capital Allocation
    print("\n4. Meta-Agent: Purpose & Capital Allocation...")
    
    total_balance = sum(balances.values())
    portfolio_pnl = sum(raw_pnls.values())
    
    # Calculate global DD (simplified)
    initial_balance = 100000.0
    total_dd_pct = max(0.0, (initial_balance - total_balance) / initial_balance)
    daily_dd_pct = 0.02  # Simulated 2% daily DD
    
    meta_result = meta_agent.compute_purpose_and_allocate(
        current_balance=total_balance,
        daily_dd_pct=daily_dd_pct,
        total_dd_pct=total_dd_pct,
        portfolio_pnl=portfolio_pnl,
    )
    
    print(f"\n   Purpose Score: {meta_result['purpose']['purpose']:.2f}")
    print(f"   Global DD: {total_dd_pct:.2%} (Daily: {daily_dd_pct:.2%})")
    print(f"   Emergency Mode: {meta_result['emergency']['emergency_mode']}")
    
    print("\n   Capital Allocation:")
    for symbol, weight in meta_result['weights'].items():
        perf = meta_result['agent_performances'].get(symbol, {})
        calmar = perf.get('calmar_ratio', 0.0)
        print(f"   {symbol:8s}: {weight:6.2%} (Calmar Ratio: {calmar:6.2f})")
    
    # 5. NRF Step (if in NRF mode)
    print("\n5. NRF Step...")
    nrf_metrics = meta_agent.nrf_step(
        state=states['EURUSD'],
        training_step=100,
    )
    
    if nrf_metrics:
        print(f"   NRF Update: Loss={nrf_metrics['loss']:.4f}")
    else:
        print(f"   NRF Mode: Not active this episode")
    
    # 6. Failure Recording (simulate DD breach)
    print("\n6. Failure Recording...")
    
    # Simulate a failure on XAUUSD
    if total_dd_pct > 0.08:  # 8% DD breach
        print(f"   ⚠ DD Breach detected: {total_dd_pct:.2%} > 8%")
        
        meta_agent.record_failure(
            state=states['XAUUSD'],
            dd_breach_size=total_dd_pct - 0.08,
            agent_weights={'XAUUSD': np.random.randn(100)},  # Simulated weights
            symbol='XAUUSD',
            episode_id=episode_config['episode'],
            step=50,
            balance=balances['XAUUSD'],
            pnl=raw_pnls['XAUUSD'],
            failure_type='total_dd',
        )
        
        print(f"   ✓ Failure recorded to bank")
    else:
        print(f"   No DD breach (current: {total_dd_pct:.2%})")
    
    # 7. Skill Checkpointing (if high Calmar Ratio)
    print("\n7. Skill Checkpointing...")
    
    best_symbol = max(
        meta_result['agent_performances'].items(),
        key=lambda x: x[1]['calmar_ratio']
    )[0]
    
    best_calmar = meta_result['agent_performances'][best_symbol]['calmar_ratio']
    
    if best_calmar > 2.0:  # High Calmar threshold
        print(f"   ✓ High Calmar detected: {best_symbol} = {best_calmar:.2f}")
        
        # Get novelty score from local shaper
        novelty_info = enriched_rewards[best_symbol][1]
        novelty_score = novelty_info.get('novelty_score', 0.0)
        
        meta_agent.checkpoint_skill(
            agent_id=f"agent_{best_symbol}",
            symbol=best_symbol,
            calmar_ratio=best_calmar,
            novelty_score=novelty_score,
            model_weights={'weights': np.random.randn(100)},  # Simulated
            episode_id=episode_config['episode'],
            steps_trained=5000,
            skill_name=f"{best_symbol} High Calmar Strategy",
        )
        
        print(f"   ✓ Skill checkpointed: {best_symbol} (Calmar={best_calmar:.2f}, Novelty={novelty_score:.3f})")
    else:
        print(f"   Best Calmar: {best_symbol} = {best_calmar:.2f} (below threshold)")
    
    # 8. End Episode
    print("\n8. Ending Episode...")
    episode_summary = meta_agent.end_episode()
    
    print(f"   Episode {episode_summary['episode']} complete")
    print(f"   Failure Bank Size: {episode_summary['failure_bank_size']}")
    print(f"   Skill Bank Size: {episode_summary['skill_bank_size']}")
    
    # 9. POEL Statistics
    print("\n9. POEL System Statistics...")
    stats = meta_agent.get_statistics()
    
    print(f"\n   Training Mode: {stats['current_mode']}")
    print(f"   Episodes: {stats['episode_count']}")
    print(f"   NRF Cycles: {stats['nrf_cycle_count']}")
    
    print(f"\n   Failure Bank:")
    fb = stats['failure_bank']
    print(f"     Total Failures: {fb['total_failures']}")
    print(f"     By Type: {fb['by_type']}")
    
    print(f"\n   Skill Bank:")
    sb = stats['skill_bank']
    print(f"     Total Skills: {sb['total_skills']}")
    print(f"     Avg Calmar: {sb.get('avg_calmar', 0.0):.2f}")
    print(f"     Avg Novelty: {sb.get('avg_novelty', 0.0):.3f}")
    
    print(f"\n   Curriculum:")
    curr = stats['curriculum']
    print(f"     Total Episodes: {curr['total_episodes']}")
    print(f"     Injected: {curr['injected_episodes']}")
    print(f"     Injection Rate: {curr['actual_injection_rate']:.1%}")
    
    print("\n" + "=" * 80)
    print("INTEGRATION COMPLETE ✓")
    print("=" * 80)


def multi_episode_training_example():
    """
    Simulate multiple episodes to show mode transitions.
    
    Demonstrates:
    - Normal training mode
    - NRF skill discovery mode
    - Curriculum injection
    - Emergency protocol
    """
    print("\n" + "=" * 80)
    print("MULTI-EPISODE TRAINING SIMULATION")
    print("=" * 80)
    
    poel_system = create_complete_poel_system()
    meta_agent = poel_system['meta_agent']
    local_shapers = poel_system['local_shapers']
    symbols = poel_system['symbols']
    
    # Simulate 30 episodes
    for episode_num in range(1, 31):
        # Start episode
        config = meta_agent.start_episode()
        
        # Simulate some steps
        total_balance = 100000.0 + np.random.randn() * 5000
        total_dd = np.random.uniform(0.0, 0.12)
        
        # Update performances
        for symbol in symbols:
            meta_agent.update_agent_performance(
                agent_id=f"agent_{symbol}",
                symbol=symbol,
                pnl=np.random.randn() * 200,
                balance=total_balance / len(symbols),
            )
        
        # Compute purpose
        result = meta_agent.compute_purpose_and_allocate(
            current_balance=total_balance,
            daily_dd_pct=total_dd / 2,
            total_dd_pct=total_dd,
            portfolio_pnl=np.random.randn() * 500,
        )
        
        # End episode
        summary = meta_agent.end_episode()
        
        # Print summary
        mode_symbol = {
            'normal': '🔄',
            'nrf': '🧠',
            'curriculum': '📚',
            'emergency': '🚨',
        }
        
        symbol_mode = mode_symbol.get(config['mode'], '❓')
        
        print(f"Episode {episode_num:2d} {symbol_mode} "
              f"Mode: {config['mode']:12s} | "
              f"DD: {total_dd:5.1%} | "
              f"Purpose: {result['purpose']['purpose']:7.1f} | "
              f"Emergency: {result['emergency']['emergency_mode']}")
    
    # Final statistics
    final_stats = meta_agent.get_statistics()
    
    print("\n" + "=" * 80)
    print("FINAL STATISTICS")
    print("=" * 80)
    print(f"\nTotal Episodes: {final_stats['episode_count']}")
    print(f"NRF Cycles: {final_stats['nrf_cycle_count']}")
    print(f"Failures Recorded: {final_stats['failure_bank']['total_failures']}")
    print(f"Skills Discovered: {final_stats['skill_bank']['total_skills']}")
    print(f"Curriculum Injections: {final_stats['curriculum']['injected_episodes']}")


if __name__ == "__main__":
    # Run complete workflow example
    episode_workflow_example()
    
    # Run multi-episode simulation
    multi_episode_training_example()
