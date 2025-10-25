"""
UNDERDOG TRADING SYSTEMS
Heterogeneous MARL Migration Validation Script

This script validates that the heterogeneous architecture migration
is complete and functional before running full training.

Tests:
1. Agent configuration system
2. Neural network architectures
3. PPO, SAC, DDPG agent instantiation
4. Agent factory compatibility
5. Training loop integration
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
from config.agent_configs import get_agent_config, get_agent_metadata, AGENT_MAPPING, print_agent_summary
from underdog.rl.models import create_policy_network, create_critic_network
from underdog.rl.multi_agents import PPOAgent, SACAgent, DDPGAgent
from underdog.rl.agents import TD3Agent

def test_configurations():
    """Test 1: Agent configurations"""
    print("\n" + "="*80)
    print("TEST 1: Agent Configuration System")
    print("="*80)
    
    for symbol in ['EURUSD', 'USDJPY', 'XAUUSD', 'GBPUSD']:
        config = get_agent_config(symbol)
        metadata = get_agent_metadata(symbol)
        print(f"\n[{symbol}]")
        print(f"  Algorithm:    {metadata['algorithm']}")
        print(f"  Architecture: {metadata['architecture']}")
        print(f"  Config Type:  {config.__class__.__name__}")
        print(f"  LR Actor:     {config.lr_actor}")
        print(f"  LR Critic:    {config.lr_critic}")
    
    print("\n✓ Configuration system working")


def test_neural_architectures():
    """Test 2: Neural network architectures"""
    print("\n" + "="*80)
    print("TEST 2: Neural Network Architectures")
    print("="*80)
    
    state_dim = 29
    action_dim = 1
    batch_size = 4
    
    architectures = ['LSTM', 'CNN1D', 'Transformer', 'Attention']
    
    for arch in architectures:
        print(f"\n[{arch}]")
        
        # Create policy
        policy = create_policy_network(arch, state_dim, action_dim, hidden_dim=128)
        print(f"  Policy:  {policy.__class__.__name__}")
        
        # Create critic
        critic = create_critic_network(arch, state_dim, action_dim, hidden_dim=128)
        print(f"  Critic:  {critic.__class__.__name__}")
        
        # Test forward pass
        if arch in ['LSTM', 'CNN1D', 'Transformer', 'Attention']:
            # Sequence input
            state = torch.randn(batch_size, 10, state_dim)  # 10 timesteps
        else:
            # Single timestep
            state = torch.randn(batch_size, state_dim)
        
        action = torch.randn(batch_size, action_dim)
        
        with torch.no_grad():
            policy_out = policy(state)
            q1, q2 = critic(state, action)
        
        print(f"  Policy Output Shape: {policy_out.shape}")
        print(f"  Critic Output Shape: {q1.shape}, {q2.shape}")
        assert policy_out.shape == (batch_size, action_dim), f"Policy shape mismatch for {arch}"
        assert q1.shape == (batch_size, 1), f"Critic shape mismatch for {arch}"
    
    print("\n✓ All architectures working")


def test_agent_instantiation():
    """Test 3: Agent instantiation"""
    print("\n" + "="*80)
    print("TEST 3: Agent Instantiation")
    print("="*80)
    
    agents = {}
    
    for symbol in ['EURUSD', 'USDJPY', 'XAUUSD', 'GBPUSD']:
        metadata = get_agent_metadata(symbol)
        config = metadata['config']
        algorithm = metadata['algorithm']
        
        # Override state/action dims
        config.state_dim = 29
        config.action_dim = 1
        
        print(f"\n[{symbol}] Creating {algorithm} agent...")
        
        if algorithm == 'TD3':
            agent = TD3Agent(config=config)
        elif algorithm == 'PPO':
            agent = PPOAgent(config=config)
        elif algorithm == 'SAC':
            agent = SACAgent(config=config)
        elif algorithm == 'DDPG':
            agent = DDPGAgent(config=config)
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")
        
        agents[symbol] = agent
        print(f"  ✓ {agent.__class__.__name__} created")
        
        # Test action selection
        state = np.random.randn(29)
        action = agent.select_action(state, explore=True)
        print(f"  ✓ Action shape: {action.shape}")
        assert action.shape == (1,), f"Action shape mismatch for {symbol}"
    
    print("\n✓ All agents instantiated successfully")
    return agents


def test_agent_compatibility(agents):
    """Test 4: Agent-environment compatibility"""
    print("\n" + "="*80)
    print("TEST 4: Agent-Environment Compatibility")
    print("="*80)
    
    # Simulate transitions
    for symbol, agent in agents.items():
        print(f"\n[{symbol}] Testing {agent.__class__.__name__}...")
        
        state = np.random.randn(29)
        action = agent.select_action(state, explore=True)
        next_state = np.random.randn(29)
        reward = np.random.randn()
        done = False
        
        agent_class = agent.__class__.__name__
        
        if agent_class == 'PPOAgent':
            # PPO stores transitions internally
            agent.store_transition(state, action, reward, done)
            print(f"  ✓ Transition stored in on-policy buffer")
            print(f"  Buffer size: {len(agent.buffer['states'])}")
        else:
            # Off-policy agents need external replay buffer
            from underdog.rl.agents import ReplayBuffer
            buffer = ReplayBuffer(state_dim=29, action_dim=1, max_size=1000)
            buffer.add(state, action, reward, next_state, done)
            print(f"  ✓ Transition stored in replay buffer")
            print(f"  Buffer size: {len(buffer)}")
    
    print("\n✓ All agents compatible with storage")


def test_training_compatibility(agents):
    """Test 5: Training compatibility"""
    print("\n" + "="*80)
    print("TEST 5: Training Compatibility")
    print("="*80)
    
    from underdog.rl.agents import ReplayBuffer
    
    for symbol, agent in agents.items():
        print(f"\n[{symbol}] Testing {agent.__class__.__name__} training...")
        
        agent_class = agent.__class__.__name__
        
        if agent_class == 'PPOAgent':
            # Fill PPO buffer
            for _ in range(128):  # PPO needs at least batch_size transitions
                state = np.random.randn(29)
                action = agent.select_action(state, explore=True)
                reward = np.random.randn()
                done = np.random.rand() > 0.95
                agent.store_transition(state, action, reward, done)
            
            # Train
            metrics = agent.train()
            print(f"  ✓ PPO training successful")
            print(f"  Metrics: {metrics}")
        else:
            # Off-policy training
            buffer = ReplayBuffer(state_dim=29, action_dim=1, max_size=10000)
            
            # Fill buffer
            for _ in range(1000):
                state = np.random.randn(29)
                action = agent.select_action(state, explore=True)
                reward = np.random.randn()
                next_state = np.random.randn(29)
                done = np.random.rand() > 0.95
                buffer.add(state, action, reward, next_state, done)
            
            # Train
            metrics = agent.train(buffer)
            print(f"  ✓ {agent_class} training successful")
            print(f"  Metrics: {metrics}")
    
    print("\n✓ All agents can train successfully")


def main():
    """Run all validation tests"""
    print("\n" + "="*80)
    print("UNDERDOG TRADING SYSTEMS")
    print("Heterogeneous MARL Migration Validation")
    print("="*80)
    
    try:
        # Test 1: Configurations
        test_configurations()
        
        # Test 2: Neural architectures
        test_neural_architectures()
        
        # Test 3: Agent instantiation
        agents = test_agent_instantiation()
        
        # Test 4: Compatibility
        test_agent_compatibility(agents)
        
        # Test 5: Training
        test_training_compatibility(agents)
        
        # Summary
        print("\n" + "="*80)
        print("✓ ALL TESTS PASSED")
        print("="*80)
        print("\nHeterogeneous MARL architecture is ready for training!")
        print("\nNext steps:")
        print("  1. Run training: poetry run python scripts/train_marl_agent.py --episodes 10")
        print("  2. Monitor dashboard: poetry run python scripts/visualize_training_v2.py")
        print("  3. Verify agent types in CSV: agent0_type, agent1_type, etc.")
        
    except Exception as e:
        print("\n" + "="*80)
        print("❌ TEST FAILED")
        print("="*80)
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
