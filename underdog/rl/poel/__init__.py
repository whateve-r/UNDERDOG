"""
Purpose-Driven Open-Ended Learning (POEL) Module

This module implements Open-Ended Reinforcement Learning with business purpose alignment
for autonomous discovery of trading strategies under strict risk constraints.

Key Components:
- Module 1: Purpose-Driven Risk Control (local agents)
- Module 2.1: Dynamic Capital Allocation (meta-level coordination)
- Module 2.2: Failure Bank & Curriculum Learning
- Neural Reward Functions: Autonomous skill discovery
"""

# Module 1: Purpose-Driven Risk Control
from .purpose import PurposeFunction, BusinessPurpose
from .novelty import NoveltyDetector, DistanceMetric
from .stability import StabilityPenalty, LocalDrawdownTracker
from .reward_shaper import POELRewardShaper

# Module 2.1: Dynamic Capital Allocation
from .capital_allocator import (
    CapitalAllocator,
    CalmarCalculator,
    AgentPerformance,
    create_agent_performance,
)

# Module 2.2: Failure Bank & Curriculum
from .failure_bank import (
    FailureBank,
    SkillRepository,
    CurriculumManager,
    FailureRecord,
    SkillCheckpoint,
)

# Neural Reward Functions
from .neural_reward import (
    NeuralRewardFunction,
    NRFSkillDiscovery,
    NRFConfig,
    RewardNetwork,
    create_nrf_system,
)

# Meta-Agent Coordinator
from .meta_agent import (
    POELMetaAgent,
    POELMetaState,
    TrainingMode,
)

__all__ = [
    # Module 1
    'PurposeFunction',
    'BusinessPurpose',
    'NoveltyDetector',
    'DistanceMetric',
    'StabilityPenalty',
    'LocalDrawdownTracker',
    'POELRewardShaper',
    # Module 2.1
    'CapitalAllocator',
    'CalmarCalculator',
    'AgentPerformance',
    'create_agent_performance',
    # Module 2.2
    'FailureBank',
    'SkillRepository',
    'CurriculumManager',
    'FailureRecord',
    'SkillCheckpoint',
    # NRF
    'NeuralRewardFunction',
    'NRFSkillDiscovery',
    'NRFConfig',
    'RewardNetwork',
    'create_nrf_system',
    # Meta-Agent
    'POELMetaAgent',
    'POELMetaState',
    'TrainingMode',
]
