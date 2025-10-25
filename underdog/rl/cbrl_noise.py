"""
CBRL: Chaos-Based Reinforcement Learning - Chaotic Noise Generator

Implementation of the Logistic Map for intelligent exploration in RL agents.
Replaces Gaussian noise with deterministic chaotic sequences for better
exploration-exploitation balance.

References:
    - "Chaos-based reinforcement learning" papers
    - Logistic Map: x_{t+1} = r * x_t * (1 - x_t), where r = 4 for full chaos

Mathematical Properties:
    - Deterministic but aperiodic (appears random)
    - Sensitive to initial conditions (butterfly effect)
    - Bounded to [0, 1] interval
    - Ergodic (explores all states uniformly over time)

Usage:
    >>> noise_gen = ChaoticNoiseGenerator(x0=0.314159)
    >>> noise = noise_gen.get_noise(shape=(1,))
    >>> # Returns chaotic value in [-1, 1]
"""

import numpy as np
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class ChaoticNoiseGenerator:
    """
    🌪️ Chaotic Noise Generator using Logistic Map
    
    Generates deterministic chaotic sequences for exploration in RL.
    Superior to Gaussian noise due to:
        1. Deterministic reproducibility (given x0)
        2. Uniform coverage of action space (ergodic)
        3. Autocorrelation structure aids learning
        4. No statistical assumptions needed
    """
    
    def __init__(
        self,
        x0: Optional[float] = None,
        r: float = 4.0,
        output_range: Tuple[float, float] = (-1.0, 1.0),
        seed: Optional[int] = None
    ):
        """
        Initialize Chaotic Noise Generator
        
        Args:
            x0: Initial condition (0, 1). If None, random initialization.
            r: Logistic map parameter (4.0 for full chaos)
            output_range: Desired output range (default: [-1, 1] for actions)
            seed: Random seed for reproducibility (only affects x0 if None)
        """
        self.r = r
        self.output_range = output_range
        
        # Initialize state
        if seed is not None:
            np.random.seed(seed)
        
        if x0 is None:
            # Random initialization in (0, 1), avoiding exact 0 or 1
            self.x = np.random.uniform(0.01, 0.99)
        else:
            if not (0 < x0 < 1):
                raise ValueError(f"x0 must be in (0, 1), got {x0}")
            self.x = x0
        
        self.initial_x = self.x
        self.step_count = 0
        
        logger.info(f"ChaoticNoiseGenerator initialized: x0={self.x:.6f}, r={self.r}")
    
    def get_noise(self, shape: Tuple[int, ...] = (1,)) -> np.ndarray:
        """
        Generate chaotic noise samples
        
        Args:
            shape: Shape of output array (e.g., (1,) for single action)
        
        Returns:
            noise: Chaotic noise array in output_range
        """
        noise_values = []
        total_samples = np.prod(shape)
        
        for _ in range(total_samples):
            # Logistic map iteration: x_{t+1} = r * x_t * (1 - x_t)
            self.x = self.r * self.x * (1 - self.x)
            self.step_count += 1
            
            # Map from [0, 1] to output_range
            noise_scaled = self._scale_to_range(self.x)
            noise_values.append(noise_scaled)
        
        # Reshape to desired shape
        noise = np.array(noise_values, dtype=np.float32).reshape(shape)
        
        return noise
    
    def _scale_to_range(self, x: float) -> float:
        """
        Scale chaotic value from [0, 1] to output_range
        
        Args:
            x: Chaotic value in [0, 1]
        
        Returns:
            Scaled value in output_range
        """
        low, high = self.output_range
        return low + (high - low) * x
    
    def reset(self, x0: Optional[float] = None):
        """
        Reset generator to initial or new state
        
        Args:
            x0: New initial condition. If None, use original initial_x
        """
        if x0 is None:
            self.x = self.initial_x
        else:
            if not (0 < x0 < 1):
                raise ValueError(f"x0 must be in (0, 1), got {x0}")
            self.x = x0
            self.initial_x = x0
        
        self.step_count = 0
        logger.debug(f"ChaoticNoiseGenerator reset to x={self.x:.6f}")
    
    def get_state(self) -> dict:
        """Get current generator state for checkpointing"""
        return {
            'x': self.x,
            'initial_x': self.initial_x,
            'step_count': self.step_count,
            'r': self.r,
            'output_range': self.output_range
        }
    
    def set_state(self, state: dict):
        """Restore generator state from checkpoint"""
        self.x = state['x']
        self.initial_x = state['initial_x']
        self.step_count = state['step_count']
        self.r = state['r']
        self.output_range = tuple(state['output_range'])
        logger.debug(f"ChaoticNoiseGenerator state restored: step={self.step_count}")


# ============================================================================
# Testing & Validation
# ============================================================================

if __name__ == "__main__":
    print("="*60)
    print("CHAOTIC NOISE GENERATOR TEST")
    print("="*60)
    
    # Test 1: Basic functionality
    print("\n[1] Testing basic noise generation...")
    gen = ChaoticNoiseGenerator(x0=0.5, seed=42)
    noise_samples = gen.get_noise(shape=(10,))
    print(f"   Generated 10 samples: {noise_samples}")
    print(f"   Mean: {np.mean(noise_samples):.4f}")
    print(f"   Std: {np.std(noise_samples):.4f}")
    print(f"   Range: [{np.min(noise_samples):.4f}, {np.max(noise_samples):.4f}]")
    
    # Test 2: Deterministic reproducibility
    print("\n[2] Testing deterministic reproducibility...")
    gen1 = ChaoticNoiseGenerator(x0=0.314159)
    gen2 = ChaoticNoiseGenerator(x0=0.314159)
    
    seq1 = gen1.get_noise(shape=(5,))
    seq2 = gen2.get_noise(shape=(5,))
    
    print(f"   Sequence 1: {seq1}")
    print(f"   Sequence 2: {seq2}")
    print(f"   Identical: {np.allclose(seq1, seq2)}")
    
    # Test 3: Divergence from small perturbations
    print("\n[3] Testing sensitivity to initial conditions...")
    gen_a = ChaoticNoiseGenerator(x0=0.5000)
    gen_b = ChaoticNoiseGenerator(x0=0.5001)  # 0.01% difference
    
    seq_a = gen_a.get_noise(shape=(20,))
    seq_b = gen_b.get_noise(shape=(20,))
    
    divergence = np.abs(seq_a - seq_b)
    print(f"   Max divergence: {np.max(divergence):.4f}")
    print(f"   Mean divergence: {np.mean(divergence):.4f}")
    print(f"   (Should increase rapidly due to butterfly effect)")
    
    # Test 4: Compare with Gaussian noise
    print("\n[4] Comparing with Gaussian noise...")
    chaotic_samples = gen.get_noise(shape=(1000,))
    gaussian_samples = np.random.randn(1000)
    
    print(f"   Chaotic - Mean: {np.mean(chaotic_samples):.4f}, Std: {np.std(chaotic_samples):.4f}")
    print(f"   Gaussian - Mean: {np.mean(gaussian_samples):.4f}, Std: {np.std(gaussian_samples):.4f}")
    
    # Test 5: State save/load
    print("\n[5] Testing state save/load...")
    gen.reset(x0=0.7)
    state = gen.get_state()
    noise_before = gen.get_noise(shape=(3,))
    
    gen.reset(x0=0.2)  # Change state
    gen.set_state(state)  # Restore
    noise_after = gen.get_noise(shape=(3,))
    
    print(f"   Before: {noise_before}")
    print(f"   After:  {noise_after}")
    print(f"   Restored correctly: {np.allclose(noise_before, noise_after)}")
    
    print("\n" + "="*60)
    print("✅ All tests completed!")
    print("="*60)
