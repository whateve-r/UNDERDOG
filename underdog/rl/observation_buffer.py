"""
Temporal Observation Buffers for Heterogeneous MARL

Implements ring buffers for sequential observations required by
temporal neural architectures (LSTM, CNN1D, Transformer).

Architecture Requirements:
- EURUSD (TD3+LSTM): [60 × Features] sequences
- USDJPY (PPO+CNN1D): [15 × Features] sequences  
- XAUUSD (SAC+Transformer): [120 × Features] sequences
- GBPUSD (DDPG+Attention): [Features] single-step (no buffer)

Critical: Without these buffers, training would feed wrong shapes
to neural networks, corrupting initialization.
"""

from collections import deque
from typing import Optional, Union
import numpy as np


class ObservationBuffer:
    """
    Ring buffer for temporal observations.
    
    Maintains a sliding window of observations with automatic padding
    during warm-up phase (first N steps of episode).
    
    Args:
        symbol: Asset symbol (determines sequence length)
        feature_dim: Observation vector dimension (default 24)
        max_sequence_length: Maximum buffer size (default 120 for Transformer)
    
    Example:
        >>> buffer = ObservationBuffer('EURUSD', feature_dim=24)
        >>> obs = env.reset()[0]  # [24,]
        >>> buffer.reset(obs)
        >>> sequence = buffer.get_sequence()  # [60, 24] (zero-padded initially)
    """
    
    # Sequence length mapping per asset (HMARL specification)
    SEQUENCE_LENGTHS = {
        'EURUSD': 60,   # TD3 + LSTM (trend-following)
        'USDJPY': 15,   # PPO + CNN1D (breakout detection)
        'XAUUSD': 120,  # SAC + Transformer (regime adaptation)
        'GBPUSD': 1,    # DDPG + Attention (no temporal context)
    }
    
    def __init__(
        self,
        symbol: str,
        feature_dim: int = 24,
        max_sequence_length: int = 120
    ):
        """Initialize buffer with symbol-specific sequence length."""
        if symbol not in self.SEQUENCE_LENGTHS:
            raise ValueError(
                f"Unknown symbol: {symbol}. "
                f"Expected one of {list(self.SEQUENCE_LENGTHS.keys())}"
            )
        
        self.symbol = symbol
        self.feature_dim = feature_dim
        self.sequence_length = self.SEQUENCE_LENGTHS[symbol]
        self.max_sequence_length = max_sequence_length
        
        # Use deque for efficient FIFO operations
        self.buffer: deque = deque(maxlen=max_sequence_length)
        
        # Track if buffer has been initialized
        self._initialized = False
    
    def reset(self, initial_observation: np.ndarray) -> None:
        """
        Reset buffer with initial observation.
        
        Clears buffer and adds initial observation. During warm-up,
        get_sequence() will zero-pad to reach sequence_length.
        
        Args:
            initial_observation: Initial state from env.reset() [Features,]
        """
        self.buffer.clear()
        self.add(initial_observation)
        self._initialized = True
    
    def add(self, observation: np.ndarray) -> None:
        """
        Add new observation to buffer.
        
        Automatically maintains sliding window via deque.maxlen.
        
        Args:
            observation: State vector from env.step() [Features,]
        """
        # Validate shape
        if observation.shape != (self.feature_dim,):
            raise ValueError(
                f"Expected observation shape ({self.feature_dim},), "
                f"got {observation.shape}"
            )
        
        self.buffer.append(observation.copy())
    
    def get_sequence(self) -> np.ndarray:
        """
        Get observation sequence for current timestep.
        
        Behavior:
        - GBPUSD (seq_len=1): Returns latest observation [Features,]
        - Others: Returns [T, Features] sequence
          - If buffer < seq_len: Zero-pads at start
          - If buffer >= seq_len: Returns last seq_len observations
        
        Returns:
            - For GBPUSD: [Features,] shape
            - For others: [T, Features] shape where T = sequence_length
        
        Raises:
            RuntimeError: If called before reset()
        """
        if not self._initialized:
            raise RuntimeError(
                "Buffer not initialized. Call reset() with initial observation first."
            )
        
        if len(self.buffer) == 0:
            raise RuntimeError("Buffer is empty. This should not happen after reset().")
        
        # Special case: GBPUSD uses single-step (no temporal context)
        if self.sequence_length == 1:
            return self.buffer[-1]  # [Features,]
        
        # Build sequence with zero-padding during warm-up
        current_length = len(self.buffer)
        
        if current_length < self.sequence_length:
            # Warm-up phase: zero-pad at start
            padding_needed = self.sequence_length - current_length
            zero_obs = np.zeros((self.feature_dim,), dtype=np.float32)
            
            padded_sequence = [zero_obs] * padding_needed + list(self.buffer)
            return np.array(padded_sequence, dtype=np.float32)  # [T, F]
        else:
            # Normal operation: sliding window
            sequence = list(self.buffer)[-self.sequence_length:]
            return np.array(sequence, dtype=np.float32)  # [T, F]
    
    def __len__(self) -> int:
        """Return current buffer size."""
        return len(self.buffer)
    
    def __repr__(self) -> str:
        return (
            f"ObservationBuffer(symbol={self.symbol}, "
            f"seq_len={self.sequence_length}, "
            f"current_size={len(self.buffer)}/{self.max_sequence_length})"
        )


class MultiAssetObservationManager:
    """
    Manages observation buffers for all assets in multi-asset environment.
    
    Creates and maintains separate buffers for each asset, handling
    heterogeneous sequence lengths.
    
    Args:
        symbols: List of asset symbols
        feature_dim: Observation dimension (default 24)
    
    Example:
        >>> manager = MultiAssetObservationManager(['EURUSD', 'USDJPY', 'XAUUSD', 'GBPUSD'])
        >>> observations = env.reset()[0]  # List of 4 observations
        >>> manager.reset(observations)
        >>> sequences = manager.get_sequences()  # List of sequences (varying shapes)
    """
    
    def __init__(self, symbols: list[str], feature_dim: int = 24):
        """Initialize buffers for all assets."""
        self.symbols = symbols
        self.feature_dim = feature_dim
        
        # Create buffer for each asset
        self.buffers = {
            symbol: ObservationBuffer(symbol, feature_dim)
            for symbol in symbols
        }
    
    def reset(self, initial_observations: list[np.ndarray]) -> None:
        """
        Reset all buffers with initial observations.
        
        Args:
            initial_observations: List of initial states from env.reset()
                Length must match number of symbols.
        """
        if len(initial_observations) != len(self.symbols):
            raise ValueError(
                f"Expected {len(self.symbols)} observations, "
                f"got {len(initial_observations)}"
            )
        
        for symbol, obs in zip(self.symbols, initial_observations):
            self.buffers[symbol].reset(obs)
    
    def add(self, observations: list[np.ndarray]) -> None:
        """
        Add new observations to all buffers.
        
        Args:
            observations: List of states from env.step()
        """
        if len(observations) != len(self.symbols):
            raise ValueError(
                f"Expected {len(self.symbols)} observations, "
                f"got {len(observations)}"
            )
        
        for symbol, obs in zip(self.symbols, observations):
            self.buffers[symbol].add(obs)
    
    def get_sequences(self) -> list[np.ndarray]:
        """
        Get observation sequences for all assets.
        
        Returns:
            List of sequences:
            - EURUSD: [60, Features]
            - USDJPY: [15, Features]
            - XAUUSD: [120, Features]
            - GBPUSD: [Features]  # No temporal dimension
        """
        return [
            self.buffers[symbol].get_sequence()
            for symbol in self.symbols
        ]
    
    def get_sequence_shapes(self) -> dict[str, tuple]:
        """
        Get expected sequence shape for each asset (debugging).
        
        Returns:
            Dict mapping symbol to expected shape tuple.
        """
        shapes = {}
        for symbol in self.symbols:
            seq_len = ObservationBuffer.SEQUENCE_LENGTHS[symbol]
            if seq_len == 1:
                shapes[symbol] = (self.feature_dim,)
            else:
                shapes[symbol] = (seq_len, self.feature_dim)
        return shapes
    
    def __repr__(self) -> str:
        buffer_info = "\n  ".join([str(buf) for buf in self.buffers.values()])
        return f"MultiAssetObservationManager(\n  {buffer_info}\n)"


# Validation function for testing
def validate_buffer_shapes(symbols: list[str], feature_dim: int = 24) -> dict:
    """
    Validate buffer shapes match neural architecture requirements.
    
    Args:
        symbols: Asset symbols
        feature_dim: Observation dimension
    
    Returns:
        Dict with validation results per symbol
    """
    manager = MultiAssetObservationManager(symbols, feature_dim)
    
    # Simulate reset with dummy observations
    dummy_obs = [np.random.randn(feature_dim) for _ in symbols]
    manager.reset(dummy_obs)
    
    # Add more observations to fill buffers
    for _ in range(150):  # More than max sequence length
        dummy_obs = [np.random.randn(feature_dim) for _ in symbols]
        manager.add(dummy_obs)
    
    # Get sequences and validate
    sequences = manager.get_sequences()
    expected_shapes = manager.get_sequence_shapes()
    
    results = {}
    for symbol, sequence, expected_shape in zip(symbols, sequences, expected_shapes.values()):
        actual_shape = sequence.shape
        matches = actual_shape == expected_shape
        
        results[symbol] = {
            'expected_shape': expected_shape,
            'actual_shape': actual_shape,
            'matches': matches,
            'sequence_length': ObservationBuffer.SEQUENCE_LENGTHS[symbol],
        }
    
    return results
