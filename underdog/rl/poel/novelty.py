"""
Novelty Detection System

Implements exploration bonuses based on state-action novelty.
Uses distance metrics (L2, Mahalanobis) to identify unexplored regions.
"""

from typing import Tuple, Optional, Literal
from enum import Enum
import numpy as np
from collections import deque
import torch


class DistanceMetric(str, Enum):
    """Distance metrics for novelty calculation"""
    L2 = "l2"  # Euclidean distance
    MAHALANOBIS = "mahalanobis"  # Accounts for feature correlation
    COSINE = "cosine"  # Angle-based distance


class NoveltyDetector:
    """
    Detects novel state-action pairs using distance-based metrics.
    
    Maintains a buffer of recent (state, action) pairs and calculates
    novelty as distance to nearest neighbor in feature space.
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        buffer_size: int = 10000,
        metric: DistanceMetric = DistanceMetric.L2,
        normalization: bool = True,
    ):
        """
        Args:
            state_dim: Initial dimension of state space (can be updated on first use)
            action_dim: Initial dimension of action space (can be updated on first use)
            buffer_size: Number of recent (s,a) pairs to track
            metric: Distance metric to use
            normalization: Whether to normalize features
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.metric = metric
        self.normalization = normalization
        self.initialized = False  # Track if dimensions have been confirmed
        
        # Buffer of recent (state, action) pairs
        self.buffer = deque(maxlen=buffer_size)
        
        # Running statistics for normalization
        if normalization:
            self.state_mean = np.zeros(state_dim)
            self.state_std = np.ones(state_dim)
            self.action_mean = np.zeros(action_dim)
            self.action_std = np.ones(action_dim)
            self.n_samples = 0
            
        # Mahalanobis covariance matrix
        if metric == DistanceMetric.MAHALANOBIS:
            feature_dim = state_dim + action_dim
            self.cov_matrix = np.eye(feature_dim)
            self.inv_cov_matrix = np.eye(feature_dim)
            
    def _reinitialize_if_needed(self, state: np.ndarray, action: np.ndarray):
        """Reinitialize dimensions if they don't match actual data"""
        actual_state_dim = state.shape[0] if state.ndim == 1 else np.prod(state.shape)
        actual_action_dim = action.shape[0] if action.ndim == 1 else np.prod(action.shape)
        
        if not self.initialized or actual_state_dim != self.state_dim or actual_action_dim != self.action_dim:
            self.state_dim = actual_state_dim
            self.action_dim = actual_action_dim
            self.initialized = True
            
            # Reinitialize statistics
            if self.normalization:
                self.state_mean = np.zeros(self.state_dim)
                self.state_std = np.ones(self.state_dim)
                self.action_mean = np.zeros(self.action_dim)
                self.action_std = np.ones(self.action_dim)
                self.n_samples = 0
                
            # Reinitialize Mahalanobis covariance
            if self.metric == DistanceMetric.MAHALANOBIS:
                feature_dim = self.state_dim + self.action_dim
                self.cov_matrix = np.eye(feature_dim)
                self.inv_cov_matrix = np.eye(feature_dim)
            
    def update_statistics(self, state: np.ndarray, action: np.ndarray):
        """Update running mean/std for normalization"""
        if not self.normalization:
            return
        
        # Ensure dimensions match actual data
        self._reinitialize_if_needed(state, action)
            
        # Incremental mean/std update (Welford's algorithm)
        self.n_samples += 1
        n = self.n_samples
        
        # State statistics
        delta_state = state - self.state_mean
        self.state_mean += delta_state / n
        delta2_state = state - self.state_mean
        if n > 1:
            self.state_std = np.sqrt(
                ((n - 2) * self.state_std**2 + delta_state * delta2_state) / (n - 1)
            )
            
        # Action statistics
        delta_action = action - self.action_mean
        self.action_mean += delta_action / n
        delta2_action = action - self.action_mean
        if n > 1:
            self.action_std = np.sqrt(
                ((n - 2) * self.action_std**2 + delta_action * delta2_action) / (n - 1)
            )
            
    def normalize_features(
        self,
        state: np.ndarray,
        action: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Normalize state and action using running statistics"""
        if not self.normalization or self.n_samples < 2:
            return state, action
            
        norm_state = (state - self.state_mean) / (self.state_std + 1e-8)
        norm_action = (action - self.action_mean) / (self.action_std + 1e-8)
        
        return norm_state, norm_action
        
    def add_experience(self, state: np.ndarray, action: np.ndarray):
        """Add (state, action) pair to novelty buffer"""
        # Update statistics first
        self.update_statistics(state, action)
        
        # Normalize if enabled
        if self.normalization:
            state, action = self.normalize_features(state, action)
            
        # Add to buffer
        self.buffer.append((state.copy(), action.copy()))
        
        # Update covariance for Mahalanobis (every 100 samples)
        if (self.metric == DistanceMetric.MAHALANOBIS and 
            len(self.buffer) % 100 == 0 and len(self.buffer) >= 100):
            self._update_covariance()
            
    def _update_covariance(self):
        """Update covariance matrix for Mahalanobis distance"""
        # Concatenate all (s,a) pairs
        features = np.vstack([
            np.concatenate([s, a]) for s, a in self.buffer
        ])
        
        # Compute covariance
        self.cov_matrix = np.cov(features.T)
        
        # Compute inverse (with regularization for stability)
        try:
            self.inv_cov_matrix = np.linalg.inv(
                self.cov_matrix + 1e-4 * np.eye(self.cov_matrix.shape[0])
            )
        except np.linalg.LinAlgError:
            # Fallback to identity if singular
            self.inv_cov_matrix = np.eye(self.cov_matrix.shape[0])
            
    def compute_novelty(
        self,
        state: np.ndarray,
        action: np.ndarray,
        k_neighbors: int = 5
    ) -> float:
        """
        Compute novelty bonus for (state, action) pair.
        
        Args:
            state: Current state
            action: Current action
            k_neighbors: Number of nearest neighbors to consider
            
        Returns:
            Novelty score (higher = more novel)
        """
        if len(self.buffer) < k_neighbors:
            # Not enough history - consider very novel
            return 1.0
            
        # Normalize if enabled
        if self.normalization:
            state, action = self.normalize_features(state, action)
            
        # Compute distances to all buffered pairs
        distances = []
        for buf_state, buf_action in self.buffer:
            dist = self._compute_distance(state, action, buf_state, buf_action)
            distances.append(dist)
            
        # Handle case when buffer has fewer samples than k_neighbors
        if len(distances) == 0:
            return 1.0  # Maximum novelty for first sample
            
        distances = np.array(distances)
        
        # Adjust k_neighbors if buffer is smaller
        actual_k = min(k_neighbors, len(distances))
        if actual_k == len(distances):
            # Use all distances if buffer is small
            k_nearest_dists = distances
        else:
            # Use k nearest neighbors
            k_nearest_dists = np.partition(distances, actual_k)[:actual_k]
            
        novelty = np.mean(k_nearest_dists)
        
        # Normalize to [0, 1] range (using tanh scaling)
        novelty_normalized = np.tanh(novelty / 5.0)
        
        return float(novelty_normalized)
        
    def _compute_distance(
        self,
        state1: np.ndarray,
        action1: np.ndarray,
        state2: np.ndarray,
        action2: np.ndarray
    ) -> float:
        """Compute distance between two (state, action) pairs"""
        if self.metric == DistanceMetric.L2:
            return self._l2_distance(state1, action1, state2, action2)
        elif self.metric == DistanceMetric.MAHALANOBIS:
            return self._mahalanobis_distance(state1, action1, state2, action2)
        elif self.metric == DistanceMetric.COSINE:
            return self._cosine_distance(state1, action1, state2, action2)
        else:
            raise ValueError(f"Unknown metric: {self.metric}")
            
    def _l2_distance(
        self,
        state1: np.ndarray,
        action1: np.ndarray,
        state2: np.ndarray,
        action2: np.ndarray
    ) -> float:
        """Euclidean distance in concatenated (s,a) space"""
        feat1 = np.concatenate([state1, action1])
        feat2 = np.concatenate([state2, action2])
        return float(np.linalg.norm(feat1 - feat2))
        
    def _mahalanobis_distance(
        self,
        state1: np.ndarray,
        action1: np.ndarray,
        state2: np.ndarray,
        action2: np.ndarray
    ) -> float:
        """Mahalanobis distance accounting for feature correlations"""
        feat1 = np.concatenate([state1, action1])
        feat2 = np.concatenate([state2, action2])
        diff = feat1 - feat2
        
        # Mahalanobis: sqrt((x-y)^T * Σ^-1 * (x-y))
        return float(np.sqrt(diff @ self.inv_cov_matrix @ diff))
        
    def _cosine_distance(
        self,
        state1: np.ndarray,
        action1: np.ndarray,
        state2: np.ndarray,
        action2: np.ndarray
    ) -> float:
        """Cosine distance (1 - cosine similarity)"""
        feat1 = np.concatenate([state1, action1])
        feat2 = np.concatenate([state2, action2])
        
        # Cosine similarity
        cos_sim = np.dot(feat1, feat2) / (
            np.linalg.norm(feat1) * np.linalg.norm(feat2) + 1e-8
        )
        
        # Convert to distance
        return float(1.0 - cos_sim)
        
    def reset(self):
        """Clear buffer and statistics"""
        self.buffer.clear()
        if self.normalization:
            self.state_mean = np.zeros(self.state_dim)
            self.state_std = np.ones(self.state_dim)
            self.action_mean = np.zeros(self.action_dim)
            self.action_std = np.ones(self.action_dim)
            self.n_samples = 0
