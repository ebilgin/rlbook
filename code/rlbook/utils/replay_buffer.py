"""
Replay Buffer implementations

Experience replay is crucial for stable training of deep RL algorithms.
Breaking temporal correlations and reusing experiences improves sample efficiency.

Referenced in:
    - Chapter: Deep Q-Networks (1040-deep-q-networks)
"""

from typing import Tuple, Optional
from collections import deque
import random
import numpy as np


class ReplayBuffer:
    """
    Simple replay buffer for storing and sampling transitions.

    Experience replay serves two purposes:
    1. Break temporal correlations between consecutive samples
    2. Reuse past experience for better sample efficiency

    Args:
        capacity: Maximum number of transitions to store
    """

    def __init__(self, capacity: int = 10000):
        self.buffer = deque(maxlen=capacity)

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """Store a transition."""
        self.buffer.append((state, action, reward, next_state, done))

    def sample(
        self, batch_size: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Sample a random batch of transitions.

        Returns:
            Tuple of (states, actions, rewards, next_states, dones)
        """
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        return (
            np.array(states),
            np.array(actions),
            np.array(rewards, dtype=np.float32),
            np.array(next_states),
            np.array(dones, dtype=np.float32),
        )

    def __len__(self) -> int:
        return len(self.buffer)

    def is_ready(self, batch_size: int) -> bool:
        """Check if buffer has enough samples for a batch."""
        return len(self.buffer) >= batch_size


class PrioritizedReplayBuffer:
    """
    Prioritized Experience Replay (PER) buffer.

    Samples transitions with probability proportional to their TD error.
    Important transitions (high error) are replayed more frequently.

    From: "Prioritized Experience Replay" (Schaul et al., 2015)

    Args:
        capacity: Maximum buffer size
        alpha: How much prioritization to use (0 = uniform, 1 = full)
        beta: Importance sampling correction (starts low, anneals to 1)
        beta_increment: How much to increase beta each sample
        epsilon: Small constant to ensure non-zero probability
    """

    def __init__(
        self,
        capacity: int = 10000,
        alpha: float = 0.6,
        beta: float = 0.4,
        beta_increment: float = 0.001,
        epsilon: float = 1e-6,
    ):
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.epsilon = epsilon

        self.buffer = []
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.position = 0
        self.size = 0

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        priority: Optional[float] = None,
    ) -> None:
        """
        Store a transition with priority.

        New transitions get max priority to ensure they're sampled at least once.
        """
        if priority is None:
            # New transitions get max priority
            priority = self.priorities.max() if self.size > 0 else 1.0

        if self.size < self.capacity:
            self.buffer.append((state, action, reward, next_state, done))
            self.size += 1
        else:
            self.buffer[self.position] = (state, action, reward, next_state, done)

        self.priorities[self.position] = priority
        self.position = (self.position + 1) % self.capacity

    def sample(
        self, batch_size: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Sample a batch with prioritized probabilities.

        Returns:
            states, actions, rewards, next_states, dones, weights, indices
            weights: Importance sampling weights to correct for bias
            indices: Buffer indices for updating priorities
        """
        # Compute sampling probabilities
        priorities = self.priorities[: self.size]
        probs = priorities ** self.alpha
        probs /= probs.sum()

        # Sample indices
        indices = np.random.choice(self.size, batch_size, p=probs, replace=False)

        # Get transitions
        batch = [self.buffer[i] for i in indices]
        states, actions, rewards, next_states, dones = zip(*batch)

        # Compute importance sampling weights
        # w_i = (N * P(i))^(-beta) / max(w)
        weights = (self.size * probs[indices]) ** (-self.beta)
        weights /= weights.max()  # Normalize

        # Anneal beta toward 1
        self.beta = min(1.0, self.beta + self.beta_increment)

        return (
            np.array(states),
            np.array(actions),
            np.array(rewards, dtype=np.float32),
            np.array(next_states),
            np.array(dones, dtype=np.float32),
            weights.astype(np.float32),
            indices,
        )

    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray) -> None:
        """
        Update priorities based on TD errors.

        Args:
            indices: Buffer indices to update
            td_errors: New TD errors for these transitions
        """
        for idx, error in zip(indices, td_errors):
            self.priorities[idx] = abs(error) + self.epsilon

    def __len__(self) -> int:
        return self.size

    def is_ready(self, batch_size: int) -> bool:
        return self.size >= batch_size
