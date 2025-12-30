"""
Multi-Armed Bandit Environments

Bandit environments for learning exploration-exploitation strategies.
These environments are stateless - each action is independent.

This environment is referenced in:
    - Chapter: Multi-Armed Bandits (0020-multi-armed-bandits)
    - Chapter: Contextual Bandits (0030-contextual-bandits)

Example:
    >>> env = MultiArmedBandit(k=10)
    >>> state, _ = env.reset()
    >>> action = 3  # Pull arm 3
    >>> _, reward, _, _, _ = env.step(action)
"""

from typing import Optional, Tuple, Dict, Any
import numpy as np
import gymnasium as gym
from gymnasium import spaces


class MultiArmedBandit(gym.Env):
    """
    Classic K-armed bandit testbed.

    Each arm has a true mean reward sampled from N(0, 1). When pulled,
    the arm returns a reward sampled from N(true_mean, 1).

    This is the environment from Sutton & Barto Chapter 2.

    Args:
        k: Number of arms (default: 10)
        stationary: If False, true values do random walks (nonstationary)
        walk_std: Standard deviation of random walk (only if nonstationary)

    Observation Space:
        Discrete(1) - There's only one "state" in a bandit problem

    Action Space:
        Discrete(k) - Choose which arm to pull
    """

    def __init__(
        self,
        k: int = 10,
        stationary: bool = True,
        walk_std: float = 0.01,
    ):
        super().__init__()

        self.k = k
        self.stationary = stationary
        self.walk_std = walk_std

        self.observation_space = spaces.Discrete(1)
        self.action_space = spaces.Discrete(k)

        self._rng = np.random.default_rng()
        self.true_values = np.zeros(k)
        self._initialize_arms()

    def _initialize_arms(self) -> None:
        """Initialize true arm values from N(0, 1)."""
        self.true_values = self._rng.standard_normal(self.k)
        self.optimal_action = int(np.argmax(self.true_values))

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[int, Dict[str, Any]]:
        """Reset the bandit (reinitialize arm values)."""
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        self._initialize_arms()
        return 0, self._get_info()

    def step(self, action: int) -> Tuple[int, float, bool, bool, Dict[str, Any]]:
        """
        Pull an arm and receive a reward.

        Args:
            action: Which arm to pull (0 to k-1)

        Returns:
            observation: Always 0 (bandits are stateless)
            reward: Sampled from N(true_value[action], 1)
            terminated: Always False (bandits don't terminate)
            truncated: Always False
            info: Contains optimal action and whether this was optimal
        """
        # Sample reward from arm's distribution
        reward = self._rng.normal(self.true_values[action], 1.0)

        # Nonstationary: random walk on true values
        if not self.stationary:
            self.true_values += self._rng.normal(0, self.walk_std, self.k)
            self.optimal_action = int(np.argmax(self.true_values))

        info = self._get_info()
        info["is_optimal"] = action == self.optimal_action

        return 0, float(reward), False, False, info

    def _get_info(self) -> Dict[str, Any]:
        """Return information about the current state."""
        return {
            "optimal_action": self.optimal_action,
            "optimal_value": float(self.true_values[self.optimal_action]),
            "true_values": self.true_values.copy(),
        }


class BernoulliBandit(gym.Env):
    """
    Bernoulli bandit where each arm has a fixed success probability.

    Useful for problems where rewards are binary (success/failure).

    Args:
        probs: List of success probabilities for each arm
    """

    def __init__(self, probs: Optional[list] = None):
        super().__init__()

        if probs is None:
            probs = [0.1, 0.3, 0.5, 0.7, 0.9]

        self.probs = np.array(probs)
        self.k = len(probs)

        self.observation_space = spaces.Discrete(1)
        self.action_space = spaces.Discrete(self.k)

        self._rng = np.random.default_rng()
        self.optimal_action = int(np.argmax(self.probs))

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[int, Dict[str, Any]]:
        """Reset the environment."""
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        return 0, {"optimal_action": self.optimal_action}

    def step(self, action: int) -> Tuple[int, float, bool, bool, Dict[str, Any]]:
        """Pull an arm, receive 1.0 (success) or 0.0 (failure)."""
        reward = float(self._rng.random() < self.probs[action])
        info = {"is_optimal": action == self.optimal_action}
        return 0, reward, False, False, info


class ContextualBandit(gym.Env):
    """
    Contextual bandit where optimal arm depends on context.

    Each step provides a context vector, and the agent must learn
    which arm is best for each context.

    Args:
        n_contexts: Number of distinct contexts
        k: Number of arms
        context_dim: Dimensionality of context features
    """

    def __init__(
        self,
        n_contexts: int = 4,
        k: int = 5,
        context_dim: int = 4,
    ):
        super().__init__()

        self.n_contexts = n_contexts
        self.k = k
        self.context_dim = context_dim

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(context_dim,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(k)

        self._rng = np.random.default_rng()
        self._initialize()

    def _initialize(self) -> None:
        """Initialize context features and optimal arms."""
        # Each context has a feature vector
        self.context_features = self._rng.standard_normal((self.n_contexts, self.context_dim))

        # Each context has expected rewards for each arm
        self.expected_rewards = self._rng.uniform(0, 1, (self.n_contexts, self.k))

        # Current context
        self.current_context = 0

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset and sample a new context."""
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)
            self._initialize()

        self.current_context = self._rng.integers(self.n_contexts)
        return self._get_obs(), self._get_info()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Take action, receive reward, get new context."""
        # Reward based on current context and action
        expected = self.expected_rewards[self.current_context, action]
        reward = self._rng.normal(expected, 0.1)

        # Sample new context
        self.current_context = self._rng.integers(self.n_contexts)

        return self._get_obs(), float(reward), False, False, self._get_info()

    def _get_obs(self) -> np.ndarray:
        """Return current context features."""
        return self.context_features[self.current_context].astype(np.float32)

    def _get_info(self) -> Dict[str, Any]:
        """Return info about current context."""
        optimal = int(np.argmax(self.expected_rewards[self.current_context]))
        return {
            "context_id": self.current_context,
            "optimal_action": optimal,
        }
