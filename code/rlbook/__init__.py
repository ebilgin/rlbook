"""
rlbook - Production-grade RL implementations from rlbook.ai

This package provides tested, educational implementations of reinforcement
learning algorithms and environments that accompany the rlbook.ai content.

Example:
    >>> from rlbook.envs import GridWorld
    >>> from rlbook.agents import QLearningAgent
    >>>
    >>> env = GridWorld(size=4)
    >>> agent = QLearningAgent(n_states=16, n_actions=4)
"""

__version__ = "0.1.0"

from rlbook import agents, envs, utils

__all__ = ["agents", "envs", "utils", "__version__"]
