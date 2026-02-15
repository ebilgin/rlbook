"""
rlbook.envs - Reinforcement learning environments

This module provides Gymnasium-compatible environments for learning RL concepts.
All environments follow the standard Gymnasium API.

Available environments:
    - GridWorld: Classic grid navigation environment
    - MultiArmedBandit: K-armed bandit testbed
    - ElevatorDispatch: Multi-elevator dispatch system
"""

from rlbook.envs.gridworld import GridWorld
from rlbook.envs.bandits import MultiArmedBandit
from rlbook.envs.elevator import ElevatorDispatch

__all__ = ["GridWorld", "MultiArmedBandit", "ElevatorDispatch"]
