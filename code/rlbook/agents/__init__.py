"""
rlbook.agents - Reinforcement learning agent implementations

This module provides clean, educational implementations of RL algorithms.
Each agent includes detailed docstrings explaining the algorithm.

Available agents:
    - QLearningAgent: Tabular Q-learning (off-policy TD control)
    - SarsaAgent: Tabular SARSA (on-policy TD control)
    - ExpectedSarsaAgent: Expected SARSA (on-policy with lower variance)
    - DQNAgent: Deep Q-Network with experience replay
    - DoubleDQNAgent: Double DQN (reduces overestimation bias)
    - ElevatorDQN: Multi-agent DQN for elevator dispatch
    - GRPOTrainer: Group Relative Policy Optimization for LLM fine-tuning
"""

from rlbook.agents.q_learning import QLearningAgent, SarsaAgent, ExpectedSarsaAgent
from rlbook.agents.dqn import DQNAgent, DoubleDQNAgent
from rlbook.agents.elevator_dqn import ElevatorDQN
from rlbook.agents.grpo import GRPOTrainer, GRPOConfig

__all__ = [
    "QLearningAgent",
    "SarsaAgent",
    "ExpectedSarsaAgent",
    "DQNAgent",
    "DoubleDQNAgent",
    "ElevatorDQN",
    "GRPOTrainer",
    "GRPOConfig",
]
