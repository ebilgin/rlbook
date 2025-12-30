"""Tests for rlbook agents."""

import pytest
import numpy as np
from rlbook.agents import QLearningAgent, SarsaAgent


class TestQLearningAgent:
    """Tests for Q-Learning agent."""

    def test_initialization(self):
        """Test basic initialization."""
        agent = QLearningAgent(n_states=16, n_actions=4)
        assert agent.q_table.shape == (16, 4)
        assert np.all(agent.q_table == 0)

    def test_select_action_exploration(self):
        """Test epsilon-greedy exploration."""
        agent = QLearningAgent(n_states=16, n_actions=4, epsilon=1.0)

        # With epsilon=1, all actions should be random
        actions = [agent.select_action(0) for _ in range(100)]

        # Should see variety in actions
        assert len(set(actions)) > 1

    def test_select_action_exploitation(self):
        """Test greedy action selection."""
        agent = QLearningAgent(n_states=16, n_actions=4, epsilon=0.0)

        # Set Q-values so action 2 is best
        agent.q_table[0, 2] = 10.0

        # Should always select action 2
        for _ in range(10):
            assert agent.select_action(0) == 2

    def test_update(self):
        """Test Q-value update."""
        agent = QLearningAgent(
            n_states=16, n_actions=4,
            learning_rate=0.1, gamma=0.9
        )

        # Initial Q-value
        assert agent.q_table[0, 0] == 0

        # Update after getting reward
        td_error = agent.update(state=0, action=0, reward=1.0, next_state=1, done=False)

        # Q should increase
        assert agent.q_table[0, 0] > 0
        assert td_error > 0

    def test_terminal_update(self):
        """Test update at terminal state."""
        agent = QLearningAgent(
            n_states=16, n_actions=4,
            learning_rate=1.0, gamma=0.9  # lr=1 for exact update
        )

        # Terminal update: Q(s,a) = r (no future value)
        agent.update(state=0, action=0, reward=10.0, next_state=15, done=True)

        # Should be exactly the reward
        assert agent.q_table[0, 0] == 10.0

    def test_epsilon_decay(self):
        """Test epsilon decays correctly."""
        agent = QLearningAgent(
            n_states=16, n_actions=4,
            epsilon=1.0, epsilon_decay=0.9, epsilon_min=0.1
        )

        initial_eps = agent.epsilon
        agent.decay_epsilon()

        assert agent.epsilon < initial_eps
        assert agent.epsilon == 0.9

    def test_epsilon_min(self):
        """Test epsilon doesn't go below minimum."""
        agent = QLearningAgent(
            n_states=16, n_actions=4,
            epsilon=0.1, epsilon_decay=0.5, epsilon_min=0.1
        )

        agent.decay_epsilon()
        assert agent.epsilon == 0.1  # Should stay at min

    def test_get_policy(self):
        """Test policy extraction."""
        agent = QLearningAgent(n_states=4, n_actions=2)

        # Set Q-values
        agent.q_table[0, 1] = 5.0  # State 0 -> action 1
        agent.q_table[1, 0] = 3.0  # State 1 -> action 0
        agent.q_table[2, 1] = 2.0  # State 2 -> action 1
        agent.q_table[3, 0] = 1.0  # State 3 -> action 0

        policy = agent.get_policy()

        assert policy[0] == 1
        assert policy[1] == 0
        assert policy[2] == 1
        assert policy[3] == 0

    def test_learning_progress(self):
        """Test agent actually learns on simple problem."""
        # Simple 2-state problem: state 0, action 1 leads to reward
        agent = QLearningAgent(
            n_states=2, n_actions=2,
            learning_rate=0.5, gamma=0.9, epsilon=0.3
        )

        # Train: state 0, action 1 gives reward 1 and terminates
        for _ in range(100):
            agent.update(state=0, action=1, reward=1.0, next_state=1, done=True)
            agent.update(state=0, action=0, reward=0.0, next_state=0, done=False)

        # Should learn action 1 is better in state 0
        assert agent.q_table[0, 1] > agent.q_table[0, 0]


class TestSarsaAgent:
    """Tests for SARSA agent."""

    def test_initialization(self):
        """Test basic initialization."""
        agent = SarsaAgent(n_states=16, n_actions=4)
        assert agent.q_table.shape == (16, 4)

    def test_update_uses_next_action(self):
        """Test SARSA uses actual next action, not max."""
        agent = SarsaAgent(
            n_states=16, n_actions=4,
            learning_rate=1.0, gamma=0.9
        )

        # Set up: Q(next_state, best_action) = 10, Q(next_state, next_action) = 2
        agent.q_table[1, 0] = 10.0  # Best action
        agent.q_table[1, 2] = 2.0   # Action we'll actually take

        # SARSA update with next_action=2
        agent.update(
            state=0, action=0, reward=0.0,
            next_state=1, next_action=2, done=False
        )

        # Should use Q(1, 2)=2, not Q(1, 0)=10
        # Q(0,0) = 0 + 1.0 * (0 + 0.9 * 2 - 0) = 1.8
        assert abs(agent.q_table[0, 0] - 1.8) < 0.01
