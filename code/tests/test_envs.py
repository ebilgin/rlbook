"""Tests for rlbook environments."""

import pytest
import numpy as np
from rlbook.envs import GridWorld, MultiArmedBandit


class TestGridWorld:
    """Tests for GridWorld environment."""

    def test_initialization(self):
        """Test basic initialization."""
        env = GridWorld(size=4)
        assert env.size == 4
        assert env.observation_space.n == 16
        assert env.action_space.n == 4

    def test_reset(self):
        """Test reset returns valid state."""
        env = GridWorld(size=4)
        state, info = env.reset(seed=42)
        assert 0 <= state < 16
        assert "agent_pos" in info
        assert info["agent_pos"] == (0, 0)

    def test_step_actions(self):
        """Test all actions work correctly."""
        env = GridWorld(size=4)
        env.reset(seed=42)

        # DOWN should increase row
        state, reward, done, _, info = env.step(GridWorld.DOWN)
        assert info["agent_pos"] == (1, 0)

        # RIGHT should increase col
        state, reward, done, _, info = env.step(GridWorld.RIGHT)
        assert info["agent_pos"] == (1, 1)

    def test_wall_collision(self):
        """Test agent stays in place when hitting wall."""
        env = GridWorld(size=4)
        env.reset(seed=42)

        # Try to go UP at top edge - should stay
        state, _, _, _, info = env.step(GridWorld.UP)
        assert info["agent_pos"] == (0, 0)

        # Try to go LEFT at left edge - should stay
        state, _, _, _, info = env.step(GridWorld.LEFT)
        assert info["agent_pos"] == (0, 0)

    def test_goal_reached(self):
        """Test episode terminates at goal."""
        env = GridWorld(size=2)  # Small grid for quick test
        env.reset(seed=42)

        # Navigate to goal (1, 1)
        env.step(GridWorld.DOWN)  # (0,0) -> (1,0)
        state, reward, done, _, info = env.step(GridWorld.RIGHT)  # (1,0) -> (1,1)

        assert done
        assert reward == env.goal_reward
        assert info["agent_pos"] == (1, 1)

    def test_obstacles(self):
        """Test obstacles block movement."""
        obstacles = [(1, 0)]
        env = GridWorld(size=4, obstacles=obstacles)
        env.reset(seed=42)

        # Try to move to obstacle
        state, _, _, _, info = env.step(GridWorld.DOWN)
        assert info["agent_pos"] == (0, 0)  # Blocked

    def test_reproducibility(self):
        """Test same seed gives same results."""
        env1 = GridWorld(size=4, stochastic=True)
        env2 = GridWorld(size=4, stochastic=True)

        env1.reset(seed=123)
        env2.reset(seed=123)

        for _ in range(10):
            action = np.random.randint(4)
            s1, r1, d1, _, _ = env1.step(action)
            s2, r2, d2, _, _ = env2.step(action)
            assert s1 == s2
            assert r1 == r2


class TestMultiArmedBandit:
    """Tests for MultiArmedBandit environment."""

    def test_initialization(self):
        """Test basic initialization."""
        env = MultiArmedBandit(k=10)
        assert env.k == 10
        assert env.action_space.n == 10
        assert env.observation_space.n == 1

    def test_reset(self):
        """Test reset initializes arm values."""
        env = MultiArmedBandit(k=5)
        state, info = env.reset(seed=42)
        assert state == 0
        assert "true_values" in info
        assert len(info["true_values"]) == 5

    def test_step_returns_reward(self):
        """Test step returns valid reward."""
        env = MultiArmedBandit(k=5)
        env.reset(seed=42)

        state, reward, done, truncated, info = env.step(0)
        assert state == 0  # Bandits are stateless
        assert not done
        assert not truncated
        assert isinstance(reward, float)

    def test_optimal_action(self):
        """Test optimal action is correctly identified."""
        env = MultiArmedBandit(k=3)
        env.reset(seed=42)

        optimal = info = env._get_info()["optimal_action"]
        assert optimal == np.argmax(env.true_values)

    def test_nonstationary(self):
        """Test nonstationary bandit changes over time."""
        env = MultiArmedBandit(k=3, stationary=False, walk_std=1.0)
        env.reset(seed=42)

        initial_values = env.true_values.copy()

        # Take many steps
        for _ in range(100):
            env.step(0)

        # Values should have changed
        assert not np.allclose(initial_values, env.true_values)
