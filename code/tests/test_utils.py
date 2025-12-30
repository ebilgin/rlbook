"""Tests for rlbook utility modules."""

import pytest
import numpy as np
from rlbook.utils.replay_buffer import ReplayBuffer, PrioritizedReplayBuffer


class TestReplayBuffer:
    """Tests for simple replay buffer."""

    def test_initialization(self):
        """Test buffer is created empty."""
        buffer = ReplayBuffer(capacity=100)
        assert len(buffer) == 0
        assert not buffer.is_ready(batch_size=10)

    def test_push_and_len(self):
        """Test adding transitions."""
        buffer = ReplayBuffer(capacity=100)

        for i in range(10):
            buffer.push(
                state=np.array([i]),
                action=0,
                reward=1.0,
                next_state=np.array([i + 1]),
                done=False,
            )

        assert len(buffer) == 10
        assert buffer.is_ready(batch_size=10)
        assert not buffer.is_ready(batch_size=11)

    def test_capacity_limit(self):
        """Test buffer doesn't exceed capacity."""
        buffer = ReplayBuffer(capacity=5)

        for i in range(10):
            buffer.push(
                state=np.array([i]),
                action=0,
                reward=1.0,
                next_state=np.array([i + 1]),
                done=False,
            )

        assert len(buffer) == 5

    def test_sample_returns_correct_shapes(self):
        """Test sample returns correctly shaped arrays."""
        buffer = ReplayBuffer(capacity=100)

        # Add transitions with 4-dimensional states
        for i in range(20):
            buffer.push(
                state=np.array([i, i + 1, i + 2, i + 3]),
                action=i % 4,
                reward=float(i),
                next_state=np.array([i + 1, i + 2, i + 3, i + 4]),
                done=(i % 5 == 0),
            )

        states, actions, rewards, next_states, dones = buffer.sample(batch_size=8)

        assert states.shape == (8, 4)
        assert actions.shape == (8,)
        assert rewards.shape == (8,)
        assert next_states.shape == (8, 4)
        assert dones.shape == (8,)

    def test_sample_returns_stored_values(self):
        """Test sample returns values that were actually stored."""
        buffer = ReplayBuffer(capacity=100)

        # Add a single transition
        buffer.push(
            state=np.array([1, 2, 3]),
            action=2,
            reward=5.0,
            next_state=np.array([4, 5, 6]),
            done=True,
        )

        states, actions, rewards, next_states, dones = buffer.sample(batch_size=1)

        np.testing.assert_array_equal(states[0], [1, 2, 3])
        assert actions[0] == 2
        assert rewards[0] == 5.0
        np.testing.assert_array_equal(next_states[0], [4, 5, 6])
        assert dones[0] == 1.0  # True becomes 1.0


class TestPrioritizedReplayBuffer:
    """Tests for prioritized replay buffer."""

    def test_initialization(self):
        """Test buffer is created empty."""
        buffer = PrioritizedReplayBuffer(capacity=100)
        assert len(buffer) == 0
        assert not buffer.is_ready(batch_size=10)

    def test_push_and_len(self):
        """Test adding transitions."""
        buffer = PrioritizedReplayBuffer(capacity=100)

        for i in range(10):
            buffer.push(
                state=np.array([i]),
                action=0,
                reward=1.0,
                next_state=np.array([i + 1]),
                done=False,
            )

        assert len(buffer) == 10

    def test_capacity_limit(self):
        """Test buffer doesn't exceed capacity."""
        buffer = PrioritizedReplayBuffer(capacity=5)

        for i in range(10):
            buffer.push(
                state=np.array([i]),
                action=0,
                reward=1.0,
                next_state=np.array([i + 1]),
                done=False,
            )

        assert len(buffer) == 5

    def test_sample_returns_weights_and_indices(self):
        """Test sample returns importance weights and indices."""
        buffer = PrioritizedReplayBuffer(capacity=100)

        for i in range(20):
            buffer.push(
                state=np.array([i]),
                action=0,
                reward=1.0,
                next_state=np.array([i + 1]),
                done=False,
            )

        result = buffer.sample(batch_size=8)
        states, actions, rewards, next_states, dones, weights, indices = result

        assert states.shape == (8, 1)
        assert weights.shape == (8,)
        assert indices.shape == (8,)

        # Weights should be positive and max-normalized to 1
        assert np.all(weights > 0)
        assert np.all(weights <= 1.0)
        assert np.max(weights) == pytest.approx(1.0)

    def test_update_priorities(self):
        """Test priority updates affect sampling."""
        buffer = PrioritizedReplayBuffer(capacity=100, alpha=1.0)

        # Add transitions with equal priority initially
        for i in range(10):
            buffer.push(
                state=np.array([i]),
                action=0,
                reward=1.0,
                next_state=np.array([i + 1]),
                done=False,
                priority=1.0,
            )

        # Give index 0 very high priority
        buffer.update_priorities(
            indices=np.array([0]),
            td_errors=np.array([100.0]),
        )

        # Sample many times and check index 0 appears more often
        samples_with_idx_0 = 0
        for _ in range(100):
            result = buffer.sample(batch_size=5)
            indices = result[6]
            if 0 in indices:
                samples_with_idx_0 += 1

        # With high priority, index 0 should appear in most samples
        assert samples_with_idx_0 > 50  # At least half the time

    def test_beta_anneals(self):
        """Test beta increases toward 1 over time."""
        buffer = PrioritizedReplayBuffer(
            capacity=100,
            beta=0.4,
            beta_increment=0.1,
        )

        for i in range(10):
            buffer.push(
                state=np.array([i]),
                action=0,
                reward=1.0,
                next_state=np.array([i + 1]),
                done=False,
            )

        initial_beta = buffer.beta
        buffer.sample(batch_size=5)

        assert buffer.beta > initial_beta
        assert buffer.beta == pytest.approx(0.5)

        # Beta should cap at 1.0
        for _ in range(20):
            buffer.sample(batch_size=5)

        assert buffer.beta == 1.0

    def test_new_transitions_get_max_priority(self):
        """Test new transitions get max priority by default."""
        buffer = PrioritizedReplayBuffer(capacity=100)

        # First transition sets baseline
        buffer.push(
            state=np.array([0]),
            action=0,
            reward=1.0,
            next_state=np.array([1]),
            done=False,
        )

        # Update its priority
        buffer.update_priorities(
            indices=np.array([0]),
            td_errors=np.array([10.0]),
        )

        # New transition should get max priority
        buffer.push(
            state=np.array([1]),
            action=0,
            reward=1.0,
            next_state=np.array([2]),
            done=False,
        )

        # Check both have high priority
        assert buffer.priorities[0] > 0
        assert buffer.priorities[1] > 0
