"""
Tests for ElevatorDispatch environment.
"""

import pytest
import numpy as np
from rlbook.envs import ElevatorDispatch


class TestElevatorDispatch:
    """Test suite for elevator dispatch environment."""

    def test_initialization(self):
        """Test environment initializes with correct parameters."""
        env = ElevatorDispatch(n_floors=10, n_elevators=3)
        assert env.n_floors == 10
        assert env.n_elevators == 3
        assert env.elevator_capacity == 8
        assert len(env.elevators) == 0  # Not initialized until reset

    def test_reset_returns_valid_state(self):
        """Test reset returns valid observation and info."""
        env = ElevatorDispatch(n_floors=10, n_elevators=3)
        obs, info = env.reset(seed=42)

        # Check observation structure
        assert isinstance(obs, dict)
        assert len(obs) == 3  # One per elevator
        assert "elevator_0" in obs
        assert "elevator_1" in obs
        assert "elevator_2" in obs

        # Check observation shapes
        for i in range(3):
            assert obs[f"elevator_{i}"].shape[0] > 0
            assert obs[f"elevator_{i}"].dtype == np.float32

        # Check info
        assert "timestep" in info
        assert info["timestep"] == 0
        assert info["waiting_passengers"] == 0
        assert info["delivered_passengers"] == 0

    def test_step_returns_correct_format(self):
        """Test step returns correct tuple format."""
        env = ElevatorDispatch(n_floors=10, n_elevators=3)
        env.reset(seed=42)

        # Take random actions
        actions = [env.action_space.sample()[i] for i in range(env.n_elevators)]
        obs, reward, terminated, truncated, info = env.step(actions)

        # Check return types
        assert isinstance(obs, dict)
        assert isinstance(reward, (int, float))
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)

        # Episode should not terminate early
        assert terminated is False

    def test_elevator_movement(self):
        """Test elevators move toward target floors."""
        env = ElevatorDispatch(n_floors=10, n_elevators=3, move_speed=1.0)
        env.reset(seed=42)

        # Command elevator 0 to floor 5
        initial_floor = env.elevators[0].floor
        actions = [5, 0, 0]  # Elevator 0 → floor 5

        # Take several steps
        for _ in range(10):
            obs, reward, _, _, info = env.step(actions)

        # Elevator should have moved toward floor 5
        final_floor = env.elevators[0].floor
        assert final_floor > initial_floor
        assert final_floor <= 5.0

    def test_passenger_spawning(self):
        """Test passengers spawn over time."""
        env = ElevatorDispatch(
            n_floors=10,
            n_elevators=3,
            passenger_spawn_rate=1.0,  # High rate for testing
        )
        env.reset(seed=42)

        # Run for several timesteps
        actions = [0] * env.n_elevators
        for _ in range(50):
            obs, reward, _, _, info = env.step(actions)

        # Should have spawned some passengers
        assert info["total_spawned"] > 0

    def test_passenger_pickup_and_delivery(self):
        """Test passengers are picked up and delivered."""
        env = ElevatorDispatch(n_floors=10, n_elevators=1, move_speed=0.5)
        env.reset(seed=42)

        # Manually spawn a passenger
        from rlbook.envs.elevator import Passenger
        passenger = Passenger(
            id=0, origin_floor=0, dest_floor=5, spawn_time=0
        )
        env.waiting_passengers.append(passenger)
        env.next_passenger_id = 1

        # Elevator picks up at floor 0, delivers at floor 5
        # Command elevator to floor 5
        actions = [5]

        # Run until passenger is delivered
        for _ in range(20):
            obs, reward, _, _, info = env.step(actions)

        # Passenger should be delivered
        assert len(env.delivered_passengers) > 0
        assert env.delivered_passengers[0].dest_floor == 5

    def test_elevator_capacity(self):
        """Test elevator respects capacity limits."""
        env = ElevatorDispatch(n_floors=10, n_elevators=1, elevator_capacity=3)
        env.reset(seed=42)

        # Manually spawn 5 passengers at floor 0
        from rlbook.envs.elevator import Passenger
        for i in range(5):
            passenger = Passenger(
                id=i, origin_floor=0, dest_floor=5, spawn_time=0
            )
            env.waiting_passengers.append(passenger)
        env.next_passenger_id = 5

        # Elevator is at floor 0, should pick up only 3 (capacity)
        actions = [5]
        obs, reward, _, _, info = env.step(actions)

        # Should have 3 in elevator, 2 still waiting
        assert env.elevators[0].passenger_count <= 3
        assert len(env.waiting_passengers) >= 2

    def test_episode_truncation(self):
        """Test episode truncates after max_timesteps."""
        env = ElevatorDispatch(n_floors=10, n_elevators=3, max_timesteps=10)
        env.reset(seed=42)

        actions = [0] * env.n_elevators
        truncated = False

        for i in range(15):
            obs, reward, terminated, truncated, info = env.step(actions)
            if truncated:
                break

        assert truncated
        assert info["timestep"] == 10

    def test_reproducibility_with_seed(self):
        """Test environment is reproducible with same seed."""
        env1 = ElevatorDispatch(n_floors=10, n_elevators=3)
        env2 = ElevatorDispatch(n_floors=10, n_elevators=3)

        obs1, _ = env1.reset(seed=42)
        obs2, _ = env2.reset(seed=42)

        # Initial observations should match
        for i in range(3):
            np.testing.assert_array_equal(
                obs1[f"elevator_{i}"], obs2[f"elevator_{i}"]
            )

        # Run for a few steps with same actions
        actions = [0, 1, 2]
        for _ in range(10):
            obs1, r1, _, _, _ = env1.step(actions)
            obs2, r2, _, _, _ = env2.step(actions)

        # Observations and rewards should match
        assert r1 == r2
        for i in range(3):
            np.testing.assert_array_almost_equal(
                obs1[f"elevator_{i}"], obs2[f"elevator_{i}"], decimal=4
            )

    def test_traffic_patterns(self):
        """Test different traffic patterns spawn appropriate trips."""
        patterns = ["morning_rush", "lunch_time", "evening_rush", "quiet"]

        for pattern in patterns:
            env = ElevatorDispatch(
                n_floors=10,
                n_elevators=3,
                traffic_pattern=pattern,
                passenger_spawn_rate=1.0,
            )
            env.reset(seed=42)

            # Run for some time
            actions = [0] * env.n_elevators
            for _ in range(50):
                env.step(actions)

            # Should have spawned passengers
            assert env.next_passenger_id > 0, f"No passengers for {pattern}"

    def test_reward_components(self):
        """Test reward includes wait time, delivery, and energy components."""
        env = ElevatorDispatch(n_floors=10, n_elevators=1)
        env.reset(seed=42)

        # Manually add waiting passengers
        from rlbook.envs.elevator import Passenger
        passenger = Passenger(
            id=0, origin_floor=0, dest_floor=3, spawn_time=0
        )
        env.waiting_passengers.append(passenger)

        # Take a step (passenger waits)
        actions = [5]  # Wrong floor
        obs, reward, _, _, info = env.step(actions)

        # Reward should be negative (waiting time penalty + energy cost)
        assert reward < 0

    def test_render_ascii(self):
        """Test ASCII rendering produces output."""
        env = ElevatorDispatch(n_floors=5, n_elevators=2, render_mode="ansi")
        env.reset(seed=42)

        output = env.render()
        assert output is not None
        assert isinstance(output, str)
        assert "Floor" in output
        assert "Elevator Dispatch" in output

    def test_metrics_calculation(self):
        """Test metrics are correctly calculated."""
        env = ElevatorDispatch(n_floors=10, n_elevators=3)
        env.reset(seed=42)

        # Run for some time
        actions = [0] * env.n_elevators
        for _ in range(100):
            obs, reward, _, _, info = env.step(actions)

        # Check info contains expected metrics
        assert "avg_wait_time" in info
        assert "max_wait_time" in info
        assert "elevator_utilization" in info
        assert "total_movements" in info
        assert info["elevator_utilization"] >= 0.0
        assert info["elevator_utilization"] <= 1.0

    def test_direction_property(self):
        """Test elevator direction property works correctly."""
        from rlbook.envs.elevator import Elevator, Direction

        elevator = Elevator(id=0, floor=5.0)

        # No target → IDLE
        assert elevator.direction == Direction.IDLE

        # Target above → UP
        elevator.target_floor = 7
        assert elevator.direction == Direction.UP

        # Target below → DOWN
        elevator.target_floor = 3
        assert elevator.direction == Direction.DOWN

        # At target → IDLE
        elevator.target_floor = 5
        assert elevator.direction == Direction.IDLE
