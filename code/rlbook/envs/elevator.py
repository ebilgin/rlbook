"""
Elevator Dispatch Environment

A multi-elevator dispatch system for a building. Multiple elevators must coordinate
to serve passengers efficiently while minimizing wait times and energy consumption.

This environment is referenced in:
    - Application: Elevator Dispatch (0030-elevator-dispatch)

Example:
    >>> env = ElevatorDispatch(n_floors=10, n_elevators=3)
    >>> obs, _ = env.reset()
    >>> actions = [env.action_space.sample() for _ in range(env.n_elevators)]
    >>> obs, reward, done, truncated, info = env.step(actions)
"""

from typing import Optional, Tuple, List, Dict, Any
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from dataclasses import dataclass, field
from enum import IntEnum


class Direction(IntEnum):
    """Elevator movement direction."""
    UP = 0
    DOWN = 1
    IDLE = 2


class TrafficPattern(IntEnum):
    """Time-based traffic patterns."""
    MORNING_RUSH = 0    # Heavy lobby → upper floors
    LUNCH_TIME = 1      # Bidirectional, moderate
    EVENING_RUSH = 2    # Heavy upper floors → lobby
    QUIET = 3           # Low, random


@dataclass
class Passenger:
    """Represents a single passenger."""
    id: int
    origin_floor: int
    dest_floor: int
    spawn_time: int
    pickup_time: Optional[int] = None

    @property
    def wait_time(self) -> int:
        """Current wait time in timesteps."""
        if self.pickup_time is None:
            return 0  # Will be calculated when needed
        return self.pickup_time - self.spawn_time

    @property
    def direction(self) -> Direction:
        """Direction of travel."""
        if self.dest_floor > self.origin_floor:
            return Direction.UP
        elif self.dest_floor < self.origin_floor:
            return Direction.DOWN
        return Direction.IDLE


@dataclass
class Elevator:
    """Represents a single elevator."""
    id: int
    floor: float = 0.0  # Can be between floors during movement
    target_floor: Optional[int] = None
    passengers: List[Passenger] = field(default_factory=list)
    max_capacity: int = 8
    is_moving: bool = False

    @property
    def current_floor(self) -> int:
        """Integer floor position."""
        return int(np.round(self.floor))

    @property
    def passenger_count(self) -> int:
        """Number of passengers currently in elevator."""
        return len(self.passengers)

    @property
    def is_full(self) -> bool:
        """Check if elevator is at capacity."""
        return self.passenger_count >= self.max_capacity

    @property
    def direction(self) -> Direction:
        """Current movement direction."""
        if self.target_floor is None:
            return Direction.IDLE
        elif self.target_floor > self.current_floor:
            return Direction.UP
        elif self.target_floor < self.current_floor:
            return Direction.DOWN
        return Direction.IDLE

    @property
    def destination_floors(self) -> List[int]:
        """List of floors where passengers want to go."""
        return sorted(set(p.dest_floor for p in self.passengers))


class ElevatorDispatch(gym.Env):
    """
    Multi-elevator dispatch environment.

    A building with multiple elevators serving passengers. The goal is to minimize
    passenger wait times while considering energy efficiency. Elevators must learn
    to coordinate without explicit communication.

    Args:
        n_floors: Number of floors in the building (default: 10)
        n_elevators: Number of elevators (default: 3)
        elevator_capacity: Maximum passengers per elevator (default: 8)
        traffic_pattern: Traffic pattern type (default: "morning_rush")
        passenger_spawn_rate: Base Poisson rate for passenger arrivals (default: 0.3)
        max_timesteps: Maximum timesteps per episode (default: 300)
        move_speed: Floors per timestep (default: 0.5, takes 2 steps per floor)
        render_mode: Visualization mode ("human", "ansi", None)

    Observation Space:
        Dict with per-elevator observations (Box of continuous features)
        Each elevator observes:
        - Own state (floor, direction, passenger count, destination floors)
        - Global state (pending requests, other elevators' positions)
        - Traffic pattern (one-hot encoded)

    Action Space:
        MultiDiscrete - one action per elevator, selecting target floor (0 to n_floors-1)

    Rewards:
        - Primary: -1 × total waiting time of all passengers this timestep
        - Bonus: +10 for each passenger delivered
        - Penalty: -5 for each passenger waiting >60 seconds
        - Energy: -0.1 per floor moved
    """

    metadata = {"render_modes": ["human", "ansi"]}

    def __init__(
        self,
        n_floors: int = 10,
        n_elevators: int = 3,
        elevator_capacity: int = 8,
        traffic_pattern: str = "morning_rush",
        passenger_spawn_rate: float = 0.3,
        max_timesteps: int = 300,
        move_speed: float = 0.5,
        render_mode: Optional[str] = None,
    ):
        super().__init__()

        self.n_floors = n_floors
        self.n_elevators = n_elevators
        self.elevator_capacity = elevator_capacity
        self.passenger_spawn_rate = passenger_spawn_rate
        self.max_timesteps = max_timesteps
        self.move_speed = move_speed
        self.render_mode = render_mode

        # Parse traffic pattern
        pattern_map = {
            "morning_rush": TrafficPattern.MORNING_RUSH,
            "lunch_time": TrafficPattern.LUNCH_TIME,
            "evening_rush": TrafficPattern.EVENING_RUSH,
            "quiet": TrafficPattern.QUIET,
        }
        self.traffic_pattern = pattern_map.get(traffic_pattern, TrafficPattern.MORNING_RUSH)

        # Spaces
        # Each elevator selects a target floor
        self.action_space = spaces.MultiDiscrete([n_floors] * n_elevators)

        # Observation: Dict with per-elevator feature vectors
        # Features per elevator: ~40 dimensions
        # - Own state: floor(1), direction(3 one-hot), passenger_count(1), destination_floors(10 binary)
        # - Requests: waiting_per_floor(10), request_directions(10 binary UP, 10 binary DOWN)
        # - Other elevators: positions(n_elevators-1), directions((n_elevators-1)*3)
        # - Traffic: pattern(4 one-hot)
        obs_dim = 15 + 20 + (n_elevators - 1) * 4 + 4
        self.observation_space = spaces.Dict({
            f"elevator_{i}": spaces.Box(
                low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
            )
            for i in range(n_elevators)
        })

        # State
        self.elevators: List[Elevator] = []
        self.waiting_passengers: List[Passenger] = []
        self.delivered_passengers: List[Passenger] = []
        self.next_passenger_id = 0
        self.timestep = 0

        # Metrics
        self.total_movements = 0

        # Random number generator
        self._rng = np.random.default_rng()

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """Reset the environment."""
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        # Initialize elevators at ground floor
        self.elevators = [
            Elevator(id=i, floor=0.0, max_capacity=self.elevator_capacity)
            for i in range(self.n_elevators)
        ]

        # Clear passengers
        self.waiting_passengers = []
        self.delivered_passengers = []
        self.next_passenger_id = 0

        # Reset counters
        self.timestep = 0
        self.total_movements = 0

        return self._get_obs(), self._get_info()

    def step(
        self, actions: List[int]
    ) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        """
        Execute one timestep.

        Args:
            actions: List of target floors, one per elevator

        Returns:
            observation: Dict of observations per elevator
            reward: Total reward for this timestep
            terminated: Always False (episodes end after max_timesteps)
            truncated: True if reached max_timesteps
            info: Metrics and debug information
        """
        # Set target floors for elevators
        for i, (elevator, target_floor) in enumerate(zip(self.elevators, actions)):
            if 0 <= target_floor < self.n_floors:
                elevator.target_floor = target_floor

        # Handle boarding and exiting BEFORE movement
        # This allows elevators to pick up passengers at current floor before moving
        for elevator in self.elevators:
            floor = elevator.current_floor

            # Drop off passengers at current floor
            passengers_to_remove = [
                p for p in elevator.passengers if p.dest_floor == floor
            ]
            for passenger in passengers_to_remove:
                elevator.passengers.remove(passenger)
                self.delivered_passengers.append(passenger)

            # Pick up waiting passengers (up to capacity)
            if not elevator.is_full:
                available_space = elevator.max_capacity - elevator.passenger_count

                # Find waiting passengers at this floor
                # Pick up passengers going in elevator's direction or if elevator is idle
                candidates = [
                    p for p in self.waiting_passengers
                    if p.origin_floor == floor and (
                        elevator.target_floor is None or
                        elevator.direction == Direction.IDLE or
                        p.direction == elevator.direction or
                        (elevator.target_floor == floor)  # At target, pick up anyone
                    )
                ]

                # Board up to available space
                to_board = candidates[:available_space]
                for passenger in to_board:
                    passenger.pickup_time = self.timestep
                    elevator.passengers.append(passenger)
                    self.waiting_passengers.remove(passenger)

        # Move elevators toward targets
        for elevator in self.elevators:
            if elevator.target_floor is not None:
                current = elevator.floor
                target = elevator.target_floor

                if abs(target - current) > 0.01:  # Not at target
                    direction = 1 if target > current else -1
                    movement = min(self.move_speed, abs(target - current))
                    elevator.floor += direction * movement
                    self.total_movements += movement
                    elevator.is_moving = True
                else:
                    elevator.floor = target  # Snap to target
                    elevator.is_moving = False

        # Spawn new passengers (Poisson process)
        self._spawn_passengers()

        # Calculate reward
        reward = self._calculate_reward()

        # Update timestep
        self.timestep += 1

        # Episode ends after max_timesteps
        terminated = False
        truncated = self.timestep >= self.max_timesteps

        return self._get_obs(), reward, terminated, truncated, self._get_info()

    def _spawn_passengers(self):
        """Spawn passengers according to traffic pattern."""
        # Adjust spawn rate based on traffic pattern
        rate_multipliers = {
            TrafficPattern.MORNING_RUSH: 1.5,
            TrafficPattern.LUNCH_TIME: 1.0,
            TrafficPattern.EVENING_RUSH: 1.5,
            TrafficPattern.QUIET: 0.5,
        }
        rate = self.passenger_spawn_rate * rate_multipliers[self.traffic_pattern]

        # Poisson arrivals
        n_arrivals = self._rng.poisson(rate)

        for _ in range(n_arrivals):
            origin, dest = self._generate_trip()
            if origin != dest:  # Valid trip
                passenger = Passenger(
                    id=self.next_passenger_id,
                    origin_floor=origin,
                    dest_floor=dest,
                    spawn_time=self.timestep,
                )
                self.waiting_passengers.append(passenger)
                self.next_passenger_id += 1

    def _generate_trip(self) -> Tuple[int, int]:
        """Generate origin-destination pair based on traffic pattern."""
        if self.traffic_pattern == TrafficPattern.MORNING_RUSH:
            # 70% ground → upper, 30% inter-floor
            if self._rng.random() < 0.7:
                origin = 0
                dest = self._rng.integers(1, self.n_floors)
            else:
                origin = self._rng.integers(0, self.n_floors)
                dest = self._rng.integers(0, self.n_floors)

        elif self.traffic_pattern == TrafficPattern.EVENING_RUSH:
            # 70% upper → ground, 30% inter-floor
            if self._rng.random() < 0.7:
                origin = self._rng.integers(1, self.n_floors)
                dest = 0
            else:
                origin = self._rng.integers(0, self.n_floors)
                dest = self._rng.integers(0, self.n_floors)

        elif self.traffic_pattern == TrafficPattern.LUNCH_TIME:
            # 40% → ground, 40% from ground, 20% inter-floor
            r = self._rng.random()
            if r < 0.4:
                origin = self._rng.integers(1, self.n_floors)
                dest = 0
            elif r < 0.8:
                origin = 0
                dest = self._rng.integers(1, self.n_floors)
            else:
                origin = self._rng.integers(0, self.n_floors)
                dest = self._rng.integers(0, self.n_floors)

        else:  # QUIET
            origin = self._rng.integers(0, self.n_floors)
            dest = self._rng.integers(0, self.n_floors)

        return origin, dest

    def _calculate_reward(self) -> float:
        """Calculate reward for current timestep."""
        reward = 0.0

        # Primary: Negative waiting time
        for passenger in self.waiting_passengers:
            wait_time = self.timestep - passenger.spawn_time
            reward -= 1.0  # -1 per passenger per timestep

            # Extra penalty for long waits (starvation prevention)
            if wait_time > 60:
                reward -= 5.0

        # Bonus for delivering passengers
        # (Count passengers delivered this timestep)
        delivered_this_step = sum(
            1 for p in self.delivered_passengers
            if p.pickup_time == self.timestep - 1  # Delivered last step
        )
        reward += 10.0 * delivered_this_step

        # Energy cost (encourage efficiency)
        # Penalize movement
        reward -= 0.1 * self.total_movements

        return reward

    def _get_obs(self) -> Dict[str, np.ndarray]:
        """Get observations for all elevators."""
        observations = {}

        # Calculate global features
        waiting_per_floor = np.zeros(self.n_floors, dtype=np.float32)
        request_up = np.zeros(self.n_floors, dtype=np.float32)
        request_down = np.zeros(self.n_floors, dtype=np.float32)

        for passenger in self.waiting_passengers:
            waiting_per_floor[passenger.origin_floor] += 1
            if passenger.direction == Direction.UP:
                request_up[passenger.origin_floor] = 1
            elif passenger.direction == Direction.DOWN:
                request_down[passenger.origin_floor] = 1

        # Traffic pattern one-hot
        traffic_onehot = np.zeros(4, dtype=np.float32)
        traffic_onehot[self.traffic_pattern] = 1

        # Per-elevator observations
        for i, elevator in enumerate(self.elevators):
            features = []

            # Own state
            features.append(elevator.floor / self.n_floors)  # Normalized floor

            # Direction (one-hot)
            direction_onehot = np.zeros(3, dtype=np.float32)
            direction_onehot[elevator.direction] = 1
            features.extend(direction_onehot)

            # Passenger count (normalized)
            features.append(elevator.passenger_count / elevator.max_capacity)

            # Destination floors (binary vector)
            dest_floors = np.zeros(self.n_floors, dtype=np.float32)
            for dest in elevator.destination_floors:
                dest_floors[dest] = 1
            features.extend(dest_floors)

            # Waiting passengers per floor
            features.extend(waiting_per_floor)

            # Request directions
            features.extend(request_up)
            features.extend(request_down)

            # Other elevators' positions and directions
            for j, other in enumerate(self.elevators):
                if i != j:
                    features.append(other.floor / self.n_floors)
                    other_direction_onehot = np.zeros(3, dtype=np.float32)
                    other_direction_onehot[other.direction] = 1
                    features.extend(other_direction_onehot)

            # Traffic pattern
            features.extend(traffic_onehot)

            observations[f"elevator_{i}"] = np.array(features, dtype=np.float32)

        return observations

    def _get_info(self) -> Dict[str, Any]:
        """Get diagnostic information."""
        # Calculate metrics
        avg_wait_time = 0.0
        max_wait_time = 0
        if self.waiting_passengers:
            wait_times = [self.timestep - p.spawn_time for p in self.waiting_passengers]
            avg_wait_time = np.mean(wait_times)
            max_wait_time = max(wait_times)

        total_spawned = self.next_passenger_id
        total_delivered = len(self.delivered_passengers)

        elevator_utilization = np.mean([
            e.passenger_count / e.max_capacity for e in self.elevators
        ])

        return {
            "timestep": self.timestep,
            "waiting_passengers": len(self.waiting_passengers),
            "delivered_passengers": total_delivered,
            "total_spawned": total_spawned,
            "avg_wait_time": avg_wait_time,
            "max_wait_time": max_wait_time,
            "elevator_utilization": elevator_utilization,
            "total_movements": self.total_movements,
        }

    def render(self) -> Optional[str]:
        """Render the environment (ASCII visualization)."""
        if self.render_mode is None:
            return None

        lines = []
        lines.append(f"\n=== Elevator Dispatch (t={self.timestep}) ===")
        lines.append(f"Traffic: {self.traffic_pattern.name}")
        lines.append("")

        # Building visualization
        for floor in range(self.n_floors - 1, -1, -1):
            line = f"Floor {floor:2d} │ "

            # Show elevators at this floor
            for elevator in self.elevators:
                if elevator.current_floor == floor:
                    symbol = "█" if elevator.passenger_count > 0 else "▒"
                    line += f"[{elevator.id}:{elevator.passenger_count}] "
                else:
                    line += "    "

            # Show waiting passengers
            waiting_here = sum(
                1 for p in self.waiting_passengers if p.origin_floor == floor
            )
            if waiting_here > 0:
                line += f" ← {waiting_here} waiting"

            lines.append(line)

        lines.append("─" * 50)

        # Metrics
        info = self._get_info()
        lines.append(f"Waiting: {info['waiting_passengers']}, "
                    f"Delivered: {info['delivered_passengers']}/{info['total_spawned']}, "
                    f"Avg Wait: {info['avg_wait_time']:.1f}s")
        lines.append("")

        output = "\n".join(lines)

        if self.render_mode == "human":
            print(output)

        return output


# Alias for convenience
ElevatorEnv = ElevatorDispatch
