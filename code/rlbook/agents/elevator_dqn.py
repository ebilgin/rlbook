"""
Multi-Agent DQN for Elevator Dispatch

Independent Q-learning with shared replay buffer. Each elevator has its own
Q-network but learns from experiences of all elevators.

This code is referenced in:
    - Application: Elevator Dispatch (0030-elevator-dispatch)

Example:
    >>> from rlbook.agents import ElevatorDQN
    >>> agent = ElevatorDQN(n_floors=10, n_elevators=3, observation_dim=50)
    >>> actions = agent.select_actions(observations)
    >>> agent.store_transitions(observations, actions, reward, next_observations, done)
    >>> loss = agent.train_step()
"""

from typing import Tuple, Optional, List, Dict
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from rlbook.agents.dqn import QNetwork, ReplayBuffer


class ElevatorDQN:
    """
    Multi-agent DQN for elevator dispatch.

    Uses independent Q-learning: each elevator has its own Q-network that
    learns from its own observations. Elevators share a replay buffer to
    learn from each other's experiences, enabling implicit coordination.

    Args:
        n_floors: Number of floors (action space size per elevator)
        n_elevators: Number of elevators
        observation_dim: Dimension of observation vector per elevator
        hidden_dims: Tuple of hidden layer sizes
        learning_rate: Optimizer learning rate
        gamma: Discount factor
        epsilon: Initial exploration rate
        epsilon_decay: Multiply epsilon by this each episode
        epsilon_min: Minimum exploration rate
        buffer_size: Shared replay buffer capacity
        batch_size: Training batch size
        target_update_freq: Steps between target network updates
        device: 'cuda' or 'cpu'
    """

    def __init__(
        self,
        n_floors: int,
        n_elevators: int,
        observation_dim: int,
        hidden_dims: Tuple[int, ...] = (128, 128),
        learning_rate: float = 1e-3,
        gamma: float = 0.99,
        epsilon: float = 1.0,
        epsilon_decay: float = 0.995,
        epsilon_min: float = 0.01,
        buffer_size: int = 50000,
        batch_size: int = 64,
        target_update_freq: int = 100,
        device: Optional[str] = None,
    ):
        self.n_floors = n_floors
        self.n_elevators = n_elevators
        self.observation_dim = observation_dim
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq

        # Device
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        # Create Q-networks for each elevator (independent learners)
        self.q_networks = []
        self.target_networks = []
        self.optimizers = []

        for i in range(n_elevators):
            # Q-network
            q_net = QNetwork(observation_dim, n_floors, hidden_dims).to(self.device)
            target_net = QNetwork(observation_dim, n_floors, hidden_dims).to(self.device)
            target_net.load_state_dict(q_net.state_dict())
            target_net.eval()

            self.q_networks.append(q_net)
            self.target_networks.append(target_net)

            # Optimizer for this elevator's network
            optimizer = optim.Adam(q_net.parameters(), lr=learning_rate)
            self.optimizers.append(optimizer)

        # Shared replay buffer (all elevators learn from all experiences)
        self.replay_buffer = ReplayBuffer(buffer_size)

        # Training step counter
        self.steps = 0

    def select_actions(
        self, observations: Dict[str, np.ndarray], training: bool = True
    ) -> List[int]:
        """
        Select actions for all elevators using ε-greedy policy.

        Args:
            observations: Dict mapping "elevator_i" to observation vector
            training: If False, always exploit

        Returns:
            List of actions, one per elevator
        """
        actions = []

        for i in range(self.n_elevators):
            obs = observations[f"elevator_{i}"]

            # ε-greedy exploration
            if training and np.random.random() < self.epsilon:
                action = np.random.randint(0, self.n_floors)
            else:
                # Exploit: select best action according to Q-network
                with torch.no_grad():
                    obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
                    q_values = self.q_networks[i](obs_tensor)
                    action = q_values.argmax(dim=1).item()

            actions.append(action)

        return actions

    def store_transitions(
        self,
        observations: Dict[str, np.ndarray],
        actions: List[int],
        reward: float,
        next_observations: Dict[str, np.ndarray],
        done: bool,
    ) -> None:
        """
        Store transitions for all elevators in shared replay buffer.

        Each elevator's experience is stored separately, but all share the buffer.
        This allows elevators to learn from each other's experiences.

        Args:
            observations: Current observations (dict)
            actions: Actions taken by each elevator
            reward: Shared reward (same for all elevators)
            next_observations: Next observations (dict)
            done: Episode termination flag
        """
        # Store each elevator's transition
        for i in range(self.n_elevators):
            obs = observations[f"elevator_{i}"]
            action = actions[i]
            next_obs = next_observations[f"elevator_{i}"]

            self.replay_buffer.push(obs, action, reward, next_obs, done)

    def train_step(self) -> Optional[float]:
        """
        Perform one training step for all elevators.

        Samples a batch from the shared replay buffer and updates all Q-networks.

        Returns:
            Average loss across all elevators, or None if buffer too small
        """
        if len(self.replay_buffer) < self.batch_size:
            return None

        total_loss = 0.0

        # Train each elevator's network
        for i in range(self.n_elevators):
            # Sample batch from shared buffer
            states, actions, rewards, next_states, dones = self.replay_buffer.sample(
                self.batch_size
            )

            # Convert to tensors
            states = torch.FloatTensor(states).to(self.device)
            actions = torch.LongTensor(actions).to(self.device)
            rewards = torch.FloatTensor(rewards).to(self.device)
            next_states = torch.FloatTensor(next_states).to(self.device)
            dones = torch.FloatTensor(dones).to(self.device)

            # Compute current Q values for this elevator
            current_q = (
                self.q_networks[i](states)
                .gather(1, actions.unsqueeze(1))
                .squeeze(1)
            )

            # Compute target Q values
            with torch.no_grad():
                next_q = self.target_networks[i](next_states).max(dim=1)[0]
                target_q = rewards + self.gamma * next_q * (1 - dones)

            # Compute loss and update
            loss = F.mse_loss(current_q, target_q)

            self.optimizers[i].zero_grad()
            loss.backward()
            self.optimizers[i].step()

            total_loss += loss.item()

        # Update target networks periodically
        self.steps += 1
        if self.steps % self.target_update_freq == 0:
            self.update_target_networks()

        return total_loss / self.n_elevators

    def update_target_networks(self) -> None:
        """Copy Q-network weights to target networks for all elevators."""
        for i in range(self.n_elevators):
            self.target_networks[i].load_state_dict(self.q_networks[i].state_dict())

    def decay_epsilon(self) -> None:
        """Decay exploration rate (call once per episode)."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save(self, filepath: str) -> None:
        """Save all Q-networks to a file."""
        checkpoint = {
            "n_floors": self.n_floors,
            "n_elevators": self.n_elevators,
            "observation_dim": self.observation_dim,
            "epsilon": self.epsilon,
            "steps": self.steps,
            "q_networks": [net.state_dict() for net in self.q_networks],
            "target_networks": [net.state_dict() for net in self.target_networks],
            "optimizers": [opt.state_dict() for opt in self.optimizers],
        }
        torch.save(checkpoint, filepath)

    def load(self, filepath: str) -> None:
        """Load Q-networks from a file."""
        checkpoint = torch.load(filepath, map_location=self.device)

        self.epsilon = checkpoint["epsilon"]
        self.steps = checkpoint["steps"]

        for i in range(self.n_elevators):
            self.q_networks[i].load_state_dict(checkpoint["q_networks"][i])
            self.target_networks[i].load_state_dict(checkpoint["target_networks"][i])
            self.optimizers[i].load_state_dict(checkpoint["optimizers"][i])


# Baseline policies for comparison


def random_policy(env) -> List[int]:
    """Random action selection for all elevators."""
    return [np.random.randint(0, env.n_floors) for _ in range(env.n_elevators)]


def nearest_car_policy(env) -> List[int]:
    """Assign each elevator to nearest waiting passenger (FCFS)."""
    actions = []

    for elevator in env.elevators:
        if not env.waiting_passengers:
            # No requests, stay put
            actions.append(elevator.current_floor)
            continue

        # Find nearest waiting passenger
        min_distance = float("inf")
        nearest_floor = elevator.current_floor

        for passenger in env.waiting_passengers:
            distance = abs(elevator.current_floor - passenger.origin_floor)
            if distance < min_distance:
                min_distance = distance
                nearest_floor = passenger.origin_floor

        actions.append(nearest_floor)

    return actions


def scan_policy(env) -> List[int]:
    """SCAN elevator algorithm (continue in current direction until end)."""
    from rlbook.envs.elevator import Direction

    actions = []

    for elevator in env.elevators:
        current_floor = elevator.current_floor

        # If has passengers, prioritize their destinations
        if elevator.destination_floors:
            if elevator.direction == Direction.UP:
                # Go to highest destination
                target = max(elevator.destination_floors)
            elif elevator.direction == Direction.DOWN:
                # Go to lowest destination
                target = min(elevator.destination_floors)
            else:  # IDLE
                # Go to nearest destination
                target = min(
                    elevator.destination_floors,
                    key=lambda f: abs(f - current_floor),
                )
        else:
            # No passengers, go to nearest request
            if env.waiting_passengers:
                nearest_floor = min(
                    (p.origin_floor for p in env.waiting_passengers),
                    key=lambda f: abs(f - current_floor),
                )
                target = nearest_floor
            else:
                # No requests, stay put
                target = current_floor

        actions.append(target)

    return actions
