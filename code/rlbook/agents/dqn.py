"""
Deep Q-Network (DQN) Agent

Neural network-based Q-learning with experience replay and target networks.

This code is referenced in:
    - Chapter: Deep Q-Networks (1040-deep-q-networks)

Example:
    >>> from rlbook.agents import DQNAgent
    >>> agent = DQNAgent(state_dim=4, n_actions=2)
    >>> action = agent.select_action(state)
    >>> agent.store_transition(state, action, reward, next_state, done)
    >>> loss = agent.train_step()
"""

from typing import Tuple, Optional
from collections import deque
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class QNetwork(nn.Module):
    """
    Simple MLP Q-network.

    Maps states to Q-values for each action.
    Architecture: state -> hidden layers -> Q(s, a) for all a
    """

    def __init__(
        self,
        state_dim: int,
        n_actions: int,
        hidden_dims: Tuple[int, ...] = (64, 64),
    ):
        super().__init__()

        layers = []
        prev_dim = state_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, n_actions))

        self.network = nn.Sequential(*layers)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Compute Q-values for all actions given state."""
        return self.network(state)


class ReplayBuffer:
    """
    Experience replay buffer for DQN.

    Stores transitions and samples random minibatches for training.
    Breaking temporal correlations is crucial for stable learning.
    """

    def __init__(self, capacity: int = 10000):
        self.buffer = deque(maxlen=capacity)

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """Store a transition."""
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> Tuple:
        """Sample a random batch of transitions."""
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        return (
            np.array(states),
            np.array(actions),
            np.array(rewards, dtype=np.float32),
            np.array(next_states),
            np.array(dones, dtype=np.float32),
        )

    def __len__(self) -> int:
        return len(self.buffer)


class DQNAgent:
    """
    Deep Q-Network agent with experience replay and target network.

    Key components:
        1. Q-Network: Neural network approximating Q(s, a)
        2. Target Network: Slowly-updated copy for stable targets
        3. Experience Replay: Buffer of past transitions for training
        4. ε-greedy exploration: Balance exploration and exploitation

    The DQN loss is:
        L = E[(r + γ * max_a' Q_target(s', a') - Q(s, a))²]

    Args:
        state_dim: Dimension of state vector
        n_actions: Number of discrete actions
        hidden_dims: Tuple of hidden layer sizes
        learning_rate: Optimizer learning rate
        gamma: Discount factor
        epsilon: Initial exploration rate
        epsilon_decay: Multiply epsilon by this each step
        epsilon_min: Minimum exploration rate
        buffer_size: Replay buffer capacity
        batch_size: Training batch size
        target_update_freq: Steps between target network updates
        device: 'cuda' or 'cpu'
    """

    def __init__(
        self,
        state_dim: int,
        n_actions: int,
        hidden_dims: Tuple[int, ...] = (64, 64),
        learning_rate: float = 1e-3,
        gamma: float = 0.99,
        epsilon: float = 1.0,
        epsilon_decay: float = 0.995,
        epsilon_min: float = 0.01,
        buffer_size: int = 10000,
        batch_size: int = 64,
        target_update_freq: int = 100,
        device: Optional[str] = None,
    ):
        self.state_dim = state_dim
        self.n_actions = n_actions
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

        # Networks
        self.q_network = QNetwork(state_dim, n_actions, hidden_dims).to(self.device)
        self.target_network = QNetwork(state_dim, n_actions, hidden_dims).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()  # Target network is never trained directly

        # Optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)

        # Replay buffer
        self.replay_buffer = ReplayBuffer(buffer_size)

        # Training step counter
        self.steps = 0

    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        """
        Select action using ε-greedy policy.

        Args:
            state: Current state observation
            training: If False, always exploit

        Returns:
            Selected action
        """
        if training and random.random() < self.epsilon:
            return random.randint(0, self.n_actions - 1)

        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.q_network(state_tensor)
            return q_values.argmax(dim=1).item()

    def store_transition(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """Store a transition in the replay buffer."""
        self.replay_buffer.push(state, action, reward, next_state, done)

    def train_step(self) -> Optional[float]:
        """
        Perform one training step.

        Returns:
            Loss value if training occurred, None otherwise
        """
        if len(self.replay_buffer) < self.batch_size:
            return None

        # Sample batch
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(
            self.batch_size
        )

        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        # Compute current Q values
        current_q = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # Compute target Q values
        with torch.no_grad():
            next_q = self.target_network(next_states).max(dim=1)[0]
            target_q = rewards + self.gamma * next_q * (1 - dones)

        # Compute loss and update
        loss = F.mse_loss(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update target network periodically
        self.steps += 1
        if self.steps % self.target_update_freq == 0:
            self.update_target_network()

        return loss.item()

    def update_target_network(self) -> None:
        """Copy Q-network weights to target network."""
        self.target_network.load_state_dict(self.q_network.state_dict())

    def decay_epsilon(self) -> None:
        """Decay exploration rate."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save(self, path: str) -> None:
        """Save model weights."""
        torch.save(
            {
                "q_network": self.q_network.state_dict(),
                "target_network": self.target_network.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "epsilon": self.epsilon,
                "steps": self.steps,
            },
            path,
        )

    def load(self, path: str) -> None:
        """Load model weights."""
        checkpoint = torch.load(path, map_location=self.device)
        self.q_network.load_state_dict(checkpoint["q_network"])
        self.target_network.load_state_dict(checkpoint["target_network"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.epsilon = checkpoint["epsilon"]
        self.steps = checkpoint["steps"]


class DoubleDQNAgent(DQNAgent):
    """
    Double DQN agent.

    Addresses overestimation bias in DQN by decoupling action selection
    from action evaluation:
        - Q-network selects the best action: a* = argmax_a Q(s', a)
        - Target network evaluates it: Q_target(s', a*)

    This reduces overestimation and often improves performance.
    """

    def train_step(self) -> Optional[float]:
        """Training step with Double DQN target computation."""
        if len(self.replay_buffer) < self.batch_size:
            return None

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(
            self.batch_size
        )

        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        # Current Q values
        current_q = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # Double DQN: select action with Q-network, evaluate with target
        with torch.no_grad():
            # Q-network selects best action
            next_actions = self.q_network(next_states).argmax(dim=1)
            # Target network evaluates that action
            next_q = self.target_network(next_states).gather(
                1, next_actions.unsqueeze(1)
            ).squeeze(1)
            target_q = rewards + self.gamma * next_q * (1 - dones)

        loss = F.mse_loss(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.steps += 1
        if self.steps % self.target_update_freq == 0:
            self.update_target_network()

        return loss.item()
