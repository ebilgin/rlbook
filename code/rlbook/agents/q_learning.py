"""
Q-Learning and SARSA Agents

Tabular temporal difference learning algorithms for discrete state/action spaces.

This code is referenced in:
    - Chapter: Introduction to TD Learning (1010-intro-to-td)
    - Chapter: Q-Learning Basics (1020-q-learning-basics)

Example:
    >>> from rlbook.agents import QLearningAgent
    >>> agent = QLearningAgent(n_states=16, n_actions=4)
    >>> action = agent.select_action(state)
    >>> agent.update(state, action, reward, next_state, done)
"""

from typing import Optional
import numpy as np


class QLearningAgent:
    """
    Tabular Q-Learning agent.

    Q-learning is an off-policy TD control algorithm that learns the optimal
    action-value function Q* directly, regardless of the policy being followed.

    The update rule is:
        Q(s, a) <- Q(s, a) + α * [r + γ * max_a' Q(s', a') - Q(s, a)]

    Key insight: We update toward the *best* action in the next state, even if
    we didn't take that action. This is what makes Q-learning "off-policy."

    Args:
        n_states: Number of discrete states
        n_actions: Number of discrete actions
        learning_rate: Step size for Q updates (α)
        gamma: Discount factor for future rewards (γ)
        epsilon: Exploration rate for ε-greedy policy
        epsilon_decay: Multiply epsilon by this each episode
        epsilon_min: Minimum exploration rate

    Attributes:
        q_table: Q(s, a) values, shape (n_states, n_actions)
    """

    def __init__(
        self,
        n_states: int,
        n_actions: int,
        learning_rate: float = 0.1,
        gamma: float = 0.99,
        epsilon: float = 1.0,
        epsilon_decay: float = 0.995,
        epsilon_min: float = 0.01,
    ):
        self.n_states = n_states
        self.n_actions = n_actions
        self.lr = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        # Initialize Q-table with zeros (optimistic init can help exploration)
        self.q_table = np.zeros((n_states, n_actions))

        self._rng = np.random.default_rng()

    def select_action(self, state: int, training: bool = True) -> int:
        """
        Select action using ε-greedy policy.

        With probability ε, choose random action (explore).
        With probability 1-ε, choose best action (exploit).

        Args:
            state: Current state index
            training: If False, always exploit (for evaluation)

        Returns:
            Selected action index
        """
        if training and self._rng.random() < self.epsilon:
            return self._rng.integers(self.n_actions)
        else:
            # Break ties randomly
            q_values = self.q_table[state]
            max_q = np.max(q_values)
            best_actions = np.where(q_values == max_q)[0]
            return self._rng.choice(best_actions)

    def update(
        self,
        state: int,
        action: int,
        reward: float,
        next_state: int,
        done: bool,
    ) -> float:
        """
        Update Q-value using the Q-learning update rule.

        Q(s, a) <- Q(s, a) + α * [r + γ * max_a' Q(s', a') - Q(s, a)]

        Args:
            state: State where action was taken
            action: Action that was taken
            reward: Reward received
            next_state: Resulting state
            done: Whether episode ended

        Returns:
            TD error (useful for debugging)
        """
        # Current Q value
        current_q = self.q_table[state, action]

        # Target: r + γ * max Q(s', a')
        # If done, there's no next state value
        if done:
            target = reward
        else:
            target = reward + self.gamma * np.max(self.q_table[next_state])

        # TD error
        td_error = target - current_q

        # Update
        self.q_table[state, action] = current_q + self.lr * td_error

        return td_error

    def decay_epsilon(self) -> None:
        """Decay exploration rate (call at end of each episode)."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def get_policy(self) -> np.ndarray:
        """
        Get the greedy policy from current Q-values.

        Returns:
            Array of shape (n_states,) with best action for each state
        """
        return np.argmax(self.q_table, axis=1)

    def get_value_function(self) -> np.ndarray:
        """
        Get state value function V(s) = max_a Q(s, a).

        Returns:
            Array of shape (n_states,) with value of each state
        """
        return np.max(self.q_table, axis=1)


class SarsaAgent(QLearningAgent):
    """
    SARSA (State-Action-Reward-State-Action) agent.

    SARSA is an on-policy TD control algorithm. Unlike Q-learning, it updates
    toward the action actually taken in the next state, not the best action.

    The update rule is:
        Q(s, a) <- Q(s, a) + α * [r + γ * Q(s', a') - Q(s, a)]

    Key difference from Q-learning: We use Q(s', a') where a' is the action
    we actually selected, not max_a' Q(s', a').

    This makes SARSA more conservative - it learns the value of the policy
    it's actually following (including exploration mistakes).
    """

    def update(
        self,
        state: int,
        action: int,
        reward: float,
        next_state: int,
        next_action: Optional[int],
        done: bool,
    ) -> float:
        """
        Update Q-value using the SARSA update rule.

        Q(s, a) <- Q(s, a) + α * [r + γ * Q(s', a') - Q(s, a)]

        Args:
            state: State where action was taken
            action: Action that was taken
            reward: Reward received
            next_state: Resulting state
            next_action: Action selected in next state (None if done)
            done: Whether episode ended

        Returns:
            TD error
        """
        current_q = self.q_table[state, action]

        if done:
            target = reward
        else:
            # Use Q(s', a') not max Q(s', a')
            target = reward + self.gamma * self.q_table[next_state, next_action]

        td_error = target - current_q
        self.q_table[state, action] = current_q + self.lr * td_error

        return td_error


class ExpectedSarsaAgent(QLearningAgent):
    """
    Expected SARSA agent.

    Instead of using the actual next action (SARSA) or max action (Q-learning),
    Expected SARSA uses the expected value under the current policy.

    The update rule is:
        Q(s, a) <- Q(s, a) + α * [r + γ * E[Q(s', a')] - Q(s, a)]

    where E[Q(s', a')] is computed over the ε-greedy action distribution.

    This reduces variance compared to SARSA while maintaining on-policy learning.
    """

    def update(
        self,
        state: int,
        action: int,
        reward: float,
        next_state: int,
        done: bool,
    ) -> float:
        """
        Update Q-value using Expected SARSA update rule.

        Args:
            state: State where action was taken
            action: Action that was taken
            reward: Reward received
            next_state: Resulting state
            done: Whether episode ended

        Returns:
            TD error
        """
        current_q = self.q_table[state, action]

        if done:
            target = reward
        else:
            # Compute expected Q under ε-greedy policy
            next_q_values = self.q_table[next_state]
            best_action = np.argmax(next_q_values)

            # Expected value: (1-ε) * max_Q + ε * mean_Q
            expected_q = (
                (1 - self.epsilon) * next_q_values[best_action]
                + self.epsilon * np.mean(next_q_values)
            )
            target = reward + self.gamma * expected_q

        td_error = target - current_q
        self.q_table[state, action] = current_q + self.lr * td_error

        return td_error
