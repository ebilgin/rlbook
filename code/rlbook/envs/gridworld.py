"""
GridWorld Environment

The canonical teaching environment for reinforcement learning. A simple grid
where an agent navigates from a start position to a goal while avoiding obstacles.

This environment is referenced in:
    - Chapter: Introduction to TD Learning (1010-intro-to-td)
    - Chapter: Q-Learning Basics (1020-q-learning-basics)
    - Environment: GridWorld Playground (0010-gridworld)

Example:
    >>> env = GridWorld(size=4)
    >>> state, _ = env.reset()
    >>> action = env.action_space.sample()
    >>> next_state, reward, done, truncated, info = env.step(action)
"""

from typing import Optional, Tuple, List, Dict, Any
import numpy as np
import gymnasium as gym
from gymnasium import spaces


class GridWorld(gym.Env):
    """
    A simple grid world environment.

    The agent starts at (0, 0) and must reach the goal at (size-1, size-1).
    Actions move the agent in cardinal directions. Hitting a wall keeps the
    agent in place.

    Args:
        size: Grid dimensions (size x size)
        obstacles: List of (row, col) positions that are impassable
        stochastic: If True, actions have 10% chance of random direction
        step_reward: Reward for each step (default: -0.1 to encourage efficiency)
        goal_reward: Reward for reaching the goal (default: 10.0)
        wall_penalty: Additional penalty for hitting a wall (default: 0.0)

    Observation Space:
        Discrete(size * size) - flattened grid position

    Action Space:
        Discrete(4) - UP(0), DOWN(1), LEFT(2), RIGHT(3)
    """

    metadata = {"render_modes": ["human", "ansi", "rgb_array"]}

    # Action constants
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3

    def __init__(
        self,
        size: int = 4,
        obstacles: Optional[List[Tuple[int, int]]] = None,
        stochastic: bool = False,
        step_reward: float = -0.1,
        goal_reward: float = 10.0,
        wall_penalty: float = 0.0,
        render_mode: Optional[str] = None,
    ):
        super().__init__()

        self.size = size
        self.obstacles = set(obstacles) if obstacles else set()
        self.stochastic = stochastic
        self.step_reward = step_reward
        self.goal_reward = goal_reward
        self.wall_penalty = wall_penalty
        self.render_mode = render_mode

        # Spaces
        self.observation_space = spaces.Discrete(size * size)
        self.action_space = spaces.Discrete(4)

        # Positions
        self.start_pos = (0, 0)
        self.goal_pos = (size - 1, size - 1)
        self.agent_pos = self.start_pos

        # Action effects: (row_delta, col_delta)
        self._action_effects = {
            self.UP: (-1, 0),
            self.DOWN: (1, 0),
            self.LEFT: (0, -1),
            self.RIGHT: (0, 1),
        }

        # For stochastic transitions
        self._rng = np.random.default_rng()

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[int, Dict[str, Any]]:
        """Reset the environment to the start state."""
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        self.agent_pos = self.start_pos
        return self._get_obs(), self._get_info()

    def step(self, action: int) -> Tuple[int, float, bool, bool, Dict[str, Any]]:
        """
        Take a step in the environment.

        Args:
            action: Integer action (0=UP, 1=DOWN, 2=LEFT, 3=RIGHT)

        Returns:
            observation: New state (flattened grid position)
            reward: Reward for this transition
            terminated: True if goal reached
            truncated: Always False (no time limit in base env)
            info: Additional information
        """
        # Stochastic transitions: 10% chance of random action
        if self.stochastic and self._rng.random() < 0.1:
            action = self._rng.integers(4)

        # Calculate new position
        dr, dc = self._action_effects[action]
        new_row = self.agent_pos[0] + dr
        new_col = self.agent_pos[1] + dc
        new_pos = (new_row, new_col)

        # Check boundaries and obstacles
        hit_wall = False
        if not self._is_valid_position(new_pos):
            hit_wall = True
            new_pos = self.agent_pos  # Stay in place

        self.agent_pos = new_pos

        # Calculate reward
        if self.agent_pos == self.goal_pos:
            reward = self.goal_reward
            terminated = True
        else:
            reward = self.step_reward
            if hit_wall:
                reward += self.wall_penalty
            terminated = False

        return self._get_obs(), reward, terminated, False, self._get_info()

    def _is_valid_position(self, pos: Tuple[int, int]) -> bool:
        """Check if a position is within bounds and not an obstacle."""
        row, col = pos
        if row < 0 or row >= self.size or col < 0 or col >= self.size:
            return False
        if pos in self.obstacles:
            return False
        return True

    def _get_obs(self) -> int:
        """Convert (row, col) position to flattened index."""
        return self.agent_pos[0] * self.size + self.agent_pos[1]

    def _get_info(self) -> Dict[str, Any]:
        """Return additional state information."""
        return {
            "agent_pos": self.agent_pos,
            "goal_pos": self.goal_pos,
            "distance_to_goal": abs(self.agent_pos[0] - self.goal_pos[0])
            + abs(self.agent_pos[1] - self.goal_pos[1]),
        }

    def render(self) -> Optional[str]:
        """Render the environment."""
        if self.render_mode == "ansi":
            return self._render_ansi()
        elif self.render_mode == "human":
            print(self._render_ansi())
            return None
        return None

    def _render_ansi(self) -> str:
        """Create ASCII representation of the grid."""
        lines = []
        for row in range(self.size):
            line = ""
            for col in range(self.size):
                pos = (row, col)
                if pos == self.agent_pos:
                    line += "A "
                elif pos == self.goal_pos:
                    line += "G "
                elif pos in self.obstacles:
                    line += "# "
                else:
                    line += ". "
            lines.append(line)
        return "\n".join(lines)

    def get_transition_prob(
        self, state: int, action: int
    ) -> List[Tuple[float, int, float, bool]]:
        """
        Get transition probabilities for dynamic programming.

        Returns list of (probability, next_state, reward, done) tuples.
        Useful for value iteration and policy iteration.
        """
        # Convert state to position
        row, col = state // self.size, state % self.size
        original_pos = self.agent_pos
        self.agent_pos = (row, col)

        transitions = []

        if self.stochastic:
            # 90% intended action, 10% random
            for a in range(4):
                prob = 0.9 if a == action else 0.1 / 3
                self.agent_pos = (row, col)
                obs, reward, done, _, _ = self.step(a)
                transitions.append((prob, obs, reward, done))
                self.agent_pos = (row, col)
        else:
            obs, reward, done, _, _ = self.step(action)
            transitions.append((1.0, obs, reward, done))

        self.agent_pos = original_pos
        return transitions


# Variant environments for different learning scenarios
class CliffWalking(GridWorld):
    """
    Cliff Walking environment from Sutton & Barto.

    A 4x12 grid where the bottom row (except start and goal) is a cliff.
    Stepping on the cliff returns the agent to start with -100 reward.
    Optimal path hugs the cliff (risky), safe path goes around.
    """

    def __init__(self, **kwargs):
        # Create cliff obstacles (bottom row except corners)
        cliff = [(3, c) for c in range(1, 11)]
        super().__init__(size=4, obstacles=cliff, **kwargs)
        # Override to make it 4x12
        self.size = 4
        self._width = 12
        self.observation_space = spaces.Discrete(4 * 12)
        self.goal_pos = (3, 11)
        self.cliff = set(cliff)

    def step(self, action: int) -> Tuple[int, float, bool, bool, Dict[str, Any]]:
        """Override to handle cliff specially."""
        result = super().step(action)
        if self.agent_pos in self.cliff:
            self.agent_pos = self.start_pos
            return self._get_obs(), -100.0, False, False, self._get_info()
        return result
