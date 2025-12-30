"""
Plotting utilities for RL training visualization.

Provides simple functions for common RL plots.
"""

from typing import List, Optional, Tuple
import numpy as np
import matplotlib.pyplot as plt


def plot_training_curve(
    rewards: List[float],
    window: int = 100,
    title: str = "Training Progress",
    xlabel: str = "Episode",
    ylabel: str = "Reward",
    figsize: Tuple[int, int] = (10, 5),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot training rewards with smoothed moving average.

    Args:
        rewards: List of episode rewards
        window: Window size for moving average
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
        figsize: Figure size
        save_path: If provided, save figure to this path

    Returns:
        matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Raw rewards (faded)
    ax.plot(rewards, alpha=0.3, color="blue", label="Raw")

    # Moving average
    if len(rewards) >= window:
        moving_avg = np.convolve(rewards, np.ones(window) / window, mode="valid")
        ax.plot(
            range(window - 1, len(rewards)),
            moving_avg,
            color="blue",
            linewidth=2,
            label=f"Moving Avg ({window})",
        )

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_value_function(
    values: np.ndarray,
    grid_size: Tuple[int, int],
    title: str = "Value Function",
    cmap: str = "RdYlGn",
    figsize: Tuple[int, int] = (8, 6),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot value function as a heatmap for grid environments.

    Args:
        values: 1D array of state values
        grid_size: (rows, cols) of the grid
        title: Plot title
        cmap: Colormap name
        figsize: Figure size
        save_path: If provided, save figure to this path

    Returns:
        matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Reshape to grid
    value_grid = values.reshape(grid_size)

    # Create heatmap
    im = ax.imshow(value_grid, cmap=cmap)
    plt.colorbar(im, ax=ax, label="V(s)")

    # Add value annotations
    for i in range(grid_size[0]):
        for j in range(grid_size[1]):
            ax.text(
                j, i, f"{value_grid[i, j]:.1f}",
                ha="center", va="center",
                color="black", fontsize=10,
            )

    ax.set_title(title)
    ax.set_xlabel("Column")
    ax.set_ylabel("Row")

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_policy(
    policy: np.ndarray,
    grid_size: Tuple[int, int],
    title: str = "Policy",
    figsize: Tuple[int, int] = (8, 6),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot policy as arrows for grid environments.

    Args:
        policy: 1D array of action indices (0=UP, 1=DOWN, 2=LEFT, 3=RIGHT)
        grid_size: (rows, cols) of the grid
        title: Plot title
        figsize: Figure size
        save_path: If provided, save figure to this path

    Returns:
        matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Arrow directions: (dx, dy) for plotting
    arrows = {
        0: (0, 0.3),   # UP
        1: (0, -0.3),  # DOWN
        2: (-0.3, 0),  # LEFT
        3: (0.3, 0),   # RIGHT
    }

    policy_grid = policy.reshape(grid_size)

    for i in range(grid_size[0]):
        for j in range(grid_size[1]):
            action = policy_grid[i, j]
            dx, dy = arrows[action]
            ax.arrow(
                j, i, dx, dy,
                head_width=0.15, head_length=0.1,
                fc="blue", ec="blue",
            )

    ax.set_xlim(-0.5, grid_size[1] - 0.5)
    ax.set_ylim(grid_size[0] - 0.5, -0.5)  # Invert y-axis
    ax.set_aspect("equal")
    ax.set_title(title)
    ax.grid(True)

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_q_values(
    q_table: np.ndarray,
    grid_size: Tuple[int, int],
    title: str = "Q-Values",
    figsize: Tuple[int, int] = (12, 8),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot Q-values for all actions in each state.

    Creates a subplot for each action showing its Q-values across states.

    Args:
        q_table: 2D array of shape (n_states, n_actions)
        grid_size: (rows, cols) of the grid
        title: Plot title
        figsize: Figure size
        save_path: If provided, save figure to this path

    Returns:
        matplotlib Figure object
    """
    n_actions = q_table.shape[1]
    action_names = ["UP", "DOWN", "LEFT", "RIGHT"]

    fig, axes = plt.subplots(2, 2, figsize=figsize)
    axes = axes.flatten()

    for a in range(min(n_actions, 4)):
        ax = axes[a]
        q_grid = q_table[:, a].reshape(grid_size)

        im = ax.imshow(q_grid, cmap="RdYlGn")
        plt.colorbar(im, ax=ax)

        for i in range(grid_size[0]):
            for j in range(grid_size[1]):
                ax.text(
                    j, i, f"{q_grid[i, j]:.1f}",
                    ha="center", va="center",
                    color="black", fontsize=8,
                )

        ax.set_title(f"Q(s, {action_names[a]})")

    fig.suptitle(title)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig
