"""
Train Q-Learning on GridWorld

Demonstrates tabular Q-learning on a simple grid environment.
This example accompanies the Q-Learning Basics chapter.

Run with: python -m rlbook.examples.train_gridworld
"""

import argparse
from rlbook.envs import GridWorld
from rlbook.agents import QLearningAgent
from rlbook.utils import plot_training_curve, plot_value_function, plot_policy


def train(
    grid_size: int = 4,
    n_episodes: int = 500,
    learning_rate: float = 0.1,
    gamma: float = 0.99,
    epsilon: float = 1.0,
    epsilon_decay: float = 0.995,
    render_final: bool = True,
    plot_results: bool = True,
):
    """
    Train Q-Learning agent on GridWorld.

    Args:
        grid_size: Size of the grid (grid_size x grid_size)
        n_episodes: Number of training episodes
        learning_rate: Q-learning step size
        gamma: Discount factor
        epsilon: Initial exploration rate
        epsilon_decay: Epsilon decay per episode
        render_final: Whether to render final policy
        plot_results: Whether to show training plots
    """
    # Create environment and agent
    env = GridWorld(size=grid_size, render_mode="ansi")
    agent = QLearningAgent(
        n_states=env.observation_space.n,
        n_actions=env.action_space.n,
        learning_rate=learning_rate,
        gamma=gamma,
        epsilon=epsilon,
        epsilon_decay=epsilon_decay,
    )

    # Training loop
    episode_rewards = []

    for episode in range(n_episodes):
        state, _ = env.reset()
        episode_reward = 0
        done = False

        while not done:
            action = agent.select_action(state)
            next_state, reward, done, _, _ = env.step(action)
            agent.update(state, action, reward, next_state, done)
            state = next_state
            episode_reward += reward

        episode_rewards.append(episode_reward)
        agent.decay_epsilon()

        # Progress logging
        if (episode + 1) % 100 == 0:
            avg_reward = sum(episode_rewards[-100:]) / 100
            print(f"Episode {episode + 1}/{n_episodes}, Avg Reward: {avg_reward:.2f}, Epsilon: {agent.epsilon:.3f}")

    # Results
    print("\n=== Training Complete ===")
    print(f"Final Avg Reward (last 100): {sum(episode_rewards[-100:]) / 100:.2f}")

    if render_final:
        print("\n=== Final Policy ===")
        policy = agent.get_policy()
        action_symbols = {0: "↑", 1: "↓", 2: "←", 3: "→"}
        for i in range(grid_size):
            row = ""
            for j in range(grid_size):
                state = i * grid_size + j
                if i == grid_size - 1 and j == grid_size - 1:
                    row += "G "
                else:
                    row += action_symbols[policy[state]] + " "
            print(row)

    if plot_results:
        # Training curve
        fig1 = plot_training_curve(
            episode_rewards,
            title=f"Q-Learning on {grid_size}x{grid_size} GridWorld",
        )

        # Value function
        fig2 = plot_value_function(
            agent.get_value_function(),
            (grid_size, grid_size),
            title="Learned Value Function",
        )

        # Policy
        fig3 = plot_policy(
            agent.get_policy(),
            (grid_size, grid_size),
            title="Learned Policy",
        )

        import matplotlib.pyplot as plt
        plt.show()

    return agent, episode_rewards


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Q-Learning on GridWorld")
    parser.add_argument("--grid-size", type=int, default=4, help="Grid size")
    parser.add_argument("--episodes", type=int, default=500, help="Number of episodes")
    parser.add_argument("--lr", type=float, default=0.1, help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--no-plot", action="store_true", help="Disable plotting")

    args = parser.parse_args()

    train(
        grid_size=args.grid_size,
        n_episodes=args.episodes,
        learning_rate=args.lr,
        gamma=args.gamma,
        plot_results=not args.no_plot,
    )
