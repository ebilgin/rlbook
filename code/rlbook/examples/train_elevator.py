"""
Training script for Elevator Dispatch with DQN.

This script trains a multi-agent DQN to control elevator dispatch in a building.
Results are saved for visualization in the interactive components.

Usage:
    python -m rlbook.examples.train_elevator

    # With custom parameters:
    python -m rlbook.examples.train_elevator --episodes 500 --n-elevators 3
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List
import numpy as np

from rlbook.envs import ElevatorDispatch
from rlbook.agents import ElevatorDQN
from rlbook.agents.elevator_dqn import random_policy, nearest_car_policy, scan_policy


def train_dqn(
    n_episodes: int = 1000,
    n_floors: int = 10,
    n_elevators: int = 3,
    traffic_pattern: str = "morning_rush",
    save_dir: str = "trained_models",
    verbose: bool = True,
) -> Dict:
    """
    Train DQN agent on elevator dispatch.

    Args:
        n_episodes: Number of training episodes
        n_floors: Number of floors in building
        n_elevators: Number of elevators
        traffic_pattern: Traffic pattern to use
        save_dir: Directory to save models and metrics
        verbose: Print training progress

    Returns:
        Dictionary with training metrics
    """
    # Create environment
    env = ElevatorDispatch(
        n_floors=n_floors,
        n_elevators=n_elevators,
        traffic_pattern=traffic_pattern,
        max_timesteps=300,
    )

    # Get observation dimension from environment
    obs, _ = env.reset()
    obs_dim = obs["elevator_0"].shape[0]

    # Create agent
    agent = ElevatorDQN(
        n_floors=n_floors,
        n_elevators=n_elevators,
        observation_dim=obs_dim,
        hidden_dims=(128, 128),
        learning_rate=1e-3,
        gamma=0.99,
        epsilon=1.0,
        epsilon_decay=0.995,
        epsilon_min=0.01,
        buffer_size=50000,
        batch_size=64,
    )

    # Training metrics
    episode_rewards = []
    avg_wait_times = []
    epsilon_history = []
    loss_history = []

    if verbose:
        print(f"Training DQN on {n_floors}-floor building with {n_elevators} elevators")
        print(f"Traffic pattern: {traffic_pattern}")
        print(f"Episodes: {n_episodes}\n")

    # Training loop
    for episode in range(n_episodes):
        obs, info = env.reset()
        episode_reward = 0
        episode_losses = []

        for step in range(env.max_timesteps):
            # Select actions
            actions = agent.select_actions(obs, training=True)

            # Take step
            next_obs, reward, terminated, truncated, info = env.step(actions)

            # Store transition
            agent.store_transitions(obs, actions, reward, next_obs, terminated or truncated)

            # Train
            loss = agent.train_step()
            if loss is not None:
                episode_losses.append(loss)

            episode_reward += reward
            obs = next_obs

            if terminated or truncated:
                break

        # Decay epsilon
        agent.decay_epsilon()

        # Record metrics
        episode_rewards.append(episode_reward)
        avg_wait_times.append(info["avg_wait_time"])
        epsilon_history.append(agent.epsilon)
        if episode_losses:
            loss_history.append(np.mean(episode_losses))

        # Print progress
        if verbose and (episode + 1) % 50 == 0:
            recent_reward = np.mean(episode_rewards[-50:])
            recent_wait = np.mean(avg_wait_times[-50:])
            print(
                f"Episode {episode + 1}/{n_episodes} | "
                f"Reward: {recent_reward:.1f} | "
                f"Avg Wait: {recent_wait:.1f}s | "
                f"ε: {agent.epsilon:.3f} | "
                f"Buffer: {len(agent.replay_buffer)}"
            )

    # Evaluate against baselines
    if verbose:
        print("\nEvaluating against baselines...")

    baseline_results = evaluate_baselines(env, n_episodes=20, verbose=verbose)

    # Evaluate trained agent
    agent_results = evaluate_agent(env, agent, n_episodes=20, verbose=verbose)

    # Save results
    save_dir_path = Path(save_dir)
    save_dir_path.mkdir(exist_ok=True, parents=True)

    # Save model
    model_path = save_dir_path / "elevator_dqn.pt"
    agent.save(str(model_path))
    if verbose:
        print(f"\nModel saved to {model_path}")

    # Save training metrics
    metrics = {
        "episode_rewards": episode_rewards,
        "avg_wait_times": avg_wait_times,
        "epsilon_history": epsilon_history,
        "loss_history": loss_history,
        "baseline_results": baseline_results,
        "agent_results": agent_results,
        "config": {
            "n_episodes": n_episodes,
            "n_floors": n_floors,
            "n_elevators": n_elevators,
            "traffic_pattern": traffic_pattern,
        },
    }

    metrics_path = save_dir_path / "training_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    if verbose:
        print(f"Metrics saved to {metrics_path}")

        print("\n" + "=" * 50)
        print("FINAL RESULTS")
        print("=" * 50)
        print(f"{'Algorithm':<20} {'Avg Wait (s)':<15} {'Passengers Served':<20}")
        print("-" * 50)
        for algo, result in {**baseline_results, "DQN (Trained)": agent_results}.items():
            print(
                f"{algo:<20} {result['avg_wait_time']:<15.2f} "
                f"{result['avg_passengers_served']:<20.1f}"
            )
        print("=" * 50)

    return metrics


def evaluate_baselines(env, n_episodes: int = 20, verbose: bool = True) -> Dict:
    """Evaluate baseline policies."""
    policies = {
        "Random": random_policy,
        "Nearest Car": nearest_car_policy,
        "SCAN": scan_policy,
    }

    results = {}

    for name, policy_fn in policies.items():
        wait_times = []
        passengers_served = []

        for _ in range(n_episodes):
            obs, info = env.reset()

            for step in range(env.max_timesteps):
                actions = policy_fn(env)
                obs, reward, terminated, truncated, info = env.step(actions)

                if terminated or truncated:
                    break

            wait_times.append(info["avg_wait_time"])
            passengers_served.append(info["delivered_passengers"])

        results[name] = {
            "avg_wait_time": np.mean(wait_times),
            "avg_passengers_served": np.mean(passengers_served),
        }

        if verbose:
            print(f"  {name}: Wait={results[name]['avg_wait_time']:.2f}s, "
                  f"Served={results[name]['avg_passengers_served']:.1f}")

    return results


def evaluate_agent(env, agent: ElevatorDQN, n_episodes: int = 20, verbose: bool = True) -> Dict:
    """Evaluate trained DQN agent."""
    wait_times = []
    passengers_served = []

    for _ in range(n_episodes):
        obs, info = env.reset()

        for step in range(env.max_timesteps):
            actions = agent.select_actions(obs, training=False)  # No exploration
            obs, reward, terminated, truncated, info = env.step(actions)

            if terminated or truncated:
                break

        wait_times.append(info["avg_wait_time"])
        passengers_served.append(info["delivered_passengers"])

    result = {
        "avg_wait_time": np.mean(wait_times),
        "avg_passengers_served": np.mean(passengers_served),
    }

    if verbose:
        print(f"  DQN (Trained): Wait={result['avg_wait_time']:.2f}s, "
              f"Served={result['avg_passengers_served']:.1f}")

    return result


def main():
    parser = argparse.ArgumentParser(description="Train DQN for elevator dispatch")
    parser.add_argument("--episodes", type=int, default=1000, help="Number of training episodes")
    parser.add_argument("--n-floors", type=int, default=10, help="Number of floors")
    parser.add_argument("--n-elevators", type=int, default=3, help="Number of elevators")
    parser.add_argument(
        "--traffic",
        type=str,
        default="morning_rush",
        choices=["morning_rush", "lunch_time", "evening_rush", "quiet"],
        help="Traffic pattern",
    )
    parser.add_argument(
        "--save-dir", type=str, default="trained_models", help="Directory to save results"
    )
    parser.add_argument("--quiet", action="store_true", help="Suppress output")

    args = parser.parse_args()

    train_dqn(
        n_episodes=args.episodes,
        n_floors=args.n_floors,
        n_elevators=args.n_elevators,
        traffic_pattern=args.traffic,
        save_dir=args.save_dir,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
