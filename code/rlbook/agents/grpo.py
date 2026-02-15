"""
Group Relative Policy Optimization (GRPO)

A simplified, educational implementation of GRPO for language model fine-tuning.
GRPO eliminates the critic network from PPO by using group-relative advantages:
instead of learning a value function, it samples G completions per prompt and
normalizes rewards within each group.

This code is referenced in:
    - Chapter: RL for Language Models (3040-rlhf)
    - Subsection: Building a Reasoning Model (hands-on)
    - Paper: GRPO (papers/grpo)

Key insight: GRPO is essentially REINFORCE with a group mean baseline and PPO
clipping. The group statistics replace the learned critic, which means:
    - No value network to train (saves memory)
    - No critic approximation error (more stable)
    - Requires G forward passes per prompt (compute tradeoff)

References:
    - DeepSeekMath: "Group Relative Policy Optimization" (Shao et al., 2024)
    - Karpathy's nanochat: github.com/karpathy/nanochat

Example:
    >>> from rlbook.agents.grpo import GRPOTrainer
    >>> trainer = GRPOTrainer(group_size=8, clip_eps=0.2, kl_coef=0.01)
    >>> advantages = trainer.compute_advantages(rewards)
    >>> loss = trainer.compute_loss(log_probs, old_log_probs, advantages)
"""

from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple

import numpy as np


@dataclass
class GRPOConfig:
    """Configuration for GRPO training.

    Args:
        group_size: Number of completions to sample per prompt (G).
            Larger groups reduce variance but cost more compute.
            Typical values: 4-32.
        clip_eps: PPO clipping parameter (epsilon).
            Limits how much the policy can change in one step.
            Default 0.2 matches the original PPO paper.
        kl_coef: KL divergence penalty coefficient (beta).
            Higher values keep the policy closer to the reference model.
            Set to 0 for short training runs (like nanochat).
        normalize_advantages: Whether to divide by group std.
            True for full GRPO, False for nanochat-style simplification.
        max_tokens: Maximum tokens per completion.
    """

    group_size: int = 8
    clip_eps: float = 0.2
    kl_coef: float = 0.01
    normalize_advantages: bool = True
    max_tokens: int = 512


class GRPOTrainer:
    """Educational GRPO trainer.

    Implements the core GRPO algorithm without deep learning framework
    dependencies. Uses numpy for clarity. For a PyTorch version suitable
    for actual LLM training, see Karpathy's nanochat or HuggingFace TRL.

    The algorithm has three phases per batch:
        1. Generate: Sample G completions per prompt
        2. Score: Compute rewards (binary for math, learned for chat)
        3. Update: Policy gradient with group-relative advantages

    Args:
        config: GRPO hyperparameters

    Attributes:
        config: Training configuration
        stats: Dictionary tracking training metrics

    Example:
        >>> config = GRPOConfig(group_size=4, clip_eps=0.2)
        >>> trainer = GRPOTrainer(config)
        >>> rewards = np.array([[1, 0, 1, 0], [0, 0, 1, 0]])
        >>> advantages = trainer.compute_advantages(rewards)
        >>> advantages.shape
        (2, 4)
    """

    def __init__(self, config: Optional[GRPOConfig] = None):
        self.config = config or GRPOConfig()
        self.stats: dict = {
            "losses": [],
            "mean_rewards": [],
            "mean_kl": [],
        }

    def compute_advantages(self, rewards: np.ndarray) -> np.ndarray:
        """Compute group-relative advantages.

        This is the core GRPO innovation: advantages are computed relative
        to the group, not from a learned value function. For each prompt,
        we normalize rewards within the group of G completions.

        With normalize_advantages=True (full GRPO):
            A_i = (r_i - mean(group)) / (std(group) + eps)

        With normalize_advantages=False (nanochat-style):
            A_i = r_i - mean(group)

        Args:
            rewards: Shape [batch_size, group_size]. Rewards for each
                completion. For verifiable rewards (math/code), these
                are typically binary (0 or 1).

        Returns:
            Advantages with same shape as rewards.

        Example:
            >>> trainer = GRPOTrainer(GRPOConfig(group_size=4))
            >>> rewards = np.array([[1.0, 0.0, 1.0, 0.0]])
            >>> adv = trainer.compute_advantages(rewards)
            >>> # Correct completions get positive advantage
            >>> assert adv[0, 0] > 0
            >>> assert adv[0, 1] < 0
        """
        rewards = np.asarray(rewards, dtype=np.float64)

        if rewards.ndim == 1:
            rewards = rewards.reshape(1, -1)

        mean = rewards.mean(axis=1, keepdims=True)
        advantages = rewards - mean

        if self.config.normalize_advantages:
            std = rewards.std(axis=1, keepdims=True)
            advantages = advantages / (std + 1e-8)

        return advantages

    def compute_loss(
        self,
        log_probs: np.ndarray,
        old_log_probs: np.ndarray,
        advantages: np.ndarray,
    ) -> Tuple[float, dict]:
        """Compute the GRPO loss with PPO clipping and KL penalty.

        The loss combines three components:
            1. Clipped surrogate objective (from PPO)
            2. KL divergence penalty (prevents reward hacking)

        The clipped objective for each completion i:
            L_i = -min(rho_i * A_i, clip(rho_i, 1-eps, 1+eps) * A_i)

        where rho_i = exp(log_prob_new - log_prob_old) is the importance ratio.

        Args:
            log_probs: Log probabilities under current policy.
                Shape [batch_size, group_size].
            old_log_probs: Log probabilities under reference policy.
                Shape [batch_size, group_size].
            advantages: Group-relative advantages from compute_advantages.
                Shape [batch_size, group_size].

        Returns:
            Tuple of (scalar loss, metrics dict).

        Example:
            >>> trainer = GRPOTrainer(GRPOConfig(group_size=2, kl_coef=0.1))
            >>> log_probs = np.array([[-1.0, -2.0]])
            >>> old_log_probs = np.array([[-1.1, -1.9]])
            >>> advantages = np.array([[1.0, -1.0]])
            >>> loss, metrics = trainer.compute_loss(log_probs, old_log_probs, advantages)
            >>> isinstance(loss, float)
            True
        """
        log_probs = np.asarray(log_probs, dtype=np.float64)
        old_log_probs = np.asarray(old_log_probs, dtype=np.float64)
        advantages = np.asarray(advantages, dtype=np.float64)

        # Importance sampling ratio: rho = pi_new / pi_old
        ratio = np.exp(log_probs - old_log_probs)

        # Clipped surrogate objective
        eps = self.config.clip_eps
        unclipped = ratio * advantages
        clipped = np.clip(ratio, 1 - eps, 1 + eps) * advantages
        policy_loss = -np.minimum(unclipped, clipped).mean()

        # KL divergence: approximate KL(pi_new || pi_old)
        kl = (old_log_probs - log_probs).mean()

        # Total loss
        total_loss = policy_loss + self.config.kl_coef * kl

        metrics = {
            "policy_loss": float(policy_loss),
            "kl_divergence": float(kl),
            "total_loss": float(total_loss),
            "mean_ratio": float(ratio.mean()),
            "clip_fraction": float(
                np.mean(np.abs(ratio - 1.0) > eps)
            ),
        }

        return float(total_loss), metrics

    def train_step(
        self,
        prompts: List[str],
        generate_fn: Callable[[str], str],
        reward_fn: Callable[[str, str], float],
        log_prob_fn: Callable[[str, str], float],
        ref_log_prob_fn: Callable[[str, str], float],
    ) -> dict:
        """Execute one full GRPO training step.

        This is the main training loop abstracted from the model:
            1. For each prompt, generate G completions
            2. Score each completion with reward_fn
            3. Compute group-relative advantages
            4. Compute loss with clipping and KL penalty

        The actual parameter update is left to the caller (since we
        don't depend on any specific deep learning framework).

        Args:
            prompts: Batch of prompts to train on.
            generate_fn: Function that generates a completion given a prompt.
            reward_fn: Function that scores (prompt, completion) -> reward.
            log_prob_fn: Function computing log P(completion|prompt) under
                the current policy.
            ref_log_prob_fn: Function computing log P(completion|prompt)
                under the reference (frozen) policy.

        Returns:
            Dictionary with training metrics and computed loss.

        Example:
            >>> config = GRPOConfig(group_size=2)
            >>> trainer = GRPOTrainer(config)
            >>> # Mock functions for testing
            >>> gen = lambda p: "42"
            >>> rew = lambda p, c: 1.0 if c == "42" else 0.0
            >>> lp = lambda p, c: -1.0
            >>> rlp = lambda p, c: -1.0
            >>> metrics = trainer.train_step(["What is 6*7?"], gen, rew, lp, rlp)
            >>> "total_loss" in metrics
            True
        """
        G = self.config.group_size
        batch_size = len(prompts)

        # Phase 1: Generate G completions per prompt
        completions = []
        for prompt in prompts:
            group = [generate_fn(prompt) for _ in range(G)]
            completions.append(group)

        # Phase 2: Score with reward function
        rewards = np.zeros((batch_size, G))
        for i, prompt in enumerate(prompts):
            for j in range(G):
                rewards[i, j] = reward_fn(prompt, completions[i][j])

        # Phase 3: Compute group-relative advantages
        advantages = self.compute_advantages(rewards)

        # Phase 4: Compute log probabilities
        log_probs = np.zeros((batch_size, G))
        old_log_probs = np.zeros((batch_size, G))
        for i, prompt in enumerate(prompts):
            for j in range(G):
                log_probs[i, j] = log_prob_fn(prompt, completions[i][j])
                old_log_probs[i, j] = ref_log_prob_fn(
                    prompt, completions[i][j]
                )

        # Phase 5: Compute loss
        loss, metrics = self.compute_loss(log_probs, old_log_probs, advantages)

        # Track statistics
        metrics["mean_reward"] = float(rewards.mean())
        metrics["reward_std"] = float(rewards.std())
        metrics["mean_advantage"] = float(advantages.mean())
        metrics["completions"] = completions

        self.stats["losses"].append(loss)
        self.stats["mean_rewards"].append(metrics["mean_reward"])
        self.stats["mean_kl"].append(metrics["kl_divergence"])

        return metrics


def simple_grpo_update(
    rewards: np.ndarray,
    normalize: bool = True,
) -> np.ndarray:
    """Minimal GRPO: just compute group-relative advantages.

    This is the simplest possible GRPO implementation, equivalent to
    what Karpathy's nanochat uses (REINFORCE with group mean baseline).

    Args:
        rewards: Shape [batch_size, group_size] or [group_size].
        normalize: Whether to divide by standard deviation.

    Returns:
        Advantages with same shape as input.

    Example:
        >>> # Binary rewards: 3 correct out of 8
        >>> rewards = np.array([1, 0, 1, 0, 0, 0, 1, 0])
        >>> adv = simple_grpo_update(rewards, normalize=False)
        >>> # Correct completions get +0.625, wrong get -0.375
        >>> np.isclose(adv[0], 0.625)
        True
        >>> np.isclose(adv[1], -0.375)
        True
    """
    rewards = np.asarray(rewards, dtype=np.float64)
    mean = rewards.mean()
    advantages = rewards - mean

    if normalize:
        std = rewards.std()
        if std > 1e-8:
            advantages = advantages / std

    return advantages
