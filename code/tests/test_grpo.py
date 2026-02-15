"""Tests for GRPO agent implementation."""

import pytest
import numpy as np
from rlbook.agents.grpo import GRPOTrainer, GRPOConfig, simple_grpo_update


class TestGRPOAdvantages:
    """Tests for group-relative advantage computation."""

    def test_basic_binary_rewards(self):
        """Test advantages with binary (0/1) rewards."""
        trainer = GRPOTrainer(GRPOConfig(group_size=4, normalize_advantages=False))
        rewards = np.array([[1.0, 0.0, 1.0, 0.0]])

        adv = trainer.compute_advantages(rewards)

        # Mean is 0.5, so correct gets +0.5, wrong gets -0.5
        assert adv.shape == (1, 4)
        np.testing.assert_allclose(adv[0, 0], 0.5)
        np.testing.assert_allclose(adv[0, 1], -0.5)
        np.testing.assert_allclose(adv[0, 2], 0.5)
        np.testing.assert_allclose(adv[0, 3], -0.5)

    def test_normalized_advantages(self):
        """Test advantages with sigma normalization."""
        trainer = GRPOTrainer(GRPOConfig(group_size=4, normalize_advantages=True))
        rewards = np.array([[1.0, 0.0, 1.0, 0.0]])

        adv = trainer.compute_advantages(rewards)

        # Should have zero mean and unit variance (approximately)
        np.testing.assert_allclose(adv.mean(), 0.0, atol=1e-7)
        # Normalized: (0.5) / std where std = 0.5
        np.testing.assert_allclose(adv[0, 0], 1.0, atol=1e-7)
        np.testing.assert_allclose(adv[0, 1], -1.0, atol=1e-7)

    def test_all_same_rewards(self):
        """Test when all completions get the same reward."""
        trainer = GRPOTrainer(GRPOConfig(group_size=4, normalize_advantages=False))
        rewards = np.array([[1.0, 1.0, 1.0, 1.0]])

        adv = trainer.compute_advantages(rewards)

        # All advantages should be zero
        np.testing.assert_allclose(adv, 0.0, atol=1e-7)

    def test_all_same_rewards_normalized(self):
        """Test normalized advantages when all rewards are identical."""
        trainer = GRPOTrainer(GRPOConfig(group_size=4, normalize_advantages=True))
        rewards = np.array([[0.5, 0.5, 0.5, 0.5]])

        adv = trainer.compute_advantages(rewards)

        # Std is 0, but epsilon prevents division by zero
        np.testing.assert_allclose(adv, 0.0, atol=1e-5)

    def test_batch_independence(self):
        """Test that advantages are computed independently per prompt."""
        trainer = GRPOTrainer(GRPOConfig(group_size=3, normalize_advantages=False))
        rewards = np.array([
            [1.0, 0.0, 0.0],  # 1/3 correct
            [1.0, 1.0, 0.0],  # 2/3 correct
        ])

        adv = trainer.compute_advantages(rewards)

        # Group 1: mean=1/3, so adv = [2/3, -1/3, -1/3]
        np.testing.assert_allclose(adv[0, 0], 2 / 3, atol=1e-7)
        np.testing.assert_allclose(adv[0, 1], -1 / 3, atol=1e-7)

        # Group 2: mean=2/3, so adv = [1/3, 1/3, -2/3]
        np.testing.assert_allclose(adv[1, 0], 1 / 3, atol=1e-7)
        np.testing.assert_allclose(adv[1, 2], -2 / 3, atol=1e-7)

    def test_1d_rewards(self):
        """Test that 1D rewards are handled correctly."""
        trainer = GRPOTrainer(GRPOConfig(group_size=4, normalize_advantages=False))
        rewards = np.array([1.0, 0.0, 1.0, 0.0])

        adv = trainer.compute_advantages(rewards)

        assert adv.shape == (1, 4)
        np.testing.assert_allclose(adv[0, 0], 0.5)


class TestGRPOLoss:
    """Tests for the GRPO loss computation."""

    def test_no_update_when_ratio_is_one(self):
        """Test that loss is well-defined when log_probs == old_log_probs."""
        trainer = GRPOTrainer(GRPOConfig(group_size=2, kl_coef=0.0))
        log_probs = np.array([[-1.0, -2.0]])
        old_log_probs = np.array([[-1.0, -2.0]])
        advantages = np.array([[1.0, -1.0]])

        loss, metrics = trainer.compute_loss(log_probs, old_log_probs, advantages)

        # ratio = 1.0 for all, so clipping doesn't activate
        assert metrics["clip_fraction"] == 0.0
        assert metrics["mean_ratio"] == 1.0

    def test_clipping_activates(self):
        """Test that clipping activates when ratio is large."""
        trainer = GRPOTrainer(GRPOConfig(group_size=2, clip_eps=0.2, kl_coef=0.0))
        # Large difference in log probs -> large ratio
        log_probs = np.array([[-0.5, -2.0]])
        old_log_probs = np.array([[-2.0, -0.5]])
        advantages = np.array([[1.0, -1.0]])

        loss, metrics = trainer.compute_loss(log_probs, old_log_probs, advantages)

        # At least some ratios should be clipped
        assert metrics["clip_fraction"] > 0.0

    def test_kl_penalty(self):
        """Test that KL penalty increases loss when policy drifts."""
        config_no_kl = GRPOConfig(group_size=2, kl_coef=0.0)
        config_with_kl = GRPOConfig(group_size=2, kl_coef=1.0)

        trainer_no_kl = GRPOTrainer(config_no_kl)
        trainer_with_kl = GRPOTrainer(config_with_kl)

        # Policy has drifted from reference (asymmetric drift)
        log_probs = np.array([[-0.3, -1.5]])
        old_log_probs = np.array([[-1.0, -1.0]])
        advantages = np.array([[1.0, -1.0]])

        loss_no_kl, _ = trainer_no_kl.compute_loss(
            log_probs, old_log_probs, advantages
        )
        loss_with_kl, metrics = trainer_with_kl.compute_loss(
            log_probs, old_log_probs, advantages
        )

        # KL penalty should make loss different
        assert loss_no_kl != loss_with_kl
        # KL divergence should be non-zero
        assert metrics["kl_divergence"] != 0.0

    def test_positive_advantage_decreases_loss(self):
        """Test that increasing prob of high-advantage completion reduces loss."""
        trainer = GRPOTrainer(GRPOConfig(group_size=1, clip_eps=1.0, kl_coef=0.0))
        advantages = np.array([[1.0]])

        # Lower log prob (less likely)
        loss_low, _ = trainer.compute_loss(
            np.array([[-2.0]]), np.array([[-2.0]]), advantages
        )

        # Higher log prob (more likely) -> ratio > 1
        loss_high, _ = trainer.compute_loss(
            np.array([[-1.0]]), np.array([[-2.0]]), advantages
        )

        # Higher prob of positive-advantage completion should give lower loss
        assert loss_high < loss_low


class TestGRPOTrainStep:
    """Tests for the full training step."""

    def test_train_step_returns_metrics(self):
        """Test that train_step returns expected metrics."""
        trainer = GRPOTrainer(GRPOConfig(group_size=2))

        # Simple mock functions
        def gen(prompt):
            return "42"

        def rew(prompt, completion):
            return 1.0 if "42" in completion else 0.0

        def lp(prompt, completion):
            return -1.0

        metrics = trainer.train_step(
            prompts=["What is 6*7?"],
            generate_fn=gen,
            reward_fn=rew,
            log_prob_fn=lp,
            ref_log_prob_fn=lp,
        )

        assert "total_loss" in metrics
        assert "mean_reward" in metrics
        assert "mean_advantage" in metrics
        assert "completions" in metrics
        assert len(metrics["completions"]) == 1
        assert len(metrics["completions"][0]) == 2  # group_size

    def test_train_step_tracks_stats(self):
        """Test that training statistics are accumulated."""
        trainer = GRPOTrainer(GRPOConfig(group_size=2))

        gen = lambda p: "answer"
        rew = lambda p, c: 0.5
        lp = lambda p, c: -1.0

        # Two training steps
        trainer.train_step(["Q1"], gen, rew, lp, lp)
        trainer.train_step(["Q2"], gen, rew, lp, lp)

        assert len(trainer.stats["losses"]) == 2
        assert len(trainer.stats["mean_rewards"]) == 2

    def test_varied_completions_affect_advantages(self):
        """Test that different completions get different advantages."""
        trainer = GRPOTrainer(GRPOConfig(group_size=4, normalize_advantages=False))

        counter = [0]

        def gen(prompt):
            counter[0] += 1
            # 2 out of 4 will be correct
            return "42" if counter[0] % 2 == 0 else "wrong"

        def rew(prompt, completion):
            return 1.0 if completion == "42" else 0.0

        lp = lambda p, c: -1.0

        metrics = trainer.train_step(["What is 6*7?"], gen, rew, lp, lp)

        # Mean reward should be 0.5 (2 correct out of 4)
        assert metrics["mean_reward"] == 0.5


class TestSimpleGRPO:
    """Tests for the minimal simple_grpo_update function."""

    def test_binary_rewards(self):
        """Test with standard binary rewards."""
        rewards = np.array([1, 0, 1, 0, 0, 0, 1, 0])
        adv = simple_grpo_update(rewards, normalize=False)

        # 3 correct out of 8: mean = 0.375
        expected_correct = 1.0 - 0.375  # 0.625
        expected_wrong = 0.0 - 0.375  # -0.375

        np.testing.assert_allclose(adv[0], expected_correct, atol=1e-7)
        np.testing.assert_allclose(adv[1], expected_wrong, atol=1e-7)

    def test_normalized(self):
        """Test with normalization enabled."""
        rewards = np.array([1.0, 0.0, 1.0, 0.0])
        adv = simple_grpo_update(rewards, normalize=True)

        # Should have zero mean
        np.testing.assert_allclose(adv.mean(), 0.0, atol=1e-7)

    def test_all_zero_rewards(self):
        """Test with all-zero rewards (no signal)."""
        rewards = np.array([0.0, 0.0, 0.0, 0.0])
        adv = simple_grpo_update(rewards, normalize=False)

        np.testing.assert_allclose(adv, 0.0, atol=1e-7)

    def test_single_correct(self):
        """Test with only one correct completion."""
        rewards = np.array([0, 0, 0, 1, 0, 0, 0, 0])
        adv = simple_grpo_update(rewards, normalize=False)

        # The one correct completion should have high advantage
        assert adv[3] > 0
        # All wrong completions should have negative advantage
        assert all(adv[i] < 0 for i in [0, 1, 2, 4, 5, 6, 7])
