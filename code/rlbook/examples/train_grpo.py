"""
GRPO Training Example: Simple Math Reasoning

Demonstrates GRPO (Group Relative Policy Optimization) on a toy math task.
This example uses a mock language model to show the training loop structure
without requiring GPU resources.

Usage:
    python -m rlbook.examples.train_grpo

This code is referenced in:
    - Chapter: RL for Language Models (3040-rlhf)
    - Subsection: Building a Reasoning Model (hands-on)
"""

import numpy as np
from rlbook.agents.grpo import GRPOTrainer, GRPOConfig


# --- Toy Math Environment ---

MATH_PROBLEMS = [
    {"prompt": "What is 7 + 5?", "answer": "12"},
    {"prompt": "What is 3 * 8?", "answer": "24"},
    {"prompt": "What is 15 - 6?", "answer": "9"},
    {"prompt": "What is 20 / 4?", "answer": "5"},
    {"prompt": "What is 9 + 11?", "answer": "20"},
    {"prompt": "What is 6 * 7?", "answer": "42"},
    {"prompt": "What is 100 - 37?", "answer": "63"},
    {"prompt": "What is 48 / 8?", "answer": "6"},
]

ANSWER_LOOKUP = {p["prompt"]: p["answer"] for p in MATH_PROBLEMS}


class MockMathModel:
    """A mock language model that gradually learns to solve math.

    Simulates a model whose accuracy improves over training steps.
    The 'accuracy' parameter controls how likely it is to produce
    the correct answer versus a random wrong answer.
    """

    def __init__(self, initial_accuracy: float = 0.3):
        self.accuracy = initial_accuracy
        self.rng = np.random.default_rng(42)

    def generate(self, prompt: str) -> str:
        """Generate a completion for the given prompt."""
        correct = ANSWER_LOOKUP.get(prompt, "0")

        if self.rng.random() < self.accuracy:
            return f"The answer is {correct}."
        else:
            # Generate a plausible but wrong answer
            wrong = str(int(correct) + self.rng.integers(-5, 6))
            if wrong == correct:
                wrong = str(int(correct) + 1)
            return f"The answer is {wrong}."

    def log_prob(self, prompt: str, completion: str) -> float:
        """Mock log probability (higher for correct answers)."""
        correct = ANSWER_LOOKUP.get(prompt, "0")
        if correct in completion:
            return -0.5 - self.rng.random() * 0.5
        else:
            return -2.0 - self.rng.random() * 1.0

    def improve(self, delta: float = 0.05):
        """Simulate the model improving from training."""
        self.accuracy = min(1.0, self.accuracy + delta)


def math_reward(prompt: str, completion: str) -> float:
    """Binary reward: 1.0 if correct answer, 0.0 otherwise."""
    correct = ANSWER_LOOKUP.get(prompt, None)
    if correct is None:
        return 0.0
    return 1.0 if correct in completion else 0.0


def main():
    """Run GRPO training on toy math problems."""
    print("=" * 60)
    print("GRPO Training Example: Simple Math Reasoning")
    print("=" * 60)
    print()

    # Configure GRPO
    config = GRPOConfig(
        group_size=8,
        clip_eps=0.2,
        kl_coef=0.01,
        normalize_advantages=True,
    )
    trainer = GRPOTrainer(config)

    # Create mock model
    model = MockMathModel(initial_accuracy=0.3)
    ref_model = MockMathModel(initial_accuracy=0.3)  # Frozen reference

    print(f"Config: G={config.group_size}, clip_eps={config.clip_eps}, "
          f"kl_coef={config.kl_coef}")
    print(f"Initial accuracy: {model.accuracy:.0%}")
    print()

    # Training loop
    n_steps = 20
    prompts_list = [p["prompt"] for p in MATH_PROBLEMS]

    for step in range(n_steps):
        # Sample a batch of prompts
        batch = list(np.random.choice(prompts_list, size=4, replace=False))

        # Run one GRPO step
        metrics = trainer.train_step(
            prompts=batch,
            generate_fn=model.generate,
            reward_fn=math_reward,
            log_prob_fn=model.log_prob,
            ref_log_prob_fn=ref_model.log_prob,
        )

        # Simulate model improvement based on rewards
        if metrics["mean_reward"] > 0.3:
            model.improve(delta=0.03)

        # Log progress
        if step % 5 == 0 or step == n_steps - 1:
            print(f"Step {step:3d} | "
                  f"Reward: {metrics['mean_reward']:.3f} | "
                  f"Loss: {metrics['total_loss']:.4f} | "
                  f"KL: {metrics['kl_divergence']:.4f} | "
                  f"Accuracy: {model.accuracy:.0%}")

    print()
    print("-" * 60)
    print("Training complete!")
    print(f"Final accuracy: {model.accuracy:.0%}")
    print(f"Average reward (last 5 steps): "
          f"{np.mean(trainer.stats['mean_rewards'][-5:]):.3f}")
    print()

    # Show sample completions from last step
    print("Sample completions from last step:")
    for prompt_completions in metrics["completions"][:2]:
        for i, comp in enumerate(prompt_completions[:3]):
            print(f"  [{i}] {comp}")
        print()


if __name__ == "__main__":
    main()
