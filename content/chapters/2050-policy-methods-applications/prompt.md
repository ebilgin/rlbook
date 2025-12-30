# Chapter 24: Policy Gradient Methods in Practice

## Chapter Metadata

**Chapter Number:** 24
**Title:** Policy Gradient Methods in Practice
**Section:** Policy Gradient Methods
**Prerequisites:**
- Chapter 23: PPO and Trust Region Methods
- Chapter 22: Actor-Critic Methods
- All previous chapters in the Policy Gradient section
**Estimated Reading Time:** 25 minutes

---

## Learning Objectives

By the end of this chapter, readers will be able to:

1. Apply policy gradient methods to robotics and continuous control tasks
2. Understand how RLHF (Reinforcement Learning from Human Feedback) uses PPO
3. Recognize practical challenges: reward shaping, sim-to-real, sample efficiency
4. Know when to choose policy gradient vs. value-based methods
5. Navigate the landscape of modern RL algorithms

---

## Core Concepts to Cover

### Primary Concepts (Must Cover)
- [ ] Continuous control: robotics with policy gradients
- [ ] RLHF: training language models with PPO
- [ ] Sim-to-real transfer: training in simulation, deploying in reality
- [ ] Reward engineering: designing rewards for real-world objectives
- [ ] Method selection guide: when to use which algorithm

### Secondary Concepts (Cover if Space Permits)
- [ ] Multi-task and meta-learning with policy gradients
- [ ] Combining with model-based RL
- [ ] Safety constraints in policy optimization
- [ ] Recent developments: PPO improvements, decision transformers

### Explicitly Out of Scope
- Detailed implementation of application-specific systems
- Off-policy continuous control (SAC, TD3 — different section)
- Full RLHF pipeline (too application-specific)
- Research frontiers (too speculative)

---

## Narrative Arc

### Opening Hook
"Policy gradient methods aren't just theoretical—they're training robots to walk, teaching language models to be helpful, and powering some of the most impressive AI systems today. Let's see how."

Ground the theory in real applications.

### Key Insight
Policy gradient methods—especially PPO—have become the go-to approach for:
1. **Continuous control**: Robots, simulators, games with continuous actions
2. **RLHF**: Fine-tuning large language models with human preferences
3. **Complex environments**: Where value-based methods struggle with dimensionality

The key is their flexibility: any parameterized policy, any differentiable objective.

### Closing Connection
"This concludes our journey through policy gradient methods. You now understand the core ideas from REINFORCE through PPO, and you've seen how they power real systems. The RL landscape continues to evolve—actor-critic ideas combine with offline RL, model-based methods, and foundation models. The tools you've learned here are the foundation for understanding what comes next."

---

## Required Interactive Elements

### Demo 1: MuJoCo-Style Walker
- **Purpose:** Show policy gradient solving continuous control
- **Interaction:**
  - Watch a simulated robot learn to walk
  - Visualize policy outputs (Gaussian mean/std for each joint)
  - See how policy improves over training
  - Try different random seeds—robustness varies
- **Expected Discovery:** Policy gradients can learn complex motor skills; training can be unstable but PPO helps.

### Demo 2: RLHF Concept Demonstration
- **Purpose:** Show how human preferences become rewards
- **Interaction:**
  - See two model outputs for a prompt
  - Choose which is better (simulated or real)
  - See how the reward model learns from preferences
  - Watch how PPO fine-tunes toward preferred outputs
- **Expected Discovery:** RLHF turns human preferences into optimization signal; PPO is the optimization engine.

### Demo 3: Algorithm Selection Guide
- **Purpose:** Help readers choose the right algorithm
- **Interaction:**
  - Answer questions about problem characteristics
  - Get algorithm recommendation with explanation
  - See comparison on relevant dimensions
- **Expected Discovery:** Different algorithms suit different problems; there's no universal winner.

---

## Recurring Examples to Use

- **HalfCheetah/Hopper/Walker**: Standard MuJoCo continuous control benchmarks
- **LunarLander-Continuous**: Accessible continuous control task
- **CartPole**: Simple baseline for comparisons
- **Custom**: RLHF conceptual example with text generation

---

## Cross-References

### Build On (Backward References)
- All previous policy gradient chapters
- Chapter 15: "Recall the limitations of Q-learning that led us to policy methods..."
- Chapter 13 (DQN): "Deep Q-learning for discrete actions, policy gradients for continuous"

### Set Up (Forward References)
- Future sections: "Off-policy actor-critic methods like SAC..."
- Future sections: "Model-based RL can improve sample efficiency..."

---

## Mathematical Depth

### Required Equations
This chapter is more application-focused; math is lighter.

1. Gaussian policy for continuous actions:
$$\pi_\theta(a|s) = \mathcal{N}(\mu_\theta(s), \sigma_\theta(s)^2)$$

2. RLHF reward model training (Bradley-Terry model):
$$P[\text{response } A \succ B] = \sigma(r(A) - r(B))$$

3. PPO objective for RLHF (with KL penalty):
$$J(\theta) = \mathbb{E}\left[r_\phi(s, a) - \beta D_\text{KL}(\pi_\theta \| \pi_\text{ref})\right]$$

### Derivations to Include (Mathematical Layer)
- How to sample from and compute log-probability of Gaussian policy
- Why KL penalty prevents mode collapse in RLHF

### Proofs to Omit
- Most proofs—this is an applications chapter
- Sim-to-real transfer theory

---

## Code Examples Needed

### Intuition Layer
```python
# Gaussian policy for continuous control
mean = actor_network(state)  # Neural network outputs mean
std = torch.exp(log_std)     # Learnable log standard deviation
action = torch.normal(mean, std)  # Sample action
log_prob = Normal(mean, std).log_prob(action).sum()  # For policy gradient
```

### Implementation Layer
- Complete PPO for continuous actions (Gaussian policy)
- Training on Pendulum or LunarLander-Continuous
- RLHF conceptual code (reward model + PPO loop)
- Sim-to-real discussion with domain randomization example

---

## Common Misconceptions to Address

1. **"Policy gradients are always better for continuous actions"**: SAC and TD3 (off-policy actor-critic) are often more sample-efficient. PPO is simpler and more stable, but not always faster.

2. **"RLHF is just PPO"**: RLHF is a pipeline: preference data collection → reward model training → PPO fine-tuning → repeat. PPO is just one component.

3. **"Sim-to-real always works"**: The reality gap can be severe. Domain randomization helps but isn't magic. Real-world learning often supplements simulation.

4. **"More training always helps"**: Overfitting to the reward model is a real problem in RLHF. Over-optimization can produce high-reward but poor-quality outputs.

5. **"You need MuJoCo or expensive simulators"**: Many continuous control tasks can be done in simpler environments. PyBullet is free and works well.

6. **"Policy gradient methods are black boxes"**: The policy is interpretable! You can visualize action distributions, see what the policy "wants" to do.

---

## Exercises

### Conceptual (4 questions)
- Why is PPO particularly well-suited for RLHF? What properties make it appropriate?
- What is the sim-to-real gap? Name three techniques to address it.
- When would you choose Q-learning over policy gradients? When policy gradients over Q-learning?
- Why do we need a KL penalty in RLHF? What goes wrong without it?

### Coding (2 challenges)
- Implement PPO with Gaussian policy for continuous actions. Train on Pendulum environment.
- Implement a simple reward model that learns from pairwise comparisons. (Doesn't need to connect to LLM—just demonstrate the concept.)

### Exploration (1 open-ended)
- Research a recent application of RL in a domain you're interested in (robotics, games, NLP, etc.). What algorithm did they use? What were the key challenges they faced?

---

## Section Summary: Method Selection Guide

| Problem Type | Recommended Method | Why |
|--------------|-------------------|-----|
| Discrete actions, tabular | Q-learning | Simple, effective, converges |
| Discrete actions, deep | DQN + variants | Sample efficient, off-policy |
| Continuous actions, on-policy | PPO | Stable, simple, works well |
| Continuous actions, sample efficiency matters | SAC | Off-policy, reuses data |
| High-dimensional observation, continuous actions | PPO or SAC | Both work, SAC more efficient |
| Human feedback available | PPO (RLHF) | Stable updates, KL control |
| Safety critical | Constrained PPO / Safe RL | Hard constraints on policy |
| Data available, no simulation | Offline RL (CQL, IQL) | No environment interaction |

---

## Additional Context for AI

- This chapter is a capstone—tie together everything from the section
- Balance theory and practice; readers should feel empowered to apply these methods
- RLHF is topical and important; explain it conceptually without getting into LLM details
- Sim-to-real is practical knowledge that researchers need
- The method selection guide is high-value practical content
- Don't be dogmatic about algorithm choices—there's rarely a clear winner
- Connect to the broader RL landscape; acknowledge what we haven't covered
- End on an inspiring note—readers have learned a powerful set of tools
- Keep it grounded: these methods power real systems (OpenAI, robotics, games)

---

## Quality Checklist

- [ ] Continuous control is demonstrated with clear example
- [ ] RLHF is explained conceptually with enough detail to understand
- [ ] Sim-to-real challenges are covered practically
- [ ] Method selection guide is actionable and useful
- [ ] Code examples show complete continuous control PPO
- [ ] Applications feel real and motivating
- [ ] Section provides satisfying conclusion to policy gradient content
- [ ] Forward references to future content where appropriate
- [ ] Interactive demos make applications tangible
