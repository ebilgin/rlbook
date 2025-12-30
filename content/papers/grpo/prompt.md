# Paper Explanation Prompt: Group Relative Policy Optimization (GRPO)

## Paper Metadata

**Slug:** grpo
**Paper Title:** DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models
**Authors:** Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, Y.K. Li, Y. Wu, Daya Guo
**Venue:** arXiv 2024, widely adopted in 2025
**ArXiv:** 2402.03300
**Paper URL:** https://arxiv.org/abs/2402.03300

---

## TL;DR

GRPO eliminates the critic network from PPO by computing advantages relative to a group of sampled completions, cutting RL training costs in half while matching or exceeding PPO performance.

---

## Target Audience

**Prerequisites:**
- Understanding of policy gradient methods (REINFORCE)
- Familiarity with PPO's clipped objective
- Basic understanding of LLM fine-tuning

**Who benefits most:**
- [x] Researchers implementing or extending this work
- [x] Practitioners applying this technique
- [x] Students understanding foundational ideas
- [x] Reading group participants

---

## Paper Summary

### The Problem

PPO requires training two models: a policy network and a critic (value) network. For LLMs, this doubles memory requirements and computational cost. Furthermore, critics struggle with sequence-level (outcome) rewards where only the final token receives a reward signal—the value function must somehow propagate this sparse signal back through all tokens.

### The Key Insight

Instead of learning what a "good" state looks like (the critic's job), we can simply ask: "How good is this completion compared to other completions for the same prompt?" By sampling multiple completions per prompt and normalizing rewards within each group, we get a stable baseline without any learned value function.

### The Approach

1. For each prompt, sample G completions from the current policy
2. Compute rewards for all completions
3. Normalize rewards within each group: advantage = (r - mean) / std
4. Apply PPO's clipped objective using these group-relative advantages
5. Add KL penalty to prevent divergence from reference policy

### Main Results

- DeepSeekMath-Instruct improved from 46.8% to 51.7% on MATH benchmark
- Approximately 50% reduction in compute requirements vs. PPO
- Used to train DeepSeek-R1, demonstrating scalability to reasoning models
- Now the standard RL algorithm for training Large Reasoning Models (LRMs)

---

## Content Structure

### Section 1: Motivation — The Critic Problem

**Cover:**
- Why PPO needs a critic (baseline for variance reduction)
- The memory cost: ~16GB per 1B parameters for the critic
- Why critics struggle with outcome rewards (sparse signal)
- The bootstrapping problem in language modeling

**Interactive element:** Show memory/compute comparison between PPO and GRPO setups

### Section 2: The Core Idea — Group-Relative Advantages

**Cover:**
- Traditional advantage: A(s,a) = Q(s,a) - V(s)
- GRPO's insight: V(s) ≈ mean reward of other completions
- Why this works: the prompt is the "state", completions are "actions"
- Normalization for stability

**Key equation:**
$$\hat{A}_i = \frac{r_i - \mu_{\text{group}}}{\sigma_{\text{group}} + \epsilon}$$

**Interactive element:** Visualize a group of completions with their rewards and show how advantages are computed

### Section 3: The Full Algorithm

**Cover:**
- Complete GRPO objective function
- The clipped surrogate (inherited from PPO)
- KL divergence penalty (DeepSeek's unbiased estimator)
- Per-token vs. per-completion rewards

**Key equations:**

The objective:
$$J_{\text{GRPO}}(\theta) = \mathbb{E}\left[\min\left(\frac{\pi_\theta}{\pi_{\text{old}}} \hat{A}, \text{clip}\left(\frac{\pi_\theta}{\pi_{\text{old}}}, 1-\epsilon, 1+\epsilon\right) \hat{A}\right)\right] - \beta D_{\text{KL}}(\pi_\theta \| \pi_{\text{ref}})$$

DeepSeek's KL estimator:
$$D_{\text{KL}} = e^{\log \pi_{\text{ref}} - \log \pi_\theta} - (\log \pi_{\text{ref}} - \log \pi_\theta) - 1$$

### Section 4: GRPO vs. PPO — A Detailed Comparison

**Cover:**
- Side-by-side architecture diagrams
- Memory requirements comparison
- Batch size requirements (GRPO needs more samples per prompt)
- When to prefer each algorithm

**Table to include:**
| Aspect | PPO | GRPO |
|--------|-----|------|
| Models trained | 2 (policy + critic) | 1 (policy only) |
| Advantage estimation | GAE with learned critic | Group statistics |
| Samples per prompt | 1 (typical) | 16-64 |
| Memory overhead | High | ~50% lower |

### Section 5: Implementation Details

**Cover:**
- Batch size considerations (B prompts × G completions)
- Single vs. multiple policy updates per batch
- Reward normalization across vs. within groups
- Reference policy management

**Pseudocode to include:**
```python
def grpo_step(prompts, policy, ref_policy, reward_fn, G=16):
    # 1. Sample G completions per prompt
    completions = [policy.generate(p, n=G) for p in prompts]

    # 2. Compute rewards
    rewards = reward_fn(completions)  # Shape: (B, G)

    # 3. Group-relative advantage
    mean_r = rewards.mean(dim=1, keepdim=True)
    std_r = rewards.std(dim=1, keepdim=True)
    advantages = (rewards - mean_r) / (std_r + 1e-8)

    # 4. Compute policy ratio and loss
    # ... (clipped objective + KL penalty)
```

### Section 6: Why GRPO Became Dominant

**Cover:**
- Timeline: DeepSeekMath → DeepSeek-R1 → widespread adoption
- Compatibility with verifiable rewards (math, code)
- Scalability to 100B+ parameter models
- The rise of Large Reasoning Models (LRMs)

### Section 7: Limitations and Extensions

**Cover:**
- Requires more samples per prompt (compute-memory tradeoff)
- Less suitable for dense (process) rewards
- Recent extensions: Training-Free GRPO, theoretical analyses
- Connection to other critic-free methods (REINFORCE, RLOO)

---

## Interactive Elements

### Demo 1: Group Advantage Visualizer
- **Purpose:** Show how advantages are computed from a group of completions
- **Interaction:** User sees 8 completions for a math problem, each with a reward. Slider adjusts one completion's reward to show how all advantages change.
- **Expected Discovery:** Advantages are relative—a completion with reward 0.7 has positive advantage in a weak group but negative advantage in a strong group.

### Demo 2: PPO vs. GRPO Memory/Compute Comparison
- **Purpose:** Visualize the efficiency gains of removing the critic
- **Interaction:** User selects model size (7B, 13B, 70B) and sees memory breakdown
- **Expected Discovery:** Critic overhead becomes significant at scale; GRPO enables training larger models with same hardware.

### Demo 3: Policy Update Dynamics
- **Purpose:** Show how GRPO updates the policy based on group statistics
- **Interaction:** Animated visualization of policy probabilities being adjusted based on relative performance
- **Expected Discovery:** Good completions get reinforced, but "good" is defined relative to siblings, not absolutely.

---

## Chapter Connections

### Builds On
- Chapter: policy-gradients — REINFORCE baseline, variance reduction concepts
- Chapter: ppo — Clipped objective, importance sampling ratios
- Chapter: rlhf — Reward modeling, KL constraints

### Referenced By
- Chapter: llm-reasoning — GRPO as the training algorithm for reasoning models
- Chapter: rlhf — As a modern alternative to PPO for LLM alignment

### Relationship Type
extends (builds on PPO with key simplifications)

---

## Mathematical Depth

### Key Equations (Intuition Layer)
1. Group-relative advantage: A = (r - mean) / std — with explanation that this replaces the critic
2. Why normalization helps: prevents one high-reward prompt from dominating

### Full Derivations (Mathematical Layer)
1. Connection to REINFORCE with baseline
2. Derivation of the unbiased KL estimator
3. Why group normalization is an unbiased estimator of advantage

### Proofs to Summarize
- Convergence properties (reference original PPO proofs, note GRPO inherits them under mild assumptions)

---

## Code Examples

### Pseudocode (Intuition Layer)
- Simplified 10-line version showing the core loop
- Focus on: sample → reward → normalize → update

### Implementation (Implementation Layer)
- Full PyTorch implementation (~50 lines)
- Integration with HuggingFace TRL library
- Common hyperparameters and their effects

---

## Common Misunderstandings

1. **"GRPO doesn't use clipping"**: It does use PPO's clipped objective; it just doesn't use a critic.

2. **"GRPO requires less compute"**: It requires less memory (no critic), but may need more samples per prompt. Total FLOPs can be similar.

3. **"GRPO only works for math problems"**: It works for any task with verifiable rewards; math was just the first application.

4. **"The advantage is per-completion"**: For outcome rewards, yes. For process rewards, advantages can vary by token.

---

## Discussion Questions

For reading groups or self-study:

1. Why might group-relative advantages be particularly well-suited to LLM training, where the "state" (prompt) is rich and complex?

2. How does the choice of group size G affect the bias-variance tradeoff? What happens with G=2 vs. G=64?

3. Could you combine GRPO with a lightweight critic for process rewards? What would be the tradeoffs?

4. GRPO normalizes within groups. What if you normalized across the entire batch instead? How would this change the learning dynamics?

---

## Additional Context for AI

- Emphasize the practical impact: GRPO enabled DeepSeek-R1, which demonstrated that open models can match proprietary reasoning capabilities
- Connect to the broader trend of "scaling RL" for LLMs
- The paper is technically about DeepSeekMath, but GRPO is the contribution that had lasting impact
- Readers may come from either RL background (familiar with PPO) or LLM background (familiar with fine-tuning)—bridge both perspectives
- Include the actual numbers: ~50% memory reduction, 16-64 samples per prompt typical

---

## Quality Checklist

Before accepting generated content, verify:

- [ ] TL;DR captures the "no critic, group normalization" insight
- [ ] Prerequisites accurately link to PPO and policy gradient chapters
- [ ] The comparison table between PPO and GRPO is accurate
- [ ] All three complexity layers are present and properly tagged
- [ ] Interactive demos are specified with clear group-based visualizations
- [ ] Chapter connections are bidirectional (update connections.yaml)
- [ ] Mathematical notation follows MATH_CONVENTIONS.md
- [ ] Code examples show both pseudocode and realistic implementation
- [ ] Discussion questions probe understanding of the group-relative insight
