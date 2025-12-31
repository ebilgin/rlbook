# Chapter: RLHF and Language Models

## Chapter Metadata

**Chapter Number:** 25
**Title:** RLHF and Language Models
**Section:** Advanced Topics
**Prerequisites:**
- PPO and Trust Region Methods
- (Recommended) Introduction to Policy Gradients
**Estimated Reading Time:** 40 minutes

---

## Learning Objectives

By the end of this chapter, readers will be able to:

1. Explain why RLHF is needed for aligning language models
2. Describe the three-stage RLHF pipeline
3. Explain how reward models learn from human preferences
4. Apply PPO concepts to language model fine-tuning
5. Understand recent alternatives like DPO
6. Identify open challenges in AI alignment

---

## Core Concepts to Cover

### Primary Concepts (Must Cover)
- [ ] The alignment problem: why supervised learning isn't enough
- [ ] The RLHF pipeline: SFT → Reward Modeling → RL
- [ ] Reward modeling from human preferences
- [ ] PPO for language models
- [ ] KL penalty to prevent reward hacking
- [ ] DPO: Direct Preference Optimization

### Secondary Concepts (Cover if Space Permits)
- [ ] Constitutional AI
- [ ] Debate and IDA
- [ ] Reward model interpretability
- [ ] Multi-objective RLHF

### Explicitly Out of Scope
- Detailed LLM architecture
- Pre-training details
- Deployment and safety considerations

---

## Narrative Arc

### Opening Hook
"ChatGPT didn't just learn to predict text—it learned what humans actually want. The secret? Reinforcement learning from human feedback (RLHF). It's the technique that transformed capable-but-quirky language models into helpful, harmless, and honest assistants."

### Key Insight
RLHF solves a fundamental problem: we can't write down exactly what we want from AI. Instead, we show examples of good behavior, train a reward model to predict what humans prefer, then use RL to optimize for that learned reward. It's preferences all the way down.

### Closing Connection
"RLHF is how we've aligned today's AI systems—but it's not the final answer. As models become more capable, we need better alignment techniques. The future of AI safety depends on understanding and improving these methods."

---

## Required Interactive Elements

### Demo 1: Preference Comparison
- **Purpose:** Show how human preferences are collected
- **Interaction:**
  - Show two AI responses to a prompt
  - User picks preferred one (or tie)
  - Show how this becomes training data
  - Accumulate preferences into a dataset
- **Expected Discovery:** Preferences are subjective but can be learned

### Demo 2: Reward Model Training
- **Purpose:** Visualize reward model learning
- **Interaction:**
  - Show preference pairs
  - Train a simple reward model
  - Visualize predictions on new responses
  - Show correlation with human preferences
- **Expected Discovery:** The model learns to predict human preferences

### Demo 3: The RLHF Pipeline
- **Purpose:** Show the complete flow
- **Interaction:**
  - Start with base model generating responses
  - Show SFT improving quality
  - Show reward model scoring responses
  - Show PPO optimizing for reward
  - Compare responses at each stage
- **Expected Discovery:** Each stage improves the model differently

### Demo 4: Reward Hacking
- **Purpose:** Show why KL penalty matters
- **Interaction:**
  - Optimize for reward without KL constraint
  - Watch model generate high-reward but garbage outputs
  - Add KL penalty; see quality maintained
- **Expected Discovery:** Unconstrained optimization exploits reward model flaws

---

## Recurring Examples to Use

- **ChatGPT-style responses:** The canonical example
- **Helpfulness comparison:** Detailed vs. terse answers
- **Harmlessness:** Refusing dangerous requests
- **Simple preference tasks:** Summarization, Q&A

---

## Cross-References

### Build On (Backward References)
- PPO: "We use PPO to optimize the policy..."
- Policy Gradients: "LLMs are policies over tokens..."
- Reward Functions: "Instead of designing reward, we learn it..."

### Set Up (Forward References)
- (This is the capstone, but mention future directions)
- AI safety research
- Scalable oversight

---

## Mathematical Depth

### Required Equations

1. **Reward model training (Bradley-Terry)**:
$$P(y_1 \succ y_2 | x) = \sigma(r_\phi(x, y_1) - r_\phi(x, y_2))$$

2. **RLHF objective with KL penalty**:
$$\max_\pi \mathbb{E}_{x \sim \mathcal{D}, y \sim \pi(\cdot|x)} \left[ r_\phi(x, y) - \beta \cdot D_{\text{KL}}(\pi(\cdot|x) \| \pi_{\text{ref}}(\cdot|x)) \right]$$

3. **PPO clipped objective (adapted for LM)**:
$$L^{\text{CLIP}} = \mathbb{E}_t \left[ \min(r_t \hat{A}_t, \text{clip}(r_t, 1-\epsilon, 1+\epsilon) \hat{A}_t) \right]$$

4. **DPO loss**:
$$L_{\text{DPO}} = -\mathbb{E}_{(x,y_w,y_l)} \left[ \log \sigma \left( \beta \log \frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)} \right) \right]$$

### Derivations to Include (Mathematical Layer)
- Why Bradley-Terry model for preferences
- Deriving DPO from RLHF objective
- Why KL penalty prevents collapse

### Proofs to Omit
- Convergence guarantees
- Sample complexity bounds

---

## Code Examples Needed

### Intuition Layer
```python
def compute_reward(response, reward_model, tokenizer):
    """Compute reward for a response."""
    inputs = tokenizer(response, return_tensors="pt")
    reward = reward_model(**inputs).logits
    return reward.item()

def rlhf_step(prompt, model, reward_model, ref_model, beta=0.1):
    """Conceptual RLHF update."""
    # Generate response
    response = model.generate(prompt)

    # Compute reward
    reward = reward_model(prompt, response)

    # Compute KL penalty
    log_prob = model.log_prob(response | prompt)
    ref_log_prob = ref_model.log_prob(response | prompt)
    kl_penalty = log_prob - ref_log_prob

    # Total reward signal
    total_reward = reward - beta * kl_penalty

    # Update model with PPO using total_reward
    ...
```

### Implementation Layer
```python
class RewardModel(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base = base_model
        self.reward_head = nn.Linear(base_model.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        outputs = self.base(input_ids, attention_mask=attention_mask)
        hidden = outputs.last_hidden_state[:, -1, :]  # Last token
        reward = self.reward_head(hidden)
        return reward


def train_reward_model(model, preference_data, epochs=3):
    """Train reward model on preference data."""
    optimizer = AdamW(model.parameters(), lr=1e-5)

    for epoch in range(epochs):
        for prompt, chosen, rejected in preference_data:
            # Get rewards for both responses
            r_chosen = model(tokenize(prompt + chosen))
            r_rejected = model(tokenize(prompt + rejected))

            # Bradley-Terry loss
            loss = -F.logsigmoid(r_chosen - r_rejected).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
```

---

## Common Misconceptions to Address

1. **"RLHF makes models truthful"**: RLHF makes models say what humans prefer—which correlates with truth but isn't the same thing.

2. **"The reward model perfectly captures human values"**: Reward models are imperfect proxies. They can be exploited.

3. **"RLHF is why models refuse harmful requests"**: That's mostly from SFT and system prompts. RLHF fine-tunes behavior.

4. **"DPO is better than PPO"**: DPO is simpler but not universally better. PPO can still outperform for some tasks.

---

## Exercises

### Conceptual (3-5 questions)
1. Why can't we just use the reward model as the loss function for supervised learning?
2. What might happen if we trained with very high β (KL penalty)?
3. Why is it hard to collect perfect preference data?

### Coding (2-3 challenges)
1. Implement a simple reward model training loop
2. Compute KL divergence between two language models on sample prompts
3. Simulate reward hacking by optimizing for a flawed reward

### Exploration (1-2 open-ended)
1. How might we improve RLHF for tasks where humans struggle to evaluate quality?

---

## Subsection Breakdown

### Subsection 1: RL for AI Alignment
- The alignment problem: we can't specify what we want
- Why supervised learning isn't enough
- The RLHF insight: learn from preferences
- Brief history: from games to language models

### Subsection 2: Reward Modeling
- Collecting human preferences
- The Bradley-Terry model
- Training the reward model
- Limitations and biases
- Interactive: preference comparison demo

### Subsection 3: PPO for Language Models
- LLMs as policies
- The RLHF objective with KL penalty
- Why KL penalty matters (reward hacking)
- PPO updates for token generation
- Interactive: reward hacking demo

### Subsection 4: Current Frontiers
- DPO: eliminating the reward model
- Constitutional AI: self-improvement
- Open challenges: scalable oversight, reward specification
- The future of alignment
- Interactive: RLHF pipeline demo

---

## Additional Context for AI

- This is the capstone chapter. Connect all the threads.
- RLHF is why readers should care about RL—it powers ChatGPT.
- Make the alignment problem vivid: we can't write down what we want.
- The demos should make the preference → reward → policy pipeline tangible.
- Include DPO as the current alternative getting attention.
- End with open questions: this is an active research area.

---

## Quality Checklist

- [ ] Alignment problem motivated
- [ ] Three-stage pipeline explained
- [ ] Reward modeling with Bradley-Terry
- [ ] PPO application to LLMs
- [ ] KL penalty justified
- [ ] DPO covered as alternative
- [ ] Interactive demos specified
- [ ] Open challenges mentioned
