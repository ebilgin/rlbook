# Chapter: RL for Language Models

## Chapter Metadata

**Title:** RL for Language Models
**Slug:** rl-for-llms
**Section:** Advanced Topics
**Prerequisites:**
- PPO (Chapter 2040)
- (Recommended) Introduction to Policy Gradients (Chapter 2010)
- (Recommended) REINFORCE (Chapter 2020)
**Estimated Reading Time:** 60 minutes

---

## Learning Objectives

By the end of this chapter, readers will be able to:

1. Explain the two motivations for RL in LLM post-training: alignment and reasoning
2. Describe how reward models learn from human preferences (Bradley-Terry model)
3. Compare PPO, DPO, and GRPO for LLM training — their tradeoffs and when to use each
4. Explain how GRPO enables reasoning through verifiable rewards (RLVR)
5. Walk through a real GRPO implementation (Karpathy's nanochat)
6. Identify open challenges: reward hacking, mode collapse, alignment tax, scalable oversight

---

## Core Concepts to Cover

### Primary Concepts (Must Cover)
- [ ] Two motivations: alignment (teaching values) AND reasoning (developing capabilities)
- [ ] The discriminator-generator gap (why preferences work)
- [ ] Reward modeling from human preferences (Bradley-Terry)
- [ ] PPO for language models (token-level decisions, KL penalty)
- [ ] DPO: Direct Preference Optimization (eliminating the reward model)
- [ ] GRPO: Group Relative Policy Optimization (eliminating the critic)
- [ ] RLVR: Reinforcement Learning from Verifiable Rewards
- [ ] DeepSeek R1: emergence of reasoning through RL
- [ ] Karpathy's nanochat GRPO implementation
- [ ] Reward hacking and mitigation

### Secondary Concepts (Cover)
- [ ] SimPO, KTO, ORPO — the modern landscape
- [ ] Constitutional AI / RLAIF
- [ ] Process vs. outcome reward models
- [ ] Mode collapse and alignment tax
- [ ] Scalable oversight

### Explicitly Out of Scope
- Detailed LLM architecture (transformer internals)
- Pre-training details (tokenization, data curation)
- Deployment infrastructure

---

## Narrative Arc

### Opening Hook
"In 2022, ChatGPT turned language models from autocomplete engines into conversational partners. The secret ingredient? Reinforcement learning. Three years later, RL did something even more remarkable — it taught models to *reason*. This chapter tells both stories."

### Key Insight
RL for LLMs has evolved through two revolutions. The first (RLHF) taught models what humans want by learning from preferences. The second (RLVR/GRPO) taught models to reason by training against verifiable rewards. Both use the same core idea — policy gradients — but apply them in fundamentally different ways.

### Closing Connection
"The techniques in this chapter — from reward modeling to GRPO — represent the cutting edge of how we shape AI behavior. But the hardest problems remain unsolved: How do we align systems smarter than us? How do we verify reasoning we can't check? The tools you've learned here are the foundation for tackling those questions."

---

## Required Interactive Elements

### Demo 1: Training Stage Pipeline (index.mdx)
- **Component:** `TrainingStagePipeline.tsx`
- **Purpose:** Show how each training stage transforms model behavior
- **Interaction:**
  - Horizontal pipeline: Pretraining → SFT → RL
  - Click each stage to see sample responses
  - Metrics bar: helpfulness, safety, reasoning accuracy
  - Before/after comparison at each transition
- **Expected Discovery:** Each stage adds a different capability; RL is what makes models aligned and capable reasoners

### Demo 2: Preference Labeler (reward-modeling.mdx)
- **Component:** `PreferenceLabeler.tsx`
- **Purpose:** Experience preference collection firsthand
- **Interaction:**
  - Display a prompt with two AI responses side-by-side
  - User clicks preferred response (A, B, or Tie)
  - Accumulated preferences shown as growing dataset
  - Mini reward model "trains" on collected preferences
  - Shows learned reward scores on new unseen response pairs
- **Expected Discovery:** Ranking is easier than rating; a few dozen preferences are enough to learn patterns

### Demo 3: Algorithm Explorer (rl-algorithms-for-llms.mdx)
- **Component:** `AlgorithmExplorer.tsx`
- **Purpose:** Compare PPO, DPO, and GRPO visually
- **Interaction:**
  - Three algorithm cards with pipeline diagrams
  - Model size selector (1B, 7B, 13B, 70B)
  - Memory bars showing GPU requirements per algorithm
  - GPU VRAM markers (12GB, 24GB, 40GB, 80GB)
  - Comparison table: models needed, data type, online/offline
- **Expected Discovery:** PPO needs 4 models in memory; DPO needs 2; GRPO needs 2 but is online

### Demo 4: GRPO Explorer (grpo-and-reasoning.mdx)
- **Component:** `GRPOExplorer.tsx`
- **Purpose:** Visualize how group-relative advantages work
- **Interaction:**
  - Math problem displayed at top
  - G=8 sample completions with rewards (correct=1.0, wrong=0.0)
  - Step-by-step advantage calculation
  - Slider: adjust one reward → watch ALL advantages change (relativity)
  - Group size selector (4, 8, 16, 32)
  - Policy update visualization: probability bars shifting
- **Expected Discovery:** Advantages are relative — the same completion can be "good" or "bad" depending on its group

### Demo 5: Reward Hacking Demo (challenges.mdx)
- **Component:** `RewardHackingDemo.tsx`
- **Purpose:** Show why unconstrained optimization is dangerous
- **Interaction:**
  - Simple optimization scenario with proxy and true reward
  - Toggle: KL penalty ON/OFF
  - Without KL: proxy reward climbs but true quality degrades
  - With KL: moderate improvement maintained
  - Two diverging metrics visualized over training steps
- **Expected Discovery:** Optimizing a proxy too hard makes things worse; KL penalty is the safety net

---

## Recurring Examples to Use

- **Math reasoning tasks:** "What is 23 × 17?" — ideal for GRPO/RLVR because verifiable
- **ChatGPT-style responses:** Helpfulness comparison (detailed vs. terse)
- **Harmlessness:** Refusing dangerous requests
- **GSM8K problems:** The canonical GRPO benchmark (used in nanochat)
- **Code generation:** Verifiable via execution

---

## Cross-References

### Build On (Backward References)
- REINFORCE (Chapter 2020): "GRPO is essentially REINFORCE with group-relative baselines"
- PPO (Chapter 2040): "PPO's clipped objective carries over, but the critic doesn't"
- Policy Gradients (Chapter 2010): "LLMs are policies over tokens"
- Actor-Critic (Chapter 2030): "The advantage function we learned there — GRPO replaces the critic"
- GRPO Paper: "For the full mathematical derivation, see the GRPO paper deep-dive"

### Set Up (Forward References)
- AI safety research
- Scalable oversight
- Process reward models

---

## Mathematical Depth

### Required Equations

1. **Reward model training (Bradley-Terry)**:
$$P(y_w \succ y_l \mid x) = \sigma(r_\phi(x, y_w) - r_\phi(x, y_l))$$

2. **RLHF objective with KL penalty**:
$$\max_\theta \mathbb{E}_{x, y \sim \pi_\theta} \left[ r_\phi(x, y) - \beta \cdot D_{\text{KL}}(\pi_\theta \| \pi_{\text{ref}}) \right]$$

3. **DPO loss**:
$$L_{\text{DPO}} = -\mathbb{E} \left[ \log \sigma \left( \beta \log \frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)} \right) \right]$$

4. **GRPO group-relative advantage**:
$$\hat{A}_i = \frac{r_i - \mu_G}{\sigma_G + \epsilon}$$

5. **GRPO objective**:
$$L_{\text{GRPO}} = -\frac{1}{G} \sum_{i=1}^{G} \min\left( \rho_i \hat{A}_i, \text{clip}(\rho_i, 1-\epsilon, 1+\epsilon) \hat{A}_i \right) + \beta D_{\text{KL}}$$

### Derivations to Include (Mathematical Layer)
- Bradley-Terry from maximum likelihood
- DPO derivation from RLHF objective (closed-form optimal policy)
- GRPO connection to REINFORCE with baseline
- Why group normalization replaces the critic

### Proofs to Omit
- Convergence guarantees
- Sample complexity bounds

---

## Karpathy References

| Resource | Where Used | How |
|----------|-----------|-----|
| [nanochat](https://github.com/karpathy/nanochat) `scripts/chat_rl.py` | hands-on.mdx | Annotated walkthrough with attribution |
| ["RLHF is Just Barely RL"](https://x.com/karpathy/status/1821277264996352246) | why-rl-for-llms.mdx | Conceptual framing: proxy vs verifiable |
| [2025 Year in Review](https://karpathy.bearblog.dev/year-in-review-2025/) | grpo-and-reasoning.mdx | RLVR as dominant paradigm |
| ["Deep Dive into LLMs"](https://www.youtube.com/watch?v=7xTGNNLPyMI) video | why-rl-for-llms.mdx | Recommended resource |
| ["Pong from Pixels"](http://karpathy.github.io/2016/05/31/rl/) | hands-on.mdx | Policy gradients → LLMs connection |
| [DeepSeek R1 analysis](https://x.com/karpathy/status/1883941452738355376) | grpo-and-reasoning.mdx | Expert commentary |

---

## Subsection Breakdown

### Subsection 1: Why RL for Language Models? (`why-rl-for-llms.mdx`)
- Two motivations: alignment (ChatGPT, 2022) and reasoning (DeepSeek R1, 2025)
- The alignment problem: we can't write a loss function for "helpful"
- The discriminator-generator gap (Karpathy's insight: ranking is easier than generating)
- The LLM as an RL agent: state = context, action = next token, policy = the model
- Three training stages: pretraining → SFT → RL (what each adds)
- Timeline: InstructGPT → ChatGPT → DPO → GRPO → DeepSeek R1
- "RLHF is just barely RL" — Karpathy's framing
- Transition: proxy rewards (RLHF) vs. verifiable rewards (RLVR)

### Subsection 2: Reward Modeling (`reward-modeling.mdx`)
- Why pairwise comparisons > absolute ratings
- Preference data collection (who labels, how, quality control)
- The Bradley-Terry probabilistic model
- Reward model architecture: language model base + scalar head
- Training: binary cross-entropy on preference pairs
- Reward model biases: length bias, sycophancy, format preference
- Probing for biases (practical diagnostic techniques)
- Improving reward models: ensembles, regularization, iterative refinement
- Interactive: PreferenceLabeler demo

### Subsection 3: RL Algorithms for LLMs (`rl-algorithms-for-llms.mdx`)
- PPO for LLMs: token-level decisions, KL penalty, per-token reward distribution
- The critic problem: 4 models in memory (policy, reference, critic, reward model)
- Value function for credit assignment in text generation
- DPO: eliminating the reward model (implicit reward via log-ratio)
- DPO derivation: closed-form solution to RLHF objective
- When DPO works well (sufficient offline data) vs. when it struggles (distribution shift)
- GRPO: eliminating the critic (group-relative advantages)
- GRPO connection to REINFORCE with baseline
- Brief coverage: SimPO, KTO, ORPO
- Algorithm selection guide: online vs offline, paired vs unpaired, verifiable vs subjective
- Interactive: AlgorithmExplorer demo

### Subsection 4: GRPO and the Reasoning Revolution (`grpo-and-reasoning.mdx`)
- RLVR: replacing learned rewards with verifiable ones
- GRPO algorithm step by step: sample → reward → normalize → clip → update
- Mathematical formulation and connection to REINFORCE
- DeepSeek R1 case study: the 4-stage pipeline
- R1-Zero: reasoning from pure RL (no SFT)
- The "aha moment" — emergent self-reflection and backtracking
- The debate: does RL create new capabilities or amplify latent ones?
- Process vs. outcome reward models (OpenAI's "Let's Verify Step by Step")
- Test-time compute: a new scaling dimension
- Interactive: GRPOExplorer demo

### Subsection 5: Building a Reasoning Model (`hands-on.mdx`)
- Karpathy's nanochat walkthrough: annotated `scripts/chat_rl.py`
  - Attribution and link to github.com/karpathy/nanochat
  - The simplified GRPO: no trust region, on-policy, GAPO-style normalization
  - "It's basically REINFORCE with group baselines"
  - Step-by-step: prompt sampling → completion generation → reward computation → advantage normalization → policy update
  - Expected results: GSM8K 60% → 75%
- Our implementation: simplified GRPO in code/rlbook/agents/grpo.py
- Connection to REINFORCE (Chapter 2020) and PPO (Chapter 2040)
- Exercises: modify group size, try different reward functions, compare with/without KL

### Subsection 6: Challenges and Frontiers (`challenges.mdx`)
- Reward hacking: real examples (METR 2025 — o3 replacing chess engines, modifying test scripts)
- Mode collapse: diversity loss during RL training
- The alignment tax: capability degradation from safety training
- Constitutional AI / RLAIF: Anthropic's approach, self-critique and revision
- Scalable oversight: what happens when models surpass human evaluators?
- What's next: process reward models, debate, interpretability
- Interactive: RewardHackingDemo

---

## Common Misconceptions to Address

1. **"RLHF makes models truthful"**: RLHF optimizes for preference, which correlates with truth but isn't truth.
2. **"The reward model perfectly captures human values"**: Reward models are imperfect proxies that can be exploited.
3. **"RLHF is why models refuse harmful requests"**: Mostly SFT and system prompts; RLHF refines behavior.
4. **"DPO replaced PPO"**: DPO is simpler but PPO/GRPO are superior for online learning and reasoning tasks.
5. **"GRPO is just REINFORCE"**: GRPO adds clipping, group normalization, and KL penalty — meaningful improvements.
6. **"Reasoning emerges magically from RL"**: Evidence suggests RL amplifies patterns already present in pretraining data.

---

## Exercises

### Conceptual
1. Why can't we use the reward model as a supervised learning loss? (Hint: reward hacking)
2. What happens with very high β (KL penalty)? Very low β?
3. Why does GRPO need multiple completions per prompt while PPO needs only one?
4. In what scenarios would DPO outperform GRPO? Vice versa?
5. Why is verifiable reward (math, code) more reliable than learned reward (helpfulness)?

### Coding
1. Implement a simple reward model training loop with Bradley-Terry loss
2. Implement simplified GRPO: sample completions, compute group advantages, update policy
3. Simulate reward hacking: optimize against a proxy reward and measure true quality degradation

### Exploration
1. How might we align systems that are smarter than human evaluators?
2. Design a reward function for a task where automated verification is impossible

---

## Quality Checklist

- [ ] Two motivations (alignment + reasoning) clearly distinguished
- [ ] Reward modeling with Bradley-Terry fully explained
- [ ] PPO, DPO, and GRPO compared with clear guidance on when to use each
- [ ] GRPO algorithm explained step-by-step with connection to REINFORCE
- [ ] DeepSeek R1 case study with 4-stage pipeline
- [ ] Karpathy's nanochat implementation walked through with attribution
- [ ] 5 interactive demos built and placed
- [ ] All three complexity layers used in every subsection
- [ ] Cross-references to policy gradient chapters
- [ ] Reward hacking demonstrated with real examples
- [ ] Exercises included (conceptual, coding, exploration)
- [ ] `npm run build` passes

---

## Iteration Notes

### Initial Design (2026-02-14)
- Expanded from 4 subsections to 6 to cover GRPO, RLVR, and hands-on implementation
- Added Karpathy's nanochat as primary code reference
- Designed 5 interactive components (up from 4 specified, 0 built)
- Restructured to tell the dual story: alignment (2022) + reasoning (2025)
- Changed slug from 'rlhf' to 'rl-for-llms' to reflect broader scope
