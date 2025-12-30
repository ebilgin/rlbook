# Chapter 02: Multi-Armed Bandits

## Chapter Metadata

**Chapter Number:** 02
**Title:** Multi-Armed Bandits
**Section:** Foundations
**Prerequisites:**
- Chapter 01: Introduction to RL (basic concepts, exploration-exploitation)
**Estimated Reading Time:** 25 minutes

---

## Learning Objectives

By the end of this chapter, readers will be able to:

1. Define the multi-armed bandit problem and its components
2. Explain why bandits are the simplest form of RL
3. Implement and compare action-value estimation methods
4. Apply ε-greedy, UCB, and Thompson Sampling strategies
5. Understand regret and how different algorithms compare

---

## Core Concepts to Cover

### Primary Concepts (Must Cover)
- [ ] The bandit problem: k arms, unknown reward distributions
- [ ] Action-value estimation: sample averages
- [ ] Greedy action selection and its failure
- [ ] ε-greedy exploration
- [ ] Upper Confidence Bound (UCB)
- [ ] Regret: measuring lost opportunity

### Secondary Concepts (Cover if Space Permits)
- [ ] Thompson Sampling (Bayesian approach)
- [ ] Optimistic initialization
- [ ] Non-stationary bandits and tracking

### Explicitly Out of Scope
- Contextual bandits (next chapter)
- Full MDP formulation
- Formal regret bounds and proofs

---

## Narrative Arc

### Opening Hook
"Imagine you're in a casino with 10 slot machines. Each has a different (unknown) payout rate. You have 1000 pulls. How do you maximize your winnings? This is the multi-armed bandit problem—and the strategies we learn here form the foundation of all exploration in RL."

### Key Insight
The bandit problem isolates the exploration-exploitation dilemma. With no states to worry about, no delayed rewards, no sequential dependencies—we can focus purely on learning which actions are best while also trying new things. Every algorithm we develop here will reappear in more complex settings.

### Closing Connection
"Bandits assume actions don't affect future states. But what if pulling an arm changed which arms were available, or changed the payouts? That's where contextual bandits and full MDPs come in. The exploration strategies you've learned here—ε-greedy, UCB—will transfer directly."

---

## Required Interactive Elements

### Demo 1: Bandit Playground
- **Purpose:** Let readers experiment with different strategies
- **Interaction:**
  - Set up k=10 arms with hidden reward distributions
  - Choose strategy: greedy, ε-greedy, UCB, Thompson
  - Watch cumulative reward and arm selection frequencies
  - Reveal true arm values after experiment
- **Expected Discovery:** Greedy gets stuck; exploration helps; UCB is systematic

### Demo 2: Regret Comparison
- **Purpose:** Visualize regret over time for different algorithms
- **Interaction:**
  - Run multiple strategies simultaneously
  - Plot cumulative regret (difference from always-optimal)
  - Show how regret grows (linear vs sublinear)
- **Expected Discovery:** Good exploration has sublinear regret; greedy is linear

### Demo 3: ε-Value Sensitivity
- **Purpose:** Show impact of exploration parameter
- **Interaction:**
  - Slider for ε from 0 to 1
  - See how performance changes
  - Identify sweet spot
- **Expected Discovery:** Too low = stuck, too high = wasted exploration

---

## Recurring Examples to Use

- **Slot machines:** The canonical example (k-armed bandit)
- **A/B testing:** Real-world application (website optimization)
- **Clinical trials:** Ethical exploration (adaptive trials)
- **Ad selection:** Industry application

---

## Cross-References

### Build On (Backward References)
- Chapter 01: "Recall the exploration-exploitation tradeoff..."
- Chapter 01: "Our goal is still to maximize cumulative reward..."

### Set Up (Forward References)
- Chapter 03: "When the best arm depends on context..."
- Chapter 12: "These same exploration strategies apply to Q-learning..."

---

## Mathematical Depth

### Required Equations
1. Sample average: $Q_n(a) = \frac{\sum_{i=1}^{n-1} R_i \cdot \mathbb{1}_{A_i=a}}{\sum_{i=1}^{n-1} \mathbb{1}_{A_i=a}}$
2. Incremental update: $Q_{n+1}(a) = Q_n(a) + \frac{1}{n}[R_n - Q_n(a)]$
3. UCB action selection: $A_t = \arg\max_a \left[ Q_t(a) + c\sqrt{\frac{\ln t}{N_t(a)}} \right]$
4. Regret definition: $L_T = \sum_{t=1}^{T} [\mu^* - \mu_{A_t}]$

### Derivations to Include (Mathematical Layer)
- Derive incremental update from sample average
- Explain UCB confidence bound intuition

### Proofs to Omit
- Formal regret bounds
- Convergence proofs

---

## Code Examples Needed

### Intuition Layer
```python
# ε-greedy selection
if random() < epsilon:
    action = random_arm()  # Explore
else:
    action = argmax(Q)     # Exploit
```

### Implementation Layer
- Complete Bandit class with k arms
- Action-value estimation with incremental update
- ε-greedy, UCB, Thompson Sampling agents
- Comparison experiment with plotting

---

## Common Misconceptions to Address

1. **"More exploration is always better"**: There's a cost. You sacrifice reward for information. The goal is efficient exploration.

2. **"UCB is always better than ε-greedy"**: UCB has better theoretical properties but ε-greedy is simpler and often competitive. Context matters.

3. **"Bandits are too simple to be useful"**: Bandits power recommendation systems, ad tech, and A/B testing at massive scale.

4. **"You need to know the true reward distributions"**: No! The whole point is learning them from experience.

---

## Exercises

### Conceptual (4 questions)
- Why does a greedy strategy fail in the bandit problem?
- What does the exploration bonus in UCB represent intuitively?
- How does Thompson Sampling balance exploration and exploitation?
- When would you prefer ε-greedy over UCB in practice?

### Coding (3 challenges)
- Implement a k-armed bandit testbed and run ε-greedy with different ε values
- Implement UCB and compare with ε-greedy on the same problem
- Modify ε-greedy to use decaying ε and measure the impact

### Exploration (1 open-ended)
- Design a non-stationary bandit where reward distributions change over time. How would you modify UCB to handle this?

---

## Additional Context for AI

- This is the first "real algorithm" chapter. Balance theory with implementation.
- The slot machine metaphor is perfect—use it consistently.
- Make incremental updates feel natural (you don't need to store all history).
- UCB's "optimism in the face of uncertainty" is a key concept that recurs.
- Thompson Sampling can be simplified: "sample from beliefs, act greedily."
- Connect to real applications: A/B testing is bandits, clinical trials are bandits.
- The interactive playground is critical—readers should feel these algorithms.

---

## Quality Checklist

- [ ] Bandit problem clearly defined
- [ ] Sample average and incremental update derived
- [ ] ε-greedy, UCB, Thompson all explained
- [ ] Interactive demo shows algorithm differences
- [ ] Regret concept introduced with visualization
- [ ] Real-world applications mentioned
- [ ] Code examples are complete and runnable
- [ ] Connection to full RL is clear (but bandits are simpler)
