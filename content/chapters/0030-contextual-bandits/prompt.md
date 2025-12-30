# Chapter 03: Contextual Bandits

## Chapter Metadata

**Chapter Number:** 03
**Title:** Contextual Bandits
**Section:** Foundations
**Prerequisites:**
- Chapter 02: Multi-Armed Bandits (action-value estimation, exploration strategies)
**Estimated Reading Time:** 25 minutes

---

## Learning Objectives

By the end of this chapter, readers will be able to:

1. Explain how contextual bandits extend the basic bandit problem
2. Understand the role of context/features in action selection
3. Implement linear contextual bandits with ε-greedy exploration
4. Recognize real-world applications (recommendations, ads, personalization)
5. Appreciate the bridge between bandits and full RL

---

## Core Concepts to Cover

### Primary Concepts (Must Cover)
- [ ] Contextual bandits: actions depend on observable context
- [ ] Feature vectors and linear models for action values
- [ ] Learning a policy that maps context → action
- [ ] ε-greedy and UCB adapted for contextual setting
- [ ] The personalization use case

### Secondary Concepts (Cover if Space Permits)
- [ ] LinUCB algorithm
- [ ] Neural contextual bandits (preview)
- [ ] Off-policy evaluation

### Explicitly Out of Scope
- Full MDP with state transitions
- Deep learning approaches (brief mention only)
- Detailed off-policy learning

---

## Narrative Arc

### Opening Hook
"A movie recommendation system faces a unique challenge: the best movie to recommend depends on who's asking. A thriller fan and a comedy lover shouldn't get the same suggestion. This is a contextual bandit: the right action depends on context. And it's everywhere—personalized ads, news feeds, treatment selection, product recommendations."

### Key Insight
Contextual bandits add a crucial element: context. Instead of learning a single best action, we learn a policy—a mapping from context to action. This is a step toward full RL, where the context would be the state. But crucially, our actions don't affect future contexts. This makes learning easier while still capturing personalization.

### Closing Connection
"We've now covered the spectrum from no context (bandits) to static context (contextual bandits). The next step is when your actions affect future states—that's the full reinforcement learning problem with Markov Decision Processes. But the exploration strategies and value estimation ideas from bandits will carry forward."

---

## Required Interactive Elements

### Demo 1: Personalized Recommendations
- **Purpose:** Show how context changes optimal actions
- **Interaction:**
  - User profiles with different preferences (features)
  - See how recommended action changes based on user
  - Watch the algorithm learn personalized policies
- **Expected Discovery:** Same algorithm, but optimal action varies with context

### Demo 2: Linear Model Visualization
- **Purpose:** Visualize how features predict action values
- **Interaction:**
  - 2D context space
  - Show decision boundaries between actions
  - Watch boundaries update as data arrives
- **Expected Discovery:** The policy is a function of context, visualizable as regions

### Demo 3: Contextual vs Standard Bandit
- **Purpose:** Show the value of using context
- **Interaction:**
  - Run standard bandit (ignores context) vs contextual
  - Compare cumulative rewards
  - Show personalization advantage
- **Expected Discovery:** Context information leads to higher rewards

---

## Recurring Examples to Use

- **Movie recommendations:** User features → movie choice
- **News article selection:** Reader profile → article
- **Ad personalization:** User + page context → ad
- **Medical treatment:** Patient features → treatment

---

## Cross-References

### Build On (Backward References)
- Chapter 02: "Extend our bandit algorithms with context features..."
- Chapter 02: "The same exploration-exploitation tradeoff applies..."

### Set Up (Forward References)
- Chapter 05: "In MDPs, context becomes state, and actions affect it..."
- Chapter 13: "Deep neural networks can represent complex contextual policies..."

---

## Mathematical Depth

### Required Equations
1. Contextual action value: $Q(x, a) = \theta_a^T x$ (linear case)
2. ε-greedy policy: $a = \arg\max_a Q(x, a)$ with probability $1-\epsilon$
3. LinUCB: $a = \arg\max_a [\theta_a^T x + \alpha \sqrt{x^T A_a^{-1} x}]$

### Derivations to Include (Mathematical Layer)
- Show how linear model updates with new data
- Explain LinUCB confidence bound derivation (intuitive)

### Proofs to Omit
- Regret bounds for LinUCB
- Theoretical analysis of convergence

---

## Code Examples Needed

### Intuition Layer
```python
# Contextual bandit: action depends on user features
def select_action(context, epsilon=0.1):
    if random() < epsilon:
        return random_action()
    else:
        # Choose action with highest predicted value for this context
        values = [model[a].predict(context) for a in actions]
        return argmax(values)
```

### Implementation Layer
- LinearContextualBandit class with feature-based value estimation
- Online learning: update model after each interaction
- Comparison: contextual vs non-contextual on personalized task
- Visualization of learned policy

---

## Common Misconceptions to Address

1. **"Contextual bandits are just supervised learning"**: No! We only observe the reward for the action taken, not for other actions. This is bandit feedback, not full labels.

2. **"Context is the same as state"**: Similar, but in contextual bandits, our action doesn't change future contexts. In MDPs, actions affect states.

3. **"More features are always better"**: Feature engineering matters. Irrelevant features add noise and slow learning.

4. **"You need to retrain the model from scratch"**: No, online learning updates the model incrementally with each interaction.

---

## Exercises

### Conceptual (3 questions)
- How does bandit feedback differ from supervised learning labels?
- Why is personalization valuable in recommendation systems?
- What's the key difference between contextual bandits and full RL?

### Coding (3 challenges)
- Implement a linear contextual bandit for a movie recommendation scenario
- Compare performance: one model per action vs. shared model with action features
- Add LinUCB exploration and compare with ε-greedy

### Exploration (1 open-ended)
- Design a contextual bandit for a real application you care about. What features would you use? What actions? How would you measure reward?

---

## Additional Context for AI

- This chapter bridges bandits and full RL. Make the connection clear.
- Recommendation systems are the killer app—use them extensively.
- The "context" concept should feel natural: it's just features about the situation.
- Emphasize that actions don't affect future contexts (yet)—that's the key simplification.
- Linear models are sufficient for teaching; mention neural approaches exist.
- The interactive personalization demo is critical for intuition.
- LinUCB is elegant but optional—focus on ε-greedy with linear models first.

---

## Quality Checklist

- [ ] Contextual bandit problem clearly defined
- [ ] Distinction from standard bandits and full RL is clear
- [ ] Linear model for action values explained
- [ ] At least 2 real-world applications discussed
- [ ] Interactive demo shows personalization in action
- [ ] Code examples are complete and runnable
- [ ] Clear setup for MDP chapter coming next
