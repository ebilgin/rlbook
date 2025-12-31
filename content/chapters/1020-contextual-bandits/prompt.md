# Chapter: Contextual Bandits

## Chapter Metadata

**Chapter Number:** 7
**Title:** Contextual Bandits
**Section:** Bandit Problems
**Prerequisites:**
- Multi-Armed Bandits
**Estimated Reading Time:** 30 minutes

---

## Learning Objectives

By the end of this chapter, readers will be able to:

1. Explain how contextual bandits extend multi-armed bandits
2. Formalize the contextual bandit problem mathematically
3. Implement LinUCB for linear reward models
4. Describe how neural networks extend contextual bandits
5. Identify real-world applications (recommendation, ads, news)
6. Understand the bridge from bandits to full RL

---

## Core Concepts to Cover

### Primary Concepts (Must Cover)
- [ ] From bandits to contextual bandits: adding features/context
- [ ] The contextual bandit formalism
- [ ] Linear reward models
- [ ] LinUCB algorithm
- [ ] Exploration in high-dimensional spaces
- [ ] Applications: personalized recommendations

### Secondary Concepts (Cover if Space Permits)
- [ ] Neural contextual bandits
- [ ] Thompson Sampling with linear models
- [ ] Offline evaluation (counterfactual estimation)
- [ ] Multi-objective bandits

### Explicitly Out of Scope
- Full MDP formulation (next section)
- Deep RL approaches
- Batch/offline bandit learning in depth

---

## Narrative Arc

### Opening Hook
"A news website needs to decide which headline to show each visitor. But here's the catch: different people like different things. A sports fan wants game scores; a tech enthusiast wants startup news. This isn't just about finding the best arm—it's about finding the best arm *for each person*."

### Key Insight
Contextual bandits bridge the gap between simple bandits (one best arm for everyone) and full RL (sequential decisions). By conditioning on context (user features, time of day, etc.), we can personalize decisions while still benefiting from the bandit framework's simplicity.

### Closing Connection
"We've now mastered single-step decision problems. But what if your action affects not just your immediate reward, but also where you end up next? That's sequential decision-making, and it requires a new mathematical framework: Markov Decision Processes."

---

## Required Interactive Elements

### Demo 1: Contextual vs Non-Contextual
- **Purpose:** Show why context matters
- **Interaction:**
  - Generate users with visible features (e.g., age, interests)
  - Run standard bandit (one arm for all) vs contextual (personalized)
  - Show per-user regret and overall performance
  - Visualize how context-aware algorithm adapts
- **Expected Discovery:** Ignoring context leaves reward on the table

### Demo 2: LinUCB Explorer
- **Purpose:** Visualize linear reward models and exploration
- **Interaction:**
  - 2D feature space with colored arms
  - Show linear decision boundaries
  - Visualize confidence ellipsoids
  - Watch boundaries update as data arrives
- **Expected Discovery:** LinUCB explores where it's uncertain about the model

### Demo 3: News Recommendation Simulator
- **Purpose:** Real-world application demo
- **Interaction:**
  - Simulated users with profiles arrive
  - Algorithm must choose article to show
  - User clicks or doesn't (reward signal)
  - Track CTR improvement over time
- **Expected Discovery:** Personalization significantly improves engagement

---

## Recurring Examples to Use

- **News recommendation:** The canonical contextual bandit application
- **Ad selection:** Different users respond to different ads
- **Medical treatment:** Patient features → treatment selection
- **A/B testing evolution:** From uniform to personalized experiments

---

## Cross-References

### Build On (Backward References)
- Multi-Armed Bandits: "We extend bandits with context..."
- UCB: "LinUCB applies the UCB principle to linear models..."
- Thompson Sampling: "Can also use Thompson Sampling with linear posteriors..."

### Set Up (Forward References)
- MDPs: "When actions affect future states, we need MDPs..."
- Function Approximation: "Linear models are a simple form of function approximation..."
- Policy Gradients: "Contextual bandits are one-step policy learning..."

---

## Mathematical Depth

### Required Equations

1. **Contextual bandit formalism**:
$$r_t = f_a(x_t) + \epsilon_t$$
where $x_t$ is context, $a$ is action, $f_a$ is the reward function for arm $a$.

2. **Linear reward model**:
$$\mathbb{E}[r_t | x_t, a] = x_t^\top \theta_a$$

3. **LinUCB arm selection**:
$$a_t = \arg\max_a \left( x_t^\top \hat{\theta}_a + \alpha \sqrt{x_t^\top A_a^{-1} x_t} \right)$$

4. **Ridge regression update**:
$$A_a \leftarrow A_a + x_t x_t^\top, \quad b_a \leftarrow b_a + r_t x_t$$
$$\hat{\theta}_a = A_a^{-1} b_a$$

### Derivations to Include (Mathematical Layer)
- Why ridge regression gives uncertainty estimates
- Connection between LinUCB and UCB
- Confidence ellipsoid interpretation

### Proofs to Omit
- Regret bounds for LinUCB
- Convergence analysis

---

## Code Examples Needed

### Intuition Layer
```python
def linucb_select(context, arms, alpha=1.0):
    """Select arm using LinUCB."""
    best_arm, best_ucb = None, -float('inf')

    for arm in arms:
        # Estimated reward
        mean = context @ arm.theta
        # Exploration bonus (uncertainty)
        bonus = alpha * np.sqrt(context @ arm.A_inv @ context)
        ucb = mean + bonus

        if ucb > best_ucb:
            best_arm, best_ucb = arm, ucb

    return best_arm
```

### Implementation Layer
```python
class LinUCBArm:
    def __init__(self, d, alpha=1.0):
        self.d = d
        self.alpha = alpha
        self.A = np.eye(d)  # d x d matrix
        self.b = np.zeros(d)  # d-dimensional vector
        self.theta = np.zeros(d)

    def update(self, context, reward):
        """Update model with new observation."""
        self.A += np.outer(context, context)
        self.b += reward * context
        self.theta = np.linalg.solve(self.A, self.b)

    def get_ucb(self, context):
        """Compute UCB for this arm given context."""
        A_inv = np.linalg.inv(self.A)
        mean = context @ self.theta
        std = np.sqrt(context @ A_inv @ context)
        return mean + self.alpha * std


class LinUCB:
    def __init__(self, n_arms, d, alpha=1.0):
        self.arms = [LinUCBArm(d, alpha) for _ in range(n_arms)]

    def select(self, context):
        """Select arm with highest UCB."""
        ucbs = [arm.get_ucb(context) for arm in self.arms]
        return np.argmax(ucbs)

    def update(self, context, arm_idx, reward):
        """Update the selected arm."""
        self.arms[arm_idx].update(context, reward)
```

---

## Common Misconceptions to Address

1. **"Context = state"**: Not quite. In contextual bandits, context is observed before action but doesn't change due to action. States in MDPs evolve.

2. **"Just use the best arm per user segment"**: Discretizing context loses information. Linear models interpolate smoothly.

3. **"More features always help"**: High-dimensional features need regularization and more data. Feature engineering matters.

4. **"Contextual bandits solve recommendation"**: They're one-step. If recommendations affect future preferences (filter bubbles), you need full RL.

---

## Exercises

### Conceptual (3-5 questions)
1. Why can't we just run separate MAB for each unique context?
2. What does the exploration bonus in LinUCB represent geometrically?
3. When would contextual bandits be preferred over full RL for recommendations?

### Coding (2-3 challenges)
1. Implement LinUCB and test on a synthetic problem with known linear rewards
2. Compare LinUCB to ε-greedy with per-arm feature learning
3. Implement disjoint LinUCB (separate models per arm) vs hybrid (shared + arm-specific)

### Exploration (1-2 open-ended)
1. Design a contextual bandit for a problem you care about. What's the context? What are the arms? How would you measure success?

---

## Subsection Breakdown

### Subsection 1: From Bandits to Decisions
- The limitation of one-size-fits-all
- Introducing context (features)
- The contextual bandit formalism
- Interactive: contextual vs non-contextual comparison

### Subsection 2: LinUCB
- Linear reward models
- Ridge regression for reward estimation
- The LinUCB algorithm
- Confidence ellipsoids and exploration
- Interactive: LinUCB explorer

### Subsection 3: Applications and Extensions
- News recommendation (the Yahoo! story)
- Ad selection
- Neural contextual bandits
- Limitations: when you need full RL
- Interactive: news recommendation simulator

---

## Additional Context for AI

- This is the bridge between bandits and MDPs.
- Make the Yahoo! news recommendation story vivid—it's the canonical example.
- LinUCB should be explained intuitively before the math.
- Confidence ellipsoids can be visualized in 2D for intuition.
- The "what if actions affect future context?" question sets up MDPs perfectly.
- Don't overcomplicate—this is still simpler than full RL.

---

## Quality Checklist

- [ ] Context vs state distinction clear
- [ ] LinUCB algorithm explained step by step
- [ ] Confidence ellipsoid intuition provided
- [ ] Real-world applications vivid
- [ ] Connection to MDPs/RL foreshadowed
- [ ] Interactive demos specified
- [ ] Code runnable and clear
