# Chapter: Multi-Armed Bandits

## Chapter Metadata

**Chapter Number:** 09
**Title:** Multi-Armed Bandits
**Section:** Bandit Problems
**Prerequisites:**
- Foundations (especially exploration-exploitation)
**Estimated Reading Time:** 35 minutes

---

## Learning Objectives

By the end of this chapter, readers will be able to:

1. Formalize the multi-armed bandit problem
2. Implement and analyze ε-greedy exploration
3. Implement Upper Confidence Bound (UCB)
4. Implement Thompson Sampling
5. Compare exploration strategies empirically
6. Choose appropriate exploration methods for different settings

---

## Core Concepts to Cover

### Primary Concepts (Must Cover)
- [ ] The bandit problem: k arms, unknown rewards
- [ ] Action-value estimation
- [ ] Greedy and ε-greedy methods
- [ ] Optimistic initialization
- [ ] Upper Confidence Bound (UCB1)
- [ ] Thompson Sampling
- [ ] Regret as the evaluation metric

### Secondary Concepts (Cover if Space Permits)
- [ ] Non-stationary bandits
- [ ] Gradient bandit algorithms
- [ ] Bayesian approaches to bandits

### Explicitly Out of Scope
- Contextual bandits (next chapter)
- Full RL with states (TD Learning section)

---

## Narrative Arc

### Opening Hook
"You're at a casino with k slot machines. Each has a different (unknown) probability of paying out. You have limited pulls. How do you maximize your winnings? Welcome to the multi-armed bandit problem—the simplest RL setting, and yet remarkably rich."

### Key Insight
Every exploration strategy reflects a different belief about how to balance learning and earning. ε-greedy explores randomly, UCB explores where uncertainty is high, Thompson Sampling explores probabilistically based on beliefs. There's no universal best—it depends on your problem.

### Closing Connection
"Bandits gave us powerful exploration strategies. But what if the context matters? What if different users should get different recommendations? That's where contextual bandits come in—bandits with features."

---

## Required Interactive Elements

### Demo 1: Bandit Playground
- **Purpose:** Let users pull arms and see algorithms in action
- **Interaction:**
  - 10 arms with hidden probabilities
  - User can manually pull or let algorithms play
  - Show running reward, regret, action counts
  - Compare multiple algorithms simultaneously
- **Expected Discovery:** Different strategies have different tradeoffs

### Demo 2: Algorithm Race
- **Purpose:** Visual comparison of exploration strategies
- **Interaction:**
  - Line plots: cumulative reward over time
  - Line plots: regret over time
  - Action heatmap: which arms each algorithm pulls
  - Multiple runs for confidence intervals
- **Expected Discovery:** UCB and Thompson often beat ε-greedy

### Demo 3: UCB Intuition
- **Purpose:** Visualize the confidence bound idea
- **Interaction:**
  - Show estimated value ± confidence interval for each arm
  - Show how UCB chooses the arm with highest upper bound
  - Watch intervals shrink as we pull more
- **Expected Discovery:** UCB is "optimism in the face of uncertainty"

### Demo 4: Thompson Sampling Visualization
- **Purpose:** Show the Bayesian approach
- **Interaction:**
  - Show Beta distributions for each arm
  - Sample from each and pick the winner
  - Watch distributions sharpen over time
- **Expected Discovery:** Sampling naturally trades off exploration and exploitation

---

## Recurring Examples to Use

- **Casino slot machines:** The classic bandit metaphor
- **A/B testing:** Real-world application
- **Clinical trials:** Ethical dimension of exploration
- **Online advertising:** Maximize clicks

---

## Cross-References

### Build On (Backward References)
- Chapter 2 (Framework): "The exploration-exploitation tradeoff..."
- Foundations: "The slot machine demo from Chapter 2..."

### Set Up (Forward References)
- Chapter 10 (Contextual): "What if context matters?"
- TD Learning: "What if actions change the state?"
- DQN: "How does ε-greedy scale to deep RL?"

---

## Mathematical Depth

### Required Equations

1. **Action-value estimate**:
$$Q_t(a) = \frac{\text{sum of rewards from action } a}{\text{number of times } a \text{ taken}}$$

2. **Incremental update**:
$$Q_{n+1}(a) = Q_n(a) + \frac{1}{n}[R_n - Q_n(a)]$$

3. **ε-greedy action selection**:
$$A_t = \begin{cases} \arg\max_a Q_t(a) & \text{with prob } 1-\varepsilon \\ \text{random action} & \text{with prob } \varepsilon \end{cases}$$

4. **UCB1 action selection**:
$$A_t = \arg\max_a \left[ Q_t(a) + c\sqrt{\frac{\ln t}{N_t(a)}} \right]$$

5. **Thompson Sampling (Bernoulli)**:
$$\theta_a \sim \text{Beta}(\alpha_a, \beta_a), \quad A_t = \arg\max_a \theta_a$$

6. **Regret**:
$$\text{Regret}(T) = \sum_{t=1}^T [\mu^* - \mu_{A_t}]$$

### Derivations to Include (Mathematical Layer)
- UCB regret bound sketch (why log(t) regret)
- Beta distribution update rules
- Why incremental update is equivalent to sample mean

### Proofs to Omit
- Formal regret bound proofs
- Optimality proofs

---

## Code Examples Needed

### Intuition Layer
```python
def epsilon_greedy(Q, epsilon=0.1):
    """Select action using epsilon-greedy."""
    if random.random() < epsilon:
        return random.choice(range(len(Q)))  # Explore
    return np.argmax(Q)  # Exploit

def update_q(Q, N, action, reward):
    """Update action-value estimate."""
    N[action] += 1
    Q[action] += (reward - Q[action]) / N[action]
```

### Implementation Layer
```python
class UCB:
    def __init__(self, n_arms, c=2.0):
        self.n_arms = n_arms
        self.c = c
        self.Q = np.zeros(n_arms)
        self.N = np.zeros(n_arms)
        self.t = 0

    def select_action(self):
        self.t += 1
        # Handle unplayed arms first
        for a in range(self.n_arms):
            if self.N[a] == 0:
                return a
        # UCB selection
        ucb_values = self.Q + self.c * np.sqrt(np.log(self.t) / self.N)
        return np.argmax(ucb_values)

    def update(self, action, reward):
        self.N[action] += 1
        self.Q[action] += (reward - self.Q[action]) / self.N[action]
```

```python
class ThompsonSampling:
    def __init__(self, n_arms):
        self.n_arms = n_arms
        self.alpha = np.ones(n_arms)  # Successes + 1
        self.beta = np.ones(n_arms)   # Failures + 1

    def select_action(self):
        samples = np.random.beta(self.alpha, self.beta)
        return np.argmax(samples)

    def update(self, action, reward):
        if reward > 0:
            self.alpha[action] += 1
        else:
            self.beta[action] += 1
```

---

## Common Misconceptions to Address

1. **"ε-greedy with small ε is always best"**: UCB and Thompson often outperform for any fixed ε.

2. **"Exploration should decrease over time"**: Not always. ε-greedy with decay can get stuck; UCB adapts naturally.

3. **"Regret should be zero at the end"**: Regret accumulates. A good algorithm has sublinear regret (O(log T)).

4. **"The best arm should be pulled most often"**: Yes, but how quickly we find it matters for total regret.

---

## Exercises

### Conceptual (3-5 questions)
1. Why is pure greedy suboptimal?
2. What happens to UCB's exploration bonus as N_t(a) increases?
3. When would you prefer Thompson Sampling over UCB?

### Coding (2-3 challenges)
1. Implement all three strategies and compare on a 10-armed testbed
2. Plot regret curves for different values of ε
3. Implement a non-stationary bandit with random walk rewards

### Exploration (1-2 open-ended)
1. Design an A/B test using bandit algorithms. How is it better than fixed allocation?

---

## Subsection Breakdown

### Subsection 1: The Bandit Problem
- Setup: k arms, unknown reward distributions
- Goal: maximize cumulative reward
- The fundamental tension: explore vs exploit
- Action-value estimation and incremental updates
- Regret as the evaluation metric

### Subsection 2: Greedy and ε-Greedy Methods
- Pure greedy: always exploit, can get stuck
- ε-greedy: random exploration
- Choosing ε: tradeoffs
- Optimistic initialization: start high
- Interactive: see ε-greedy in action

### Subsection 3: Upper Confidence Bound
- The idea: optimism in the face of uncertainty
- UCB1 formula: value + confidence bonus
- Why it works: explores uncertain arms
- The exploration bonus shrinks as you learn
- Interactive: UCB visualization

### Subsection 4: Thompson Sampling
- The Bayesian approach: maintain beliefs
- Beta distributions for Bernoulli bandits
- Sample and pick the winner
- Natural exploration from uncertainty
- Interactive: Thompson Sampling visualization

### Subsection 5: Comparing Strategies
- When to use which method
- Empirical comparison on testbeds
- Regret bounds summary
- Practical recommendations
- Interactive: algorithm race

---

## Additional Context for AI

- This is the most "hands-on" chapter yet. Lots of code and interactive demos.
- The demos are crucial: let users feel the exploration-exploitation tension.
- Work through specific numerical examples.
- Include the classic 10-armed testbed from Sutton & Barto.
- Make regret intuitive: "how much did we leave on the table?"
- Thompson Sampling should feel magical when visualized correctly.

---

## Quality Checklist

- [ ] All three main strategies implemented
- [ ] Interactive bandit playground
- [ ] Algorithm comparison demo
- [ ] Regret curves shown
- [ ] UCB and Thompson Sampling visualized
- [ ] Real-world applications mentioned
- [ ] Clear progression from simple to sophisticated
