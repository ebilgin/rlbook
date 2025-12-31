# Chapter: Q-Learning

## Chapter Metadata

**Chapter Number:** 13
**Title:** Q-Learning: Off-Policy TD Control
**Section:** Temporal Difference Learning
**Prerequisites:**
- SARSA
**Estimated Reading Time:** 30 minutes

---

## Learning Objectives

By the end of this chapter, readers will be able to:

1. Explain the key difference between SARSA and Q-learning
2. Implement Q-learning for control
3. Explain what "off-policy" means and why it matters
4. Demonstrate Q-learning on Cliff Walking
5. Identify the "deadly triad" and its implications

---

## Core Concepts to Cover

### Primary Concepts (Must Cover)
- [ ] The Q-learning update: uses max instead of a'
- [ ] Off-policy learning: behavior vs target policy
- [ ] Learning Q* directly
- [ ] Exploration during Q-learning
- [ ] SARSA vs Q-learning: the Cliff Walking contrast
- [ ] The deadly triad preview

### Secondary Concepts (Cover if Space Permits)
- [ ] Double Q-learning (tabular version)
- [ ] Maximization bias
- [ ] Convergence conditions

### Explicitly Out of Scope
- DQN (next section)
- Experience replay
- Target networks

---

## Narrative Arc

### Opening Hook
"SARSA is safe but learns the value of being ε-greedy. What if we could learn the optimal policy while still exploring? Q-learning does exactly this—it learns Q* regardless of what behavior policy we follow. It's perhaps the most important algorithm in RL."

### Key Insight
Q-learning's trick is using max over next actions instead of the actual next action. This means we're always learning toward the optimal policy, even while behaving suboptimally. The behavior policy can be anything—as long as it tries all actions eventually.

### Closing Connection
"Q-learning gives us Q*. But in its tabular form, it's limited to small state spaces. What happens when we have millions of states—or continuous states? That's where function approximation comes in, and where things get interesting... and dangerous."

---

## Required Interactive Elements

### Demo 1: Q-learning Training
- **Purpose:** Watch Q-learning find optimal values
- **Interaction:**
  - Same GridWorld as SARSA
  - Train Q-learning in real-time
  - Show Q-values and derived policy
  - Compare final policy to SARSA
- **Expected Discovery:** Q-learning finds the optimal policy; SARSA's is more conservative

### Demo 2: SARSA vs Q-learning on Cliff Walking
- **Purpose:** THE defining comparison
- **Interaction:**
  - Run both algorithms side by side
  - Show paths taken during and after learning
  - Show reward curves
  - Q-learning: risky optimal path; SARSA: safe path
- **Expected Discovery:** Q-learning finds optimal but is risky during learning; SARSA is safer

### Demo 3: Off-Policy Visualization
- **Purpose:** Make off-policy learning intuitive
- **Interaction:**
  - Show behavior policy (random or ε-greedy)
  - Show target policy (greedy w.r.t. Q)
  - Highlight that updates target the greedy policy
  - Show how Q-learning learns Q* despite exploration
- **Expected Discovery:** The behavior and target policies are different

### Demo 4: Maximization Bias
- **Purpose:** Show why Double Q-learning helps
- **Interaction:**
  - MDP where maximization bias is problematic
  - Compare Q-learning vs Double Q-learning
  - Show how Q-learning overestimates
- **Expected Discovery:** Max can cause overestimation; Double Q-learning fixes it

---

## Recurring Examples to Use

- **GridWorld:** Standard environment
- **Cliff Walking:** SARSA vs Q-learning showcase
- **Maximization Bias MDP:** From Sutton & Barto

---

## Cross-References

### Build On (Backward References)
- SARSA: "Replace a' with max..."
- TD(0): "Still using TD targets..."
- Value Functions: "Learning Q* directly..."

### Set Up (Forward References)
- Function Approximation: "Tables don't scale..."
- DQN: "Q-learning with neural networks..."
- Deadly Triad: "What can go wrong..."

---

## Mathematical Depth

### Required Equations

1. **Q-learning update**:
$$Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \alpha \left[ R_{t+1} + \gamma \max_{a'} Q(S_{t+1}, a') - Q(S_t, A_t) \right]$$

2. **TD error for Q-learning**:
$$\delta_t = R_{t+1} + \gamma \max_{a'} Q(S_{t+1}, a') - Q(S_t, A_t)$$

3. **Optimal policy from Q***:
$$\pi^*(s) = \arg\max_a Q^*(s, a)$$

4. **Double Q-learning** (to address maximization bias):
$$Q_1(S, A) \leftarrow Q_1(S, A) + \alpha \left[ R + \gamma Q_2(S', \arg\max_a Q_1(S', a)) - Q_1(S, A) \right]$$

### Derivations to Include (Mathematical Layer)
- Why Q-learning converges to Q* (sketch)
- Maximization bias explanation
- The deadly triad conditions

### Proofs to Omit
- Formal convergence proof
- Complexity analysis

---

## Code Examples Needed

### Intuition Layer
```python
def q_learning_update(Q, s, a, r, s_next, alpha=0.1, gamma=0.99):
    """Single Q-learning update."""
    # The key difference from SARSA: max over all actions
    td_target = r + gamma * np.max(Q[s_next])
    td_error = td_target - Q[s, a]
    Q[s, a] = Q[s, a] + alpha * td_error
    return td_error
```

### Implementation Layer
```python
def q_learning_episode(env, Q, epsilon=0.1, alpha=0.1, gamma=0.99):
    """Run one episode of Q-learning."""
    state = env.reset()
    total_reward = 0
    done = False

    while not done:
        # Behavior policy: ε-greedy
        action = epsilon_greedy(Q[state], epsilon)

        next_state, reward, done = env.step(action)
        total_reward += reward

        # Q-learning update: uses max, not actual next action
        if done:
            td_target = reward
        else:
            td_target = reward + gamma * np.max(Q[next_state])

        Q[state, action] += alpha * (td_target - Q[state, action])
        state = next_state

    return total_reward
```

```python
class DoubleQLearning:
    def __init__(self, n_states, n_actions, alpha=0.1, gamma=0.99):
        self.Q1 = np.zeros((n_states, n_actions))
        self.Q2 = np.zeros((n_states, n_actions))
        self.alpha = alpha
        self.gamma = gamma

    def update(self, s, a, r, s_next, done):
        if np.random.random() < 0.5:
            # Update Q1 using Q2 for evaluation
            a_star = np.argmax(self.Q1[s_next])
            target = r if done else r + self.gamma * self.Q2[s_next, a_star]
            self.Q1[s, a] += self.alpha * (target - self.Q1[s, a])
        else:
            # Update Q2 using Q1 for evaluation
            a_star = np.argmax(self.Q2[s_next])
            target = r if done else r + self.gamma * self.Q1[s_next, a_star]
            self.Q2[s, a] += self.alpha * (target - self.Q2[s, a])

    def get_action(self, s, epsilon):
        Q_sum = self.Q1[s] + self.Q2[s]
        return epsilon_greedy(Q_sum, epsilon)
```

---

## Common Misconceptions to Address

1. **"Q-learning is always better than SARSA"**: Not in risky environments! SARSA's conservatism can be valuable.

2. **"Off-policy means we can ignore exploration"**: We still need exploration! But the update targets optimality.

3. **"Q-learning can learn from any data"**: Yes, but convergence requires adequate coverage of state-action pairs.

4. **"Max is the same as the actual next action for a greedy policy"**: Only if we follow a greedy policy. With ε-greedy, they differ.

---

## Exercises

### Conceptual (3-5 questions)
1. What's the one-word difference between SARSA and Q-learning updates?
2. Why does Q-learning learn Q* instead of Q^π?
3. Can Q-learning learn from data collected by a random policy?

### Coding (2-3 challenges)
1. Implement Q-learning and compare with SARSA on Cliff Walking
2. Implement Double Q-learning
3. Create a visualization of Q-values during learning

### Exploration (1-2 open-ended)
1. Think of a scenario where Q-learning's optimism could be dangerous in the real world.

---

## Subsection Breakdown

### Subsection 1: The Q-Learning Idea
- From SARSA: what if we used max instead of a'?
- Learning Q* directly
- The target policy is greedy; the behavior policy explores
- Why this is profound: optimal learning from suboptimal behavior

### Subsection 2: The Q-Learning Algorithm
- The update rule explained
- Why max makes it off-policy
- Full algorithm with pseudocode
- ε-greedy for exploration
- Interactive: Q-learning training demo

### Subsection 3: SARSA vs Q-Learning
- The Cliff Walking experiment
- SARSA: safe path (accounts for ε)
- Q-learning: optimal path (ignores ε in target)
- When each is preferred
- Interactive: side-by-side comparison

### Subsection 4: Convergence and the Deadly Triad
- When Q-learning converges: tabular, adequate exploration, decaying α
- The deadly triad: off-policy + function approximation + bootstrapping
- Why this matters for DQN
- Maximization bias and Double Q-learning
- Interactive: maximization bias demo

---

## Additional Context for AI

- This is THE most famous RL algorithm. Make it memorable.
- The Cliff Walking comparison is essential and should be vivid.
- The deadly triad is a crucial setup for DQN's solutions.
- Show that Q-learning is one line different from SARSA—but that one line changes everything.
- Include the Double Q-learning idea—it's used everywhere in deep RL.
- Make off-policy learning intuitive: "learn about the best behavior from any data."

---

## Quality Checklist

- [ ] Q-learning clearly distinguished from SARSA
- [ ] Off-policy concept explained
- [ ] Cliff Walking comparison with demo
- [ ] Deadly triad introduced
- [ ] Double Q-learning covered
- [ ] Clear setup for DQN chapter
