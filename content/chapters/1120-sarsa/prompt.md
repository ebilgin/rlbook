# Chapter: SARSA

## Chapter Metadata

**Chapter Number:** 12
**Title:** SARSA: On-Policy TD Control
**Section:** Temporal Difference Learning
**Prerequisites:**
- Introduction to TD Learning
**Estimated Reading Time:** 25 minutes

---

## Learning Objectives

By the end of this chapter, readers will be able to:

1. Explain the transition from prediction to control
2. Implement SARSA for learning action values
3. Explain what "on-policy" means
4. Recognize when SARSA's cautious behavior is desirable
5. Train an agent to solve GridWorld using SARSA

---

## Core Concepts to Cover

### Primary Concepts (Must Cover)
- [ ] From V to Q: why we need action values
- [ ] The SARSA update: Q(s,a) ← Q(s,a) + α[r + γQ(s',a') - Q(s,a)]
- [ ] On-policy learning: learn about the policy you're following
- [ ] ε-greedy policy for exploration during SARSA
- [ ] SARSA learns the value of being ε-greedy, not being optimal

### Secondary Concepts (Cover if Space Permits)
- [ ] Expected SARSA
- [ ] SARSA(λ) preview
- [ ] Convergence conditions

### Explicitly Out of Scope
- Q-learning (next chapter)
- Function approximation
- Actor-critic methods

---

## Narrative Arc

### Opening Hook
"TD(0) learns V^π. But to control behavior, we need to evaluate actions, not just states. SARSA is TD for action values—and it gets its name from the quintuple (S, A, R, S', A') that defines each update."

### Key Insight
SARSA learns the value of the policy it's following, including the exploration. This makes it "safe"—it learns to avoid dangerous states even if there's a risky shortcut. The policy and value function evolve together.

### Closing Connection
"SARSA is cautious because it accounts for its own exploration. But what if we want to learn the optimal policy while following an exploratory one? That's Q-learning—and it changes everything."

---

## Required Interactive Elements

### Demo 1: SARSA Training
- **Purpose:** Watch an agent learn with SARSA
- **Interaction:**
  - GridWorld with obstacles
  - Train SARSA agent in real-time
  - Show Q-values as colors in each cell
  - Show learned policy as arrows
  - Episode count and cumulative reward
- **Expected Discovery:** Agent learns to navigate; Q-values propagate from goal

### Demo 2: Cliff Walking
- **Purpose:** Classic SARSA demonstration
- **Interaction:**
  - The Cliff Walking environment from Sutton & Barto
  - SARSA learns the safe path (far from cliff)
  - Show path taken over episodes
  - Compare to Q-learning (which takes the risky path)
- **Expected Discovery:** SARSA is safer because it accounts for ε-exploration

### Demo 3: On-Policy Intuition
- **Purpose:** Show what on-policy means
- **Interaction:**
  - Highlight how SARSA samples a' from the same policy
  - Show that learned values reflect ε-greedy behavior
  - Contrast with "learning optimal values" (Q-learning preview)
- **Expected Discovery:** SARSA's values include the exploration penalty

---

## Recurring Examples to Use

- **GridWorld:** Standard learning environment
- **Cliff Walking:** THE classic SARSA example
- **Windy GridWorld:** Stochastic environment

---

## Cross-References

### Build On (Backward References)
- TD(0): "Now we extend TD to action values..."
- Bandits: "We used ε-greedy there too..."
- Value Functions: "Recall Q^π(s,a)..."

### Set Up (Forward References)
- Q-Learning: "What if we want optimal values?"
- DQN: "SARSA with neural networks..."
- Actor-Critic: "Another on-policy method..."

---

## Mathematical Depth

### Required Equations

1. **SARSA update**:
$$Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \alpha \left[ R_{t+1} + \gamma Q(S_{t+1}, A_{t+1}) - Q(S_t, A_t) \right]$$

2. **ε-greedy policy from Q**:
$$\pi(a|s) = \begin{cases} 1 - \varepsilon + \frac{\varepsilon}{|\mathcal{A}|} & \text{if } a = \arg\max_a Q(s,a) \\ \frac{\varepsilon}{|\mathcal{A}|} & \text{otherwise} \end{cases}$$

3. **TD error for SARSA**:
$$\delta_t = R_{t+1} + \gamma Q(S_{t+1}, A_{t+1}) - Q(S_t, A_t)$$

### Derivations to Include (Mathematical Layer)
- Why SARSA converges to Q^π (not Q*)
- Expected SARSA as variance reduction

### Proofs to Omit
- Formal convergence proofs
- Asymptotic analysis

---

## Code Examples Needed

### Intuition Layer
```python
def sarsa_update(Q, s, a, r, s_next, a_next, alpha=0.1, gamma=0.99):
    """Single SARSA update."""
    td_target = r + gamma * Q[s_next, a_next]
    td_error = td_target - Q[s, a]
    Q[s, a] = Q[s, a] + alpha * td_error
    return td_error
```

### Implementation Layer
```python
def sarsa_episode(env, Q, epsilon=0.1, alpha=0.1, gamma=0.99):
    """Run one episode of SARSA."""
    state = env.reset()
    action = epsilon_greedy(Q[state], epsilon)
    total_reward = 0

    done = False
    while not done:
        next_state, reward, done = env.step(action)
        total_reward += reward

        if done:
            # Terminal update
            Q[state, action] += alpha * (reward - Q[state, action])
        else:
            next_action = epsilon_greedy(Q[next_state], epsilon)
            td_target = reward + gamma * Q[next_state, next_action]
            Q[state, action] += alpha * (td_target - Q[state, action])

            state = next_state
            action = next_action

    return total_reward
```

---

## Common Misconceptions to Address

1. **"SARSA learns Q*"**: No! SARSA learns Q^π for the ε-greedy policy it follows.

2. **"SARSA is inferior to Q-learning"**: Not always. SARSA's conservatism is an advantage in risky environments.

3. **"The 'A'' in SARSA is unnecessary"**: It's essential! It makes SARSA on-policy.

4. **"Lower ε means better SARSA"**: Not necessarily—you trade off exploration for exploitation.

---

## Exercises

### Conceptual (3-5 questions)
1. Why is the algorithm called SARSA?
2. Would SARSA work if we chose a' randomly instead of from our policy?
3. What happens to SARSA as ε → 0?

### Coding (2-3 challenges)
1. Implement SARSA for GridWorld
2. Solve Cliff Walking with SARSA; plot the learning curve
3. Compare different ε values

### Exploration (1-2 open-ended)
1. In what real-world situations would SARSA's caution be valuable?

---

## Subsection Breakdown

### Subsection 1: From Prediction to Control
- TD(0) estimates V^π—but we need actions
- GPI: evaluate Q^π, improve policy, repeat
- Why action values? Can't compute argmax without model
- The SARSA tuple: (S, A, R, S', A')

### Subsection 2: The SARSA Algorithm
- The update rule explained
- Why we need a' before updating
- ε-greedy exploration
- Full algorithm pseudocode
- Interactive: SARSA training demo

### Subsection 3: On-Policy Behavior
- What on-policy means: learn about what you do
- SARSA learns Q^π for ε-greedy π
- The Cliff Walking example
- Why SARSA is "safe"
- Interactive: Cliff Walking demo
- Trade-offs: safety vs optimality

---

## Additional Context for AI

- The Cliff Walking example is essential—it's the clearest demonstration of on-policy behavior.
- Emphasize the SARSA name etymology: S-A-R-S-A.
- Make clear that SARSA doesn't learn Q*—this is the key distinction from Q-learning.
- The demos should show learning in real-time.
- Prepare readers for Q-learning by highlighting SARSA's limitations.

---

## Quality Checklist

- [ ] SARSA update explained and implemented
- [ ] On-policy nature emphasized
- [ ] Cliff Walking example included
- [ ] Interactive training demo specified
- [ ] Clear distinction set up for Q-learning
- [ ] ε-greedy policy integration shown
