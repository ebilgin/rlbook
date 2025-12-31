# Chapter: Introduction to TD Learning

## Chapter Metadata

**Chapter Number:** 11
**Title:** Introduction to Temporal Difference Learning
**Section:** Temporal Difference Learning
**Prerequisites:**
- Markov Decision Processes (all 3 chapters)
- Dynamic Programming (both chapters)
**Estimated Reading Time:** 30 minutes

---

## Learning Objectives

By the end of this chapter, readers will be able to:

1. Explain the TD learning idea: bootstrapping from estimates
2. Implement TD(0) for value prediction
3. Identify the TD error and its role in learning
4. Compare TD learning with Monte Carlo and DP
5. Explain bias-variance tradeoffs in TD vs MC

---

## Core Concepts to Cover

### Primary Concepts (Must Cover)
- [ ] The TD idea: learn from incomplete episodes
- [ ] TD(0) algorithm for V^π estimation
- [ ] The TD error: δ = r + γV(s') - V(s)
- [ ] Bootstrapping: using estimates to update estimates
- [ ] TD vs Monte Carlo comparison
- [ ] Bias and variance in TD learning

### Secondary Concepts (Cover if Space Permits)
- [ ] TD(λ) preview
- [ ] Eligibility traces intuition
- [ ] Batch TD

### Explicitly Out of Scope
- Control (SARSA, Q-learning) - next chapters
- Function approximation - Deep RL section
- Actor-critic - Policy Gradients section

---

## Narrative Arc

### Opening Hook
"Monte Carlo methods wait until the end of an episode to learn. Dynamic programming needs a model of the environment. What if we could learn step-by-step, from experience, without a model? That's TD learning—the heart of modern reinforcement learning."

### Key Insight
TD learning combines the best of both worlds: like MC, it learns from experience (no model needed); like DP, it updates estimates from estimates (bootstrapping), allowing learning before an episode ends. This is the key insight that enables practical RL.

### Closing Connection
"TD(0) lets us estimate V^π from experience. But for control, we need to learn about actions, not just states. That's where SARSA comes in—TD for action values."

---

## Required Interactive Elements

### Demo 1: TD vs MC Learning Curves
- **Purpose:** Compare learning dynamics
- **Interaction:**
  - Run an agent in GridWorld
  - Plot value estimates for a specific state over episodes
  - Toggle between TD and MC updates
  - Show variance in estimates
- **Expected Discovery:** TD learns faster; MC estimates are more stable but slower

### Demo 2: TD Error Visualization
- **Purpose:** Make TD error intuitive
- **Interaction:**
  - Step through an episode
  - Show V(s), r, V(s') at each step
  - Compute δ = r + γV(s') - V(s)
  - Show update: V(s) ← V(s) + αδ
- **Expected Discovery:** TD error is "surprise"—how much reality differed from expectation

### Demo 3: Bootstrapping Animation
- **Purpose:** Show value propagation
- **Interaction:**
  - Start with random values
  - Show how reward information propagates backward step by step
  - Compare: MC propagates only at episode end; TD propagates every step
- **Expected Discovery:** TD spreads information faster

---

## Recurring Examples to Use

- **GridWorld:** Perfect for step-by-step TD
- **Random Walk:** Classic TD example (Sutton & Barto)
- **Simple Chain MDP:** Easy to trace calculations

---

## Cross-References

### Build On (Backward References)
- DP (Policy Evaluation): "DP uses the model; TD uses experience..."
- Bellman equations: "TD targets are sample-based Bellman updates..."
- Value functions: "We're still estimating V^π..."

### Set Up (Forward References)
- SARSA: "TD for action values..."
- Q-Learning: "Off-policy TD..."
- Function Approximation: "What if states are continuous?"

---

## Mathematical Depth

### Required Equations

1. **TD(0) update**:
$$V(S_t) \leftarrow V(S_t) + \alpha \left[ R_{t+1} + \gamma V(S_{t+1}) - V(S_t) \right]$$

2. **TD error**:
$$\delta_t = R_{t+1} + \gamma V(S_{t+1}) - V(S_t)$$

3. **TD target**:
$$\text{Target} = R_{t+1} + \gamma V(S_{t+1})$$

4. **MC update (for comparison)**:
$$V(S_t) \leftarrow V(S_t) + \alpha \left[ G_t - V(S_t) \right]$$

### Derivations to Include (Mathematical Layer)
- Why TD is biased but consistent
- Bias-variance decomposition sketch
- Relationship to Bellman equation

### Proofs to Omit
- Formal convergence proof
- TD(λ) theory

---

## Code Examples Needed

### Intuition Layer
```python
def td_zero_step(V, s, r, s_next, alpha=0.1, gamma=0.99):
    """Single TD(0) update."""
    td_target = r + gamma * V[s_next]
    td_error = td_target - V[s]
    V[s] = V[s] + alpha * td_error
    return td_error
```

### Implementation Layer
```python
def td_zero_episode(env, V, policy, alpha=0.1, gamma=0.99):
    """Run one episode of TD(0) learning."""
    state = env.reset()
    done = False
    total_reward = 0

    while not done:
        action = policy(state)
        next_state, reward, done = env.step(action)
        total_reward += reward

        # TD update (don't update terminal state)
        if not done:
            td_target = reward + gamma * V[next_state]
        else:
            td_target = reward

        td_error = td_target - V[state]
        V[state] += alpha * td_error

        state = next_state

    return total_reward
```

---

## Common Misconceptions to Address

1. **"TD has no bias"**: TD is biased because it bootstraps from estimated values. MC is unbiased.

2. **"Lower variance means TD is always better"**: Not necessarily—bias can hurt. But TD often works better in practice.

3. **"TD only works for episodic tasks"**: TD works for continuing tasks too! Unlike MC.

4. **"α should be very small"**: Too small = slow learning. Too large = instability. Tuning matters.

---

## Exercises

### Conceptual (3-5 questions)
1. What would happen if we used α = 1 in TD(0)?
2. Why is TD called "temporal difference"?
3. In what situation would MC outperform TD?

### Coding (2-3 challenges)
1. Implement TD(0) for the Random Walk example
2. Compare TD and MC learning curves on GridWorld
3. Plot TD error over an episode

### Exploration (1-2 open-ended)
1. How might TD learning relate to how humans learn?

---

## Subsection Breakdown

### Subsection 1: The TD Idea
- Motivation: learning from incomplete episodes
- The key insight: use V(s') as a stand-in for future returns
- Bootstrapping: estimates from estimates
- Why this is powerful: no need to wait, no need for a model
- The cost: bias from using estimates

### Subsection 2: TD(0) Prediction
- The algorithm: step-by-step updates
- The TD target: r + γV(s')
- The TD error: surprise signal
- Learning rate α and its role
- Worked example with numbers
- Interactive: TD error visualization

### Subsection 3: TD vs Monte Carlo
- MC: wait until episode ends, use actual return
- TD: update every step, use estimated return
- Bias-variance tradeoff
- Sample efficiency comparison
- When to use which
- Interactive: TD vs MC comparison demo

---

## Additional Context for AI

- This is a pivotal chapter. TD is the foundation of Q-learning, SARSA, and beyond.
- The TD error should be made vivid: "surprise" or "prediction error".
- The Random Walk example is classic and should be included.
- Interactive demos are crucial for understanding the difference from MC.
- Emphasize that TD works without episodes ending—key for continuing tasks.
- Make the bias-variance tradeoff concrete with examples.

---

## Quality Checklist

- [ ] TD(0) algorithm clearly explained
- [ ] TD error defined and visualized
- [ ] Comparison with MC and DP
- [ ] Worked numerical example
- [ ] Interactive demos specified
- [ ] Bias-variance tradeoff discussed
- [ ] Clear transition to SARSA
