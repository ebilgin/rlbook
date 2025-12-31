# Chapter: Policy Improvement

## Chapter Metadata

**Chapter Number:** 08
**Title:** Policy Improvement
**Section:** Dynamic Programming
**Prerequisites:**
- Policy Evaluation
**Estimated Reading Time:** 25 minutes

---

## Learning Objectives

By the end of this chapter, readers will be able to:

1. Explain the policy improvement theorem
2. Implement policy iteration
3. Implement value iteration
4. Compare policy iteration vs value iteration
5. Solve simple MDPs using DP methods

---

## Core Concepts to Cover

### Primary Concepts (Must Cover)
- [ ] Policy Improvement Theorem: greedy improvement works
- [ ] Greedy policy from value function
- [ ] Policy Iteration: alternate evaluation and improvement
- [ ] Value Iteration: combine evaluation and improvement
- [ ] Convergence guarantees

### Secondary Concepts (Cover if Space Permits)
- [ ] Modified policy iteration
- [ ] Asynchronous DP
- [ ] When to use which algorithm

### Explicitly Out of Scope
- Model-free methods (TD, Q-learning)
- Function approximation
- Large-scale applications

---

## Narrative Arc

### Opening Hook
"We can evaluate any policy. But how do we find a *better* policy? The policy improvement theorem gives us a remarkable guarantee: if we act greedily with respect to the value function, we can only get better—never worse."

### Key Insight
Policy iteration is a ping-pong game: evaluate the current policy, then greedify to get a new policy, then evaluate again. Value iteration collapses this into a single backup operation per state per iteration.

### Closing Connection
"Dynamic Programming gives us exact solutions—but only when we know the MDP completely. In the real world, we often don't know P(s'|s,a). That's where reinforcement learning methods come in: learning from experience when the model is unknown."

---

## Required Interactive Elements

### Demo 1: Policy Iteration Visualizer
- **Purpose:** Watch policy and values evolve together
- **Interaction:**
  - Show GridWorld with policy arrows and V(s) values
  - Step through: evaluation phase → improvement phase
  - Watch arrows change as policy improves
  - Count iterations until convergence
- **Expected Discovery:** Policy converges quickly (often in few iterations)

### Demo 2: Value Iteration Visualizer
- **Purpose:** Show V* emerging directly
- **Interaction:**
  - Same GridWorld, but now value iteration
  - Watch values converge to V*
  - Extract final policy at the end
  - Compare iteration count to policy iteration
- **Expected Discovery:** Value iteration often needs more iterations but is simpler

### Demo 3: Policy Iteration vs Value Iteration Race
- **Purpose:** Compare the two algorithms
- **Interaction:**
  - Run both side by side on same MDP
  - Show computation cost (evaluations, backups)
  - Show final policies match
- **Expected Discovery:** Same answer, different paths

---

## Recurring Examples to Use

- **GridWorld:** The canonical example for DP
- **Frozen Lake:** Slippery version for stochastic transitions
- **Small MDP:** 3-4 states for hand calculation

---

## Cross-References

### Build On (Backward References)
- Chapter 7 (Evaluation): "We can compute V^π..."
- Chapter 6 (Bellman): "The optimality equation suggests..."

### Set Up (Forward References)
- Bandits: "What if we don't know P? Start simple..."
- TD Learning: "What if we learn from samples instead?"
- Q-Learning: "Off-policy learning without a model..."

---

## Mathematical Depth

### Required Equations

1. **Greedy policy**:
$$\pi'(s) = \arg\max_a \left[ R(s,a) + \gamma \sum_{s'} P(s'|s,a) V^\pi(s') \right]$$

2. **Policy Improvement Theorem**:
$$Q^\pi(s, \pi'(s)) \geq V^\pi(s) \text{ for all } s \implies V^{\pi'}(s) \geq V^\pi(s) \text{ for all } s$$

3. **Value Iteration update**:
$$V_{k+1}(s) = \max_a \left[ R(s,a) + \gamma \sum_{s'} P(s'|s,a) V_k(s') \right]$$

### Derivations to Include (Mathematical Layer)
- Sketch of policy improvement theorem proof
- Why value iteration converges to V*

### Proofs to Omit
- Full proof of policy improvement theorem
- Convergence rate analysis

---

## Code Examples Needed

### Intuition Layer
```python
def policy_improvement(V, mdp, gamma=0.99):
    """Improve policy by acting greedily w.r.t. V."""
    policy = {}
    for s in mdp.states:
        action_values = {}
        for a in mdp.actions(s):
            action_values[a] = sum(
                p * (r + gamma * V[s_next])
                for s_next, p, r in mdp.transitions(s, a)
            )
        policy[s] = max(action_values, key=action_values.get)
    return policy
```

```python
def value_iteration(mdp, gamma=0.99, theta=1e-6):
    """Find optimal values using value iteration."""
    V = {s: 0.0 for s in mdp.states}

    while True:
        delta = 0
        for s in mdp.states:
            v = V[s]
            # Bellman optimality backup
            V[s] = max(
                sum(p * (r + gamma * V[s_next])
                    for s_next, p, r in mdp.transitions(s, a))
                for a in mdp.actions(s)
            )
            delta = max(delta, abs(v - V[s]))

        if delta < theta:
            break

    return V
```

### Implementation Layer
- Complete policy iteration with logging
- Value iteration with policy extraction
- Comparison script

---

## Common Misconceptions to Address

1. **"Policy iteration always takes more iterations than value iteration"**: Often the opposite! Policy iteration converges in fewer policy updates.

2. **"Value iteration finds the policy during training"**: No—it finds V*, then we extract the policy at the end.

3. **"We need many policy improvement steps"**: Often just 2-4 policy iterations suffice for simple MDPs.

4. **"DP is practical for large problems"**: Not really—it requires the full model and scales poorly with state space.

---

## Exercises

### Conceptual (3-5 questions)
1. Can policy iteration ever make the policy worse?
2. Why does value iteration use max instead of Σπ(a|s)?
3. What's the advantage of policy iteration over value iteration?

### Coding (2-3 challenges)
1. Implement policy iteration for GridWorld
2. Implement value iteration and compare
3. Solve Frozen Lake with DP

### Exploration (1-2 open-ended)
1. Why is DP not used directly in complex games like Go?

---

## Subsection Breakdown

### Subsection 1: The Policy Improvement Theorem
- Greedy action selection
- The theorem: greedy never hurts
- Why this is remarkable: local improvement → global improvement
- Building intuition with examples

### Subsection 2: Policy Iteration
- The algorithm: evaluate → improve → repeat
- Guaranteed convergence to optimal policy
- Typically very few iterations needed
- Interactive: policy iteration visualizer
- Worked example with numbers

### Subsection 3: Value Iteration
- Combine evaluation and improvement in one step
- Apply Bellman optimality backup directly
- Converges to V*, then extract policy
- Interactive: value iteration visualizer
- Comparison with policy iteration

---

## Additional Context for AI

- Both algorithms should be fully worked through with a simple example.
- The interactive demos are essential—show both algorithms on the same GridWorld.
- Emphasize the "ping-pong" nature of policy iteration.
- Make clear that DP requires the model—this sets up the need for RL.
- End with the question: "What if we don't know P?"

---

## Quality Checklist

- [ ] Policy improvement theorem explained
- [ ] Both algorithms with pseudocode
- [ ] Worked numerical examples
- [ ] Interactive visualizers specified
- [ ] Comparison between the two methods
- [ ] Clear transition to model-free learning
