# Chapter: Policy Evaluation

## Chapter Metadata

**Chapter Number:** 07
**Title:** Policy Evaluation
**Section:** Dynamic Programming
**Prerequisites:**
- Introduction to MDPs
- Value Functions
- The Bellman Equations
**Estimated Reading Time:** 20 minutes

---

## Learning Objectives

By the end of this chapter, readers will be able to:

1. Implement iterative policy evaluation
2. Explain convergence of policy evaluation
3. Choose appropriate stopping criteria
4. Trace through policy evaluation on a simple MDP
5. Understand the computational requirements

---

## Core Concepts to Cover

### Primary Concepts (Must Cover)
- [ ] Iterative Policy Evaluation algorithm
- [ ] Synchronous vs asynchronous updates
- [ ] Convergence: when to stop
- [ ] Computational complexity
- [ ] Worked example with GridWorld

### Secondary Concepts (Cover if Space Permits)
- [ ] In-place updates vs two-array updates
- [ ] Sweep orderings
- [ ] Matrix form of policy evaluation

### Explicitly Out of Scope
- Policy improvement (next chapter)
- Function approximation
- Model-free evaluation (Monte Carlo, TD)

---

## Narrative Arc

### Opening Hook
"We have the Bellman equations. We know that the value function must satisfy them. But how do we actually find V^π? The answer is surprisingly simple: just keep applying the Bellman equation until the values stop changing."

### Key Insight
Policy evaluation is a fixed-point algorithm. We start with any guess and repeatedly apply the Bellman operator. Because the operator is a contraction, we're guaranteed to converge to the unique solution.

### Closing Connection
"Now we can compute V^π for any policy. But our goal is to find the *optimal* policy. How do we use value functions to improve policies? That's the subject of the next chapter."

---

## Required Interactive Elements

### Demo 1: Policy Evaluation Visualizer
- **Purpose:** Watch values converge in real-time
- **Interaction:**
  - Display GridWorld with a fixed policy (arrows)
  - Show V(s) values in each cell
  - Step button: one full sweep
  - Play/pause: automatic stepping
  - Show iteration count and max change
- **Expected Discovery:** Values converge from initial guesses to true V^π

### Demo 2: Convergence Analysis
- **Purpose:** Show convergence behavior
- **Interaction:**
  - Plot max value change (Δ) over iterations
  - Show different γ values affect convergence speed
  - Show how initial values don't affect final result
- **Expected Discovery:** Higher γ = slower convergence; same endpoint regardless of start

---

## Recurring Examples to Use

- **4×4 GridWorld:** Perfect size for visualization
  - Random policy: even probability for each action
  - Show how values converge from 0 to true V^π
- **Numerical trace:** Step-by-step calculation for a 3-state MDP

---

## Cross-References

### Build On (Backward References)
- Chapter 6 (Bellman): "Recall the Bellman expectation equation..."
- Chapter 5 (Values): "V^π is the expected return..."

### Set Up (Forward References)
- Chapter 8 (Improvement): "Now we can improve the policy..."
- TD Learning: "What if we don't know the transitions?"

---

## Mathematical Depth

### Required Equations

1. **Policy Evaluation Update**:
$$V_{k+1}(s) = \sum_a \pi(a|s) \left[ R(s,a) + \gamma \sum_{s'} P(s'|s,a) V_k(s') \right]$$

2. **Convergence condition**:
$$\max_s |V_{k+1}(s) - V_k(s)| < \theta$$

### Derivations to Include (Mathematical Layer)
- Why iteration converges (contraction argument sketch)
- Relationship to solving linear system

### Proofs to Omit
- Formal contraction mapping proof
- Rate of convergence analysis

---

## Code Examples Needed

### Intuition Layer
```python
def policy_evaluation(mdp, policy, gamma=0.99, theta=1e-6):
    """Iterative policy evaluation."""
    V = {s: 0.0 for s in mdp.states}

    while True:
        delta = 0
        for s in mdp.states:
            v = V[s]
            # Bellman backup
            V[s] = sum(
                policy[s][a] * sum(
                    p * (r + gamma * V[s_next])
                    for s_next, p, r in mdp.transitions(s, a)
                )
                for a in mdp.actions(s)
            )
            delta = max(delta, abs(v - V[s]))

        if delta < theta:
            break

    return V
```

### Implementation Layer
- Full implementation with logging
- Visualization of convergence
- Comparison of stopping thresholds

---

## Common Misconceptions to Address

1. **"We need to start with good initial values"**: Any initialization works; convergence is guaranteed.

2. **"More iterations is always better"**: After convergence, additional iterations are wasted computation.

3. **"θ should be as small as possible"**: Smaller θ means more computation. Choose based on precision needs.

---

## Exercises

### Conceptual (3-5 questions)
1. How does γ affect convergence speed?
2. What would happen if we used γ = 1 in a continuing task?
3. Why can we start with V(s) = 0 for all states?

### Coding (2-3 challenges)
1. Implement policy evaluation for GridWorld
2. Plot convergence curves for different γ values

### Exploration (1-2 open-ended)
1. How would you parallelize policy evaluation?

---

## Subsection Breakdown

### Subsection 1: Iterative Policy Evaluation
- The algorithm: initialize, sweep, repeat
- Synchronous updates: update all states, then move to next iteration
- Worked example with small MDP
- Interactive: step-by-step visualizer

### Subsection 2: Convergence and Stopping
- Why it converges: contraction intuition
- Stopping criterion: when max change is small
- How to choose θ
- Computational cost: O(|S|²|A|) per sweep
- Interactive: convergence analysis plots

---

## Additional Context for AI

- This is the first "algorithm" chapter. Make the algorithm crystal clear.
- Show actual numbers—don't just describe; compute.
- The interactive demo is crucial: watch values change in real-time.
- Keep it focused: evaluation only, no improvement yet.
- Set up the question: "We can evaluate, but how do we improve?"

---

## Quality Checklist

- [ ] Algorithm presented as clear pseudocode
- [ ] Worked numerical example
- [ ] Convergence visualized
- [ ] Stopping criterion explained
- [ ] GridWorld demo specified
- [ ] Clean transition to policy improvement
