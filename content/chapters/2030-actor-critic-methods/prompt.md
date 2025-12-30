# Chapter 22: Actor-Critic Methods

## Chapter Metadata

**Chapter Number:** 22
**Title:** Actor-Critic Methods
**Section:** Policy Gradient Methods
**Prerequisites:**
- Chapter 21: Policy Gradient Theorem and REINFORCE (policy gradients, variance problem)
- Chapter 10: Introduction to TD Learning (TD error, bootstrapping)
- Chapter 11: Q-Learning Basics (value function learning)
**Estimated Reading Time:** 30 minutes

---

## Learning Objectives

By the end of this chapter, readers will be able to:

1. Explain the actor-critic architecture and the roles of actor and critic
2. Understand how the critic reduces variance compared to REINFORCE
3. Derive and implement the advantage function $A(s,a) = Q(s,a) - V(s)$
4. Implement A2C (Advantage Actor-Critic) from scratch
5. Understand the bias-variance trade-off in actor-critic methods
6. Compare on-policy and off-policy actor-critic variants

---

## Core Concepts to Cover

### Primary Concepts (Must Cover)
- [ ] The actor-critic idea: policy (actor) + value function (critic)
- [ ] Critic as a learned baseline: lower variance than REINFORCE
- [ ] TD error as advantage estimate: $\delta_t = r + \gamma V(s') - V(s) \approx A(s,a)$
- [ ] Advantage function: $A^{\pi}(s,a) = Q^{\pi}(s,a) - V^{\pi}(s)$
- [ ] A2C: Advantage Actor-Critic algorithm
- [ ] Bootstrapping trade-off: reduced variance but introduces bias

### Secondary Concepts (Cover if Space Permits)
- [ ] n-step returns in actor-critic
- [ ] Generalized Advantage Estimation (GAE) preview
- [ ] Shared vs separate networks for actor and critic
- [ ] Entropy regularization for exploration

### Explicitly Out of Scope
- PPO and trust region methods (next chapter)
- Off-policy actor-critic in depth (SAC, TD3 — later section)
- Deterministic policy gradient (DPG)
- Asynchronous methods (A3C)

---

## Narrative Arc

### Opening Hook
"REINFORCE works, but it's slow and noisy because it waits for complete episodes and uses high-variance return estimates. What if we could use the TD trick—learning from single steps by bootstrapping—while still doing policy gradient? That's exactly what actor-critic methods do."

Bridge from REINFORCE's limitation to the actor-critic solution.

### Key Insight
Actor-critic combines the best of both worlds:
- **Actor**: Policy $\pi_\theta(a|s)$ that we improve via gradients
- **Critic**: Value function $V_\phi(s)$ that estimates expected returns

Instead of using Monte Carlo returns $G_t$ (high variance), we use the TD error:
$$\delta_t = r_{t+1} + \gamma V_\phi(s_{t+1}) - V_\phi(s_t)$$

This is an unbiased estimate of the advantage $A(s_t, a_t)$! Lower variance because $V_\phi$ is learned, not sampled.

### Closing Connection
"Actor-critic gives us the core architecture for modern RL: a policy network and a value network trained together. But there's still a stability problem—large policy updates can be catastrophic. Next, we'll see how PPO and TRPO add constraints to make training rock-solid."

---

## Required Interactive Elements

### Demo 1: Actor-Critic Architecture Visualizer
- **Purpose:** Show the flow of information in actor-critic
- **Interaction:**
  - Visualize state → actor → action, state → critic → value
  - Step through an episode, see TD error computed
  - Watch how TD error is used to update actor
  - Toggle to show gradient flow
- **Expected Discovery:** Actor and critic work together—critic evaluates, actor improves.

### Demo 2: Variance Comparison: REINFORCE vs Actor-Critic
- **Purpose:** Demonstrate the variance reduction from using a critic
- **Interaction:**
  - Run REINFORCE and A2C on same environment
  - Show gradient estimate variance over time
  - Compare learning curves
  - Visualize the critic's value estimates improving
- **Expected Discovery:** A2C has lower variance in gradient estimates, learns more stably.

### Demo 3: Bias-Variance Trade-off Visualization
- **Purpose:** Show how n-step returns trade off bias and variance
- **Interaction:**
  - Slider for n (1-step to full Monte Carlo)
  - See how variance decreases and bias increases with smaller n
  - Watch learning progress for different n values
- **Expected Discovery:** There's a sweet spot; 1-step is low variance but biased, Monte Carlo is unbiased but high variance.

---

## Recurring Examples to Use

- **CartPole:** Primary environment for A2C implementation
- **GridWorld:** For visualizing value function and policy together
- **LunarLander:** More challenging environment to show A2C scaling up
- **Pendulum:** Preview for continuous control (continuous actions)

---

## Cross-References

### Build On (Backward References)
- Chapter 21: "Recall REINFORCE's update: $\theta \leftarrow \theta + \alpha G_t \nabla_\theta \log \pi_\theta$..."
- Chapter 10: "The TD error $\delta_t = r + \gamma V(s') - V(s)$ we saw in TD learning..."
- Chapter 11: "Like Q-learning's value update, the critic learns from TD errors"

### Set Up (Forward References)
- Chapter 23: "PPO adds a constraint to prevent the actor from changing too much"
- Later chapters: "SAC extends actor-critic to off-policy learning with continuous actions"

---

## Mathematical Depth

### Required Equations
1. Advantage function:
$$A^{\pi}(s,a) = Q^{\pi}(s,a) - V^{\pi}(s)$$

2. TD error as advantage estimate:
$$\delta_t = R_{t+1} + \gamma V_\phi(S_{t+1}) - V_\phi(S_t) \approx A(S_t, A_t)$$

3. Actor update (policy gradient with advantage):
$$\theta \leftarrow \theta + \alpha \nabla_\theta \log \pi_\theta(A_t|S_t) \cdot \hat{A}_t$$

4. Critic update (TD learning):
$$\phi \leftarrow \phi + \alpha_v \delta_t \nabla_\phi V_\phi(S_t)$$

5. n-step advantage:
$$\hat{A}_t^{(n)} = \sum_{k=0}^{n-1} \gamma^k R_{t+k+1} + \gamma^n V_\phi(S_{t+n}) - V_\phi(S_t)$$

6. Policy gradient with advantage:
$$\nabla_\theta J(\theta) = \mathbb{E}_{\pi_\theta}\left[\nabla_\theta \log \pi_\theta(a|s) \cdot A^{\pi_\theta}(s,a)\right]$$

### Derivations to Include (Mathematical Layer)
- Show that TD error is an unbiased estimate of advantage (in expectation)
- Derive the n-step return formula
- Show why $A(s,a)$ is centered (sum over actions is zero under policy)

### Proofs to Omit
- GAE derivation (mention it, link to paper)
- Convergence analysis of two-timescale learning
- Optimal critic learning rate relative to actor

---

## Code Examples Needed

### Intuition Layer
```python
# Actor-Critic update in one snippet
value = critic(state)
next_value = critic(next_state) if not done else 0
advantage = reward + gamma * next_value - value  # TD error

# Update actor
actor_loss = -log_prob * advantage.detach()

# Update critic
critic_loss = advantage ** 2
```

### Implementation Layer
- Complete A2C agent with separate actor and critic networks
- Training loop for CartPole and LunarLander
- Comparison with REINFORCE (learning curves)
- Version with entropy regularization

---

## Common Misconceptions to Address

1. **"Actor-critic is just REINFORCE with a baseline"**: Almost, but the critic is *learned*, not a fixed baseline. The critic also uses bootstrapping, introducing some bias for lower variance.

2. **"The critic learns Q(s,a)"**: In basic actor-critic, the critic learns V(s), not Q(s,a). We use the TD error as an advantage estimate. (Some variants do learn Q, like SAC.)

3. **"Lower variance is always better"**: Not if it comes with too much bias. Bootstrapping introduces bias when the value function is inaccurate. This is the trade-off.

4. **"Actor and critic should have the same learning rate"**: Usually not—the critic often needs to learn faster (higher learning rate) so it provides a useful signal to the actor.

5. **"TD error and advantage are the same thing"**: TD error is an *estimate* of advantage. They're equal in expectation (when the critic is accurate) but not identical.

6. **"You need separate networks for actor and critic"**: Not necessarily—shared layers can work well and are more parameter-efficient. The trade-off is training stability.

---

## Exercises

### Conceptual (4 questions)
- Explain why the TD error $\delta_t$ is an estimate of the advantage $A(s,a)$. Under what conditions is this estimate unbiased?
- What are the roles of actor and critic? Which one is like a coach, and which is like a player?
- Why does using a learned critic reduce variance compared to Monte Carlo returns?
- Compare actor-critic to Q-learning with $\epsilon$-greedy. What are the similarities and differences?

### Coding (3 challenges)
- Implement A2C with separate actor and critic networks. Train on CartPole and plot learning curves.
- Compare learning with n=1 (TD) vs n=10 vs full episode returns. Which learns faster? Which is more stable?
- Add entropy regularization to your A2C implementation. How does it affect exploration?

### Exploration (1 open-ended)
- Try different learning rates for actor and critic. What happens when the critic learns much faster than the actor? Much slower? Can you find a ratio that works well?

---

## Additional Context for AI

- Actor-critic is the foundation of most modern RL algorithms (PPO, SAC, TD3, etc.)
- Emphasize the "two timescales" idea: critic learns faster to provide a stable signal
- The TD error as advantage estimate is the key insight—make sure readers understand it
- Connect back to TD learning: the critic is just doing TD(0) or n-step TD
- Entropy regularization is important for exploration—cover it at least briefly
- Show both separate and shared network architectures
- CartPole should be solved reliably with A2C; LunarLander is a good stretch goal
- Prepare readers for PPO by noting that actor updates can be unstable if too large
- On-policy focus for now; off-policy actor-critic (SAC, DDPG) comes later

---

## Quality Checklist

- [ ] Actor-critic architecture is clearly explained with diagram/visualization
- [ ] TD error as advantage estimate is derived and explained intuitively
- [ ] Bias-variance trade-off is covered (why bootstrapping helps and hurts)
- [ ] A2C implementation is complete and runnable
- [ ] Comparison with REINFORCE shows improvement
- [ ] Entropy regularization is covered
- [ ] Interactive demos help build intuition
- [ ] Code follows conventions (PyTorch patterns, clear variable names)
- [ ] Clear forward reference to PPO and trust regions
