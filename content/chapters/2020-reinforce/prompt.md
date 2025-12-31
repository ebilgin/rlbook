# Chapter: REINFORCE and the Policy Gradient Theorem

## Chapter Metadata

**Chapter Number:** 19
**Title:** REINFORCE and the Policy Gradient Theorem
**Section:** Policy Gradient Methods
**Prerequisites:**
- Introduction to Policy Gradients
**Estimated Reading Time:** 35 minutes

---

## Learning Objectives

By the end of this chapter, readers will be able to:

1. State and explain the Policy Gradient Theorem
2. Implement the REINFORCE algorithm
3. Explain why we use log probabilities in the gradient
4. Identify the variance problem in REINFORCE
5. Implement baselines to reduce variance
6. Understand why REINFORCE is Monte Carlo policy gradient

---

## Core Concepts to Cover

### Primary Concepts (Must Cover)
- [ ] The Policy Gradient Theorem
- [ ] REINFORCE algorithm
- [ ] Log-derivative trick
- [ ] Why Monte Carlo returns? (complete episode requirement)
- [ ] The variance problem
- [ ] Baselines for variance reduction

### Secondary Concepts (Cover if Space Permits)
- [ ] Whitening returns
- [ ] Entropy bonus for exploration
- [ ] Connection to supervised learning (weighted MLE)
- [ ] Historical context (Williams 1992)

### Explicitly Out of Scope
- Actor-critic methods (next chapter)
- Trust regions (PPO chapter)
- Advanced variance reduction (GAE)

---

## Narrative Arc

### Opening Hook
"We want to improve our policy by gradient ascent. But there's a problem: the gradient of expected return involves the environment dynamics, which we don't know. The Policy Gradient Theorem provides an elegant solution—we can compute the gradient using only samples from our policy."

### Key Insight
The Policy Gradient Theorem tells us that the gradient of expected return is the expected value of (return × gradient of log-probability). We don't need to know environment dynamics—we just need to sample trajectories and weight the log-probability gradients by returns. High-return actions get reinforced; low-return actions get suppressed.

### Closing Connection
"REINFORCE is elegant but high-variance—we must wait until episode end and deal with noisy return estimates. What if we could get gradient signals at every step using value function estimates? That's actor-critic methods."

---

## Required Interactive Elements

### Demo 1: Policy Gradient Theorem Visualization
- **Purpose:** Show how gradient updates shift probability mass
- **Interaction:**
  - Simple 3-action bandit
  - Run episodes, show returns
  - Visualize how log-prob gradient shifts probabilities
  - Show how high-reward actions get more probable
- **Expected Discovery:** REINFORCE increases probability of actions that led to high returns

### Demo 2: REINFORCE Training
- **Purpose:** Watch REINFORCE learn
- **Interaction:**
  - CartPole environment
  - Show policy probabilities evolving
  - Show return variance across episodes
  - Plot learning curve
- **Expected Discovery:** REINFORCE works but learning is noisy

### Demo 3: Variance Visualization
- **Purpose:** Demonstrate the variance problem
- **Interaction:**
  - Show gradient estimates from different episodes
  - Visualize how noisy the estimates are
  - Compare with and without baseline
  - Show variance reduction
- **Expected Discovery:** Baselines dramatically reduce variance without changing expected gradient

### Demo 4: Baseline Impact
- **Purpose:** Show how baselines help learning
- **Interaction:**
  - Side-by-side: REINFORCE with and without baseline
  - Same environment, same seeds where possible
  - Compare learning speed and stability
- **Expected Discovery:** Baseline makes learning faster and more stable

---

## Recurring Examples to Use

- **CartPole:** Standard REINFORCE benchmark
- **Simple bandit:** For theorem intuition
- **GridWorld:** Discrete actions, familiar environment
- **Short corridor:** For illustrating why stochastic policies can be necessary

---

## Cross-References

### Build On (Backward References)
- Intro to Policy Gradients: "We established the objective; now we derive the gradient..."
- Monte Carlo Methods: "Like MC, REINFORCE uses complete episode returns..."
- TD Learning: "Unlike TD, REINFORCE can't update mid-episode..."

### Set Up (Forward References)
- Actor-Critic: "Next we'll use value functions to reduce variance further..."
- PPO: "Trust regions will address learning stability..."
- A2C: "Advantage functions formalize the baseline idea..."

---

## Mathematical Depth

### Required Equations

1. **Policy Gradient Theorem**:
$$\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta} \left[ \sum_{t=0}^{T} \nabla_\theta \log \pi_\theta(a_t|s_t) \cdot G_t \right]$$

2. **Log-derivative trick**:
$$\nabla_\theta \pi_\theta(a|s) = \pi_\theta(a|s) \nabla_\theta \log \pi_\theta(a|s)$$

3. **REINFORCE update**:
$$\theta \leftarrow \theta + \alpha \sum_{t=0}^{T} \nabla_\theta \log \pi_\theta(a_t|s_t) \cdot G_t$$

4. **With baseline**:
$$\theta \leftarrow \theta + \alpha \sum_{t=0}^{T} \nabla_\theta \log \pi_\theta(a_t|s_t) \cdot (G_t - b(s_t))$$

5. **Baseline doesn't bias gradient**:
$$\mathbb{E}[\nabla_\theta \log \pi_\theta(a|s) \cdot b(s)] = 0$$

### Derivations to Include (Mathematical Layer)
- Derivation of policy gradient theorem from first principles
- Why log-derivative trick works
- Proof that baseline doesn't change expected gradient
- Variance formula showing baseline reduction

### Proofs to Omit
- Convergence guarantees
- Optimal baseline derivation

---

## Code Examples Needed

### Intuition Layer
```python
def reinforce_update(episode, policy, optimizer):
    """REINFORCE: weight log-probs by returns."""
    states, actions, rewards = episode

    # Compute returns (reward-to-go)
    returns = compute_returns(rewards, gamma=0.99)

    # Policy gradient: log_prob * return
    log_probs = [policy.log_prob(s, a) for s, a in zip(states, actions)]
    loss = -sum(lp * G for lp, G in zip(log_probs, returns))

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

### Implementation Layer
```python
class REINFORCE:
    def __init__(self, policy, lr=1e-3, gamma=0.99, baseline=None):
        self.policy = policy
        self.optimizer = torch.optim.Adam(policy.parameters(), lr=lr)
        self.gamma = gamma
        self.baseline = baseline  # Optional value function

    def compute_returns(self, rewards):
        """Compute discounted returns for each timestep."""
        returns = []
        G = 0
        for r in reversed(rewards):
            G = r + self.gamma * G
            returns.insert(0, G)
        return torch.tensor(returns)

    def update(self, episode):
        """Update policy from a complete episode."""
        states, actions, rewards = zip(*episode)

        states = torch.stack(states)
        actions = torch.tensor(actions)
        returns = self.compute_returns(rewards)

        # Optional: normalize returns for stability
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        # Compute log probabilities
        log_probs = self.policy.log_prob(states, actions)

        # Apply baseline if available
        if self.baseline is not None:
            values = self.baseline(states).squeeze()
            advantages = returns - values.detach()
            policy_loss = -(log_probs * advantages).mean()

            # Also update baseline
            value_loss = F.mse_loss(values, returns)
            loss = policy_loss + 0.5 * value_loss
        else:
            policy_loss = -(log_probs * returns).mean()
            loss = policy_loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return policy_loss.item()


def train_reinforce(env, agent, episodes=1000):
    """Train REINFORCE agent."""
    returns_history = []

    for ep in range(episodes):
        episode = []
        state = env.reset()
        done = False
        total_reward = 0

        while not done:
            action = agent.policy.sample(state)
            next_state, reward, done, _ = env.step(action)
            episode.append((state, action, reward))
            state = next_state
            total_reward += reward

        agent.update(episode)
        returns_history.append(total_reward)

        if ep % 100 == 0:
            print(f"Episode {ep}, Avg Return: {np.mean(returns_history[-100:]):.2f}")

    return returns_history
```

---

## Common Misconceptions to Address

1. **"REINFORCE can update during an episode"**: It cannot. It needs the full return, which is only known at episode end.

2. **"The baseline should be the return"**: No! The baseline must be independent of the action taken. Using V(s) is correct.

3. **"Higher learning rate = faster learning"**: High variance means high LR causes instability. Lower LR is often necessary.

4. **"REINFORCE always converges"**: It can get stuck in local optima or diverge with poor hyperparameters.

---

## Exercises

### Conceptual (3-5 questions)
1. Why does multiplying the log-probability by the return create the correct gradient direction?
2. Prove that subtracting a state-dependent baseline doesn't change the expected gradient.
3. Why can't we use TD-style updates in REINFORCE?

### Coding (2-3 challenges)
1. Implement REINFORCE from scratch and train on CartPole
2. Add a learned baseline (value function) and compare variance
3. Implement the "reward-to-go" version of REINFORCE (only future rewards)

### Exploration (1-2 open-ended)
1. Experiment with different baseline choices. What happens with a constant baseline? A running mean of returns?

---

## Subsection Breakdown

### Subsection 1: The Policy Gradient Theorem
- The challenge: gradient of expectation
- The log-derivative trick
- Derivation of the theorem
- Intuition: why log-probability times return?
- Interactive: gradient visualization

### Subsection 2: The REINFORCE Algorithm
- Putting the theorem into practice
- Complete episode requirement
- The algorithm step by step
- Implementation
- Interactive: REINFORCE training

### Subsection 3: The Variance Problem
- Why REINFORCE is noisy
- Sources of variance
- Visualizing gradient variance
- Impact on learning
- Interactive: variance visualization

### Subsection 4: Baselines
- Reducing variance without bias
- State-value baseline
- Proof of unbiasedness
- Advantage formulation preview
- Interactive: baseline impact demo

---

## Additional Context for AI

- The policy gradient theorem is THE key insight. Derive it carefully.
- The log-derivative trick should feel like magic but be understood.
- REINFORCE's simplicity is a feature—it's pure policy gradient.
- The variance problem sets up actor-critic perfectly.
- Baselines are crucial—don't skip them.
- CartPole is the canonical REINFORCE benchmark.
- Make the "reinforcing good actions" intuition clear.

---

## Quality Checklist

- [ ] Policy Gradient Theorem derived and explained
- [ ] Log-derivative trick understood
- [ ] REINFORCE algorithm implementable
- [ ] Variance problem viscerally understood
- [ ] Baseline unbiasedness proven
- [ ] Code complete and runnable
- [ ] Interactive demos specified
- [ ] Connection to actor-critic made
