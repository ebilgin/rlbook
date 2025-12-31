# Chapter: Actor-Critic Methods

## Chapter Metadata

**Chapter Number:** 20
**Title:** Actor-Critic Methods
**Section:** Policy Gradient Methods
**Prerequisites:**
- REINFORCE and Policy Gradient Theorem
- (Recommended) TD Learning
**Estimated Reading Time:** 40 minutes

---

## Learning Objectives

By the end of this chapter, readers will be able to:

1. Explain the actor-critic architecture and its motivation
2. Implement Advantage Actor-Critic (A2C)
3. Describe the advantage function and its benefits
4. Explain how TD learning provides bootstrap targets
5. Understand the bias-variance tradeoff in actor-critic
6. Implement A2C with n-step returns and GAE

---

## Core Concepts to Cover

### Primary Concepts (Must Cover)
- [ ] Actor-critic architecture: policy (actor) + value (critic)
- [ ] The advantage function: A(s,a) = Q(s,a) - V(s)
- [ ] TD error as advantage estimate
- [ ] A2C algorithm
- [ ] n-step returns in actor-critic
- [ ] Bias-variance tradeoff

### Secondary Concepts (Cover if Space Permits)
- [ ] Generalized Advantage Estimation (GAE)
- [ ] A3C (asynchronous variant)
- [ ] Entropy regularization for exploration
- [ ] Shared vs separate networks

### Explicitly Out of Scope
- PPO (next chapter)
- SAC and off-policy actor-critic
- Continuous action implementation details

---

## Narrative Arc

### Opening Hook
"REINFORCE taught us to weight log-probabilities by returns. But waiting for episode end is slow, and returns are noisy. What if we had a critic—a value function that could estimate how good a state is? We could get feedback every step, not just at episode end."

### Key Insight
Actor-critic methods combine the best of policy gradients and value-based learning. The actor learns the policy; the critic learns the value function. The critic provides low-variance estimates of how good actions are, enabling the actor to learn faster and more stably than pure REINFORCE.

### Closing Connection
"A2C gives us stable, efficient policy learning. But there's still a problem: large policy updates can be catastrophic. PPO addresses this with trust regions, preventing updates that change the policy too much."

---

## Required Interactive Elements

### Demo 1: Actor-Critic Architecture
- **Purpose:** Visualize the dual learning
- **Interaction:**
  - Show actor (policy) and critic (value) networks side by side
  - Watch both learn simultaneously
  - Show how critic values inform actor updates
  - Compare to REINFORCE learning speed
- **Expected Discovery:** Critic accelerates actor learning

### Demo 2: Advantage Function Intuition
- **Purpose:** Show what advantages mean
- **Interaction:**
  - GridWorld with visible state values
  - Show Q-values and V-values
  - Compute and display advantages
  - Show how advantages indicate relative action quality
- **Expected Discovery:** Advantages tell you which actions are better than average

### Demo 3: REINFORCE vs A2C
- **Purpose:** Compare learning curves and stability
- **Interaction:**
  - Same environment (e.g., CartPole)
  - Side-by-side training
  - Show variance in gradient estimates
  - Compare learning speed and stability
- **Expected Discovery:** A2C learns faster with less variance

### Demo 4: n-step Returns Tradeoff
- **Purpose:** Visualize bias-variance tradeoff
- **Interaction:**
  - Slider to adjust n in n-step returns
  - Show how n=1 has high bias, n=∞ has high variance
  - Visualize target estimates for different n
  - Show learning curves for different n
- **Expected Discovery:** Intermediate n balances bias and variance

---

## Recurring Examples to Use

- **CartPole:** Standard benchmark for A2C
- **LunarLander:** More complex, shows A2C strength
- **GridWorld:** For visualizing values and advantages
- **Pendulum:** Continuous control extension

---

## Cross-References

### Build On (Backward References)
- REINFORCE: "We reduce REINFORCE's variance using a critic..."
- TD Learning: "The critic uses TD updates to learn values..."
- Value Functions: "We combine V and Q into the advantage..."

### Set Up (Forward References)
- PPO: "Trust regions will stabilize large policy updates..."
- A3C: "Parallelism accelerates learning..."
- SAC: "Off-policy actor-critic with entropy..."

---

## Mathematical Depth

### Required Equations

1. **Advantage function**:
$$A^\pi(s, a) = Q^\pi(s, a) - V^\pi(s)$$

2. **TD error as advantage estimate**:
$$\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t) \approx A(s_t, a_t)$$

3. **Actor (policy) gradient with advantage**:
$$\nabla_\theta J(\theta) = \mathbb{E}\left[ \nabla_\theta \log \pi_\theta(a_t|s_t) \cdot A(s_t, a_t) \right]$$

4. **Critic (value) loss**:
$$L_V(\phi) = \mathbb{E}\left[ (V_\phi(s_t) - G_t)^2 \right]$$

5. **n-step return**:
$$G_t^{(n)} = \sum_{k=0}^{n-1} \gamma^k r_{t+k} + \gamma^n V(s_{t+n})$$

6. **GAE (λ-weighted advantages)**:
$$\hat{A}_t^{\text{GAE}} = \sum_{l=0}^{\infty} (\gamma \lambda)^l \delta_{t+l}$$

### Derivations to Include (Mathematical Layer)
- Why TD error approximates advantage
- n-step returns derivation
- GAE as exponentially-weighted TD errors
- Why advantage reduces variance (informal)

### Proofs to Omit
- Convergence guarantees
- Optimal λ for GAE

---

## Code Examples Needed

### Intuition Layer
```python
def actor_critic_step(state, action, reward, next_state, done,
                      actor, critic, optimizer):
    """One-step actor-critic update."""
    # Critic: estimate values
    value = critic(state)
    next_value = 0 if done else critic(next_state)

    # TD error = advantage estimate
    td_target = reward + gamma * next_value
    advantage = td_target - value

    # Actor loss: log prob weighted by advantage
    log_prob = actor.log_prob(state, action)
    actor_loss = -log_prob * advantage.detach()

    # Critic loss: TD error squared
    critic_loss = (td_target.detach() - value) ** 2

    # Combined update
    loss = actor_loss + 0.5 * critic_loss
    loss.backward()
    optimizer.step()
```

### Implementation Layer
```python
class ActorCritic(nn.Module):
    """Combined actor-critic network with shared features."""

    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super().__init__()
        # Shared feature extraction
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        # Actor head (policy)
        self.actor = nn.Linear(hidden_dim, action_dim)
        # Critic head (value)
        self.critic = nn.Linear(hidden_dim, 1)

    def forward(self, state):
        features = self.shared(state)
        logits = self.actor(features)
        value = self.critic(features)
        return logits, value

    def get_action(self, state):
        logits, _ = self.forward(state)
        probs = F.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        return action.item(), dist.log_prob(action)

    def evaluate(self, states, actions):
        logits, values = self.forward(states)
        probs = F.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        return log_probs, values.squeeze(), entropy


class A2C:
    def __init__(self, model, lr=3e-4, gamma=0.99, n_steps=5,
                 entropy_coef=0.01, value_coef=0.5):
        self.model = model
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        self.gamma = gamma
        self.n_steps = n_steps
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef

    def compute_returns(self, rewards, values, dones, next_value):
        """Compute n-step returns."""
        returns = []
        R = next_value
        for r, d in zip(reversed(rewards), reversed(dones)):
            R = r + self.gamma * R * (1 - d)
            returns.insert(0, R)
        return torch.tensor(returns)

    def update(self, states, actions, rewards, dones, next_state):
        """Update actor and critic."""
        states = torch.stack(states)
        actions = torch.tensor(actions)
        dones = torch.tensor(dones, dtype=torch.float32)

        # Get values and next value
        with torch.no_grad():
            _, next_value = self.model(next_state)
            next_value = next_value.item() * (1 - dones[-1])

        returns = self.compute_returns(rewards, None, dones, next_value)

        # Forward pass
        log_probs, values, entropy = self.model.evaluate(states, actions)

        # Advantages
        advantages = returns - values.detach()
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Losses
        actor_loss = -(log_probs * advantages).mean()
        critic_loss = F.mse_loss(values, returns)
        entropy_loss = -entropy.mean()

        # Combined loss
        loss = actor_loss + self.value_coef * critic_loss + self.entropy_coef * entropy_loss

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
        self.optimizer.step()

        return {
            'actor_loss': actor_loss.item(),
            'critic_loss': critic_loss.item(),
            'entropy': entropy.mean().item(),
        }
```

---

## Common Misconceptions to Address

1. **"Actor and critic are independent"**: They share features and train together. The critic enables the actor.

2. **"TD error equals the advantage"**: TD error *estimates* the advantage. True advantage requires knowing Q.

3. **"More bootstrapping is always better"**: Heavy bootstrapping (low n) introduces bias. There's a tradeoff.

4. **"A2C is just REINFORCE + baseline"**: Close, but A2C uses TD learning for the critic and can update online.

---

## Exercises

### Conceptual (3-5 questions)
1. Why is the TD error a good estimate of the advantage?
2. What's the tradeoff between 1-step and n-step returns?
3. Why do we use `advantage.detach()` when computing actor loss?

### Coding (2-3 challenges)
1. Implement A2C with shared actor-critic network
2. Experiment with different values of n in n-step returns
3. Add entropy regularization and tune the coefficient

### Exploration (1-2 open-ended)
1. Design an experiment to measure the bias-variance tradeoff for different n values

---

## Subsection Breakdown

### Subsection 1: The Actor-Critic Idea
- Motivation: REINFORCE variance
- Two components: actor and critic
- How critic helps actor
- Interactive: architecture visualization

### Subsection 2: The Advantage Function
- Q, V, and A relationships
- Why advantages work better than returns
- TD error as advantage estimate
- Interactive: advantage intuition

### Subsection 3: The A2C Algorithm
- Combining policy gradient with TD
- The update procedure
- Shared vs separate networks
- Implementation
- Interactive: REINFORCE vs A2C

### Subsection 4: n-step Returns and GAE
- Beyond 1-step TD
- n-step returns
- The bias-variance spectrum
- GAE for optimal blending
- Interactive: n-step tradeoff

---

## Additional Context for AI

- Actor-critic is THE workhorse architecture of modern RL.
- The advantage formulation is crucial—explain it well.
- TD error ≈ advantage is a key insight.
- A2C is synchronous; mention A3C as asynchronous variant.
- GAE is important but can be secondary—show the idea.
- CartPole and LunarLander are good benchmarks.
- Set up PPO by mentioning stability issues.

---

## Quality Checklist

- [ ] Actor-critic motivation clear
- [ ] Advantage function explained intuitively
- [ ] TD error as advantage estimate understood
- [ ] A2C algorithm implementable
- [ ] n-step returns and bias-variance tradeoff covered
- [ ] GAE introduced
- [ ] Code complete and runnable
- [ ] Interactive demos specified
