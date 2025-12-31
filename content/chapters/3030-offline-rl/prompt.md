# Chapter: Offline Reinforcement Learning

## Chapter Metadata

**Chapter Number:** 24
**Title:** Offline Reinforcement Learning
**Section:** Advanced Topics
**Prerequisites:**
- Q-Learning
- DQN
**Estimated Reading Time:** 35 minutes

---

## Learning Objectives

By the end of this chapter, readers will be able to:

1. Explain why offline RL is important for real-world applications
2. Describe the distribution shift problem
3. Implement Conservative Q-Learning (CQL)
4. Explain behavior cloning and its limitations
5. Understand the tradeoff between conservatism and optimality
6. Identify when offline RL is appropriate

---

## Core Concepts to Cover

### Primary Concepts (Must Cover)
- [ ] Online vs offline RL: the fundamental distinction
- [ ] Why offline RL matters (safety, cost, existing data)
- [ ] Distribution shift / OOD actions problem
- [ ] Behavior cloning as a baseline
- [ ] Conservative Q-Learning (CQL)
- [ ] Implicit behavior constraints

### Secondary Concepts (Cover if Space Permits)
- [ ] Batch-constrained Q-learning (BCQ)
- [ ] Decision Transformer
- [ ] Offline policy selection
- [ ] Hybrid online-offline training

### Explicitly Out of Scope
- Detailed imitation learning
- Inverse RL
- Detailed Decision Transformer implementation

---

## Narrative Arc

### Opening Hook
"What if you can't explore? In healthcare, you can't experiment on patients to learn a treatment policy. In autonomous driving, you can't crash cars to learn safe behavior. But you have years of logged data from doctors and human drivers. Can you learn good policies from this fixed dataset, without any new interaction?"

### Key Insight
Offline RL learns from a fixed dataset without environment interaction. The core challenge is distribution shift: the learned policy might choose actions never seen in the data, and we have no way to know if those actions are good or catastrophic. Conservative methods explicitly discourage out-of-distribution actions.

### Closing Connection
"Offline RL bridges the gap between the data we have and the policies we want. It's crucial for deploying RL in the real world, where exploration is expensive or dangerous. Combined with RLHF, it's how we train language models to be helpful without letting them explore harmful behaviors."

---

## Required Interactive Elements

### Demo 1: Online vs Offline Learning
- **Purpose:** Show the fundamental difference
- **Interaction:**
  - Same environment
  - Online agent: interacts freely
  - Offline agent: only sees pre-collected dataset
  - Compare what each can learn
  - Show dataset coverage matters
- **Expected Discovery:** Offline agent is constrained by dataset quality

### Demo 2: Distribution Shift Visualization
- **Purpose:** Make OOD problem visceral
- **Interaction:**
  - Show dataset with limited state-action coverage
  - Train Q-learning on this data
  - Show how Q-values become unreliable for unseen actions
  - Demonstrate policy failure when choosing OOD actions
- **Expected Discovery:** Q-learning extrapolates badly outside data distribution

### Demo 3: Conservative Q-Learning
- **Purpose:** Show how CQL addresses distribution shift
- **Interaction:**
  - Same dataset as Demo 2
  - Train CQL with conservatism penalty
  - Show lower Q-values for OOD actions
  - Compare final policy quality
- **Expected Discovery:** Conservatism prevents overconfident OOD actions

### Demo 4: Dataset Quality Impact
- **Purpose:** Show how data quality affects learning
- **Interaction:**
  - Slider to control dataset quality (expert vs random)
  - Train offline agent on each quality level
  - Show learning curves and final performance
  - Demonstrate that good data = good policies
- **Expected Discovery:** Offline RL performance is bounded by dataset quality

---

## Recurring Examples to Use

- **GridWorld with fixed dataset:** Simple, transparent
- **Healthcare treatment:** Motivating real-world example
- **Autonomous driving:** Safety-critical motivation
- **Atari from human demos:** Connecting to DQN

---

## Cross-References

### Build On (Backward References)
- Q-Learning: "Offline Q-learning is Q-learning without new environment interaction..."
- DQN: "DQN's experience replay is a step toward offline, but still collects new data..."
- Experience Replay: "What if the replay buffer was all we had?"

### Set Up (Forward References)
- RLHF: "Language model training uses offline RL principles..."
- Real-world deployment: "Offline RL enables safe initial policies..."

---

## Mathematical Depth

### Required Equations

1. **Offline RL objective**:
$$\max_\pi J(\pi) = \mathbb{E}_{(s,a,r,s') \sim D}[\sum_t \gamma^t r_t]$$
where $D$ is the fixed offline dataset

2. **Distribution shift problem**:
$$Q^\pi(s, a) \text{ is unreliable when } (s, a) \notin \text{supp}(D)$$

3. **Behavior cloning loss**:
$$L_{BC}(\theta) = -\mathbb{E}_{(s,a) \sim D}[\log \pi_\theta(a|s)]$$

4. **CQL penalty**:
$$\alpha \cdot \mathbb{E}_{s \sim D}\left[ \log \sum_a \exp(Q(s,a)) - \mathbb{E}_{a \sim D(a|s)}[Q(s,a)] \right]$$

5. **CQL objective**:
$$\min_Q \frac{1}{2} \mathbb{E}_{(s,a,r,s') \sim D}[(Q(s,a) - y)^2] + \alpha \cdot \text{CQL penalty}$$

where $y = r + \gamma \max_{a'} Q(s', a')$

### Derivations to Include (Mathematical Layer)
- Why standard Q-learning overestimates for OOD actions
- CQL as lower bound on Q-values
- Information-theoretic view of distribution shift

### Proofs to Omit
- CQL convergence guarantees
- Optimality bounds

---

## Code Examples Needed

### Intuition Layer
```python
def offline_q_learning(dataset, Q, episodes=1000):
    """Naive offline Q-learning (will fail due to distribution shift)."""
    for _ in range(episodes):
        # Sample from fixed dataset
        s, a, r, s_next, done = sample_batch(dataset)

        # Standard Q-learning update
        target = r + gamma * (1 - done) * Q(s_next).max(dim=1).values
        loss = (Q(s).gather(1, a) - target.detach()).pow(2).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Problem: Q-values for unseen actions are unreliable!
```

### Implementation Layer
```python
class CQL:
    """Conservative Q-Learning for offline RL."""

    def __init__(self, state_dim, n_actions, alpha=1.0, hidden_dim=256):
        self.q_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_actions)
        )
        self.target_net = copy.deepcopy(self.q_net)
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=3e-4)
        self.alpha = alpha
        self.n_actions = n_actions

    def compute_cql_penalty(self, states, actions):
        """Compute conservative penalty."""
        q_values = self.q_net(states)

        # Log-sum-exp over all actions (pushes down all Q-values)
        logsumexp = torch.logsumexp(q_values, dim=1)

        # Q-value for dataset actions (pushes up dataset Q-values)
        q_dataset = q_values.gather(1, actions).squeeze()

        # Penalty: logsumexp - dataset actions
        penalty = (logsumexp - q_dataset).mean()
        return penalty

    def update(self, batch):
        """CQL update step."""
        states, actions, rewards, next_states, dones = batch

        # Standard Q-learning target
        with torch.no_grad():
            next_q = self.target_net(next_states).max(dim=1).values
            targets = rewards + (1 - dones) * 0.99 * next_q

        # Current Q-values
        q_values = self.q_net(states)
        q_selected = q_values.gather(1, actions).squeeze()

        # Bellman error
        bellman_loss = F.mse_loss(q_selected, targets)

        # CQL penalty
        cql_penalty = self.compute_cql_penalty(states, actions)

        # Combined loss
        loss = bellman_loss + self.alpha * cql_penalty

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {
            'bellman_loss': bellman_loss.item(),
            'cql_penalty': cql_penalty.item(),
            'total_loss': loss.item()
        }

    def soft_update(self, tau=0.005):
        """Update target network."""
        for target_param, param in zip(self.target_net.parameters(),
                                       self.q_net.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)


class BehaviorCloning:
    """Simple behavior cloning baseline."""

    def __init__(self, state_dim, n_actions, hidden_dim=256):
        self.policy = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_actions)
        )
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=1e-3)

    def update(self, states, actions):
        """Supervised learning: imitate dataset actions."""
        logits = self.policy(states)
        loss = F.cross_entropy(logits, actions)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def select_action(self, state):
        """Select action greedily."""
        with torch.no_grad():
            logits = self.policy(state)
            return logits.argmax().item()


def create_offline_dataset(env, policy, n_episodes=100):
    """Create fixed dataset from behavior policy."""
    dataset = []

    for _ in range(n_episodes):
        state = env.reset()
        done = False

        while not done:
            action = policy(state)  # Behavior policy
            next_state, reward, done, _ = env.step(action)
            dataset.append((state, action, reward, next_state, done))
            state = next_state

    return dataset
```

---

## Common Misconceptions to Address

1. **"Offline RL is just supervised learning"**: Behavior cloning is supervised, but offline RL tries to improve beyond the behavior policy.

2. **"More data always helps"**: Data quality matters more than quantity. Expert data beats random data.

3. **"CQL is too conservative"**: The conservatism parameter α can be tuned. Too conservative = mimic behavior policy. Too liberal = distribution shift.

4. **"Offline RL can't beat the behavior policy"**: It can, by stitching together good parts of different trajectories.

---

## Exercises

### Conceptual (3-5 questions)
1. Why does standard Q-learning fail in the offline setting?
2. What's the tradeoff in choosing the conservatism parameter α in CQL?
3. When would behavior cloning be sufficient? When is it insufficient?

### Coding (2-3 challenges)
1. Implement behavior cloning and evaluate on offline dataset
2. Implement CQL and compare to naive offline Q-learning
3. Create datasets of varying quality and measure how offline RL performance scales

### Exploration (1-2 open-ended)
1. Design an experiment to show stitching: when offline RL outperforms any individual trajectory in the dataset

---

## Subsection Breakdown

### Subsection 1: Why Offline RL?
- The case for learning from fixed data
- Real-world motivations (safety, cost)
- Online vs offline distinction
- Interactive: online vs offline demo

### Subsection 2: The Distribution Shift Problem
- What happens with OOD actions
- Q-learning's overestimation outside data
- Visualizing the problem
- Interactive: distribution shift visualization

### Subsection 3: Conservative Approaches
- Behavior cloning baseline
- Conservative Q-Learning (CQL)
- BCQ and implicit constraints
- Interactive: CQL demo

### Subsection 4: Practical Considerations
- Dataset quality and coverage
- When offline RL is appropriate
- Combining with online fine-tuning
- Decision Transformer preview
- Interactive: dataset quality impact

---

## Additional Context for AI

- Offline RL is crucial for real-world deployment.
- The distribution shift problem is THE key insight.
- CQL is the go-to algorithm—explain it well.
- Healthcare/autonomous driving are compelling motivations.
- Connect to RLHF—LLM training is largely offline.
- Don't oversell—offline RL has real limitations.
- Dataset quality visualization is important.

---

## Quality Checklist

- [ ] Offline RL motivation compelling
- [ ] Distribution shift problem viscerally understood
- [ ] Behavior cloning explained as baseline
- [ ] CQL algorithm implementable
- [ ] Conservatism tradeoff clear
- [ ] Dataset quality impact demonstrated
- [ ] Interactive demos specified
- [ ] Code runnable
