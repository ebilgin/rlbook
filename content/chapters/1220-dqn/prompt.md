# Chapter: Deep Q-Networks

## Chapter Metadata

**Chapter Number:** 15
**Title:** Deep Q-Networks (DQN)
**Section:** Deep Reinforcement Learning
**Prerequisites:**
- Q-Learning
- Function Approximation
**Estimated Reading Time:** 35 minutes

---

## Learning Objectives

By the end of this chapter, readers will be able to:

1. Explain why naive Q-learning with neural networks fails
2. Describe how experience replay breaks correlation
3. Explain target networks and why they stabilize training
4. Implement DQN from scratch
5. Train an agent to play simple games

---

## Core Concepts to Cover

### Primary Concepts (Must Cover)
- [ ] The challenge: applying Q-learning to high-dimensional inputs
- [ ] The architecture: CNN for processing frames
- [ ] Experience replay: store and sample transitions
- [ ] Target networks: frozen targets for stable learning
- [ ] The complete DQN algorithm
- [ ] Frame stacking and preprocessing

### Secondary Concepts (Cover if Space Permits)
- [ ] Huber loss vs MSE
- [ ] Gradient clipping
- [ ] Exploration scheduling (ε-decay)
- [ ] Atari preprocessing details

### Explicitly Out of Scope
- DQN improvements (Double, Prioritized, Dueling) - next chapter
- Continuous actions
- Actor-critic methods

---

## Narrative Arc

### Opening Hook
"In 2013, a paper from DeepMind shook the AI world: a single algorithm, with the same hyperparameters, learned to play 49 different Atari games from raw pixels—some at superhuman level. That algorithm was DQN, and it showed that deep learning and reinforcement learning could work together."

### Key Insight
The deadly triad (off-policy + function approximation + bootstrapping) seems fatal. DQN survives by breaking two correlations: experience replay breaks the correlation between consecutive samples, and target networks break the correlation between Q-values and their targets.

### Closing Connection
"DQN was a breakthrough, but it had limitations. Overestimation bias. Uniform experience sampling. Coupled state-action values. The next chapter shows how researchers improved each of these—leading to algorithms that crushed DQN's performance."

---

## Required Interactive Elements

### Demo 1: Experience Replay Visualization
- **Purpose:** Show why correlation is a problem and how replay helps
- **Interaction:**
  - Visualize a replay buffer as a queue/ring buffer
  - Show transitions being stored as agent plays
  - Show random sampling for updates
  - Contrast with "on-policy" sequential updates
  - Toggle between correlated vs replay updates
- **Expected Discovery:** Random sampling breaks the sequence dependence

### Demo 2: Target Network Stability
- **Purpose:** Show why fixed targets help
- **Interaction:**
  - Side-by-side: DQN with and without target network
  - Show Q-value trajectories during training
  - No target network: oscillations, instability
  - With target network: smooth convergence
- **Expected Discovery:** Chasing a moving target causes oscillations

### Demo 3: DQN Training Visualization
- **Purpose:** Watch DQN learn a game
- **Interaction:**
  - Simple game environment (CartPole or simplified Atari)
  - Show agent's view (stacked frames)
  - Show Q-values for each action
  - Training curves: reward and loss
  - Play button to watch trained agent
- **Expected Discovery:** DQN learns complex behaviors from pixels

### Demo 4: Frame Stacking Intuition
- **Purpose:** Show why we stack frames
- **Interaction:**
  - Single frame: can you tell direction?
  - Stacked frames: motion is visible
  - Show how a ball's trajectory becomes clear
- **Expected Discovery:** Temporal information from multiple frames

---

## Recurring Examples to Use

- **Pong:** Classic DQN game, easy to understand
- **Breakout:** Shows the famous "tunnel" strategy
- **CartPole:** Simpler for first implementation
- **Simple GridWorld with images:** If needed for demos

---

## Cross-References

### Build On (Backward References)
- Q-Learning: "DQN is Q-learning with a neural network..."
- Function Approximation: "We saw linear Q-learning struggles..."
- Deadly Triad: "Remember the three ingredients for instability..."

### Set Up (Forward References)
- DQN Improvements: "We can do better than DQN..."
- Rainbow: "Combining all the tricks..."
- Policy Gradients: "A different approach to deep RL..."

---

## Mathematical Depth

### Required Equations

1. **DQN loss function**:
$$L(\theta) = \mathbb{E}_{(s,a,r,s') \sim \mathcal{D}} \left[ \left( r + \gamma \max_{a'} Q(s', a'; \theta^-) - Q(s, a; \theta) \right)^2 \right]$$

2. **Target network update** (hard update):
$$\theta^- \leftarrow \theta \quad \text{every } C \text{ steps}$$

3. **Gradient**:
$$\nabla_\theta L = \mathbb{E} \left[ \left( r + \gamma \max_{a'} Q(s', a'; \theta^-) - Q(s, a; \theta) \right) \nabla_\theta Q(s, a; \theta) \right]$$

4. **ε-greedy with decay**:
$$\varepsilon_t = \max(\varepsilon_{\text{final}}, \varepsilon_{\text{start}} - t \cdot \frac{\varepsilon_{\text{start}} - \varepsilon_{\text{final}}}{\text{decay steps}})$$

### Derivations to Include (Mathematical Layer)
- Why the target network reduces variance
- Connection to fitted Q-iteration
- Why replay buffer size matters

### Proofs to Omit
- Convergence guarantees (they don't exist for DQN)
- Sample complexity analysis

---

## Code Examples Needed

### Intuition Layer
```python
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (np.array(states), np.array(actions), np.array(rewards),
                np.array(next_states), np.array(dones))
```

### Implementation Layer
```python
class DQN(nn.Module):
    def __init__(self, in_channels, n_actions):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )
        self.fc = nn.Sequential(
            nn.Linear(7 * 7 * 64, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions),
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


def dqn_train_step(q_network, target_network, optimizer, replay_buffer,
                   batch_size=32, gamma=0.99):
    """Single training step."""
    states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)

    # Current Q values
    q_values = q_network(states).gather(1, actions.unsqueeze(1))

    # Target Q values (from target network)
    with torch.no_grad():
        next_q_values = target_network(next_states).max(1)[0]
        targets = rewards + gamma * next_q_values * (1 - dones)

    # Loss and update
    loss = F.mse_loss(q_values.squeeze(), targets)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()
```

---

## Common Misconceptions to Address

1. **"DQN learns from the reward signal alone"**: It learns from the TD error, which involves bootstrapped estimates.

2. **"Bigger replay buffer is always better"**: Too big = old, irrelevant transitions. Too small = correlated samples.

3. **"Target network should update every step"**: That defeats the purpose! Update infrequently (e.g., every 10K steps).

4. **"DQN is sample efficient"**: It's not! DQN needs millions of frames to learn Atari.

5. **"DQN converges to Q*"**: No guarantees! It works in practice but can diverge in theory.

---

## Exercises

### Conceptual (3-5 questions)
1. Why would training fail without experience replay?
2. What would happen if we updated the target network every step?
3. Why stack 4 frames instead of just using the current frame?

### Coding (2-3 challenges)
1. Implement a replay buffer with numpy
2. Train DQN on CartPole
3. Visualize Q-values during training

### Exploration (1-2 open-ended)
1. The original DQN paper used the same hyperparameters for all 49 games. Is this surprising? What are the tradeoffs?

---

## Subsection Breakdown

### Subsection 1: The DQN Architecture
- From Q-table to Q-network
- CNNs for processing pixels
- Frame stacking for temporal information
- The architecture from the Nature paper
- Interactive: frame stacking demo

### Subsection 2: Experience Replay
- The correlation problem
- Store transitions in a buffer
- Sample randomly for updates
- Benefits: data efficiency, decorrelation
- Buffer size considerations
- Interactive: replay buffer visualization

### Subsection 3: Target Networks
- The moving target problem
- Solution: freeze the target
- Hard vs soft updates
- Update frequency
- Interactive: stability comparison demo

### Subsection 4: Putting It Together
- The complete DQN algorithm
- ε-decay schedule
- Training loop
- Hyperparameters from the paper
- Interactive: DQN training visualization

---

## Additional Context for AI

- DQN is a landmark paper. Give it appropriate weight.
- The two key tricks (replay, target networks) should be very clear.
- Show real training curves if possible—DQN takes millions of frames.
- The Atari preprocessing is important but can be in Mathematical/Implementation layer.
- Emphasize that DQN is Q-learning + two tricks to fix instability.
- Set up the improvements chapter: "DQN is great, but we can do better."

---

## Quality Checklist

- [ ] Experience replay explained with visualization
- [ ] Target networks explained with visualization
- [ ] Complete architecture described
- [ ] Frame stacking justified
- [ ] Full algorithm presented
- [ ] Working code provided
- [ ] Clear limitations mentioned (setting up improvements chapter)
