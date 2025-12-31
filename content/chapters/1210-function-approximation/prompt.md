# Chapter: Function Approximation

## Chapter Metadata

**Chapter Number:** 14
**Title:** Function Approximation in RL
**Section:** Deep Reinforcement Learning
**Prerequisites:**
- Q-Learning
- (Recommended) Bellman Equations
**Estimated Reading Time:** 35 minutes

---

## Learning Objectives

By the end of this chapter, readers will be able to:

1. Explain why tabular methods fail in large/continuous state spaces
2. Describe the function approximation approach to RL
3. Implement linear function approximation for value estimation
4. Understand the deadly triad and its implications
5. Explain how neural networks enable deep RL
6. Identify when function approximation is necessary

---

## Core Concepts to Cover

### Primary Concepts (Must Cover)
- [ ] The curse of dimensionality in tabular RL
- [ ] Value function approximation: V̂(s; w) and Q̂(s, a; w)
- [ ] Linear function approximation with feature vectors
- [ ] Gradient descent for value learning
- [ ] The deadly triad: function approximation + bootstrapping + off-policy
- [ ] From linear to neural networks

### Secondary Concepts (Cover if Space Permits)
- [ ] Feature engineering and representations
- [ ] Tile coding and radial basis functions
- [ ] Semi-gradient methods
- [ ] Target networks (preview for DQN)

### Explicitly Out of Scope
- Full DQN architecture (next chapter)
- Experience replay (next chapter)
- Specific deep learning architectures

---

## Narrative Arc

### Opening Hook
"Our Q-learning agent mastered a 4×4 grid. But what about a robot navigating a room? With continuous position (x, y) and orientation (θ), there are infinite states. We can't have a table entry for every possible configuration. We need a way to generalize."

### Key Insight
Function approximation lets us represent value functions compactly and generalize across similar states. Instead of storing Q(s,a) for every state, we learn parameters w such that Q̂(s,a;w) ≈ Q*(s,a). Similar states get similar values automatically.

### Closing Connection
"Linear function approximation shows the core ideas, but the real power comes from neural networks. They can learn their own features from raw pixels, enabling agents to play Atari games and control robots. That's Deep Q-Networks—coming next."

---

## Required Interactive Elements

### Demo 1: Tabular vs Continuous
- **Purpose:** Show why tables fail
- **Interaction:**
  - Continuous 2D state space (e.g., mountain car position/velocity)
  - Show what tabular discretization looks like
  - Demonstrate aliasing and generalization failure
  - Contrast with smooth function approximator
- **Expected Discovery:** Discretization is either too coarse or too memory-intensive

### Demo 2: Linear Value Function
- **Purpose:** Visualize linear approximation
- **Interaction:**
  - 2D state space with features
  - Show value surface as linear combination of basis functions
  - Adjust weights and see surface change
  - Watch gradient descent update weights from experience
- **Expected Discovery:** Linear approximation creates smooth, generalizable value functions

### Demo 3: The Deadly Triad
- **Purpose:** Show instability from the triad
- **Interaction:**
  - Toggle: function approximation on/off
  - Toggle: bootstrapping on/off
  - Toggle: off-policy on/off
  - Show training stability/instability
  - Demonstrate that any two elements are fine; all three causes problems
- **Expected Discovery:** The combination of all three creates instability

---

## Recurring Examples to Use

- **Mountain Car:** Continuous state space, needs function approximation
- **CartPole:** Another continuous control classic
- **GridWorld with continuous position:** Extension of familiar example
- **Atari frames:** Preview of DQN (high-dimensional inputs)

---

## Cross-References

### Build On (Backward References)
- Q-Learning: "We now extend Q-learning beyond tables..."
- Bellman Equations: "The Bellman equation still holds, but now we approximate..."
- TD Learning: "TD updates become gradient descent on parameters..."

### Set Up (Forward References)
- DQN: "Next we'll see how neural networks + two key tricks enable stable learning..."
- Policy Gradients: "Function approximation for policies, not just values..."
- Actor-Critic: "Approximating both value and policy..."

---

## Mathematical Depth

### Required Equations

1. **Parameterized value function**:
$$\hat{V}(s; \mathbf{w}) \approx V^\pi(s)$$
$$\hat{Q}(s, a; \mathbf{w}) \approx Q^\pi(s, a)$$

2. **Linear approximation with features**:
$$\hat{V}(s; \mathbf{w}) = \mathbf{w}^\top \phi(s) = \sum_{i=1}^{d} w_i \phi_i(s)$$

3. **Mean squared value error**:
$$\overline{\text{VE}}(\mathbf{w}) = \sum_{s \in \mathcal{S}} \mu(s) \left[ V^\pi(s) - \hat{V}(s; \mathbf{w}) \right]^2$$

4. **Semi-gradient TD(0)**:
$$\mathbf{w} \leftarrow \mathbf{w} + \alpha \left[ R + \gamma \hat{V}(S'; \mathbf{w}) - \hat{V}(S; \mathbf{w}) \right] \nabla_\mathbf{w} \hat{V}(S; \mathbf{w})$$

5. **Semi-gradient Q-learning**:
$$\mathbf{w} \leftarrow \mathbf{w} + \alpha \left[ R + \gamma \max_{a'} \hat{Q}(S', a'; \mathbf{w}) - \hat{Q}(S, A; \mathbf{w}) \right] \nabla_\mathbf{w} \hat{Q}(S, A; \mathbf{w})$$

### Derivations to Include (Mathematical Layer)
- Why the gradient is only on the prediction, not the target (semi-gradient)
- Linear approximation gradient: ∇V̂ = φ(s)
- Why the deadly triad causes divergence

### Proofs to Omit
- Convergence guarantees for linear TD
- PAC bounds

---

## Code Examples Needed

### Intuition Layer
```python
class LinearValueFunction:
    """Value function as linear combination of features."""

    def __init__(self, num_features):
        self.w = np.zeros(num_features)

    def __call__(self, state):
        features = self.get_features(state)
        return np.dot(self.w, features)

    def update(self, state, target, alpha=0.01):
        """Semi-gradient update toward target."""
        features = self.get_features(state)
        prediction = np.dot(self.w, features)
        error = target - prediction
        self.w += alpha * error * features  # Gradient is just features!
```

### Implementation Layer
```python
class TileCodedQ:
    """Q-function with tile coding for continuous states."""

    def __init__(self, n_tilings, tiles_per_dim, n_actions, alpha=0.1):
        self.n_tilings = n_tilings
        self.tiles_per_dim = tiles_per_dim
        self.n_actions = n_actions
        self.alpha = alpha / n_tilings  # Divide by tilings

        # Each tiling has tiles_per_dim^state_dim * n_actions weights
        n_tiles = tiles_per_dim ** 2 * n_actions
        self.w = np.zeros((n_tilings, n_tiles))

        # Random offsets for each tiling
        self.offsets = np.random.uniform(0, 1/tiles_per_dim, (n_tilings, 2))

    def get_tiles(self, state, action):
        """Get active tile indices for state-action pair."""
        tiles = []
        for t in range(self.n_tilings):
            # Offset state
            offset_state = state + self.offsets[t]
            # Discretize
            idx = (offset_state * self.tiles_per_dim).astype(int)
            idx = np.clip(idx, 0, self.tiles_per_dim - 1)
            # Flatten to single index
            tile_idx = idx[0] * self.tiles_per_dim + idx[1]
            tile_idx = tile_idx * self.n_actions + action
            tiles.append((t, tile_idx))
        return tiles

    def __call__(self, state, action):
        """Get Q-value."""
        tiles = self.get_tiles(state, action)
        return sum(self.w[t, idx] for t, idx in tiles)

    def update(self, state, action, target):
        """Semi-gradient TD update."""
        prediction = self(state, action)
        error = target - prediction
        tiles = self.get_tiles(state, action)
        for t, idx in tiles:
            self.w[t, idx] += self.alpha * error


def semi_gradient_sarsa(env, Q, episodes=1000, gamma=0.99, epsilon=0.1):
    """Semi-gradient SARSA with function approximation."""
    for ep in range(episodes):
        state = env.reset()
        action = epsilon_greedy(Q, state, epsilon)

        while True:
            next_state, reward, done, _ = env.step(action)

            if done:
                target = reward
                Q.update(state, action, target)
                break

            next_action = epsilon_greedy(Q, next_state, epsilon)
            target = reward + gamma * Q(next_state, next_action)
            Q.update(state, action, target)

            state, action = next_state, next_action
```

---

## Common Misconceptions to Address

1. **"Function approximation is just interpolation"**: It's generalization—predicting values for unseen states based on learned patterns.

2. **"More features = better"**: Overfitting is possible. Features should capture relevant structure, not noise.

3. **"TD learning with function approximation always converges"**: The deadly triad can cause divergence. DQN's tricks address this.

4. **"Neural networks are always better than linear"**: Linear methods can be faster, more stable, and sufficient for many problems.

---

## Exercises

### Conceptual (3-5 questions)
1. Why is the TD update called "semi-gradient"?
2. What's the difference between state aliasing in discretization vs generalization in function approximation?
3. Which element of the deadly triad does on-policy learning remove? Why does that help?

### Coding (2-3 challenges)
1. Implement linear TD(0) for a simple prediction task
2. Create tile coding for Mountain Car and compare to tabular Q-learning
3. Demonstrate the deadly triad by showing divergence with all three elements vs stability with only two

### Exploration (1-2 open-ended)
1. Design features for a problem domain you're interested in. What structure should they capture?

---

## Subsection Breakdown

### Subsection 1: Beyond Tabular Methods
- The curse of dimensionality
- Continuous and high-dimensional state spaces
- Why we need generalization
- Interactive: tabular vs continuous demo

### Subsection 2: Linear Function Approximation
- Feature vectors and basis functions
- Linear value functions
- Gradient descent for value learning
- Tile coding and RBFs
- Interactive: linear value function demo

### Subsection 3: Challenges and Solutions
- The deadly triad explained
- Why bootstrapping + function approximation + off-policy is unstable
- Semi-gradient methods
- Preview: how DQN solves these issues
- Interactive: deadly triad demo

### Subsection 4: From Linear to Deep
- Limitations of linear approximation
- Neural networks as universal function approximators
- End-to-end feature learning
- The promise of deep RL

---

## Additional Context for AI

- This chapter bridges tabular RL and deep RL.
- The deadly triad is crucial—it explains why DQN needed its tricks.
- Keep Mountain Car as the running example for continuity.
- Tile coding is a good concrete example of feature engineering.
- Neural networks are introduced conceptually here; DQN gets the details.
- Make the "why" of function approximation visceral—show the exponential blowup.

---

## Quality Checklist

- [ ] Curse of dimensionality made vivid
- [ ] Linear approximation fully explained
- [ ] Semi-gradient derivation clear
- [ ] Deadly triad understood
- [ ] Feature engineering examples provided
- [ ] Bridge to neural networks established
- [ ] Interactive demos specified
