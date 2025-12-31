# Chapter: Model-Based Reinforcement Learning

## Chapter Metadata

**Chapter Number:** 22
**Title:** Model-Based Reinforcement Learning
**Section:** Advanced Topics
**Prerequisites:**
- Q-Learning
- (Recommended) TD Learning
**Estimated Reading Time:** 40 minutes

---

## Learning Objectives

By the end of this chapter, readers will be able to:

1. Distinguish between model-free and model-based RL
2. Explain the sample efficiency advantage of model-based methods
3. Implement the Dyna architecture
4. Describe how to learn environment models
5. Explain the model bias problem
6. Describe modern model-based methods (MuZero preview)

---

## Core Concepts to Cover

### Primary Concepts (Must Cover)
- [ ] Model-free vs model-based: the fundamental distinction
- [ ] What is a model? (transition dynamics + reward function)
- [ ] Planning with a learned model
- [ ] The Dyna architecture
- [ ] Model learning from experience
- [ ] Model errors and their impact

### Secondary Concepts (Cover if Space Permits)
- [ ] Model predictive control (MPC)
- [ ] Ensemble models for uncertainty
- [ ] MuZero: learning the model for planning
- [ ] Model-based policy optimization (MBPO)

### Explicitly Out of Scope
- Detailed MuZero implementation
- Continuous control with models
- Differentiable simulation

---

## Narrative Arc

### Opening Hook
"Every model-free method we've seen learns by trial and error in the real world. But what if the agent could *imagine* experiences? What if it had a model of how the world works and could plan ahead, like a chess player thinking several moves ahead?"

### Key Insight
Model-based RL learns a model of the environment (how states transition, what rewards occur) and uses it to plan or generate synthetic experience. This can be far more sample-efficient than model-free learning, but introduces the challenge of model errors.

### Closing Connection
"Model-based RL trades sample efficiency for the challenge of learning accurate models. When it works, it's remarkably efficient. Modern systems like MuZero show that learned models can achieve superhuman performance across diverse games—sometimes learning to plan in ways we don't fully understand."

---

## Required Interactive Elements

### Demo 1: Model-Free vs Model-Based Comparison
- **Purpose:** Show sample efficiency difference
- **Interaction:**
  - Same GridWorld, two agents
  - Model-free: learns from real experience only
  - Model-based: learns model, simulates additional experience
  - Count real environment interactions
  - Compare learning curves
- **Expected Discovery:** Model-based learns much faster from fewer real samples

### Demo 2: Dyna Architecture
- **Purpose:** Visualize integrated learning and planning
- **Interaction:**
  - Show model learning alongside Q-learning
  - Visualize simulated experience from model
  - Toggle planning depth (0-20 simulated steps per real step)
  - Show how more planning accelerates learning
- **Expected Discovery:** Planning amplifies each real experience

### Demo 3: Model Errors
- **Purpose:** Show how model errors hurt
- **Interaction:**
  - Introduce inaccurate model (wrong transitions)
  - Show how agent plans with wrong model
  - Watch agent fail when following bad plans
  - Compare to model-free agent
- **Expected Discovery:** Garbage model in, garbage policy out

### Demo 4: Model Learning
- **Purpose:** Visualize model learning from experience
- **Interaction:**
  - Agent explores environment
  - Show transition table filling in
  - Visualize model uncertainty decreasing
  - Show how model predictions improve
- **Expected Discovery:** Model gets better with more diverse experience

---

## Recurring Examples to Use

- **GridWorld:** Perfect for deterministic model learning
- **Stochastic GridWorld:** Show model learning with noise
- **Maze:** Planning through learned corridors
- **Board games:** Preview for MuZero

---

## Cross-References

### Build On (Backward References)
- Q-Learning: "Dyna integrates model-based planning with Q-learning..."
- TD Learning: "Model-free TD learns from real experience; Dyna adds simulated..."
- Value Functions: "Models let us compute value functions via planning..."

### Set Up (Forward References)
- MuZero: "Modern model-based methods learn abstract models..."
- Multi-Agent RL: "Models of other agents are crucial in multi-agent..."
- Offline RL: "Learned models can extend offline datasets..."

---

## Mathematical Depth

### Required Equations

1. **Environment model**:
$$\hat{P}(s'|s, a) \approx P(s'|s, a)$$
$$\hat{R}(s, a) \approx R(s, a)$$

2. **Dyna-Q update (real experience)**:
$$Q(s, a) \leftarrow Q(s, a) + \alpha[r + \gamma \max_{a'} Q(s', a') - Q(s, a)]$$

3. **Dyna-Q update (simulated experience)**:
$$Q(s, a) \leftarrow Q(s, a) + \alpha[\hat{r} + \gamma \max_{a'} Q(\hat{s}', a') - Q(s, a)]$$
where $\hat{s}', \hat{r}$ come from the model

4. **Model learning (tabular)**:
$$\hat{P}(s'|s, a) = \frac{\text{count}(s, a, s')}{\text{count}(s, a)}$$

5. **Model-based policy optimization objective**:
$$J(\theta) = \mathbb{E}_{s_0, a_0, ...} \left[ \sum_{t=0}^{H} \gamma^t \hat{r}_t \right]$$
where rollouts use the learned model

### Derivations to Include (Mathematical Layer)
- Why model-based is more sample efficient (amplification intuition)
- Model uncertainty propagation
- Dyna convergence conditions

### Proofs to Omit
- Sample complexity bounds
- Model learning convergence

---

## Code Examples Needed

### Intuition Layer
```python
class Dyna:
    """Dyna: Q-learning + model-based planning."""

    def __init__(self, n_states, n_actions, planning_steps=5):
        self.Q = np.zeros((n_states, n_actions))
        self.model = {}  # Learned model: (s,a) -> (s', r)
        self.planning_steps = planning_steps

    def update(self, s, a, r, s_prime):
        # Direct RL: learn from real experience
        self.Q[s, a] += alpha * (r + gamma * self.Q[s_prime].max() - self.Q[s, a])

        # Model learning: remember what happened
        self.model[(s, a)] = (s_prime, r)

        # Planning: simulate additional experience
        for _ in range(self.planning_steps):
            s_sim, a_sim = random.choice(list(self.model.keys()))
            s_next_sim, r_sim = self.model[(s_sim, a_sim)]
            self.Q[s_sim, a_sim] += alpha * (
                r_sim + gamma * self.Q[s_next_sim].max() - self.Q[s_sim, a_sim]
            )
```

### Implementation Layer
```python
class TabularModel:
    """Tabular environment model."""

    def __init__(self, n_states, n_actions):
        self.n_states = n_states
        self.n_actions = n_actions
        # Count-based transition model
        self.transition_counts = np.zeros((n_states, n_actions, n_states))
        self.reward_sum = np.zeros((n_states, n_actions))
        self.visit_counts = np.zeros((n_states, n_actions))

    def update(self, s, a, r, s_prime):
        """Update model from experience."""
        self.transition_counts[s, a, s_prime] += 1
        self.reward_sum[s, a] += r
        self.visit_counts[s, a] += 1

    def predict(self, s, a):
        """Predict next state and reward."""
        if self.visit_counts[s, a] == 0:
            return None, None

        # Transition probabilities
        probs = self.transition_counts[s, a] / self.visit_counts[s, a]
        s_prime = np.random.choice(self.n_states, p=probs)

        # Expected reward
        r = self.reward_sum[s, a] / self.visit_counts[s, a]

        return s_prime, r

    def sample_experienced(self):
        """Sample a previously experienced (s, a) pair."""
        experienced = np.where(self.visit_counts > 0)
        if len(experienced[0]) == 0:
            return None
        idx = np.random.randint(len(experienced[0]))
        return experienced[0][idx], experienced[1][idx]


class DynaQ:
    """Full Dyna-Q implementation."""

    def __init__(self, n_states, n_actions, alpha=0.1, gamma=0.99,
                 epsilon=0.1, planning_steps=10):
        self.n_states = n_states
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.planning_steps = planning_steps

        self.Q = np.zeros((n_states, n_actions))
        self.model = TabularModel(n_states, n_actions)

    def select_action(self, state):
        if np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)
        return np.argmax(self.Q[state])

    def update(self, s, a, r, s_prime, done):
        # Direct RL update
        target = r if done else r + self.gamma * self.Q[s_prime].max()
        self.Q[s, a] += self.alpha * (target - self.Q[s, a])

        # Model learning
        self.model.update(s, a, r, s_prime)

        # Planning
        for _ in range(self.planning_steps):
            sim = self.model.sample_experienced()
            if sim is None:
                break
            s_sim, a_sim = sim
            s_next, r_sim = self.model.predict(s_sim, a_sim)
            if s_next is not None:
                target = r_sim + self.gamma * self.Q[s_next].max()
                self.Q[s_sim, a_sim] += self.alpha * (target - self.Q[s_sim, a_sim])
```

---

## Common Misconceptions to Address

1. **"Models must be accurate to be useful"**: Even approximate models help. The key is using uncertainty appropriately.

2. **"Model-based is always more sample efficient"**: Only if the model is good. Bad models can be worse than model-free.

3. **"Dyna is model-based, Q-learning is model-free"**: Dyna IS Q-learning + model. It's a hybrid.

4. **"Learning the model is easy"**: For complex environments, model learning is extremely challenging.

---

## Exercises

### Conceptual (3-5 questions)
1. Why is model-based RL more sample efficient than model-free?
2. What happens in Dyna if the environment changes but the model isn't updated?
3. How does planning depth affect learning speed vs computational cost?

### Coding (2-3 challenges)
1. Implement Dyna-Q and compare to Q-learning on a maze
2. Experiment with different planning steps (1, 5, 20, 50) and plot learning curves
3. Implement a stochastic model and compare to deterministic

### Exploration (1-2 open-ended)
1. Design an experiment to show when model-based fails (hint: model errors)

---

## Subsection Breakdown

### Subsection 1: Model-Free vs Model-Based
- What we've been doing: model-free
- What is a model?
- Sample efficiency intuition
- Interactive: comparison demo

### Subsection 2: The Dyna Architecture
- Integrating learning and planning
- The Dyna-Q algorithm
- Planning as simulated experience
- Implementation
- Interactive: Dyna visualization

### Subsection 3: Learning the Model
- Tabular model learning
- Neural network models (preview)
- Model uncertainty
- Interactive: model learning demo

### Subsection 4: Challenges and Modern Methods
- Model errors and compounding
- Ensemble methods for uncertainty
- MuZero: learning abstract models
- When to use model-based
- Interactive: model errors demo

---

## Additional Context for AI

- Model-based RL is gaining importance in real-world applications.
- Dyna is the classic algorithm—explain it thoroughly.
- The sample efficiency advantage is the key motivation.
- Model errors are the key challenge—make this visceral.
- MuZero is the modern capstone—mention as inspiration.
- GridWorld/maze makes model learning transparent.
- Connect planning to how humans think ahead.

---

## Quality Checklist

- [ ] Model-free vs model-based clearly distinguished
- [ ] Dyna architecture explained and implemented
- [ ] Sample efficiency advantage demonstrated
- [ ] Model learning covered
- [ ] Model error problem illustrated
- [ ] MuZero previewed
- [ ] Interactive demos specified
- [ ] Code runnable
