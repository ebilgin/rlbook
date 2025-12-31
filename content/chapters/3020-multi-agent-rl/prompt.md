# Chapter: Multi-Agent Reinforcement Learning

## Chapter Metadata

**Chapter Number:** 23
**Title:** Multi-Agent Reinforcement Learning
**Section:** Advanced Topics
**Prerequisites:**
- Q-Learning
- Policy Gradients (recommended)
**Estimated Reading Time:** 40 minutes

---

## Learning Objectives

By the end of this chapter, readers will be able to:

1. Explain challenges unique to multi-agent settings
2. Distinguish cooperative, competitive, and mixed settings
3. Describe independent learning and its limitations
4. Explain centralized training with decentralized execution (CTDE)
5. Implement simple multi-agent algorithms
6. Understand game-theoretic concepts (Nash equilibrium, self-play)

---

## Core Concepts to Cover

### Primary Concepts (Must Cover)
- [ ] Multi-agent problem formulation
- [ ] Cooperative vs competitive vs mixed
- [ ] Independent learners (IQL)
- [ ] Non-stationarity problem
- [ ] Centralized training, decentralized execution (CTDE)
- [ ] Self-play for competitive games

### Secondary Concepts (Cover if Space Permits)
- [ ] Nash equilibrium basics
- [ ] Communication in multi-agent systems
- [ ] Emergent behavior
- [ ] OpenAI Five / AlphaStar examples

### Explicitly Out of Scope
- Detailed game theory
- Extensive-form games
- Mean-field approximations
- Mechanism design

---

## Narrative Arc

### Opening Hook
"So far, our agent has been alone in its environment. But most interesting problems involve multiple decision-makers: autonomous vehicles sharing roads, robots cooperating in a warehouse, or AIs competing in games. When multiple agents learn simultaneously, everything changes."

### Key Insight
In multi-agent RL, each agent faces a moving target: other agents are learning too, changing the environment dynamics. Simple independent learning often fails because the environment appears non-stationary. The solution: train agents together (centralized) but deploy them independently (decentralized).

### Closing Connection
"Multi-agent RL is where RL meets game theory and economics. The emergent behaviors—cooperation, competition, even deception—can be surprising and powerful. AlphaStar's strategies and OpenAI Five's teamwork emerged from multi-agent learning."

---

## Required Interactive Elements

### Demo 1: Cooperative vs Competitive
- **Purpose:** Show different multi-agent settings
- **Interaction:**
  - Simple grid world with two agents
  - Toggle: cooperative (shared reward) vs competitive (zero-sum)
  - Watch different behaviors emerge
  - Show how incentives shape behavior
- **Expected Discovery:** Game structure determines emergent behavior

### Demo 2: Non-Stationarity Problem
- **Purpose:** Show why independent learning struggles
- **Interaction:**
  - Two agents learning independently
  - Show how each agent's "optimal" action changes
  - Visualize the moving target problem
  - Compare to single-agent learning stability
- **Expected Discovery:** Other learning agents make the environment non-stationary

### Demo 3: Self-Play Training
- **Purpose:** Demonstrate self-play for games
- **Interaction:**
  - Simple game (tic-tac-toe or similar)
  - Agent plays against copies of itself
  - Show policy improvement over generations
  - Track win rate against earlier versions
- **Expected Discovery:** Self-play creates curriculum of increasingly strong opponents

### Demo 4: Emergent Coordination
- **Purpose:** Show surprising cooperative behavior
- **Interaction:**
  - Coordination game (e.g., meeting point)
  - Agents must coordinate without explicit communication
  - Watch strategies emerge
  - Show multiple possible equilibria
- **Expected Discovery:** Agents can learn to coordinate implicitly

---

## Recurring Examples to Use

- **GridWorld tag:** One agent chases another
- **Cooperative navigation:** Agents reach goals without collision
- **Rock-Paper-Scissors:** Game theory basics
- **Simple pursuit-evasion:** Predator-prey dynamics
- **Prisoner's Dilemma:** Classic cooperation problem

---

## Cross-References

### Build On (Backward References)
- Q-Learning: "Independent Q-learning is just Q-learning per agent..."
- Policy Gradients: "Multi-agent policy gradients face additional challenges..."
- TD Learning: "TD methods must handle non-stationary targets..."

### Set Up (Forward References)
- RLHF: "LLM training can be seen as a multi-agent game..."
- Advanced Topics: "Communication protocols, population-based training..."

---

## Mathematical Depth

### Required Equations

1. **Markov Game (Stochastic Game)**:
- $N$ agents, states $S$, actions $A_1 \times A_2 \times ... \times A_N$
- Transition: $P(s' | s, a_1, ..., a_N)$
- Rewards: $R_i(s, a_1, ..., a_N)$ for each agent $i$

2. **Independent Q-learning**:
$$Q_i(s, a_i) \leftarrow Q_i(s, a_i) + \alpha[r_i + \gamma \max_{a'_i} Q_i(s', a'_i) - Q_i(s, a_i)]$$

3. **Joint action value (centralized)**:
$$Q^{tot}(s, a_1, ..., a_N) = f(Q_1(s, a_1), ..., Q_N(s, a_N))$$

4. **Nash equilibrium condition**:
$$V_i(\pi_1^*, ..., \pi_N^*) \geq V_i(\pi_1^*, ..., \pi_i, ..., \pi_N^*) \quad \forall \pi_i, \forall i$$

5. **Fictitious play update**:
$$\hat{\pi}_{-i}(a_{-i}) = \frac{1}{t} \sum_{k=1}^{t} \mathbf{1}[a_{-i}^k = a_{-i}]$$

### Derivations to Include (Mathematical Layer)
- Why independent learning doesn't converge in general
- Nash equilibrium interpretation
- Self-play as best response iteration

### Proofs to Omit
- Nash equilibrium existence
- Fictitious play convergence conditions

---

## Code Examples Needed

### Intuition Layer
```python
# Independent Q-learning: each agent ignores others
class IndependentQLearning:
    def __init__(self, n_agents, n_states, n_actions):
        self.Q = [np.zeros((n_states, n_actions)) for _ in range(n_agents)]

    def update(self, agent_idx, s, a, r, s_next):
        # Each agent updates its own Q-table
        # Ignoring that other agents are also learning
        target = r + gamma * self.Q[agent_idx][s_next].max()
        self.Q[agent_idx][s, a] += alpha * (target - self.Q[agent_idx][s, a])
```

### Implementation Layer
```python
class MultiAgentEnv:
    """Simple multi-agent gridworld."""

    def __init__(self, grid_size=5, n_agents=2):
        self.grid_size = grid_size
        self.n_agents = n_agents
        self.reset()

    def reset(self):
        # Random starting positions
        self.positions = [
            np.array([np.random.randint(self.grid_size),
                     np.random.randint(self.grid_size)])
            for _ in range(self.n_agents)
        ]
        return self._get_observations()

    def _get_observations(self):
        # Each agent sees all positions
        return [np.concatenate(self.positions) for _ in range(self.n_agents)]

    def step(self, actions):
        """All agents act simultaneously."""
        rewards = np.zeros(self.n_agents)

        # Move agents
        for i, action in enumerate(actions):
            delta = {0: [0, 1], 1: [0, -1], 2: [1, 0], 3: [-1, 0], 4: [0, 0]}[action]
            self.positions[i] = np.clip(
                self.positions[i] + delta, 0, self.grid_size - 1
            )

        # Compute rewards (example: cooperative - get close)
        if np.allclose(self.positions[0], self.positions[1]):
            rewards[:] = 10  # Shared reward for meeting

        done = np.allclose(self.positions[0], self.positions[1])
        return self._get_observations(), rewards, done, {}


class CTDE_QMix:
    """Simplified QMIX-style value decomposition."""

    def __init__(self, n_agents, obs_dim, n_actions, hidden_dim=64):
        self.n_agents = n_agents
        self.n_actions = n_actions

        # Individual agent networks
        self.agent_nets = nn.ModuleList([
            nn.Sequential(
                nn.Linear(obs_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, n_actions)
            ) for _ in range(n_agents)
        ])

        # Mixing network (monotonic combination)
        self.mixer = nn.Sequential(
            nn.Linear(n_agents, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def get_individual_q(self, observations):
        """Get Q-values for each agent."""
        return [net(obs) for net, obs in zip(self.agent_nets, observations)]

    def get_total_q(self, observations, actions):
        """Get mixed Q-value for joint action."""
        individual_q = self.get_individual_q(observations)

        # Select Q-values for taken actions
        q_values = torch.stack([
            q[a] for q, a in zip(individual_q, actions)
        ])

        # Mix individual Q-values
        q_total = self.mixer(q_values)
        return q_total

    def select_actions(self, observations, epsilon=0.1):
        """Decentralized action selection."""
        actions = []
        for i, obs in enumerate(observations):
            if np.random.random() < epsilon:
                actions.append(np.random.randint(self.n_actions))
            else:
                q = self.agent_nets[i](obs)
                actions.append(q.argmax().item())
        return actions


def train_self_play(game, policy, episodes=1000):
    """Simple self-play training loop."""
    for ep in range(episodes):
        # Create opponent as copy of current policy
        opponent = copy.deepcopy(policy)
        opponent.eval()

        state = game.reset()
        done = False
        trajectory = []

        while not done:
            # Current policy plays as player 0
            if game.current_player == 0:
                action = policy.select_action(state)
            else:
                with torch.no_grad():
                    action = opponent.select_action(state)

            next_state, reward, done, _ = game.step(action)

            if game.current_player == 0:
                trajectory.append((state, action, reward))

            state = next_state

        # Update policy from trajectory
        policy.update(trajectory)

        if ep % 100 == 0:
            print(f"Episode {ep}: evaluating against previous versions...")
```

---

## Common Misconceptions to Address

1. **"Just run single-agent algorithms independently"**: Independent learning ignores non-stationarity. It can fail or oscillate.

2. **"Multi-agent RL needs explicit communication"**: Agents can coordinate through behavior without explicit messages.

3. **"Nash equilibrium is the goal"**: Finding Nash equilibria is computationally hard. We often settle for approximate or cooperative solutions.

4. **"Self-play only works for two-player games"**: Population-based training extends self-play to many agents.

---

## Exercises

### Conceptual (3-5 questions)
1. Why does independent Q-learning face non-stationarity?
2. What's the difference between a Nash equilibrium and a Pareto optimal solution?
3. How does CTDE address the scalability problem?

### Coding (2-3 challenges)
1. Implement independent Q-learning for a two-agent gridworld
2. Create a simple self-play training loop for tic-tac-toe
3. Implement a cooperative multi-agent task and compare independent vs joint training

### Exploration (1-2 open-ended)
1. Design a multi-agent scenario where cooperation is essential. What reward structure would encourage cooperation?

---

## Subsection Breakdown

### Subsection 1: The Multi-Agent Setting
- From single to multiple agents
- Cooperative, competitive, and mixed games
- Markov games formulation
- Interactive: cooperative vs competitive demo

### Subsection 2: Independent Learning
- The naive approach: independent learners
- Non-stationarity problem
- When independent learning works
- Interactive: non-stationarity visualization

### Subsection 3: Centralized Training, Decentralized Execution
- The CTDE paradigm
- Value decomposition (QMIX intuition)
- Training together, acting alone
- Implementation sketch

### Subsection 4: Self-Play and Competition
- Self-play for games
- Population-based training
- Emergent strategies
- AlphaStar and OpenAI Five
- Interactive: self-play training

---

## Additional Context for AI

- Multi-agent RL is increasingly important (robotics, games, LLMs).
- The non-stationarity problem is key—make it visceral.
- CTDE is the dominant paradigm—explain it clearly.
- Self-play is beautiful—AlphaGo/AlphaStar are inspiring examples.
- Keep the game theory light—focus on RL aspects.
- Simple gridworlds make concepts clear before scaling up.
- Emergent behavior is a highlight—show surprising outcomes.

---

## Quality Checklist

- [ ] Multi-agent setting properly formulated
- [ ] Cooperative/competitive/mixed distinguished
- [ ] Non-stationarity problem understood
- [ ] CTDE paradigm explained
- [ ] Self-play covered
- [ ] Simple implementations provided
- [ ] Interactive demos specified
- [ ] Emergent behavior showcased
