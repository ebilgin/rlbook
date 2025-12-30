# Environments

Interactive playgrounds for experimenting with reinforcement learning algorithms.

## Purpose

Each environment provides:
- Browser-based interactive demo (TensorFlow.js)
- Formal MDP specification
- Baseline algorithm performance
- Suggested experiments
- Variants for different difficulty levels

## Structure

```
environments/
├── XXXX-env-slug/
│   ├── index.mdx          # Documentation
│   ├── prompt.md          # AI generation prompt
│   ├── specification.md   # Formal MDP spec
│   ├── baselines.md       # Expected performance
│   ├── src/               # TF.js implementation
│   └── assets/            # Screenshots, diagrams
```

## Planned Environments

### Classic Control
- [ ] **GridWorld** - The canonical teaching environment
- [ ] **CliffWalking** - Risk vs. reward visualization
- [ ] **FrozenLake** - Stochastic transitions
- [ ] **CartPole** - Continuous state, discrete action
- [ ] **MountainCar** - Sparse rewards, momentum

### Bandits
- [ ] **Multi-Armed Bandit Testbed** - Exploration strategies
- [ ] **Contextual Bandit Arena** - Personalization

### Custom Teaching Environments
- [ ] **RewardShaping Lab** - See how rewards affect behavior
- [ ] **ExplorationWorld** - Visualize exploration strategies
- [ ] **CreditAssignment Maze** - Delayed reward challenges

### Mini-Games
- [ ] **Snake** - Classic game with RL
- [ ] **Pong** - Two-player dynamics
- [ ] **Simple Trading** - Financial decisions

## Environment Specification Format

Every environment documents:

```yaml
name: GridWorld
state_space:
  type: discrete
  size: 16
  description: "Agent position on 4x4 grid"

action_space:
  type: discrete
  size: 4
  actions: [up, down, left, right]

reward:
  goal: +10
  step: -0.1
  wall: -0.5

dynamics:
  deterministic: true
  episode_length: 100

difficulty_variants:
  - easy: "4x4, no obstacles"
  - medium: "6x6, walls"
  - hard: "8x8, stochastic wind"
```

## Using Environments

### In Browser
Each environment page has an interactive demo. Adjust parameters, watch training, and experiment.

### In Code (Python)
```python
from rlbook.envs import GridWorld

env = GridWorld(size=4, obstacles=False)
state = env.reset()
action = env.action_space.sample()
next_state, reward, done, info = env.step(action)
```

### In Code (JavaScript)
```javascript
import { GridWorld } from '@rlbook/environments';

const env = new GridWorld({ size: 4 });
let state = env.reset();
const action = env.sampleAction();
const { nextState, reward, done } = env.step(action);
```

## Baseline Results

Each environment includes expected performance:

| Algorithm | GridWorld 4x4 | CliffWalking | CartPole |
|-----------|---------------|--------------|----------|
| Random | ~-50 | ~-100 | ~20 |
| Q-Learning | ~-5 | ~-15 | N/A |
| DQN | ~-3 | ~-13 | ~195 |
| Optimal | -3 | -13 | 500 |

## Contributing

See [CONTENT_TYPES.md](../../docs/CONTENT_TYPES.md) for guidelines on creating environments.

Key requirements:
- Must run in browser (TensorFlow.js)
- Must have formal MDP specification
- Must include baseline results
- Must suggest meaningful experiments
