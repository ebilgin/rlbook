# rlbook Code Library

Production-grade, tested implementations of RL algorithms and environments from rlbook.ai.

## Overview

This package provides:

- **Environments**: GridWorld, bandits, and custom teaching environments
- **Agents**: Q-Learning, SARSA, DQN implementations
- **Utils**: Replay buffers, plotting, and training utilities
- **Examples**: Runnable scripts demonstrating each algorithm

All code is designed to work coherently with the book's chapters, applications, and interactive demos.

## Installation

```bash
# From the rlbook repository root
pip install -e ./code

# Or install dependencies only
pip install -r code/requirements.txt
```

## Quick Start

```python
from rlbook.envs import GridWorld
from rlbook.agents import QLearningAgent

# Create environment
env = GridWorld(size=4)

# Create agent
agent = QLearningAgent(
    n_states=env.observation_space.n,
    n_actions=env.action_space.n,
    learning_rate=0.1,
    gamma=0.99,
    epsilon=0.1,
)

# Training loop
for episode in range(1000):
    state, _ = env.reset()
    done = False

    while not done:
        action = agent.select_action(state)
        next_state, reward, done, _, _ = env.step(action)
        agent.update(state, action, reward, next_state, done)
        state = next_state
```

## Package Structure

```
code/
├── rlbook/
│   ├── envs/           # Environment implementations
│   │   ├── gridworld.py    # GridWorld, CliffWalking
│   │   └── bandits.py      # Multi-armed, Bernoulli, Contextual bandits
│   ├── agents/         # Agent implementations
│   │   ├── q_learning.py   # Q-Learning, SARSA, Expected SARSA
│   │   └── dqn.py          # DQN, Double DQN
│   ├── utils/          # Shared utilities
│   │   ├── replay_buffer.py
│   │   └── plotting.py
│   └── examples/       # Runnable examples
│       └── train_gridworld.py
└── tests/              # Unit tests
```

## Relationship to Book Content

Code in this package is referenced throughout the book:

| Chapter | Code Reference |
|---------|----------------|
| Introduction to TD Learning | `rlbook/agents/q_learning.py` |
| Q-Learning Basics | `rlbook/agents/q_learning.py` |
| Deep Q-Networks | `rlbook/agents/dqn.py` |
| Multi-Armed Bandits | `rlbook/envs/bandits.py` |
| GridWorld Environment | `rlbook/envs/gridworld.py` |

## Running Examples

```bash
# Train Q-learning on GridWorld
python -m rlbook.examples.train_gridworld
```

## Testing

```bash
# Run all tests
pytest code/tests/

# Run with coverage
pytest code/tests/ --cov=rlbook
```

## Design Principles

1. **Educational clarity over optimization**: Code prioritizes readability
2. **Minimal dependencies**: Core algorithms use only NumPy and PyTorch
3. **Gymnasium compatible**: All environments follow the Gymnasium API
4. **Tested**: Every algorithm has unit tests verifying correctness
5. **Documented**: Docstrings explain the "why" not just the "what"

## Dependencies

- Python 3.9+
- NumPy
- PyTorch
- Gymnasium
- Matplotlib (for plotting)

## Contributing

See [CONTRIBUTING.md](../docs/CONTRIBUTING.md) for guidelines. Key points:

- All code must have tests
- Follow the existing code style
- Add docstrings with educational explanations
- Reference the corresponding chapter/content

## License

MIT License - see [LICENSE](../LICENSE) for details.
