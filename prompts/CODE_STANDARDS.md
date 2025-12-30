# Code Standards

Guidelines for all code examples in rlbook.ai content.

## General Principles

### Educational First, Production Second

Code examples exist to teach concepts. Prioritize clarity over:
- Performance optimizations
- Edge case handling (unless teaching edge cases)
- Production-ready error handling
- Framework-specific patterns

**Good:**
```python
def q_learning_update(Q, state, action, reward, next_state, alpha=0.1, gamma=0.99):
    """Update Q-value using the Q-learning rule."""
    best_next_value = max(Q[next_state].values())
    td_target = reward + gamma * best_next_value
    Q[state][action] += alpha * (td_target - Q[state][action])
```

**Overly Production:**
```python
def q_learning_update(
    Q: Dict[State, Dict[Action, float]],
    state: State,
    action: Action,
    reward: float,
    next_state: State,
    alpha: float = 0.1,
    gamma: float = 0.99,
    done: bool = False,
    clip_range: Optional[Tuple[float, float]] = None,
) -> float:
    """Update Q-value using Q-learning rule with full type hints and options."""
    # ... 30 more lines of production code
```

### Complete and Runnable

Every code block should work if copied. Never use:
- `...` to skip implementation
- `# TODO: implement`
- `pass` as placeholder
- Undefined variables from "previous examples"

Exception: When explicitly showing a pattern or interface, mark it clearly:
```python
# PATTERN: This shows the structure, not runnable code
class CustomEnvironment:
    def step(self, action): ...
    def reset(self): ...
```

## Python Standards

### Style

- Follow PEP 8 with 88-character line limit (Black default)
- Use type hints for function signatures in Implementation sections
- Omit type hints in Intuition sections for clarity
- Prefer f-strings over .format() or %

### Imports

Show imports in the first code block of each chapter section:

```python
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
```

For subsequent blocks, assume imports are present unless using a new library.

### Naming Conventions

| Type | Convention | Example |
|------|------------|---------|
| Variables | snake_case | `learning_rate`, `next_state` |
| Functions | snake_case | `compute_td_error()` |
| Classes | PascalCase | `GridWorldEnv`, `DQNAgent` |
| Constants | UPPER_SNAKE | `MAX_EPISODES`, `GAMMA` |

RL-specific names:
```python
# States and actions
state, s = ...           # Current state
next_state, s_prime = ... # Next state
action, a = ...          # Current action

# Values
q_value = ...            # Single Q-value
Q = ...                  # Q-table or Q-function
V = ...                  # Value function/table

# Parameters
alpha = 0.1              # Learning rate (not lr)
gamma = 0.99             # Discount factor
epsilon = 0.1            # Exploration rate (not eps)
```

### Comments

Explain *why*, not *what*:

**Bad:**
```python
# Increment counter
counter += 1
```

**Good:**
```python
# Track episodes for learning rate decay
episode_count += 1
```

Use docstrings for functions that will be reused:
```python
def epsilon_greedy(Q, state, epsilon):
    """
    Select action using epsilon-greedy policy.

    Args:
        Q: Q-table as dict[state][action] -> value
        state: Current state
        epsilon: Exploration probability

    Returns:
        Selected action
    """
```

## JavaScript/TypeScript Standards

### For Interactive Demos

```typescript
// Use TypeScript for component code
interface QLearningParams {
  alpha: number;      // Learning rate
  gamma: number;      // Discount factor
  epsilon: number;    // Exploration rate
}

function updateQValue(
  Q: Map<string, number>,
  state: string,
  action: string,
  reward: number,
  nextState: string,
  params: QLearningParams
): void {
  const key = `${state}-${action}`;
  const currentQ = Q.get(key) ?? 0;
  const maxNextQ = getMaxQ(Q, nextState);

  const tdTarget = reward + params.gamma * maxNextQ;
  const newQ = currentQ + params.alpha * (tdTarget - currentQ);

  Q.set(key, newQ);
}
```

### TensorFlow.js Patterns

```javascript
// Always dispose tensors to prevent memory leaks
const prediction = tf.tidy(() => {
  const stateTensor = tf.tensor2d([state]);
  const qValues = model.predict(stateTensor);
  return qValues.argMax(1).dataSync()[0];
});

// For training loops, use tf.tidy and explicit dispose
async function trainStep(batch) {
  const loss = tf.tidy(() => {
    // Compute loss
    return model.trainOnBatch(batch.states, batch.targets);
  });

  // Dispose batch tensors if not needed
  batch.states.dispose();
  batch.targets.dispose();

  return loss;
}
```

## Code Block Types

### Concept Blocks (Intuition Layer)

Minimal, focused on the key idea:

```python
# Q-learning update: learn from the best possible next action
Q[state][action] += alpha * (reward + gamma * max(Q[next_state]) - Q[state][action])
```

### Walkthrough Blocks (Mathematical Layer)

Annotated, showing each step:

```python
def q_learning_update(Q, s, a, r, s_prime, alpha, gamma):
    # Step 1: Find the best action value in the next state
    best_next_value = max(Q[s_prime].values())

    # Step 2: Compute the TD target (what we think the value should be)
    td_target = r + gamma * best_next_value

    # Step 3: Compute TD error (how wrong we were)
    td_error = td_target - Q[s][a]

    # Step 4: Update towards the target
    Q[s][a] = Q[s][a] + alpha * td_error

    return td_error  # Useful for monitoring
```

### Implementation Blocks (Implementation Layer)

Complete, practical implementations:

```python
class QLearningAgent:
    """Tabular Q-learning agent with epsilon-greedy exploration."""

    def __init__(self, n_actions, alpha=0.1, gamma=0.99, epsilon=0.1):
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.Q = defaultdict(lambda: np.zeros(n_actions))

    def select_action(self, state):
        """Epsilon-greedy action selection."""
        if np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)
        return np.argmax(self.Q[state])

    def update(self, state, action, reward, next_state, done):
        """Perform Q-learning update."""
        if done:
            td_target = reward
        else:
            td_target = reward + self.gamma * np.max(self.Q[next_state])

        td_error = td_target - self.Q[state][action]
        self.Q[state][action] += self.alpha * td_error

        return td_error
```

## Interactive Code Considerations

### Browser Execution

When code runs in the browser:
- Avoid blocking operations (use async/await)
- Limit iterations per frame for smooth animation
- Provide progress callbacks for long operations
- Use Web Workers for heavy computation

```javascript
// Good: Non-blocking training loop
async function trainEpisode(agent, env, updateCallback) {
  let state = env.reset();
  let done = false;

  while (!done) {
    const action = agent.selectAction(state);
    const [nextState, reward, isDone] = env.step(action);

    agent.update(state, action, reward, nextState, isDone);

    state = nextState;
    done = isDone;

    // Allow UI updates
    await updateCallback(state, action, reward);
    await new Promise(r => setTimeout(r, 0));
  }
}
```

### State Visualization

Always provide a way to inspect agent state:

```javascript
// Expose Q-values for visualization
function getQTableForVisualization(agent) {
  return Object.entries(agent.Q).map(([state, actions]) => ({
    state,
    values: actions,
    bestAction: actions.indexOf(Math.max(...actions))
  }));
}
```

## Testing Expectations

While we don't require tests in content, code should be:
- Manually tested before publication
- Deterministic when seeded (`np.random.seed(42)`)
- Consistent across common environments (Python 3.9+, modern browsers)

## Version and Dependency Notes

Specify versions when relevant:
```python
# Requires: Python 3.9+, NumPy 1.21+, Gymnasium 0.29+
import gymnasium as gym
import numpy as np
```

For browser code, note compatibility:
```javascript
// Works in Chrome 113+, Firefox 141+, Edge 113+ (WebGPU)
// Falls back to WebGL in Safari
```
