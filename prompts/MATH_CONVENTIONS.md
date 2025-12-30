# Mathematical Notation Conventions

Consistent notation across all rlbook.ai content. Follow these conventions strictly.

## Core RL Notation

### States, Actions, Rewards

| Symbol | Meaning | Notes |
|--------|---------|-------|
| $s, s'$ | State, next state | Lowercase for instances |
| $\mathcal{S}$ | State space | Calligraphic for sets |
| $a, a'$ | Action, next action | |
| $\mathcal{A}$ | Action space | $\mathcal{A}(s)$ if state-dependent |
| $r$ | Reward (single step) | Lowercase scalar |
| $R(s,a)$ | Reward function | $R(s,a,s')$ if next-state dependent |
| $R_t$ | Reward at time $t$ | Random variable |

### Returns and Value Functions

| Symbol | Meaning | Definition |
|--------|---------|------------|
| $G_t$ | Return from time $t$ | $G_t = R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + ...$ |
| $V(s)$ | State-value function | $V(s) = \mathbb{E}[G_t \| S_t = s]$ |
| $V^\pi(s)$ | Value under policy $\pi$ | Superscript for policy |
| $V^*(s)$ | Optimal value function | Asterisk for optimal |
| $Q(s,a)$ | Action-value function | |
| $Q^\pi(s,a)$ | Q-value under policy $\pi$ | |
| $Q^*(s,a)$ | Optimal Q-function | |

### Policy

| Symbol | Meaning | Notes |
|--------|---------|-------|
| $\pi$ | Policy | Function $\pi: \mathcal{S} \rightarrow \mathcal{A}$ or distribution |
| $\pi(a\|s)$ | Probability of $a$ in state $s$ | Stochastic policy |
| $\pi(s)$ | Action in state $s$ | Deterministic policy |
| $\pi^*$ | Optimal policy | |
| $\mu$ | Behavior policy | For off-policy learning |
| $\pi_\theta$ | Parameterized policy | $\theta$ = parameters |

### Learning Parameters

| Symbol | Meaning | Typical Values |
|--------|---------|----------------|
| $\alpha$ | Learning rate | 0.001 - 0.1 |
| $\gamma$ | Discount factor | 0.9 - 0.999 |
| $\epsilon$ | Exploration rate | 0.01 - 0.1 |
| $\lambda$ | Eligibility trace decay | 0 - 1 |
| $\tau$ | Temperature / soft update rate | Context dependent |

### Time and Episodes

| Symbol | Meaning |
|--------|---------|
| $t$ | Time step |
| $T$ | Terminal time step |
| $k$ | Episode number |
| $n$ | Number of steps (n-step returns) |

## Deep RL Notation

### Networks and Parameters

| Symbol | Meaning |
|--------|---------|
| $\theta$ | Network parameters (general) |
| $\theta^-$ | Target network parameters |
| $\phi$ | Critic parameters (actor-critic) |
| $\psi$ | Encoder / other auxiliary parameters |
| $\nabla_\theta$ | Gradient with respect to $\theta$ |

### Function Approximation

| Symbol | Meaning |
|--------|---------|
| $\hat{V}(s; \theta)$ | Approximate value function |
| $\hat{Q}(s,a; \theta)$ | Approximate Q-function |
| $\mathbf{x}(s)$ | Feature vector for state $s$ |
| $\mathbf{w}$ | Linear function weights |

## Probability and Expectation

### Distributions

| Symbol | Meaning |
|--------|---------|
| $p(s'\|s,a)$ | Transition probability |
| $\mathbb{P}$ | Probability measure |
| $\mathbb{E}$ | Expectation |
| $\mathbb{E}_\pi$ | Expectation under policy $\pi$ |
| $\sim$ | "Distributed as" or "sampled from" |
| $\mathcal{N}(\mu, \sigma^2)$ | Normal distribution |
| $\mathcal{U}(a, b)$ | Uniform distribution |

### Common Expressions

**TD Error:**
$$\delta_t = R_{t+1} + \gamma V(S_{t+1}) - V(S_t)$$

**Bellman Equation (expectation form):**
$$V^\pi(s) = \sum_a \pi(a|s) \sum_{s',r} p(s',r|s,a)[r + \gamma V^\pi(s')]$$

**Bellman Optimality:**
$$V^*(s) = \max_a \sum_{s',r} p(s',r|s,a)[r + \gamma V^*(s')]$$

**Q-Learning Update:**
$$Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \alpha[R_{t+1} + \gamma \max_a Q(S_{t+1}, a) - Q(S_t, A_t)]$$

**Policy Gradient:**
$$\nabla_\theta J(\theta) = \mathbb{E}_{\pi_\theta}\left[\nabla_\theta \log \pi_\theta(a|s) Q^{\pi_\theta}(s,a)\right]$$

## Formatting Rules

### Inline vs Display

**Inline** (single variables or short expressions):
- "The learning rate $\alpha$ controls step size"
- "We want to maximize $Q(s,a)$"

**Display** (important equations, derivations):
$$V(s) = \mathbb{E}\left[\sum_{k=0}^{\infty} \gamma^k R_{t+k+1} \middle| S_t = s\right]$$

### Equation Numbering

Number only equations that are referenced later:

$$Q(s,a) = r + \gamma \max_{a'} Q(s', a') \tag{1}$$

Then reference as "Equation (1)" or "the Q-learning target (1)".

### Breaking Down Equations

For complex equations, use underbrace for explanation:

$$Q(s,a) \leftarrow Q(s,a) + \alpha[\underbrace{r + \gamma \max_{a'} Q(s',a')}_{\text{TD target}} - \underbrace{Q(s,a)}_{\text{current estimate}}]$$

### Multi-line Equations

Use aligned environment for derivations:

$$\begin{aligned}
V(s) &= \mathbb{E}[G_t | S_t = s] \\
&= \mathbb{E}[R_{t+1} + \gamma G_{t+1} | S_t = s] \\
&= \mathbb{E}[R_{t+1} + \gamma V(S_{t+1}) | S_t = s]
\end{aligned}$$

## Symbol Conflicts to Avoid

| Avoid | Problem | Use Instead |
|-------|---------|-------------|
| $r$ for both reward and radius | Confusion | $r$ for reward, $\rho$ for radius |
| $T$ for both temperature and terminal time | Context clash | $\tau$ for temperature, $T$ for time |
| $A$ for both action and advantage | Common conflict | $A$ for advantage, $a$ for action |
| $p$ for both probability and parameters | Overloaded | $p$ for probability, $\theta$ for parameters |

## Subscript and Superscript Conventions

- **Time index**: Subscript $t$ → $V_t$, $Q_t$, $\pi_t$
- **Policy**: Superscript $\pi$ → $V^\pi$, $Q^\pi$
- **Optimal**: Superscript $*$ → $V^*$, $Q^*$, $\pi^*$
- **Parameters**: Subscript or semicolon → $Q_\theta$ or $Q(s,a;\theta)$
- **Components**: Subscript with comma → $\theta_{i,j}$

## When Introducing New Notation

1. Introduce intuitively first: "We need a way to measure how good an action is..."
2. Define the symbol: "We call this the Q-function, written $Q(s,a)$"
3. Explain the notation: "The $Q$ stands for 'quality' of taking action $a$ in state $s$"
4. Give a concrete example with numbers
5. Show how it connects to previously defined terms
