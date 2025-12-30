# Content Types

rlbook.ai organizes content into six distinct categories, each serving a different learning purpose. Understanding these categories helps readers find the right content for their goals and helps contributors know where their work fits.

## Overview

| Category | Purpose | Reader Goal | Example |
|----------|---------|-------------|---------|
| **Chapters** | Teach concepts | "I want to understand Q-learning" | Q-Learning Basics |
| **Papers** | Analyze research | "I want to understand the DQN paper" | DQN Deep Dive |
| **Applications** | Formulate problems | "I want to apply RL to my robotics project" | RL for Inventory Management |
| **Infrastructure** | Scale & deploy | "I want to train on 100 GPUs" | Distributed RL with Ray |
| **Environments** | Experiment hands-on | "I want a playground to test ideas" | GridWorld Playground |
| **Code** | Run & extend | "I want tested implementations to build on" | rlbook Python package |

---

## 1. Chapters

### Purpose
Teach reinforcement learning concepts progressively, building from foundations to advanced topics. Each chapter synthesizes knowledge from multiple sources (papers, textbooks, tutorials) to provide comprehensive understanding.

### Characteristics
- **Progressive complexity**: Builds on previous chapters
- **Three layers**: Intuition → Mathematical → Implementation
- **Multiple sources**: Draws from papers, textbooks, and best practices
- **Exercises**: Conceptual questions and coding challenges
- **Interactive demos**: Browser-based visualizations

### Structure
```
content/chapters/XXXX-chapter-slug/
├── index.mdx          # Main content
├── prompt.md          # AI generation prompt
├── exercises/         # Additional exercises
└── assets/            # Images, diagrams
```

### When to Create a Chapter
- Teaching a core RL concept (TD learning, policy gradients)
- Covering a category of methods (value-based methods, actor-critic)
- Explaining foundational knowledge (MDPs, Bellman equations)

### Examples
- Introduction to Reinforcement Learning
- Multi-Armed Bandits
- Deep Q-Networks
- Policy Gradient Methods

---

## 2. Papers

### Purpose
Provide deep analysis of seminal or important research papers. Focus on understanding one specific contribution in depth: the problem it solved, the method, key insights, and modern relevance.

### Characteristics
- **Single paper focus**: Deep dive into one contribution
- **Historical context**: What problem existed, what came before
- **Method breakdown**: Step-by-step explanation of the approach
- **Key insights**: What made this paper important
- **Modern relevance**: How it influences current research
- **Critical analysis**: Limitations and what came after

### Structure
```
content/papers/XXXX-paper-slug/
├── index.mdx          # Analysis content
├── prompt.md          # AI generation prompt
├── summary.md         # One-page summary
└── assets/            # Figures from paper, diagrams
```

### When to Create a Paper Analysis
- Seminal papers that defined the field (DQN, PPO, AlphaGo)
- Papers frequently cited or referenced in chapters
- Papers that introduced widely-used techniques
- Recent papers with significant impact

### Examples
- "Playing Atari with Deep Reinforcement Learning" (DQN)
- "Proximal Policy Optimization Algorithms" (PPO)
- "Mastering the Game of Go" (AlphaGo)
- "Decision Transformer"

---

## 3. Applications

### Purpose
Teach how to formulate real-world problems as reinforcement learning problems. Focus on the full pipeline from problem definition to deployment considerations, not just the algorithm.

### Characteristics
- **Problem-first**: Starts with a real domain, not an algorithm
- **Complete formulation**: State, action, reward, constraints
- **Practical challenges**: Data, simulation, evaluation
- **Domain knowledge**: What RL practitioners need to know about the domain
- **Case study format**: End-to-end walkthrough
- **Deployment considerations**: How to actually use this

### Structure
```
content/applications/XXXX-application-slug/
├── index.mdx          # Main content
├── prompt.md          # AI generation prompt
├── formulation.md     # Detailed MDP formulation
├── code/              # Reference implementation
└── assets/            # Domain-specific diagrams
```

### Key Sections for Every Application
1. **Domain Introduction**: What is this problem? Why does it matter?
2. **State Space**: What information does the agent observe?
3. **Action Space**: What can the agent do?
4. **Reward Design**: How do we encode the objective?
5. **Challenges**: What makes this hard? (sparse rewards, safety, etc.)
6. **Baseline Approaches**: What do people do without RL?
7. **RL Approaches**: What algorithms fit? Why?
8. **Evaluation**: How do we know if it's working?
9. **Deployment**: How do we put this in production?

### When to Create an Application
- Important RL application domain (robotics, games, finance)
- Novel problem formulation worth documenting
- Common use case with non-obvious design decisions

### Examples
- RL for Robotics Manipulation
- Recommendation Systems as Contextual Bandits
- Autonomous Trading Agents
- Game AI for Strategy Games
- Resource Scheduling with RL

---

## 4. Infrastructure

### Purpose
Cover the engineering aspects of reinforcement learning: scaling training, managing experiments, deploying models, and building production systems. Focus on "how to make it work at scale" rather than algorithms.

### Characteristics
- **Engineering focus**: Code, systems, tools
- **Practical guidance**: Specific tools and configurations
- **Scale considerations**: From laptop to cluster
- **Production concerns**: Monitoring, debugging, reliability
- **Tool comparisons**: When to use what

### Structure
```
content/infrastructure/XXXX-infra-slug/
├── index.mdx          # Main content
├── prompt.md          # AI generation prompt
├── code/              # Configuration files, scripts
└── assets/            # Architecture diagrams
```

### Key Topics
- **Distributed Training**: Ray, multi-GPU, cluster setup
- **Experiment Tracking**: Weights & Biases, MLflow, logging
- **Hyperparameter Tuning**: Grid search, Bayesian optimization, PBT
- **Simulation**: Building environments, parallel simulation
- **Deployment**: Model serving, online learning, A/B testing
- **Debugging**: Visualization, common issues, profiling

### When to Create an Infrastructure Guide
- Common scaling challenge (distributed training)
- Important tooling (experiment tracking)
- Production deployment patterns
- Debugging and monitoring approaches

### Examples
- Distributed RL Training with Ray RLlib
- Experiment Tracking for RL Projects
- Building Custom Gym Environments
- Deploying RL Models in Production
- Hyperparameter Tuning Strategies

---

## 5. Environments

### Purpose
Provide interactive playgrounds for experimentation. Each environment is a self-contained world where readers can test ideas, visualize algorithms, and build intuition through hands-on exploration.

### Characteristics
- **Interactive**: Run in the browser with adjustable parameters
- **Well-documented**: Clear specification of state, action, reward
- **Baseline results**: What performance to expect from standard algorithms
- **Suggested experiments**: Guided exploration
- **Progressive complexity**: Simple to complex variants

### Structure
```
content/environments/XXXX-env-slug/
├── index.mdx          # Documentation and usage
├── prompt.md          # AI generation prompt
├── specification.md   # Formal MDP specification
├── baselines.md       # Expected performance
├── src/               # Environment implementation (TF.js)
└── assets/            # Screenshots, diagrams
```

### Key Sections for Every Environment
1. **Overview**: What is this environment? What does it teach?
2. **Visual Preview**: Screenshot or GIF of the environment
3. **Specification**: State space, action space, rewards, dynamics
4. **Getting Started**: How to use it (interactive or code)
5. **Baseline Results**: Performance of standard algorithms
6. **Suggested Experiments**: What to try
7. **Variants**: Harder/easier versions, modifications

### Types of Environments
- **Classic**: GridWorld, CliffWalking, FrozenLake
- **Control**: CartPole, MountainCar, Pendulum (simplified)
- **Custom**: Book-specific environments for teaching
- **Mini-games**: Simple games for demonstration

### When to Create an Environment
- Teaching a specific concept (exploration, credit assignment)
- Providing a playground for a chapter
- Demonstrating algorithm differences
- Creating a challenge for readers

### Examples
- GridWorld Playground
- Multi-Armed Bandit Testbed
- CliffWalking: Safe vs Optimal Paths
- CartPole Balance Challenge
- Trading Simulation Environment

---

## 6. Code

### Purpose
Provide production-grade, tested Python implementations that readers can install, run, and extend. The code package serves as a reference implementation for concepts taught in chapters and can be used for real projects.

### Characteristics
- **Tested**: Full pytest coverage for reliability
- **Type-hinted**: Modern Python with type annotations
- **Documented**: Comprehensive docstrings and inline comments
- **Consistent**: Same examples as chapters (GridWorld, bandits)
- **Gymnasium-compatible**: Environments follow the gym.Env interface
- **PyTorch-based**: Deep learning agents use PyTorch

### Structure
```
code/
├── rlbook/                  # Main package
│   ├── envs/                # Gymnasium-compatible environments
│   │   ├── gridworld.py     # GridWorld, CliffWalking
│   │   └── bandits.py       # Multi-armed, contextual bandits
│   ├── agents/              # Agent implementations
│   │   ├── q_learning.py    # Q-Learning, SARSA, Expected SARSA
│   │   └── dqn.py           # DQN, Double DQN
│   ├── utils/               # Utilities
│   │   ├── replay_buffer.py # Simple and prioritized buffers
│   │   └── plotting.py      # Visualization helpers
│   └── examples/            # Training scripts
│       └── train_gridworld.py
├── tests/                   # pytest test suite
├── pyproject.toml           # Package configuration
└── README.md                # Installation and usage guide
```

### When to Add Code
- Reference implementation for a chapter algorithm
- New environment for experiments
- Utility functions used across multiple agents
- Training scripts demonstrating complete workflows

### Installation & Usage
```bash
cd code
pip install -e .          # Install in development mode
pytest                     # Run test suite
python -m rlbook.examples.train_gridworld  # Run example
```

### Referencing Code from Chapters
```mdx
<Implementation>
For the full tested implementation, see
[code/rlbook/agents/q_learning.py](https://github.com/ebilgin/rlbook/tree/main/code/rlbook/agents/q_learning.py).
</Implementation>
```

---

## Content Relationships

Content types reference each other to create a cohesive learning experience:

```
Chapters ←→ Papers ←→ Code
    ↓           ↓        ↑
Applications ← Environments
    ↓
Infrastructure
```

- **Chapters** cite **Papers** for deep dives on specific methods
- **Chapters** use **Environments** for interactive demonstrations
- **Chapters** reference **Code** for tested implementations
- **Code** implements algorithms from **Chapters** and **Papers**
- **Applications** reference **Chapters** for algorithm background
- **Applications** may introduce custom **Environments**
- **Infrastructure** supports scaling work from all other categories

### Cross-Reference Patterns

```mdx
{/* In a Chapter, reference a Paper */}
<CrossRef type="paper" slug="dqn-paper">
  For the original DQN paper, see our deep dive analysis.
</CrossRef>

{/* In a Chapter, embed an Environment */}
<Environment slug="gridworld" variant="basic" />

{/* In an Application, reference a Chapter */}
<CrossRef type="chapter" slug="reward-shaping">
  See the Reward Engineering chapter for shaping techniques.
</CrossRef>
```

---

## Contribution Guidelines by Type

**Important:** All content is written in MDX. Before contributing, review:
- [MDX_AUTHORING.md](../prompts/MDX_AUTHORING.md) - Critical syntax rules to avoid build failures

### Chapters
- Follow the chapter prompt template
- Include all three complexity layers
- Add at least 3 exercises
- Suggest interactive demo specifications
- **Test build before committing** (`npm run build`)

### Papers
- Include proper citation
- Don't just summarize—add value with analysis
- Connect to other content in the book
- Note what the paper got wrong or what changed since

### Applications
- Start from a real problem, not an algorithm
- Be explicit about design decisions and alternatives
- Include gotchas and common mistakes
- Provide working code

### Infrastructure
- Be specific about versions and configurations
- Include both "quick start" and "production" guidance
- Compare alternatives fairly
- Update as tools evolve

### Environments
- Make them run in the browser (TensorFlow.js)
- Document the MDP formally
- Provide baseline results
- Suggest meaningful experiments

### Code
- Include pytest tests for all new code
- Add type hints and docstrings
- Follow existing patterns (check similar files)
- Update `__init__.py` exports
- Run `pytest` before committing

---

## Status Tracking

All content types use the same status system:

| Status | Icon | Meaning |
|--------|------|---------|
| `draft` | 📝 | AI-generated, pending review |
| `editor_reviewed` | ✅ | Reviewed by editor |
| `community_reviewed` | 👥 | Incorporates community feedback |
| `verified` | 🔒 | Code tested, demos working, ready for production |
