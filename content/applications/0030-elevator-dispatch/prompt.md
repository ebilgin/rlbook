# Elevator Dispatch Application - Generation Prompt

## Context

This application demonstrates how to formulate a real-world multi-agent coordination problem (elevator dispatch) as a reinforcement learning problem. It was chosen because:

1. **Universal relatability**: Everyone has experienced elevator systems
2. **Rich pedagogy**: Demonstrates multi-agent coordination, reward shaping, baseline comparisons
3. **Visual appeal**: Animated building with moving elevators makes for compelling visualizations
4. **Practical relevance**: Actual elevator systems use similar approaches
5. **Not a toy problem**: Avoids CartPole-style trivial examples

## Learning Objectives

By the end of this application, readers should be able to:

1. Formulate a multi-agent coordination problem as a Dec-POMDP
2. Design reward functions that balance competing objectives (wait time, energy, fairness)
3. Implement independent Q-learning with shared replay buffers
4. Understand when simple baselines (SCAN, nearest-car) are sufficient vs when RL adds value
5. Recognize deployment challenges (sim-to-real gap, safety constraints, monitoring)

## Content Structure

### 1. Introduction (Hook)
- **The Daily Frustration**: Relatable elevator waiting experience
- **Why RL?**: Limitations of rule-based algorithms (FCFS, SCAN)
- **The Challenge**: Multi-objective optimization under uncertainty

### 2. Problem Domain
- **Building specs**: 10 floors, 3 elevators, capacity constraints
- **Traffic patterns**: Morning rush, lunch, evening rush, quiet
- **Performance metrics**: Wait time, throughput, utilization, energy

### 3. MDP Formulation
- **State space**: Per-elevator observations (own state + global requests + other elevators + traffic pattern)
- **Action space**: Target floor selection (discrete 0-9)
- **Reward design**: Multi-component reward (wait time penalty + delivery bonus + starvation prevention + energy cost)
- **Episode structure**: 300 timesteps, Poisson arrivals

### 4. Multi-Agent Challenge
- **Credit assignment**: Which elevator deserves credit for improvement?
- **Non-stationarity**: Other elevators' policies changing during training
- **Coordination without communication**: Implicit coordination through shared experiences
- **Safe exploration**: Can't make passengers wait too long while exploring

### 5. Baseline Approaches
- **Random**: Baseline to beat
- **Nearest Car (FCFS)**: Intuitive heuristic
- **SCAN**: Elevator algorithm (continue in direction until done)
- **Performance comparison**: Establishes what RL needs to beat

### 6. RL Solution
- **Independent Q-learning**: Each elevator has own Q-network
- **Shared replay buffer**: Learn from all experiences
- **Network architecture**: [128, 128] fully connected
- **Training details**: 1000 episodes, ε-greedy exploration

### 7. Results
- **Metrics table**: DQN vs baselines on wait time and throughput
- **Analysis**: What did the agent learn? (implicit zoning, anticipatory positioning)
- **Emergent coordination**: Behaviors not explicitly programmed

### 8. Challenges & Solutions
- **Slow initial learning**: Guided buffer initialization
- **Non-stationarity**: Shared buffer + target networks
- **Exploration in production**: Low ε + safety fallbacks

### 9. Deployment Considerations
- **Sim-to-real transfer**: Domain randomization, continuous learning
- **Safety & failsafes**: Hard timeouts, emergency overrides
- **Monitoring**: Key metrics to track in production

### 10. Extensions
- Larger buildings (scaling up)
- Express elevators (skip floors)
- Multi-objective optimization (Pareto fronts)
- Destination dispatch (full observability)
- Lifelong learning (adaptation)

### 11. Try It Yourself
- Code examples (training script)
- Exercises (reward shaping, architecture experiments, baseline improvements)
- Colab notebook link

## Mathematical Depth

**Intuition Layer**:
- Use plain English for core concepts
- Tables for comparisons
- Visual analogies (no equations)

**Mathematical Layer**:
- Formalize as Dec-POMDP
- Show observation vector composition
- Q-learning update rule
- ε-greedy policy
- Multi-objective reward function

**Implementation Layer**:
- Complete training loop code
- Reference to tested implementation
- Hyperparameter specifications
- Evaluation scripts

## Implementation References

The content references these code files:
- `code/rlbook/envs/elevator.py` - Gymnasium environment
- `code/rlbook/agents/elevator_dqn.py` - Multi-agent DQN
- `code/rlbook/examples/train_elevator.py` - Training script
- `tests/test_elevator.py` - Test suite

## Interactive Elements (Placeholders)

1. **ElevatorSimulation**: Live building view showing elevators moving, passengers waiting
2. **TrainingProgress**: Episode rewards, wait times, epsilon decay over 1000 episodes
3. **PolicyComparison**: Side-by-side RL vs baselines with same traffic

## Cross-References

**Prerequisites**:
- [Deep Q-Networks](/chapters/deep-q-networks) - DQN algorithm foundation
- [Exploration-Exploitation](/chapters/exploration-exploitation) - ε-greedy policies

**Related Applications**:
- Traffic Signal Control (similar multi-agent coordination)
- Warehouse Robotics (fleet coordination)

**Related Papers**:
- MADDPG (Lowe et al., 2017)
- QMIX (Rashid et al., 2018)

## Key Pedagogical Decisions

1. **Start with relatability**: Everyone has elevator frustrations → instant engagement
2. **Show baseline failures**: Demonstrate *why* RL is needed, not just that it works
3. **Three-layer structure**: Allow readers to control depth
4. **Reward engineering spotlight**: Highlight design decisions and tradeoffs
5. **Deployment reality**: Don't oversell - show sim-to-real challenges
6. **Emergent behavior**: Emphasize what *wasn't* programmed but emerged from learning

## Common Pitfalls to Avoid

1. ❌ Don't use `\begin{cases}` in LaTeX (breaks MDX parser) → use bullet lists
2. ❌ Don't make elevators too smart (violates partial observability)
3. ❌ Don't skip baselines (readers need context for RL performance)
4. ❌ Don't oversimplify deployment (sim-to-real is hard!)
5. ✅ Do show actual code (readers can run it)
6. ✅ Do admit limitations (RL isn't always best)

## Iteration Notes

### 2026-02-14: Initial Implementation
- Created comprehensive application guide following CONTENT_TYPES.md pattern
- Implemented full Python environment, agent, and training script
- All tests passing (14/14 pytest tests)
- Build succeeds (`npm run build`)
- Fixed MDX parsing issue with `\begin{cases}` (used bullet list instead)
- Focused on content first, interactive components pending

### Future Iterations
- Add ElevatorSimulation React component for live visualization
- Create TrainingProgress component showing learning curves
- Build PolicyComparison component for side-by-side algorithm demos
- Develop Colab notebook for hands-on training
- Potential: Add video of trained agent in action

## Generation Instructions

When regenerating this content:

1. **Read foundation docs**:
   - `/prompts/PRINCIPLES.md` - Core principles
   - `/prompts/STYLE_GUIDE.md` - Writing style
   - `/prompts/MDX_AUTHORING.md` - **Critical** for avoiding syntax errors
   - `/prompts/MATH_CONVENTIONS.md` - Notation standards

2. **Follow structure**: Use the 11-section structure above

3. **Maintain tone**: Knowledgeable friend, not textbook or blog

4. **Code examples**: Reference actual working code in `/code/rlbook/`

5. **Three layers**: Wrap content in `<Intuition>`, `<Mathematical>`, `<Implementation>`

6. **Test build**: Always run `npm run build` before considering complete

7. **Cross-reference**: Link to prerequisite chapters and related content

## Success Criteria

✅ Build succeeds without errors
✅ All three complexity layers present
✅ Code examples are runnable
✅ Baselines clearly explained
✅ Deployment challenges acknowledged
✅ Mathematics are correct and well-explained
✅ No `\begin{cases}` or other MDX-breaking constructs
✅ Interactive component placeholders (even if not yet implemented)

---

This prompt documents the design decisions and structure for the Elevator Dispatch application. Use it to regenerate or update this content while maintaining consistency with the project's pedagogical philosophy.
