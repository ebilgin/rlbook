# Chapter: Introduction to MDPs

## Chapter Metadata

**Chapter Number:** 04
**Title:** Introduction to Markov Decision Processes
**Section:** Markov Decision Processes
**Prerequisites:**
- Foundations (all 3 chapters)
**Estimated Reading Time:** 25 minutes

---

## Learning Objectives

By the end of this chapter, readers will be able to:

1. Explain why bandits are insufficient for sequential decision problems
2. Define all components of an MDP (states, actions, transitions, rewards, discount)
3. Construct simple MDPs from problem descriptions
4. Explain the Markov property and why it matters
5. Distinguish between episodic and continuing tasks

---

## Core Concepts to Cover

### Primary Concepts (Must Cover)
- [ ] From bandits to sequential decisions: why we need states
- [ ] The 5-tuple MDP definition: (S, A, P, R, γ)
- [ ] Transition dynamics: P(s'|s,a)
- [ ] Reward function: R(s,a,s') vs R(s,a) vs R(s)
- [ ] Discount factor γ and its interpretation
- [ ] The Markov property: the future depends only on the present
- [ ] Episodic vs continuing tasks

### Secondary Concepts (Cover if Space Permits)
- [ ] Finite vs infinite MDPs
- [ ] Deterministic vs stochastic transitions
- [ ] State vs observation (partially observable preview)

### Explicitly Out of Scope
- Value functions (next chapter)
- Bellman equations (chapter after that)
- Solving MDPs (Dynamic Programming section)

---

## Narrative Arc

### Opening Hook
"In bandits, every pull of the lever was independent—your choice didn't change the world. But what if your actions have lasting consequences? What if where you go affects where you can go next? Welcome to Markov Decision Processes, the mathematical language of sequential decision-making."

### Key Insight
The Markov property is both a simplification and a design choice. It says: if you know the current state, history doesn't matter for predicting the future. This isn't always true in the real world, but it's true *by construction* when we design our state representation well. Understanding MDPs means understanding how to define states.

### Closing Connection
"Now that we can formally describe sequential decision problems, we need a way to evaluate how good different states and actions are. That's where value functions come in—they're our way of asking 'how good is it to be here?'"

---

## Required Interactive Elements

### Demo 1: MDP Builder
- **Purpose:** Let users construct their own simple MDP
- **Interaction:**
  - Drag-and-drop to add states
  - Click to connect with transitions (set probabilities)
  - Assign rewards to transitions or states
  - Visualize the MDP graph
- **Expected Discovery:** MDPs are flexible—you can model many problems with them

### Demo 2: Markov Property Illustration
- **Purpose:** Show what Markov property means
- **Interaction:**
  - Show a sequence of states
  - Ask: "Given you're in state X, does knowing the history help predict the next state?"
  - Contrast Markovian vs non-Markovian examples
- **Expected Discovery:** The state should encode everything relevant about history

---

## Recurring Examples to Use

- **GridWorld:** Perfect for illustrating MDP concepts
  - States = positions
  - Actions = up/down/left/right
  - Transitions = deterministic or stochastic (slippery ice)
  - Rewards = -1 per step, +10 at goal, -10 at pit
- **Robot Navigation:** Real-world analogy
- **Simple Game:** Tic-tac-toe or similar to show sequential nature

---

## Cross-References

### Build On (Backward References)
- Chapter 1 (Intro): "Recall the agent-environment loop..."
- Chapter 2 (Framework): "We introduced states and transitions informally..."
- Foundations: "We saw exploration-exploitation in bandits..."

### Set Up (Forward References)
- Chapter 5 (Value Functions): "Next we'll measure how good states are..."
- Chapter 6 (Bellman): "The recursive structure of MDPs..."
- TD Learning: "When we don't know the transition probabilities..."

---

## Mathematical Depth

### Required Equations

1. **MDP tuple**: $\mathcal{M} = (\mathcal{S}, \mathcal{A}, P, R, \gamma)$

2. **Transition function**: $P(s' | s, a) = \Pr(S_{t+1} = s' | S_t = s, A_t = a)$

3. **Reward function** (expected reward): $R(s, a) = \mathbb{E}[R_{t+1} | S_t = s, A_t = a]$

4. **Discounted return**: $G_t = R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + \ldots = \sum_{k=0}^{\infty} \gamma^k R_{t+k+1}$

5. **Markov property**: $P(S_{t+1} | S_t, A_t, S_{t-1}, A_{t-1}, \ldots) = P(S_{t+1} | S_t, A_t)$

### Derivations to Include (Mathematical Layer)
- Why discounting ensures finite returns for continuing tasks
- How to convert rewards-on-transitions to rewards-on-states

### Proofs to Omit
- Formal measure-theoretic foundations
- Proofs about MDP solution existence

---

## Code Examples Needed

### Intuition Layer
```python
# A simple MDP represented as a dictionary
mdp = {
    'states': ['s1', 's2', 's3', 'terminal'],
    'actions': ['left', 'right'],
    'transitions': {
        ('s1', 'right'): [('s2', 1.0)],  # deterministic
        ('s2', 'right'): [('s3', 0.8), ('s1', 0.2)],  # stochastic
        ...
    },
    'rewards': {
        ('s3', 'right', 'terminal'): 10.0,
        ...
    },
    'gamma': 0.99
}
```

### Implementation Layer
- Complete MDP class with transition sampling
- GridWorld MDP implementation
- Visualization of MDP as graph

---

## Common Misconceptions to Address

1. **"States must be physical locations"**: States can be anything—game boards, robot configurations, conversation histories. The key is the Markov property.

2. **"Transitions are always deterministic"**: Many real MDPs have stochastic transitions (robot uncertainty, opponent behavior, etc.)

3. **"γ = 1 is always fine"**: Only for episodic tasks. For continuing tasks, γ < 1 is essential for finite returns.

4. **"The MDP is given"**: In practice, choosing the state representation is a crucial design decision.

---

## Exercises

### Conceptual (3-5 questions)
1. Why can't we model chess as a bandit problem?
2. What would happen if we used γ = 1 in a continuing task?
3. Is "the weather" a Markovian state for predicting tomorrow's weather?

### Coding (2-3 challenges)
1. Implement a GridWorld MDP class
2. Simulate an episode in a given MDP

### Exploration (1-2 open-ended)
1. Design an MDP for a problem you care about. What are the states? Actions? Rewards?

---

## Subsection Breakdown

### Subsection 1: From Bandits to Sequential Decisions
- Review bandit setting: actions don't change the world
- Motivate: what if your action affects the next situation?
- Examples: robot navigation, game playing, dialogue
- The key addition: STATE

### Subsection 2: The MDP Components
- States: where you are
- Actions: what you can do
- Transitions: how actions change states
- Rewards: feedback for state-action-state triples
- Discount factor: caring about the future
- Interactive: MDP Builder demo

### Subsection 3: The Markov Property
- Definition: the future depends only on the present
- Why it matters: simplifies planning dramatically
- When it doesn't hold: partial observability preview
- How to make it hold: design better states
- Interactive: Markov vs non-Markov examples

---

## Additional Context for AI

- This is the first "mathematical" chapter. Be rigorous but accessible.
- GridWorld should be THE example. Readers will see it throughout.
- Emphasize that MDPs are a modeling framework, not a solution method.
- The Markov property section is crucial—it's what makes RL tractable.
- Don't rush to solutions. This chapter is about understanding the problem.
- Visual/interactive elements are essential for making math accessible.

---

## Quality Checklist

- [ ] All three complexity layers present
- [ ] MDP Builder demo specified
- [ ] GridWorld example used consistently
- [ ] Mathematical notation follows conventions
- [ ] Clear distinction from bandit problems
- [ ] Markov property explained with intuition AND math
- [ ] Forward references to value functions and Bellman

---

## Iteration Notes

### Visual Style
- Use the same card/box styling as Foundations
- MDP diagrams: circles for states, arrows for transitions, labels for probabilities
- Color scheme: cyan for states, emerald for actions, amber for rewards

### Interactive Components
- MDPBuilder: Allow drag-and-drop state creation
- TransitionDemo: Show how actions lead to next states
