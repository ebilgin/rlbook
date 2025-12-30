# Writing Style Guide

This guide ensures consistent voice, tone, and formatting across all rlbook.ai content.

## Voice and Tone

### Be a Knowledgeable Friend

Write as if explaining to a smart colleague who's new to RL. Not a textbook, not a blog post—something in between.

**Do:**
- "Here's the key insight..."
- "You might wonder why..."
- "A common mistake is..."
- "Let's work through this step by step"

**Don't:**
- "The reader should note that..." (too formal)
- "Obviously..." or "Simply..." (dismissive)
- "In this section, we will discuss..." (bureaucratic)
- Excessive exclamation marks or forced enthusiasm

### Active Voice

Prefer active constructions that emphasize agents and actions—fitting for RL content.

**Do:** "The agent updates its Q-values after each action."
**Don't:** "Q-values are updated by the agent after each action is taken."

### Second Person for Instructions

When guiding the reader through exercises or demos:

**Do:** "Try increasing the learning rate. Notice how..."
**Don't:** "One can increase the learning rate to observe..."

## Structure

### Chapter Structure

```
1. Opening Hook (1-2 paragraphs)
   - Motivating problem or question
   - Why this matters
   - What you'll learn

2. Intuition Section
   - Core concept in plain terms
   - Interactive demo or animation
   - Real-world analogies

3. Formal Treatment (expandable)
   - Definitions
   - Key equations
   - Derivations if illuminating

4. Implementation (expandable)
   - Code walkthrough
   - Practical considerations
   - Common pitfalls

5. Deeper Understanding
   - Edge cases
   - Connections to other concepts
   - Historical context or alternatives

6. Summary
   - Key takeaways (3-5 bullets)
   - What's next

7. Exercises
   - Conceptual questions
   - Coding challenges
   - Open-ended exploration
```

### Section Length

- **Paragraphs**: 3-5 sentences max. White space aids comprehension.
- **Sections**: Aim for 500-1000 words before a new heading.
- **Chapters**: 2000-4000 words total (excluding code and expanded sections).

### Headings

Use descriptive headings that convey meaning, not just topics.

**Do:** "Why Random Exploration Fails"
**Don't:** "Exploration Methods"

**Do:** "The Deadly Triad: When Function Approximation Goes Wrong"
**Don't:** "Function Approximation Challenges"

## Formatting

### Emphasis

- **Bold** for key terms on first introduction
- *Italics* for emphasis within sentences
- `code` for variables, function names, file names
- Never ALL CAPS for emphasis

### Lists

Use bullets for unordered collections, numbers for sequences or rankings.

Keep list items parallel in structure:

**Good:**
- Observe the current state
- Select an action
- Receive a reward
- Transition to the next state

**Bad:**
- Observing what state you're in
- Action selection
- The agent gets a reward
- Next state

### Math Formatting

Inline math for short expressions: The discount factor $\gamma$ controls...

Display math for important equations:

$$Q(s,a) \leftarrow Q(s,a) + \alpha[r + \gamma \max_{a'} Q(s',a') - Q(s,a)]$$

Always introduce notation before using it.

### Code Formatting

````markdown
```python
# Brief comment explaining purpose
def function_name(clear, parameter_names):
    """Docstring for non-trivial functions."""
    # Implementation with explanatory comments
    return result
```
````

For interactive code references:
- `variable_name` for inline references
- Highlight changed lines when showing modifications

### Callouts

Use callouts sparingly for genuinely important asides:

```mdx
<Note>
This is important context that doesn't fit the main flow.
</Note>

<Warning>
This is a common mistake or pitfall to avoid.
</Warning>

<Tip>
This is a practical suggestion or shortcut.
</Tip>

<DeepDive>
This is optional advanced content for curious readers.
</DeepDive>
```

## Terminology

### Consistent Terms

Use these terms consistently (not their alternatives):

| Use This | Not This |
|----------|----------|
| reward | payoff, return (for single-step) |
| return | cumulative reward, total reward |
| state | observation (unless partial observability) |
| action | move, decision, choice |
| policy | strategy |
| value function | utility function |
| agent | learner, actor |
| environment | world, simulator |
| episode | trial, game |
| step | timestep, iteration |

### Acronym Policy

- Define on first use: "Deep Q-Network (DQN)"
- Use acronym thereafter: "DQN extends..."
- In new chapters, redefine or link to definition
- Common ML terms (MLP, CNN, RNN) can be used without definition

## Analogies and Examples

### Good Analogies

- Grounded in common experience
- Highlight the key insight, not just surface similarity
- Acknowledge where the analogy breaks down

**Example:**
> Think of the value function like a heat map showing "how good" each location is. Just as you might prefer spots in a room closer to the heater in winter, an agent prefers states with higher value. But unlike physical heat, value depends on the policy—what the agent plans to do next.

### Recurring Examples

Maintain consistency with established examples:

- **GridWorld**: 4x4 or 5x5 grid, standard movements (up/down/left/right), goal in corner
- **CliffWalking**: Standard Sutton & Barto setup
- **CartPole**: OpenAI Gym standard configuration
- **Custom envs**: Document clearly in first introduction

## Inclusive Language

- Use "they" for hypothetical individuals
- Avoid gendered examples unnecessarily
- "Folks" or "everyone" rather than "guys"
- Acknowledge diverse paths into RL (not just academic)

## Citations and Attribution

- Cite seminal papers for major concepts: "Mnih et al. (2015) introduced..."
- Link to papers where readers might want to go deeper
- Acknowledge when simplifying: "We're glossing over some details here—see [paper] for the full treatment."
- Credit visualization inspirations: "Inspired by [source]"

## Quality Checklist

Before finalizing any content:

- [ ] Can this be understood without expanding math/code sections?
- [ ] Is there an interactive element or visualization?
- [ ] Are all technical terms defined or linked?
- [ ] Do code examples run without modification?
- [ ] Are forward/backward references in place?
- [ ] Is the length appropriate for the depth?
- [ ] Would all three reader personas find value?
