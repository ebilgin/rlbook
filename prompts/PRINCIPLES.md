# Content Generation Principles

These principles guide all AI-generated content for rlbook.ai. Every prompt should align with these foundations.

## 1. The Reader Comes First

### Assumed Background
- Comfortable with Python programming
- Familiar with basic ML concepts (supervised learning, gradient descent, neural networks)
- Understanding of probability and basic linear algebra
- **Not assumed**: Prior RL knowledge, advanced mathematics, research paper familiarity

### Reader Personas
When writing, consider these learner types:

1. **The Practitioner**: Wants to build RL systems. Cares about code, debugging, practical tips.
2. **The Theorist**: Wants deep understanding. Cares about proofs, convergence, edge cases.
3. **The Explorer**: Wants intuition. Cares about visualizations, interactive demos, big picture.

All three should find value in every chapter.

## 2. Progressive Disclosure

### The Three Layers

Every concept should be explainable at three levels:

**Layer 1 - Intuition (Always Visible)**
- What is this concept in plain English?
- Why does it matter? What problem does it solve?
- Visual/interactive demonstration
- Analogies to familiar concepts

**Layer 2 - Mathematical (Expandable)**
- Formal definitions and notation
- Key equations with explanation of each term
- Derivations where they build understanding
- Connections to other mathematical frameworks

**Layer 3 - Implementation (Expandable)**
- Clean, documented code
- Edge cases and gotchas
- Performance considerations
- Debugging tips

### Writing for Layers

```markdown
<!-- BAD: Forces math on everyone -->
The Q-function Q(s,a) represents the expected return...

<!-- GOOD: Intuition first, math available -->
The Q-function tells us "how good is this action in this situation?"

<Mathematical>
Formally, $Q(s,a) = \mathbb{E}[G_t | S_t = s, A_t = a]$
where $G_t$ is the discounted return from time $t$.
</Mathematical>
```

## 3. Coherent Narrative

### Recurring Examples

Use consistent examples throughout the book to build familiarity:

| Example | Purpose | Chapters |
|---------|---------|----------|
| **GridWorld** | Core concepts, easy to visualize | Foundations through Value Methods |
| **CliffWalking** | Risk, exploration, on/off policy | TD Learning, Q-Learning |
| **CartPole** | Continuous states, function approx | Deep RL onwards |
| **LunarLander** | Multi-dimensional actions, reward shaping | Policy Gradient onwards |
| **Custom Trading Env** | Real-world complexity | Applications |

### Building Complexity

Each chapter should:
1. Reference what came before ("Recall from Chapter X...")
2. Show limitations of previous approaches
3. Introduce new concepts as solutions to those limitations
4. Foreshadow what comes next

## 4. Interactivity Standards

### Every Concept Needs a Demo

If you can't make it interactive, you haven't explained it well enough.

**Required for each concept:**
- At minimum: Animated visualization showing the concept
- Ideal: Interactive demo where reader can change parameters
- Best: Sandboxed environment where reader can experiment freely

### Demo Design Principles

1. **Immediate feedback**: Changes should reflect instantly
2. **Sensible defaults**: Demo should be interesting without any interaction
3. **Guided exploration**: Suggest what to try ("Notice what happens when...")
4. **Failure is learning**: Let readers break things to understand boundaries

## 5. Visual Language

### Consistent Design

- **States**: Circles or grid cells
- **Actions**: Arrows showing direction/choice
- **Values**: Color gradients (blue=low, red=high) or numbers
- **Policies**: Thicker arrows for preferred actions
- **Rewards**: Yellow/gold highlights
- **Transitions**: Dashed lines for stochastic, solid for deterministic

### Animation Guidelines

- Use animation to show **processes**, not just to look fancy
- Play/pause controls on all animations
- Speed control for complex sequences
- Option to step through frame by frame

## 6. Mathematical Rigor Without Intimidation

### Equation Introduction Pattern

1. State the intuition first
2. Introduce notation one piece at a time
3. Show the full equation
4. Walk through each component
5. Provide a concrete example with numbers

```markdown
The agent updates its value estimate based on what actually happened
versus what it expected.

<Mathematical>
Let's build this up:
- $V(s)$ is our current estimate of the state's value
- $r$ is the reward we just received
- $V(s')$ is our estimate of the next state's value
- $\alpha$ is our learning rate

The update rule becomes:
$$V(s) \leftarrow V(s) + \alpha[\underbrace{r + \gamma V(s')}_{\text{what happened}} - \underbrace{V(s)}_{\text{what we expected}}]$$
</Mathematical>
```

## 7. Code Quality

### Standards for Code Examples

- **Complete**: No `...` or "implementation left as exercise" for core concepts
- **Runnable**: Every code block should work if copied
- **Documented**: Comments explain the "why", not just the "what"
- **Idiomatic**: Follow Python/JS best practices
- **Minimal**: Show the concept, not production-ready code

### Referencing the Code Package

The `code/rlbook/` directory contains production-grade, tested implementations. For full documentation, see [docs/CONTENT_TYPES.md](../docs/CONTENT_TYPES.md#6-code).

When writing chapters:
1. **Keep inline code minimal**: Show the core concept, not full implementation
2. **Reference tested code**: Point readers to `code/rlbook/` for complete implementations
3. **Maintain consistency**: Ensure inline examples match the tested code's API

### Code Block Types

```python
# CONCEPT: Core algorithm, educational focus
def q_learning_update(Q, s, a, r, s_prime, alpha=0.1, gamma=0.99):
    """Single Q-learning update step."""
    best_next_value = max(Q[s_prime].values())
    td_target = r + gamma * best_next_value
    td_error = td_target - Q[s][a]
    Q[s][a] += alpha * td_error
```

```python
# PRACTICAL: Production considerations
def q_learning_update(Q, s, a, r, s_prime, alpha=0.1, gamma=0.99, done=False):
    """
    Q-learning update with terminal state handling.

    Note: When s_prime is terminal (done=True), there is no future value.
    This is a common bug source in RL implementations.
    """
    if done:
        td_target = r  # No future value from terminal state
    else:
        best_next_value = max(Q[s_prime].values())
        td_target = r + gamma * best_next_value

    td_error = td_target - Q[s][a]
    Q[s][a] += alpha * td_error
    return td_error  # Useful for monitoring convergence
```

## 8. Chapter Endings: Summary & Exercises

Every chapter should end with consolidation and practice opportunities.

### Summary Structure

Use **numbered takeaway cards** (see [VISUAL_PATTERNS.md](VISUAL_PATTERNS.md#pattern-numbered-takeaway-cards)) to highlight 3-5 key concepts. Each takeaway should:
- Have a clear, actionable title
- Explain the "so what" in 1-2 sentences
- Use code formatting for technical terms

### Quiz Questions

Include 4-6 conceptual questions using the **expandable answer pattern**. Good quiz questions:
- Test understanding, not memorization
- Include technical terms in code formatting
- Provide thorough explanations in the answer (not just the answer itself)
- Cover the range of topics in the chapter

### Coding Exercises

For hands-on practice:
- **Link to Colab notebooks** rather than embedding long code blocks
- Use **exercise preview cards** to show what each exercise covers
- Include solutions in the notebook (in collapsed cells or separate sections)
- Structure exercises from foundational to advanced

### What's Next

End with a `<Note>` callout that:
- Summarizes what the reader can now do
- Lists 2-3 advanced topics for further exploration
- Connects to upcoming chapters where relevant

### Dedicated Subsection

For substantial chapters, consider making "Summary & Exercises" a dedicated subsection (separate MDX file) rather than embedding at the end of the main content. This keeps the main content focused and makes exercises easier to find.

## 9. Cross-Referencing

### Link Generously

- Forward references: "We'll see in [Chapter X](link) how this extends to..."
- Backward references: "Recall from [the exploration section](link) that..."
- Concept references: "This is an example of <ConceptRef id="bootstrapping" />"

### Glossary Integration

Every technical term should:
1. Be defined on first use
2. Link to a glossary entry
3. Show a tooltip on hover with brief definition

## 10. Accessibility

- Alt text for all images describing the concept, not just the visual
- Transcripts for any video content
- Color choices that work for colorblind readers
- Math available as both rendered and source

## 11. Evolution and Feedback

### Content Versioning

- Major conceptual changes: Update prompt, regenerate content
- Minor fixes: Direct edit to content
- Community feedback: Collected via Giscus, reviewed for prompt improvements

### Living Document

This is not a static textbook. Content should:
- Reference current research where relevant
- Update examples to reflect best practices
- Incorporate reader feedback and common questions
- Evolve as the field evolves
