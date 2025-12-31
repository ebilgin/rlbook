# Subsection Prompt Template

Use this template for generating individual subsection content within a chapter.

---

## Required Context

**Before generating content, read [CLAUDE.md](../../CLAUDE.md)** — specifically the "How to Generate Content" section.

Additionally for subsections:
- Read the **parent chapter prompt** (`content/chapters/{dirName}/prompt.md`)
- Review **previous subsection content** for continuity (if not the first subsection)

---

## Relationship to Chapter Prompt

Subsections provide **finer-grained control** over content generation:

- **Chapter prompt** (`prompt.md`): Defines the overall chapter scope, learning objectives, narrative arc, and which subsections exist
- **Subsection prompts** (optional): Provide specific guidance for individual subsections when more control is needed

### When to Use Subsection Prompts

1. **Complex topics** that need detailed specification
2. **Multiple authors/sessions** working on different parts
3. **Iterative refinement** of specific sections without regenerating entire chapter
4. **Quality issues** in a specific subsection that need targeted fixing

### When Chapter Prompt is Sufficient

- Standard topics that follow naturally from chapter outline
- Initial content generation
- Topics where the chapter prompt's "Core Concepts" section provides enough detail

---

## Subsection Metadata

**Chapter:** [Parent chapter title and number]
**Subsection:** [Subsection title]
**Position:** [N of M - e.g., "2 of 5"]
**Prerequisites Within Chapter:** [Previous subsections that must be read first]
**Estimated Reading Time:** [X minutes]

---

## Subsection Objectives

By the end of this subsection, readers will be able to:

1. [Specific outcome - should be subset of chapter objectives]
2. [Another outcome]

---

## Content Scope

### Must Cover
- [ ] [Concept 1 - specific to this subsection]
- [ ] [Concept 2]

### Explicitly Defer (covered in later subsections)
- [Topic covered in subsection N+1]
- [Topic covered in subsection N+2]

### Assume Known (from previous subsections)
- [Concept from subsection N-1]
- [Concept from subsection N-2]

---

## Narrative Flow

### Opening
How does this subsection connect to what came before?
[E.g., "In the previous section, we saw that greedy selection gets stuck. Now we'll fix that with exploration."]

### Core Content
What's the main thing this subsection teaches?
[E.g., "The ε-greedy algorithm and why it works"]

### Transition
How does this set up the next subsection?
[E.g., "ε-greedy explores randomly. Next, we'll see how UCB explores systematically."]

---

## Required Elements

### Complexity Layers
- **Intuition**: [Brief description of intuitive content]
- **Mathematical**: [What equations/derivations to include]
- **Implementation**: [What code to show]

### Interactive Elements
- [Demo name if any for this subsection]

### Code Examples
```python
# Brief indication of what code this subsection needs
```

---

## Content Layer Guidelines

When writing content for this subsection:

```mdx
<Intuition>
  Core concept explained in plain terms.
  This should be understandable without math or code.
</Intuition>

<Mathematical title="Formal Definition">
  $$Q(a) = \frac{1}{N(a)} \sum_{i=1}^{N(a)} R_i$$

  Where:
  - $Q(a)$ is the estimated value of action $a$
  - $N(a)$ is the count of times action $a$ was selected
</Mathematical>

<Implementation>
```python
def select_action(Q, epsilon):
    if random() < epsilon:
        return random_action()  # Explore
    return argmax(Q)  # Exploit
```
</Implementation>
```

---

## Iteration Notes

> **Purpose:** Capture learnings from subsection generation/revision that should persist. Update after each significant revision.

### Decisions Made
<!-- Choices made during iteration that future sessions should respect -->
- [Date]: [Decision and rationale]

### Issues Fixed
<!-- Problems encountered and how they were resolved -->
- [Issue]: [Resolution]

---

## Quality Checklist

Before accepting generated subsection:

- [ ] Connects smoothly to previous subsection
- [ ] All three complexity layers present
- [ ] Code examples are complete and runnable
- [ ] Mathematical notation follows MATH_CONVENTIONS.md
- [ ] MDX syntax follows MDX_AUTHORING.md
- [ ] Build passes: `npm run build`
- [ ] Sets up next subsection appropriately

---

## File Locations

Subsection content goes in:
```
content/chapters/{dirName}/{subsection-slug}.mdx
```

Example:
```
content/chapters/0020-multi-armed-bandits/epsilon-greedy.mdx
```

The subsection must also be defined in `src/lib/chapters.ts` under the parent chapter's `subsections` array.
