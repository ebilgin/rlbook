# Chapter Prompt Template

Use this template for generating complete chapter content.

---

## Required Context

**Before generating content, read [CLAUDE.md](../../CLAUDE.md)** — specifically the "How to Generate Content" section. It references all required foundation documents (PRINCIPLES.md, STYLE_GUIDE.md, MDX_AUTHORING.md, etc.).

---

## Chapter Metadata

**Chapter Number:** [XX]
**Title:** [Chapter Title]
**Section:** [Parent section, e.g., "Q-Learning", "Policy Gradients"]
**Prerequisites:** [List of chapters/concepts that should be understood first]
**Estimated Reading Time:** [X minutes]

---

## Learning Objectives

By the end of this chapter, readers will be able to:

1. [Specific, measurable outcome]
2. [Another outcome]
3. [Another outcome]

---

## Core Concepts to Cover

### Primary Concepts (Must Cover)
- [ ] [Concept 1]
- [ ] [Concept 2]

### Secondary Concepts (Cover if Space Permits)
- [ ] [Concept 1]
- [ ] [Concept 2]

### Explicitly Out of Scope
- [Topic to defer to later chapter]
- [Advanced topic not needed here]

---

## Narrative Arc

### Opening Hook
[Describe the motivating problem or question to open with]

### Key Insight
[What's the "aha moment" this chapter delivers?]

### Closing Connection
[How does this connect to what comes next?]

---

## Required Interactive Elements

### Demo 1: [Name]
- **Purpose:** [What concept does this demonstrate?]
- **Interaction:** [What can users control?]
- **Expected Discovery:** [What should users learn by playing?]

### Demo 2: [Name]
[Repeat structure]

---

## Recurring Examples to Use

Specify which standard examples to use and how:

- **GridWorld:** [How it's used in this chapter]
- **CliffWalking:** [If used]
- **CartPole:** [If used]
- **Custom:** [Any chapter-specific examples]

---

## Cross-References

### Build On (Backward References)
- Chapter XX: [Specific concept to reference]
- Chapter YY: [Specific concept to reference]

### Set Up (Forward References)
- Chapter XX: [What this prepares readers for]

---

## Mathematical Depth

### Required Equations
1. [Equation name/description]
2. [Equation name/description]

### Derivations to Include (Mathematical Layer)
- [Derivation 1]

### Proofs to Omit
- [What to state without proof and why]

---

## Code Examples Needed

### Intuition Layer
- [Simple code snippet purpose]

### Implementation Layer
- [Complete implementation purpose]

---

## Common Misconceptions to Address

1. [Misconception]: [Correction]
2. [Misconception]: [Correction]

---

## Exercises

### Conceptual (3-5 questions)
- [Question type/topic]

### Coding (2-3 challenges)
- [Challenge description]

### Exploration (1-2 open-ended)
- [Open-ended prompt]

---

## Additional Context for AI

[Any specific instructions, constraints, or context that doesn't fit above. For example:
- "Emphasize the connection to dynamic programming"
- "Avoid discussion of function approximation—that comes later"
- "Use the trading example from the previous chapter"
]

---

## Iteration Notes

> **Purpose:** Capture learnings from content generation sessions that should persist across iterations. Update this section after each significant revision.

### Decisions Made
<!-- Document choices made during iteration that future sessions should respect -->
- [Date]: [Decision and rationale]

### Style Preferences
<!-- Specific style choices for this chapter that differ from or extend the global style guide -->
- [Preference and why]

### Known Issues
<!-- Problems identified but not yet fixed, with context -->
- [ ] [Issue description]

### What Worked Well
<!-- Approaches that produced good results, to replicate -->
- [Approach and outcome]

---

## Quality Checklist

Before accepting generated content, verify:

- [ ] All three complexity layers are present and properly tagged
- [ ] Interactive demos are described with clear specifications
- [ ] Cross-references use proper component syntax
- [ ] Code examples are complete and runnable
- [ ] Mathematical notation follows MATH_CONVENTIONS.md
- [ ] **MDX syntax follows MDX_AUTHORING.md** (critical for build success)
- [ ] Writing follows STYLE_GUIDE.md
- [ ] Exercises span difficulty levels
- [ ] Build passes: `npm run build`

---

## MDX Syntax Requirements

**Critical:** Review [MDX_AUTHORING.md](../MDX_AUTHORING.md) before generating content.

### Must Avoid (causes build failures)

| Pattern | Problem | Alternative |
|---------|---------|-------------|
| `\begin{cases}...\end{cases}` | MDX parser breaks | Use bullet list description |
| `\|x\|` in table cells | Conflicts with table delimiters | Use prose or move outside table |
| Raw `<` or `>` in prose | Interpreted as JSX tags | Use `$<$` or `&lt;` |
| Unescaped `{` or `}` | Interpreted as JS expression | Use `\{` or `$\{...\}$` |

### Always Test

```bash
npm run build
```

If you see errors like:
- `Expected a closing tag for <Component>` → Check for problematic LaTeX
- `Unexpected end of file in expression` → Check for `|` in tables or unbalanced braces
