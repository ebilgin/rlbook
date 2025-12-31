# Paper Explanation Template

Use this template for generating explanations of research papers. Paper explanations are standalone content pieces that provide accessible breakdowns of RL research papers.

---

## Required Context

**Before generating content, read [CLAUDE.md](../../CLAUDE.md)** — specifically the "How to Generate Content" and "For AI Assistants" sections.

Also read:
- **`content/connections.yaml`** - To update chapter-paper relationships
- **Related chapter content** - To ensure accurate cross-references

---

## Paper Metadata

**Slug:** [url-friendly-name, e.g., "prioritized-experience-replay"]
**Paper Title:** [Full paper title]
**Authors:** [Author list]
**Venue:** [Conference/Journal and year, e.g., "ICLR 2016"]
**ArXiv:** [ArXiv ID if available, e.g., "1511.05952"]
**Paper URL:** [Direct link to paper]

---

## TL;DR

[One sentence summary of the paper's key contribution. This appears in listings and previews.]

---

## Target Audience

**Prerequisites:** [What should readers already understand?]
- [Chapter or concept 1]
- [Chapter or concept 2]

**Who benefits most:**
- [ ] Researchers implementing or extending this work
- [ ] Practitioners applying this technique
- [ ] Students understanding foundational ideas
- [ ] Reading group participants

---

## Paper Summary

### The Problem

[What problem does this paper address? Why does it matter?]

### The Key Insight

[What's the core idea? The "aha moment" that makes this paper work.]

### The Approach

[High-level description of the method. Not a full algorithm walkthrough—that comes in the detailed sections.]

### Main Results

[What did they achieve? Key experimental findings.]

---

## Content Structure

### Section 1: Motivation and Background
- [What context is needed?]
- [What prior work does this build on?]

### Section 2: Core Method
- [Main algorithm or technique]
- [Key equations to include]

### Section 3: Why It Works
- [Intuition for why this approach is effective]
- [Any theoretical justification]

### Section 4: Practical Considerations
- [Implementation details that matter]
- [Hyperparameters and their effects]
- [Common pitfalls]

### Section 5: Results Walkthrough
- [Key experiments to highlight]
- [What the results show]

### Section 6: Limitations and Extensions
- [What are the paper's limitations?]
- [How has subsequent work extended this?]

---

## Interactive Elements

Papers benefit from focused demos that illustrate the key insight:

### Demo: [Name]
- **Purpose:** [What aspect of the paper does this demonstrate?]
- **Comparison:** [What baseline does it compare against?]
- **Interaction:** [What can users control?]
- **Expected Discovery:** [What should users observe?]

---

## Chapter Connections

### Builds On
- Chapter: [slug] — [What concept from this chapter is assumed?]
- Chapter: [slug] — [Another dependency]

### Referenced By
- Chapter: [slug] — [How does this chapter use this paper?]

### Relationship Type
[Choose: foundational | extends | application | theoretical | empirical]

---

## Mathematical Depth

### Key Equations (Intuition Layer)
[List equations that should be shown even at intuition level, with plain-language explanations]

### Full Derivations (Mathematical Layer)
[List derivations to include for readers who want the math]

### Proofs to Summarize
[Any proofs that should be summarized rather than reproduced in full]

---

## Code Examples

### Pseudocode (Intuition Layer)
[Simplified pseudocode that captures the essence]

### Implementation (Implementation Layer)
[What code examples to include? Reference existing components if applicable]

---

## Common Misunderstandings

1. [Misunderstanding]: [Correction]
2. [Misunderstanding]: [Correction]

---

## Discussion Questions

For reading groups or self-study:

1. [Question about the paper's assumptions]
2. [Question about alternative approaches]
3. [Question about applicability]

---

## Additional Context for AI

[Any specific instructions for content generation:
- "Focus on the prioritization mechanism, not the DQN baseline"
- "Include comparison with uniform sampling"
- "Emphasize the importance sampling correction"
]

---

## Quality Checklist

Before accepting generated content, verify:

- [ ] TL;DR is genuinely one sentence and captures the key contribution
- [ ] Prerequisites are accurate and link to correct chapters
- [ ] Key insight is clearly articulated
- [ ] All three complexity layers are present and properly tagged
- [ ] At least one interactive demo is specified
- [ ] Chapter connections are bidirectional (update connections.yaml)
- [ ] Mathematical notation follows MATH_CONVENTIONS.md
- [ ] Code examples are complete and runnable
- [ ] Discussion questions are thought-provoking
