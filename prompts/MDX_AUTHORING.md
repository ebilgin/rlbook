# MDX Authoring Guide

This guide documents MDX syntax pitfalls and best practices specific to rlbook.ai content. MDX combines Markdown with JSX, which creates some parsing conflicts that must be avoided.

## Critical: Syntax Pitfalls

### 1. LaTeX `\begin{cases}` Blocks Break MDX

**Problem:** The `\begin{cases}...\end{cases}` LaTeX environment causes MDX parsing errors because the parser interprets characters inside as JSX.

**Error you'll see:**
```
[@mdx-js/rollup] Expected a closing tag for `<Mathematical>`
```

**Don't do this:**
```mdx
<Mathematical>

$$\pi(a|s) = \begin{cases}
1 - \epsilon & \text{if } a = \arg\max_{a'} Q(s, a') \\
\frac{\epsilon}{|\mathcal{A}|} & \text{otherwise}
\end{cases}$$

</Mathematical>
```

**Do this instead:**
```mdx
<Mathematical>

The policy selects actions as follows:
- For the greedy action $a = \arg\max_{a'} Q(s, a')$: probability is $1 - \epsilon$
- For all other actions: probability is $\frac{\epsilon}{|\mathcal{A}|}$

</Mathematical>
```

**Alternative:** If you must use piecewise notation, describe it in prose or use a simpler inline format.

### 2. Pipe Characters `|` in LaTeX Inside Tables

**Problem:** Markdown tables use `|` as column delimiters. LaTeX expressions containing `|` (like `|A|` for cardinality) inside table cells cause parsing conflicts.

**Error you'll see:**
```
[@mdx-js/rollup] Unexpected end of file in expression, expected a corresponding closing brace
```

**Don't do this:**
```mdx
| Method | Formula |
|--------|---------|
| Dueling DQN | $Q = V + A - \frac{1}{|A|}\sum_{a'} A(s,a')$ |
```

**Do this instead:**
```mdx
| Method | Formula |
|--------|---------|
| Dueling DQN | $Q = V + A - \text{mean}(A)$ |
```

**Or use a different notation:**
- `\lvert A \rvert` instead of `|A|` (may still conflict)
- `\text{size}(A)` or `n_{\text{actions}}` as alternatives
- Move complex formulas outside the table

### 3. Angle Brackets `<` and `>` in Content

**Problem:** MDX interprets `<` as the start of a JSX tag. This affects:
- Mathematical comparisons: `x < y`
- Generics-like notation: `Array<number>`
- XML/HTML snippets in prose

**Workarounds:**
```mdx
{/* Use LaTeX for math comparisons */}
When $x < y$, the condition holds.

{/* Use HTML entities in prose */}
The value should be &lt; 10.

{/* Use code blocks for code with generics */}
```typescript
const arr: Array<number> = [];
```
```

### 4. Curly Braces `{` and `}` Escaping

**Problem:** MDX interprets `{...}` as JavaScript expressions. Stray braces in prose cause errors.

**Workarounds:**
```mdx
{/* Escape with backslash in text */}
The set is defined as \{1, 2, 3\}.

{/* Or use LaTeX for mathematical sets */}
The set is $\{1, 2, 3\}$.

{/* Use code formatting */}
The object syntax is `{ key: value }`.
```

## Component Usage

### Content Layer Components

Always import the components at the top of the file:

```mdx
import { Intuition, Mathematical, Implementation, DeepDive } from '@/components/ui/ContentLayers';
```

**Structure:**
```mdx
<Intuition>

Plain-language explanation here. No complex LaTeX.

</Intuition>

<Mathematical>

Formal definitions and equations here.
Keep LaTeX simple - avoid `\begin{cases}`.

</Mathematical>

<Implementation>

```python
# Code examples here
```

</Implementation>
```

### Important Notes

1. **Blank lines matter:** Always leave a blank line after opening tags and before closing tags
2. **No nesting:** Don't nest content layer components inside each other
3. **SSR compatibility:** Components render on the server with default values, so content should be self-contained

### Callout Components

```mdx
import { Note, Warning, Tip } from '@/components/ui/Callouts';

<Note>
Important context that doesn't interrupt the flow.
</Note>

<Warning>
Common mistake or pitfall to avoid.
</Warning>

<Tip>
Practical suggestion or shortcut.
</Tip>
```

### Navigation Components

```mdx
import { ChapterObjectives, KeyTakeaways, NextChapter, CrossRef } from '@/components/ui/ChapterNav';

<ChapterObjectives>
- Objective 1
- Objective 2
</ChapterObjectives>

<KeyTakeaways>
- Takeaway 1
- Takeaway 2
</KeyTakeaways>

<NextChapter slug="next-chapter-slug" title="Next Chapter Title" />

{/* Reference another chapter */}
See <CrossRef slug="other-chapter">the other chapter</CrossRef> for details.
```

## LaTeX Best Practices

### Inline Math
```mdx
The discount factor $\gamma$ is typically between 0.9 and 0.99.
```

### Display Math
```mdx
$$Q(s,a) \leftarrow Q(s,a) + \alpha[r + \gamma \max_{a'} Q(s',a') - Q(s,a)]$$
```

### Avoid These Constructs

| Problematic | Alternative |
|-------------|-------------|
| `\begin{cases}...\end{cases}` | Bullet list or prose description |
| `\begin{align}...\end{align}` | Separate `$$...$$` blocks |
| `\begin{bmatrix}...\end{bmatrix}` | Use with caution; test build |
| `|x|` in tables | `\lVert x \rVert` or move outside table |

### Safe LaTeX Constructs

These work reliably:
- `\frac{a}{b}` - fractions
- `\sum`, `\prod`, `\int` - big operators
- `\mathcal{A}`, `\mathbb{R}` - special fonts
- `\theta`, `\alpha`, `\gamma` - Greek letters
- `\leftarrow`, `\rightarrow` - arrows
- `\max`, `\min`, `\arg\max` - operators
- `^{superscript}`, `_{subscript}` - scripts

## Testing Your Content

### Local Build Test

Always run a build before committing:

```bash
npm run build
```

If you see MDX errors, they'll point to the file and approximate line number.

### Common Error Messages

| Error | Likely Cause |
|-------|--------------|
| `Expected a closing tag for <Component>` | Unclosed tag OR problematic LaTeX inside component |
| `Unexpected end of file in expression` | Unbalanced `{` braces or `|` in tables |
| `Could not parse expression` | JavaScript syntax issue in `{...}` block |

### Debugging Steps

1. Check line number in error message
2. Look for `\begin{...}` blocks near that line
3. Look for `|` characters in table cells
4. Look for unescaped `<`, `>`, `{`, `}` in prose
5. Try simplifying the LaTeX progressively until it builds

## File Template

```mdx
---
title: "Chapter Title"
slug: "chapter-slug"
section: "Section Name"
description: "Brief description"
status: "draft"
lastReviewed: null
prerequisites:
  - slug: "prereq-slug"
    title: "Prerequisite Title"
---

import { Intuition, Mathematical, Implementation, DeepDive } from '@/components/ui/ContentLayers';
import { ChapterObjectives, KeyTakeaways, NextChapter, CrossRef } from '@/components/ui/ChapterNav';
import { Note, Warning, Tip } from '@/components/ui/Callouts';

# Chapter Title

<ChapterObjectives>
- Objective 1
- Objective 2
</ChapterObjectives>

## Section Heading

Opening paragraph...

<Intuition>

Plain-language explanation.

</Intuition>

<Mathematical>

Formal definitions using safe LaTeX only.

$$Q(s,a) = r + \gamma \max_{a'} Q(s', a')$$

</Mathematical>

<Implementation>

```python
def example():
    pass
```

</Implementation>

## Summary

<KeyTakeaways>
- Key point 1
- Key point 2
</KeyTakeaways>

<NextChapter slug="next-chapter" title="Next Chapter" />
```

## Quick Reference Card

```
SAFE:
  $\gamma$, $\theta$, $\alpha$          Greek letters
  $\frac{a}{b}$                         Fractions
  $\sum_{i=1}^{n}$, $\max_a$            Big operators
  $$...$$                               Display equations

AVOID:
  \begin{cases}...\end{cases}           Use bullet lists
  \begin{align}...\end{align}           Use separate $$ blocks
  |x| inside tables                     Use text description
  < > in prose                          Use $<$ or &lt;
  { } in prose                          Use \{ \} or $\{...\}$
```
