# CLAUDE.md - rlbook.ai Project Guide

## Project Overview

**rlbook.ai** is an open-source, community-driven resource for learning reinforcement learning. Content is AI-generated from carefully designed prompts and refined through human review.

- **Website**: https://rlbook.ai
- **Repository**: https://github.com/ebilgin/rlbook
- **Discord**: https://discord.gg/mJ7n3zNf7r
- **Author/Curator**: Enes Bilgin (author of "Mastering Reinforcement Learning with Python")

## Documentation Overview

| Document | Audience | Purpose |
|----------|----------|---------|
| **CLAUDE.md** (this file) | AI (Claude) | How to generate/edit content, what docs to read |
| **README.md** | New visitors | Project overview, quick start, deployment |
| **docs/CONTRIBUTING.md** | Human contributors | PR process, code standards, community guidelines |
| **docs/CONTENT_WORKFLOW.md** | Content creators | Phases from prompt to publication, iteration notes |
| **docs/CONTENT_TYPES.md** | Content creators | Six content categories with structure/templates |
| **docs/ARCHITECTURE.md** | Developers | Technical stack, component patterns, deployment |
| **prompts/*.md** | AI generators | Foundation docs (PRINCIPLES, STYLE_GUIDE, etc.) |

## Core Philosophy

1. **Prompts are First-Class Assets**: The prompts that generate content are as important as the content itself. They are versioned, reviewed, and evolved with community input.

2. **Progressive Complexity**: Readers control their depth. Math and code details are available but not forced. Content works at multiple levels.

3. **Interactivity First**: Every concept should have an interactive demonstration. Learning happens through exploration, not just reading.

4. **Coherent Narrative**: Examples build on each other across chapters. The same GridWorld, the same agent, evolving techniques.

5. **Browser-Native**: All simulations run client-side. No server load, instant feedback, works offline.

## Repository Structure (Content Focus)

For full project structure, see [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md). Here's what matters for content generation:

```
rlbook/
├── CLAUDE.md                 # This file - AI collaboration guide
├── prompts/                  # Foundation documents for content generation
│   ├── PRINCIPLES.md         # Core content principles (always read first)
│   ├── STYLE_GUIDE.md        # Writing style, tone, formatting
│   ├── MATH_CONVENTIONS.md   # Mathematical notation standards
│   ├── CODE_STANDARDS.md     # Code example standards
│   ├── MDX_AUTHORING.md      # MDX syntax rules (critical!)
│   └── templates/            # Prompt templates
│       ├── chapter.md        # Template for chapter prompts
│       ├── subsection.md     # Template for subsection prompts
│       └── paper.md          # Template for paper prompts
├── content/                  # All educational content
│   ├── chapters/             # Progressive lessons
│   │   └── XXXX-chapter-name/
│   │       ├── prompt.md     # Chapter prompt (required)
│   │       ├── index.mdx     # Chapter overview
│   │       ├── {subsection}.mdx  # Subsection pages
│   │       └── assets/       # Images, data files
│   ├── papers/               # Research paper deep dives
│   ├── applications/         # Problem formulation guides
│   └── connections.yaml      # Chapter ↔ Paper relationships
├── code/rlbook/              # Python implementations (reference in content)
├── src/lib/chapters.ts       # Chapter/subsection definitions
└── docs/                     # Project documentation
    ├── CONTENT_TYPES.md      # Six content categories
    └── CONTENT_WORKFLOW.md   # Prompt → Publication workflow
```

## Content Status System

All content has a review status tracked in MDX frontmatter (`draft` → `editor_reviewed` → `community_reviewed` → `verified`). See [docs/CONTENT_WORKFLOW.md](docs/CONTENT_WORKFLOW.md#content-status-flow) for full details.

## Content Complexity Levels

All content supports three complexity levels, controlled by reader preference:

1. **Intuition** (default): Core concepts, visual explanations, interactive demos
2. **Mathematical**: Formal definitions, equations, derivations
3. **Implementation**: Code examples, algorithmic details, edge cases

Use these markers in MDX:
```mdx
<Intuition>
  Q-learning learns the value of actions by trial and error...
</Intuition>

<Mathematical>
  $$Q(s,a) \leftarrow Q(s,a) + \alpha[r + \gamma \max_{a'} Q(s',a') - Q(s,a)]$$
</Mathematical>

<Implementation>
  ```python
  def update_q(state, action, reward, next_state):
      ...
  ```
</Implementation>
```

## Dual Learning Paths

Each chapter offers two ways to engage with code:

1. **In-Browser Demos** (TensorFlow.js)
   - Instant interactivity, no setup required
   - Quick experimentation with parameters
   - Ideal for building intuition

2. **Colab Notebooks** (PyTorch)
   - Deeper experimentation and modification
   - Production-like code patterns
   - Extended exercises and projects
   - Link in each chapter: `<ColabLink notebook="1020_q_learning_basics.ipynb" />`

## Python Package (`code/rlbook/`)

The `code/` directory contains production-grade, tested Python implementations. For full documentation, see [docs/CONTENT_TYPES.md](docs/CONTENT_TYPES.md#6-code) and [code/README.md](code/README.md).

**Quick reference:**
```bash
cd code
pip install -e .
pytest  # Run tests
python -m rlbook.examples.train_gridworld  # Run example
```

When writing chapters, reference the Python package:
```mdx
<Implementation>
For the full tested implementation, see [code/rlbook/agents/q_learning.py](/code/rlbook/agents/q_learning.py).
</Implementation>
```

## Papers: Deep Dives into Research

Papers are standalone explanations of influential RL research papers. They complement chapters by providing focused, in-depth coverage of specific techniques.

### Chapters vs. Papers

| Aspect | Chapters | Papers |
|--------|----------|--------|
| **Scope** | Broad topic coverage | Single paper/technique |
| **Goal** | Build understanding progressively | Explain a specific contribution |
| **Narrative** | Part of learning path | Standalone, linkable |
| **Use case** | Learning RL | Reading groups, research, deep-dives |

### Paper Frontmatter

```yaml
---
title: "Prioritized Experience Replay"
slug: "prioritized-experience-replay"
paper:
  authors: ["Schaul", "Quan", "Antonoglou", "Silver"]
  venue: "ICLR 2016"
  arxiv: "1511.05952"
  url: "https://arxiv.org/abs/1511.05952"
tldr: "Sample important transitions more often to learn faster"
status: draft
lastReviewed: null
---
```

### Cross-Referencing

Use standard markdown links for cross-references:

**In chapters**, reference papers for deep-dives:
```mdx
For the full derivation of importance sampling corrections, see the
[Prioritized Experience Replay paper analysis](/papers/prioritized-experience-replay).
```

**In papers**, reference prerequisite chapters:
```mdx
**Prerequisites:**
- [Q-Learning Basics](/chapters/q-learning-basics)
- [Deep Q-Networks](/chapters/deep-q-networks) (especially the replay buffer section)
```

### connections.yaml

The `content/connections.yaml` file explicitly tracks all chapter ↔ paper relationships:

```yaml
papers:
  - slug: prioritized-experience-replay
    chapters:
      - slug: deep-q-networks
        relationship: extends      # This paper extends concepts from this chapter
        context: "PER improves upon the uniform replay buffer introduced here"
      - slug: exploration-exploitation
        relationship: related      # Tangentially related
        context: "Priority can be seen as a form of curiosity-driven sampling"

  - slug: dqn-nature-2015
    chapters:
      - slug: deep-q-networks
        relationship: foundational  # This paper IS the chapter's foundation
        context: "The seminal DQN paper that this chapter explains"
```

**Relationship types:**
- `foundational`: The paper is central to the chapter's content
- `extends`: The paper builds on chapter concepts
- `applies`: The paper applies techniques from the chapter
- `related`: Tangential connection worth noting
- `critiques`: The paper challenges ideas from the chapter

## Working with This Repository

### Prompt System Overview

This project uses a layered prompt system for content generation:

```
prompts/
├── PRINCIPLES.md         # Core content principles (always read first)
├── STYLE_GUIDE.md        # Writing style and tone
├── MATH_CONVENTIONS.md   # Mathematical notation standards
├── CODE_STANDARDS.md     # Code example standards
├── MDX_AUTHORING.md      # MDX syntax rules (critical!)
└── templates/
    ├── chapter.md        # Template for chapter prompts
    ├── subsection.md     # Template for subsection prompts
    ├── paper.md          # Template for paper prompts
    └── ...

content/chapters/{dirName}/
├── prompt.md             # Chapter-specific prompt (required)
├── index.mdx             # Chapter overview content
├── {subsection}.mdx      # Individual subsection content
└── assets/               # Images, data files
```

### Chapter vs Subsection Prompts

**Chapter prompt** (`content/chapters/{dirName}/prompt.md`):
- Defines overall chapter scope, learning objectives, narrative arc
- Lists all subsections and what each covers
- Specifies cross-references, interactive elements, exercises
- **Required** for every chapter

**Subsection prompts** (optional, for finer control):
- Use when a specific subsection needs detailed guidance
- Useful for complex topics, iterative refinement, or quality fixes
- Follow template at `prompts/templates/subsection.md`

### How to Generate Content

**Step 1: Read the foundation documents**
1. `prompts/PRINCIPLES.md` - Core philosophy
2. `prompts/STYLE_GUIDE.md` - Writing style
3. `prompts/MDX_AUTHORING.md` - **Critical** syntax rules

**Step 2: Read the chapter prompt**
- `content/chapters/{dirName}/prompt.md`
- Contains learning objectives, concepts, narrative arc

**Step 3: Check existing content for context**
- Read `src/lib/chapters.ts` for chapter/subsection structure
- Review adjacent chapters for continuity

**Step 4: Generate content following the prompt**
- Create `index.mdx` for chapter overview
- Create `{subsection-slug}.mdx` for each subsection
- Use all three complexity layers (`<Intuition>`, `<Mathematical>`, `<Implementation>`)

**Step 5: Test**
```bash
npm run build  # Must pass!
```

### Updating Prompts

**Process for prompt changes:**
1. **Open an issue** describing the proposed change
2. **Discuss impact** on existing content
3. **Get editor approval** (Enes)
4. **Update the prompt**
5. **Regenerate affected content**
6. **Review and publish**

**Why this matters:** Prompts are versioned assets. Changes affect all content generated from them. Small prompt tweaks can have large downstream effects.

### For AI Assistants (Claude)

When generating or modifying content:

1. **Always read the relevant prompt first**: Check both `/prompts/PRINCIPLES.md` and the content-specific `prompt.md` (chapter or paper)

2. **Follow MDX syntax rules**: Read `/prompts/MDX_AUTHORING.md` before writing content. Critical issues:
   - **Avoid `\begin{cases}`** - MDX parser breaks on this LaTeX construct. Use bullet lists instead.
   - **Avoid `|x|` in table cells** - Pipe characters conflict with markdown table syntax.
   - **Escape `<`, `>`, `{`, `}`** in prose - They're interpreted as JSX.
   - **Always test with `npm run build`** before considering content complete.

3. **Maintain continuity**: Reference the recurring examples (GridWorld, CliffWalking, etc.) established in earlier chapters

4. **Structure for complexity toggle**: Wrap content in appropriate `<Intuition>`, `<Mathematical>`, `<Implementation>` components

5. **Interactive-first**: Every concept explanation should suggest or include an interactive demo

6. **Cross-references**: Link to foundational concepts when building on them. Use standard markdown links: `[Q-Learning Basics](/chapters/q-learning-basics)`

7. **Consistent notation**: Follow `/prompts/MATH_CONVENTIONS.md` for all mathematical expressions

8. **Update connections.yaml**: When creating or modifying papers, always update `content/connections.yaml` to reflect chapter relationships

9. **Papers are standalone**: Paper explanations should be self-contained for readers coming from a direct link (e.g., reading group). Include a **Prerequisites** section with links to required chapters

### Iterating on Existing Content

When the user is iterating on a chapter or subsection (giving feedback, requesting changes):

1. **At session start**: Read the chapter's `prompt.md` file, including its **Iteration Notes** section
2. **During session**: Make requested changes to the content (.mdx files)
3. **At session end**: When the user says they're done or the session is wrapping up:
   - **Update the prompt's Iteration Notes** with significant decisions, issues fixed, and what worked well
   - Ask: "Should I update the chapter prompt with notes from this session?"

**What to capture in Iteration Notes:**
- Decisions with rationale (e.g., "Used CliffWalking instead of GridWorld for TD error demo")
- Style preferences specific to this chapter
- Known issues not yet fixed
- Approaches that worked well

**Example update:**
```markdown
## Iteration Notes

### Decisions Made
- 2024-12-30: Simplified the bootstrap diagram after user feedback
- 2024-12-30: Added explicit numpy import to all code examples

### Known Issues
- [ ] Section 3 code example needs error handling for edge case
```

This ensures context persists across sessions. If a pattern emerges across multiple chapters, elevate it to CLAUDE.md or the foundation docs.

## Technical Implementation Notes

### Content Structure

This project uses a three-level hierarchy:
- **Section**: Groups of related chapters (e.g., "Foundations", "Q-Learning Foundations")
- **Chapter**: Main topic (e.g., "Multi-Armed Bandits")
- **Subsection**: Individual lessons within a chapter (e.g., "UCB", "Thompson Sampling")

Chapters are defined in `src/lib/chapters.ts`. Each chapter can have subsections that appear as separate pages.

### Dark Mode

The site defaults to dark mode. Dark theme is set via the `dark` class on the `<html>` element.
- Always include `dark:` variants for all color classes
- Test all new components in dark mode
- ThemeToggle has been removed - dark mode is the default and only theme

### Content Layers and Complexity Toggle

The complexity toggle (`<ComplexityToggle />`) controls visibility of `<Mathematical>` and `<Implementation>` sections.

**How it works:**
- ContentLayers components (`Mathematical`, `Implementation`) have `data-layer="math"` and `data-layer="code"` attributes
- ComplexityToggle adds/removes `hide-math` and `hide-code` classes on `document.body`
- CSS rules in `global.css` hide layers when body has corresponding class:
  ```css
  body.hide-math [data-layer="math"] { display: none; }
  body.hide-code [data-layer="code"] { display: none; }
  ```

**Why not React Context?** MDX content is server-rendered at build time, so React hooks/context don't work in MDX components. The CSS-based approach works because it's applied at runtime via client-side JS.

### Prerequisites

Prerequisites in chapter frontmatter must reference **existing** chapter slugs. Before adding a prerequisite, verify the chapter exists in `src/lib/chapters.ts`.

```yaml
# Good - chapter exists
prerequisites:
  - slug: "intro-to-rl"
    title: "Introduction to RL"

# Bad - chapter doesn't exist, will create broken link
prerequisites:
  - slug: "markov-decision-processes"  # Not in chapters.ts!
    title: "MDPs"
```

### Giscus Comments

Comments use Giscus connected to GitHub Discussions. The configuration is in ChapterLayout.astro and SubsectionLayout.astro:
- repo: ebilgin/rlbook
- Theme is hardcoded to "dark" since that's the site's only theme

### For Human Contributors

1. **Prompt changes require review**: Prompts affect all generated content. Discuss in issues first.

2. **Content corrections**: Fix typos and errors directly. For substantial rewrites, update the prompt too.

3. **New interactives**: Add to `/src/components/interactive/` with documentation.

4. **Testing locally**: `npm run dev` starts the dev server with hot reload.

## Current Chapters

```
content/chapters/
├── 0010-intro-to-rl/              # Introduction to RL
├── 0020-multi-armed-bandits/      # Exploration-exploitation basics
├── 0030-contextual-bandits/       # Context-dependent decisions
├── 1010-intro-to-td/              # TD Learning foundations
├── 1020-q-learning-basics/        # Tabular Q-learning
├── 1030-exploration-exploitation/ # ε-greedy, UCB, etc.
├── 1040-deep-q-networks/          # DQN and variants
├── 1050-q-learning-applications/  # Practical applications
├── 1060-q-learning-frontiers/     # Current research
├── 2010-intro-to-policy-gradients/    # Policy-based methods intro
├── 2020-policy-gradient-theorem/      # REINFORCE algorithm
├── 2030-actor-critic-methods/         # A2C, A3C
├── 2040-ppo-and-advanced-pg/          # PPO, TRPO
└── 2050-policy-methods-applications/  # RLHF, robotics, etc.
```

**Note**: Directory numbering uses increments of 10 (1010, 1020, etc.) to allow inserting new chapters between existing ones without renumbering. Navigation uses slugs (e.g., `q-learning-basics`) rather than numbers to ensure links remain stable.

## Build Commands

```bash
npm run build        # Production build (must pass before committing content!)
npm run dev          # Start dev server for local preview
```

For full setup instructions, see [README.md](README.md#quick-start).
