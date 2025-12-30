# Content Workflow: From Prompt to Publication

This document describes the workflow for creating and maintaining content on rlbook.ai.

## Overview

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  Prompt Design  │ ──► │  AI Generation  │ ──► │  Editor Review  │
└─────────────────┘     └─────────────────┘     └─────────────────┘
         │                                               │
         │                                               ▼
         │                                      ┌─────────────────┐
         │                                      │   Publication   │
         │                                      └─────────────────┘
         │                                               │
         ▼                                               ▼
┌─────────────────┐                             ┌─────────────────┐
│ Prompt Iteration│ ◄────────────────────────── │Community Review │
└─────────────────┘                             └─────────────────┘
```

## Phase 1: Prompt Design

### Who

- Curriculum Designer (Enes) for chapter scope and objectives
- Community for suggestions and feedback

### Inputs

- Learning objectives for the chapter
- Prerequisites and connections to other chapters
- Required interactive elements
- Target audience considerations

### Process

1. Create chapter prompt using [templates/chapter.md](../prompts/templates/chapter.md)
2. Define interactive demo specifications
3. Specify cross-references and continuity points
4. Review against [PRINCIPLES.md](../prompts/PRINCIPLES.md)

### Outputs

- `content/chapters/XX-name/prompt.md`
- Demo specifications in prompt
- Exercise outlines

---

## Phase 2: AI Generation

### Who

- AI (Claude) with human oversight

### Inputs

- Chapter prompt
- Global principles (PRINCIPLES.md, STYLE_GUIDE.md, etc.)
- Existing content for continuity

### Process

1. Load all relevant prompts and principles
2. Generate content following structure in prompt
3. Create all three complexity layers (Intuition, Mathematical, Implementation)
4. Generate exercises with solutions
5. Output as MDX with component placeholders

### Outputs

- `content/chapters/XX-name/index.mdx` (status: draft)
- Exercise files
- Code examples

### Quality Checks

- [ ] All learning objectives addressed
- [ ] Three complexity layers present
- [ ] Cross-references in place
- [ ] Code examples complete and syntactically correct
- [ ] Mathematical notation follows conventions

---

## Phase 3: Editor Review

### Who

- Editor (Enes)

### Inputs

- Generated content (status: draft)
- Original prompt
- Principles documents

### Process

1. Read through entire chapter
2. Verify accuracy of explanations
3. Check flow and coherence
4. Test code examples
5. Review interactive component specifications
6. Mark issues or approve

### Decision Points

| Finding | Action |
|---------|--------|
| Minor issues (typos, phrasing) | Edit directly |
| Structural issues | Note for prompt revision |
| Conceptual errors | Major revision needed |
| Missing content | Regenerate section |

### Outputs

- Updated content (status: editor_reviewed)
- Issues logged for prompt improvement
- Notes for interactive implementation

---

## Phase 4: Interactive Implementation

### Who

- Developers (community or core team)

### Inputs

- Interactive specifications from prompts
- Editor-reviewed content

### Process

1. Implement React components in TypeScript
2. Integrate TensorFlow.js for ML demos
3. Test across browsers
4. Ensure accessibility
5. Optimize performance

### Outputs

- Components in `/src/components/interactive/`
- Working demos embedded in content

---

## Phase 5: Publication

### Who

- Automated via CI/CD

### Process

1. PR merged to main branch
2. Build triggered (Astro)
3. Tests run (type checking, link validation)
4. Deploy to Cloudflare Pages
5. Cache invalidation

### Outputs

- Live content on rlbook.ai
- Preview links for review

---

## Phase 6: Community Review

### Who

- Readers via Giscus discussions
- Contributors via GitHub

### Inputs

- Published content
- Reader questions and feedback

### Process

1. Readers comment via Giscus
2. Issues/discussions triaged
3. Valid improvements identified
4. Content or prompts updated
5. Status updated to `community_reviewed`

### Feedback Types

| Type | Response |
|------|----------|
| Typo/error | Direct fix |
| Clarification needed | Improve explanation |
| New example request | Consider for prompt update |
| Technical inaccuracy | Investigate and correct |
| Interactive bug | File issue for fix |

---

## Phase 7: Verification

### Who

- Technical reviewers

### Process

1. All code examples tested
2. Interactives verified working
3. Colab notebooks run end-to-end
4. Cross-browser testing for demos

### Outputs

- Status updated to `verified`
- Verification timestamp recorded

---

## Prompt Iteration

Prompts evolve based on:

1. **Editor findings** during review
2. **Community feedback** from discussions
3. **Technical updates** (new best practices, tools)
4. **Curriculum changes** (new chapters, reorganization)

### Prompt Change Process

1. Open issue describing proposed change
2. Discuss impact on existing content
3. Get editor approval
4. Update prompt
5. Regenerate affected content
6. Review and publish

---

## Content Status Flow

```
draft ──► editor_reviewed ──► published
                                  │
                                  ▼
                         community_reviewed
                                  │
                                  ▼
                             verified
```

### Frontmatter Fields

```yaml
---
title: "Chapter Title"
chapter: 10
status: "draft"  # draft | editor_reviewed | community_reviewed | verified
lastReviewed: "2024-01-15"  # null if never reviewed
reviewedBy: "ebilgin"       # null if auto-generated
generatedFrom: "prompt-v2"  # prompt version used
---
```

---

## Colab Notebook Workflow

### Creation

1. Base notebook created from chapter content
2. PyTorch implementations added
3. Extended examples for deeper exploration
4. Testing on Colab free tier

### Maintenance

1. Keep in sync with chapter content
2. Update for library version changes
3. Add new experiments based on feedback

### Structure

Each notebook should include:
- Setup cell (installs, imports)
- Concept explanation (from chapter)
- Implementation with detailed comments
- Experiments and visualizations
- Exercises for reader to complete

---

## Version Control

### Branching Strategy

- `main`: Production content
- `content/chapter-XX`: Chapter development
- `feature/component-name`: Interactive development
- `fix/issue-number`: Bug fixes

### Commit Messages

```
content: Update chapter 11 with better Q-table visualization
component: Add GridWorld interactive demo
prompt: Improve exploration chapter prompt for clarity
fix: Correct TD error formula in chapter 10
```

---

## Quality Metrics

Track over time:
- Time from prompt to publication
- Community feedback volume and sentiment
- Code example test pass rate
- Interactive demo load times
- Reader engagement (time on page, demo interactions)
