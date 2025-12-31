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
5. (Optional) Create subsection prompts for finer control

### Outputs

- `content/chapters/XXXX-name/prompt.md` (chapter prompt)
- `content/chapters/XXXX-name/prompts/subsection-name.md` (optional subsection prompts)
- Demo specifications in prompt
- Exercise outlines

### Chapter vs Subsection Prompts

**Chapter prompts** define the overall scope:
- Learning objectives for the entire chapter
- Narrative arc and story structure
- Which subsections exist and their order
- Cross-chapter connections

**Subsection prompts** provide finer-grained control when needed:
- Specific objectives for individual subsections
- Detailed content scope (what to cover, what to defer)
- Narrative flow (how to connect to previous/next subsections)
- Particular code examples or mathematical derivations

Use subsection prompts when:
- A topic is complex and needs detailed specification
- Multiple authors/sessions are working on different parts
- Iterative refinement of specific sections is needed
- Quality issues in a specific subsection need targeted fixing

Skip subsection prompts when:
- Topics follow naturally from the chapter outline
- The chapter prompt's "Core Concepts" section provides enough detail
- Doing initial content generation

---

## Phase 2: AI Generation

### Who

- AI (Claude) with human oversight

### Inputs

- Chapter prompt (required)
- Subsection prompts (optional, for finer control)
- Global principles (PRINCIPLES.md, STYLE_GUIDE.md, etc.)
- Existing content for continuity

### Process

#### Generating a Full Chapter

1. Load all relevant prompts and principles
2. Generate content following structure in prompt
3. Create all three complexity layers (Intuition, Mathematical, Implementation)
4. Generate exercises with solutions
5. Output as MDX with component placeholders

#### Generating Individual Subsections

When using subsection prompts for finer control:

1. Load the chapter prompt for context
2. Load the specific subsection prompt
3. Load global principles (PRINCIPLES.md, STYLE_GUIDE.md, MDX_AUTHORING.md, MATH_CONVENTIONS.md)
4. Review previous subsection content for continuity
5. Generate the subsection with proper transitions
6. Output as `{subsection-slug}.mdx` in the chapter directory

### Outputs

**Chapter-level generation:**
- `content/chapters/XXXX-name/index.mdx` (status: draft)
- Exercise files
- Code examples

**Subsection-level generation:**
- `content/chapters/XXXX-name/{subsection-slug}.mdx`
- Must be registered in `src/lib/chapters.ts` under the chapter's `subsections` array

### Quality Checks

- [ ] All learning objectives addressed
- [ ] Three complexity layers present
- [ ] Cross-references in place
- [ ] Code examples complete and syntactically correct
- [ ] Mathematical notation follows conventions
- [ ] Subsection transitions flow naturally

### Ensuring Prompts and Context are Used

**For Claude Code users:**
1. `CLAUDE.md` is automatically loaded — it references all foundation documents
2. Reference specific prompts in your request:
   ```
   Generate content for intro-to-td following the prompt at
   content/chapters/1010-intro-to-td/prompt.md
   ```

**For other AI tools:**
1. Load `CLAUDE.md` first — it lists all required foundation documents
2. Load the specific chapter/subsection prompt
3. Generate content following the loaded context

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

### Capturing Iteration Context

When iterating on content, learnings arise that should persist across sessions. Use the **Iteration Notes** section in chapter/subsection prompts to capture:

- **Decisions Made**: Choices with rationale (e.g., "2024-01-15: Used numpy instead of pure Python for performance")
- **Style Preferences**: Chapter-specific style choices
- **Known Issues**: Problems identified but not yet fixed
- **What Worked Well**: Approaches to replicate

**Workflow:**
1. After each significant revision, update the prompt's Iteration Notes section
2. Next session reads these notes for context
3. Global patterns should be elevated to CLAUDE.md or foundation docs

**Example:**
```markdown
## Iteration Notes

### Decisions Made
- 2024-01-15: Kept epsilon-greedy before UCB to build intuition gradually
- 2024-01-16: Added GridWorld visualization after user feedback on clarity

### Known Issues
- [ ] Code example in section 3 needs NumPy 2.0 compatibility check
```

### Prompt Change Process

1. Open issue describing proposed change
2. Discuss impact on existing content
3. Get editor approval
4. Update prompt
5. Regenerate affected content
6. Review and publish

### Updating Prompts: Step by Step

**For minor fixes (typos, clarifications):**
1. Edit the prompt file directly
2. Commit with message: `prompt: Fix typo in chapter X prompt`
3. No regeneration needed unless content is affected

**For content-affecting changes:**
1. Edit the prompt in `content/chapters/XXXX-name/prompt.md`
2. If updating subsection prompts, edit in `content/chapters/XXXX-name/prompts/`
3. Regenerate affected content
4. Review generated output
5. Commit both prompt and content changes together
6. PR with description of what changed and why

**For template changes (affects all future content):**
1. Update template in `prompts/templates/chapter.md` or `prompts/templates/subsection.md`
2. Consider if existing prompts need updating to match
3. Document the change in the template file itself
4. Update CLAUDE.md if the change affects conventions

### Prompt Directory Structure

```
content/chapters/XXXX-chapter-name/
├── prompt.md              # Main chapter prompt
├── prompts/               # Optional subsection prompts
│   ├── subsection-1.md
│   └── subsection-2.md
├── index.mdx              # Chapter overview content
├── subsection-1.mdx       # Subsection content files
└── subsection-2.mdx
```

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
