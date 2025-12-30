# Contributing to rlbook.ai

Thank you for your interest in contributing to rlbook.ai! This project aims to be the ultimate interactive resource for learning reinforcement learning, and community input is essential to achieving that goal.

## Ways to Contribute

### 1. Report Issues

Found a bug, typo, or unclear explanation?

- **Small fixes** (typos, broken links): Open an issue or PR directly
- **Conceptual issues** (incorrect explanations): Open an issue for discussion first
- **Interactive demo bugs**: Include browser/OS info and steps to reproduce

### 2. Suggest Improvements

Have ideas for better explanations or examples?

- Open an issue with the `enhancement` label
- Describe what's unclear and how it could be improved
- Reference specific chapters/sections

### 3. Contribute Content

Want to write or improve content?

- **Prompt improvements**: Suggest changes to prompts that generate content
- **Direct content edits**: For reviewed content, submit PRs
- **New exercises**: Add conceptual or coding challenges
- **Colab notebooks**: Contribute PyTorch notebooks for deeper exploration

### 4. Build Interactive Demos

- Check open issues labeled `interactive`
- Follow the demo specification format in [prompts/templates/interactive.md](../prompts/templates/interactive.md)
- Test across browsers (Chrome, Firefox, Safari)

### 5. Review and Discuss

- Participate in GitHub Discussions (via Giscus on the site)
- Review PRs from other contributors
- Help answer questions in discussions

---

## Content Review Process

All content goes through a review pipeline:

```
Draft (AI-Generated) → Editor Reviewed → Community Reviewed → Verified
```

### Status Definitions

| Status | Meaning | Who Can Change |
|--------|---------|----------------|
| `draft` | Generated from prompts, not yet reviewed | AI generation |
| `editor_reviewed` | Reviewed and approved by Enes | Editor only |
| `community_reviewed` | Incorporates community feedback | After discussion resolution |
| `verified` | Code tested, demos verified working | After technical verification |

### How to Request Review

1. Comment on the chapter's discussion thread
2. Tag specific issues or improvements
3. Wait for editor response before major changes

---

## Prompt Contribution Guidelines

Prompts are first-class assets. Changes to prompts affect all generated content.

### Proposing Prompt Changes

1. **Open an issue first** — discuss the change before implementing
2. Explain: What's wrong with current prompt? What's the expected improvement?
3. Provide: Example of content before/after (if possible)

### Prompt PR Requirements

- [ ] Issue link explaining motivation
- [ ] Editor approval on the approach
- [ ] No breaking changes to existing content structure
- [ ] Updated any affected content (if regeneration needed)

---

## Code Contribution Guidelines

### Interactive Components (TypeScript/React)

```bash
# Setup
npm install
npm run dev

# Check types and lint
npm run check

# Build for production
npm run build
```

**Standards:**
- TypeScript strict mode
- React functional components with hooks
- Follow existing component patterns in `/src/components`
- Include accessibility features (keyboard nav, ARIA labels)

### Colab Notebooks (Python/PyTorch)

**Notebook Requirements:**
- Clear markdown explanations between code cells
- All dependencies installable via `!pip install`
- Deterministic results (set random seeds)
- Tested on Colab's free tier (no GPU required for basic notebooks)

**Naming Convention:**
```
notebooks/
├── 10_intro_to_td.ipynb
├── 11_q_learning_basics.ipynb
└── ...
```

### Python Code Standards

- Python 3.9+
- Follow PEP 8 (use Black formatter)
- Type hints for function signatures
- NumPy/PyTorch for numerical code
- Gymnasium for environments

---

## Pull Request Process

### 1. Before Submitting

- [ ] Fork the repository
- [ ] Create a feature branch (`git checkout -b feature/your-feature`)
- [ ] Follow relevant code standards
- [ ] Test your changes locally
- [ ] Update documentation if needed

### 2. PR Description

Include:
- **Summary**: What does this PR do?
- **Motivation**: Why is this change needed?
- **Testing**: How did you verify it works?
- **Related Issues**: Link to relevant issues

### 3. Review Process

1. Automated checks run (linting, build, link validation)
2. Preview deployment generated for visual review
3. Maintainer reviews code and content
4. Discussion and iteration as needed
5. Merge when approved

---

## Development Setup

### Prerequisites

- Node.js 18+
- npm 9+
- Git

### Local Development

```bash
# Clone the repository
git clone https://github.com/ebilgin/rlbook.git
cd rlbook

# Install dependencies
npm install

# Start development server
npm run dev

# Open http://localhost:4321
```

### Project Structure

```
rlbook/
├── content/chapters/     # MDX content
├── prompts/              # Prompt templates and principles
├── src/components/       # React components
├── notebooks/            # Colab notebooks (PyTorch)
├── public/               # Static assets
└── docs/                 # Project documentation
```

---

## Style Guidelines

### Writing Style

- Follow [STYLE_GUIDE.md](../prompts/STYLE_GUIDE.md)
- Write for three reader types: practitioners, theorists, explorers
- Use progressive disclosure (intuition → math → code)
- Include interactive elements where possible

### Code Examples

- Follow [CODE_STANDARDS.md](../prompts/CODE_STANDARDS.md)
- Complete and runnable (no `...` or `pass`)
- Well-commented for educational purposes
- Tested before submission

### Mathematical Notation

- Follow [MATH_CONVENTIONS.md](../prompts/MATH_CONVENTIONS.md)
- Use consistent symbols throughout
- Explain notation when introducing it

---

## Community Guidelines

### Be Respectful

- Welcome newcomers
- Provide constructive feedback
- Assume good intentions
- Credit others' contributions

### Be Collaborative

- Discuss before major changes
- Share knowledge generously
- Help others succeed

### Be Patient

- Reviews take time
- Iteration is normal
- Quality over speed

---

## Getting Help

- **Questions about contributing**: Open a discussion
- **Technical issues**: Open an issue with reproduction steps
- **Content questions**: Comment on the relevant chapter discussion

---

## License

By contributing, you agree that your contributions will be licensed under:
- **Content**: CC BY-NC-SA 4.0
- **Code**: MIT License

---

Thank you for helping make rlbook.ai the best resource for learning reinforcement learning!
