# rlbook.ai

An open-source, community-driven resource for learning reinforcement learning.

[![Deploy to Cloudflare Pages](https://github.com/ebilgin/rlbook/actions/workflows/deploy.yml/badge.svg)](https://github.com/ebilgin/rlbook/actions/workflows/deploy.yml)

## About

rlbook.ai combines AI-assisted content generation with human curation to create high-quality educational content for reinforcement learning. Features include:

- **Interactive demos** running in your browser (TensorFlow.js)
- **Progressive complexity** - toggle math and code visibility
- **PyTorch notebooks** for deeper exploration (Google Colab)
- **Community-driven** content with transparent review status

Created by [Enes Bilgin](https://github.com/ebilgin), author of "Mastering Reinforcement Learning with Python".

## Content Types

rlbook.ai organizes content into five categories, each serving different learning goals:

| Category | Purpose | Description |
|----------|---------|-------------|
| **📚 Chapters** | Learn Concepts | Progressive lessons teaching RL from foundations to advanced topics |
| **📄 Papers** | Deep Dives | In-depth analysis of seminal research papers |
| **🎯 Applications** | Problem Formulation | End-to-end guides for formulating real problems as RL |
| **🔧 Infrastructure** | Scale & Deploy | Engineering guides for distributed training and production |
| **🎮 Environments** | Experiment | Interactive playgrounds for hands-on experimentation |
| **💻 Code** | Run & Test | Production-grade Python implementations |

See [docs/CONTENT_TYPES.md](docs/CONTENT_TYPES.md) for detailed descriptions and contribution guidelines.

## Quick Start

### Prerequisites

- Node.js 18+
- Python 3.9+ (for the RL code package)

### One-Command Setup

```bash
# Clone the repository
git clone https://github.com/ebilgin/rlbook.git
cd rlbook

# Run setup script (installs everything)
./scripts/setup.sh
```

This will:
- Install npm dependencies
- Create a Python virtual environment (`.venv/`)
- Install the `rlbook` Python package
- Run tests to verify everything works

### Manual Setup

If you prefer to set things up manually:

```bash
# Node.js setup
npm install
npm run dev  # Start dev server at http://localhost:4321

# Python setup (optional, for RL code)
python3 -m venv .venv
source .venv/bin/activate
pip install -e ./code
pytest code/tests/  # Verify installation
```

### Available Commands

| Command | Description |
|---------|-------------|
| `npm run dev` | Start development server |
| `npm run build` | Build for production |
| `npm run preview` | Preview production build locally |
| `npm run check` | Run TypeScript and Astro checks |
| `pytest code/tests/` | Run Python tests (requires venv activation) |

## Project Structure

```
rlbook/
├── content/
│   ├── chapters/           # 📚 Progressive learning content
│   │   └── XXXX-slug/      # Numbered for ordering (0010, 0020, 1010...)
│   │       ├── index.mdx   # Content
│   │       └── prompt.md   # AI generation prompt
│   ├── papers/             # 📄 Paper deep dives
│   ├── applications/       # 🎯 Problem formulation guides
│   ├── infrastructure/     # 🔧 Engineering guides
│   └── environments/       # 🎮 Interactive playgrounds
├── prompts/                # AI prompt templates and guidelines
│   ├── PRINCIPLES.md       # Core content principles
│   ├── STYLE_GUIDE.md      # Writing style guide
│   ├── MATH_CONVENTIONS.md # Math notation standards
│   ├── MDX_AUTHORING.md    # MDX syntax rules (critical!)
│   └── templates/          # Reusable prompt templates
├── code/                   # 💻 Python package
│   ├── rlbook/             # Installable package
│   │   ├── envs/           # Environment implementations
│   │   ├── agents/         # Agent implementations
│   │   ├── utils/          # Utilities (replay buffer, plotting)
│   │   └── examples/       # Runnable training scripts
│   └── tests/              # Unit tests
├── src/
│   ├── components/         # React components
│   │   ├── interactive/    # RL demos (GridWorld, etc.)
│   │   └── ui/             # UI components
│   ├── layouts/            # Astro layouts
│   ├── pages/              # Astro pages
│   └── styles/             # Global CSS
├── notebooks/              # Google Colab notebooks (PyTorch)
├── scripts/                # Development scripts
│   └── setup.sh            # One-command project setup
├── public/                 # Static assets
└── docs/                   # Project documentation
```

## Content Creation

This project uses AI-assisted content generation. Prompts are first-class assets that define what content gets generated.

### Creating Content

Each content type has its own structure and guidelines. See [docs/CONTENT_TYPES.md](docs/CONTENT_TYPES.md) for details.

**Quick example for chapters:**

1. Create the directory: `mkdir -p content/chapters/1025-my-chapter/{exercises,assets}`
2. Write the prompt: `content/chapters/1025-my-chapter/prompt.md`
3. Generate content using Claude Code, Claude.ai, or the API
4. Review using `prompts/EDITOR_REVIEW.md`

### Generating Content from Prompts

#### Option 1: Claude Code CLI (Recommended)

```bash
# Install Claude Code
npm install -g @anthropic-ai/claude-code

# Navigate to the repo and start Claude
cd rlbook
claude

# Then ask Claude:
# "Read content/chapters/1020-q-learning-basics/prompt.md and generate
# the chapter content following all principles in prompts/PRINCIPLES.md."
```

#### Option 2: Claude.ai Web Interface

1. Open [claude.ai](https://claude.ai)
2. Upload: `prompts/PRINCIPLES.md`, `prompts/STYLE_GUIDE.md`, and the content's `prompt.md`
3. Ask Claude to generate the content
4. Copy output to `index.mdx`

#### Option 3: API Integration

```python
import anthropic

client = anthropic.Anthropic()

with open("prompts/PRINCIPLES.md") as f:
    principles = f.read()
with open("content/chapters/1020-q-learning-basics/prompt.md") as f:
    chapter_prompt = f.read()

message = client.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=8000,
    messages=[{
        "role": "user",
        "content": f"{principles}\n\n---\n\n{chapter_prompt}\n\nGenerate the chapter content in MDX format."
    }]
)

print(message.content[0].text)
```

### Key Principles

- **Prompts before content**: Write/refine the prompt, then generate content
- **Use slugs, not numbers**: Reference content by slug (e.g., `q-learning-basics`)
- **Three complexity layers**: Intuition → Mathematical → Implementation
- **Interactive-first**: Every concept should have or suggest an interactive demo
- **Test builds**: Always run `npm run build` before committing MDX content

See [prompts/PRINCIPLES.md](prompts/PRINCIPLES.md) for detailed guidelines.

### MDX Syntax (Critical)

Content is written in MDX, which has some parsing quirks. **Read [prompts/MDX_AUTHORING.md](prompts/MDX_AUTHORING.md) before writing content.**

Quick rules:
- Avoid `\begin{cases}` in LaTeX (use bullet lists instead)
- Avoid `|x|` in table cells (conflicts with markdown tables)
- Escape `<`, `>`, `{`, `}` in prose

## Content Status

All content shows its review status:

| Status | Icon | Meaning |
|--------|------|---------|
| Draft | 📝 | AI-generated, pending review |
| Editor Reviewed | ✅ | Approved by editor |
| Community Reviewed | 👥 | Incorporates community feedback |
| Verified | 🔒 | Code tested, demos working |

## Deployment

### Cloudflare Pages (Recommended)

1. **GitHub Integration:**
   - Go to [Cloudflare Dashboard](https://dash.cloudflare.com/) → Pages
   - Connect to Git and select this repository
   - Build command: `npm run build`
   - Output directory: `dist`

2. **GitHub Actions (Current Setup):**
   - Add secrets: `CLOUDFLARE_API_TOKEN`, `CLOUDFLARE_ACCOUNT_ID`
   - Commits to `main` trigger production deployment
   - PRs get preview deployments

### Manual Deployment

```bash
npm run build
npx wrangler pages deploy dist --project-name=rlbook
```

## Contributing

We welcome contributions! See [docs/CONTRIBUTING.md](docs/CONTRIBUTING.md) for guidelines.

### Quick Ways to Contribute

- **Report issues**: Typos, bugs, unclear explanations
- **Improve prompts**: Better prompts = better content
- **Build demos**: Check issues labeled `interactive`
- **Review content**: Use checklist in `prompts/EDITOR_REVIEW.md`
- **Add applications**: Share your RL problem formulations
- **Write paper analyses**: Deep dive into seminal papers
- **Contribute code**: Add tested implementations to `code/rlbook/`

## Tech Stack

- **Framework**: [Astro](https://astro.build/) with islands architecture
- **Content**: MDX with React components
- **Styling**: Tailwind CSS
- **Math**: KaTeX for LaTeX rendering
- **ML Runtime**: TensorFlow.js (browser)
- **Python Package**: PyTorch + Gymnasium (code/)
- **Notebooks**: PyTorch (Colab)
- **Hosting**: Cloudflare Pages + R2
- **Comments**: Giscus (GitHub Discussions)

## License

- **Content**: [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/)
- **Code**: [MIT](LICENSE)
