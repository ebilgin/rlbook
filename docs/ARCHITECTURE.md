# Technical Architecture

This document describes the technical architecture of rlbook.ai.

## System Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        Cloudflare Edge                          │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐ │
│  │  Cloudflare     │  │  Cloudflare     │  │  Cloudflare     │ │
│  │  Pages          │  │  R2 (Assets)    │  │  KV (Optional)  │ │
│  └────────┬────────┘  └────────┬────────┘  └────────┬────────┘ │
└───────────┼─────────────────────┼─────────────────────┼─────────┘
            │                     │                     │
            ▼                     ▼                     ▼
┌─────────────────────────────────────────────────────────────────┐
│                         User Browser                             │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐ │
│  │  Static HTML    │  │  React Islands  │  │  TensorFlow.js  │ │
│  │  (Content)      │  │  (Interactives) │  │  (ML Runtime)   │ │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

## Technology Stack

### Build & Framework

| Component | Technology | Purpose |
|-----------|------------|---------|
| Framework | Astro 4.x | Static site generation, islands architecture |
| Content | MDX | Markdown with React components |
| Components | React 18 | Interactive elements |
| Styling | Tailwind CSS | Utility-first styling |
| Type Safety | TypeScript | Static type checking |

### Client-Side Runtime

| Component | Technology | Purpose |
|-----------|------------|---------|
| ML Runtime | TensorFlow.js | In-browser RL training/inference |
| Physics | MuJoCo WASM / Rapier.js | Environment simulation |
| Visualization | D3.js | Data visualization |
| 3D | React Three Fiber | Robotics visualizations |
| Animation | GSAP | Scroll-based animations |
| Math | KaTeX | LaTeX rendering |

### Hosting & Infrastructure

| Component | Technology | Purpose |
|-----------|------------|---------|
| Hosting | Cloudflare Pages | Static hosting, edge delivery |
| Assets | Cloudflare R2 | Large file storage (WASM, models) |
| Comments | Giscus | GitHub Discussions-powered |
| Analytics | (TBD) | Privacy-friendly analytics |

---

## Astro Islands Architecture

Astro's islands architecture is key to performance:

```astro
---
// Server-side: runs at build time
import StaticContent from './StaticContent.astro';
import InteractiveDemo from './InteractiveDemo.tsx';
---

<!-- Static HTML: 0 KB JavaScript -->
<StaticContent />

<!-- Hydrates only when visible: loads JS on demand -->
<InteractiveDemo client:visible />
```

### Hydration Strategies

| Directive | When JS Loads | Use Case |
|-----------|---------------|----------|
| `client:load` | Immediately | Critical interactives |
| `client:visible` | When scrolled into view | Most demos |
| `client:idle` | When browser is idle | Non-critical interactives |
| `client:only="react"` | Client-only, no SSR | Heavy demos |

---

## Directory Structure

```
rlbook/
├── astro.config.mjs          # Astro configuration
├── tailwind.config.js        # Tailwind configuration
├── tsconfig.json             # TypeScript configuration
├── package.json              # Dependencies
│
├── content/
│   └── chapters/
│       └── XXXX-chapter-name/
│           ├── prompt.md           # Chapter prompt (required)
│           ├── index.mdx           # Chapter overview
│           ├── {subsection}.mdx    # Subsection pages
│           └── assets/             # Chapter-specific assets
│
├── prompts/
│   ├── PRINCIPLES.md         # Core content principles
│   ├── STYLE_GUIDE.md        # Writing style guide
│   ├── MATH_CONVENTIONS.md   # Math notation standards
│   ├── CODE_STANDARDS.md     # Code example standards
│   └── templates/            # Prompt templates
│
├── src/
│   ├── components/
│   │   ├── interactive/      # RL demos (React)
│   │   │   ├── GridWorld.tsx
│   │   │   ├── QLearningDemo.tsx
│   │   │   └── ...
│   │   ├── visualization/    # Charts, graphs (D3/React)
│   │   ├── ui/               # UI components
│   │   │   ├── ComplexityToggle.tsx
│   │   │   ├── ContentStatus.tsx
│   │   │   └── ...
│   │   └── math/             # Math rendering
│   │       └── MathBlock.tsx
│   │
│   ├── layouts/
│   │   ├── BaseLayout.astro
│   │   └── ChapterLayout.astro
│   │
│   ├── styles/
│   │   └── global.css
│   │
│   └── lib/
│       ├── rl/               # RL primitives (shared)
│       │   ├── environments.ts
│       │   ├── agents.ts
│       │   └── utils.ts
│       └── utils/            # General utilities
│
├── notebooks/                # Colab notebooks (PyTorch)
│   ├── 10_intro_to_td.ipynb
│   └── ...
│
├── public/
│   ├── models/              # Pre-trained models (ONNX)
│   ├── wasm/                # WASM binaries
│   └── assets/              # Static images, etc.
│
├── docs/
│   ├── CONTRIBUTING.md
│   ├── CONTENT_WORKFLOW.md
│   └── ARCHITECTURE.md
│
└── .github/
    ├── ISSUE_TEMPLATE/
    └── workflows/
        ├── build.yml        # Build and type checking
        ├── preview.yml      # PR preview deployments
        └── deploy.yml       # Production deployment
```

---

## Component Architecture

### Complexity Toggle System

```typescript
// Context for complexity preferences
interface ComplexityContext {
  showMath: boolean;
  showCode: boolean;
  setShowMath: (show: boolean) => void;
  setShowCode: (show: boolean) => void;
}

// Wrapper components for content layers
<Intuition>Always visible content</Intuition>
<Mathematical>Expandable math content</Mathematical>
<Implementation>Expandable code content</Implementation>
```

### Interactive Demo Pattern

```typescript
// Standard demo component structure
interface DemoProps {
  // Configuration
  width?: number;
  height?: number;

  // Callbacks
  onStateChange?: (state: GameState) => void;

  // Control
  autoPlay?: boolean;
  speed?: number;
}

function GridWorldDemo({ width = 5, height = 5, ...props }: DemoProps) {
  // State management
  const [gameState, setGameState] = useState<GameState>(initialState);

  // RL logic
  const agent = useAgent(config);
  const env = useEnvironment({ width, height });

  // Render
  return (
    <DemoContainer>
      <Canvas>{/* Visualization */}</Canvas>
      <Controls>{/* Sliders, buttons */}</Controls>
      <Metrics>{/* Q-values, rewards */}</Metrics>
    </DemoContainer>
  );
}
```

---

## Data Flow

### Content Build Pipeline

```
MDX Files → Astro Build → Static HTML + Islands → Cloudflare Pages
    │
    ├── Parse frontmatter (status, prerequisites)
    ├── Transform MDX to React
    ├── Extract component islands
    ├── Bundle island JavaScript
    └── Generate static HTML
```

### Interactive Demo Runtime

```
User Interaction → State Update → RL Step → Visualization Update
        │              │            │              │
        ▼              ▼            ▼              ▼
   Event Handler  React State  TF.js/WASM    Canvas/SVG
```

---

## Performance Targets

| Metric | Target | Strategy |
|--------|--------|----------|
| First Contentful Paint | < 1.5s | Static HTML, edge caching |
| Largest Contentful Paint | < 2.5s | Lazy load images, islands |
| Time to Interactive | < 3.5s | Defer non-critical JS |
| Demo Load Time | < 2s | Code splitting, lazy hydration |
| Bundle Size (main) | < 100KB | Tree shaking, component islands |
| WASM Load | < 5s | R2 + edge caching |

### Optimization Techniques

1. **Static Generation**: All content pre-rendered at build time
2. **Island Hydration**: Only interactive components ship JS
3. **Code Splitting**: Each demo is a separate bundle
4. **Edge Caching**: Cloudflare caches at 200+ locations
5. **Asset Optimization**: Images compressed, WASM cached
6. **Lazy Loading**: Demos load when scrolled into view

---

## Browser Support

| Browser | Version | Notes |
|---------|---------|-------|
| Chrome | 113+ | Full support including WebGPU |
| Firefox | 141+ | Full support including WebGPU |
| Safari | 17+ | WebGL fallback for TensorFlow.js |
| Edge | 113+ | Full support |

### Feature Detection

```typescript
// Check for WebGPU availability
const hasWebGPU = 'gpu' in navigator;

// Fallback for TensorFlow.js backend
if (hasWebGPU) {
  await tf.setBackend('webgpu');
} else {
  await tf.setBackend('webgl');
}
```

---

## Deployment Pipeline

### Preview Deployments (PRs)

```yaml
# .github/workflows/preview.yml
on: pull_request

jobs:
  preview:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-node@v4
      - run: npm ci
      - run: npm run build
      - uses: cloudflare/pages-action@v1
        with:
          branch: ${{ github.head_ref }}
```

### Production Deployment

```yaml
# .github/workflows/deploy.yml
on:
  push:
    branches: [main]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-node@v4
      - run: npm ci
      - run: npm run build
      - uses: cloudflare/pages-action@v1
        with:
          branch: main
          production: true
```

---

## Security Considerations

1. **No Server-Side Code**: Static site, no server vulnerabilities
2. **Content Security Policy**: Strict CSP headers
3. **Dependency Auditing**: Regular npm audit
4. **HTTPS Only**: Enforced by Cloudflare
5. **No User Data Storage**: Progress in localStorage only

---

## Monitoring & Analytics

### Build Monitoring

- GitHub Actions for CI/CD status
- Cloudflare Pages deployment logs

### Runtime Monitoring (Planned)

- Privacy-friendly analytics (Plausible or similar)
- Core Web Vitals tracking
- Demo error reporting

---

## Local Development

```bash
# Install dependencies
npm install

# Start development server
npm run dev

# Type checking
npm run check

# Build for production
npm run build

# Preview production build
npm run preview
```

### Environment Variables

```bash
# .env (optional)
PUBLIC_GISCUS_REPO=ebilgin/rlbook-discussions
PUBLIC_GISCUS_CATEGORY=Chapter Discussions
```
