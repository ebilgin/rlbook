# Visual Patterns Guide

This guide documents reusable visual patterns for rlbook.ai content. Use these patterns to create consistent, engaging visuals across chapters.

## Philosophy

- **Dark mode first**: All patterns assume dark theme
- **Inline over external**: Prefer Tailwind-styled JSX over images when possible
- **Scannable**: Visual elements should communicate at a glance
- **Consistent color semantics**: Colors carry meaning across the site

---

## Color Conventions

Use consistent colors to convey meaning:

### RL-Specific Colors (for RL chapters)

| Color | Semantic Meaning | Use For |
|-------|------------------|---------|
| **Cyan** | Supervised learning, labeled data | Supervised learning concepts, ground truth |
| **Violet** | Unsupervised learning, discovery | Clustering, pattern finding, unsupervised concepts |
| **Amber** | Reinforcement learning, rewards | RL concepts, rewards, the "hero" option |
| **Blue** | Agent, learner, decision-maker | Agent boxes, policy, actions |
| **Purple** | Environment, external world | Environment boxes, states, dynamics |
| **Emerald** | Actions, outputs, forward flow | Action arrows, outputs |
| **Red** | Limitations, warnings, problems | Disadvantages, pitfalls, things to avoid |
| **Slate** | Neutral, supporting | Backgrounds, secondary text, containers |

### General Technical Content (for ML Concepts, reference chapters)

For non-RL content like the Quantization chapter, colors are used to distinguish concepts rather than carry RL-specific meaning:

| Color | Use For |
|-------|---------|
| **Blue** | Primary/baseline option (e.g., `float16`) |
| **Amber** | Alternative option (e.g., `bfloat16`, `int4`) |
| **Emerald** | Practical/common choice (e.g., `int8`) |
| **Violet** | Advanced/specialized option |
| **Slate** | Neutral baseline (e.g., `float32`) |

The key is **consistency within a chapter**—pick a color for each concept and stick with it.

---

## Pattern: Comparison Cards

Use instead of markdown tables when comparing 2-4 distinct concepts.

### When to Use
- Comparing learning paradigms (supervised vs unsupervised vs RL)
- Comparing algorithm categories
- Comparing approaches with distinct trade-offs

### Template

```jsx
<div className="grid md:grid-cols-3 gap-4 my-8">
  {/* Card 1 */}
  <div className="bg-gradient-to-br from-cyan-900/30 to-cyan-800/10 border border-cyan-700/50 rounded-xl p-5">
    <div className="flex items-center gap-3 mb-3">
      <div className="text-2xl">🏷️</div>
      <div className="text-cyan-400 font-bold">Title</div>
    </div>
    <div className="text-slate-300 text-sm mb-3">
      Description with <span className="text-cyan-300 font-medium">highlighted term</span>
    </div>
    <div className="text-slate-400 text-xs bg-slate-800/50 rounded-lg p-3">
      "Quote or example in context"
    </div>
  </div>

  {/* Card 2 - Highlighted (add ring for emphasis) */}
  <div className="bg-gradient-to-br from-amber-900/30 to-amber-800/10 border-2 border-amber-500/70 rounded-xl p-5 ring-2 ring-amber-500/20">
    {/* Same structure */}
  </div>
</div>
```

### Key Patterns
- Gradient backgrounds: `from-{color}-900/30 to-{color}-800/10`
- Borders: `border border-{color}-700/50`
- Highlight with ring: `ring-2 ring-{color}-500/20`
- Quote boxes: `bg-slate-800/50 rounded-lg p-3`

---

## Pattern: Inline Diagrams

Use for conceptual diagrams like the agent-environment loop.

### When to Use
- Showing relationships between concepts
- Flow diagrams
- Simple state diagrams

### Template: Two-Box with Arrows

```jsx
<div className="my-8 flex justify-center">
  <div className="bg-slate-800/50 rounded-xl p-6 border border-slate-700 font-mono text-sm">
    <div className="flex items-center justify-center gap-8">
      {/* Left Box */}
      <div className="bg-blue-600/20 border-2 border-blue-500 rounded-lg px-6 py-4 text-center">
        <div className="text-blue-400 font-bold text-lg">Agent</div>
        <div className="text-slate-400 text-xs mt-1">Subtitle</div>
      </div>

      {/* Arrows */}
      <div className="flex flex-col gap-4">
        <div className="flex items-center gap-2">
          <span className="text-emerald-400">action</span>
          <span className="text-slate-500">→</span>
        </div>
        <div className="flex items-center gap-2">
          <span className="text-slate-500">←</span>
          <span className="text-amber-400">state, reward</span>
        </div>
      </div>

      {/* Right Box */}
      <div className="bg-purple-600/20 border-2 border-purple-500 rounded-lg px-6 py-4 text-center">
        <div className="text-purple-400 font-bold text-lg">Environment</div>
        <div className="text-slate-400 text-xs mt-1">Subtitle</div>
      </div>
    </div>

    {/* Caption */}
    <div className="text-center mt-4 text-slate-500 text-xs">
      ↻ Caption text
    </div>
  </div>
</div>
```

### Key Patterns
- Container: `bg-slate-800/50 rounded-xl p-6 border border-slate-700`
- Boxes: `bg-{color}-600/20 border-2 border-{color}-500 rounded-lg`
- Use `font-mono` for technical diagrams
- Add `mt-8` after diagrams before prose text

---

## Pattern: Limitation/Problem Cards

Use for listing disadvantages, limitations, or problems with an approach.

### When to Use
- Explaining why an approach doesn't work
- Listing trade-offs or downsides
- Warning about pitfalls

### Template

```jsx
<div className="my-6 space-y-3">
  <div className="flex items-start gap-3 bg-red-900/20 border border-red-800/40 rounded-lg p-4">
    <div className="text-red-400 font-bold text-lg">1.</div>
    <div>
      <div className="text-red-300 font-medium">Problem title</div>
      <div className="text-slate-400 text-sm">Explanation of the problem.</div>
    </div>
  </div>

  <div className="flex items-start gap-3 bg-red-900/20 border border-red-800/40 rounded-lg p-4">
    <div className="text-red-400 font-bold text-lg">2.</div>
    <div>
      <div className="text-red-300 font-medium">Another problem</div>
      <div className="text-slate-400 text-sm">More details here.</div>
    </div>
  </div>
</div>
```

---

## Pattern: Advantage/Success Cards

For listing benefits or advantages (inverse of limitation cards).

### Template

```jsx
<div className="my-6 space-y-3">
  <div className="flex items-start gap-3 bg-emerald-900/20 border border-emerald-800/40 rounded-lg p-4">
    <div className="text-emerald-400 font-bold text-lg">✓</div>
    <div>
      <div className="text-emerald-300 font-medium">Benefit title</div>
      <div className="text-slate-400 text-sm">Why this is good.</div>
    </div>
  </div>
</div>
```

---

## Using Callout Components

The existing callout components should be used liberally:

| Component | When to Use |
|-----------|-------------|
| `<Example>` | Real-world examples, analogies, relatable scenarios |
| `<Definition>` | Formal definitions of key terms |
| `<Note>` | Important context, caveats, additional information |
| `<Tip>` | Practical advice, shortcuts, best practices |
| `<Warning>` | Common mistakes, pitfalls, things to avoid |

### Example Usage

```mdx
<Example title="RL in Everyday Life">
Every time you teach a dog a trick, you're doing reinforcement learning.
</Example>

<Definition title="Reinforcement Learning">
**Reinforcement learning** is the science of learning through trial and error...
</Definition>
```

---

## Spacing Guidelines

- Add `my-8` around major visual elements (diagrams, card grids)
- Use `mt-8` before prose that follows a visual element
- Use `space-y-3` or `space-y-4` for vertical card lists
- Use `gap-4` for grid layouts

---

## Pattern: Horizontal Bar Charts

Use for comparing relative values or speedups. **Important**: Use inline styles for percentage widths as Tailwind JIT may not compile arbitrary values like `w-[75%]`.

### When to Use
- Comparing speedups or performance metrics
- Showing relative sizes (memory, time, etc.)
- Visualizing ratios

### Template

```jsx
<div className="bg-slate-800/50 rounded-xl p-5 border border-slate-700 my-6">
  <div className="text-slate-200 font-medium mb-4">Chart Title</div>
  <div className="space-y-3">
    {/* Each bar */}
    <div className="flex items-center gap-3">
      <span className="text-xs text-slate-400 w-16 font-mono shrink-0">label</span>
      <div className="flex-1">
        <div style={{width: '50%', backgroundColor: '#3b82f6'}} className="h-6 rounded flex items-center justify-end pr-2">
          <span className="text-xs text-white font-semibold">1.0×</span>
        </div>
      </div>
    </div>

    <div className="flex items-center gap-3">
      <span className="text-xs text-slate-400 w-16 font-mono shrink-0">label2</span>
      <div className="flex-1">
        <div style={{width: '75%', backgroundColor: '#f59e0b'}} className="h-6 rounded flex items-center justify-end pr-2">
          <span className="text-xs text-white font-semibold">1.5×</span>
        </div>
      </div>
    </div>
  </div>
</div>
```

### Key Patterns
- **Use inline `style` for widths**: `style={{width: '75%'}}` instead of `w-[75%]`
- **Use hex colors for backgrounds**: `backgroundColor: '#3b82f6'` instead of `bg-blue-500`
- **`shrink-0` on labels**: Prevents labels from compressing
- **`flex-1` wrapper**: Ensures bars align regardless of label width
- **Color values**: Blue `#3b82f6`, Amber `#f59e0b`, Emerald `#10b981`, Violet `#8b5cf6`

---

## Pattern: Numbered Takeaway Cards

Use for chapter summaries or key points lists.

### When to Use
- Chapter summary sections
- Key takeaways at end of content
- Numbered learning objectives

### Template

```jsx
<div className="space-y-4 my-8">
  <div className="flex gap-4 p-4 bg-slate-800/50 rounded-lg border border-slate-700">
    <div className="text-2xl">1</div>
    <div>
      <div className="font-bold text-slate-200 mb-1">Takeaway title</div>
      <div className="text-slate-400 text-sm">Explanation of the key point with <code className="text-blue-300">highlighted code</code> terms.</div>
    </div>
  </div>

  <div className="flex gap-4 p-4 bg-slate-800/50 rounded-lg border border-slate-700">
    <div className="text-2xl">2</div>
    <div>
      <div className="font-bold text-slate-200 mb-1">Second takeaway</div>
      <div className="text-slate-400 text-sm">More details here.</div>
    </div>
  </div>
</div>
```

### Key Patterns
- Large number as visual anchor (no emoji needed)
- Consistent card styling with `bg-slate-800/50`
- Use `<code className="text-{color}-300">` for inline technical terms

---

## Pattern: Quiz with Expandable Answers

Use for self-assessment questions with hidden answers.

### When to Use
- Chapter quizzes
- Concept checks
- Self-assessment sections

### Template

```jsx
<div className="space-y-6 my-8">
  <div className="p-5 bg-slate-800/30 rounded-xl border border-slate-700">
    <div className="font-bold text-slate-200 mb-3">1. Question text with <code className="text-amber-300">code terms</code>?</div>
    <details className="mt-3">
      <summary className="cursor-pointer text-cyan-400 hover:text-cyan-300">Show answer</summary>
      <div className="mt-3 p-3 bg-slate-900/50 rounded-lg text-slate-300">
        Answer explanation here. Use <strong>bold</strong> for emphasis and <code>code</code> for technical terms.
      </div>
    </details>
  </div>

  <div className="p-5 bg-slate-800/30 rounded-xl border border-slate-700">
    <div className="font-bold text-slate-200 mb-3">2. Another question?</div>
    <details className="mt-3">
      <summary className="cursor-pointer text-cyan-400 hover:text-cyan-300">Show answer</summary>
      <div className="mt-3 p-3 bg-slate-900/50 rounded-lg text-slate-300">
        Another answer here.
      </div>
    </details>
  </div>
</div>
```

### Key Patterns
- `<details>/<summary>` for native expandable behavior
- Cyan color for interactive "Show answer" link
- Slightly darker background (`bg-slate-900/50`) for answer container
- Number questions for easy reference

---

## Pattern: Colab/External Link Buttons

Use for linking to external resources like Colab notebooks.

### When to Use
- Linking to Colab notebooks
- External documentation
- Downloadable resources

### Template

```jsx
<div className="flex flex-wrap gap-3 my-6">
  <a href="https://colab.research.google.com/..." target="_blank" rel="noopener noreferrer"
     className="inline-flex items-center gap-2 px-4 py-3 bg-gradient-to-r from-amber-600 to-orange-600 text-white rounded-lg hover:from-amber-500 hover:to-orange-500 transition-all">
    <span className="text-xl">🚀</span>
    <div>
      <div className="font-semibold">Open in Colab</div>
      <div className="text-xs text-amber-100">Includes solutions</div>
    </div>
  </a>
</div>
```

### Color Variants
- **Colab (exercises)**: `from-amber-600 to-orange-600` with 🚀
- **GitHub**: `from-slate-600 to-slate-700` with GitHub icon
- **Documentation**: `from-blue-600 to-blue-700` with 📖

---

## Pattern: Exercise/Feature Preview Cards

Use for previewing exercises or features in a grid.

### When to Use
- Exercise overview before Colab link
- Feature lists
- Module previews

### Template

```jsx
<div className="grid md:grid-cols-2 gap-4 my-6">
  <div className="p-4 bg-slate-800/30 rounded-lg border border-slate-700">
    <div className="font-bold text-slate-200 mb-2">Exercise 1: Title</div>
    <div className="text-sm text-slate-400">Brief description of what this exercise covers.</div>
  </div>

  <div className="p-4 bg-slate-800/30 rounded-lg border border-slate-700">
    <div className="font-bold text-slate-200 mb-2">Exercise 2: Title</div>
    <div className="text-sm text-slate-400">Another exercise description.</div>
  </div>
</div>
```

### Key Patterns
- Use `md:grid-cols-2` for 2-column layout on medium+ screens
- Lighter background (`bg-slate-800/30`) than full cards
- Keep descriptions brief (1-2 sentences)

---

## Pattern: Stat Cards

Use for displaying key metrics or specifications at a glance.

### When to Use
- System specifications (building size, parameters, etc.)
- Key metrics overview
- Episode parameters
- Quick reference stats

### Template

```jsx
<div className="grid grid-cols-2 md:grid-cols-4 gap-4 my-4">
  <div className="text-center p-3 bg-gray-800 rounded">
    <div className="text-3xl font-bold text-blue-400">10</div>
    <div className="text-xs text-gray-400">Floors</div>
  </div>
  <div className="text-center p-3 bg-gray-800 rounded">
    <div className="text-3xl font-bold text-blue-400">3</div>
    <div className="text-xs text-gray-400">Elevators</div>
  </div>
  <div className="text-center p-3 bg-gray-800 rounded">
    <div className="text-3xl font-bold text-blue-400">8</div>
    <div className="text-xs text-gray-400">Capacity</div>
  </div>
  <div className="text-center p-3 bg-gray-800 rounded">
    <div className="text-3xl font-bold text-blue-400">10m</div>
    <div className="text-xs text-gray-400">Episode</div>
  </div>
</div>
```

### Key Patterns
- Large bold number (3xl) as focal point
- Small gray label below
- Grid layout (2 or 4 columns)
- Centered text alignment
- Use for quick-scan information

---

## Pattern: Performance Comparison Bars with Emoji

Use for visually comparing algorithm performance with clear visual hierarchy.

### When to Use
- Comparing algorithm results
- Showing performance rankings
- Benchmark comparisons
- Before/after metrics

### Template

```jsx
<div className="my-6 space-y-3">
  {/* Baseline - Red (worst) */}
  <div className="flex items-center gap-4 p-4 bg-red-900/10 border border-red-800 rounded">
    <div className="text-3xl">🎲</div>
    <div className="flex-1">
      <div className="font-bold text-red-400">Random</div>
      <div className="text-xs text-gray-400">Baseline - terrible</div>
    </div>
    <div className="text-right">
      <div className="text-2xl font-bold text-red-300">45.2s</div>
      <div className="text-xs text-gray-400">87 served</div>
    </div>
  </div>

  {/* Winner - Green with thicker border */}
  <div className="flex items-center gap-4 p-4 bg-green-900/20 border-2 border-green-500 rounded">
    <div className="text-3xl">🧠</div>
    <div className="flex-1">
      <div className="font-bold text-green-400">DQN (Trained RL)</div>
      <div className="text-xs text-green-300">26% better than SCAN!</div>
    </div>
    <div className="text-right">
      <div className="text-3xl font-bold text-green-400">12.4s</div>
      <div className="text-xs text-gray-300">156 served (+5%)</div>
    </div>
  </div>
</div>
```

### Key Patterns
- Emoji on left for visual scanning
- Color-coded by performance (red=bad, yellow=ok, blue=good, green=best)
- Winner gets `border-2` instead of `border`
- Metric on right, large and bold
- Secondary metric below in smaller text

---

## Pattern: Border-Left Accent Cards

Use for highlighting distinct items in a list with colored accents.

### When to Use
- Reward components
- Feature lists
- Emergent behaviors
- Multi-part explanations

### Template

```jsx
<div className="space-y-3 my-4">
  <div className="p-4 bg-red-900/20 border-l-4 border-red-500 rounded">
    <div className="font-bold text-red-300">Wait Time Penalty</div>
    <div className="text-sm mt-1">-1.0 per waiting passenger per timestep</div>
    <div className="text-xs text-gray-400 mt-1">Primary goal: minimize wait</div>
  </div>

  <div className="p-4 bg-green-900/20 border-l-4 border-green-500 rounded">
    <div className="font-bold text-green-300">Delivery Bonus</div>
    <div className="text-sm mt-1">+10.0 per passenger delivered</div>
    <div className="text-xs text-gray-400 mt-1">Reward completion, not just attempts</div>
  </div>
</div>
```

### Key Patterns
- `border-l-4` creates thick left accent
- Background color matches border (with lower opacity)
- Three-tier text hierarchy: bold title, description, explanation
- Use different colors for different categories

---

## Pattern: Challenge-Solution Cards

Use for problem/solution pairs with clear visual structure.

### When to Use
- Common challenges and how to address them
- Troubleshooting guides
- FAQ-style content
- Debugging workflows

### Template

```jsx
<div className="space-y-4 my-6">
  <div className="border-l-4 border-red-500 bg-gray-800 p-4 rounded">
    <div className="font-bold text-red-400 mb-2">❌ Challenge: Slow Initial Learning</div>
    <div className="text-sm text-gray-300 mb-2">With ε=1.0, early episodes are random walks → very negative rewards → slow buffer fill</div>
    <div className="text-sm text-green-400">✓ <strong>Solution:</strong></div>
    <ul className="text-sm text-gray-300 ml-4 mt-1 space-y-1">
      <li>• Pre-fill buffer with nearest-car policy (100 episodes)</li>
      <li>• Or use shaped rewards (bonus for approaching requests)</li>
    </ul>
  </div>
</div>
```

### Key Patterns
- Red border-left for challenge
- ❌ emoji before "Challenge"
- Green checkmark (✓) before "Solution"
- Problem statement, then bulleted solutions
- Solid `bg-gray-800` background (not transparent)

---

## Pattern: Extension/Future Work Cards

Use for listing potential extensions or advanced topics.

### When to Use
- Extension ideas
- Future research directions
- Advanced variations
- "What's next" sections

### Template

```jsx
<div className="space-y-3 my-4">
  <div className="p-4 bg-gray-800 rounded border border-blue-600">
    <div className="font-bold text-blue-400 mb-1">🏢 Larger Buildings (50 floors × 10 elevators)</div>
    <div className="text-sm text-gray-300">Observation dimension explodes → use **graph neural networks** (GNNs) to encode elevator-floor relationships. Consider CTDE methods like QMIX.</div>
  </div>

  <div className="p-4 bg-gray-800 rounded border border-purple-600">
    <div className="font-bold text-purple-400 mb-1">⚡ Express Elevators</div>
    <div className="text-sm text-gray-300">Some elevators skip floors (1, 10, 20...). RL learns when to use express vs local based on destinations—better than fixed rules.</div>
  </div>
</div>
```

### Key Patterns
- Colored border (not border-left, full border)
- Simple structure: emoji + title, then description
- Solid background `bg-gray-800`
- Different colors distinguish different extensions
- Keep descriptions concise (1-2 sentences)

---

## Pattern: Gradient Numbered Takeaway Cards

Use for key takeaways with enhanced visual appeal (alternative to plain numbered takeaways).

### When to Use
- Chapter summaries
- Key insights that deserve emphasis
- Major conclusions
- Learning objectives review

### Template

```jsx
<div className="space-y-3 my-6">
  <div className="flex gap-4 p-4 bg-gradient-to-r from-blue-900 to-blue-800 rounded-lg">
    <div className="text-3xl font-bold text-blue-300">1</div>
    <div>
      <div className="font-bold text-blue-200">RL shines for coordination problems</div>
      <div className="text-sm text-gray-300">When multiple agents must work together without explicit communication, RL discovers emergent coordination.</div>
    </div>
  </div>

  <div className="flex gap-4 p-4 bg-gradient-to-r from-purple-900 to-purple-800 rounded-lg">
    <div className="text-3xl font-bold text-purple-300">2</div>
    <div>
      <div className="font-bold text-purple-200">Reward engineering is critical</div>
      <div className="text-sm text-gray-300">Small changes drastically affect learned behavior.</div>
    </div>
  </div>
</div>
```

### Key Patterns
- Gradient backgrounds: `bg-gradient-to-r from-{color}-900 to-{color}-800`
- Large number (3xl) on left
- Different color per item for visual distinction
- Title color: `text-{color}-200`, number: `text-{color}-300`
- More visually striking than plain numbered cards—use for major takeaways

---

## Accessibility Notes

- Always include text labels (don't rely solely on color)
- Use semantic HTML where possible
- Ensure sufficient contrast (the dark theme patterns above are tested)
- Emojis in card headers provide visual anchors but content should work without them
