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

## Accessibility Notes

- Always include text labels (don't rely solely on color)
- Use semantic HTML where possible
- Ensure sufficient contrast (the dark theme patterns above are tested)
- Emojis in card headers provide visual anchors but content should work without them
