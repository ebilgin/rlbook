# Interactive Demo Template

Use this template for specifying interactive demonstrations.

---

## Required Context

**Before generating content, read [CLAUDE.md](../../CLAUDE.md)** — specifically the "Core Philosophy" (Interactivity First) and "For AI Assistants" sections.

Also review existing components in `src/components/interactive/` for patterns and reusability.

---

## Demo Metadata

**Demo Name:** [e.g., "Q-Value Heatmap Explorer"]
**Associated Chapter:** [Chapter number and name]
**Concept Demonstrated:** [Primary concept this illustrates]
**Complexity:** [Simple / Medium / Complex]

---

## Learning Goal

**What should users understand after interacting with this demo?**

[Clear statement of the insight this demo delivers]

---

## Demo Specification

### Initial State
[Describe what users see when the demo first loads]

### User Controls

| Control | Type | Range/Options | Default | Effect |
|---------|------|---------------|---------|--------|
| [Name] | [slider/button/dropdown/etc] | [min-max or options] | [default] | [what it changes] |

### Visual Elements

| Element | Represents | Visual Encoding |
|---------|------------|-----------------|
| [e.g., Grid cells] | [States] | [Color = value, border = current] |
| [e.g., Arrows] | [Policy/actions] | [Direction, thickness = probability] |

### Animation/Dynamics
[Describe any automatic animations or transitions]

---

## Guided Exploration

### Suggested Experiments

Provide prompts that guide users to discover key insights:

1. **"Try this:"** [Specific action to take]
   **"Notice that:"** [What they should observe]
   **"This shows:"** [The underlying concept]

2. [Repeat for 2-4 key experiments]

### Edge Cases to Explore
[Interesting parameter settings that reveal important behaviors]

---

## Technical Specification

### State Requirements
```typescript
interface DemoState {
  // Define the state shape
}
```

### Key Functions
```typescript
// Core computation needed
function computeX(params): Result {
  // Describe algorithm
}
```

### Performance Constraints
- Max iterations per frame: [number]
- Target frame rate: [number]
- Memory limit: [if applicable]

### Browser Requirements
- [List any specific APIs needed: WebGL, WebGPU, etc.]

---

## Accessibility

### Keyboard Navigation
[How users can interact without mouse]

### Screen Reader
[What should be announced]

### Color Alternatives
[How to convey information without relying solely on color]

---

## Fallback Behavior

### If JavaScript Disabled
[Static alternative to show]

### If Performance Issues
[Simplified version or degradation strategy]

---

## Integration Notes

### Data Flow
[How this demo connects to the chapter content]

### Cross-References
[Other demos that relate to this one]

### Reusability
[Can components be reused? How?]
