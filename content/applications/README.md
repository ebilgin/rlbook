# Applications

End-to-end guides for formulating real-world problems as reinforcement learning problems.

## Purpose

Each application guide covers:
- Domain introduction and why RL is relevant
- Complete MDP formulation (state, action, reward)
- Practical challenges and solutions
- Algorithm selection and comparison
- Evaluation strategies
- Deployment considerations

## Structure

```
applications/
├── XXXX-application-slug/
│   ├── index.mdx        # Main content
│   ├── prompt.md        # AI generation prompt
│   ├── formulation.md   # Detailed MDP spec
│   ├── code/            # Reference implementation
│   └── assets/          # Domain diagrams
```

## Planned Applications

### Robotics
- [ ] Robotic Manipulation: Grasping and Assembly
- [ ] Locomotion: Learning to Walk
- [ ] Drone Navigation

### Games
- [ ] Game AI for Strategy Games
- [ ] Procedural Content Generation
- [ ] NPC Behavior with RL

### Business & Finance
- [ ] Recommendation Systems as Bandits
- [ ] Dynamic Pricing
- [ ] Inventory Management
- [ ] Algorithmic Trading (Educational)

### Operations
- [ ] Resource Scheduling
- [ ] Network Routing
- [ ] Energy Management

### Language & Agents
- [ ] RLHF for Language Models
- [ ] Dialogue Systems
- [ ] Autonomous Agents

## Key Questions Every Application Answers

1. **What is the problem?** Domain context and objectives
2. **What does the agent observe?** State/observation design
3. **What can the agent do?** Action space design
4. **What is success?** Reward engineering
5. **What makes it hard?** Domain-specific challenges
6. **What works?** Algorithm recommendations
7. **How do we evaluate?** Metrics and baselines
8. **How do we deploy?** Production considerations

## Contributing

See [CONTENT_TYPES.md](../../docs/CONTENT_TYPES.md) for guidelines on writing application guides.
