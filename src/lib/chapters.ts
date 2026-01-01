/**
 * Chapter data definitions
 *
 * This is the single source of truth for chapter metadata.
 *
 * Structure:
 * - Level 1: Section (e.g., "Foundations", "Markov Decision Processes")
 * - Level 2: Chapter (e.g., "Introduction to RL", "Multi-Armed Bandits")
 * - Level 3: Subsection/Topic (e.g., "What is RL?", "The Agent-Environment Interface")
 *
 * Directory numbering scheme:
 * - 00XX: Foundations
 * - 01XX: Markov Decision Processes
 * - 02XX: Dynamic Programming
 * - 10XX: Bandit Problems
 * - 11XX: Temporal Difference Learning
 * - 12XX: Deep Reinforcement Learning
 * - 20XX: Policy Gradient Methods
 * - 30XX: Advanced Topics
 * - 40XX: ML Concepts (reference material for ML fundamentals used in RL)
 */

export type ChapterStatus = 'draft' | 'editor_reviewed' | 'community_reviewed' | 'verified';

export interface SubsectionData {
  slug: string;           // e.g., "what-is-rl"
  title: string;          // e.g., "What is Reinforcement Learning?"
  description: string;
  order: number;          // For sorting (10, 20, 30...)
}

export interface ChapterData {
  slug: string;           // e.g., "intro-to-rl"
  dirName: string;        // e.g., "0010-intro-to-rl"
  title: string;
  section: string;
  description: string;
  status: ChapterStatus;
  subsections?: SubsectionData[];
}

// Full URL path for a subsection: /chapters/{chapterSlug}/{subsectionSlug}
export interface NavigationItem {
  type: 'chapter' | 'subsection';
  chapterSlug: string;
  subsectionSlug?: string;
  title: string;
  section: string;
  fullPath: string;
}

export const chapters: Record<string, ChapterData> = {
  // ============================================================================
  // PART I: FOUNDATIONS (Complete - editor_reviewed)
  // ============================================================================

  'intro-to-rl': {
    slug: 'intro-to-rl',
    dirName: '0010-intro-to-rl',
    title: 'What is Reinforcement Learning?',
    section: 'Foundations',
    description: 'The big picture: what RL is, where it came from, and where you see it today',
    status: 'editor_reviewed',
    subsections: [
      {
        slug: 'what-is-rl',
        title: 'The Core Idea',
        description: 'Learning from interaction: the essence of RL',
        order: 10,
      },
      {
        slug: 'rl-in-the-wild',
        title: 'RL in the Wild',
        description: 'Real-world examples from everyday life to LLMs',
        order: 20,
      },
      {
        slug: 'rl-history',
        title: 'A Brief History',
        description: 'From Bellman to ChatGPT: the milestones that shaped RL',
        order: 30,
      },
    ],
  },

  'rl-framework': {
    slug: 'rl-framework',
    dirName: '0011-rl-framework',
    title: 'The RL Framework',
    section: 'Foundations',
    description: 'The building blocks: agents, environments, states, actions, rewards, and policies',
    status: 'editor_reviewed',
    subsections: [
      {
        slug: 'agent-environment',
        title: 'The Agent-Environment Interface',
        description: 'States, actions, and the interaction loop',
        order: 10,
      },
      {
        slug: 'rewards-returns',
        title: 'Rewards and Returns',
        description: 'Defining goals through reward signals',
        order: 20,
      },
      {
        slug: 'policies-values',
        title: 'Policies and Value Functions',
        description: 'How agents represent knowledge',
        order: 30,
      },
      {
        slug: 'exploration-exploitation',
        title: 'Exploration vs Exploitation',
        description: 'The fundamental tradeoff at the heart of RL',
        order: 40,
      },
    ],
  },

  'getting-started': {
    slug: 'getting-started',
    dirName: '0012-getting-started',
    title: 'Getting Started',
    section: 'Foundations',
    description: 'The algorithm landscape, your roadmap, and your first hands-on demo',
    status: 'editor_reviewed',
    subsections: [
      {
        slug: 'rl-landscape',
        title: 'The RL Landscape',
        description: 'Model-free vs model-based, value vs policy methods',
        order: 10,
      },
      {
        slug: 'try-it-yourself',
        title: 'Try It Yourself',
        description: 'Experience the RL loop with an interactive GridWorld demo',
        order: 20,
      },
    ],
  },

  // ============================================================================
  // PART II: MARKOV DECISION PROCESSES (Mathematical Foundation)
  // ============================================================================

  'intro-to-mdps': {
    slug: 'intro-to-mdps',
    dirName: '0110-intro-to-mdps',
    title: 'Introduction to MDPs',
    section: 'Markov Decision Processes',
    description: 'The mathematical framework that formalizes sequential decision-making',
    status: 'draft',
    subsections: [
      {
        slug: 'from-bandits-to-mdps',
        title: 'From Bandits to Sequential Decisions',
        description: 'Why we need states: when actions have lasting consequences',
        order: 10,
      },
      {
        slug: 'mdp-components',
        title: 'The MDP Components',
        description: 'States, actions, transitions, rewards, and discount factors',
        order: 20,
      },
      {
        slug: 'markov-property',
        title: 'The Markov Property',
        description: 'Why the present is all you need to predict the future',
        order: 30,
      },
    ],
  },

  'value-functions': {
    slug: 'value-functions',
    dirName: '0120-value-functions',
    title: 'Value Functions',
    section: 'Markov Decision Processes',
    description: 'Measuring how good states and actions are',
    status: 'draft',
    subsections: [
      {
        slug: 'state-values',
        title: 'State Value Functions',
        description: 'How good is it to be in a state?',
        order: 10,
      },
      {
        slug: 'action-values',
        title: 'Action Value Functions',
        description: 'How good is it to take an action in a state?',
        order: 20,
      },
      {
        slug: 'optimal-values',
        title: 'Optimal Value Functions',
        description: 'The best possible values and what they tell us',
        order: 30,
      },
    ],
  },

  'bellman-equations': {
    slug: 'bellman-equations',
    dirName: '0130-bellman-equations',
    title: 'The Bellman Equations',
    section: 'Markov Decision Processes',
    description: 'The recursive equations that make RL possible',
    status: 'draft',
    subsections: [
      {
        slug: 'bellman-expectation',
        title: 'Bellman Expectation Equations',
        description: 'How values relate to each other under a policy',
        order: 10,
      },
      {
        slug: 'bellman-optimality',
        title: 'Bellman Optimality Equations',
        description: 'The equations that define optimal behavior',
        order: 20,
      },
      {
        slug: 'why-bellman-matters',
        title: 'Why Bellman Matters',
        description: 'Bootstrapping: the key insight behind all RL algorithms',
        order: 30,
      },
    ],
  },

  // ============================================================================
  // PART III: DYNAMIC PROGRAMMING (Planning with Known Models)
  // ============================================================================

  'policy-evaluation': {
    slug: 'policy-evaluation',
    dirName: '0210-policy-evaluation',
    title: 'Policy Evaluation',
    section: 'Dynamic Programming',
    description: 'Computing the value of a policy when you know the model',
    status: 'draft',
    subsections: [
      {
        slug: 'iterative-evaluation',
        title: 'Iterative Policy Evaluation',
        description: 'Repeatedly applying Bellman until convergence',
        order: 10,
      },
      {
        slug: 'convergence',
        title: 'Convergence and Stopping',
        description: 'How do we know when we\'re done?',
        order: 20,
      },
    ],
  },

  'policy-improvement': {
    slug: 'policy-improvement',
    dirName: '0220-policy-improvement',
    title: 'Policy Improvement',
    section: 'Dynamic Programming',
    description: 'Finding better policies through value functions',
    status: 'draft',
    subsections: [
      {
        slug: 'improvement-theorem',
        title: 'The Policy Improvement Theorem',
        description: 'Why greedy improvement always works',
        order: 10,
      },
      {
        slug: 'policy-iteration',
        title: 'Policy Iteration',
        description: 'Alternating evaluation and improvement until optimal',
        order: 20,
      },
      {
        slug: 'value-iteration',
        title: 'Value Iteration',
        description: 'Finding optimal values directly',
        order: 30,
      },
    ],
  },

  // ============================================================================
  // PART IV: BANDIT PROBLEMS (Exploration Without States)
  // ============================================================================

  'multi-armed-bandits': {
    slug: 'multi-armed-bandits',
    dirName: '1010-multi-armed-bandits',
    title: 'Multi-Armed Bandits',
    section: 'Bandit Problems',
    description: 'Master the exploration-exploitation tradeoff in the simplest RL setting',
    status: 'draft',
    subsections: [
      {
        slug: 'bandit-problem',
        title: 'The Bandit Problem',
        description: 'Slot machines and the exploration dilemma',
        order: 10,
      },
      {
        slug: 'greedy-methods',
        title: 'Greedy and ε-Greedy Methods',
        description: 'Simple but powerful exploration strategies',
        order: 20,
      },
      {
        slug: 'ucb',
        title: 'Upper Confidence Bound',
        description: 'Optimism in the face of uncertainty',
        order: 30,
      },
      {
        slug: 'thompson-sampling',
        title: 'Thompson Sampling',
        description: 'The Bayesian approach to exploration',
        order: 40,
      },
      {
        slug: 'comparing-strategies',
        title: 'Comparing Strategies',
        description: 'When to use which exploration method',
        order: 50,
      },
    ],
  },

  'contextual-bandits': {
    slug: 'contextual-bandits',
    dirName: '1020-contextual-bandits',
    title: 'Contextual Bandits',
    section: 'Bandit Problems',
    description: 'Personalized decisions based on context features',
    status: 'draft',
    subsections: [
      {
        slug: 'context-matters',
        title: 'Why Context Matters',
        description: 'From bandits to personalized decisions',
        order: 10,
      },
      {
        slug: 'linucb',
        title: 'Linear UCB',
        description: 'Contextual exploration with linear models',
        order: 20,
      },
      {
        slug: 'applications',
        title: 'Real-World Applications',
        description: 'Recommendations, ads, and clinical trials',
        order: 30,
      },
    ],
  },

  // ============================================================================
  // PART V: TEMPORAL DIFFERENCE LEARNING (Learning from Experience)
  // ============================================================================

  'intro-to-td': {
    slug: 'intro-to-td',
    dirName: '1110-intro-to-td',
    title: 'Introduction to TD Learning',
    section: 'Temporal Difference Learning',
    description: 'Learning from experience without waiting for the episode to end',
    status: 'draft',
    subsections: [
      {
        slug: 'td-idea',
        title: 'The TD Idea',
        description: 'Bootstrapping: learning from incomplete returns',
        order: 10,
      },
      {
        slug: 'td-zero',
        title: 'TD(0) Prediction',
        description: 'The simplest TD method for value estimation',
        order: 20,
      },
      {
        slug: 'td-vs-mc',
        title: 'TD vs Monte Carlo',
        description: 'Bias, variance, and sample efficiency',
        order: 30,
      },
    ],
  },

  'sarsa': {
    slug: 'sarsa',
    dirName: '1120-sarsa',
    title: 'SARSA',
    section: 'Temporal Difference Learning',
    description: 'On-policy TD control: learning while following your current policy',
    status: 'draft',
    subsections: [
      {
        slug: 'from-prediction-to-control',
        title: 'From Prediction to Control',
        description: 'Using TD for action-value estimation',
        order: 10,
      },
      {
        slug: 'sarsa-algorithm',
        title: 'The SARSA Algorithm',
        description: 'State-Action-Reward-State-Action learning',
        order: 20,
      },
      {
        slug: 'on-policy-behavior',
        title: 'On-Policy Behavior',
        description: 'Why SARSA is safe but sometimes suboptimal',
        order: 30,
      },
    ],
  },

  'q-learning': {
    slug: 'q-learning',
    dirName: '1130-q-learning',
    title: 'Q-Learning',
    section: 'Temporal Difference Learning',
    description: 'Off-policy TD control: learning the optimal policy while exploring',
    status: 'draft',
    subsections: [
      {
        slug: 'q-learning-idea',
        title: 'The Q-Learning Idea',
        description: 'Learning optimal values regardless of behavior',
        order: 10,
      },
      {
        slug: 'q-learning-algorithm',
        title: 'The Q-Learning Algorithm',
        description: 'The most famous RL algorithm',
        order: 20,
      },
      {
        slug: 'sarsa-vs-q-learning',
        title: 'SARSA vs Q-Learning',
        description: 'The CliffWalking experiment',
        order: 30,
      },
      {
        slug: 'convergence',
        title: 'Convergence and the Deadly Triad',
        description: 'When Q-learning works and when it breaks',
        order: 40,
      },
    ],
  },

  // ============================================================================
  // PART VI: DEEP REINFORCEMENT LEARNING (Function Approximation)
  // ============================================================================

  'function-approximation': {
    slug: 'function-approximation',
    dirName: '1210-function-approximation',
    title: 'Function Approximation',
    section: 'Deep Reinforcement Learning',
    description: 'Scaling RL to large state spaces with learned representations',
    status: 'draft',
    subsections: [
      {
        slug: 'why-tables-fail',
        title: 'Why Tables Fail',
        description: 'The curse of dimensionality in RL',
        order: 10,
      },
      {
        slug: 'linear-approximation',
        title: 'Linear Function Approximation',
        description: 'Features, weights, and gradient descent',
        order: 20,
      },
      {
        slug: 'neural-networks',
        title: 'Neural Network Approximators',
        description: 'Deep learning meets reinforcement learning',
        order: 30,
      },
    ],
  },

  'dqn': {
    slug: 'dqn',
    dirName: '1220-dqn',
    title: 'Deep Q-Networks',
    section: 'Deep Reinforcement Learning',
    description: 'The breakthrough that made deep RL work',
    status: 'draft',
    subsections: [
      {
        slug: 'dqn-architecture',
        title: 'The DQN Architecture',
        description: 'CNNs for processing visual observations',
        order: 10,
      },
      {
        slug: 'experience-replay',
        title: 'Experience Replay',
        description: 'Breaking correlations through random sampling',
        order: 20,
      },
      {
        slug: 'target-networks',
        title: 'Target Networks',
        description: 'Stabilizing training with frozen targets',
        order: 30,
      },
      {
        slug: 'putting-it-together',
        title: 'Putting It Together',
        description: 'The complete DQN algorithm',
        order: 40,
      },
    ],
  },

  'dqn-improvements': {
    slug: 'dqn-improvements',
    dirName: '1230-dqn-improvements',
    title: 'DQN Improvements',
    section: 'Deep Reinforcement Learning',
    description: 'Enhancements that make DQN even better',
    status: 'draft',
    subsections: [
      {
        slug: 'double-dqn',
        title: 'Double DQN',
        description: 'Fixing overestimation bias',
        order: 10,
      },
      {
        slug: 'prioritized-replay',
        title: 'Prioritized Experience Replay',
        description: 'Learning more from important transitions',
        order: 20,
      },
      {
        slug: 'dueling-networks',
        title: 'Dueling Networks',
        description: 'Separating state value from action advantage',
        order: 30,
      },
      {
        slug: 'rainbow',
        title: 'Rainbow: Combining Improvements',
        description: 'The sum is greater than its parts',
        order: 40,
      },
    ],
  },

  // ============================================================================
  // PART VII: POLICY GRADIENT METHODS (Learning Policies Directly)
  // ============================================================================

  'intro-to-policy-gradients': {
    slug: 'intro-to-policy-gradients',
    dirName: '2010-intro-to-policy-gradients',
    title: 'Introduction to Policy Gradients',
    section: 'Policy Gradient Methods',
    description: 'A fundamentally different approach: learning policies directly',
    status: 'draft',
    subsections: [
      {
        slug: 'why-policies',
        title: 'Why Learn Policies Directly?',
        description: 'Advantages over value-based methods',
        order: 10,
      },
      {
        slug: 'stochastic-policies',
        title: 'Stochastic Policies',
        description: 'Probability distributions over actions',
        order: 20,
      },
      {
        slug: 'policy-objective',
        title: 'The Policy Objective',
        description: 'What we\'re trying to maximize',
        order: 30,
      },
    ],
  },

  'reinforce': {
    slug: 'reinforce',
    dirName: '2020-reinforce',
    title: 'REINFORCE',
    section: 'Policy Gradient Methods',
    description: 'The foundational policy gradient algorithm',
    status: 'draft',
    subsections: [
      {
        slug: 'policy-gradient-theorem',
        title: 'The Policy Gradient Theorem',
        description: 'How to compute gradients for policies',
        order: 10,
      },
      {
        slug: 'reinforce-algorithm',
        title: 'The REINFORCE Algorithm',
        description: 'Monte Carlo policy gradients',
        order: 20,
      },
      {
        slug: 'variance-problem',
        title: 'The Variance Problem',
        description: 'Why REINFORCE needs help',
        order: 30,
      },
      {
        slug: 'baselines',
        title: 'Baselines and Variance Reduction',
        description: 'Making gradients more stable',
        order: 40,
      },
    ],
  },

  'actor-critic': {
    slug: 'actor-critic',
    dirName: '2030-actor-critic',
    title: 'Actor-Critic Methods',
    section: 'Policy Gradient Methods',
    description: 'Combining policy and value learning for stability',
    status: 'draft',
    subsections: [
      {
        slug: 'actor-critic-idea',
        title: 'The Actor-Critic Idea',
        description: 'Two networks working together',
        order: 10,
      },
      {
        slug: 'advantage-functions',
        title: 'Advantage Functions',
        description: 'How much better is this action than average?',
        order: 20,
      },
      {
        slug: 'a2c',
        title: 'Advantage Actor-Critic (A2C)',
        description: 'Synchronous actor-critic training',
        order: 30,
      },
      {
        slug: 'gae',
        title: 'Generalized Advantage Estimation',
        description: 'Balancing bias and variance in advantage estimation',
        order: 40,
      },
    ],
  },

  'ppo': {
    slug: 'ppo',
    dirName: '2040-ppo',
    title: 'Proximal Policy Optimization',
    section: 'Policy Gradient Methods',
    description: 'The most popular deep RL algorithm in practice',
    status: 'draft',
    subsections: [
      {
        slug: 'trust-regions',
        title: 'Trust Regions',
        description: 'Why we need to limit policy updates',
        order: 10,
      },
      {
        slug: 'ppo-algorithm',
        title: 'The PPO Algorithm',
        description: 'Clipped surrogate objectives',
        order: 20,
      },
      {
        slug: 'why-ppo-works',
        title: 'Why PPO Works',
        description: 'Simplicity, stability, and performance',
        order: 30,
      },
      {
        slug: 'ppo-in-practice',
        title: 'PPO in Practice',
        description: 'Hyperparameters and implementation tips',
        order: 40,
      },
    ],
  },

  // ============================================================================
  // PART VIII: ADVANCED TOPICS (Cutting Edge)
  // ============================================================================

  'model-based-rl': {
    slug: 'model-based-rl',
    dirName: '3010-model-based-rl',
    title: 'Model-Based RL',
    section: 'Advanced Topics',
    description: 'Learning world models for sample-efficient planning',
    status: 'draft',
    subsections: [
      {
        slug: 'learning-models',
        title: 'Learning World Models',
        description: 'Predicting transitions and rewards',
        order: 10,
      },
      {
        slug: 'planning',
        title: 'Planning with Learned Models',
        description: 'Using imagination for better decisions',
        order: 20,
      },
      {
        slug: 'dyna',
        title: 'Dyna Architecture',
        description: 'Combining real and simulated experience',
        order: 30,
      },
      {
        slug: 'muzero',
        title: 'MuZero and Beyond',
        description: 'Learning models for planning without rules',
        order: 40,
      },
    ],
  },

  'multi-agent-rl': {
    slug: 'multi-agent-rl',
    dirName: '3020-multi-agent-rl',
    title: 'Multi-Agent RL',
    section: 'Advanced Topics',
    description: 'When multiple agents learn and interact together',
    status: 'draft',
    subsections: [
      {
        slug: 'multi-agent-settings',
        title: 'Multi-Agent Settings',
        description: 'Cooperation, competition, and mixed motives',
        order: 10,
      },
      {
        slug: 'independent-learning',
        title: 'Independent Learning',
        description: 'Each agent learns on its own',
        order: 20,
      },
      {
        slug: 'centralized-training',
        title: 'Centralized Training, Decentralized Execution',
        description: 'Sharing information during training only',
        order: 30,
      },
    ],
  },

  'offline-rl': {
    slug: 'offline-rl',
    dirName: '3030-offline-rl',
    title: 'Offline RL',
    section: 'Advanced Topics',
    description: 'Learning from logged data without environment interaction',
    status: 'draft',
    subsections: [
      {
        slug: 'offline-setting',
        title: 'The Offline Setting',
        description: 'When you can\'t interact with the environment',
        order: 10,
      },
      {
        slug: 'distribution-shift',
        title: 'Distribution Shift',
        description: 'The core challenge of offline RL',
        order: 20,
      },
      {
        slug: 'conservative-methods',
        title: 'Conservative Methods',
        description: 'Staying close to the data',
        order: 30,
      },
    ],
  },

  'rlhf': {
    slug: 'rlhf',
    dirName: '3040-rlhf',
    title: 'RLHF and Language Models',
    section: 'Advanced Topics',
    description: 'How RL powers modern AI systems like ChatGPT',
    status: 'draft',
    subsections: [
      {
        slug: 'rl-for-alignment',
        title: 'RL for AI Alignment',
        description: 'Teaching models what humans want',
        order: 10,
      },
      {
        slug: 'reward-modeling',
        title: 'Reward Modeling',
        description: 'Learning rewards from human preferences',
        order: 20,
      },
      {
        slug: 'ppo-for-llms',
        title: 'PPO for Language Models',
        description: 'Applying policy gradients to text generation',
        order: 30,
      },
      {
        slug: 'frontiers',
        title: 'Current Frontiers',
        description: 'DPO, constitutional AI, and what\'s next',
        order: 40,
      },
    ],
  },

  // ============================================================================
  // ML CONCEPTS (Reference Material for ML Fundamentals)
  // ============================================================================

  'quantization': {
    slug: 'quantization',
    dirName: '4010-quantization',
    title: 'Quantization',
    section: 'ML Concepts',
    description: 'Reducing model size and speeding up inference by using lower-precision numbers',
    status: 'draft',
    subsections: [
      {
        slug: 'why-quantization',
        title: 'Why Quantization Matters',
        description: 'Memory, speed, and the precision tradeoff',
        order: 10,
      },
      {
        slug: 'number-representations',
        title: 'Number Representations',
        description: 'From float32 to int8: how computers store numbers',
        order: 20,
      },
      {
        slug: 'quantization-methods',
        title: 'Quantization Methods',
        description: 'Post-training quantization vs quantization-aware training',
        order: 30,
      },
      {
        slug: 'quantization-in-practice',
        title: 'Quantization in Practice',
        description: 'Tools, techniques, and hands-on examples',
        order: 40,
      },
    ],
  },
};

/**
 * Get all chapter slugs
 */
export function getChapterSlugs(): string[] {
  return Object.keys(chapters);
}

/**
 * Get chapter data by slug
 */
export function getChapter(slug: string): ChapterData | undefined {
  return chapters[slug];
}

/**
 * Get all chapters as an array, sorted by directory name
 */
export function getChaptersArray(): ChapterData[] {
  return Object.values(chapters).sort((a, b) => a.dirName.localeCompare(b.dirName));
}

/**
 * Get chapters grouped by section, maintaining order
 */
export function getChaptersBySection(): Map<string, ChapterData[]> {
  const sorted = getChaptersArray();
  const sections = new Map<string, ChapterData[]>();

  for (const chapter of sorted) {
    const existing = sections.get(chapter.section) || [];
    existing.push(chapter);
    sections.set(chapter.section, existing);
  }

  return sections;
}

/**
 * Get previous and next chapters for navigation
 */
export function getAdjacentChapters(currentSlug: string): { prev?: ChapterData; next?: ChapterData } {
  const sorted = getChaptersArray();
  const currentIndex = sorted.findIndex(c => c.slug === currentSlug);

  if (currentIndex === -1) {
    return {};
  }

  return {
    prev: currentIndex > 0 ? sorted[currentIndex - 1] : undefined,
    next: currentIndex < sorted.length - 1 ? sorted[currentIndex + 1] : undefined,
  };
}

/**
 * Get a subsection by chapter and subsection slug
 */
export function getSubsection(chapterSlug: string, subsectionSlug: string): SubsectionData | undefined {
  const chapter = chapters[chapterSlug];
  if (!chapter?.subsections) return undefined;
  return chapter.subsections.find(s => s.slug === subsectionSlug);
}

/**
 * Get sorted subsections for a chapter
 */
export function getSubsectionsForChapter(chapterSlug: string): SubsectionData[] {
  const chapter = chapters[chapterSlug];
  if (!chapter?.subsections) return [];
  return [...chapter.subsections].sort((a, b) => a.order - b.order);
}

/**
 * Get adjacent subsections for navigation within a chapter
 */
export function getAdjacentSubsections(
  chapterSlug: string,
  subsectionSlug: string
): { prev?: SubsectionData; next?: SubsectionData; prevChapter?: ChapterData; nextChapter?: ChapterData } {
  const subsections = getSubsectionsForChapter(chapterSlug);
  const currentIndex = subsections.findIndex(s => s.slug === subsectionSlug);

  if (currentIndex === -1) {
    return {};
  }

  const result: ReturnType<typeof getAdjacentSubsections> = {};

  if (currentIndex > 0) {
    result.prev = subsections[currentIndex - 1];
  } else {
    // First subsection - link to previous chapter
    const { prev } = getAdjacentChapters(chapterSlug);
    result.prevChapter = prev;
  }

  if (currentIndex < subsections.length - 1) {
    result.next = subsections[currentIndex + 1];
  } else {
    // Last subsection - link to next chapter
    const { next } = getAdjacentChapters(chapterSlug);
    result.nextChapter = next;
  }

  return result;
}

/**
 * Build the full navigation path for a subsection
 */
export function getSubsectionPath(chapterSlug: string, subsectionSlug: string): string {
  return `/chapters/${chapterSlug}/${subsectionSlug}`;
}

/**
 * Get all navigation items (chapters + subsections) as a flat list
 * Useful for generating all static paths
 */
export function getAllNavigationItems(): NavigationItem[] {
  const items: NavigationItem[] = [];
  const sorted = getChaptersArray();

  for (const chapter of sorted) {
    // Add chapter as navigation item
    items.push({
      type: 'chapter',
      chapterSlug: chapter.slug,
      title: chapter.title,
      section: chapter.section,
      fullPath: `/chapters/${chapter.slug}`,
    });

    // Add subsections if they exist
    if (chapter.subsections) {
      const sortedSubs = [...chapter.subsections].sort((a, b) => a.order - b.order);
      for (const sub of sortedSubs) {
        items.push({
          type: 'subsection',
          chapterSlug: chapter.slug,
          subsectionSlug: sub.slug,
          title: sub.title,
          section: chapter.section,
          fullPath: `/chapters/${chapter.slug}/${sub.slug}`,
        });
      }
    }
  }

  return items;
}
