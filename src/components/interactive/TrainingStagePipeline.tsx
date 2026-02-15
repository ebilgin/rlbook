/**
 * Training Stage Pipeline
 *
 * Interactive visualization of how LLM training stages
 * (Pretraining → SFT → RL) transform model responses.
 * Users can click each stage to see example responses and metrics.
 */

import React, { useState } from 'react';

interface Stage {
  id: string;
  label: string;
  shortLabel: string;
  color: string;
  borderColor: string;
  bgGradient: string;
  prompt: string;
  response: string;
  metrics: { helpfulness: number; safety: number; reasoning: number };
  description: string;
}

const stages: Stage[] = [
  {
    id: 'pretrain',
    label: 'Pretraining',
    shortLabel: 'Pretrain',
    color: 'text-slate-300',
    borderColor: 'border-slate-500',
    bgGradient: 'from-slate-800/60 to-slate-800/20',
    prompt: 'What is the capital of France?',
    response:
      'The capital of France is Paris, which is also the largest city in the country. Paris is known for its iconic landmarks such as the Eiffel Tower. The capital of Germany is Berlin. The capital of Italy is Rome. The capital of Spain is',
    metrics: { helpfulness: 25, safety: 20, reasoning: 15 },
    description:
      'The model has absorbed vast knowledge from the internet, but it just predicts the next token. It rambles, continues listing facts, and doesn\'t know when to stop.',
  },
  {
    id: 'sft',
    label: 'Supervised Fine-Tuning',
    shortLabel: 'SFT',
    color: 'text-blue-400',
    borderColor: 'border-blue-500',
    bgGradient: 'from-blue-900/30 to-blue-800/10',
    prompt: 'What is the capital of France?',
    response:
      'The capital of France is Paris. It has been the capital since the 10th century and is the country\'s largest city with a population of about 2.1 million in the city proper.',
    metrics: { helpfulness: 60, safety: 50, reasoning: 35 },
    description:
      'After learning from curated instruction-response pairs, the model follows the question-answer format. Responses are relevant and concise, but can be generic.',
  },
  {
    id: 'rlhf',
    label: 'RL (Alignment)',
    shortLabel: 'RLHF',
    color: 'text-cyan-400',
    borderColor: 'border-cyan-500',
    bgGradient: 'from-cyan-900/30 to-cyan-800/10',
    prompt: 'What is the capital of France?',
    response:
      'Paris is the capital of France. It\'s located in the north-central part of the country along the Seine River. Would you like to know anything specific about Paris, such as its history, landmarks, or culture?',
    metrics: { helpfulness: 85, safety: 90, reasoning: 45 },
    description:
      'RLHF teaches the model what humans prefer: helpful, engaging responses that invite follow-up. The model is proactive, safe, and conversational.',
  },
  {
    id: 'rlvr',
    label: 'RL (Reasoning)',
    shortLabel: 'GRPO',
    color: 'text-amber-400',
    borderColor: 'border-amber-500',
    bgGradient: 'from-amber-900/30 to-amber-800/10',
    prompt: 'A train travels 120 km in 2 hours. It then speeds up and covers 180 km in 2 hours. What is the average speed for the entire trip?',
    response:
      'Let me think through this step by step.\n\nTotal distance: 120 + 180 = 300 km\nTotal time: 2 + 2 = 4 hours\n\nAverage speed = Total distance / Total time\nAverage speed = 300 / 4 = 75 km/h\n\nThe average speed for the entire trip is 75 km/h.\n\nNote: A common mistake is to average the two speeds (60 and 90), getting 75 — which happens to give the correct answer here because the time intervals are equal. But in general, you must use total distance over total time.',
    metrics: { helpfulness: 90, safety: 90, reasoning: 92 },
    description:
      'GRPO with verifiable rewards teaches the model to reason step by step. It breaks problems down, shows its work, catches potential errors, and explains nuances.',
  },
];

function MetricBar({ label, value, color }: { label: string; value: number; color: string }) {
  return (
    <div className="flex items-center gap-2">
      <div className="text-xs text-slate-500 w-16 sm:w-20 text-right flex-shrink-0">{label}</div>
      <div className="flex-1 h-2 bg-slate-800 rounded-full overflow-hidden">
        <div
          className={`h-full rounded-full transition-all duration-500 ${color}`}
          style={{ width: `${value}%` }}
        />
      </div>
      <div className="text-xs text-slate-400 w-8">{value}%</div>
    </div>
  );
}

export function TrainingStagePipeline() {
  const [activeStage, setActiveStage] = useState(0);
  const stage = stages[activeStage];

  return (
    <div className="my-8 bg-slate-800/30 rounded-xl border border-slate-700 overflow-hidden">
      {/* Header */}
      <div className="bg-slate-800/50 px-6 py-4 border-b border-slate-700">
        <h3 className="text-lg font-semibold text-slate-200">LLM Training Pipeline</h3>
        <p className="text-sm text-slate-400">Click each stage to see how responses evolve</p>
      </div>

      <div className="p-6">
        {/* Pipeline stages */}
        <div className="flex items-center gap-0 mb-8">
          {stages.map((s, i) => (
            <React.Fragment key={s.id}>
              <button
                onClick={() => setActiveStage(i)}
                className={`relative flex-1 rounded-lg px-3 py-3 text-center transition-all duration-200 border-2 ${
                  activeStage === i
                    ? `bg-gradient-to-br ${s.bgGradient} ${s.borderColor} shadow-lg`
                    : 'bg-slate-800/30 border-slate-700 hover:border-slate-600'
                }`}
              >
                <div className={`text-sm font-bold ${activeStage === i ? s.color : 'text-slate-400'}`}>
                  <span className="hidden md:inline">{s.label}</span>
                  <span className="md:hidden">{s.shortLabel}</span>
                </div>
                <div className="text-xs text-slate-500 mt-1 hidden md:block">Stage {i + 1}</div>
              </button>
              {i < stages.length - 1 && (
                <div className="text-slate-600 px-1 text-lg flex-shrink-0">&rarr;</div>
              )}
            </React.Fragment>
          ))}
        </div>

        {/* Active stage content */}
        <div className={`rounded-xl border-2 ${stage.borderColor} bg-gradient-to-br ${stage.bgGradient} p-5 mb-6 transition-all duration-300`}>
          {/* Prompt */}
          <div className="mb-4">
            <div className="text-xs text-slate-500 mb-1 font-medium">PROMPT</div>
            <div className="text-slate-300 text-sm bg-slate-900/40 rounded-lg px-4 py-2.5">
              {stage.prompt}
            </div>
          </div>

          {/* Response */}
          <div className="mb-4">
            <div className="text-xs text-slate-500 mb-1 font-medium flex items-center gap-2">
              RESPONSE
              <span className={`${stage.color} text-xs`}>after {stage.label}</span>
            </div>
            <div className="text-slate-200 text-sm bg-slate-900/40 rounded-lg px-4 py-3 whitespace-pre-line leading-relaxed">
              {stage.response}
            </div>
          </div>

          {/* Description */}
          <div className={`text-sm ${stage.color} bg-slate-900/30 rounded-lg px-4 py-2.5`}>
            {stage.description}
          </div>
        </div>

        {/* Metrics */}
        <div className="bg-slate-900/50 rounded-lg p-4">
          <div className="text-sm text-slate-400 mb-3">Model Capabilities</div>
          <div className="space-y-2">
            <MetricBar label="Helpfulness" value={stage.metrics.helpfulness} color="bg-emerald-400" />
            <MetricBar label="Safety" value={stage.metrics.safety} color="bg-blue-400" />
            <MetricBar label="Reasoning" value={stage.metrics.reasoning} color="bg-amber-400" />
          </div>
        </div>
      </div>
    </div>
  );
}

export default TrainingStagePipeline;
