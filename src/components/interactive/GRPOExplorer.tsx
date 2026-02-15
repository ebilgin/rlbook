/**
 * GRPO Explorer
 *
 * Interactive visualization of Group Relative Policy Optimization.
 * Users can:
 * - See sample completions for a math problem with rewards
 * - Adjust rewards and watch advantages change (demonstrating relativity)
 * - Change group size to see bias-variance tradeoff
 * - See how policy probabilities shift based on advantages
 */

import React, { useState, useMemo, useCallback } from 'react';

interface Completion {
  id: number;
  text: string;
  reward: number;
  correct: boolean;
}

interface MathProblem {
  question: string;
  answer: number;
  completions: Completion[];
}

const problems: MathProblem[] = [
  {
    question: 'What is 23 × 17?',
    answer: 391,
    completions: [
      { id: 0, text: '23 × 17 = 23 × 10 + 23 × 7 = 230 + 161 = 391', reward: 1.0, correct: true },
      { id: 1, text: '23 × 17 = 20 × 17 + 3 × 17 = 340 + 51 = 391', reward: 1.0, correct: true },
      { id: 2, text: '23 × 17... let me think... 23 × 20 = 460, minus 23 × 3 = 69... 460 - 69 = 391', reward: 1.0, correct: true },
      { id: 3, text: '23 × 17 = 381', reward: 0.0, correct: false },
      { id: 4, text: '23 × 17 = 23 + 17 = 40... no wait, multiply. 23 × 17 = 400 - 9 = 391', reward: 1.0, correct: true },
      { id: 5, text: '23 × 17 = 23 × 17 = 401', reward: 0.0, correct: false },
      { id: 6, text: '23 × 17 = 389', reward: 0.0, correct: false },
      { id: 7, text: '23 × 17, hmm, 25 × 17 = 425, minus 2 × 17 = 34, so 425 - 34 = 391', reward: 1.0, correct: true },
    ],
  },
];

function computeGroupAdvantages(
  rewards: number[],
  groupSize: number
): { advantages: number[]; mean: number; std: number } {
  const group = rewards.slice(0, groupSize);
  const mean = group.reduce((a, b) => a + b, 0) / group.length;
  const variance = group.reduce((a, b) => a + (b - mean) ** 2, 0) / group.length;
  const std = Math.sqrt(variance);
  const epsilon = 1e-8;

  const advantages = group.map((r) => (r - mean) / (std + epsilon));
  return { advantages, mean, std };
}

function AdvantageBar({
  value,
  maxAbs,
  color,
  label,
  reward,
  correct,
}: {
  value: number;
  maxAbs: number;
  color: string;
  label: string;
  reward: number;
  correct: boolean;
}) {
  const barWidth = maxAbs > 0 ? (Math.abs(value) / maxAbs) * 50 : 0;
  const isPositive = value >= 0;

  return (
    <div className="flex items-center gap-1.5 sm:gap-2 h-8">
      {/* Label + correctness */}
      <div className="text-xs text-slate-500 flex-shrink-0">#{label}</div>
      <div className={`text-xs flex-shrink-0 ${correct ? 'text-emerald-400' : 'text-red-400'}`}>
        {correct ? '✓' : '✗'}
      </div>

      {/* Bar area */}
      <div className="flex-1 relative h-5 min-w-0">
        {/* Center line */}
        <div className="absolute left-1/2 top-0 bottom-0 w-px bg-slate-600" />

        {/* Bar */}
        <div
          className={`absolute top-0.5 bottom-0.5 rounded transition-all duration-300 ${color}`}
          style={{
            left: isPositive ? '50%' : `${50 - barWidth}%`,
            width: `${barWidth}%`,
          }}
        />
      </div>

      {/* Value */}
      <div className={`text-xs text-right flex-shrink-0 tabular-nums ${value >= 0 ? 'text-emerald-400' : 'text-red-400'}`}>
        {value >= 0 ? '+' : ''}{value.toFixed(2)}
      </div>

      {/* Reward (hidden on very small screens) */}
      <div className="text-xs text-slate-500 text-right flex-shrink-0 hidden sm:block">
        r={reward.toFixed(1)}
      </div>
    </div>
  );
}

function PolicyBar({
  completionId,
  advantage,
  baseProb,
  correct,
}: {
  completionId: number;
  advantage: number;
  baseProb: number;
  correct: boolean;
}) {
  // Simulate policy update: probability shift proportional to advantage
  const logitShift = advantage * 0.3;
  const newProb = Math.max(0.01, Math.min(0.95, baseProb * Math.exp(logitShift)));
  const normalizedProb = newProb; // simplified, not renormalizing for visualization

  return (
    <div className="flex items-center gap-1.5 sm:gap-2">
      <div className="text-xs text-slate-500 flex-shrink-0">#{completionId + 1}</div>
      <div className="flex-1 h-4 bg-slate-800 rounded-full overflow-hidden relative min-w-0">
        {/* Old probability (faded) */}
        <div
          className="absolute top-0 bottom-0 left-0 bg-slate-600/40 rounded-full"
          style={{ width: `${baseProb * 100}%` }}
        />
        {/* New probability */}
        <div
          className={`absolute top-0 bottom-0 left-0 rounded-full transition-all duration-500 ${
            correct ? 'bg-emerald-500/70' : 'bg-red-500/70'
          }`}
          style={{ width: `${normalizedProb * 100}%` }}
        />
      </div>
      <div className="text-xs text-slate-400 flex-shrink-0 tabular-nums">
        {(baseProb * 100).toFixed(0)}%→{(normalizedProb * 100).toFixed(0)}%
      </div>
    </div>
  );
}

export function GRPOExplorer() {
  const problem = problems[0];
  const [groupSize, setGroupSize] = useState(8);
  const [adjustedRewards, setAdjustedRewards] = useState<number[]>(
    problem.completions.map((c) => c.reward)
  );
  const [adjustIdx, setAdjustIdx] = useState<number | null>(null);

  const { advantages, mean, std } = useMemo(
    () => computeGroupAdvantages(adjustedRewards, groupSize),
    [adjustedRewards, groupSize]
  );

  const maxAbsAdvantage = Math.max(...advantages.map(Math.abs), 0.1);

  const handleRewardChange = useCallback(
    (idx: number, value: number) => {
      setAdjustedRewards((prev) => {
        const next = [...prev];
        next[idx] = value;
        return next;
      });
    },
    []
  );

  const resetRewards = useCallback(() => {
    setAdjustedRewards(problem.completions.map((c) => c.reward));
    setAdjustIdx(null);
  }, [problem]);

  // Base probabilities (uniform before training)
  const baseProb = 1 / groupSize;

  return (
    <div className="my-8 bg-slate-800/30 rounded-xl border border-slate-700 overflow-hidden">
      {/* Header */}
      <div className="bg-slate-800/50 px-6 py-4 border-b border-slate-700">
        <div className="flex items-center justify-between">
          <div>
            <h3 className="text-lg font-semibold text-slate-200">Try It: GRPO Explorer</h3>
            <p className="text-sm text-slate-400">
              See how group-relative advantages work — adjust rewards and watch everything change
            </p>
          </div>
          <button
            onClick={resetRewards}
            className="px-3 py-1.5 text-sm bg-slate-700 hover:bg-slate-600 text-slate-300 rounded-lg transition-colors"
          >
            Reset
          </button>
        </div>
      </div>

      <div className="p-6">
        {/* Problem */}
        <div className="bg-slate-900/50 rounded-lg p-4 mb-6">
          <div className="text-xs text-slate-500 mb-1 font-medium">MATH PROBLEM</div>
          <div className="text-lg text-slate-200 font-medium">{problem.question}</div>
          <div className="text-xs text-slate-500 mt-1">Correct answer: {problem.answer}</div>
        </div>

        {/* Group size selector */}
        <div className="flex items-center gap-4 mb-6">
          <span className="text-sm text-slate-400">Group size (G):</span>
          <div className="flex gap-2">
            {[4, 6, 8].map((g) => (
              <button
                key={g}
                onClick={() => setGroupSize(g)}
                className={`px-3 py-1.5 rounded text-sm transition-colors ${
                  groupSize === g
                    ? 'bg-amber-600 text-white'
                    : 'bg-slate-700 text-slate-300 hover:bg-slate-600'
                }`}
              >
                G={g}
              </button>
            ))}
          </div>
        </div>

        {/* Completions with rewards */}
        <div className="mb-6">
          <div className="text-sm text-slate-400 mb-3">
            Sampled Completions (G={groupSize}) — click a reward to adjust it
          </div>
          <div className="space-y-2">
            {problem.completions.slice(0, groupSize).map((comp, i) => (
              <div
                key={comp.id}
                className={`bg-slate-900/40 rounded-lg p-3 border transition-colors ${
                  adjustIdx === i ? 'border-amber-500' : 'border-slate-700/50'
                }`}
              >
                <div className="flex items-start gap-3">
                  <div
                    className={`w-5 h-5 rounded-full flex items-center justify-center flex-shrink-0 mt-0.5 text-xs ${
                      adjustedRewards[i] > 0.5
                        ? 'bg-emerald-900/50 text-emerald-400'
                        : 'bg-red-900/50 text-red-400'
                    }`}
                  >
                    {adjustedRewards[i] > 0.5 ? '✓' : '✗'}
                  </div>
                  <div className="flex-1 text-sm text-slate-300 font-mono break-all sm:break-normal min-w-0">{comp.text}</div>
                  <button
                    onClick={() => setAdjustIdx(adjustIdx === i ? null : i)}
                    className={`px-2 py-1 rounded text-xs flex-shrink-0 transition-colors ${
                      adjustIdx === i
                        ? 'bg-amber-600 text-white'
                        : 'bg-slate-700 text-slate-400 hover:bg-slate-600'
                    }`}
                  >
                    r={adjustedRewards[i].toFixed(1)}
                  </button>
                </div>
                {adjustIdx === i && (
                  <div className="mt-3 flex items-center gap-3 pl-8">
                    <span className="text-xs text-slate-500">0</span>
                    <input
                      type="range"
                      min="0"
                      max="1"
                      step="0.1"
                      value={adjustedRewards[i]}
                      onChange={(e) => handleRewardChange(i, parseFloat(e.target.value))}
                      className="flex-1 h-2 bg-slate-700 rounded-lg appearance-none cursor-pointer accent-amber-400"
                    />
                    <span className="text-xs text-slate-500">1</span>
                  </div>
                )}
              </div>
            ))}
          </div>
        </div>

        {/* Step-by-step advantage calculation */}
        <div className="bg-slate-900/50 rounded-lg p-5 mb-6">
          <div className="text-sm text-slate-400 mb-1">Step 1: Compute Group Statistics</div>
          <div className="grid grid-cols-2 gap-4 mb-4">
            <div className="bg-slate-800/50 rounded-lg p-3 text-center">
              <div className="text-xs text-slate-500">Mean Reward (μ)</div>
              <div className="text-xl font-bold text-cyan-400">{mean.toFixed(3)}</div>
            </div>
            <div className="bg-slate-800/50 rounded-lg p-3 text-center">
              <div className="text-xs text-slate-500">Std Dev (σ)</div>
              <div className="text-xl font-bold text-cyan-400">{std.toFixed(3)}</div>
            </div>
          </div>

          <div className="text-sm text-slate-400 mb-1">Step 2: Compute Advantages</div>
          <div className="text-xs text-slate-500 mb-3">
            Â_i = (r_i - μ) / (σ + ε)
          </div>
          <div className="space-y-1">
            {advantages.map((adv, i) => (
              <AdvantageBar
                key={i}
                value={adv}
                maxAbs={maxAbsAdvantage}
                color={adv >= 0 ? 'bg-emerald-500/60' : 'bg-red-500/60'}
                label={String(i + 1)}
                reward={adjustedRewards[i]}
                correct={problem.completions[i].correct}
              />
            ))}
          </div>
        </div>

        {/* Policy update visualization */}
        <div className="bg-slate-900/50 rounded-lg p-5 mb-6">
          <div className="text-sm text-slate-400 mb-1">Step 3: Update Policy</div>
          <div className="text-xs text-slate-500 mb-3">
            Increase probability of positive-advantage completions, decrease negative ones
          </div>
          <div className="space-y-2">
            {advantages.map((adv, i) => (
              <PolicyBar
                key={i}
                completionId={i}
                advantage={adv}
                baseProb={baseProb}
                correct={problem.completions[i].correct}
              />
            ))}
          </div>
          <div className="mt-3 flex justify-between text-xs text-slate-500">
            <span className="flex items-center gap-1">
              <span className="w-3 h-2 rounded bg-slate-600/40"></span> Before
            </span>
            <span className="flex items-center gap-1">
              <span className="w-3 h-2 rounded bg-emerald-500/70"></span> After (correct)
            </span>
            <span className="flex items-center gap-1">
              <span className="w-3 h-2 rounded bg-red-500/70"></span> After (wrong)
            </span>
          </div>
        </div>

        {/* Dynamic insight */}
        <div className="rounded-lg p-4 bg-gradient-to-r from-amber-900/30 to-amber-900/10 border border-amber-700/50">
          <div className="flex items-start gap-3">
            <span className="text-xl">💡</span>
            <div>
              <div className="font-semibold text-amber-400">Key Insight: Advantages Are Relative</div>
              <div className="text-sm text-slate-300 mt-1">
                {mean > 0.7 ? (
                  <>
                    Most completions got it right (mean reward = {mean.toFixed(2)}). So even correct answers get near-zero or slightly negative advantages — because they're average, not special. Only the wrong answers get strong negative signal.
                    <span className="text-amber-400"> Try adjusting rewards to see how one change affects ALL advantages.</span>
                  </>
                ) : mean < 0.3 ? (
                  <>
                    Most completions got it wrong (mean reward = {mean.toFixed(2)}). The few correct ones get very large positive advantages — they stand out from the crowd. The wrong ones get moderate negative advantages since failing is typical.
                  </>
                ) : (
                  <>
                    With mixed results (mean = {mean.toFixed(2)}), correct and incorrect completions get clear positive and negative advantages respectively. This is where GRPO learning is most effective — there's a clear signal about what works and what doesn't.
                  </>
                )}
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

export default GRPOExplorer;
