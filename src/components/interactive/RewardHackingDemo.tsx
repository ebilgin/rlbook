/**
 * Reward Hacking Demo
 *
 * Interactive visualization showing how optimizing a proxy reward
 * without KL penalty leads to reward hacking. Users can toggle
 * KL penalty on/off and watch the divergence between proxy reward
 * and true quality.
 */

import React, { useState, useMemo } from 'react';

interface TrainingStep {
  step: number;
  proxyReward: number;
  trueQuality: number;
  klDivergence: number;
  sampleResponse: string;
}

// Simulate training with and without KL penalty
function generateTrainingCurve(withKL: boolean): TrainingStep[] {
  const steps: TrainingStep[] = [];
  const numSteps = 20;

  const responses = {
    withKL: [
      'Paris is the capital of France.',
      'Paris is the capital of France. It is located in the north-central part of the country.',
      'Paris is the capital of France, located on the Seine River. It has been the capital since the 10th century.',
      "Paris is the capital of France, known for landmarks like the Eiffel Tower and the Louvre. It's a major European cultural and economic center.",
      "Paris is the capital of France. It's a vibrant city known for its art, culture, and cuisine. The city has been the country's capital since the 10th century.",
    ],
    withoutKL: [
      'Paris is the capital of France.',
      'Paris is the capital of France, a wonderful magnificent extraordinary city of great importance.',
      'Paris is the capital of France!!!!! Absolutely AMAZING, INCREDIBLE, FANTASTIC, WONDERFUL, MAGNIFICENT, SPECTACULAR city!!!',
      'PARIS PARIS PARIS is the BEST MOST AMAZING GREATEST MOST WONDERFUL INCREDIBLE SPECTACULAR MAGNIFICENT OUTSTANDING EXTRAORDINARY PHENOMENAL REMARKABLE capital!!!!!!!!',
      'INCREDIBLE AMAZING WONDERFUL SPECTACULAR MAGNIFICENT OUTSTANDING EXTRAORDINARY PHENOMENAL REMARKABLE SENSATIONAL FANTASTIC BRILLIANT SUPERB EXCELLENT EXCEPTIONAL GLORIOUS MAJESTIC the the the the the',
    ],
  };

  for (let i = 0; i <= numSteps; i++) {
    const t = i / numSteps;

    if (withKL) {
      // With KL: proxy reward increases moderately, true quality also improves
      const proxyReward = 0.3 + 0.45 * (1 - Math.exp(-3 * t));
      const trueQuality = 0.3 + 0.4 * (1 - Math.exp(-2.5 * t));
      const klDivergence = 0.15 * t;
      const responseIdx = Math.min(Math.floor(t * responses.withKL.length), responses.withKL.length - 1);

      steps.push({
        step: i * 50,
        proxyReward,
        trueQuality,
        klDivergence,
        sampleResponse: responses.withKL[responseIdx],
      });
    } else {
      // Without KL: proxy reward skyrockets, true quality degrades
      const proxyReward = 0.3 + 0.65 * (1 - Math.exp(-4 * t));
      const peakT = 0.25; // quality peaks early then degrades
      const trueQuality =
        t < peakT
          ? 0.3 + 0.35 * (t / peakT)
          : 0.65 - 0.55 * ((t - peakT) / (1 - peakT));
      const klDivergence = 2.5 * t * t;
      const responseIdx = Math.min(Math.floor(t * responses.withoutKL.length), responses.withoutKL.length - 1);

      steps.push({
        step: i * 50,
        proxyReward,
        trueQuality: Math.max(0.05, trueQuality),
        klDivergence,
        sampleResponse: responses.withoutKL[responseIdx],
      });
    }
  }

  return steps;
}

function MiniChart({
  data,
  dataKey,
  color,
  label,
  maxVal,
}: {
  data: TrainingStep[];
  dataKey: keyof TrainingStep;
  color: string;
  label: string;
  maxVal: number;
}) {
  const width = 100;
  const height = 60;
  const padding = 2;

  const points = data.map((d, i) => {
    const x = padding + (i / (data.length - 1)) * (width - 2 * padding);
    const y = height - padding - ((d[dataKey] as number) / maxVal) * (height - 2 * padding);
    return `${x},${y}`;
  });

  return (
    <div>
      <div className="text-xs text-slate-500 mb-1">{label}</div>
      <svg viewBox={`0 0 ${width} ${height}`} className="w-full h-16">
        <polyline
          points={points.join(' ')}
          fill="none"
          stroke={color}
          strokeWidth="2"
          strokeLinecap="round"
          strokeLinejoin="round"
        />
      </svg>
    </div>
  );
}

export function RewardHackingDemo() {
  const [withKL, setWithKL] = useState(false);
  const [currentStep, setCurrentStep] = useState(10);

  const dataWithKL = useMemo(() => generateTrainingCurve(true), []);
  const dataWithoutKL = useMemo(() => generateTrainingCurve(false), []);

  const data = withKL ? dataWithKL : dataWithoutKL;
  const currentData = data[currentStep];
  const visibleData = data.slice(0, currentStep + 1);

  return (
    <div className="my-8 bg-slate-800/30 rounded-xl border border-slate-700 overflow-hidden">
      {/* Header */}
      <div className="bg-slate-800/50 px-6 py-4 border-b border-slate-700">
        <h3 className="text-lg font-semibold text-slate-200">Try It: Reward Hacking</h3>
        <p className="text-sm text-slate-400">
          Watch what happens when a model optimizes a proxy reward — with and without the KL safety net
        </p>
      </div>

      <div className="p-6">
        {/* KL toggle */}
        <div className="flex items-center justify-center gap-2 sm:gap-4 mb-6">
          <button
            onClick={() => { setWithKL(false); setCurrentStep(10); }}
            className={`px-3 sm:px-4 py-2.5 rounded-lg text-xs sm:text-sm font-medium transition-colors ${
              !withKL
                ? 'bg-red-600 text-white'
                : 'bg-slate-700 text-slate-300 hover:bg-slate-600'
            }`}
          >
            Without KL Penalty
          </button>
          <button
            onClick={() => { setWithKL(true); setCurrentStep(10); }}
            className={`px-3 sm:px-4 py-2.5 rounded-lg text-xs sm:text-sm font-medium transition-colors ${
              withKL
                ? 'bg-emerald-600 text-white'
                : 'bg-slate-700 text-slate-300 hover:bg-slate-600'
            }`}
          >
            With KL Penalty
          </button>
        </div>

        {/* Training step slider */}
        <div className="mb-6">
          <div className="flex items-center justify-between mb-2">
            <span className="text-sm text-slate-400">Training Progress</span>
            <span className="text-sm text-slate-300">Step {currentData.step}</span>
          </div>
          <input
            type="range"
            min="0"
            max={data.length - 1}
            value={currentStep}
            onChange={(e) => setCurrentStep(parseInt(e.target.value))}
            className={`w-full h-3 bg-slate-700 rounded-lg appearance-none cursor-pointer ${
              withKL ? 'accent-emerald-400' : 'accent-red-400'
            }`}
          />
        </div>

        {/* Metrics */}
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-3 sm:gap-4 mb-6">
          <div className="bg-slate-900/50 rounded-lg p-4 text-center">
            <div className="text-xs text-slate-500 mb-1">Proxy Reward</div>
            <div className="text-2xl font-bold text-amber-400">
              {currentData.proxyReward.toFixed(2)}
            </div>
            <div className="text-xs text-slate-500 mt-1">What the model optimizes</div>
          </div>
          <div className="bg-slate-900/50 rounded-lg p-4 text-center">
            <div className="text-xs text-slate-500 mb-1">True Quality</div>
            <div
              className={`text-2xl font-bold ${
                currentData.trueQuality > 0.5 ? 'text-emerald-400' : 'text-red-400'
              }`}
            >
              {currentData.trueQuality.toFixed(2)}
            </div>
            <div className="text-xs text-slate-500 mt-1">What we actually want</div>
          </div>
          <div className="bg-slate-900/50 rounded-lg p-4 text-center">
            <div className="text-xs text-slate-500 mb-1">KL Divergence</div>
            <div
              className={`text-2xl font-bold ${
                currentData.klDivergence < 0.5 ? 'text-cyan-400' : 'text-red-400'
              }`}
            >
              {currentData.klDivergence.toFixed(2)}
            </div>
            <div className="text-xs text-slate-500 mt-1">Drift from base model</div>
          </div>
        </div>

        {/* Charts side by side */}
        <div className="grid grid-cols-2 gap-4 mb-6">
          <div className="bg-slate-900/50 rounded-lg p-4">
            <MiniChart data={visibleData} dataKey="proxyReward" color="#fbbf24" label="Proxy Reward" maxVal={1} />
            <MiniChart data={visibleData} dataKey="trueQuality" color="#34d399" label="True Quality" maxVal={1} />
          </div>
          <div className="bg-slate-900/50 rounded-lg p-4">
            {/* Gap visualization */}
            <div className="text-xs text-slate-500 mb-2">Reward-Quality Gap</div>
            <div className="flex items-end gap-1 h-20">
              {visibleData.map((d, i) => {
                const gap = d.proxyReward - d.trueQuality;
                const barHeight = Math.abs(gap) * 100;
                return (
                  <div
                    key={i}
                    className={`flex-1 rounded-t transition-all ${
                      gap > 0.15 ? 'bg-red-500/60' : 'bg-slate-600/40'
                    }`}
                    style={{ height: `${Math.min(barHeight, 100)}%` }}
                  />
                );
              })}
            </div>
            <div className="text-xs text-slate-500 mt-1">
              {!withKL && currentStep > 8
                ? 'Gap widening — reward hacking detected!'
                : 'Gap between proxy and true reward'}
            </div>
          </div>
        </div>

        {/* Sample response */}
        <div className="bg-slate-900/50 rounded-lg p-4 mb-6">
          <div className="text-xs text-slate-500 mb-1 font-medium">
            SAMPLE RESPONSE AT STEP {currentData.step}
          </div>
          <div className="text-xs text-slate-500 mb-2">Prompt: "What is the capital of France?"</div>
          <div
            className={`text-sm rounded-lg px-4 py-3 border ${
              currentData.trueQuality > 0.5
                ? 'bg-emerald-900/10 border-emerald-700/30 text-slate-200'
                : 'bg-red-900/10 border-red-700/30 text-slate-300'
            }`}
          >
            {currentData.sampleResponse}
          </div>
        </div>

        {/* Insight */}
        <div
          className={`rounded-lg p-4 bg-gradient-to-r border ${
            withKL
              ? 'from-emerald-900/30 to-emerald-900/10 border-emerald-700/50'
              : 'from-red-900/30 to-red-900/10 border-red-700/50'
          }`}
        >
          <div className="flex items-start gap-3">
            <span className="text-xl">{withKL ? '🛡️' : '⚠️'}</span>
            <div>
              <div className={`font-semibold ${withKL ? 'text-emerald-400' : 'text-red-400'}`}>
                {withKL ? 'KL Penalty Prevents Collapse' : 'Reward Hacking in Action'}
              </div>
              <div className="text-sm text-slate-300 mt-1">
                {withKL ? (
                  <>
                    The KL penalty keeps the model close to its base behavior, preventing it from
                    finding degenerate solutions. The proxy reward improves moderately, and true
                    quality improves along with it. The model gets better without going off the rails.
                  </>
                ) : (
                  <>
                    Without the KL constraint, the model discovers that repeating superlatives,
                    exclamation marks, and emphatic language scores high with the reward model — even
                    though the responses become nonsensical. The proxy reward keeps climbing while
                    true quality collapses. This is why every RLHF system uses a KL penalty.
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

export default RewardHackingDemo;
