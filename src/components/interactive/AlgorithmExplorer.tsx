/**
 * Algorithm Explorer
 *
 * Interactive comparison of PPO, DPO, and GRPO for LLM training.
 * Users can select model sizes and see memory requirements,
 * pipeline diagrams, and tradeoff comparisons.
 */

import React, { useState, useMemo } from 'react';

interface Algorithm {
  id: string;
  name: string;
  fullName: string;
  color: string;
  textColor: string;
  borderColor: string;
  bgGradient: string;
  modelsInMemory: string[];
  modelMultiplier: number; // How many copies of the base model
  dataType: string;
  online: boolean;
  bestFor: string;
  keyInsight: string;
  pipelineSteps: string[];
}

const algorithms: Algorithm[] = [
  {
    id: 'ppo',
    name: 'PPO',
    fullName: 'Proximal Policy Optimization',
    color: 'bg-blue-500',
    textColor: 'text-blue-400',
    borderColor: 'border-blue-500',
    bgGradient: 'from-blue-900/30 to-blue-800/10',
    modelsInMemory: ['Policy', 'Reference', 'Critic', 'Reward Model'],
    modelMultiplier: 4,
    dataType: 'Online generation',
    online: true,
    bestFor: 'Maximum control, established pipeline',
    keyInsight: 'Most flexible but most expensive — 4 models in GPU memory',
    pipelineSteps: ['Generate response', 'Score with reward model', 'Estimate advantages (critic)', 'Update policy (clipped)'],
  },
  {
    id: 'dpo',
    name: 'DPO',
    fullName: 'Direct Preference Optimization',
    color: 'bg-emerald-500',
    textColor: 'text-emerald-400',
    borderColor: 'border-emerald-500',
    bgGradient: 'from-emerald-900/30 to-emerald-800/10',
    modelsInMemory: ['Policy', 'Reference'],
    modelMultiplier: 2,
    dataType: 'Offline preference pairs',
    online: false,
    bestFor: 'Simple alignment with existing preference data',
    keyInsight: 'Eliminates reward model — treats alignment as classification',
    pipelineSteps: ['Load preference pair (chosen, rejected)', 'Compute log-ratios vs reference', 'Apply sigmoid loss', 'Update policy'],
  },
  {
    id: 'grpo',
    name: 'GRPO',
    fullName: 'Group Relative Policy Optimization',
    color: 'bg-amber-500',
    textColor: 'text-amber-400',
    borderColor: 'border-amber-500',
    bgGradient: 'from-amber-900/30 to-amber-800/10',
    modelsInMemory: ['Policy', 'Reference'],
    modelMultiplier: 2,
    dataType: 'Online + verifiable rewards',
    online: true,
    bestFor: 'Reasoning tasks with checkable answers',
    keyInsight: 'Eliminates critic — uses group statistics as baseline',
    pipelineSteps: ['Sample G completions per prompt', 'Check correctness (reward)', 'Compute group-relative advantages', 'Update policy (clipped + KL)'],
  },
];

interface ModelPreset {
  name: string;
  params: number; // billions
}

const modelPresets: ModelPreset[] = [
  { name: 'Qwen-1.5B', params: 1.5 },
  { name: 'Llama-7B', params: 7 },
  { name: 'Llama-13B', params: 13 },
  { name: 'Llama-70B', params: 70 },
];

interface GPUSpec {
  name: string;
  vram: number;
}

const gpus: GPUSpec[] = [
  { name: 'RTX 4090', vram: 24 },
  { name: 'A100 40GB', vram: 40 },
  { name: 'A100 80GB', vram: 80 },
  { name: 'H100 80GB', vram: 80 },
];

function MemoryBar({
  algo,
  memoryGB,
  maxMemory,
}: {
  algo: Algorithm;
  memoryGB: number;
  maxMemory: number;
}) {
  const widthPercent = Math.min((memoryGB / maxMemory) * 100, 100);

  return (
    <div className="mb-4">
      <div className="flex items-center justify-between mb-1">
        <span className={`text-sm font-medium ${algo.textColor}`}>{algo.name}</span>
        <span className="text-sm text-slate-400">{memoryGB.toFixed(1)} GB</span>
      </div>
      <div className="relative h-6 bg-slate-800 rounded-full overflow-hidden">
        <div
          className={`absolute left-0 top-0 bottom-0 ${algo.color} rounded-full transition-all duration-500 opacity-80`}
          style={{ width: `${widthPercent}%` }}
        />
        {/* GPU markers */}
        {gpus.map((gpu) => {
          const markerPos = (gpu.vram / maxMemory) * 100;
          if (markerPos > 100) return null;
          return (
            <div
              key={gpu.name}
              className="absolute top-0 bottom-0 w-px bg-slate-400/40"
              style={{ left: `${markerPos}%` }}
            >
              <div className="absolute -top-5 -translate-x-1/2 text-[10px] text-slate-500 whitespace-nowrap">
                {gpu.vram}GB
              </div>
            </div>
          );
        })}
      </div>
      {/* Fit indicator */}
      <div className="mt-1 text-xs">
        {memoryGB <= 24 ? (
          <span className="text-emerald-400">Fits on RTX 4090 (24 GB)</span>
        ) : memoryGB <= 80 ? (
          <span className="text-amber-400">Needs A100/H100 (80 GB)</span>
        ) : (
          <span className="text-red-400">Needs multi-GPU</span>
        )}
      </div>
    </div>
  );
}

export function AlgorithmExplorer() {
  const [selectedModel, setSelectedModel] = useState(1); // default: 7B
  const [activeAlgo, setActiveAlgo] = useState<string | null>(null);

  const modelParams = modelPresets[selectedModel].params;
  const selectedAlgo = activeAlgo ? algorithms.find((a) => a.id === activeAlgo) : null;

  // Memory calculation: params * 2 bytes (float16) * multiplier + overhead
  const memoryCalcs = useMemo(() => {
    return algorithms.map((algo) => ({
      algo,
      memoryGB: modelParams * 2 * algo.modelMultiplier, // float16 = 2 bytes per param
    }));
  }, [modelParams]);

  const maxMemory = Math.max(...memoryCalcs.map((m) => m.memoryGB), 80) * 1.15;

  return (
    <div className="my-8 bg-slate-800/30 rounded-xl border border-slate-700 overflow-hidden">
      {/* Header */}
      <div className="bg-slate-800/50 px-6 py-4 border-b border-slate-700">
        <h3 className="text-lg font-semibold text-slate-200">Algorithm Explorer</h3>
        <p className="text-sm text-slate-400">
          Compare PPO, DPO, and GRPO — select a model size to see memory requirements
        </p>
      </div>

      <div className="p-6">
        {/* Model size selector */}
        <div className="mb-8">
          <div className="text-sm text-slate-400 mb-3">Select model size:</div>
          <div className="flex gap-2 flex-wrap">
            {modelPresets.map((model, i) => (
              <button
                key={model.name}
                onClick={() => setSelectedModel(i)}
                className={`px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
                  selectedModel === i
                    ? 'bg-violet-600 text-white'
                    : 'bg-slate-700 text-slate-300 hover:bg-slate-600'
                }`}
              >
                {model.name}
              </button>
            ))}
          </div>
        </div>

        {/* Algorithm cards */}
        <div className="grid md:grid-cols-3 gap-4 mb-8">
          {algorithms.map((algo) => (
            <button
              key={algo.id}
              onClick={() => setActiveAlgo(activeAlgo === algo.id ? null : algo.id)}
              className={`text-left rounded-xl p-4 border-2 transition-all ${
                activeAlgo === algo.id
                  ? `bg-gradient-to-br ${algo.bgGradient} ${algo.borderColor}`
                  : 'bg-slate-800/50 border-slate-700 hover:border-slate-600'
              }`}
            >
              <div className={`text-lg font-bold ${algo.textColor}`}>{algo.name}</div>
              <div className="text-xs text-slate-500 mb-2">{algo.fullName}</div>
              <div className="text-sm text-slate-300 mb-3">{algo.keyInsight}</div>
              <div className="flex flex-wrap gap-1">
                {algo.modelsInMemory.map((model) => (
                  <span
                    key={model}
                    className={`text-xs px-2 py-0.5 rounded ${
                      activeAlgo === algo.id
                        ? `bg-slate-900/50 ${algo.textColor}`
                        : 'bg-slate-700 text-slate-400'
                    }`}
                  >
                    {model}
                  </span>
                ))}
              </div>
            </button>
          ))}
        </div>

        {/* Memory comparison bars */}
        <div className="bg-slate-900/50 rounded-lg p-5 mb-6">
          <div className="text-sm text-slate-400 mb-1">
            Memory Requirements ({modelPresets[selectedModel].name}, float16)
          </div>
          <div className="text-xs text-slate-500 mb-5">
            Each model copy = {modelParams}B params × 2 bytes = {(modelParams * 2).toFixed(1)} GB
          </div>

          {memoryCalcs.map(({ algo, memoryGB }) => (
            <MemoryBar key={algo.id} algo={algo} memoryGB={memoryGB} maxMemory={maxMemory} />
          ))}
        </div>

        {/* Key insight box */}
        <div className="rounded-lg p-4 bg-gradient-to-r from-violet-900/30 to-violet-900/10 border border-violet-700/50 mb-6">
          <div className="flex items-start gap-3">
            <span className="text-xl">💡</span>
            <div>
              <div className="font-semibold text-violet-400">The Memory Tradeoff</div>
              <div className="text-sm text-slate-300 mt-1">
                For a {modelPresets[selectedModel].name} model, PPO needs{' '}
                <span className="text-blue-400 font-medium">
                  {(modelParams * 2 * 4).toFixed(0)} GB
                </span>{' '}
                (4 models) while DPO and GRPO need only{' '}
                <span className="text-emerald-400 font-medium">
                  {(modelParams * 2 * 2).toFixed(0)} GB
                </span>{' '}
                (2 models) — a{' '}
                <span className="text-amber-400 font-medium">2× reduction</span>.
                {modelParams >= 13 && ' At this scale, that difference determines whether you need 1 GPU or 4.'}
              </div>
            </div>
          </div>
        </div>

        {/* Pipeline detail (when an algo is selected) */}
        {selectedAlgo && (
          <div className="bg-slate-900/50 rounded-lg p-5 mb-6">
            <div className={`text-sm font-medium ${selectedAlgo.textColor} mb-3`}>
              {selectedAlgo.name} Training Loop
            </div>
            <div className="space-y-2">
              {selectedAlgo.pipelineSteps.map((step, i) => (
                <div key={i} className="flex items-center gap-3">
                  <div
                    className={`w-6 h-6 rounded-full ${selectedAlgo.color} flex items-center justify-center text-xs text-white font-bold flex-shrink-0`}
                  >
                    {i + 1}
                  </div>
                  <div className="text-sm text-slate-300">{step}</div>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Comparison table */}
        <div className="bg-slate-900/50 rounded-lg p-5">
          <div className="text-sm text-slate-400 mb-3">Quick Comparison</div>
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="text-slate-500 border-b border-slate-700">
                  <th className="text-left py-2 px-2"></th>
                  <th className="text-left py-2 px-2 text-blue-400">PPO</th>
                  <th className="text-left py-2 px-2 text-emerald-400">DPO</th>
                  <th className="text-left py-2 px-2 text-amber-400">GRPO</th>
                </tr>
              </thead>
              <tbody className="text-slate-300">
                <tr className="border-b border-slate-800">
                  <td className="py-2 px-2 text-slate-500">Models in memory</td>
                  <td className="py-2 px-2">4</td>
                  <td className="py-2 px-2">2</td>
                  <td className="py-2 px-2">2</td>
                </tr>
                <tr className="border-b border-slate-800">
                  <td className="py-2 px-2 text-slate-500">Data type</td>
                  <td className="py-2 px-2">Online</td>
                  <td className="py-2 px-2">Offline pairs</td>
                  <td className="py-2 px-2">Online</td>
                </tr>
                <tr className="border-b border-slate-800">
                  <td className="py-2 px-2 text-slate-500">Reward source</td>
                  <td className="py-2 px-2">Learned RM</td>
                  <td className="py-2 px-2">Implicit</td>
                  <td className="py-2 px-2">Verifiable</td>
                </tr>
                <tr className="border-b border-slate-800">
                  <td className="py-2 px-2 text-slate-500">Critic needed</td>
                  <td className="py-2 px-2">Yes</td>
                  <td className="py-2 px-2">No</td>
                  <td className="py-2 px-2">No</td>
                </tr>
                <tr>
                  <td className="py-2 px-2 text-slate-500">Best for</td>
                  <td className="py-2 px-2 text-xs">{algorithms[0].bestFor}</td>
                  <td className="py-2 px-2 text-xs">{algorithms[1].bestFor}</td>
                  <td className="py-2 px-2 text-xs">{algorithms[2].bestFor}</td>
                </tr>
              </tbody>
            </table>
          </div>
        </div>
      </div>
    </div>
  );
}

export default AlgorithmExplorer;
