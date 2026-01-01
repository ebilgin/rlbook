/**
 * Interactive Memory Calculator
 *
 * Shows how model size changes with different precisions.
 * Users can:
 * - Select model size (or enter custom)
 * - See memory bars for each precision
 * - See which GPUs can fit each configuration
 */

import React, { useState, useMemo } from 'react';

interface GPUSpec {
  name: string;
  vram: number; // in GB
  color: string;
}

const gpus: GPUSpec[] = [
  { name: 'RTX 3060', vram: 12, color: 'bg-slate-500' },
  { name: 'RTX 4070', vram: 12, color: 'bg-slate-500' },
  { name: 'RTX 3080', vram: 10, color: 'bg-blue-500' },
  { name: 'RTX 4080', vram: 16, color: 'bg-blue-500' },
  { name: 'RTX 4090', vram: 24, color: 'bg-violet-500' },
  { name: 'A100 40GB', vram: 40, color: 'bg-emerald-500' },
  { name: 'A100 80GB', vram: 80, color: 'bg-amber-500' },
];

const presetModels = [
  { name: 'Qwen2.5-0.5B', params: 0.5 },
  { name: 'Llama 3-1B', params: 1 },
  { name: 'Llama 3-3B', params: 3 },
  { name: 'Llama 3-7B', params: 7 },
  { name: 'Llama 3-13B', params: 13 },
  { name: 'Llama 3-70B', params: 70 },
];

interface PrecisionConfig {
  name: string;
  bits: number;
  color: string;
  description: string;
}

const precisions: PrecisionConfig[] = [
  { name: 'float32', bits: 32, color: 'bg-slate-400', description: 'Full precision (baseline)' },
  { name: 'float16', bits: 16, color: 'bg-blue-400', description: 'Half precision' },
  { name: 'int8', bits: 8, color: 'bg-emerald-400', description: 'Standard quantized' },
  { name: 'int4', bits: 4, color: 'bg-amber-400', description: 'Aggressive quantized' },
];

function formatMemory(gb: number): string {
  if (gb >= 1) return `${gb.toFixed(1)} GB`;
  return `${(gb * 1024).toFixed(0)} MB`;
}

function MemoryBar({
  memory,
  maxMemory,
  precision,
  fitsGpu,
}: {
  memory: number;
  maxMemory: number;
  precision: PrecisionConfig;
  fitsGpu: GPUSpec | null;
}) {
  const widthPercent = Math.min((memory / maxMemory) * 100, 100);

  return (
    <div className="mb-3">
      <div className="flex items-center justify-between mb-1">
        <span className="text-sm font-medium text-slate-300">{precision.name}</span>
        <span className="text-sm text-slate-400">{formatMemory(memory)}</span>
      </div>
      <div className="relative h-8 bg-slate-800 rounded-lg overflow-hidden">
        {/* Memory bar */}
        <div
          className={`h-full ${precision.color} transition-all duration-500 rounded-lg`}
          style={{ width: `${widthPercent}%` }}
        />

        {/* GPU VRAM markers */}
        {gpus.slice(0, 5).map((gpu, i) => {
          const markerPos = (gpu.vram / maxMemory) * 100;
          if (markerPos > 100) return null;
          return (
            <div
              key={i}
              className="absolute top-0 bottom-0 w-px bg-slate-500/50"
              style={{ left: `${markerPos}%` }}
              title={`${gpu.name}: ${gpu.vram}GB`}
            />
          );
        })}

        {/* Compression badge */}
        {precision.bits < 32 && (
          <div className="absolute right-2 top-1/2 -translate-y-1/2 text-xs font-bold text-white drop-shadow-[0_1px_2px_rgba(0,0,0,0.8)]">
            {(32 / precision.bits).toFixed(0)}× smaller
          </div>
        )}
      </div>

      {/* Fits indicator */}
      <div className="mt-1 text-xs">
        {fitsGpu ? (
          <span className="text-emerald-300 font-medium">✓ Fits on {fitsGpu.name}</span>
        ) : (
          <span className="text-red-300 font-medium">✗ Needs multi-GPU or larger</span>
        )}
      </div>
    </div>
  );
}

export function MemoryCalculator() {
  const [selectedPreset, setSelectedPreset] = useState(3); // Default to 7B
  const [customParams, setCustomParams] = useState<number | null>(null);

  const params = customParams ?? presetModels[selectedPreset].params;
  const modelName = customParams ? `${customParams}B model` : presetModels[selectedPreset].name;

  // Calculate memory for each precision
  const memories = useMemo(() => {
    return precisions.map(p => ({
      precision: p,
      memory: (params * p.bits) / 8, // params in billions, result in GB
    }));
  }, [params]);

  const maxMemory = Math.max(memories[0].memory, 80); // At least show up to 80GB scale

  // Find smallest GPU that fits each
  const fitsGpu = (memory: number): GPUSpec | null => {
    return gpus.find(g => g.vram >= memory) || null;
  };

  return (
    <div className="my-8 bg-slate-800/30 rounded-xl border border-slate-700 overflow-hidden">
      {/* Header */}
      <div className="bg-slate-800/50 px-6 py-4 border-b border-slate-700">
        <h3 className="text-lg font-semibold text-slate-200">Memory Calculator</h3>
        <p className="text-sm text-slate-400">See how quantization shrinks model memory</p>
      </div>

      <div className="p-6">
        {/* Model selector */}
        <div className="mb-6">
          <label className="block text-sm text-slate-400 mb-2">Select a model:</label>
          <div className="flex flex-wrap gap-2">
            {presetModels.map((model, i) => (
              <button
                key={i}
                onClick={() => {
                  setSelectedPreset(i);
                  setCustomParams(null);
                }}
                className={`px-3 py-1.5 rounded-lg text-sm transition-colors ${
                  selectedPreset === i && customParams === null
                    ? 'bg-cyan-600 text-white'
                    : 'bg-slate-700 text-slate-300 hover:bg-slate-600'
                }`}
              >
                {model.name}
              </button>
            ))}
          </div>
        </div>

        {/* Custom input */}
        <div className="mb-6">
          <label className="block text-sm text-slate-400 mb-2">Or enter custom size:</label>
          <div className="flex items-center gap-2">
            <input
              type="number"
              min="0.1"
              max="1000"
              step="0.1"
              placeholder="e.g., 13"
              value={customParams ?? ''}
              onChange={(e) => {
                const val = parseFloat(e.target.value);
                setCustomParams(isNaN(val) ? null : val);
              }}
              className="w-24 px-3 py-2 bg-slate-700 border border-slate-600 rounded-lg text-slate-200 text-sm focus:outline-none focus:border-cyan-500"
            />
            <span className="text-slate-400">billion parameters</span>
          </div>
        </div>

        {/* Selected model summary */}
        <div className="mb-6 p-4 bg-slate-900/50 rounded-lg">
          <div className="text-center">
            <div className="text-2xl font-bold text-cyan-400">{modelName}</div>
            <div className="text-slate-400 text-sm">{params}B parameters</div>
          </div>
        </div>

        {/* Memory bars */}
        <div className="mb-6">
          <div className="text-sm text-slate-400 mb-3">Memory requirements by precision:</div>
          {memories.map(({ precision, memory }, i) => (
            <MemoryBar
              key={i}
              memory={memory}
              maxMemory={maxMemory}
              precision={precision}
              fitsGpu={fitsGpu(memory)}
            />
          ))}
        </div>

        {/* Key insight */}
        <div className="p-4 bg-gradient-to-r from-cyan-900/30 to-cyan-800/10 border border-cyan-700/50 rounded-lg">
          <div className="flex items-start gap-3">
            <span className="text-xl">💡</span>
            <div className="text-sm text-slate-300">
              <strong className="text-cyan-400">The takeaway:</strong> A {params}B model that needs{' '}
              <span className="text-slate-100">{formatMemory(memories[0].memory)}</span> in float32 can fit in just{' '}
              <span className="text-amber-400 font-semibold">{formatMemory(memories[3].memory)}</span> with int4 quantization—
              that's <span className="text-emerald-400 font-semibold">8× smaller</span>.
            </div>
          </div>
        </div>

        {/* GPU legend */}
        <div className="mt-6 pt-4 border-t border-slate-700">
          <div className="text-xs text-slate-500 mb-2">Common GPU VRAM:</div>
          <div className="flex flex-wrap gap-3 text-xs">
            {gpus.slice(0, 5).map((gpu, i) => (
              <span key={i} className="text-slate-400">
                {gpu.name}: {gpu.vram}GB
              </span>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}

export default MemoryCalculator;
