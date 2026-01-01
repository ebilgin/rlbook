/**
 * Interactive Quantization Demo
 *
 * A visual, hands-on exploration of how quantization works.
 * Users can:
 * - Drag a precision slider and watch values "snap" to a grid
 * - See before/after comparisons
 * - Understand the compression-error tradeoff intuitively
 */

import React, { useState, useMemo, useCallback } from 'react';

// Generate realistic weight distribution (normal, centered near 0)
function generateWeights(n: number, seed: number = 42): number[] {
  const weights: number[] = [];
  let s = seed;
  const random = () => {
    s = (s * 1103515245 + 12345) & 0x7fffffff;
    return s / 0x7fffffff;
  };

  // Box-Muller transform for normal distribution
  for (let i = 0; i < n; i++) {
    const u1 = random();
    const u2 = random();
    const z = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
    weights.push(z * 0.1);
  }

  return weights;
}

// Symmetric quantization
function quantize(weights: number[], bits: number) {
  const qmax = Math.pow(2, bits - 1) - 1;
  const maxVal = Math.max(...weights.map(Math.abs));
  const scale = maxVal / qmax;

  const quantized = weights.map(w => {
    const q = Math.round(w / scale);
    return Math.max(-qmax, Math.min(qmax, q));
  });

  const dequantized = quantized.map(q => q * scale);

  const errors = weights.map((w, i) => Math.pow(w - dequantized[i], 2));
  const mse = errors.reduce((a, b) => a + b, 0) / errors.length;
  const maxError = Math.max(...weights.map((w, i) => Math.abs(w - dequantized[i])));

  return {
    original: weights,
    quantized,
    dequantized,
    scale,
    mse,
    maxError,
    compressionRatio: 32 / bits,
    numLevels: Math.pow(2, bits),
  };
}

// Animated dot that shows snapping
function SnappingDot({ original, quantized, animate }: { original: number; quantized: number; animate: boolean }) {
  const range = 0.4; // Display range
  const origPos = ((original + range) / (2 * range)) * 100;
  const quantPos = ((quantized + range) / (2 * range)) * 100;

  return (
    <div className="relative h-3">
      {/* Original position (faded) */}
      <div
        className="absolute w-2 h-2 rounded-full bg-cyan-400/40 -translate-x-1/2 -translate-y-1/2 top-1/2"
        style={{ left: `${Math.max(0, Math.min(100, origPos))}%` }}
      />
      {/* Quantized position (solid) */}
      <div
        className={`absolute w-3 h-3 rounded-full bg-amber-400 -translate-x-1/2 -translate-y-1/2 top-1/2 ${
          animate ? 'transition-all duration-300' : ''
        }`}
        style={{ left: `${Math.max(0, Math.min(100, quantPos))}%` }}
      />
      {/* Connection line */}
      <div
        className="absolute h-0.5 bg-slate-600 top-1/2 -translate-y-1/2"
        style={{
          left: `${Math.min(origPos, quantPos)}%`,
          width: `${Math.abs(origPos - quantPos)}%`,
        }}
      />
    </div>
  );
}

// Visual quantization grid
function QuantizationGridVisual({ bits, scale }: { bits: number; scale: number }) {
  const qmax = Math.pow(2, bits - 1) - 1;
  const numLevels = Math.pow(2, bits);

  // Show up to 33 levels (for visual clarity)
  const maxVisible = 33;
  const step = numLevels <= maxVisible ? 1 : Math.ceil(numLevels / maxVisible);

  const levels = [];
  for (let i = -qmax; i <= qmax; i += step) {
    levels.push(i);
  }

  return (
    <div className="relative h-12 bg-slate-800/50 rounded-lg overflow-hidden">
      {/* Grid lines */}
      <div className="absolute inset-0 flex">
        {levels.map((level, i) => {
          const pos = ((level + qmax) / (2 * qmax)) * 100;
          return (
            <div
              key={i}
              className="absolute top-0 bottom-0 w-px bg-cyan-500/30"
              style={{ left: `${pos}%` }}
            />
          );
        })}
      </div>

      {/* Center line (zero) */}
      <div className="absolute left-1/2 top-0 bottom-0 w-0.5 bg-slate-400" />

      {/* Labels */}
      <div className="absolute bottom-0 left-0 right-0 flex justify-between px-2 text-xs text-slate-500">
        <span>-{(qmax * scale).toFixed(2)}</span>
        <span>0</span>
        <span>+{(qmax * scale).toFixed(2)}</span>
      </div>
    </div>
  );
}

// Mini histogram
function MiniHistogram({ values, color, height = 50 }: { values: number[]; color: 'cyan' | 'amber'; height?: number }) {
  const bins = 25;
  const min = -0.4;
  const max = 0.4;
  const range = max - min;
  const binWidth = range / bins;

  const counts = useMemo(() => {
    const c = new Array(bins).fill(0);
    values.forEach(v => {
      const idx = Math.floor((v - min) / binWidth);
      if (idx >= 0 && idx < bins) c[idx]++;
    });
    const maxCount = Math.max(...c);
    return c.map(x => x / maxCount);
  }, [values]);

  const barColor = color === 'cyan' ? 'bg-cyan-400' : 'bg-amber-400';

  return (
    <div className="flex items-end gap-px" style={{ height }}>
      {counts.map((h, i) => (
        <div
          key={i}
          className={`flex-1 ${barColor} rounded-t opacity-70`}
          style={{ height: `${h * 100}%` }}
        />
      ))}
    </div>
  );
}

export function QuantizationDemo() {
  const [bits, setBits] = useState(32);
  const [seed, setSeed] = useState(42);
  const [showDetails, setShowDetails] = useState(false);

  const numWeights = 200;
  const weights = useMemo(() => generateWeights(numWeights, seed), [seed]);
  const result = useMemo(() => quantize(weights, bits), [weights, bits]);

  // Sample 8 weights for the snapping visualization
  const sampleIndices = useMemo(() => {
    const indices = [];
    const step = Math.floor(numWeights / 8);
    for (let i = 0; i < 8; i++) {
      indices.push(i * step);
    }
    return indices;
  }, []);

  const regenerate = useCallback(() => setSeed(s => s + 1), []);

  // Dynamic insight message
  const insight = useMemo(() => {
    if (bits >= 32) {
      return {
        color: 'slate',
        icon: '📦',
        title: 'Full precision (baseline)',
        text: `32-bit float is the standard training format. No compression, no error—but uses the most memory.`,
      };
    } else if (bits >= 16) {
      return {
        color: 'emerald',
        icon: '✓',
        title: 'High precision',
        text: `${bits}-bit gives ${result.compressionRatio.toFixed(0)}× compression with negligible quality loss. Great for training and inference.`,
      };
    } else if (bits >= 8) {
      return {
        color: 'emerald',
        icon: '✓',
        title: 'Excellent precision',
        text: `${bits}-bit quantization offers ${result.compressionRatio.toFixed(0)}× compression. This is the standard for production inference.`,
      };
    } else if (bits >= 4) {
      return {
        color: 'amber',
        icon: '⚡',
        title: 'Good compression',
        text: `${bits}-bit quantization offers ${result.compressionRatio.toFixed(0)}× compression. Quality loss is usually acceptable for LLMs.`,
      };
    } else {
      return {
        color: 'red',
        icon: '⚠',
        title: 'Aggressive quantization',
        text: `${bits}-bit precision loses significant information. Only use with specialized training techniques.`,
      };
    }
  }, [bits, result.compressionRatio]);

  const insightColors: Record<string, string> = {
    slate: 'from-slate-800/40 to-slate-800/10 border-slate-600/50 text-slate-300',
    emerald: 'from-emerald-900/40 to-emerald-900/10 border-emerald-700/50 text-emerald-400',
    amber: 'from-amber-900/40 to-amber-900/10 border-amber-700/50 text-amber-400',
    red: 'from-red-900/40 to-red-900/10 border-red-700/50 text-red-400',
  };

  return (
    <div className="my-8 bg-slate-800/30 rounded-xl border border-slate-700 overflow-hidden">
      {/* Header */}
      <div className="bg-slate-800/50 px-6 py-4 border-b border-slate-700">
        <div className="flex items-center justify-between">
          <div>
            <h3 className="text-lg font-semibold text-slate-200">Try It: Quantization Explorer</h3>
            <p className="text-sm text-slate-400">Drag the slider to see how precision affects values</p>
          </div>
          <button
            onClick={regenerate}
            className="px-3 py-1.5 text-sm bg-slate-700 hover:bg-slate-600 text-slate-300 rounded-lg transition-colors"
          >
            New Weights
          </button>
        </div>
      </div>

      <div className="p-6">
        {/* Main slider - the hero element */}
        <div className="mb-8">
          <div className="flex items-center justify-between mb-3">
            <span className="text-slate-400">Precision</span>
            <span className="text-2xl font-bold text-cyan-400">{bits}-bit</span>
          </div>
          <input
            type="range"
            min="2"
            max="32"
            value={bits}
            onChange={(e) => setBits(parseInt(e.target.value))}
            className="w-full h-3 bg-slate-700 rounded-lg appearance-none cursor-pointer accent-cyan-400"
          />
          <div className="flex justify-between text-xs text-slate-500 mt-2">
            <span>2-bit (extreme)</span>
            <span>16-bit</span>
            <span>32-bit (baseline)</span>
          </div>
        </div>

        {/* Visual: Values snapping to grid */}
        <div className="mb-8">
          <div className="text-sm text-slate-400 mb-3">Watch values snap to the quantization grid:</div>
          <div className="bg-slate-900/50 rounded-lg p-4">
            <QuantizationGridVisual bits={bits} scale={result.scale} />
            <div className="mt-4 space-y-1">
              {sampleIndices.map((idx, i) => (
                <SnappingDot
                  key={i}
                  original={result.original[idx]}
                  quantized={result.dequantized[idx]}
                  animate={true}
                />
              ))}
            </div>
            <div className="mt-3 flex justify-between text-xs">
              <span className="flex items-center gap-1">
                <span className="w-2 h-2 rounded-full bg-cyan-400/40"></span>
                <span className="text-slate-500">Original</span>
              </span>
              <span className="flex items-center gap-1">
                <span className="w-3 h-3 rounded-full bg-amber-400"></span>
                <span className="text-slate-500">Quantized</span>
              </span>
            </div>
          </div>
        </div>

        {/* Side-by-side histograms */}
        <div className="grid md:grid-cols-2 gap-4 mb-8">
          <div className="bg-slate-900/50 rounded-lg p-4">
            <div className="text-sm text-slate-400 mb-2">Original Distribution (float32)</div>
            <MiniHistogram values={result.original} color="cyan" />
          </div>
          <div className="bg-slate-900/50 rounded-lg p-4">
            <div className="text-sm text-slate-400 mb-2">After {bits}-bit Quantization</div>
            <MiniHistogram values={result.dequantized} color="amber" />
          </div>
        </div>

        {/* Key metrics */}
        <div className="grid grid-cols-3 gap-4 mb-6">
          <div className="bg-slate-900/50 rounded-lg p-4 text-center">
            <div className="text-3xl font-bold text-emerald-400">{result.compressionRatio.toFixed(0)}×</div>
            <div className="text-xs text-slate-500 mt-1">Compression</div>
          </div>
          <div className="bg-slate-900/50 rounded-lg p-4 text-center">
            <div className="text-3xl font-bold text-cyan-400">{result.numLevels}</div>
            <div className="text-xs text-slate-500 mt-1">Quantization Levels</div>
          </div>
          <div className="bg-slate-900/50 rounded-lg p-4 text-center">
            <div className="text-3xl font-bold text-amber-400">{(Math.sqrt(result.mse) * 100).toFixed(1)}%</div>
            <div className="text-xs text-slate-500 mt-1">Avg Error</div>
          </div>
        </div>

        {/* Dynamic insight */}
        <div className={`rounded-lg p-4 bg-gradient-to-r border ${insightColors[insight.color]}`}>
          <div className="flex items-start gap-3">
            <span className="text-xl">{insight.icon}</span>
            <div>
              <div className="font-semibold">{insight.title}</div>
              <div className="text-sm text-slate-300 mt-1">{insight.text}</div>
            </div>
          </div>
        </div>

        {/* Expandable details */}
        <button
          onClick={() => setShowDetails(!showDetails)}
          className="mt-4 text-sm text-slate-400 hover:text-slate-300 flex items-center gap-1"
        >
          {showDetails ? '▼' : '▶'} {showDetails ? 'Hide' : 'Show'} sample values
        </button>

        {showDetails && (
          <div className="mt-3 bg-slate-900/50 rounded-lg p-4 overflow-x-auto">
            <table className="w-full text-xs font-mono">
              <thead>
                <tr className="text-slate-500 border-b border-slate-700">
                  <th className="text-left py-2 px-2">Original</th>
                  <th className="text-left py-2 px-2">Quantized Int</th>
                  <th className="text-left py-2 px-2">Dequantized</th>
                  <th className="text-left py-2 px-2">Error</th>
                </tr>
              </thead>
              <tbody>
                {sampleIndices.slice(0, 5).map((idx, i) => (
                  <tr key={i} className="text-slate-300 border-b border-slate-800">
                    <td className="py-1.5 px-2 text-cyan-400">{result.original[idx].toFixed(5)}</td>
                    <td className="py-1.5 px-2">{result.quantized[idx]}</td>
                    <td className="py-1.5 px-2">{result.dequantized[idx].toFixed(5)}</td>
                    <td className="py-1.5 px-2 text-amber-400">
                      {Math.abs(result.original[idx] - result.dequantized[idx]).toFixed(5)}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  );
}

export default QuantizationDemo;
