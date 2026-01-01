/**
 * Interactive Quantization Demo
 *
 * Visualizes how quantization affects values and shows the precision-accuracy tradeoff.
 * Users can:
 * - See how weights are distributed
 * - Adjust bit precision
 * - Watch values get "snapped" to a quantization grid
 * - See the resulting error
 */

import React, { useState, useMemo, useCallback } from 'react';

interface QuantizationResult {
  original: number[];
  quantized: number[];
  dequantized: number[];
  scale: number;
  mse: number;
  maxError: number;
  compressionRatio: number;
}

// Generate fake "weight" distribution (normal distribution)
function generateWeights(n: number, seed: number = 42): number[] {
  const weights: number[] = [];
  // Simple pseudo-random generator for reproducibility
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
    weights.push(z * 0.1); // Scale to typical weight magnitude
  }

  return weights;
}

// Symmetric quantization
function quantize(weights: number[], bits: number): QuantizationResult {
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

  const compressionRatio = 32 / bits;

  return {
    original: weights,
    quantized,
    dequantized,
    scale,
    mse,
    maxError,
    compressionRatio,
  };
}

// Mini histogram visualization
function Histogram({
  values,
  bins = 20,
  height = 60,
  color = 'cyan',
  label,
}: {
  values: number[];
  bins?: number;
  height?: number;
  color?: string;
  label: string;
}) {
  const histogram = useMemo(() => {
    const min = Math.min(...values);
    const max = Math.max(...values);
    const range = max - min || 1;
    const binWidth = range / bins;

    const counts = new Array(bins).fill(0);
    values.forEach(v => {
      const binIndex = Math.min(Math.floor((v - min) / binWidth), bins - 1);
      counts[binIndex]++;
    });

    const maxCount = Math.max(...counts);
    return counts.map(c => c / maxCount);
  }, [values, bins]);

  return (
    <div className="mb-4">
      <div className="text-xs text-slate-400 mb-1">{label}</div>
      <div className="flex items-end gap-px" style={{ height }}>
        {histogram.map((h, i) => (
          <div
            key={i}
            className="flex-1 rounded-t"
            style={{
              height: `${h * 100}%`,
              backgroundColor: color === 'cyan' ? 'rgb(34, 211, 238)' : 'rgb(251, 191, 36)',
              opacity: 0.7,
            }}
          />
        ))}
      </div>
    </div>
  );
}

// Quantization grid visualization
function QuantizationGrid({ bits, scale }: { bits: number; scale: number }) {
  const levels = Math.pow(2, bits);
  const qmax = Math.pow(2, bits - 1) - 1;

  // Show a subset of quantization levels
  const visibleLevels = Math.min(levels, 17);
  const step = Math.max(1, Math.floor(levels / visibleLevels));

  const markers = [];
  for (let i = -qmax; i <= qmax; i += step) {
    const value = i * scale;
    markers.push({ q: i, value });
  }

  return (
    <div className="my-4">
      <div className="text-xs text-slate-400 mb-2">
        Quantization Grid ({levels} levels)
      </div>
      <div className="relative h-8 bg-slate-700/50 rounded">
        {/* Zero marker */}
        <div
          className="absolute top-0 h-full w-0.5 bg-slate-400"
          style={{ left: '50%' }}
        />

        {/* Quantization levels */}
        {markers.map(({ q, value }, i) => {
          const position = ((q + qmax) / (2 * qmax)) * 100;
          return (
            <div
              key={i}
              className="absolute top-0 h-full w-0.5 bg-cyan-500/60"
              style={{ left: `${position}%` }}
              title={`q=${q}, v=${value.toFixed(4)}`}
            />
          );
        })}
      </div>
      <div className="flex justify-between text-xs text-slate-500 mt-1">
        <span>-{(qmax * scale).toFixed(3)}</span>
        <span>0</span>
        <span>+{(qmax * scale).toFixed(3)}</span>
      </div>
    </div>
  );
}

export function QuantizationDemo() {
  const [bits, setBits] = useState(8);
  const [numWeights, setNumWeights] = useState(100);
  const [seed, setSeed] = useState(42);

  const weights = useMemo(() => generateWeights(numWeights, seed), [numWeights, seed]);
  const result = useMemo(() => quantize(weights, bits), [weights, bits]);

  const regenerate = useCallback(() => {
    setSeed(s => s + 1);
  }, []);

  // Format for display
  const formatPercent = (n: number) => `${(n * 100).toFixed(2)}%`;
  const formatNumber = (n: number) => n.toFixed(6);

  return (
    <div className="my-8 p-6 bg-slate-800/50 rounded-xl border border-slate-700">
      {/* Header */}
      <div className="text-center mb-6">
        <h3 className="text-xl font-bold text-slate-200 mb-2">
          Interactive Quantization Demo
        </h3>
        <p className="text-slate-400 text-sm">
          See how reducing bit precision affects weight values
        </p>
      </div>

      {/* Controls */}
      <div className="grid md:grid-cols-2 gap-6 mb-6">
        {/* Bit precision slider */}
        <div>
          <label className="block text-sm text-slate-400 mb-2">
            Bit Precision: <span className="text-cyan-400 font-bold">{bits}-bit</span>
          </label>
          <input
            type="range"
            min="2"
            max="16"
            value={bits}
            onChange={(e) => setBits(parseInt(e.target.value))}
            className="w-full h-2 bg-slate-700 rounded-lg appearance-none cursor-pointer"
          />
          <div className="flex justify-between text-xs text-slate-500 mt-1">
            <span>2-bit</span>
            <span>8-bit</span>
            <span>16-bit</span>
          </div>
        </div>

        {/* Number of weights */}
        <div>
          <label className="block text-sm text-slate-400 mb-2">
            Number of Weights: <span className="text-cyan-400 font-bold">{numWeights}</span>
          </label>
          <input
            type="range"
            min="50"
            max="500"
            step="50"
            value={numWeights}
            onChange={(e) => setNumWeights(parseInt(e.target.value))}
            className="w-full h-2 bg-slate-700 rounded-lg appearance-none cursor-pointer"
          />
        </div>
      </div>

      {/* Regenerate button */}
      <div className="text-center mb-6">
        <button
          onClick={regenerate}
          className="px-4 py-2 bg-slate-600 text-white rounded-lg hover:bg-slate-500 transition-colors text-sm"
        >
          Regenerate Weights
        </button>
      </div>

      {/* Visualizations */}
      <div className="grid md:grid-cols-2 gap-6 mb-6">
        <div className="bg-slate-900/50 p-4 rounded-lg">
          <Histogram values={result.original} color="cyan" label="Original Weights (float32)" />
        </div>
        <div className="bg-slate-900/50 p-4 rounded-lg">
          <Histogram values={result.dequantized} color="amber" label={`Quantized (${bits}-bit)`} />
        </div>
      </div>

      {/* Quantization grid */}
      <div className="bg-slate-900/50 p-4 rounded-lg mb-6">
        <QuantizationGrid bits={bits} scale={result.scale} />
      </div>

      {/* Metrics */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <div className="bg-slate-900/50 p-4 rounded-lg text-center">
          <div className="text-2xl font-bold text-cyan-400">{bits}</div>
          <div className="text-xs text-slate-500">Bits/Weight</div>
        </div>
        <div className="bg-slate-900/50 p-4 rounded-lg text-center">
          <div className="text-2xl font-bold text-emerald-400">{result.compressionRatio.toFixed(1)}×</div>
          <div className="text-xs text-slate-500">Compression</div>
        </div>
        <div className="bg-slate-900/50 p-4 rounded-lg text-center">
          <div className="text-2xl font-bold text-amber-400">{formatNumber(Math.sqrt(result.mse))}</div>
          <div className="text-xs text-slate-500">RMSE</div>
        </div>
        <div className="bg-slate-900/50 p-4 rounded-lg text-center">
          <div className="text-2xl font-bold text-red-400">{formatNumber(result.maxError)}</div>
          <div className="text-xs text-slate-500">Max Error</div>
        </div>
      </div>

      {/* Insight based on current settings */}
      <div className="mt-6 p-4 bg-slate-900/30 rounded-lg">
        <div className="text-slate-300 text-sm">
          {bits >= 8 ? (
            <>
              <strong className="text-emerald-400">Great precision!</strong> With {bits}-bit quantization,
              the error is negligible for most applications. You get {result.compressionRatio.toFixed(1)}×
              compression with minimal quality loss.
            </>
          ) : bits >= 4 ? (
            <>
              <strong className="text-amber-400">Moderate precision.</strong> {bits}-bit quantization
              introduces noticeable error but often works well for neural networks.
              {result.compressionRatio.toFixed(1)}× compression can enable deployment on smaller devices.
            </>
          ) : (
            <>
              <strong className="text-red-400">Extreme quantization.</strong> {bits}-bit precision
              loses significant information. Only use this with quantization-aware training or
              when memory is extremely constrained.
            </>
          )}
        </div>
      </div>

      {/* Sample values */}
      <div className="mt-6">
        <div className="text-xs text-slate-400 mb-2">Sample Values (first 5):</div>
        <div className="overflow-x-auto">
          <table className="w-full text-xs">
            <thead>
              <tr className="text-slate-500">
                <th className="text-left p-1">Original</th>
                <th className="text-left p-1">Quantized</th>
                <th className="text-left p-1">Dequantized</th>
                <th className="text-left p-1">Error</th>
              </tr>
            </thead>
            <tbody>
              {result.original.slice(0, 5).map((orig, i) => (
                <tr key={i} className="text-slate-300">
                  <td className="p-1 font-mono">{orig.toFixed(6)}</td>
                  <td className="p-1 font-mono text-cyan-400">{result.quantized[i]}</td>
                  <td className="p-1 font-mono">{result.dequantized[i].toFixed(6)}</td>
                  <td className="p-1 font-mono text-amber-400">
                    {Math.abs(orig - result.dequantized[i]).toFixed(6)}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}

export default QuantizationDemo;
