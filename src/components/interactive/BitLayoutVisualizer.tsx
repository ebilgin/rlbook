/**
 * Interactive Bit Layout Visualizer
 *
 * Shows how float32, float16, and bfloat16 represent numbers.
 * Users can:
 * - See the bit layout (sign, exponent, mantissa)
 * - Enter a number and see its binary representation
 * - Compare formats side-by-side
 */

import React, { useState, useMemo } from 'react';

interface FormatSpec {
  name: string;
  totalBits: number;
  signBits: number;
  exponentBits: number;
  mantissaBits: number;
  exponentBias: number;
  color: string;
}

const formats: Record<string, FormatSpec> = {
  float32: {
    name: 'float32',
    totalBits: 32,
    signBits: 1,
    exponentBits: 8,
    mantissaBits: 23,
    exponentBias: 127,
    color: 'cyan',
  },
  float16: {
    name: 'float16',
    totalBits: 16,
    signBits: 1,
    exponentBits: 5,
    mantissaBits: 10,
    exponentBias: 15,
    color: 'violet',
  },
  bfloat16: {
    name: 'bfloat16',
    totalBits: 16,
    signBits: 1,
    exponentBits: 8,
    mantissaBits: 7,
    exponentBias: 127,
    color: 'amber',
  },
};

// Convert a float to its bit representation (simplified, for visualization)
function floatToBits(value: number, format: FormatSpec): { sign: string; exponent: string; mantissa: string } {
  if (value === 0) {
    return {
      sign: '0',
      exponent: '0'.repeat(format.exponentBits),
      mantissa: '0'.repeat(format.mantissaBits),
    };
  }

  const sign = value < 0 ? '1' : '0';
  const absValue = Math.abs(value);

  // Calculate exponent and mantissa
  let exp = Math.floor(Math.log2(absValue));
  let mantissa = absValue / Math.pow(2, exp) - 1; // Normalized: 1.xxxxx

  // Bias the exponent
  const biasedExp = exp + format.exponentBias;

  // Clamp exponent to valid range
  const maxExp = Math.pow(2, format.exponentBits) - 1;
  const clampedExp = Math.max(0, Math.min(maxExp - 1, biasedExp));

  // Convert to binary strings
  const exponentBinary = clampedExp.toString(2).padStart(format.exponentBits, '0');

  // Mantissa: multiply by 2^mantissaBits and take integer part
  const mantissaInt = Math.floor(mantissa * Math.pow(2, format.mantissaBits));
  const mantissaBinary = mantissaInt.toString(2).padStart(format.mantissaBits, '0');

  return {
    sign,
    exponent: exponentBinary,
    mantissa: mantissaBinary,
  };
}

function BitBox({
  bit,
  type,
  index,
}: {
  bit: string;
  type: 'sign' | 'exponent' | 'mantissa';
  index: number;
}) {
  const colors = {
    sign: 'bg-red-500/70 border-red-400',
    exponent: 'bg-emerald-500/70 border-emerald-400',
    mantissa: 'bg-blue-500/70 border-blue-400',
  };

  return (
    <div
      className={`w-5 h-7 flex items-center justify-center text-xs font-mono font-bold border rounded ${colors[type]} text-white`}
    >
      {bit}
    </div>
  );
}

function FormatVisualizer({
  format,
  value,
  showValue = true,
}: {
  format: FormatSpec;
  value: number;
  showValue?: boolean;
}) {
  const bits = useMemo(() => floatToBits(value, format), [value, format]);

  return (
    <div className="bg-slate-900/50 rounded-lg p-4">
      <div className="flex items-center justify-between mb-3">
        <span className="font-semibold text-slate-200">{format.name}</span>
        <span className="text-sm text-slate-400">{format.totalBits} bits total</span>
      </div>

      {/* Bit layout */}
      <div className="flex gap-1 flex-wrap mb-3">
        {/* Sign bit */}
        <BitBox bit={bits.sign} type="sign" index={0} />

        {/* Exponent bits */}
        {bits.exponent.split('').map((b, i) => (
          <BitBox key={`exp-${i}`} bit={b} type="exponent" index={i} />
        ))}

        {/* Mantissa bits */}
        {bits.mantissa.split('').map((b, i) => (
          <BitBox key={`man-${i}`} bit={b} type="mantissa" index={i} />
        ))}
      </div>

      {/* Legend */}
      <div className="flex gap-4 text-xs mb-3">
        <span className="flex items-center gap-1">
          <span className="w-3 h-3 rounded bg-red-500/70"></span>
          <span className="text-slate-400">Sign ({format.signBits})</span>
        </span>
        <span className="flex items-center gap-1">
          <span className="w-3 h-3 rounded bg-emerald-500/70"></span>
          <span className="text-slate-400">Exponent ({format.exponentBits})</span>
        </span>
        <span className="flex items-center gap-1">
          <span className="w-3 h-3 rounded bg-blue-500/70"></span>
          <span className="text-slate-400">Mantissa ({format.mantissaBits})</span>
        </span>
      </div>

      {/* Characteristics */}
      <div className="grid grid-cols-2 gap-2 text-xs">
        <div className="bg-slate-800/50 rounded p-2">
          <span className="text-slate-500">Range:</span>
          <span className="text-slate-300 ml-1">
            ±{Math.pow(2, Math.pow(2, format.exponentBits - 1)).toExponential(0)}
          </span>
        </div>
        <div className="bg-slate-800/50 rounded p-2">
          <span className="text-slate-500">Precision:</span>
          <span className="text-slate-300 ml-1">
            ~{Math.ceil(format.mantissaBits * Math.log10(2))} decimal digits
          </span>
        </div>
      </div>
    </div>
  );
}

export function BitLayoutVisualizer() {
  const [inputValue, setInputValue] = useState('3.14');
  const value = parseFloat(inputValue) || 0;
  const [selectedFormats, setSelectedFormats] = useState<string[]>(['float32', 'float16', 'bfloat16']);

  const toggleFormat = (name: string) => {
    setSelectedFormats(prev =>
      prev.includes(name) ? prev.filter(f => f !== name) : [...prev, name]
    );
  };

  return (
    <div className="my-8 bg-slate-800/30 rounded-xl border border-slate-700 overflow-hidden">
      {/* Header */}
      <div className="bg-slate-800/50 px-6 py-4 border-b border-slate-700">
        <h3 className="text-lg font-semibold text-slate-200">Float Format Explorer</h3>
        <p className="text-sm text-slate-400">See how numbers are stored in different formats</p>
      </div>

      <div className="p-6">
        {/* Value input */}
        <div className="mb-6">
          <label className="block text-sm text-slate-400 mb-2">Enter a number:</label>
          <input
            type="text"
            value={inputValue}
            onChange={(e) => setInputValue(e.target.value)}
            className="w-full max-w-xs px-4 py-2 bg-slate-700 border border-slate-600 rounded-lg text-slate-200 text-lg font-mono focus:outline-none focus:border-cyan-500"
            placeholder="e.g., 3.14"
          />
          <div className="mt-2 text-sm text-slate-500">
            Parsed value: <span className="text-cyan-400 font-mono">{value}</span>
          </div>
        </div>

        {/* Quick presets */}
        <div className="mb-6">
          <div className="text-sm text-slate-400 mb-2">Try these values:</div>
          <div className="flex flex-wrap gap-2">
            {['1.0', '0.1', '3.14159', '1000', '0.0001', '-42.5'].map(v => (
              <button
                key={v}
                onClick={() => setInputValue(v)}
                className="px-3 py-1 bg-slate-700 hover:bg-slate-600 text-slate-300 text-sm rounded-lg transition-colors font-mono"
              >
                {v}
              </button>
            ))}
          </div>
        </div>

        {/* Format toggles */}
        <div className="mb-6">
          <div className="text-sm text-slate-400 mb-2">Compare formats:</div>
          <div className="flex gap-2">
            {Object.values(formats).map(f => (
              <button
                key={f.name}
                onClick={() => toggleFormat(f.name)}
                className={`px-3 py-1.5 rounded-lg text-sm transition-colors ${
                  selectedFormats.includes(f.name)
                    ? 'bg-cyan-600 text-white'
                    : 'bg-slate-700 text-slate-400 hover:bg-slate-600'
                }`}
              >
                {f.name}
              </button>
            ))}
          </div>
        </div>

        {/* Format visualizations */}
        <div className="space-y-4">
          {selectedFormats.map(name => (
            <FormatVisualizer key={name} format={formats[name]} value={value} />
          ))}
        </div>

        {/* Comparison insight */}
        <div className="mt-6 p-4 bg-gradient-to-r from-violet-900/30 to-violet-800/10 border border-violet-700/50 rounded-lg">
          <div className="text-sm text-slate-300">
            <strong className="text-violet-400">Key difference:</strong>
            <ul className="mt-2 space-y-1 text-slate-400">
              <li>• <strong className="text-slate-300">float16</strong>: More mantissa bits = better precision for small values</li>
              <li>• <strong className="text-slate-300">bfloat16</strong>: More exponent bits = larger range, used for training</li>
              <li>• <strong className="text-slate-300">float32</strong>: Full precision baseline</li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  );
}

export default BitLayoutVisualizer;
