/**
 * Policy Gradient Demo Interactive Component
 *
 * A simple 2-action environment that demonstrates:
 * - Policy probabilities as a bar chart
 * - How gradients shift probability mass
 * - The "reinforce good actions" intuition
 * - Reward history visualization
 *
 * Usage:
 *   <PolicyGradientDemo />
 */

import React, { useState, useCallback, useEffect, useRef } from 'react';

// Types
interface Action {
  id: number;
  name: string;
  emoji: string;
  trueReward: number; // Mean reward
  rewardStd: number;  // Reward standard deviation
}

interface EpisodeResult {
  action: number;
  reward: number;
  probability: number;
}

// Two actions with different expected rewards
const ACTIONS: Action[] = [
  { id: 0, name: 'Action A', emoji: '🔵', trueReward: 0.3, rewardStd: 0.5 },
  { id: 1, name: 'Action B', emoji: '🔴', trueReward: 0.7, rewardStd: 0.5 },
];

// Softmax function to convert logits to probabilities
function softmax(logits: number[]): number[] {
  const maxLogit = Math.max(...logits);
  const expLogits = logits.map(l => Math.exp(l - maxLogit));
  const sumExp = expLogits.reduce((a, b) => a + b, 0);
  return expLogits.map(e => e / sumExp);
}

// Sample from a categorical distribution
function sampleAction(probs: number[]): number {
  const r = Math.random();
  let cumsum = 0;
  for (let i = 0; i < probs.length; i++) {
    cumsum += probs[i];
    if (r < cumsum) return i;
  }
  return probs.length - 1;
}

// Sample reward from Gaussian (Box-Muller transform)
function sampleReward(mean: number, std: number): number {
  const u1 = Math.random();
  const u2 = Math.random();
  const z = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
  return mean + std * z;
}

export function PolicyGradientDemo() {
  // Policy parameters (logits for softmax)
  const [logits, setLogits] = useState<number[]>([0, 0]);
  const [learningRate, setLearningRate] = useState(0.3);
  const [history, setHistory] = useState<EpisodeResult[]>([]);
  const [isTraining, setIsTraining] = useState(false);
  const [baseline, setBaseline] = useState(0); // Running average baseline
  const [showTrueRewards, setShowTrueRewards] = useState(false);
  const [lastGradient, setLastGradient] = useState<number[] | null>(null);
  const [totalReward, setTotalReward] = useState(0);
  const intervalRef = useRef<NodeJS.Timeout | null>(null);

  const probs = softmax(logits);

  // REINFORCE step
  const trainStep = useCallback(() => {
    const currentProbs = softmax(logits);
    const action = sampleAction(currentProbs);
    const reward = sampleReward(ACTIONS[action].trueReward, ACTIONS[action].rewardStd);

    // Record episode
    const episode: EpisodeResult = {
      action,
      reward,
      probability: currentProbs[action],
    };

    // Compute advantage (reward - baseline)
    const advantage = reward - baseline;

    // Policy gradient: grad_theta log(pi(a|s)) * (R - b)
    // For softmax: grad_theta_i = (1{a=i} - pi(i)) * advantage
    const gradient = currentProbs.map((p, i) =>
      (i === action ? 1 - p : -p) * advantage
    );

    // Update logits
    const newLogits = logits.map((l, i) => l + learningRate * gradient[i]);

    // Update baseline (exponential moving average)
    const newBaseline = 0.9 * baseline + 0.1 * reward;

    setLogits(newLogits);
    setBaseline(newBaseline);
    setHistory(prev => [...prev.slice(-99), episode]); // Keep last 100
    setLastGradient(gradient);
    setTotalReward(prev => prev + reward);
  }, [logits, baseline, learningRate]);

  // Training loop
  useEffect(() => {
    if (isTraining) {
      intervalRef.current = setInterval(trainStep, 200);
    }
    return () => {
      if (intervalRef.current) clearInterval(intervalRef.current);
    };
  }, [isTraining, trainStep]);

  const reset = () => {
    setLogits([0, 0]);
    setHistory([]);
    setBaseline(0);
    setLastGradient(null);
    setTotalReward(0);
    setIsTraining(false);
  };

  // Calculate statistics
  const actionCounts = ACTIONS.map((_, i) =>
    history.filter(h => h.action === i).length
  );
  const actionRewards = ACTIONS.map((_, i) => {
    const actionHistory = history.filter(h => h.action === i);
    if (actionHistory.length === 0) return 0;
    return actionHistory.reduce((sum, h) => sum + h.reward, 0) / actionHistory.length;
  });

  // Recent rewards for mini chart
  const recentRewards = history.slice(-30);
  const maxReward = Math.max(...recentRewards.map(h => h.reward), 1);
  const minReward = Math.min(...recentRewards.map(h => h.reward), -1);
  const rewardRange = maxReward - minReward || 1;

  return (
    <div className="my-8 p-6 bg-slate-800/50 rounded-xl border border-slate-700">
      {/* Header */}
      <div className="text-center mb-6">
        <h3 className="text-xl font-bold text-slate-200 mb-2">Policy Gradient: REINFORCE</h3>
        <p className="text-slate-400 text-sm">
          Watch how the policy learns to favor higher-reward actions by shifting probability mass.
        </p>
      </div>

      {/* Policy Visualization */}
      <div className="mb-6">
        <div className="text-slate-300 text-sm font-medium text-center mb-3">
          Current Policy (Action Probabilities)
        </div>
        <div className="flex justify-center gap-8">
          {ACTIONS.map((action, i) => (
            <div key={action.id} className="flex flex-col items-center">
              {/* Probability bar */}
              <div className="relative w-16 h-40 bg-slate-900/50 rounded-lg overflow-hidden border border-slate-600">
                <div
                  className={`absolute bottom-0 w-full transition-all duration-300 ${
                    i === 0 ? 'bg-blue-500' : 'bg-red-500'
                  }`}
                  style={{ height: `${probs[i] * 100}%` }}
                />
                <div className="absolute inset-0 flex items-center justify-center">
                  <span className="text-white font-bold text-lg drop-shadow-lg">
                    {(probs[i] * 100).toFixed(0)}%
                  </span>
                </div>
              </div>
              {/* Action label */}
              <div className="mt-2 text-center">
                <span className="text-2xl">{action.emoji}</span>
                <div className="text-slate-400 text-xs">{action.name}</div>
              </div>
              {/* True reward (hidden by default) */}
              {showTrueRewards && (
                <div className="mt-1 text-xs text-violet-400">
                  True: {action.trueReward.toFixed(1)}
                </div>
              )}
              {/* Observed average */}
              <div className="mt-1 text-xs text-slate-500">
                Avg: {actionRewards[i].toFixed(2)}
              </div>
              {/* Count */}
              <div className="text-xs text-slate-600">
                n={actionCounts[i]}
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Gradient Visualization */}
      {lastGradient && (
        <div className="mb-6 p-4 bg-slate-900/30 rounded-lg">
          <div className="text-slate-300 text-sm font-medium text-center mb-2">
            Last Gradient Update
          </div>
          <div className="flex justify-center gap-4">
            {lastGradient.map((g, i) => (
              <div key={i} className="flex items-center gap-2">
                <span className="text-lg">{ACTIONS[i].emoji}</span>
                <div className={`text-sm font-mono ${g >= 0 ? 'text-emerald-400' : 'text-red-400'}`}>
                  {g >= 0 ? '+' : ''}{g.toFixed(3)}
                </div>
                <span className={`text-xs ${g >= 0 ? 'text-emerald-600' : 'text-red-600'}`}>
                  {g >= 0 ? '↑' : '↓'}
                </span>
              </div>
            ))}
          </div>
          <div className="text-slate-500 text-xs text-center mt-2">
            {lastGradient[0] > lastGradient[1]
              ? 'Increasing probability of Action A'
              : 'Increasing probability of Action B'}
          </div>
        </div>
      )}

      {/* Reward History Chart */}
      <div className="mb-6">
        <div className="text-slate-300 text-sm font-medium text-center mb-2">
          Reward History (last 30 episodes)
        </div>
        <div className="flex justify-center">
          <div className="relative w-80 h-20 bg-slate-900/50 rounded-lg border border-slate-600 overflow-hidden">
            {/* Zero line */}
            <div
              className="absolute left-0 right-0 h-px bg-slate-600"
              style={{ top: `${((maxReward - 0) / rewardRange) * 100}%` }}
            />
            {/* Reward bars */}
            <div className="absolute inset-0 flex items-end justify-end gap-px p-1">
              {recentRewards.map((episode, i) => {
                const height = Math.abs(episode.reward - 0) / rewardRange;
                const isPositive = episode.reward >= 0;
                const barColor = episode.action === 0 ? 'bg-blue-500' : 'bg-red-500';
                const zeroPosition = (maxReward - 0) / rewardRange;

                return (
                  <div
                    key={i}
                    className="relative flex-1"
                    style={{ height: '100%' }}
                  >
                    <div
                      className={`absolute w-full ${barColor} opacity-80`}
                      style={{
                        height: `${height * 100}%`,
                        [isPositive ? 'bottom' : 'top']: `${(isPositive ? 1 - zeroPosition : zeroPosition) * 100}%`,
                      }}
                    />
                  </div>
                );
              })}
            </div>
            {recentRewards.length === 0 && (
              <div className="absolute inset-0 flex items-center justify-center text-slate-500 text-sm">
                No data yet
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Metrics */}
      <div className="flex justify-center gap-6 mb-6">
        <div className="text-center">
          <div className="text-xl font-bold text-slate-200">{history.length}</div>
          <div className="text-slate-500 text-xs">Episodes</div>
        </div>
        <div className="text-center">
          <div className={`text-xl font-bold ${totalReward >= 0 ? 'text-emerald-400' : 'text-red-400'}`}>
            {totalReward.toFixed(1)}
          </div>
          <div className="text-slate-500 text-xs">Total Reward</div>
        </div>
        <div className="text-center">
          <div className="text-xl font-bold text-amber-400">
            {history.length > 0 ? (totalReward / history.length).toFixed(2) : '0.00'}
          </div>
          <div className="text-slate-500 text-xs">Avg Reward</div>
        </div>
        <div className="text-center">
          <div className="text-xl font-bold text-slate-300">{baseline.toFixed(2)}</div>
          <div className="text-slate-500 text-xs">Baseline</div>
        </div>
      </div>

      {/* Controls */}
      <div className="flex justify-center gap-2 flex-wrap mb-4">
        <button
          onClick={() => setIsTraining(!isTraining)}
          className={`px-4 py-2 rounded-lg font-medium transition-colors ${
            isTraining
              ? 'bg-red-600 text-white hover:bg-red-500'
              : 'bg-emerald-600 text-white hover:bg-emerald-500'
          }`}
        >
          {isTraining ? 'Pause' : 'Train'}
        </button>
        <button
          onClick={trainStep}
          disabled={isTraining}
          className="px-4 py-2 rounded-lg bg-blue-600 text-white font-medium hover:bg-blue-500 disabled:opacity-50 transition-colors"
        >
          Step
        </button>
        <button
          onClick={reset}
          className="px-4 py-2 rounded-lg bg-slate-600 text-white font-medium hover:bg-slate-500 transition-colors"
        >
          Reset
        </button>
        <button
          onClick={() => setShowTrueRewards(!showTrueRewards)}
          className={`px-4 py-2 rounded-lg font-medium transition-colors ${
            showTrueRewards
              ? 'bg-violet-600 text-white'
              : 'bg-slate-700 text-slate-300 hover:bg-slate-600'
          }`}
        >
          {showTrueRewards ? 'Hide True Values' : 'Reveal True Values'}
        </button>
      </div>

      {/* Learning rate slider */}
      <div className="flex justify-center items-center gap-3 mb-4">
        <label className="text-sm text-slate-400">Learning Rate:</label>
        <input
          type="range"
          min={0.05}
          max={1}
          step={0.05}
          value={learningRate}
          onChange={e => setLearningRate(Number(e.target.value))}
          className="w-32"
        />
        <span className="text-sm text-slate-300 w-12">{learningRate.toFixed(2)}</span>
      </div>

      {/* Explanation */}
      <div className="mt-4 p-4 bg-slate-900/30 rounded-lg">
        <div className="text-slate-400 text-sm">
          <strong className="text-slate-300">How it works:</strong>{' '}
          The REINFORCE algorithm samples an action from the current policy, observes the reward,
          and updates the policy to increase the probability of actions that led to higher-than-average
          rewards (the &quot;baseline&quot;). Actions with rewards above the baseline get reinforced;
          those below get suppressed. Over time, the policy converges to favor the higher-reward action.
        </div>
        {showTrueRewards && (
          <div className="mt-2 text-violet-300 text-sm">
            <strong>Ground truth:</strong> Action B (red) has a true expected reward of 0.7, while
            Action A (blue) has only 0.3. The policy should converge to strongly prefer Action B.
          </div>
        )}
      </div>
    </div>
  );
}

export default PolicyGradientDemo;
