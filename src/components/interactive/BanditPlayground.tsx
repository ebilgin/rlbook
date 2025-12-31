/**
 * Multi-Armed Bandit Playground
 *
 * An interactive demo for exploring multi-armed bandits:
 * - 5-10 slot machines with hidden probabilities
 * - Manual play mode for hands-on exploration
 * - Algorithm simulation mode comparing epsilon-greedy, UCB, and Thompson Sampling
 * - Regret tracking over time
 * - Visualization of algorithm behavior
 *
 * Usage:
 *   <BanditPlayground />
 */

import React, { useState, useCallback, useEffect, useRef } from 'react';

interface Arm {
  id: number;
  trueProbability: number;
  pulls: number;
  rewards: number;
  // For Thompson Sampling
  alpha: number; // Beta distribution parameter (successes + 1)
  beta: number;  // Beta distribution parameter (failures + 1)
}

interface HistoryEntry {
  step: number;
  armPulled: number;
  reward: number;
  regret: number;
  cumulativeRegret: number;
  algorithm: string;
}

type Algorithm = 'manual' | 'epsilon-greedy' | 'ucb' | 'thompson';

// Generate random arm probabilities
function generateArms(numArms: number): Arm[] {
  const arms: Arm[] = [];
  for (let i = 0; i < numArms; i++) {
    arms.push({
      id: i,
      trueProbability: Math.random() * 0.6 + 0.2, // Between 0.2 and 0.8
      pulls: 0,
      rewards: 0,
      alpha: 1,
      beta: 1,
    });
  }
  return arms;
}

// Sample from Beta distribution using the inverse transform method
function sampleBeta(alpha: number, beta: number): number {
  // Use a simple approximation for Beta sampling
  // This uses the gamma distribution relationship
  const gammaAlpha = gammaVariate(alpha);
  const gammaBeta = gammaVariate(beta);
  return gammaAlpha / (gammaAlpha + gammaBeta);
}

// Simple gamma variate using Marsaglia and Tsang's method
function gammaVariate(shape: number): number {
  if (shape < 1) {
    return gammaVariate(shape + 1) * Math.pow(Math.random(), 1 / shape);
  }
  const d = shape - 1 / 3;
  const c = 1 / Math.sqrt(9 * d);
  while (true) {
    let x: number;
    let v: number;
    do {
      x = normalVariate();
      v = 1 + c * x;
    } while (v <= 0);
    v = v * v * v;
    const u = Math.random();
    if (u < 1 - 0.0331 * (x * x) * (x * x)) return d * v;
    if (Math.log(u) < 0.5 * x * x + d * (1 - v + Math.log(v))) return d * v;
  }
}

// Box-Muller transform for normal distribution
function normalVariate(): number {
  const u1 = Math.random();
  const u2 = Math.random();
  return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
}

export function BanditPlayground() {
  const [numArms, setNumArms] = useState(7);
  const [arms, setArms] = useState<Arm[]>(() => generateArms(7));
  const [algorithm, setAlgorithm] = useState<Algorithm>('manual');
  const [epsilon, setEpsilon] = useState(0.1);
  const [ucbC, setUcbC] = useState(2);
  const [totalSteps, setTotalSteps] = useState(0);
  const [totalReward, setTotalReward] = useState(0);
  const [cumulativeRegret, setCumulativeRegret] = useState(0);
  const [history, setHistory] = useState<HistoryEntry[]>([]);
  const [isPlaying, setIsPlaying] = useState(false);
  const [speed, setSpeed] = useState(200);
  const [showTrueValues, setShowTrueValues] = useState(false);
  const [lastPulled, setLastPulled] = useState<number | null>(null);
  const [lastReward, setLastReward] = useState<number | null>(null);
  const intervalRef = useRef<NodeJS.Timeout | null>(null);

  const bestProbability = Math.max(...arms.map(a => a.trueProbability));

  // Get estimated value for an arm
  const getEstimate = (arm: Arm): number => {
    if (arm.pulls === 0) return 0;
    return arm.rewards / arm.pulls;
  };

  // Get UCB value for an arm
  const getUCB = (arm: Arm, t: number): number => {
    if (arm.pulls === 0) return Infinity;
    const estimate = getEstimate(arm);
    const exploration = ucbC * Math.sqrt(Math.log(t) / arm.pulls);
    return estimate + exploration;
  };

  // Select action based on algorithm
  const selectAction = useCallback((): number => {
    const t = totalSteps + 1;

    switch (algorithm) {
      case 'epsilon-greedy': {
        if (Math.random() < epsilon) {
          // Explore: random arm
          return Math.floor(Math.random() * arms.length);
        }
        // Exploit: best estimated arm
        let bestArm = 0;
        let bestValue = getEstimate(arms[0]);
        for (let i = 1; i < arms.length; i++) {
          const value = getEstimate(arms[i]);
          if (value > bestValue) {
            bestValue = value;
            bestArm = i;
          }
        }
        return bestArm;
      }

      case 'ucb': {
        let bestArm = 0;
        let bestUCB = getUCB(arms[0], t);
        for (let i = 1; i < arms.length; i++) {
          const ucb = getUCB(arms[i], t);
          if (ucb > bestUCB) {
            bestUCB = ucb;
            bestArm = i;
          }
        }
        return bestArm;
      }

      case 'thompson': {
        let bestArm = 0;
        let bestSample = sampleBeta(arms[0].alpha, arms[0].beta);
        for (let i = 1; i < arms.length; i++) {
          const sample = sampleBeta(arms[i].alpha, arms[i].beta);
          if (sample > bestSample) {
            bestSample = sample;
            bestArm = i;
          }
        }
        return bestArm;
      }

      default:
        return 0;
    }
  }, [algorithm, arms, epsilon, totalSteps, ucbC]);

  // Pull an arm
  const pullArm = useCallback((armIndex: number) => {
    const arm = arms[armIndex];
    const reward = Math.random() < arm.trueProbability ? 1 : 0;
    const instantRegret = bestProbability - arm.trueProbability;
    const newCumulativeRegret = cumulativeRegret + instantRegret;

    setArms(prev => prev.map((a, i) =>
      i === armIndex
        ? {
            ...a,
            pulls: a.pulls + 1,
            rewards: a.rewards + reward,
            alpha: a.alpha + reward,
            beta: a.beta + (1 - reward),
          }
        : a
    ));

    setTotalSteps(prev => prev + 1);
    setTotalReward(prev => prev + reward);
    setCumulativeRegret(newCumulativeRegret);
    setLastPulled(armIndex);
    setLastReward(reward);

    setHistory(prev => [...prev, {
      step: totalSteps + 1,
      armPulled: armIndex,
      reward,
      regret: instantRegret,
      cumulativeRegret: newCumulativeRegret,
      algorithm: algorithm === 'manual' ? 'manual' : algorithm,
    }]);
  }, [arms, bestProbability, cumulativeRegret, totalSteps, algorithm]);

  // Auto-play step
  const autoStep = useCallback(() => {
    if (algorithm === 'manual') return;
    const action = selectAction();
    pullArm(action);
  }, [algorithm, selectAction, pullArm]);

  // Auto-play loop
  useEffect(() => {
    if (isPlaying && algorithm !== 'manual') {
      intervalRef.current = setInterval(autoStep, speed);
    }
    return () => {
      if (intervalRef.current) clearInterval(intervalRef.current);
    };
  }, [isPlaying, algorithm, speed, autoStep]);

  // Reset
  const reset = () => {
    const newArms = generateArms(numArms);
    setArms(newArms);
    setTotalSteps(0);
    setTotalReward(0);
    setCumulativeRegret(0);
    setHistory([]);
    setIsPlaying(false);
    setLastPulled(null);
    setLastReward(null);
  };

  // Update arm count
  const handleArmCountChange = (newCount: number) => {
    setNumArms(newCount);
    setArms(generateArms(newCount));
    setTotalSteps(0);
    setTotalReward(0);
    setCumulativeRegret(0);
    setHistory([]);
    setIsPlaying(false);
    setLastPulled(null);
    setLastReward(null);
  };

  // Get best arm index
  const getBestArmIndex = () => {
    let bestIdx = 0;
    let bestProb = arms[0].trueProbability;
    for (let i = 1; i < arms.length; i++) {
      if (arms[i].trueProbability > bestProb) {
        bestProb = arms[i].trueProbability;
        bestIdx = i;
      }
    }
    return bestIdx;
  };

  const bestArmIdx = getBestArmIndex();

  return (
    <div className="my-8 p-6 bg-slate-800/50 rounded-xl border border-slate-700">
      {/* Header */}
      <div className="text-center mb-6">
        <h3 className="text-xl font-bold text-slate-200 mb-2">Multi-Armed Bandit Playground</h3>
        <p className="text-slate-400 text-sm">
          Explore the exploration-exploitation tradeoff. Try manual play or compare algorithms.
        </p>
      </div>

      {/* Configuration */}
      <div className="mb-6 p-4 bg-slate-900/30 rounded-lg">
        <div className="flex flex-wrap gap-4 items-center justify-center">
          <label className="flex items-center gap-2 text-sm text-slate-300">
            Arms:
            <select
              value={numArms}
              onChange={e => handleArmCountChange(Number(e.target.value))}
              className="px-2 py-1 rounded bg-slate-700 text-slate-200 border border-slate-600"
            >
              {[5, 6, 7, 8, 9, 10].map(n => (
                <option key={n} value={n}>{n}</option>
              ))}
            </select>
          </label>

          <label className="flex items-center gap-2 text-sm text-slate-300">
            Mode:
            <select
              value={algorithm}
              onChange={e => {
                setAlgorithm(e.target.value as Algorithm);
                setIsPlaying(false);
              }}
              className="px-2 py-1 rounded bg-slate-700 text-slate-200 border border-slate-600"
            >
              <option value="manual">Manual Play</option>
              <option value="epsilon-greedy">Epsilon-Greedy</option>
              <option value="ucb">UCB</option>
              <option value="thompson">Thompson Sampling</option>
            </select>
          </label>

          {algorithm === 'epsilon-greedy' && (
            <label className="flex items-center gap-2 text-sm text-slate-300">
              Epsilon:
              <input
                type="range"
                min={0}
                max={0.5}
                step={0.05}
                value={epsilon}
                onChange={e => setEpsilon(Number(e.target.value))}
                className="w-20"
              />
              <span className="w-10">{epsilon.toFixed(2)}</span>
            </label>
          )}

          {algorithm === 'ucb' && (
            <label className="flex items-center gap-2 text-sm text-slate-300">
              c:
              <input
                type="range"
                min={0.5}
                max={4}
                step={0.5}
                value={ucbC}
                onChange={e => setUcbC(Number(e.target.value))}
                className="w-20"
              />
              <span className="w-8">{ucbC.toFixed(1)}</span>
            </label>
          )}

          {algorithm !== 'manual' && (
            <label className="flex items-center gap-2 text-sm text-slate-300">
              Speed:
              <input
                type="range"
                min={50}
                max={500}
                step={50}
                value={speed}
                onChange={e => setSpeed(Number(e.target.value))}
                className="w-20"
              />
            </label>
          )}
        </div>
      </div>

      {/* Slot Machines */}
      <div className="mb-6 overflow-x-auto">
        <div className="flex gap-2 justify-center min-w-fit px-2">
          {arms.map((arm, index) => {
            const estimate = getEstimate(arm);
            const isLastPulledArm = lastPulled === index;
            const isBest = showTrueValues && index === bestArmIdx;

            return (
              <div
                key={arm.id}
                className={`p-3 rounded-lg border-2 transition-all duration-300 w-20 flex-shrink-0 ${
                  isLastPulledArm
                    ? lastReward === 1
                      ? 'border-emerald-500 bg-emerald-900/20'
                      : 'border-red-500 bg-red-900/20'
                    : isBest
                      ? 'border-amber-500 bg-amber-900/10'
                      : 'border-slate-600 bg-slate-900/30'
                }`}
              >
                {/* Machine visual */}
                <div className="text-center mb-2">
                  <div className="text-2xl mb-1">🎰</div>
                  <div className="text-slate-400 text-xs">Arm {index + 1}</div>
                </div>

                {/* Stats */}
                <div className="text-xs space-y-1">
                  <div className="flex justify-between">
                    <span className="text-slate-500">n:</span>
                    <span className="text-slate-300">{arm.pulls}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-slate-500">Est:</span>
                    <span className="text-amber-400">
                      {arm.pulls > 0 ? `${(estimate * 100).toFixed(0)}%` : '?'}
                    </span>
                  </div>
                  {showTrueValues && (
                    <div className="flex justify-between border-t border-slate-700 pt-1">
                      <span className="text-slate-500">True:</span>
                      <span className="text-violet-400">
                        {(arm.trueProbability * 100).toFixed(0)}%
                      </span>
                    </div>
                  )}
                </div>

                {/* Estimate bar */}
                <div className="h-1.5 bg-slate-700 rounded-full overflow-hidden mt-2 mb-2">
                  <div
                    className="h-full bg-amber-500 transition-all duration-300"
                    style={{ width: `${estimate * 100}%` }}
                  />
                </div>

                {/* Pull button (manual mode only) */}
                {algorithm === 'manual' && (
                  <button
                    onClick={() => pullArm(index)}
                    className="w-full py-1.5 px-2 rounded bg-blue-600 text-white text-xs font-medium hover:bg-blue-500 transition-colors"
                  >
                    Pull
                  </button>
                )}
              </div>
            );
          })}
        </div>
      </div>

      {/* Last Result */}
      <div className="mb-4 min-h-[50px]">
        {lastPulled !== null && lastReward !== null ? (
          <div className={`text-center p-3 rounded-lg ${
            lastReward === 1
              ? 'bg-emerald-900/30 text-emerald-400'
              : 'bg-red-900/30 text-red-400'
          }`}>
            <span className="text-xl mr-2">{lastReward === 1 ? '🎉' : '😞'}</span>
            Arm {lastPulled + 1}: {lastReward === 1 ? 'WIN (+1)' : 'No win (0)'}
            {algorithm !== 'manual' && (
              <span className="text-slate-400 ml-2">
                ({algorithm === 'epsilon-greedy' ? 'ε-Greedy' : algorithm === 'ucb' ? 'UCB' : 'Thompson'})
              </span>
            )}
          </div>
        ) : (
          <div className="text-center p-3 rounded-lg bg-slate-900/30 text-slate-500">
            {algorithm === 'manual'
              ? 'Click an arm to pull it!'
              : 'Press Play to run the algorithm'}
          </div>
        )}
      </div>

      {/* Metrics */}
      <div className="flex justify-center gap-6 mb-6 flex-wrap">
        <div className="text-center">
          <div className="text-2xl font-bold text-slate-200">{totalSteps}</div>
          <div className="text-slate-500 text-sm">Steps</div>
        </div>
        <div className="text-center">
          <div className="text-2xl font-bold text-emerald-400">{totalReward}</div>
          <div className="text-slate-500 text-sm">Rewards</div>
        </div>
        <div className="text-center">
          <div className="text-2xl font-bold text-amber-400">
            {totalSteps > 0 ? `${((totalReward / totalSteps) * 100).toFixed(0)}%` : '-'}
          </div>
          <div className="text-slate-500 text-sm">Win Rate</div>
        </div>
        <div className="text-center">
          <div className="text-2xl font-bold text-red-400">{cumulativeRegret.toFixed(2)}</div>
          <div className="text-slate-500 text-sm">Regret</div>
        </div>
        {showTrueValues && (
          <div className="text-center">
            <div className="text-2xl font-bold text-violet-400">
              {(bestProbability * 100).toFixed(0)}%
            </div>
            <div className="text-slate-500 text-sm">Best Arm</div>
          </div>
        )}
      </div>

      {/* Regret Plot (simple bar visualization) */}
      {history.length > 0 && (
        <div className="mb-6 p-4 bg-slate-900/30 rounded-lg">
          <div className="text-slate-300 text-sm font-medium mb-2">Cumulative Regret Over Time</div>
          <div className="h-16 flex items-end gap-px">
            {history.slice(-100).map((entry, i) => {
              const maxRegret = Math.max(...history.slice(-100).map(h => h.cumulativeRegret), 1);
              const height = (entry.cumulativeRegret / maxRegret) * 100;
              return (
                <div
                  key={i}
                  className="flex-1 bg-red-500/60 rounded-t-sm transition-all"
                  style={{ height: `${Math.max(height, 2)}%` }}
                  title={`Step ${entry.step}: Regret ${entry.cumulativeRegret.toFixed(2)}`}
                />
              );
            })}
          </div>
          <div className="flex justify-between text-xs text-slate-500 mt-1">
            <span>{Math.max(0, history.length - 100)}</span>
            <span>{history.length}</span>
          </div>
        </div>
      )}

      {/* Controls */}
      <div className="flex justify-center gap-3 flex-wrap">
        {algorithm !== 'manual' && (
          <>
            <button
              onClick={autoStep}
              className="px-4 py-2 rounded-lg bg-blue-600 text-white font-medium hover:bg-blue-500 transition-colors"
            >
              Step
            </button>
            <button
              onClick={() => setIsPlaying(!isPlaying)}
              className="px-4 py-2 rounded-lg bg-slate-600 text-white font-medium hover:bg-slate-500 transition-colors"
            >
              {isPlaying ? 'Pause' : 'Play'}
            </button>
          </>
        )}
        <button
          onClick={reset}
          className="px-4 py-2 rounded-lg bg-slate-600 text-white font-medium hover:bg-slate-500 transition-colors"
        >
          Reset
        </button>
        <button
          onClick={() => setShowTrueValues(!showTrueValues)}
          className={`px-4 py-2 rounded-lg font-medium transition-colors ${
            showTrueValues
              ? 'bg-violet-600 text-white hover:bg-violet-500'
              : 'bg-slate-600 text-white hover:bg-slate-500'
          }`}
        >
          {showTrueValues ? 'Hide True Values' : 'Show True Values'}
        </button>
      </div>

      {/* Algorithm Explanation */}
      <div className="mt-6 p-4 bg-slate-900/30 rounded-lg">
        <div className="text-slate-400 text-sm">
          {algorithm === 'manual' && (
            <>
              <strong className="text-slate-300">Manual Mode:</strong> You decide which arm to pull.
              Can you find the best arm while minimizing regret?
            </>
          )}
          {algorithm === 'epsilon-greedy' && (
            <>
              <strong className="text-slate-300">Epsilon-Greedy:</strong> With probability epsilon,
              explore a random arm. Otherwise, exploit the arm with the highest estimated value.
              Simple but effective!
            </>
          )}
          {algorithm === 'ucb' && (
            <>
              <strong className="text-slate-300">Upper Confidence Bound (UCB):</strong> Balance
              exploitation and exploration using confidence intervals. Arms with high uncertainty
              get a bonus, encouraging exploration of under-sampled arms.
            </>
          )}
          {algorithm === 'thompson' && (
            <>
              <strong className="text-slate-300">Thompson Sampling:</strong> Maintain a probability
              distribution over each arms true value. Sample from these distributions and pick the
              arm with the highest sample. Naturally balances exploration and exploitation!
            </>
          )}
        </div>
      </div>
    </div>
  );
}

export default BanditPlayground;
