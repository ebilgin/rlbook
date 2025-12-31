/**
 * TD Learning vs Monte Carlo Demo
 *
 * A side-by-side comparison of TD(0) and Monte Carlo learning:
 * - Random Walk environment (states 1-5, terminal at edges)
 * - Show value estimates updating in real-time
 * - Visualize the difference in update timing (TD updates every step, MC at episode end)
 * - Plot learning curves showing convergence
 *
 * Usage:
 *   <TDLearningDemo />
 */

import React, { useState, useCallback, useEffect, useRef } from 'react';

const NUM_STATES = 5;
const TRUE_VALUES = [1/6, 2/6, 3/6, 4/6, 5/6]; // True values for states 0-4

interface LearnerState {
  values: number[];
  episodes: number;
  steps: number;
  history: number[][]; // History of value estimates per episode
  currentPosition: number;
  episodeRewards: number[];
  episodeStates: number[];
}

interface EpisodeStep {
  state: number;
  reward: number;
  nextState: number | null;
}

function initLearner(): LearnerState {
  return {
    values: [0.5, 0.5, 0.5, 0.5, 0.5], // Initialize to 0.5
    episodes: 0,
    steps: 0,
    history: [[0.5, 0.5, 0.5, 0.5, 0.5]],
    currentPosition: 2, // Start in middle
    episodeRewards: [],
    episodeStates: [2],
  };
}

export function TDLearningDemo() {
  const [tdLearner, setTdLearner] = useState<LearnerState>(initLearner);
  const [mcLearner, setMcLearner] = useState<LearnerState>(initLearner);
  const [alpha, setAlpha] = useState(0.1);
  const [gamma, setGamma] = useState(1.0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [speed, setSpeed] = useState(300);
  const [episodeInProgress, setEpisodeInProgress] = useState(false);
  const [lastAction, setLastAction] = useState<'left' | 'right' | null>(null);
  const intervalRef = useRef<NodeJS.Timeout | null>(null);

  // Take a random step: 50% left, 50% right
  const takeStep = useCallback((position: number): { nextPosition: number | null; reward: number; action: 'left' | 'right' } => {
    const goRight = Math.random() < 0.5;
    const action = goRight ? 'right' : 'left';

    if (goRight) {
      if (position === NUM_STATES - 1) {
        // Reached right terminal (reward = 1)
        return { nextPosition: null, reward: 1, action };
      }
      return { nextPosition: position + 1, reward: 0, action };
    } else {
      if (position === 0) {
        // Reached left terminal (reward = 0)
        return { nextPosition: null, reward: 0, action };
      }
      return { nextPosition: position - 1, reward: 0, action };
    }
  }, []);

  // TD(0) update
  const tdUpdate = useCallback((state: number, reward: number, nextState: number | null, values: number[], alpha: number, gamma: number): number[] => {
    const newValues = [...values];
    const nextValue = nextState !== null ? values[nextState] : 0;
    const tdTarget = reward + gamma * nextValue;
    const tdError = tdTarget - values[state];
    newValues[state] = values[state] + alpha * tdError;
    return newValues;
  }, []);

  // Monte Carlo update (at end of episode)
  const mcUpdate = useCallback((states: number[], rewards: number[], values: number[], alpha: number, gamma: number): number[] => {
    const newValues = [...values];

    // Calculate returns for each state visit
    let G = 0;
    for (let t = states.length - 1; t >= 0; t--) {
      G = rewards[t] + gamma * G;
      const state = states[t];
      newValues[state] = newValues[state] + alpha * (G - newValues[state]);
    }

    return newValues;
  }, []);

  // Run one step
  const runStep = useCallback(() => {
    // Get current position (same for both, synchronized)
    const currentPos = tdLearner.currentPosition;

    // Take a step
    const { nextPosition, reward, action } = takeStep(currentPos);
    setLastAction(action);

    const episodeEnded = nextPosition === null;

    // Update TD learner (updates immediately)
    setTdLearner(prev => {
      const newValues = tdUpdate(currentPos, reward, nextPosition, prev.values, alpha, gamma);
      const newHistory = episodeEnded
        ? [...prev.history, [...newValues]]
        : prev.history;

      return {
        ...prev,
        values: newValues,
        steps: prev.steps + 1,
        episodes: episodeEnded ? prev.episodes + 1 : prev.episodes,
        history: newHistory,
        currentPosition: episodeEnded ? 2 : nextPosition!,
        episodeRewards: episodeEnded ? [] : [...prev.episodeRewards, reward],
        episodeStates: episodeEnded ? [2] : [...prev.episodeStates, nextPosition!],
      };
    });

    // Update MC learner (only updates at episode end)
    setMcLearner(prev => {
      const newEpisodeRewards = [...prev.episodeRewards, reward];
      const newEpisodeStates = [...prev.episodeStates];

      if (episodeEnded) {
        // Episode ended - do MC update
        const newValues = mcUpdate(newEpisodeStates, newEpisodeRewards, prev.values, alpha, gamma);
        return {
          ...prev,
          values: newValues,
          steps: prev.steps + 1,
          episodes: prev.episodes + 1,
          history: [...prev.history, [...newValues]],
          currentPosition: 2,
          episodeRewards: [],
          episodeStates: [2],
        };
      } else {
        // Episode continues - just record state/reward
        return {
          ...prev,
          steps: prev.steps + 1,
          currentPosition: nextPosition!,
          episodeRewards: newEpisodeRewards,
          episodeStates: [...newEpisodeStates, nextPosition!],
        };
      }
    });

    setEpisodeInProgress(!episodeEnded);

    if (episodeEnded) {
      setLastAction(null);
    }
  }, [tdLearner.currentPosition, takeStep, tdUpdate, mcUpdate, alpha, gamma]);

  // Run one complete episode
  const runEpisode = useCallback(() => {
    // Run steps until episode ends
    let pos = tdLearner.currentPosition;
    let tdValues = [...tdLearner.values];
    let mcStates = [...mcLearner.episodeStates];
    let mcRewards = [...mcLearner.episodeRewards];
    let stepsTaken = 0;

    while (pos !== null && stepsTaken < 100) {
      const { nextPosition, reward, action } = takeStep(pos);
      stepsTaken++;

      // TD update
      tdValues = tdUpdate(pos, reward, nextPosition, tdValues, alpha, gamma);

      // MC accumulation
      mcRewards.push(reward);
      if (nextPosition !== null) {
        mcStates.push(nextPosition);
      }

      pos = nextPosition as number;
    }

    // MC update at episode end
    const mcValues = mcUpdate(mcStates, mcRewards, mcLearner.values, alpha, gamma);

    setTdLearner(prev => ({
      ...prev,
      values: tdValues,
      steps: prev.steps + stepsTaken,
      episodes: prev.episodes + 1,
      history: [...prev.history, [...tdValues]],
      currentPosition: 2,
      episodeRewards: [],
      episodeStates: [2],
    }));

    setMcLearner(prev => ({
      ...prev,
      values: mcValues,
      steps: prev.steps + stepsTaken,
      episodes: prev.episodes + 1,
      history: [...prev.history, [...mcValues]],
      currentPosition: 2,
      episodeRewards: [],
      episodeStates: [2],
    }));

    setEpisodeInProgress(false);
    setLastAction(null);
  }, [tdLearner, mcLearner, takeStep, tdUpdate, mcUpdate, alpha, gamma]);

  // Auto-play loop
  useEffect(() => {
    if (isPlaying) {
      intervalRef.current = setInterval(runStep, speed);
    }
    return () => {
      if (intervalRef.current) clearInterval(intervalRef.current);
    };
  }, [isPlaying, speed, runStep]);

  // Reset
  const reset = () => {
    setTdLearner(initLearner());
    setMcLearner(initLearner());
    setIsPlaying(false);
    setEpisodeInProgress(false);
    setLastAction(null);
  };

  // Calculate RMSE from true values
  const calculateRMSE = (values: number[]): number => {
    let sum = 0;
    for (let i = 0; i < NUM_STATES; i++) {
      const diff = values[i] - TRUE_VALUES[i];
      sum += diff * diff;
    }
    return Math.sqrt(sum / NUM_STATES);
  };

  // Render state visualization
  const renderStateBar = (learner: LearnerState, label: string, color: string) => {
    return (
      <div className="flex-1">
        <div className="text-center mb-2">
          <span className={`font-medium ${color}`}>{label}</span>
        </div>

        {/* State visualization */}
        <div className="flex items-center justify-center gap-1 mb-4">
          {/* Left terminal */}
          <div className="w-10 h-12 bg-slate-900 border border-slate-700 rounded flex items-center justify-center text-slate-500 text-xs">
            0
          </div>

          {/* States */}
          {learner.values.map((value, i) => {
            const isCurrentPos = learner.currentPosition === i;
            const trueValue = TRUE_VALUES[i];

            return (
              <div
                key={i}
                className={`w-14 h-12 border rounded flex flex-col items-center justify-center transition-all duration-200 ${
                  isCurrentPos
                    ? 'border-amber-500 bg-amber-900/30'
                    : 'border-slate-600 bg-slate-800'
                }`}
              >
                <div className="text-xs text-slate-400">S{i + 1}</div>
                <div className={`text-sm font-mono ${color}`}>{value.toFixed(2)}</div>
              </div>
            );
          })}

          {/* Right terminal */}
          <div className="w-10 h-12 bg-emerald-900/30 border border-emerald-700 rounded flex items-center justify-center text-emerald-400 text-xs">
            +1
          </div>
        </div>

        {/* Value bar chart */}
        <div className="flex items-end justify-center gap-1 h-20 mb-2">
          {learner.values.map((value, i) => {
            const trueValue = TRUE_VALUES[i];
            const height = Math.max(value * 100, 2);
            const trueHeight = trueValue * 100;

            return (
              <div key={i} className="flex flex-col items-center gap-1">
                <div className="relative w-10 h-16 bg-slate-900/50 rounded-t">
                  {/* True value indicator */}
                  <div
                    className="absolute w-full border-t-2 border-dashed border-slate-500"
                    style={{ bottom: `${trueHeight}%` }}
                  />
                  {/* Estimated value bar */}
                  <div
                    className={`absolute bottom-0 w-full rounded-t transition-all duration-200 ${
                      color === 'text-blue-400' ? 'bg-blue-500/60' : 'bg-violet-500/60'
                    }`}
                    style={{ height: `${height}%` }}
                  />
                </div>
                <div className="text-xs text-slate-500">S{i + 1}</div>
              </div>
            );
          })}
        </div>

        {/* Stats */}
        <div className="text-center text-sm">
          <span className="text-slate-500">Episodes: </span>
          <span className="text-slate-300">{learner.episodes}</span>
          <span className="text-slate-600 mx-2">|</span>
          <span className="text-slate-500">RMSE: </span>
          <span className={color}>{calculateRMSE(learner.values).toFixed(3)}</span>
        </div>
      </div>
    );
  };

  // Render learning curve
  const renderLearningCurve = () => {
    const tdHistory = tdLearner.history;
    const mcHistory = mcLearner.history;
    const maxLen = Math.max(tdHistory.length, mcHistory.length);

    if (maxLen <= 1) return null;

    const tdRMSE = tdHistory.map(values => calculateRMSE(values));
    const mcRMSE = mcHistory.map(values => calculateRMSE(values));

    const maxRMSE = Math.max(...tdRMSE, ...mcRMSE, 0.5);

    return (
      <div className="mt-6 p-4 bg-slate-900/30 rounded-lg">
        <div className="text-slate-300 text-sm font-medium mb-2">Learning Curves (RMSE over Episodes)</div>
        <div className="relative h-24">
          {/* TD line */}
          <svg className="absolute inset-0 w-full h-full" preserveAspectRatio="none">
            <polyline
              fill="none"
              stroke="rgb(96, 165, 250)"
              strokeWidth="2"
              points={tdRMSE.map((rmse, i) => {
                const x = (i / (maxLen - 1)) * 100;
                const y = 100 - (rmse / maxRMSE) * 100;
                return `${x}%,${y}%`;
              }).join(' ')}
            />
          </svg>

          {/* MC line */}
          <svg className="absolute inset-0 w-full h-full" preserveAspectRatio="none">
            <polyline
              fill="none"
              stroke="rgb(167, 139, 250)"
              strokeWidth="2"
              points={mcRMSE.map((rmse, i) => {
                const x = (i / (maxLen - 1)) * 100;
                const y = 100 - (rmse / maxRMSE) * 100;
                return `${x}%,${y}%`;
              }).join(' ')}
            />
          </svg>

          {/* Zero line */}
          <div className="absolute bottom-0 w-full border-t border-slate-700" />
        </div>
        <div className="flex justify-between text-xs text-slate-500 mt-1">
          <span>0</span>
          <span>{maxLen - 1} episodes</span>
        </div>
        <div className="flex justify-center gap-4 mt-2 text-xs">
          <div className="flex items-center gap-1">
            <div className="w-3 h-0.5 bg-blue-400"></div>
            <span className="text-slate-400">TD(0)</span>
          </div>
          <div className="flex items-center gap-1">
            <div className="w-3 h-0.5 bg-violet-400"></div>
            <span className="text-slate-400">Monte Carlo</span>
          </div>
        </div>
      </div>
    );
  };

  return (
    <div className="my-8 p-6 bg-slate-800/50 rounded-xl border border-slate-700">
      {/* Header */}
      <div className="text-center mb-6">
        <h3 className="text-xl font-bold text-slate-200 mb-2">TD(0) vs Monte Carlo Learning</h3>
        <p className="text-slate-400 text-sm">
          Compare how TD and MC learn value estimates on the Random Walk problem.
        </p>
      </div>

      {/* Environment description */}
      <div className="mb-6 p-3 bg-slate-900/30 rounded-lg text-center">
        <div className="text-sm text-slate-400">
          <strong className="text-slate-300">Random Walk:</strong> Agent starts at S3, moves left or right with 50% probability.
          Left terminal gives 0 reward, right terminal gives +1 reward.
        </div>
      </div>

      {/* Parameters */}
      <div className="mb-6 p-4 bg-slate-900/30 rounded-lg">
        <div className="flex flex-wrap gap-4 items-center justify-center">
          <label className="flex items-center gap-2 text-sm text-slate-300">
            Alpha (α):
            <input
              type="range"
              min={0.01}
              max={0.5}
              step={0.01}
              value={alpha}
              onChange={e => setAlpha(Number(e.target.value))}
              className="w-20"
            />
            <span className="w-10">{alpha.toFixed(2)}</span>
          </label>

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
        </div>
      </div>

      {/* Last action indicator */}
      {lastAction && (
        <div className="text-center mb-4">
          <span className="px-3 py-1 rounded-full bg-slate-700 text-slate-300 text-sm">
            Action: {lastAction === 'left' ? '← Left' : 'Right →'}
          </span>
        </div>
      )}

      {/* Side by side comparison */}
      <div className="flex gap-4 mb-6">
        {renderStateBar(tdLearner, 'TD(0)', 'text-blue-400')}
        <div className="w-px bg-slate-700"></div>
        {renderStateBar(mcLearner, 'Monte Carlo', 'text-violet-400')}
      </div>

      {/* True values reference */}
      <div className="mb-6 p-3 bg-slate-900/30 rounded-lg">
        <div className="text-xs text-slate-500 text-center mb-2">True Values (dashed lines)</div>
        <div className="flex justify-center gap-4 text-sm">
          {TRUE_VALUES.map((v, i) => (
            <span key={i} className="text-slate-400">
              S{i + 1}: <span className="text-slate-300">{v.toFixed(2)}</span>
            </span>
          ))}
        </div>
      </div>

      {/* Learning curves */}
      {renderLearningCurve()}

      {/* Controls */}
      <div className="flex justify-center gap-3 flex-wrap mt-6">
        <button
          onClick={runStep}
          className="px-4 py-2 rounded-lg bg-blue-600 text-white font-medium hover:bg-blue-500 transition-colors"
        >
          Step
        </button>
        <button
          onClick={runEpisode}
          className="px-4 py-2 rounded-lg bg-emerald-600 text-white font-medium hover:bg-emerald-500 transition-colors"
        >
          Run Episode
        </button>
        <button
          onClick={() => setIsPlaying(!isPlaying)}
          className="px-4 py-2 rounded-lg bg-slate-600 text-white font-medium hover:bg-slate-500 transition-colors"
        >
          {isPlaying ? 'Pause' : 'Play'}
        </button>
        <button
          onClick={reset}
          className="px-4 py-2 rounded-lg bg-slate-600 text-white font-medium hover:bg-slate-500 transition-colors"
        >
          Reset
        </button>
      </div>

      {/* Explanation */}
      <div className="mt-6 p-4 bg-slate-900/30 rounded-lg">
        <div className="text-slate-400 text-sm">
          <strong className="text-slate-300">Key Difference:</strong>
          <ul className="mt-2 space-y-1 list-disc list-inside">
            <li>
              <span className="text-blue-400">TD(0)</span> updates after <em>every step</em> using bootstrap estimates
              (V(s) depends on V(s&apos;))
            </li>
            <li>
              <span className="text-violet-400">Monte Carlo</span> waits until <em>episode end</em> to update,
              using actual returns
            </li>
          </ul>
          <div className="mt-2">
            Watch how TD values change immediately while MC values only update when an episode terminates.
            TD typically learns faster due to its online updates!
          </div>
        </div>
      </div>
    </div>
  );
}

export default TDLearningDemo;
