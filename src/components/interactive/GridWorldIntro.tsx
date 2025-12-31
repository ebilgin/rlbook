/**
 * GridWorld Interactive Demo for Introduction Chapter
 *
 * A simplified, intuition-focused GridWorld that demonstrates
 * the RL loop without diving into algorithms. Shows:
 * - Agent taking steps toward goal
 * - Rewards accumulating
 * - The sense of "learning" through a pre-computed optimal policy
 *
 * Usage:
 *   <GridWorldIntro />
 */

import React, { useState, useCallback, useEffect, useRef } from 'react';

interface Position {
  x: number;
  y: number;
}

// 4x4 Grid with optimal policy pre-computed
// The agent uses this to show "smart" behavior
const GRID_SIZE = 4;
const GOAL: Position = { x: 3, y: 3 };
const START: Position = { x: 0, y: 0 };

// Optimal action for each cell (0=up, 1=right, 2=down, 3=left)
// This gives a simple shortest path to the goal
const OPTIMAL_POLICY: Record<string, number> = {
  '0,0': 2, // down
  '0,1': 1, // right
  '0,2': 1, // right
  '1,0': 2, // down
  '1,1': 2, // down
  '1,2': 1, // right
  '2,0': 2, // down
  '2,1': 2, // down
  '2,2': 2, // down
  '3,0': 1, // right
  '3,1': 1, // right
  '3,2': 1, // right
  '0,3': 1, // right
  '1,3': 1, // right
  '2,3': 1, // right
};

const ACTION_DELTAS: Position[] = [
  { x: 0, y: -1 }, // up
  { x: 1, y: 0 },  // right
  { x: 0, y: 1 },  // down
  { x: -1, y: 0 }, // left
];

const ACTION_NAMES = ['Up', 'Right', 'Down', 'Left'];
const ACTION_ARROWS = ['↑', '→', '↓', '←'];

interface StepInfo {
  action: string;
  reward: number;
  description: string;
}

export function GridWorldIntro() {
  const [agentPos, setAgentPos] = useState<Position>(START);
  const [totalReward, setTotalReward] = useState(0);
  const [stepCount, setStepCount] = useState(0);
  const [isTerminal, setIsTerminal] = useState(false);
  const [lastStep, setLastStep] = useState<StepInfo | null>(null);
  const [isPlaying, setIsPlaying] = useState(false);
  const [showPolicy, setShowPolicy] = useState(false);
  const intervalRef = useRef<NodeJS.Timeout | null>(null);

  const posToKey = (pos: Position) => `${pos.x},${pos.y}`;
  const isGoal = (pos: Position) => pos.x === GOAL.x && pos.y === GOAL.y;

  const step = useCallback(() => {
    if (isTerminal) return;

    const key = posToKey(agentPos);
    const action = OPTIMAL_POLICY[key];

    if (action === undefined) {
      // Already at goal or error
      return;
    }

    const delta = ACTION_DELTAS[action];
    const newPos = {
      x: agentPos.x + delta.x,
      y: agentPos.y + delta.y,
    };

    const reachedGoal = isGoal(newPos);
    const reward = reachedGoal ? 10 : -1;

    setLastStep({
      action: ACTION_NAMES[action],
      reward,
      description: reachedGoal
        ? "Reached the goal! +10 reward"
        : `Moved ${ACTION_NAMES[action].toLowerCase()}, -1 step penalty`,
    });

    setAgentPos(newPos);
    setTotalReward(prev => prev + reward);
    setStepCount(prev => prev + 1);
    setIsTerminal(reachedGoal);

    if (reachedGoal) {
      setIsPlaying(false);
    }
  }, [agentPos, isTerminal]);

  const reset = () => {
    setAgentPos(START);
    setTotalReward(0);
    setStepCount(0);
    setIsTerminal(false);
    setLastStep(null);
    setIsPlaying(false);
  };

  // Auto-play
  useEffect(() => {
    if (isPlaying && !isTerminal) {
      intervalRef.current = setInterval(step, 800);
    }
    return () => {
      if (intervalRef.current) clearInterval(intervalRef.current);
    };
  }, [isPlaying, isTerminal, step]);

  const renderCell = (x: number, y: number) => {
    const isAgent = agentPos.x === x && agentPos.y === y;
    const isGoalCell = x === GOAL.x && y === GOAL.y;
    const key = posToKey({ x, y });
    const action = OPTIMAL_POLICY[key];

    let bgClass = 'bg-slate-700/50';
    if (isGoalCell) bgClass = 'bg-emerald-600/40';
    if (isAgent && isGoalCell) bgClass = 'bg-emerald-500/60';
    else if (isAgent) bgClass = 'bg-blue-600/40';

    return (
      <div
        key={key}
        className={`w-16 h-16 ${bgClass} border border-slate-600 rounded-lg flex items-center justify-center relative transition-all duration-300`}
      >
        {isAgent && (
          <span className="text-2xl animate-pulse">🤖</span>
        )}
        {isGoalCell && !isAgent && (
          <span className="text-2xl">🎯</span>
        )}
        {showPolicy && !isGoalCell && action !== undefined && !isAgent && (
          <span className="text-slate-400 text-lg">{ACTION_ARROWS[action]}</span>
        )}
      </div>
    );
  };

  return (
    <div className="my-8 p-6 bg-slate-800/50 rounded-xl border border-slate-700">
      {/* Header */}
      <div className="text-center mb-6">
        <h3 className="text-xl font-bold text-slate-200 mb-2">Interactive GridWorld</h3>
        <p className="text-slate-400 text-sm">
          Watch an agent learn to reach the goal. Click "Step" to see each action.
        </p>
      </div>

      {/* Grid */}
      <div className="flex justify-center mb-6">
        <div style={{display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '4px'}}>
          {Array.from({ length: GRID_SIZE }, (_, y) =>
            Array.from({ length: GRID_SIZE }, (_, x) => renderCell(x, y))
          ).flat()}
        </div>
      </div>

      {/* Step Info */}
      <div className="mb-4 min-h-[60px]">
        {lastStep ? (
          <div className="bg-slate-900/50 rounded-lg p-4 text-center">
            <div className="text-slate-300">
              <span className="text-amber-400 font-medium">Action:</span> {lastStep.action}
              <span className="mx-4">|</span>
              <span className={lastStep.reward > 0 ? 'text-emerald-400' : 'text-red-400'}>
                {lastStep.reward > 0 ? '+' : ''}{lastStep.reward} reward
              </span>
            </div>
            <div className="text-slate-500 text-sm mt-1">{lastStep.description}</div>
          </div>
        ) : (
          <div className="bg-slate-900/50 rounded-lg p-4 text-center text-slate-500">
            Click "Step" to begin, or "Play" for automatic stepping
          </div>
        )}
      </div>

      {/* Metrics */}
      <div className="flex justify-center gap-8 mb-6">
        <div className="text-center">
          <div className="text-2xl font-bold text-slate-200">{stepCount}</div>
          <div className="text-slate-500 text-sm">Steps</div>
        </div>
        <div className="text-center">
          <div className={`text-2xl font-bold ${totalReward >= 0 ? 'text-emerald-400' : 'text-red-400'}`}>
            {totalReward > 0 ? '+' : ''}{totalReward}
          </div>
          <div className="text-slate-500 text-sm">Total Reward</div>
        </div>
        {isTerminal && (
          <div className="text-center">
            <div className="text-2xl">🎉</div>
            <div className="text-emerald-400 text-sm font-medium">Goal!</div>
          </div>
        )}
      </div>

      {/* Controls */}
      <div className="flex justify-center gap-3 flex-wrap">
        <button
          onClick={step}
          disabled={isTerminal}
          className="px-4 py-2 rounded-lg bg-blue-600 text-white font-medium hover:bg-blue-500 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
        >
          Step
        </button>
        <button
          onClick={() => setIsPlaying(!isPlaying)}
          disabled={isTerminal}
          className="px-4 py-2 rounded-lg bg-slate-600 text-white font-medium hover:bg-slate-500 disabled:opacity-50 transition-colors"
        >
          {isPlaying ? 'Pause' : 'Play'}
        </button>
        <button
          onClick={reset}
          className="px-4 py-2 rounded-lg bg-slate-600 text-white font-medium hover:bg-slate-500 transition-colors"
        >
          Reset
        </button>
        <button
          onClick={() => setShowPolicy(!showPolicy)}
          className={`px-4 py-2 rounded-lg font-medium transition-colors ${
            showPolicy
              ? 'bg-amber-600 text-white hover:bg-amber-500'
              : 'bg-slate-600 text-white hover:bg-slate-500'
          }`}
        >
          {showPolicy ? 'Hide Policy' : 'Show Policy'}
        </button>
      </div>

      {/* Legend */}
      <div className="flex justify-center gap-6 mt-6 text-sm">
        <div className="flex items-center gap-2">
          <span>🤖</span>
          <span className="text-slate-400">Agent</span>
        </div>
        <div className="flex items-center gap-2">
          <span>🎯</span>
          <span className="text-slate-400">Goal (+10)</span>
        </div>
        <div className="flex items-center gap-2">
          <div className="w-4 h-4 bg-slate-700/50 border border-slate-600 rounded"></div>
          <span className="text-slate-400">Empty (-1 per step)</span>
        </div>
      </div>

      {/* Explanation */}
      <div className="mt-6 p-4 bg-slate-900/30 rounded-lg">
        <div className="text-slate-400 text-sm">
          <strong className="text-slate-300">What's happening?</strong> The agent has already learned an optimal policy (shown with arrows when you click "Show Policy").
          Each step costs -1 reward, encouraging the shortest path. Reaching the goal gives +10 reward.
          A well-trained agent maximizes total reward.
        </div>
      </div>
    </div>
  );
}

export default GridWorldIntro;
