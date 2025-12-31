/**
 * Value Iteration Demo
 *
 * A dynamic programming visualization for GridWorld:
 * - 4x4 GridWorld with goal and optional obstacles
 * - Step-by-step value iteration
 * - V(s) values displayed in each cell
 * - Policy arrows emerging as values converge
 * - Controls: Step, Play/Pause, Reset, Speed slider
 *
 * Usage:
 *   <ValueIterationDemo />
 */

import React, { useState, useCallback, useEffect, useRef } from 'react';

interface Position {
  x: number;
  y: number;
}

interface GridCell {
  value: number;
  policy: number; // 0=up, 1=right, 2=down, 3=left, -1=terminal
  isObstacle: boolean;
  isGoal: boolean;
  isStart: boolean;
}

const GRID_SIZE = 4;
const ACTIONS = ['up', 'right', 'down', 'left'];
const ACTION_ARROWS = ['↑', '→', '↓', '←'];
const ACTION_DELTAS: Position[] = [
  { x: 0, y: -1 }, // up
  { x: 1, y: 0 },  // right
  { x: 0, y: 1 },  // down
  { x: -1, y: 0 }, // left
];

// Default obstacle positions
const DEFAULT_OBSTACLES: Position[] = [
  { x: 1, y: 1 },
];

const GOAL: Position = { x: 3, y: 3 };
const START: Position = { x: 0, y: 0 };

// Initialize grid
function initializeGrid(obstacles: Position[]): GridCell[][] {
  const grid: GridCell[][] = [];
  for (let y = 0; y < GRID_SIZE; y++) {
    const row: GridCell[] = [];
    for (let x = 0; x < GRID_SIZE; x++) {
      const isObstacle = obstacles.some(o => o.x === x && o.y === y);
      const isGoal = x === GOAL.x && y === GOAL.y;
      const isStart = x === START.x && y === START.y;
      row.push({
        value: isGoal ? 0 : 0, // Terminal states have 0 value
        policy: isGoal || isObstacle ? -1 : 0,
        isObstacle,
        isGoal,
        isStart,
      });
    }
    grid.push(row);
  }
  return grid;
}

export function ValueIterationDemo() {
  const [obstacles, setObstacles] = useState<Position[]>(DEFAULT_OBSTACLES);
  const [grid, setGrid] = useState<GridCell[][]>(() => initializeGrid(DEFAULT_OBSTACLES));
  const [iteration, setIteration] = useState(0);
  const [gamma, setGamma] = useState(0.9);
  const [stepReward, setStepReward] = useState(-1);
  const [goalReward, setGoalReward] = useState(10);
  const [isPlaying, setIsPlaying] = useState(false);
  const [speed, setSpeed] = useState(500);
  const [converged, setConverged] = useState(false);
  const [maxDelta, setMaxDelta] = useState<number>(0);
  const [showPolicy, setShowPolicy] = useState(true);
  const [editMode, setEditMode] = useState(false);
  const intervalRef = useRef<NodeJS.Timeout | null>(null);

  const convergenceThreshold = 0.001;

  // Check if position is valid
  const isValidPosition = useCallback((x: number, y: number): boolean => {
    return x >= 0 && x < GRID_SIZE && y >= 0 && y < GRID_SIZE;
  }, []);

  // Get next state given current position and action
  const getNextState = useCallback((x: number, y: number, action: number): Position => {
    const delta = ACTION_DELTAS[action];
    const newX = x + delta.x;
    const newY = y + delta.y;

    // Check bounds and obstacles
    if (!isValidPosition(newX, newY)) {
      return { x, y }; // Stay in place
    }
    if (grid[newY][newX].isObstacle) {
      return { x, y }; // Stay in place
    }
    return { x: newX, y: newY };
  }, [grid, isValidPosition]);

  // Perform one iteration of value iteration
  const valueIterationStep = useCallback(() => {
    if (converged) return;

    let maxChange = 0;
    const newGrid = grid.map(row => row.map(cell => ({ ...cell })));

    for (let y = 0; y < GRID_SIZE; y++) {
      for (let x = 0; x < GRID_SIZE; x++) {
        const cell = grid[y][x];

        // Skip terminal states and obstacles
        if (cell.isGoal || cell.isObstacle) continue;

        // Compute value for each action and find the best
        let bestValue = -Infinity;
        let bestAction = 0;

        for (let a = 0; a < ACTIONS.length; a++) {
          const nextState = getNextState(x, y, a);
          const nextCell = grid[nextState.y][nextState.x];

          // Reward: goal gives goalReward, otherwise stepReward
          const reward = nextCell.isGoal ? goalReward : stepReward;

          // Bellman equation: R + gamma * V(s')
          const value = reward + gamma * nextCell.value;

          if (value > bestValue) {
            bestValue = value;
            bestAction = a;
          }
        }

        // Update value and policy
        const change = Math.abs(bestValue - cell.value);
        maxChange = Math.max(maxChange, change);

        newGrid[y][x].value = bestValue;
        newGrid[y][x].policy = bestAction;
      }
    }

    setGrid(newGrid);
    setIteration(prev => prev + 1);
    setMaxDelta(maxChange);

    if (maxChange < convergenceThreshold) {
      setConverged(true);
      setIsPlaying(false);
    }
  }, [grid, gamma, stepReward, goalReward, converged, getNextState]);

  // Auto-play loop
  useEffect(() => {
    if (isPlaying && !converged) {
      intervalRef.current = setInterval(valueIterationStep, speed);
    }
    return () => {
      if (intervalRef.current) clearInterval(intervalRef.current);
    };
  }, [isPlaying, converged, speed, valueIterationStep]);

  // Reset
  const reset = () => {
    setGrid(initializeGrid(obstacles));
    setIteration(0);
    setConverged(false);
    setMaxDelta(0);
    setIsPlaying(false);
  };

  // Toggle obstacle
  const toggleObstacle = (x: number, y: number) => {
    if (!editMode) return;
    if (x === GOAL.x && y === GOAL.y) return; // Cannot place obstacle on goal
    if (x === START.x && y === START.y) return; // Cannot place obstacle on start

    const isCurrentlyObstacle = obstacles.some(o => o.x === x && o.y === y);

    let newObstacles: Position[];
    if (isCurrentlyObstacle) {
      newObstacles = obstacles.filter(o => !(o.x === x && o.y === y));
    } else {
      newObstacles = [...obstacles, { x, y }];
    }

    setObstacles(newObstacles);
    setGrid(initializeGrid(newObstacles));
    setIteration(0);
    setConverged(false);
    setMaxDelta(0);
    setIsPlaying(false);
  };

  // Get color based on value
  const getValueColor = (value: number): string => {
    if (value === 0) return 'rgb(55, 65, 81)'; // slate-700

    // Find min/max for normalization
    let minVal = 0;
    let maxVal = 0;
    for (const row of grid) {
      for (const cell of row) {
        if (!cell.isObstacle && !cell.isGoal) {
          minVal = Math.min(minVal, cell.value);
          maxVal = Math.max(maxVal, cell.value);
        }
      }
    }

    if (maxVal === minVal) return 'rgb(55, 65, 81)';

    // Normalize value to 0-1 range
    const normalized = (value - minVal) / (maxVal - minVal);

    // Color gradient: red (low) -> yellow -> green (high)
    if (normalized < 0.5) {
      const r = 220;
      const g = Math.round(normalized * 2 * 180);
      return `rgb(${r}, ${g}, 80)`;
    } else {
      const r = Math.round((1 - normalized) * 2 * 180);
      const g = 180;
      return `rgb(${r}, ${g}, 80)`;
    }
  };

  // Render a single cell
  const renderCell = (x: number, y: number) => {
    const cell = grid[y][x];
    const key = `${x},${y}`;

    let bgColor = getValueColor(cell.value);
    let borderClass = 'border-slate-600';

    if (cell.isObstacle) {
      bgColor = 'rgb(17, 24, 39)'; // slate-900
      borderClass = 'border-slate-700';
    } else if (cell.isGoal) {
      bgColor = 'rgb(16, 185, 129)'; // emerald-500
    }

    return (
      <div
        key={key}
        onClick={() => toggleObstacle(x, y)}
        className={`w-20 h-20 border ${borderClass} rounded-lg flex flex-col items-center justify-center relative transition-all duration-300 ${
          editMode && !cell.isGoal && !(x === START.x && y === START.y)
            ? 'cursor-pointer hover:border-amber-500'
            : ''
        }`}
        style={{ backgroundColor: bgColor }}
      >
        {/* Value display */}
        {!cell.isObstacle && (
          <div className={`text-sm font-mono ${cell.isGoal ? 'text-white' : 'text-slate-200'}`}>
            {cell.value.toFixed(1)}
          </div>
        )}

        {/* Policy arrow */}
        {showPolicy && !cell.isObstacle && !cell.isGoal && cell.policy >= 0 && iteration > 0 && (
          <div className="text-xl text-slate-100 mt-1">
            {ACTION_ARROWS[cell.policy]}
          </div>
        )}

        {/* Goal marker */}
        {cell.isGoal && (
          <div className="text-2xl">🎯</div>
        )}

        {/* Start marker */}
        {cell.isStart && !cell.isObstacle && (
          <div className="absolute top-1 left-1 text-xs text-slate-400">S</div>
        )}

        {/* Obstacle marker */}
        {cell.isObstacle && (
          <div className="text-2xl">🧱</div>
        )}
      </div>
    );
  };

  return (
    <div className="my-8 p-6 bg-slate-800/50 rounded-xl border border-slate-700">
      {/* Header */}
      <div className="text-center mb-6">
        <h3 className="text-xl font-bold text-slate-200 mb-2">Value Iteration Visualization</h3>
        <p className="text-slate-400 text-sm">
          Watch value iteration solve the GridWorld problem step by step.
        </p>
      </div>

      {/* Parameters */}
      <div className="mb-6 p-4 bg-slate-900/30 rounded-lg">
        <div className="flex flex-wrap gap-4 items-center justify-center">
          <label className="flex items-center gap-2 text-sm text-slate-300">
            Gamma (γ):
            <input
              type="range"
              min={0.5}
              max={0.99}
              step={0.01}
              value={gamma}
              onChange={e => {
                setGamma(Number(e.target.value));
                reset();
              }}
              className="w-20"
            />
            <span className="w-10">{gamma.toFixed(2)}</span>
          </label>

          <label className="flex items-center gap-2 text-sm text-slate-300">
            Step Reward:
            <input
              type="number"
              min={-10}
              max={0}
              step={0.5}
              value={stepReward}
              onChange={e => {
                setStepReward(Number(e.target.value));
                reset();
              }}
              className="w-16 px-2 py-1 rounded bg-slate-700 text-slate-200 border border-slate-600"
            />
          </label>

          <label className="flex items-center gap-2 text-sm text-slate-300">
            Goal Reward:
            <input
              type="number"
              min={1}
              max={100}
              step={1}
              value={goalReward}
              onChange={e => {
                setGoalReward(Number(e.target.value));
                reset();
              }}
              className="w-16 px-2 py-1 rounded bg-slate-700 text-slate-200 border border-slate-600"
            />
          </label>

          <label className="flex items-center gap-2 text-sm text-slate-300">
            Speed:
            <input
              type="range"
              min={100}
              max={1000}
              step={100}
              value={speed}
              onChange={e => setSpeed(Number(e.target.value))}
              className="w-20"
            />
          </label>
        </div>
      </div>

      {/* Grid */}
      <div className="flex justify-center mb-6">
        <div style={{ display: 'grid', gridTemplateColumns: `repeat(${GRID_SIZE}, 1fr)`, gap: '4px' }}>
          {Array.from({ length: GRID_SIZE }, (_, y) =>
            Array.from({ length: GRID_SIZE }, (_, x) => renderCell(x, y))
          ).flat()}
        </div>
      </div>

      {/* Status */}
      <div className="mb-4 min-h-[50px]">
        <div className={`text-center p-3 rounded-lg ${
          converged
            ? 'bg-emerald-900/30 text-emerald-400'
            : 'bg-slate-900/30 text-slate-400'
        }`}>
          {converged ? (
            <>
              <span className="text-xl mr-2">✓</span>
              Converged after {iteration} iterations!
            </>
          ) : iteration === 0 ? (
            'Click "Step" or "Play" to start value iteration'
          ) : (
            <>
              Iteration {iteration} | Max change: {maxDelta.toFixed(4)}
            </>
          )}
        </div>
      </div>

      {/* Metrics */}
      <div className="flex justify-center gap-8 mb-6">
        <div className="text-center">
          <div className="text-2xl font-bold text-slate-200">{iteration}</div>
          <div className="text-slate-500 text-sm">Iteration</div>
        </div>
        <div className="text-center">
          <div className="text-2xl font-bold text-amber-400">{maxDelta.toFixed(4)}</div>
          <div className="text-slate-500 text-sm">Max Delta</div>
        </div>
        <div className="text-center">
          <div className="text-2xl font-bold text-slate-200">{gamma.toFixed(2)}</div>
          <div className="text-slate-500 text-sm">Gamma</div>
        </div>
      </div>

      {/* Controls */}
      <div className="flex justify-center gap-3 flex-wrap">
        <button
          onClick={valueIterationStep}
          disabled={converged}
          className="px-4 py-2 rounded-lg bg-blue-600 text-white font-medium hover:bg-blue-500 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
        >
          Step
        </button>
        <button
          onClick={() => setIsPlaying(!isPlaying)}
          disabled={converged}
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
        <button
          onClick={() => setEditMode(!editMode)}
          className={`px-4 py-2 rounded-lg font-medium transition-colors ${
            editMode
              ? 'bg-violet-600 text-white hover:bg-violet-500'
              : 'bg-slate-600 text-white hover:bg-slate-500'
          }`}
        >
          {editMode ? 'Done Editing' : 'Edit Obstacles'}
        </button>
      </div>

      {/* Legend */}
      <div className="flex justify-center gap-6 mt-6 text-sm flex-wrap">
        <div className="flex items-center gap-2">
          <span>🎯</span>
          <span className="text-slate-400">Goal (+{goalReward})</span>
        </div>
        <div className="flex items-center gap-2">
          <span>🧱</span>
          <span className="text-slate-400">Obstacle</span>
        </div>
        <div className="flex items-center gap-2">
          <div className="w-4 h-4 rounded" style={{ backgroundColor: 'rgb(180, 180, 80)' }}></div>
          <span className="text-slate-400">High Value</span>
        </div>
        <div className="flex items-center gap-2">
          <div className="w-4 h-4 rounded" style={{ backgroundColor: 'rgb(220, 80, 80)' }}></div>
          <span className="text-slate-400">Low Value</span>
        </div>
      </div>

      {/* Explanation */}
      <div className="mt-6 p-4 bg-slate-900/30 rounded-lg">
        <div className="text-slate-400 text-sm">
          <strong className="text-slate-300">How it works:</strong> Value iteration repeatedly applies
          the Bellman equation V(s) = max_a [R(s,a) + γ V(s&apos;)] until values converge. Each cell shows
          its estimated value, and arrows show the optimal policy. Higher gamma values make the agent
          care more about future rewards. More negative step rewards encourage shorter paths.
          {editMode && (
            <div className="mt-2 text-violet-300">
              <strong>Edit Mode:</strong> Click on empty cells to add/remove obstacles.
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

export default ValueIterationDemo;
