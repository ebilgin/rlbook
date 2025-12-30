/**
 * GridWorld Interactive Component
 *
 * A versatile GridWorld environment visualization that supports:
 * - Agent movement visualization
 * - Q-value heatmap display
 * - Policy arrows
 * - Step-by-step or continuous execution
 *
 * This is the primary recurring example throughout the Q-learning section.
 *
 * Usage:
 *   <GridWorld
 *     width={5}
 *     height={5}
 *     obstacles={[[1,1], [2,2]]}
 *     goal={[4,4]}
 *     showQValues={true}
 *     showPolicy={true}
 *   />
 */

import React, { useState, useCallback, useRef, useEffect } from 'react';

// Types
interface Position {
  x: number;
  y: number;
}

interface GridWorldProps {
  width?: number;
  height?: number;
  obstacles?: Position[];
  goal?: Position;
  start?: Position;
  showQValues?: boolean;
  showPolicy?: boolean;
  cellSize?: number;
  onStateChange?: (state: GridWorldState) => void;
}

interface GridWorldState {
  agentPosition: Position;
  qValues: Map<string, number[]>; // "x,y" -> [up, right, down, left]
  episodeReward: number;
  stepCount: number;
  isTerminal: boolean;
}

// Actions: 0=up, 1=right, 2=down, 3=left
const ACTIONS = ['↑', '→', '↓', '←'];
const ACTION_DELTAS: Position[] = [
  { x: 0, y: -1 }, // up
  { x: 1, y: 0 }, // right
  { x: 0, y: 1 }, // down
  { x: -1, y: 0 }, // left
];

// Placeholder component - full implementation will use TensorFlow.js
export function GridWorld({
  width = 5,
  height = 5,
  obstacles = [],
  goal = { x: 4, y: 4 },
  start = { x: 0, y: 0 },
  showQValues = true,
  showPolicy = true,
  cellSize = 60,
  onStateChange,
}: GridWorldProps) {
  // State
  const [state, setState] = useState<GridWorldState>({
    agentPosition: start,
    qValues: new Map(),
    episodeReward: 0,
    stepCount: 0,
    isTerminal: false,
  });

  const [isPlaying, setIsPlaying] = useState(false);
  const [speed, setSpeed] = useState(500); // ms per step
  const intervalRef = useRef<NodeJS.Timeout | null>(null);

  // Helper functions
  const posToKey = (pos: Position) => `${pos.x},${pos.y}`;

  const isObstacle = (pos: Position) =>
    obstacles.some(obs => obs.x === pos.x && obs.y === pos.y);

  const isGoal = (pos: Position) => pos.x === goal.x && pos.y === goal.y;

  const isValidPosition = (pos: Position) =>
    pos.x >= 0 && pos.x < width && pos.y >= 0 && pos.y < height && !isObstacle(pos);

  // Get color based on Q-value
  const getValueColor = (value: number, maxValue: number) => {
    if (maxValue === 0) return 'rgb(200, 200, 200)';
    const intensity = Math.min(value / maxValue, 1);
    // Blue (low) to Red (high)
    const r = Math.round(intensity * 255);
    const b = Math.round((1 - intensity) * 255);
    return `rgb(${r}, 100, ${b})`;
  };

  // Render a single cell
  const renderCell = (x: number, y: number) => {
    const pos = { x, y };
    const key = posToKey(pos);
    const isAgent = state.agentPosition.x === x && state.agentPosition.y === y;
    const isGoalCell = isGoal(pos);
    const isObstacleCell = isObstacle(pos);

    const qVals = state.qValues.get(key) || [0, 0, 0, 0];
    const maxQ = Math.max(...qVals);
    const bestAction = qVals.indexOf(maxQ);

    let bgColor = 'white';
    if (isObstacleCell) bgColor = '#333';
    else if (isGoalCell) bgColor = '#4ade80';
    else if (showQValues && maxQ > 0) bgColor = getValueColor(maxQ, 10);

    return (
      <div
        key={key}
        className="grid-cell"
        style={{
          width: cellSize,
          height: cellSize,
          backgroundColor: bgColor,
          border: '1px solid #ccc',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          position: 'relative',
          fontSize: cellSize * 0.3,
        }}
      >
        {isAgent && <span className="agent">🤖</span>}
        {isGoalCell && !isAgent && <span className="goal">🎯</span>}
        {showPolicy && !isObstacleCell && !isGoalCell && (
          <span className="policy-arrow" style={{ opacity: maxQ > 0 ? 1 : 0.3 }}>
            {ACTIONS[bestAction]}
          </span>
        )}
      </div>
    );
  };

  // Controls
  const step = useCallback(() => {
    // TODO: Implement actual Q-learning step
    // This is a placeholder that moves randomly
    setState(prev => {
      if (prev.isTerminal) return prev;

      const action = Math.floor(Math.random() * 4);
      const delta = ACTION_DELTAS[action];
      const newPos = {
        x: prev.agentPosition.x + delta.x,
        y: prev.agentPosition.y + delta.y,
      };

      const validPos = isValidPosition(newPos) ? newPos : prev.agentPosition;
      const reachedGoal = isGoal(validPos);

      return {
        ...prev,
        agentPosition: validPos,
        stepCount: prev.stepCount + 1,
        episodeReward: prev.episodeReward + (reachedGoal ? 10 : -0.1),
        isTerminal: reachedGoal,
      };
    });
  }, [goal, obstacles, width, height]);

  const reset = () => {
    setState({
      agentPosition: start,
      qValues: new Map(),
      episodeReward: 0,
      stepCount: 0,
      isTerminal: false,
    });
    setIsPlaying(false);
  };

  // Auto-play loop
  useEffect(() => {
    if (isPlaying && !state.isTerminal) {
      intervalRef.current = setInterval(step, speed);
    } else if (intervalRef.current) {
      clearInterval(intervalRef.current);
    }
    return () => {
      if (intervalRef.current) clearInterval(intervalRef.current);
    };
  }, [isPlaying, speed, state.isTerminal, step]);

  // Notify parent of state changes
  useEffect(() => {
    onStateChange?.(state);
  }, [state, onStateChange]);

  return (
    <div className="gridworld-container">
      {/* Grid */}
      <div
        className="grid"
        style={{
          display: 'grid',
          gridTemplateColumns: `repeat(${width}, ${cellSize}px)`,
          gap: 0,
          margin: '20px auto',
        }}
      >
        {Array.from({ length: height }, (_, y) =>
          Array.from({ length: width }, (_, x) => renderCell(x, y))
        )}
      </div>

      {/* Controls */}
      <div className="controls" style={{ display: 'flex', gap: 10, justifyContent: 'center' }}>
        <button onClick={step} disabled={state.isTerminal}>
          Step
        </button>
        <button onClick={() => setIsPlaying(!isPlaying)}>
          {isPlaying ? 'Pause' : 'Play'}
        </button>
        <button onClick={reset}>Reset</button>
        <label>
          Speed:
          <input
            type="range"
            min={100}
            max={1000}
            value={speed}
            onChange={e => setSpeed(Number(e.target.value))}
          />
        </label>
      </div>

      {/* Metrics */}
      <div className="metrics" style={{ textAlign: 'center', marginTop: 10 }}>
        <span>Steps: {state.stepCount}</span>
        <span style={{ marginLeft: 20 }}>Reward: {state.episodeReward.toFixed(1)}</span>
        {state.isTerminal && <span style={{ marginLeft: 20, color: 'green' }}>Goal Reached!</span>}
      </div>
    </div>
  );
}

export default GridWorld;
