/**
 * Q-Learning Demo Interactive Component
 *
 * A CliffWalking environment (4x12 grid) that demonstrates:
 * - Q-values for each state-action pair
 * - Comparison between SARSA (safe path) vs Q-Learning (optimal path)
 * - Animated agent training episodes
 * - Learned policy with arrows
 *
 * Usage:
 *   <QLearningDemo />
 */

import React, { useState, useCallback, useEffect, useRef } from 'react';

// Types
interface Position {
  x: number;
  y: number;
}

type Algorithm = 'q-learning' | 'sarsa';

interface QTable {
  [key: string]: number[]; // "x,y" -> [up, right, down, left]
}

interface TrainingState {
  agentPos: Position;
  qTable: QTable;
  episode: number;
  step: number;
  totalReward: number;
  episodeReward: number;
  isTerminal: boolean;
  path: Position[];
}

// Environment constants
const GRID_WIDTH = 12;
const GRID_HEIGHT = 4;
const START: Position = { x: 0, y: 3 };
const GOAL: Position = { x: 11, y: 3 };
const CLIFF_START = 1;
const CLIFF_END = 10;
const CLIFF_ROW = 3;

// Actions: 0=up, 1=right, 2=down, 3=left
const ACTIONS = ['up', 'right', 'down', 'left'];
const ACTION_ARROWS = ['↑', '→', '↓', '←'];
const ACTION_DELTAS: Position[] = [
  { x: 0, y: -1 }, // up
  { x: 1, y: 0 },  // right
  { x: 0, y: 1 },  // down
  { x: -1, y: 0 }, // left
];

// Hyperparameters
const ALPHA = 0.5;    // Learning rate
const GAMMA = 0.95;   // Discount factor
const EPSILON = 0.1;  // Exploration rate

function posToKey(pos: Position): string {
  return `${pos.x},${pos.y}`;
}

function isCliff(pos: Position): boolean {
  return pos.y === CLIFF_ROW && pos.x >= CLIFF_START && pos.x <= CLIFF_END;
}

function isGoal(pos: Position): boolean {
  return pos.x === GOAL.x && pos.y === GOAL.y;
}

function isValidPosition(pos: Position): boolean {
  return pos.x >= 0 && pos.x < GRID_WIDTH && pos.y >= 0 && pos.y < GRID_HEIGHT;
}

function clamp(pos: Position): Position {
  return {
    x: Math.max(0, Math.min(GRID_WIDTH - 1, pos.x)),
    y: Math.max(0, Math.min(GRID_HEIGHT - 1, pos.y)),
  };
}

function getNextState(pos: Position, action: number): Position {
  const delta = ACTION_DELTAS[action];
  const nextPos = { x: pos.x + delta.x, y: pos.y + delta.y };
  return isValidPosition(nextPos) ? nextPos : pos;
}

function getReward(pos: Position): number {
  if (isCliff(pos)) return -100;
  if (isGoal(pos)) return 0;
  return -1;
}

function initQTable(): QTable {
  const qTable: QTable = {};
  for (let y = 0; y < GRID_HEIGHT; y++) {
    for (let x = 0; x < GRID_WIDTH; x++) {
      qTable[posToKey({ x, y })] = [0, 0, 0, 0];
    }
  }
  return qTable;
}

function epsilonGreedy(qValues: number[], epsilon: number): number {
  if (Math.random() < epsilon) {
    return Math.floor(Math.random() * 4);
  }
  const maxQ = Math.max(...qValues);
  const bestActions = qValues
    .map((q, i) => (q === maxQ ? i : -1))
    .filter(i => i !== -1);
  return bestActions[Math.floor(Math.random() * bestActions.length)];
}

function greedyAction(qValues: number[]): number {
  const maxQ = Math.max(...qValues);
  const bestActions = qValues
    .map((q, i) => (q === maxQ ? i : -1))
    .filter(i => i !== -1);
  return bestActions[Math.floor(Math.random() * bestActions.length)];
}

export function QLearningDemo() {
  const [algorithm, setAlgorithm] = useState<Algorithm>('q-learning');
  const [qLearningState, setQLearningState] = useState<TrainingState>(() => ({
    agentPos: START,
    qTable: initQTable(),
    episode: 0,
    step: 0,
    totalReward: 0,
    episodeReward: 0,
    isTerminal: false,
    path: [START],
  }));
  const [sarsaState, setSarsaState] = useState<TrainingState>(() => ({
    agentPos: START,
    qTable: initQTable(),
    episode: 0,
    step: 0,
    totalReward: 0,
    episodeReward: 0,
    isTerminal: false,
    path: [START],
  }));

  const [isTraining, setIsTraining] = useState(false);
  const [speed, setSpeed] = useState(50); // ms per step
  const [showQValues, setShowQValues] = useState(false);
  const [showComparison, setShowComparison] = useState(false);
  const intervalRef = useRef<NodeJS.Timeout | null>(null);
  const sarsaActionRef = useRef<number | null>(null);

  const currentState = algorithm === 'q-learning' ? qLearningState : sarsaState;
  const setCurrentState = algorithm === 'q-learning' ? setQLearningState : setSarsaState;

  // Q-Learning update step
  const qLearningStep = useCallback(() => {
    setQLearningState(prev => {
      if (prev.isTerminal) {
        // Start new episode
        return {
          ...prev,
          agentPos: START,
          step: 0,
          episodeReward: 0,
          isTerminal: false,
          path: [START],
          episode: prev.episode + 1,
        };
      }

      const state = prev.agentPos;
      const key = posToKey(state);
      const qValues = prev.qTable[key] || [0, 0, 0, 0];

      // Choose action using epsilon-greedy
      const action = epsilonGreedy(qValues, EPSILON);
      const nextState = getNextState(state, action);
      const reward = getReward(nextState);

      // Q-Learning update: Q(s,a) += alpha * (r + gamma * max_a' Q(s',a') - Q(s,a))
      const nextKey = posToKey(nextState);
      const nextQValues = prev.qTable[nextKey] || [0, 0, 0, 0];
      const maxNextQ = Math.max(...nextQValues);

      const newQTable = { ...prev.qTable };
      newQTable[key] = [...qValues];
      newQTable[key][action] = qValues[action] + ALPHA * (reward + GAMMA * maxNextQ - qValues[action]);

      const fellOffCliff = isCliff(nextState);
      const reachedGoal = isGoal(nextState);
      const terminal = fellOffCliff || reachedGoal;

      const newPos = fellOffCliff ? START : nextState;
      const newPath = fellOffCliff ? [START] : [...prev.path, nextState];

      return {
        ...prev,
        agentPos: newPos,
        qTable: newQTable,
        step: prev.step + 1,
        episodeReward: prev.episodeReward + reward,
        totalReward: prev.totalReward + reward,
        isTerminal: terminal,
        path: newPath,
      };
    });
  }, []);

  // SARSA update step
  const sarsaStep = useCallback(() => {
    setSarsaState(prev => {
      if (prev.isTerminal) {
        sarsaActionRef.current = null;
        return {
          ...prev,
          agentPos: START,
          step: 0,
          episodeReward: 0,
          isTerminal: false,
          path: [START],
          episode: prev.episode + 1,
        };
      }

      const state = prev.agentPos;
      const key = posToKey(state);
      const qValues = prev.qTable[key] || [0, 0, 0, 0];

      // If we don't have an action selected, choose one
      let action = sarsaActionRef.current;
      if (action === null) {
        action = epsilonGreedy(qValues, EPSILON);
      }

      const nextState = getNextState(state, action);
      const reward = getReward(nextState);

      // Choose next action using epsilon-greedy (for SARSA)
      const nextKey = posToKey(nextState);
      const nextQValues = prev.qTable[nextKey] || [0, 0, 0, 0];
      const nextAction = epsilonGreedy(nextQValues, EPSILON);

      // SARSA update: Q(s,a) += alpha * (r + gamma * Q(s',a') - Q(s,a))
      const newQTable = { ...prev.qTable };
      newQTable[key] = [...qValues];
      newQTable[key][action] = qValues[action] + ALPHA * (reward + GAMMA * nextQValues[nextAction] - qValues[action]);

      const fellOffCliff = isCliff(nextState);
      const reachedGoal = isGoal(nextState);
      const terminal = fellOffCliff || reachedGoal;

      // Store the next action for the next step
      sarsaActionRef.current = terminal ? null : nextAction;

      const newPos = fellOffCliff ? START : nextState;
      const newPath = fellOffCliff ? [START] : [...prev.path, nextState];

      return {
        ...prev,
        agentPos: newPos,
        qTable: newQTable,
        step: prev.step + 1,
        episodeReward: prev.episodeReward + reward,
        totalReward: prev.totalReward + reward,
        isTerminal: terminal,
        path: newPath,
      };
    });
  }, []);

  // Training loop
  useEffect(() => {
    if (isTraining) {
      const stepFn = showComparison
        ? () => { qLearningStep(); sarsaStep(); }
        : algorithm === 'q-learning'
          ? qLearningStep
          : sarsaStep;

      intervalRef.current = setInterval(stepFn, speed);
    }
    return () => {
      if (intervalRef.current) clearInterval(intervalRef.current);
    };
  }, [isTraining, speed, algorithm, showComparison, qLearningStep, sarsaStep]);

  const reset = () => {
    setIsTraining(false);
    sarsaActionRef.current = null;
    const newState: TrainingState = {
      agentPos: START,
      qTable: initQTable(),
      episode: 0,
      step: 0,
      totalReward: 0,
      episodeReward: 0,
      isTerminal: false,
      path: [START],
    };
    setQLearningState(newState);
    setSarsaState({ ...newState, qTable: initQTable() });
  };

  // Train for multiple episodes quickly
  const trainBatch = (episodes: number) => {
    const trainAlgorithm = (
      prevState: TrainingState,
      isQLearning: boolean
    ): TrainingState => {
      let state = { ...prevState, qTable: { ...prevState.qTable } };

      for (let ep = 0; ep < episodes; ep++) {
        let agentPos = START;
        let episodeReward = 0;
        let action = isQLearning ? null : epsilonGreedy(state.qTable[posToKey(agentPos)] || [0, 0, 0, 0], EPSILON);

        for (let step = 0; step < 500; step++) {
          const key = posToKey(agentPos);
          const qValues = state.qTable[key] || [0, 0, 0, 0];

          const currentAction = isQLearning
            ? epsilonGreedy(qValues, EPSILON)
            : (action as number);

          const nextState = getNextState(agentPos, currentAction);
          const reward = getReward(nextState);
          episodeReward += reward;

          const nextKey = posToKey(nextState);
          const nextQValues = state.qTable[nextKey] || [0, 0, 0, 0];

          if (isQLearning) {
            // Q-Learning update
            const maxNextQ = Math.max(...nextQValues);
            if (!state.qTable[key]) state.qTable[key] = [0, 0, 0, 0];
            else state.qTable[key] = [...state.qTable[key]];
            state.qTable[key][currentAction] = qValues[currentAction] +
              ALPHA * (reward + GAMMA * maxNextQ - qValues[currentAction]);
          } else {
            // SARSA update
            const nextAction = epsilonGreedy(nextQValues, EPSILON);
            if (!state.qTable[key]) state.qTable[key] = [0, 0, 0, 0];
            else state.qTable[key] = [...state.qTable[key]];
            state.qTable[key][currentAction] = qValues[currentAction] +
              ALPHA * (reward + GAMMA * nextQValues[nextAction] - qValues[currentAction]);
            action = nextAction;
          }

          const fellOffCliff = isCliff(nextState);
          const reachedGoal = isGoal(nextState);

          if (fellOffCliff || reachedGoal) {
            break;
          }
          agentPos = nextState;
        }

        state.episode++;
        state.totalReward += episodeReward;
      }

      state.agentPos = START;
      state.path = [START];
      state.episodeReward = 0;
      state.isTerminal = false;
      state.step = 0;

      return state;
    };

    if (showComparison) {
      setQLearningState(prev => trainAlgorithm(prev, true));
      setSarsaState(prev => trainAlgorithm(prev, false));
    } else if (algorithm === 'q-learning') {
      setQLearningState(prev => trainAlgorithm(prev, true));
    } else {
      setSarsaState(prev => trainAlgorithm(prev, false));
    }
  };

  // Run greedy policy to show learned path
  const runGreedyPolicy = (qTable: QTable): Position[] => {
    const path: Position[] = [START];
    let pos = START;
    const visited = new Set<string>();

    for (let i = 0; i < 100; i++) {
      const key = posToKey(pos);
      if (visited.has(key)) break; // Avoid infinite loops
      visited.add(key);

      const qValues = qTable[key] || [0, 0, 0, 0];
      const action = greedyAction(qValues);
      pos = getNextState(pos, action);
      path.push(pos);

      if (isGoal(pos) || isCliff(pos)) break;
    }
    return path;
  };

  const getQValueColor = (value: number): string => {
    // Color scale from red (negative) through white (zero) to green (positive)
    const maxAbs = 20;
    const normalized = Math.max(-1, Math.min(1, value / maxAbs));
    if (normalized >= 0) {
      const intensity = Math.round((1 - normalized) * 150 + 100);
      return `rgb(${intensity}, 255, ${intensity})`;
    } else {
      const intensity = Math.round((1 + normalized) * 150 + 100);
      return `rgb(255, ${intensity}, ${intensity})`;
    }
  };

  const renderCell = (x: number, y: number, state: TrainingState, isSecondary: boolean = false) => {
    const pos = { x, y };
    const key = posToKey(pos);
    const isStart = x === START.x && y === START.y;
    const isGoalCell = isGoal(pos);
    const isCliffCell = isCliff(pos);
    const isAgent = state.agentPos.x === x && state.agentPos.y === y;
    const isOnPath = state.path.some(p => p.x === x && p.y === y);

    const qValues = state.qTable[key] || [0, 0, 0, 0];
    const maxQ = Math.max(...qValues);
    const bestAction = greedyAction(qValues);

    let bgClass = 'bg-slate-700/50';
    if (isCliffCell) bgClass = 'bg-red-900/60';
    else if (isGoalCell) bgClass = 'bg-emerald-600/50';
    else if (isStart) bgClass = 'bg-blue-600/30';
    else if (isOnPath && !showQValues) bgClass = 'bg-amber-600/30';
    else if (showQValues && maxQ !== 0) {
      // Don't override with Q-value colors for special cells
    }

    let cellStyle: React.CSSProperties = {};
    if (showQValues && !isCliffCell && !isGoalCell && maxQ !== 0) {
      cellStyle.backgroundColor = getQValueColor(maxQ);
    }

    const cellSize = showComparison ? 'w-6 h-6 text-xs' : 'w-10 h-10 text-sm';

    return (
      <div
        key={key}
        className={`${cellSize} ${bgClass} border border-slate-600/50 flex items-center justify-center relative transition-all duration-150`}
        style={cellStyle}
      >
        {isAgent && !isSecondary && (
          <span className={showComparison ? 'text-sm' : 'text-lg'}>🤖</span>
        )}
        {isGoalCell && !isAgent && (
          <span className={showComparison ? 'text-sm' : 'text-lg'}>🎯</span>
        )}
        {isCliffCell && (
          <span className={`${showComparison ? 'text-xs' : 'text-sm'} text-red-300`}>X</span>
        )}
        {!isCliffCell && !isGoalCell && !isAgent && !showQValues && maxQ !== 0 && (
          <span className="text-slate-400 opacity-70">{ACTION_ARROWS[bestAction]}</span>
        )}
        {showQValues && !isCliffCell && !isGoalCell && !isAgent && (
          <span className="text-slate-900 font-mono" style={{ fontSize: showComparison ? '6px' : '8px' }}>
            {maxQ.toFixed(1)}
          </span>
        )}
      </div>
    );
  };

  const renderGrid = (state: TrainingState, label?: string, isSecondary: boolean = false) => (
    <div className="flex flex-col items-center">
      {label && (
        <div className={`text-slate-300 font-medium mb-2 ${showComparison ? 'text-sm' : 'text-base'}`}>
          {label}
        </div>
      )}
      <div
        style={{
          display: 'grid',
          gridTemplateColumns: `repeat(${GRID_WIDTH}, 1fr)`,
          gap: '1px',
        }}
      >
        {Array.from({ length: GRID_HEIGHT }, (_, y) =>
          Array.from({ length: GRID_WIDTH }, (_, x) => renderCell(x, y, state, isSecondary))
        ).flat()}
      </div>
      <div className="flex gap-4 mt-2 text-xs text-slate-400">
        <span>Episodes: {state.episode}</span>
        <span>Reward: {state.totalReward.toFixed(0)}</span>
      </div>
    </div>
  );

  return (
    <div className="my-8 p-6 bg-slate-800/50 rounded-xl border border-slate-700">
      {/* Header */}
      <div className="text-center mb-6">
        <h3 className="text-xl font-bold text-slate-200 mb-2">Q-Learning vs SARSA: CliffWalking</h3>
        <p className="text-slate-400 text-sm">
          Compare how Q-Learning (off-policy) and SARSA (on-policy) learn different paths.
        </p>
      </div>

      {/* Algorithm selector */}
      {!showComparison && (
        <div className="flex justify-center gap-2 mb-4">
          <button
            onClick={() => setAlgorithm('q-learning')}
            className={`px-4 py-2 rounded-lg font-medium transition-colors ${
              algorithm === 'q-learning'
                ? 'bg-blue-600 text-white'
                : 'bg-slate-700 text-slate-300 hover:bg-slate-600'
            }`}
          >
            Q-Learning
          </button>
          <button
            onClick={() => setAlgorithm('sarsa')}
            className={`px-4 py-2 rounded-lg font-medium transition-colors ${
              algorithm === 'sarsa'
                ? 'bg-amber-600 text-white'
                : 'bg-slate-700 text-slate-300 hover:bg-slate-600'
            }`}
          >
            SARSA
          </button>
        </div>
      )}

      {/* Grid(s) */}
      <div className={`flex justify-center mb-6 ${showComparison ? 'gap-8' : ''}`}>
        {showComparison ? (
          <>
            {renderGrid(qLearningState, 'Q-Learning (risky optimal path)')}
            {renderGrid(sarsaState, 'SARSA (safe path)', true)}
          </>
        ) : (
          renderGrid(currentState)
        )}
      </div>

      {/* Legend */}
      <div className="flex justify-center gap-4 mb-4 text-xs flex-wrap">
        <div className="flex items-center gap-1">
          <span>🤖</span>
          <span className="text-slate-400">Agent</span>
        </div>
        <div className="flex items-center gap-1">
          <span>🎯</span>
          <span className="text-slate-400">Goal</span>
        </div>
        <div className="flex items-center gap-1">
          <div className="w-3 h-3 bg-red-900/60 border border-slate-600 rounded"></div>
          <span className="text-slate-400">Cliff (-100)</span>
        </div>
        <div className="flex items-center gap-1">
          <div className="w-3 h-3 bg-blue-600/30 border border-slate-600 rounded"></div>
          <span className="text-slate-400">Start</span>
        </div>
      </div>

      {/* Metrics */}
      <div className="flex justify-center gap-6 mb-6 text-sm">
        <div className="text-center">
          <div className="text-xl font-bold text-slate-200">{currentState.episode}</div>
          <div className="text-slate-500 text-xs">Episodes</div>
        </div>
        <div className="text-center">
          <div className="text-xl font-bold text-slate-200">{currentState.step}</div>
          <div className="text-slate-500 text-xs">Steps</div>
        </div>
        <div className="text-center">
          <div className={`text-xl font-bold ${currentState.episodeReward >= 0 ? 'text-emerald-400' : 'text-red-400'}`}>
            {currentState.episodeReward.toFixed(0)}
          </div>
          <div className="text-slate-500 text-xs">Episode Reward</div>
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
          onClick={() => trainBatch(100)}
          className="px-4 py-2 rounded-lg bg-slate-600 text-white font-medium hover:bg-slate-500 transition-colors"
        >
          +100 Episodes
        </button>
        <button
          onClick={() => trainBatch(500)}
          className="px-4 py-2 rounded-lg bg-slate-600 text-white font-medium hover:bg-slate-500 transition-colors"
        >
          +500 Episodes
        </button>
        <button
          onClick={reset}
          className="px-4 py-2 rounded-lg bg-slate-600 text-white font-medium hover:bg-slate-500 transition-colors"
        >
          Reset
        </button>
      </div>

      {/* Toggle buttons */}
      <div className="flex justify-center gap-2 flex-wrap mb-4">
        <button
          onClick={() => setShowQValues(!showQValues)}
          className={`px-3 py-1.5 rounded-lg text-sm font-medium transition-colors ${
            showQValues
              ? 'bg-violet-600 text-white'
              : 'bg-slate-700 text-slate-300 hover:bg-slate-600'
          }`}
        >
          {showQValues ? 'Hide Q-Values' : 'Show Q-Values'}
        </button>
        <button
          onClick={() => setShowComparison(!showComparison)}
          className={`px-3 py-1.5 rounded-lg text-sm font-medium transition-colors ${
            showComparison
              ? 'bg-amber-600 text-white'
              : 'bg-slate-700 text-slate-300 hover:bg-slate-600'
          }`}
        >
          {showComparison ? 'Single View' : 'Compare Both'}
        </button>
        <label className="flex items-center gap-2 text-sm text-slate-400">
          Speed:
          <input
            type="range"
            min={10}
            max={200}
            value={200 - speed}
            onChange={e => setSpeed(200 - Number(e.target.value))}
            className="w-20"
          />
        </label>
      </div>

      {/* Explanation */}
      <div className="mt-4 p-4 bg-slate-900/30 rounded-lg">
        <div className="text-slate-400 text-sm">
          <strong className="text-slate-300">Key Insight:</strong>{' '}
          Q-Learning learns the optimal policy (walking along the cliff edge) because it uses the
          maximum Q-value for updates, regardless of the exploration policy. SARSA learns a safer
          path (going up first) because its updates reflect the actual exploratory actions taken,
          which might accidentally fall off the cliff.
        </div>
      </div>
    </div>
  );
}

export default QLearningDemo;
