import { useState, useCallback, useEffect, useRef } from 'react';

// Types
interface Elevator {
  id: number;
  floor: number;
  targetFloor: number | null;
  passengerCount: number;
  maxCapacity: number;
  movementOffset: number; // Phase offset for animation (0-1)
}

interface Passenger {
  id: number;
  originFloor: number;
  destFloor: number;
  spawnTime: number;
}

interface SimulationState {
  elevators: Elevator[];
  waitingPassengers: Passenger[];
  deliveredPassengers: Passenger[];
  timestep: number;
  totalSpawned: number;
  avgWaitTime: number;
  maxWaitTime: number;
}

type Algorithm = 'random' | 'nearest' | 'scan' | 'rl';
type TrafficPattern = 'morning' | 'lunch' | 'evening' | 'quiet';

const N_FLOORS = 10;
const N_ELEVATORS = 3;
const ELEVATOR_CAPACITY = 8;
const MOVE_SPEED = 0.5;

// Initialize simulation state
function initializeState(): SimulationState {
  return {
    elevators: Array.from({ length: N_ELEVATORS }, (_, i) => ({
      id: i,
      floor: 0,
      targetFloor: null,
      passengerCount: 0,
      maxCapacity: ELEVATOR_CAPACITY,
      movementOffset: i * 0.33, // Stagger the elevators (0, 0.33, 0.66)
    })),
    waitingPassengers: [],
    deliveredPassengers: [],
    timestep: 0,
    totalSpawned: 0,
    avgWaitTime: 0,
    maxWaitTime: 0,
  };
}

// Spawn passengers based on traffic pattern
function spawnPassengers(
  state: SimulationState,
  pattern: TrafficPattern,
  rate: number
): Passenger[] {
  const newPassengers: Passenger[] = [];

  // Poisson arrivals
  const lambda = rate * (pattern === 'morning' || pattern === 'evening' ? 1.5 : pattern === 'quiet' ? 0.5 : 1.0);
  const nArrivals = Math.random() < lambda ? (Math.random() < lambda - 1 ? 2 : 1) : 0;

  for (let i = 0; i < nArrivals; i++) {
    let origin: number, dest: number;

    // Traffic pattern determines origin-destination
    if (pattern === 'morning') {
      // 70% lobby → upper
      if (Math.random() < 0.7) {
        origin = 0;
        dest = Math.floor(Math.random() * (N_FLOORS - 1)) + 1;
      } else {
        origin = Math.floor(Math.random() * N_FLOORS);
        dest = Math.floor(Math.random() * N_FLOORS);
      }
    } else if (pattern === 'evening') {
      // 70% upper → lobby
      if (Math.random() < 0.7) {
        origin = Math.floor(Math.random() * (N_FLOORS - 1)) + 1;
        dest = 0;
      } else {
        origin = Math.floor(Math.random() * N_FLOORS);
        dest = Math.floor(Math.random() * N_FLOORS);
      }
    } else {
      origin = Math.floor(Math.random() * N_FLOORS);
      dest = Math.floor(Math.random() * N_FLOORS);
    }

    if (origin !== dest) {
      newPassengers.push({
        id: state.totalSpawned + i,
        originFloor: origin,
        destFloor: dest,
        spawnTime: state.timestep,
      });
    }
  }

  return newPassengers;
}

// Select actions based on algorithm
function selectActions(state: SimulationState, algorithm: Algorithm): number[] {
  if (algorithm === 'random') {
    return state.elevators.map(() => Math.floor(Math.random() * N_FLOORS));
  }

  if (algorithm === 'nearest') {
    return state.elevators.map(elevator => {
      if (state.waitingPassengers.length === 0) return elevator.floor;

      // Find nearest waiting passenger
      let minDist = Infinity;
      let nearestFloor = elevator.floor;

      for (const p of state.waitingPassengers) {
        const dist = Math.abs(elevator.floor - p.originFloor);
        if (dist < minDist) {
          minDist = dist;
          nearestFloor = p.originFloor;
        }
      }

      return Math.round(nearestFloor);
    });
  }

  if (algorithm === 'scan') {
    return state.elevators.map(elevator => {
      const current = Math.round(elevator.floor);

      // If has target, continue toward it
      if (elevator.targetFloor !== null && Math.abs(elevator.floor - elevator.targetFloor) > 0.1) {
        return elevator.targetFloor;
      }

      // Find nearest waiting passenger
      if (state.waitingPassengers.length > 0) {
        const nearest = state.waitingPassengers.reduce((prev, curr) =>
          Math.abs(curr.originFloor - current) < Math.abs(prev.originFloor - current) ? curr : prev
        );
        return nearest.originFloor;
      }

      return current;
    });
  }

  // RL - simplified (would load trained model in real implementation)
  return state.elevators.map(elevator => {
    if (state.waitingPassengers.length === 0) return Math.round(elevator.floor);

    // Simple heuristic: go to floor with most waiting passengers
    const floorCounts = new Array(N_FLOORS).fill(0);
    for (const p of state.waitingPassengers) {
      floorCounts[p.originFloor]++;
    }

    const maxFloor = floorCounts.indexOf(Math.max(...floorCounts));
    return maxFloor;
  });
}

// Simulation step
function simulationStep(
  state: SimulationState,
  actions: number[],
  pattern: TrafficPattern,
  spawnRate: number
): SimulationState {
  const newState = { ...state };

  // Set targets
  newState.elevators = state.elevators.map((e, i) => ({
    ...e,
    targetFloor: actions[i],
  }));

  // Board/exit passengers
  newState.elevators.forEach(elevator => {
    const floor = Math.round(elevator.floor);

    // Exit passengers at current floor
    if (elevator.passengerCount > 0) {
      // Simplified: assume passengers exit (no tracking of destinations in this simple sim)
      const exitCount = Math.floor(Math.random() * elevator.passengerCount * 0.3);
      elevator.passengerCount = Math.max(0, elevator.passengerCount - exitCount);

      if (exitCount > 0) {
        newState.deliveredPassengers = [...newState.deliveredPassengers, ...Array(exitCount).fill(null)];
      }
    }

    // Board waiting passengers
    if (elevator.passengerCount < elevator.maxCapacity) {
      const passengersHere = newState.waitingPassengers.filter(p => p.originFloor === floor);
      const canBoard = Math.min(
        passengersHere.length,
        elevator.maxCapacity - elevator.passengerCount
      );

      if (canBoard > 0) {
        elevator.passengerCount += canBoard;
        newState.waitingPassengers = newState.waitingPassengers.filter(p => p.originFloor !== floor || passengersHere.indexOf(p) >= canBoard);
      }
    }
  });

  // Move elevators (with offset to make them move at different times)
  newState.elevators.forEach(elevator => {
    if (elevator.targetFloor !== null) {
      const target = elevator.targetFloor;
      const current = elevator.floor;

      if (Math.abs(target - current) > 0.01) {
        const direction = target > current ? 1 : -1;
        // Each elevator moves at slightly different speed based on offset
        const speed = MOVE_SPEED * (0.8 + 0.4 * elevator.movementOffset);
        const movement = Math.min(speed, Math.abs(target - current));
        elevator.floor += direction * movement;
      } else {
        elevator.floor = target;
      }
    }
  });

  // Spawn new passengers
  const newPassengers = spawnPassengers(newState, pattern, spawnRate);
  newState.waitingPassengers = [...newState.waitingPassengers, ...newPassengers];
  newState.totalSpawned += newPassengers.length;

  // Calculate metrics
  if (newState.waitingPassengers.length > 0) {
    const waitTimes = newState.waitingPassengers.map(p => newState.timestep - p.spawnTime);
    newState.avgWaitTime = waitTimes.reduce((a, b) => a + b, 0) / waitTimes.length;
    newState.maxWaitTime = Math.max(...waitTimes);
  } else {
    newState.avgWaitTime = 0;
    newState.maxWaitTime = 0;
  }

  newState.timestep++;

  return newState;
}

export function ElevatorSimulation() {
  const [state, setState] = useState<SimulationState>(initializeState);
  const [isPlaying, setIsPlaying] = useState(false);
  const [speed, setSpeed] = useState(500);
  const [algorithm, setAlgorithm] = useState<Algorithm>('nearest');
  const [trafficPattern, setTrafficPattern] = useState<TrafficPattern>('morning');
  const [spawnRate] = useState(0.3);
  const [isDarkMode, setIsDarkMode] = useState(false);

  const intervalRef = useRef<number | null>(null);

  // Dark mode detection
  useEffect(() => {
    const checkDarkMode = () => {
      setIsDarkMode(document.documentElement.classList.contains('dark'));
    };
    checkDarkMode();

    const observer = new MutationObserver(checkDarkMode);
    observer.observe(document.documentElement, {
      attributes: true,
      attributeFilter: ['class'],
    });

    return () => observer.disconnect();
  }, []);

  const step = useCallback(() => {
    setState(prevState => {
      const actions = selectActions(prevState, algorithm);
      return simulationStep(prevState, actions, trafficPattern, spawnRate);
    });
  }, [algorithm, trafficPattern, spawnRate]);

  // Auto-play loop
  useEffect(() => {
    if (isPlaying) {
      intervalRef.current = window.setInterval(step, speed);
    }
    return () => {
      if (intervalRef.current) clearInterval(intervalRef.current);
    };
  }, [isPlaying, speed, step]);

  const reset = useCallback(() => {
    setState(initializeState());
    setIsPlaying(false);
  }, []);

  const togglePlay = useCallback(() => {
    setIsPlaying(prev => !prev);
  }, []);

  // Colors
  const bgColor = isDarkMode ? '#1f2937' : '#f3f4f6';
  const floorColor = isDarkMode ? '#374151' : '#d1d5db';
  const textColor = isDarkMode ? '#f3f4f6' : '#1f2937';

  return (
    <div className="w-full max-w-6xl mx-auto p-4" style={{ backgroundColor: bgColor, color: textColor, borderRadius: '8px' }}>
      <h2 className="text-2xl font-bold mb-4">Elevator Dispatch Simulation</h2>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-4">
        {/* Building View */}
        <div className="md:col-span-2 border rounded-lg p-4" style={{ borderColor: floorColor }}>
          <h3 className="text-lg font-semibold mb-2">Building View</h3>
          <div className="relative" style={{ height: '400px' }}>
            {/* Floors */}
            {Array.from({ length: N_FLOORS }, (_, i) => N_FLOORS - 1 - i).map(floor => (
              <div
                key={floor}
                className="absolute w-full flex items-center"
                style={{
                  top: `${(N_FLOORS - 1 - floor) * (400 / N_FLOORS)}px`,
                  height: `${400 / N_FLOORS}px`,
                  borderBottom: `1px solid ${floorColor}`,
                }}
              >
                <span className="w-12 text-sm">{floor}</span>

                {/* Elevators */}
                <div className="flex-1 relative" style={{ height: '100%' }}>
                  {state.elevators.map(elevator => {
                    const elevatorFloor = Math.round(elevator.floor);
                    if (elevatorFloor === floor) {
                      const utilization = elevator.passengerCount / elevator.maxCapacity;
                      const color = utilization > 0.7 ? '#ef4444' : utilization > 0.3 ? '#eab308' : '#3b82f6';

                      return (
                        <div
                          key={elevator.id}
                          className="absolute flex items-center justify-center text-white text-xs font-bold"
                          style={{
                            left: `${20 + elevator.id * 25}%`,
                            width: '60px',
                            height: '80%',
                            backgroundColor: color,
                            borderRadius: '4px',
                            border: '2px solid ' + (isDarkMode ? '#fff' : '#000'),
                            transition: 'all 0.3s ease',
                          }}
                        >
                          <div className="text-center">
                            <div>E{elevator.id}</div>
                            <div className="text-xs">{elevator.passengerCount}/{elevator.maxCapacity}</div>
                          </div>
                        </div>
                      );
                    }
                    return null;
                  })}
                </div>

                {/* Waiting passengers */}
                <div className="w-24 flex items-center">
                  {state.waitingPassengers.filter(p => p.originFloor === floor).slice(0, 5).map((p, i) => (
                    <div
                      key={p.id}
                      className="w-3 h-3 rounded-full mx-0.5"
                      style={{ backgroundColor: '#f97316' }}
                      title={`Waiting ${state.timestep - p.spawnTime} steps`}
                    />
                  ))}
                  {state.waitingPassengers.filter(p => p.originFloor === floor).length > 5 && (
                    <span className="text-xs ml-1">+{state.waitingPassengers.filter(p => p.originFloor === floor).length - 5}</span>
                  )}
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Metrics */}
        <div className="border rounded-lg p-4" style={{ borderColor: floorColor }}>
          <h3 className="text-lg font-semibold mb-2">Metrics</h3>
          <div className="space-y-3">
            <div>
              <div className="text-sm opacity-70">Timestep</div>
              <div className="text-xl font-bold">{state.timestep}</div>
            </div>
            <div>
              <div className="text-sm opacity-70">Waiting</div>
              <div className="text-xl font-bold">{state.waitingPassengers.length}</div>
            </div>
            <div>
              <div className="text-sm opacity-70">Avg Wait</div>
              <div className="text-xl font-bold">{state.avgWaitTime.toFixed(1)}s</div>
            </div>
            <div>
              <div className="text-sm opacity-70">Max Wait</div>
              <div className="text-xl font-bold">{state.maxWaitTime.toFixed(0)}s</div>
            </div>
            <div>
              <div className="text-sm opacity-70">Served</div>
              <div className="text-xl font-bold">{state.deliveredPassengers.length}/{state.totalSpawned}</div>
            </div>
            <div>
              <div className="text-sm opacity-70">Utilization</div>
              <div className="text-xl font-bold">
                {state.elevators.length > 0
                  ? ((state.elevators.reduce((sum, e) => sum + e.passengerCount, 0) /
                     (state.elevators.length * ELEVATOR_CAPACITY)) * 100).toFixed(0)
                  : 0}%
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Controls */}
      <div className="border rounded-lg p-4" style={{ borderColor: floorColor }}>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
          <div>
            <label className="block text-sm mb-1">Algorithm</label>
            <select
              value={algorithm}
              onChange={(e) => setAlgorithm(e.target.value as Algorithm)}
              className="w-full px-3 py-2 rounded border"
              style={{
                backgroundColor: isDarkMode ? '#374151' : '#fff',
                borderColor: floorColor,
                color: textColor,
              }}
            >
              <option value="random">Random</option>
              <option value="nearest">Nearest Car</option>
              <option value="scan">SCAN</option>
              <option value="rl">RL (Simple)</option>
            </select>
          </div>

          <div>
            <label className="block text-sm mb-1">Traffic Pattern</label>
            <select
              value={trafficPattern}
              onChange={(e) => setTrafficPattern(e.target.value as TrafficPattern)}
              className="w-full px-3 py-2 rounded border"
              style={{
                backgroundColor: isDarkMode ? '#374151' : '#fff',
                borderColor: floorColor,
                color: textColor,
              }}
            >
              <option value="morning">Morning Rush</option>
              <option value="lunch">Lunch Time</option>
              <option value="evening">Evening Rush</option>
              <option value="quiet">Quiet Period</option>
            </select>
          </div>

          <div>
            <label className="block text-sm mb-1">Speed: {speed}ms</label>
            <input
              type="range"
              min="100"
              max="1000"
              step="50"
              value={speed}
              onChange={(e) => setSpeed(parseInt(e.target.value))}
              className="w-full"
            />
          </div>

          <div className="flex flex-wrap items-end gap-2">
            <button
              onClick={togglePlay}
              className="px-3 py-2 sm:px-4 text-sm sm:text-base rounded font-semibold flex-1 sm:flex-none min-w-[80px]"
              style={{
                backgroundColor: isPlaying ? '#ef4444' : '#10b981',
                color: '#fff',
              }}
            >
              {isPlaying ? 'Pause' : 'Play'}
            </button>
            <button
              onClick={step}
              disabled={isPlaying}
              className="px-3 py-2 sm:px-4 text-sm sm:text-base rounded font-semibold flex-1 sm:flex-none min-w-[80px]"
              style={{
                backgroundColor: '#06b6d4',
                color: '#fff',
                opacity: isPlaying ? 0.5 : 1,
              }}
            >
              Step
            </button>
            <button
              onClick={reset}
              className="px-3 py-2 sm:px-4 text-sm sm:text-base rounded font-semibold flex-1 sm:flex-none min-w-[80px]"
              style={{
                backgroundColor: '#8b5cf6',
                color: '#fff',
              }}
            >
              Reset
            </button>
          </div>
        </div>
      </div>

      <div className="mt-4 text-sm opacity-70">
        <p><strong>Legend:</strong> Blue = Low utilization, Yellow = Medium, Red = High | Orange dots = Waiting passengers</p>
      </div>
    </div>
  );
}
