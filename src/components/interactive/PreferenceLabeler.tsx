/**
 * Preference Labeler
 *
 * Interactive demo where users collect preference data by comparing
 * AI responses. A simple reward model "trains" on their preferences
 * and shows predictions on new pairs.
 */

import React, { useState, useMemo, useCallback } from 'react';

interface ResponsePair {
  id: number;
  prompt: string;
  responseA: string;
  responseB: string;
  // Hidden quality scores (used to simulate reward model learning)
  qualityA: number;
  qualityB: number;
  // Feature signals the reward model can learn
  lengthA: number;
  lengthB: number;
  specificity: 'A' | 'B'; // which response is more specific
}

interface Preference {
  pairId: number;
  choice: 'A' | 'B' | 'tie';
  prompt: string;
}

const responsePairs: ResponsePair[] = [
  {
    id: 0,
    prompt: 'Explain what a neural network is.',
    responseA:
      'A neural network is a computing system inspired by biological neural networks. It consists of layers of interconnected nodes that process information.',
    responseB:
      'A neural network is like a series of filters. Data flows through layers — each layer detects patterns, from simple edges to complex concepts. Input goes in, gets transformed layer by layer, and a prediction comes out. The key: the network learns which patterns matter by adjusting connection weights during training.',
    qualityA: 0.4,
    qualityB: 0.85,
    lengthA: 30,
    lengthB: 55,
    specificity: 'B',
  },
  {
    id: 1,
    prompt: 'What causes seasons on Earth?',
    responseA:
      "Earth's seasons are caused by the 23.5° tilt of its axis. When the Northern Hemisphere tilts toward the Sun, it receives more direct sunlight — that's summer. Six months later, it tilts away — winter. It's not about distance from the Sun (Earth is actually closest to the Sun during Northern Hemisphere winter).",
    responseB:
      'Seasons happen because the Earth orbits the Sun and different parts get more or less sunlight at different times of the year.',
    qualityA: 0.9,
    qualityB: 0.3,
    lengthA: 60,
    lengthB: 22,
    specificity: 'A',
  },
  {
    id: 2,
    prompt: 'How do I make pasta?',
    responseA:
      'Boil water, add salt, cook pasta according to the package instructions, drain, and add sauce.',
    responseB:
      'Here\'s a simple method:\n1. Bring a large pot of water to a rolling boil\n2. Add a generous pinch of salt (the water should taste like the sea)\n3. Add your pasta and stir immediately to prevent sticking\n4. Cook 1 minute less than the package says (it\'ll finish in the sauce)\n5. Save a cup of pasta water before draining\n6. Toss the pasta with your sauce, adding pasta water to get a silky consistency',
    qualityA: 0.35,
    qualityB: 0.88,
    lengthA: 18,
    lengthB: 80,
    specificity: 'B',
  },
  {
    id: 3,
    prompt: 'Is it safe to eat raw cookie dough?',
    responseA:
      'Yes, raw cookie dough is delicious! Many people eat it straight from the bowl without any problems. It\'s a classic treat.',
    responseB:
      'Traditional raw cookie dough carries some risk due to raw eggs (salmonella) and uncooked flour (E. coli). If you want to eat it raw safely, use heat-treated flour (bake at 350°F for 5 min) and pasteurized eggs or skip eggs entirely. Many stores now sell "edible cookie dough" made with these substitutions.',
    qualityA: 0.15,
    qualityB: 0.92,
    lengthA: 25,
    lengthB: 60,
    specificity: 'B',
  },
  {
    id: 4,
    prompt: 'Why is the sky blue?',
    responseA:
      'The sky is blue because of Rayleigh scattering. Sunlight contains all colors. When it hits our atmosphere, shorter wavelengths (blue/violet) scatter more than longer ones (red/yellow). Our eyes are more sensitive to blue than violet, so we see a blue sky. At sunset, light travels through more atmosphere, scattering away the blue and leaving reds and oranges.',
    responseB:
      'The sky appears blue because of the way sunlight interacts with Earth\'s atmosphere. The atmosphere scatters shorter wavelengths of light more effectively than longer ones, making the sky appear blue to our eyes.',
    qualityA: 0.88,
    qualityB: 0.55,
    lengthA: 65,
    lengthB: 35,
    specificity: 'A',
  },
  {
    id: 5,
    prompt: 'What is reinforcement learning?',
    responseA:
      'Reinforcement learning is a type of machine learning where an agent learns by interacting with an environment. The agent takes actions, receives rewards or penalties, and gradually discovers which actions lead to the best outcomes. Think of it like training a dog: good behavior gets treats (positive reward), and bad behavior gets nothing (or a negative signal). Over time, the agent develops a policy — a strategy for choosing actions in each situation to maximize its total reward.',
    responseB:
      'Reinforcement learning (RL) is a machine learning paradigm. It involves an agent, environment, states, actions, and rewards. The agent uses a policy to select actions. RL is used in robotics, game playing, and other applications.',
    qualityA: 0.92,
    qualityB: 0.3,
    lengthA: 85,
    lengthB: 35,
    specificity: 'A',
  },
];

// Simple "reward model" that learns from preferences
// Uses a weighted combination of features to predict quality
function predictReward(
  length: number,
  isSpecific: boolean,
  weights: { lengthW: number; specificW: number; bias: number }
): number {
  const normalizedLength = Math.min(length / 100, 1);
  const score =
    weights.bias +
    weights.lengthW * normalizedLength +
    weights.specificW * (isSpecific ? 1 : 0);
  return 1 / (1 + Math.exp(-score)); // sigmoid
}

// Update weights based on preference (simplified SGD)
function updateWeights(
  weights: { lengthW: number; specificW: number; bias: number },
  pair: ResponsePair,
  choice: 'A' | 'B' | 'tie',
  lr: number = 0.3
): { lengthW: number; specificW: number; bias: number } {
  if (choice === 'tie') return weights;

  const chosenLength = choice === 'A' ? pair.lengthA : pair.lengthB;
  const rejectedLength = choice === 'A' ? pair.lengthB : pair.lengthA;
  const chosenSpecific = pair.specificity === choice;
  const rejectedSpecific = pair.specificity !== choice;

  const rChosen = predictReward(chosenLength, chosenSpecific, weights);
  const rRejected = predictReward(rejectedLength, rejectedSpecific, weights);

  // Bradley-Terry gradient: d/dw log sigma(r_chosen - r_rejected)
  const diff = rChosen - rRejected;
  const grad = 1 / (1 + Math.exp(diff)) - 0; // sigmoid(-diff)

  const normalizedChosenLen = Math.min(chosenLength / 100, 1);
  const normalizedRejectedLen = Math.min(rejectedLength / 100, 1);

  return {
    lengthW: weights.lengthW + lr * grad * (normalizedChosenLen - normalizedRejectedLen),
    specificW: weights.specificW + lr * grad * ((chosenSpecific ? 1 : 0) - (rejectedSpecific ? 1 : 0)),
    bias: weights.bias + lr * grad * 0.1,
  };
}

export function PreferenceLabeler() {
  const [currentPairIdx, setCurrentPairIdx] = useState(0);
  const [preferences, setPreferences] = useState<Preference[]>([]);
  const [modelWeights, setModelWeights] = useState({ lengthW: 0, specificW: 0, bias: 0 });
  const [showPredictions, setShowPredictions] = useState(false);

  const currentPair = responsePairs[currentPairIdx];
  const isComplete = currentPairIdx >= responsePairs.length;

  const handleChoice = useCallback(
    (choice: 'A' | 'B' | 'tie') => {
      const newPref: Preference = {
        pairId: currentPair.id,
        choice,
        prompt: currentPair.prompt,
      };
      setPreferences((prev) => [...prev, newPref]);

      // Update reward model
      setModelWeights((prev) => updateWeights(prev, currentPair, choice));

      // Advance to next pair
      setCurrentPairIdx((prev) => prev + 1);
    },
    [currentPair]
  );

  // Calculate reward model accuracy on collected preferences
  const accuracy = useMemo(() => {
    if (preferences.length === 0) return 0;
    let correct = 0;
    preferences.forEach((pref) => {
      if (pref.choice === 'tie') return;
      const pair = responsePairs[pref.pairId];
      const rA = predictReward(pair.lengthA, pair.specificity === 'A', modelWeights);
      const rB = predictReward(pair.lengthB, pair.specificity === 'B', modelWeights);
      const predicted = rA > rB ? 'A' : 'B';
      if (predicted === pref.choice) correct++;
    });
    const total = preferences.filter((p) => p.choice !== 'tie').length;
    return total > 0 ? Math.round((correct / total) * 100) : 0;
  }, [preferences, modelWeights]);

  const reset = useCallback(() => {
    setCurrentPairIdx(0);
    setPreferences([]);
    setModelWeights({ lengthW: 0, specificW: 0, bias: 0 });
    setShowPredictions(false);
  }, []);

  return (
    <div className="my-8 bg-slate-800/30 rounded-xl border border-slate-700 overflow-hidden">
      {/* Header */}
      <div className="bg-slate-800/50 px-6 py-4 border-b border-slate-700">
        <div className="flex items-center justify-between">
          <div>
            <h3 className="text-lg font-semibold text-slate-200">Try It: Preference Labeling</h3>
            <p className="text-sm text-slate-400">
              Compare responses and pick the better one — just like real RLHF data collection
            </p>
          </div>
          <button
            onClick={reset}
            className="px-3 py-1.5 text-sm bg-slate-700 hover:bg-slate-600 text-slate-300 rounded-lg transition-colors"
          >
            Reset
          </button>
        </div>
      </div>

      <div className="p-6">
        {!isComplete ? (
          <>
            {/* Progress */}
            <div className="flex items-center gap-3 mb-6">
              <div className="text-sm text-slate-400">
                Pair {currentPairIdx + 1} of {responsePairs.length}
              </div>
              <div className="flex-1 h-1.5 bg-slate-800 rounded-full overflow-hidden">
                <div
                  className="h-full bg-violet-500 rounded-full transition-all duration-300"
                  style={{ width: `${(currentPairIdx / responsePairs.length) * 100}%` }}
                />
              </div>
            </div>

            {/* Prompt */}
            <div className="mb-4">
              <div className="text-xs text-slate-500 mb-1 font-medium">PROMPT</div>
              <div className="text-slate-200 bg-slate-900/40 rounded-lg px-4 py-2.5 text-sm">
                {currentPair.prompt}
              </div>
            </div>

            {/* Response comparison */}
            <div className="grid md:grid-cols-2 gap-4 mb-6">
              <button
                onClick={() => handleChoice('A')}
                className="text-left bg-slate-900/40 rounded-xl p-4 border-2 border-slate-700 hover:border-violet-500 transition-colors group"
              >
                <div className="text-xs text-slate-500 mb-2 font-medium group-hover:text-violet-400">
                  RESPONSE A
                </div>
                <div className="text-slate-300 text-sm whitespace-pre-line leading-relaxed">
                  {currentPair.responseA}
                </div>
              </button>
              <button
                onClick={() => handleChoice('B')}
                className="text-left bg-slate-900/40 rounded-xl p-4 border-2 border-slate-700 hover:border-violet-500 transition-colors group"
              >
                <div className="text-xs text-slate-500 mb-2 font-medium group-hover:text-violet-400">
                  RESPONSE B
                </div>
                <div className="text-slate-300 text-sm whitespace-pre-line leading-relaxed">
                  {currentPair.responseB}
                </div>
              </button>
            </div>

            {/* Tie button */}
            <div className="text-center">
              <button
                onClick={() => handleChoice('tie')}
                className="px-4 py-2 text-sm text-slate-400 hover:text-slate-300 border border-slate-700 hover:border-slate-600 rounded-lg transition-colors"
              >
                About equal (tie)
              </button>
            </div>
          </>
        ) : (
          /* Results phase */
          <div>
            <div className="text-center mb-8">
              <div className="text-2xl font-bold text-violet-400 mb-2">Data Collection Complete</div>
              <div className="text-slate-400">
                You labeled {preferences.length} preference pairs — enough to train a simple reward model.
              </div>
            </div>

            {/* Reward model accuracy */}
            <div className="grid md:grid-cols-3 gap-4 mb-6">
              <div className="bg-slate-900/50 rounded-lg p-4 text-center">
                <div className="text-3xl font-bold text-violet-400">{preferences.length}</div>
                <div className="text-xs text-slate-500 mt-1">Preferences Collected</div>
              </div>
              <div className="bg-slate-900/50 rounded-lg p-4 text-center">
                <div className="text-3xl font-bold text-emerald-400">{accuracy}%</div>
                <div className="text-xs text-slate-500 mt-1">Model Accuracy</div>
              </div>
              <div className="bg-slate-900/50 rounded-lg p-4 text-center">
                <div className="text-3xl font-bold text-cyan-400">
                  {preferences.filter((p) => p.choice !== 'tie').length}
                </div>
                <div className="text-xs text-slate-500 mt-1">Training Signal</div>
              </div>
            </div>

            {/* Collected preference summary */}
            <div className="bg-slate-900/50 rounded-lg p-4 mb-6">
              <div className="text-sm text-slate-400 mb-3">Your Preference Dataset</div>
              <div className="space-y-2">
                {preferences.map((pref, i) => (
                  <div key={i} className="flex items-center gap-3 text-sm">
                    <div className="text-slate-500 w-6">#{i + 1}</div>
                    <div className="flex-1 text-slate-300 truncate">{pref.prompt}</div>
                    <div
                      className={`px-2 py-0.5 rounded text-xs font-medium ${
                        pref.choice === 'tie'
                          ? 'bg-slate-700 text-slate-400'
                          : pref.choice === 'A'
                            ? 'bg-violet-900/50 text-violet-400'
                            : 'bg-violet-900/50 text-violet-400'
                      }`}
                    >
                      {pref.choice === 'tie' ? 'Tie' : `Response ${pref.choice}`}
                    </div>
                  </div>
                ))}
              </div>
            </div>

            {/* Model predictions */}
            <button
              onClick={() => setShowPredictions(!showPredictions)}
              className="w-full mb-4 px-4 py-2.5 bg-violet-900/30 hover:bg-violet-900/50 text-violet-400 border border-violet-700/50 rounded-lg transition-colors text-sm"
            >
              {showPredictions ? 'Hide' : 'Show'} Reward Model Predictions
            </button>

            {showPredictions && (
              <div className="bg-slate-900/50 rounded-lg p-4 mb-6">
                <div className="text-sm text-slate-400 mb-3">
                  Learned reward scores (higher = model predicts you'd prefer it):
                </div>
                <div className="space-y-3">
                  {responsePairs.map((pair) => {
                    const rA = predictReward(
                      pair.lengthA,
                      pair.specificity === 'A',
                      modelWeights
                    );
                    const rB = predictReward(
                      pair.lengthB,
                      pair.specificity === 'B',
                      modelWeights
                    );
                    const predicted = rA > rB ? 'A' : 'B';
                    const pref = preferences.find((p) => p.pairId === pair.id);
                    const correct = pref && pref.choice !== 'tie' && predicted === pref.choice;
                    return (
                      <div key={pair.id} className="flex items-center gap-3 text-sm">
                        <div className="flex-1 text-slate-400 truncate">{pair.prompt}</div>
                        <div className="flex gap-2 items-center">
                          <span className={`${rA > rB ? 'text-emerald-400' : 'text-slate-500'}`}>
                            A: {rA.toFixed(2)}
                          </span>
                          <span className="text-slate-600">vs</span>
                          <span className={`${rB > rA ? 'text-emerald-400' : 'text-slate-500'}`}>
                            B: {rB.toFixed(2)}
                          </span>
                          {pref && pref.choice !== 'tie' && (
                            <span className={`text-xs ${correct ? 'text-emerald-400' : 'text-red-400'}`}>
                              {correct ? '✓' : '✗'}
                            </span>
                          )}
                        </div>
                      </div>
                    );
                  })}
                </div>
              </div>
            )}

            {/* Insight */}
            <div className="rounded-lg p-4 bg-gradient-to-r from-violet-900/40 to-violet-900/10 border border-violet-700/50 text-violet-300">
              <div className="flex items-start gap-3">
                <span className="text-xl">💡</span>
                <div>
                  <div className="font-semibold text-violet-400">What the Reward Model Learned</div>
                  <div className="text-sm text-slate-300 mt-1">
                    From just {preferences.length} comparisons, the model learned patterns in your
                    preferences — like favoring detailed, specific answers over vague ones. Real RLHF
                    systems use thousands to millions of comparisons, but the principle is the same:
                    pairwise preferences become a scalar reward signal.
                  </div>
                </div>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

export default PreferenceLabeler;
