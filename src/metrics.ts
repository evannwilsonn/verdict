export interface Scores {
  n: number;
  accuracy: number;
  accuracyLow: number;  // 95% Wilson interval
  accuracyHigh: number;
  precision: number;    // for the positive class
  recall: number;
  f1: number;
  truePos: number; falsePos: number; falseNeg: number; trueNeg: number;
}

/** Wilson score interval for a proportion: honest at the edges, unlike ±1.96·√(p(1−p)/n). */
export function wilson(successes: number, n: number, z = 1.96): [number, number] {
  if (n === 0) return [0, 0];
  const p = successes / n, z2 = z * z;
  const centre = (p + z2 / (2 * n)) / (1 + z2 / n);
  const half = (z / (1 + z2 / n)) * Math.sqrt((p * (1 - p)) / n + z2 / (4 * n * n));
  return [centre - half, centre + half];
}

/** Scores from log-odds (or any score) and a threshold: predicted positive when score ≥ threshold. */
export function evaluate(scores: readonly number[], labels: readonly (0 | 1)[], threshold = 0): Scores {
  let tp = 0, fp = 0, fn = 0, tn = 0;
  scores.forEach((s, i) => {
    const pred = s >= threshold, y = labels[i] === 1;
    if (pred && y) tp++; else if (pred) fp++; else if (y) fn++; else tn++;
  });
  const n = scores.length, correct = tp + tn;
  const precision = tp + fp ? tp / (tp + fp) : 0, recall = tp + fn ? tp / (tp + fn) : 0;
  const [lo, hi] = wilson(correct, n);
  return {
    n, accuracy: correct / n, accuracyLow: lo, accuracyHigh: hi, precision, recall,
    f1: precision + recall ? (2 * precision * recall) / (precision + recall) : 0,
    truePos: tp, falsePos: fp, falseNeg: fn, trueNeg: tn,
  };
}

/** The threshold that maximises accuracy on these scores (checked between every pair of sorted scores). */
export function bestThreshold(scores: readonly number[], labels: readonly (0 | 1)[]): number {
  const order = scores.map((s, i) => [s, labels[i]] as const).sort((a, b) => a[0] - b[0]);
  const pos = labels.filter(l => l === 1).length;
  // Threshold below everything: all predicted positive, so correct = positives.
  let correct = pos, best = correct, bestT = order[0][0] - 1e-9;
  for (let k = 0; k < order.length; k++) {
    correct += order[k][1] === 1 ? -1 : 1; // item k now falls below the threshold
    if (k + 1 < order.length && order[k + 1][0] === order[k][0]) continue;
    const t = k + 1 < order.length ? (order[k][0] + order[k + 1][0]) / 2 : order[k][0] + 1e-9;
    if (correct > best) { best = correct; bestT = t; }
  }
  return bestT;
}

/**
 * McNemar's exact test on paired predictions: is model B's accuracy different from A's on the
 * same reviews, beyond chance? Uses only the reviews where exactly one of them is right.
 */
export function mcnemar(aCorrect: readonly boolean[], bCorrect: readonly boolean[]): { onlyA: number; onlyB: number; p: number } {
  let onlyA = 0, onlyB = 0;
  aCorrect.forEach((a, i) => { if (a && !bCorrect[i]) onlyA++; else if (!a && bCorrect[i]) onlyB++; });
  const n = onlyA + onlyB, k = Math.min(onlyA, onlyB);
  // Two-sided exact binomial with p = 0.5, summed in log space to stay stable for large n.
  let logC = 0, tail = 0;
  const logHalfN = n * Math.log(0.5);
  for (let i = 0; i <= k; i++) {
    if (i > 0) logC += Math.log(n - i + 1) - Math.log(i);
    tail += Math.exp(logC + logHalfN);
  }
  return { onlyA, onlyB, p: Math.min(1, 2 * tail) };
}
