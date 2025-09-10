// CART core logic (classification, Gini) for client-side visualization.
// Designed for the small 20-sample Iris subset.

export function gini(counts) {
  const total = counts.reduce((a, b) => a + b, 0);
  if (total === 0) return 0;
  return 1 - counts.reduce((sum, c) => sum + Math.pow(c / total, 2), 0);
}

export function classCounts(targets, nClasses) {
  const counts = Array(nClasses).fill(0);
  for (const t of targets) counts[t]++;
  return counts;
}

export function computeThresholds(values) {
  // unique sorted
  const uniq = Array.from(new Set(values.slice().sort((a, b) => a - b)));
  const thresholds = [];
  for (let i = 0; i < uniq.length - 1; i++) {
    thresholds.push((uniq[i] + uniq[i + 1]) / 2);
  }
  return thresholds;
}

export function evaluateSplit(featureValues, targets, threshold, nClasses) {
  const leftTargets = [];
  const rightTargets = [];
  for (let i = 0; i < featureValues.length; i++) {
    if (featureValues[i] <= threshold) leftTargets.push(targets[i]);
    else rightTargets.push(targets[i]);
  }
  const leftCounts = classCounts(leftTargets, nClasses);
  const rightCounts = classCounts(rightTargets, nClasses);
  const leftGini = gini(leftCounts);
  const rightGini = gini(rightCounts);
  const total = featureValues.length;
  const weighted = (leftTargets.length / total) * leftGini + (rightTargets.length / total) * rightGini;
  return {
    threshold,
    left: { size: leftTargets.length, counts: leftCounts, gini: leftGini },
    right: { size: rightTargets.length, counts: rightCounts, gini: rightGini },
    impurity: weighted
  };
}

export function candidateSplits(data, featureIndex, nClasses) {
  const featureValues = data.map(s => s.features[featureIndex]);
  const targets = data.map(s => s.target);
  const thresholds = computeThresholds(featureValues);
  return thresholds.map(th => evaluateSplit(featureValues, targets, th, nClasses));
}

export function bestSplitForFeature(data, featureIndex, nClasses) {
  const cands = candidateSplits(data, featureIndex, nClasses);
  if (cands.length === 0) return null;
  cands.sort((a, b) => a.impurity - b.impurity || a.threshold - b.threshold);
  return cands[0];
}

export function bestSplit(data, nClasses, featureNames) {
  let best = null;
  for (let f = 0; f < featureNames.length; f++) {
    const split = bestSplitForFeature(data, f, nClasses);
    if (!split) continue;
    if (!best || split.impurity < best.impurity || (split.impurity === best.impurity && f < best.featureIndex)) {
      best = { ...split, featureIndex: f };
    }
  }
  return best; // {featureIndex, threshold, left{...}, right{...}, impurity}
}

let nodeIdCounter = 0;
export function nextNodeId() { return ++nodeIdCounter; }

export function makeLeaf(data, nClasses) {
  const counts = classCounts(data.map(s => s.target), nClasses);
  const impurity = gini(counts);
  return {
    id: nextNodeId(),
    type: 'leaf',
    size: data.length,
    counts,
    impurity,
    samples: data
  };
}

export function trySplit(node, featureNames, nClasses, minSamplesSplit = 2) {
  if (node.type !== 'leaf') return node; // already split
  if (node.size < minSamplesSplit) return node;
  if (node.counts.filter(c => c > 0).length <= 1) return node; // pure
  const best = bestSplit(node.samples, nClasses, featureNames);
  if (!best) return node;

  const leftSamples = node.samples.filter(s => s.features[best.featureIndex] <= best.threshold);
  const rightSamples = node.samples.filter(s => s.features[best.featureIndex] > best.threshold);

  const leftNode = makeLeaf(leftSamples, nClasses);
  const rightNode = makeLeaf(rightSamples, nClasses);

  return {
    id: node.id,
    type: 'internal',
    featureIndex: best.featureIndex,
    featureName: featureNames[best.featureIndex],
    threshold: best.threshold,
    impurity: node.impurity,
    size: node.size,
    counts: node.counts,
    left: leftNode,
    right: rightNode,
    splitDetail: best
  };
}

export function anySplittable(node) {
  if (node.type === 'leaf') {
    return node.counts.filter(c => c > 0).length > 1 && node.size >= 2;
  }
  return anySplittable(node.left) || anySplittable(node.right);
}

export function expandOne(node, featureNames, nClasses) {
  if (node.type === 'leaf') {
    return trySplit(node, featureNames, nClasses);
  }
  // DFS preference left then right
  const left = expandOne(node.left, featureNames, nClasses);
  if (left !== node.left) return { ...node, left };
  const right = expandOne(node.right, featureNames, nClasses);
  if (right !== node.right) return { ...node, right };
  return node;
}

export function clone(obj) { return JSON.parse(JSON.stringify(obj)); }
