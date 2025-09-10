import fs from 'fs';
import path from 'path';
import url from 'url';
import { gini, makeLeaf, bestSplit, expandOne, anySplittable } from '../js/cart_core.js';

const __dirname = path.dirname(url.fileURLToPath(import.meta.url));
const dataPath = path.join(__dirname, '..', 'data', 'iris_subset.json');
const raw = JSON.parse(fs.readFileSync(dataPath, 'utf-8'));

const featureNames = raw.feature_names;
const targetNames = raw.target_names;
const nClasses = targetNames.length;
const root = makeLeaf(raw.samples, nClasses);

console.log('Root counts:', root.counts, 'gini=', root.impurity.toFixed(4));

const best = bestSplit(raw.samples, nClasses, featureNames);
console.log('Best initial split feature:', featureNames[best.featureIndex], 'threshold', best.threshold, 'impurity', best.impurity.toFixed(4));

let tree = root;
let steps = 0;
while (anySplittable(tree) && steps < 10) {
  tree = expandOne(tree, featureNames, nClasses);
  steps++;
}
console.log('Expanded steps:', steps);

function countLeaves(node) {
  if (node.type === 'leaf') return 1;
  return countLeaves(node.left) + countLeaves(node.right);
}

console.log('Leaf count:', countLeaves(tree));
console.log('Done.');
