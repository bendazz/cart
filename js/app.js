// Simplified app: only feature projection visualization with regeneratable stratified 20-sample subset.
import { featureNames as FULL_FEATURES, targetNames as FULL_TARGETS, buildSamples } from './iris_full.js';

const COLORS = [
  '#2d7dd2', // class 0
  '#f4a261', // class 1
  '#2a9d8f'  // class 2
];

const state = {
  data: null,
  featureNames: [],
  targetNames: [],
  nClasses: 0,
  // thresholds[i] holds numeric threshold value for feature i (in original feature units)
  thresholds: {},
  // midpoints[i] = sorted array of candidate split midpoints between distinct adjacent values
  midpoints: {},
  // scaleRanges[i] = {scaledMin, scaledMax} for mapping back and forth
  scaleRanges: {},
  dragging: {
    active: false,
    featureIndex: null,
    container: null,
  },
  activeFeature: 0,
  bestSplit: null,
};

function seededRng(seed) {
  // simple LCG
  let s = seed >>> 0;
  return () => {
    s = (s * 1664525 + 1013904223) >>> 0;
    return s / 0x100000000;
  };
}

function stratifiedSubset(fullSamples, nPerClassTotal = 20, seed = 42) {
  const byClass = new Map();
  for (const s of fullSamples) {
    if (!byClass.has(s.target)) byClass.set(s.target, []);
    byClass.get(s.target).push(s);
  }
  const nClasses = byClass.size;
  const base = Math.floor(nPerClassTotal / nClasses);
  let remainder = nPerClassTotal - base * nClasses;
  const allocation = [];
  for (let c = 0; c < nClasses; c++) {
    allocation[c] = base + (remainder > 0 ? 1 : 0);
    if (remainder > 0) remainder--;
  }
  const rng = seededRng(seed);
  const subset = [];
  for (let c = 0; c < nClasses; c++) {
    const arr = byClass.get(c).slice();
    // shuffle Fisher-Yates using rng
    for (let i = arr.length - 1; i > 0; i--) {
      const j = Math.floor(rng() * (i + 1));
      [arr[i], arr[j]] = [arr[j], arr[i]];
    }
    subset.push(...arr.slice(0, allocation[c]));
  }
  // sort by original index just for consistency
  subset.sort((a,b)=>a.index - b.index);
  return subset;
}

function loadData(seed = 42) {
  const full = buildSamples();
  const subset = stratifiedSubset(full, 20, seed);
  state.data = {
    feature_names: FULL_FEATURES.slice(),
    target_names: FULL_TARGETS.slice(),
    samples: subset
  };
  state.featureNames = state.data.feature_names;
  state.targetNames = state.data.target_names;
  state.nClasses = state.targetNames.length;
}

// Removed legacy CART-related logic.

function renderAll() { renderAllProjections(); }

async function init() {
  loadData(42);
  buildLegend();
  buildFeatureButtons();
  // Add feature titles to rows
  document.querySelectorAll('.projection-row').forEach(row => {
    const idx = parseInt(row.getAttribute('data-feature-index'), 10);
    const titleEl = row.querySelector('.projection-title');
    if (state.featureNames[idx]) titleEl.textContent = state.featureNames[idx];
  });
  renderAll();
  updateActiveFeatureStyles();
  window.addEventListener('resize', () => { renderAllProjections(); });
}

init().catch(err => {
  console.error(err);
  document.body.innerHTML = `<pre style="color:red">Initialization failed: ${err.message}</pre>`;
});

// Regenerate subset public function (will be wired to UI controls later)
function regenerateSubset(seed) {
  // reset state pieces that depend on data
  state.thresholds = {};
  state.midpoints = {};
  state.scaleRanges = {};
  state.bestSplit = null;
  loadData(seed);
  // update legend & feature buttons if first time or feature names same
  buildLegend();
  buildFeatureButtons();
  // update projection titles
  document.querySelectorAll('.projection-row').forEach(row => {
    const idx = parseInt(row.getAttribute('data-feature-index'), 10);
    const titleEl = row.querySelector('.projection-title');
    if (state.featureNames[idx]) titleEl.textContent = state.featureNames[idx];
  });
  renderAllProjections();
  updateActiveFeatureStyles();
  // re-enable best split button if exists
  const btn = document.getElementById('show-best-split');
  const out = document.getElementById('best-split-output');
  if (btn) btn.disabled = false;
  if (out) { out.hidden = true; out.textContent=''; }
}
window.regenerateSubset = regenerateSubset;

// Projection rendering
function buildLegend() {
  const legend = document.getElementById('projection-legend');
  legend.innerHTML = state.targetNames.map((n,i)=>`<span class="legend-item"><span class="legend-swatch" style="background:${COLORS[i]}"></span>${n}</span>`).join('');
}

function buildFeatureButtons() {
  const bar = document.getElementById('feature-select-bar');
  if (!bar) return;
  bar.innerHTML = '';
  state.featureNames.forEach((name, idx) => {
    const btn = document.createElement('button');
    btn.className = 'feature-btn';
    btn.textContent = name;
    btn.dataset.idx = idx;
    if (idx === state.activeFeature) btn.classList.add('active');
    btn.addEventListener('click', () => {
      state.activeFeature = idx;
      updateActiveFeatureStyles();
    });
    bar.appendChild(btn);
  });
}

function updateActiveFeatureStyles() {
  document.querySelectorAll('.feature-btn').forEach(b => {
    const idx = parseInt(b.dataset.idx, 10);
    b.classList.toggle('active', idx === state.activeFeature);
  });
  document.querySelectorAll('.threshold-line').forEach(line => {
    const f = parseInt(line.dataset.featureIndex, 10);
    line.classList.toggle('active', f === state.activeFeature);
  });
  // refresh gini labels
  refreshGiniForActive();
  updateGlobalGiniDisplay();
}

function renderFeatureProjection(rowEl, featureIndex) {
  const container = rowEl.querySelector('.projection-container');
  if (!container) return;
  const featureName = state.featureNames[featureIndex];
  const samples = state.data.samples.slice();
  samples.sort((a,b)=> a.features[featureIndex] - b.features[featureIndex]);
  const values = samples.map(s => s.features[featureIndex]);
  // Compute unique sorted values
  const uniqueVals = [];
  for (const v of values) {
    if (uniqueVals.length === 0 || uniqueVals[uniqueVals.length-1] !== v) uniqueVals.push(v);
  }
  // Midpoints between consecutive unique values
  const mids = [];
  for (let i=0;i<uniqueVals.length-1;i++) {
    mids.push((uniqueVals[i] + uniqueVals[i+1]) / 2);
  }
  state.midpoints[featureIndex] = mids;
  const min = Math.min(...values);
  const max = Math.max(...values);
  const pad = (max - min) * 0.05 || 1;
  const scaledMin = min - pad; const scaledMax = max + pad;
  state.scaleRanges[featureIndex] = { scaledMin, scaledMax };
  container.innerHTML = '';
  const line = document.createElement('div');
  line.className = 'number-line';
  container.appendChild(line);
  const ticks = 6;
  for (let i=0;i<ticks;i++) {
    const frac = i/(ticks-1);
    const xPct = frac*100;
    const v = scaledMin + frac*(scaledMax-scaledMin);
    const tick = document.createElement('div');
    tick.className = 'tick';
    tick.style.left = `calc(${xPct}% + 8px - 0px)`;
    container.appendChild(tick);
    const label = document.createElement('div');
    label.className = 'tick-label';
    label.style.left = `calc(${xPct}% + 8px)`;
    label.textContent = v.toFixed(2);
    container.appendChild(label);
  }
  samples.forEach(s => {
    const val = s.features[featureIndex];
    const frac = (val - scaledMin) / (scaledMax - scaledMin);
    const circle = document.createElement('div');
    circle.className = 'sample-circle';
    circle.style.left = `calc(${(frac*100)}% + 8px)`;
    circle.style.background = COLORS[s.target];
    circle.title = `idx ${s.index}\n${featureName}=${val}\nclass=${state.targetNames[s.target]}`;
    circle.textContent = s.target;
    container.appendChild(circle);
  });

  // Threshold line (create if nonexistent; default to central midpoint or middle of range)
  if (state.thresholds[featureIndex] == null) {
    if (mids.length > 0) {
      const midIdx = Math.floor(mids.length / 2);
      state.thresholds[featureIndex] = mids[midIdx];
    } else {
      state.thresholds[featureIndex] = (min + max) / 2;
    }
  }
  const thrVal = state.thresholds[featureIndex];
  const thrFrac = (thrVal - scaledMin) / (scaledMax - scaledMin);
  const lineEl = document.createElement('div');
  lineEl.className = 'threshold-line';
  lineEl.style.left = `calc(${(thrFrac*100)}% + 8px)`;
  lineEl.dataset.featureIndex = featureIndex;
  const labelEl = document.createElement('div');
  labelEl.className = 'threshold-handle';
  labelEl.textContent = formatThresholdLabel(featureIndex, thrVal);
  lineEl.appendChild(labelEl);
  container.appendChild(lineEl);

  lineEl.addEventListener('pointerdown', (e) => startDrag(e, featureIndex, container, lineEl));
}

function renderAllProjections() {
  if (!state.data) return;
  document.querySelectorAll('.projection-row').forEach(row => {
    const idx = parseInt(row.getAttribute('data-feature-index'), 10);
    if (!isNaN(idx)) renderFeatureProjection(row, idx);
  });
  // Recompute best split after rendering (midpoints prepared)
  computeBestSplit();
  updateGlobalGiniDisplay();
}

function startDrag(e, featureIndex, container, lineEl) {
  e.preventDefault();
  state.dragging.active = true;
  state.dragging.featureIndex = featureIndex;
  state.dragging.container = container;
  lineEl.classList.add('dragging');
  window.addEventListener('pointermove', onDragMove);
  window.addEventListener('pointerup', endDrag, { once: true });
}

function onDragMove(e) {
  if (!state.dragging.active) return;
  const { featureIndex, container } = state.dragging;
  const isActive = featureIndex === state.activeFeature;
  const range = state.scaleRanges[featureIndex];
  if (!range) return;
  const rect = container.getBoundingClientRect();
  // container has 8px left padding before scale starts
  const x = e.clientX - rect.left - 8; // adjust for padding
  const width = rect.width - 16; // minus left+right padding
  const clamped = Math.max(0, Math.min(width, x));
  const frac = clamped / width;
  let value = range.scaledMin + frac * (range.scaledMax - range.scaledMin);
  // Snap to nearest midpoint if available
  value = snapToMidpoint(featureIndex, value);
  state.thresholds[featureIndex] = value;
  // update line position & label
  const lineEl = container.querySelector('.threshold-line');
  if (lineEl) {
    // recompute fractional position after snapping
    const snappedFrac = (value - range.scaledMin) / (range.scaledMax - range.scaledMin);
    lineEl.style.left = `calc(${(snappedFrac*100)}% + 8px)`;
    if (isActive) {
      const label = lineEl.querySelector('.threshold-handle');
      if (label) label.textContent = formatThresholdLabel(featureIndex, value);
  updateGlobalGiniDisplay();
    }
  }
}

function endDrag() {
  const { featureIndex, container } = state.dragging;
  const lineEl = container?.querySelector('.threshold-line');
  if (lineEl) lineEl.classList.remove('dragging');
  state.dragging.active = false;
  state.dragging.featureIndex = null;
  state.dragging.container = null;
  window.removeEventListener('pointermove', onDragMove);
}

// --- Gini calculations for active feature ---
function giniFromCounts(counts) {
  const total = counts.reduce((a,b)=>a+b,0);
  if (total === 0) return 0;
  let sumSq = 0;
  for (const c of counts) { const p = c/total; sumSq += p*p; }
  return 1 - sumSq;
}

function countsForSide(samples, featureIndex, threshold, side) {
  const counts = Array(state.nClasses).fill(0);
  for (const s of samples) {
    const v = s.features[featureIndex];
    if (side === 'left') {
      if (v <= threshold) counts[s.target]++;
    } else {
      if (v > threshold) counts[s.target]++;
    }
  }
  return counts;
}

function giniForFeatureThreshold(featureIndex, threshold) {
  const samples = state.data.samples;
  const leftCounts = countsForSide(samples, featureIndex, threshold, 'left');
  const rightCounts = countsForSide(samples, featureIndex, threshold, 'right');
  const leftN = leftCounts.reduce((a,b)=>a+b,0);
  const rightN = rightCounts.reduce((a,b)=>a+b,0);
  const total = leftN + rightN;
  const gLeft = giniFromCounts(leftCounts);
  const gRight = giniFromCounts(rightCounts);
  const weighted = (leftN/total)*gLeft + (rightN/total)*gRight;
  return { weighted, leftN, rightN, gLeft, gRight };
}

function formatThresholdLabel(featureIndex, threshold) {
  if (featureIndex !== state.activeFeature) return threshold.toFixed(2);
  const { weighted, leftN, rightN } = giniForFeatureThreshold(featureIndex, threshold);
  return `${threshold.toFixed(2)} | G=${weighted.toFixed(4)} (L${leftN}/R${rightN})`;
}

function refreshGiniForActive() {
  const featureIndex = state.activeFeature;
  const container = document.querySelector(`.projection-row[data-feature-index="${featureIndex}"] .projection-container`);
  if (!container) return;
  const lineEl = container.querySelector('.threshold-line');
  if (!lineEl) return;
  const val = state.thresholds[featureIndex];
  const label = lineEl.querySelector('.threshold-handle');
  if (label) label.textContent = formatThresholdLabel(featureIndex, val);
}

function updateGlobalGiniDisplay() {
  const el = document.getElementById('global-gini');
  if (!el) return;
  const valSpan = el.querySelector('.gini-value');
  if (!state.data || state.thresholds[state.activeFeature] == null) {
    if (valSpan) valSpan.textContent = '--';
    return;
  }
  const thr = state.thresholds[state.activeFeature];
  const { weighted } = giniForFeatureThreshold(state.activeFeature, thr);
  if (valSpan) valSpan.textContent = weighted.toFixed(4);
}

// --- Best split computation (mimicking scikit-learn first split) ---
function computeParentGini() {
  const counts = Array(state.nClasses).fill(0);
  for (const s of state.data.samples) counts[s.target]++;
  return giniFromCounts(counts);
}

function computeBestSplit() {
  if (!state.data) return;
  const parentG = computeParentGini();
  let best = null;
  for (let f = 0; f < state.featureNames.length; f++) {
    const mids = state.midpoints[f];
    if (!mids || mids.length === 0) continue;
    for (const m of mids) {
      const detail = giniForFeatureThreshold(f, m);
      const gain = parentG - detail.weighted;
      if (!best || detail.weighted < best.weighted || (detail.weighted === best.weighted && f < best.featureIndex)) {
        best = {
          featureIndex: f,
            featureName: state.featureNames[f],
          threshold: m,
          weighted: detail.weighted,
          leftN: detail.leftN,
          rightN: detail.rightN,
          gLeft: detail.gLeft,
          gRight: detail.gRight,
          gain,
          parentG
        };
      }
    }
  }
  state.bestSplit = best;
  return best;
}

// Button handler to reveal best split
document.addEventListener('DOMContentLoaded', () => {
  const btn = document.getElementById('show-best-split');
  const out = document.getElementById('best-split-output');
  if (!btn || !out) return;
  btn.addEventListener('click', () => {
    if (!state.bestSplit) computeBestSplit();
    if (!state.bestSplit) {
      out.textContent = 'No valid split found.';
    } else {
      const b = state.bestSplit;
      out.innerHTML = `Feature: <strong>${b.featureName}</strong><br>`+
        `Threshold: ${b.threshold.toFixed(3)}<br>`+
        `Parent Gini: ${b.parentG.toFixed(4)}<br>`+
        `Child G (weighted): ${b.weighted.toFixed(4)}<br>`+
        `Gain (reduction): ${(b.gain).toFixed(4)}<br>`+
        `Left: N=${b.leftN}, G=${b.gLeft.toFixed(4)} | Right: N=${b.rightN}, G=${b.gRight.toFixed(4)}`;
    }
    out.hidden = false;
    btn.disabled = true;
  });
});

// --- Snapping helper ---
function snapToMidpoint(featureIndex, rawValue) {
  const mids = state.midpoints[featureIndex];
  if (!mids || mids.length === 0) return rawValue; // nothing to snap to
  // Binary search for nearest midpoint
  let lo = 0, hi = mids.length - 1;
  if (rawValue <= mids[lo]) return mids[lo];
  if (rawValue >= mids[hi]) return mids[hi];
  while (hi - lo > 1) {
    const mid = Math.floor((lo + hi)/2);
    if (mids[mid] === rawValue) return rawValue;
    if (mids[mid] < rawValue) lo = mid; else hi = mid;
  }
  // rawValue between mids[lo] and mids[hi]; choose closer
  return (rawValue - mids[lo]) <= (mids[hi] - rawValue) ? mids[lo] : mids[hi];
}
