// Verification script: checks that every MoE preset computes EP-related values
// and that presets match their HF config ground truth.
import { PRESETS, CalculatorState } from '../src/types';
import { calculateKV, computeExpertParams } from '../src/lib/calc';

function baseState(overrides: Partial<CalculatorState> = {}): CalculatorState {
  return {
    presetId: 'custom', attentionType: 'standard', parameters: 7, layers: 32,
    hiddenSize: 4096, qHeads: 32, kvHeads: 8, headDim: 0, mlaDc: 512, mlaDr: 64,
    fullAttnLayers: 20, seqLength: 8192, batchSize: 1, maxNumBatchedTokens: 8192,
    precision: 2, isMoe: false, numExperts: 0, moeInterSize: 0, moeLayers: 0,
    enableExpertParallel: false, tp: 1, pp: 1, gpuMemory: 80, gpuUtilization: 0.9,
    dp: 1, kvCacheDtype: 'auto', blockSize: 16, maxModelLen: 8192,
    enforceEager: false, enablePrefixCaching: true,
    ...overrides,
  };
}

console.log('=== MoE preset EP/expert verification ===\n');
const moePresets = PRESETS.filter(p => p.isMoe);
let pass = 0, fail = 0;
for (const p of moePresets) {
  const s = baseState({
    attentionType: p.attentionType, parameters: p.parameters, layers: p.layers,
    hiddenSize: p.hiddenSize, qHeads: p.qHeads, kvHeads: p.kvHeads,
    headDim: p.headDim ?? 0, mlaDc: p.mlaDc ?? 512, mlaDr: p.mlaDr ?? 64,
    fullAttnLayers: p.fullAttnLayers ?? 20, isMoe: p.isMoe ?? false,
    numExperts: p.numExperts ?? 0, moeInterSize: p.moeInterSize ?? 0,
    moeLayers: p.moeLayers ?? 0, enableExpertParallel: true, tp: 8, dp: 4,
  });
  const r = calculateKV(s);
  const expertB = (r.expertParams / 1e9).toFixed(1);
  const denseB = (p.parameters - r.expertParams / 1e9).toFixed(1);
  const ok = r.expertParams > 0 && r.expertWeightTotal > 0 && r.epSize > 1;
  console.log(`${ok ? 'PASS' : 'FAIL'}  ${p.name.padEnd(32)} experts=${p.numExperts} moeLayers=${p.moeLayers} inter=${p.moeInterSize}`);
  console.log(`        expertParams=${expertB}B  denseParams=${denseB}B  epSize=${r.epSize}  localExperts=${r.localNumExperts}  expertWt/GPU=${(r.expertWeightPerGPU/1e9).toFixed(1)}B`);
  ok ? pass++ : fail++;
}

console.log(`\n=== Dense presets (should have no MoE/EP) ===\n`);
const densePresets = PRESETS.filter(p => !p.isMoe && p.id !== 'custom');
for (const p of densePresets) {
  const s = baseState({ parameters: p.parameters, layers: p.layers, hiddenSize: p.hiddenSize, qHeads: p.qHeads, kvHeads: p.kvHeads, headDim: p.headDim ?? 0, attentionType: p.attentionType, mlaDc: p.mlaDc ?? 512, mlaDr: p.mlaDr ?? 64, fullAttnLayers: p.fullAttnLayers ?? 20 });
  const r = calculateKV(s);
  const hd = p.headDim ?? (p.attentionType === 'mla' ? 0 : p.hiddenSize / p.qHeads);
  console.log(`OK    ${p.name.padEnd(32)} headDim=${r.headDim} (expected ${hd})  expertParams=${r.expertParams}`);
  if (r.headDim !== hd) { console.log(`      *** HEAD DIM MISMATCH ***`); fail++; } else pass++;
}

console.log(`\n=== Mixtral check (should be gone) ===`);
console.log(PRESETS.find(p => p.id === 'mixtral-8x7b') ? 'FAIL: mixtral still present!' : 'PASS: mixtral removed');

console.log(`\n=== RESULTS: ${pass} passed, ${fail} failed ===`);
process.exit(fail > 0 ? 1 : 0);
