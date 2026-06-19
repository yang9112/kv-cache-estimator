// Verification script: reads every JSON preset file in src/configs/models/ and
// checks that MoE presets compute EP/expert values and dense presets resolve the
// expected head_dim. Presets are loaded straight from the config folder (not via
// the Vite glob) so this runs under plain Node/tsx without a build step.
//
// Run:  npm run verify   (or: npx tsx scripts/verify-presets.ts)
import { readdirSync, readFileSync } from 'node:fs';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import { calculateKV, computeExpertParams } from '../src/lib/calc';
import type { CalculatorState } from '../src/types';
import { normalizePreset, type ModelPreset } from '../src/configs/schema';

const __dirname = dirname(fileURLToPath(import.meta.url));
const modelsDir = join(__dirname, '..', 'src', 'configs', 'models');

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

const files = readdirSync(modelsDir).filter((f) => f.endsWith('.json'));
const presets: ModelPreset[] = files.map((f) => {
  const raw = JSON.parse(readFileSync(join(modelsDir, f), 'utf8'));
  return normalizePreset(raw);
});

console.log(`Loaded ${presets.length} preset files from ${modelsDir}\n`);

let pass = 0;
let fail = 0;

console.log('=== MoE preset EP/expert verification ===\n');
const moePresets = presets.filter((p) => p.isMoe);
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
  console.log(`        expertParams=${expertB}B  denseParams=${denseB}B  epSize=${r.epSize}  localExperts=${r.localNumExperts}  expertWt/GPU=${(r.expertWeightPerGPU / 1e9).toFixed(1)}B`);
  ok ? pass++ : fail++;
}

console.log(`\n=== Dense presets (should have no MoE/EP) ===\n`);
const densePresets = presets.filter((p) => !p.isMoe && p.id !== 'custom');
for (const p of densePresets) {
  const s = baseState({
    parameters: p.parameters, layers: p.layers, hiddenSize: p.hiddenSize,
    qHeads: p.qHeads, kvHeads: p.kvHeads, headDim: p.headDim ?? 0,
    attentionType: p.attentionType, mlaDc: p.mlaDc ?? 512, mlaDr: p.mlaDr ?? 64,
    fullAttnLayers: p.fullAttnLayers ?? 20,
  });
  const r = calculateKV(s);
  const hd = p.headDim ?? (p.attentionType === 'mla' ? 0 : p.hiddenSize / p.qHeads);
  console.log(`OK    ${p.name.padEnd(32)} headDim=${r.headDim} (expected ${hd})  expertParams=${r.expertParams}`);
  if (r.headDim !== hd) {
    console.log(`      *** HEAD DIM MISMATCH ***`);
    fail++;
  } else {
    pass++;
  }
}

console.log(`\n=== RESULTS: ${pass} passed, ${fail} failed ===`);
process.exit(fail > 0 ? 1 : 0);
