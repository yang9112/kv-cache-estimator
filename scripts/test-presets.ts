// Pure unit checks for the preset-management path: normalization, dedup,
// presetFromState round-trips, and family grouping. No Vite/glob, no React —
// runnable directly under Node via `npm test` (or `tsx scripts/test-presets.ts`).
import assert from 'node:assert/strict';
import {
  normalizePreset,
  dedupeById,
  DEFAULT_PRESET_TEMPLATE,
  type ModelPreset,
} from '../src/configs/schema';
import {
  presetFromState,
  groupPresets,
} from '../src/lib/presetUtils';
import type { CalculatorState } from '../src/types';

// ── helpers ─────────────────────────────────────────────────────────────────

function mkPreset(overrides: Partial<ModelPreset> = {}): ModelPreset {
  return { ...DEFAULT_PRESET_TEMPLATE, ...overrides };
}

function baseState(overrides: Partial<CalculatorState> = {}): CalculatorState {
  return {
    presetId: 'custom', attentionType: 'standard', parameters: 8, layers: 32,
    hiddenSize: 4096, qHeads: 32, kvHeads: 8, headDim: 0, mlaDc: 512, mlaDr: 64,
    fullAttnLayers: 20, seqLength: 8192, batchSize: 1, maxNumBatchedTokens: 8192,
    precision: 2, isMoe: false, numExperts: 0, moeInterSize: 0, moeLayers: 0,
    enableExpertParallel: false, tp: 1, pp: 1, gpuMemory: 80, gpuUtilization: 0.9,
    dp: 1, kvCacheDtype: 'auto', blockSize: 16, maxModelLen: 8192,
    enforceEager: false, enablePrefixCaching: true,
    ...overrides,
  };
}

let pass = 0;
let fail = 0;

function ok(condition: boolean, message: string) {
  if (condition) {
    pass++;
    console.log(`PASS  ${message}`);
  } else {
    fail++;
    console.log(`FAIL  ${message}`);
  }
}

function expectThrow(fn: () => void, message: string) {
  try {
    fn();
    ok(false, `${message} (did not throw)`);
  } catch {
    ok(true, message);
  }
}

// ── normalizePreset ─────────────────────────────────────────────────────────

console.log('=== normalizePreset: valid presets ===\n');

{
  const p = normalizePreset({
    id: 'x', name: 'X', family: 'F', attentionType: 'standard',
    parameters: 8, layers: 32, hiddenSize: 4096, qHeads: 32, kvHeads: 8,
  });
  ok(p.id === 'x' && p.layers === 32 && p.headDim === undefined, 'minimal valid preset normalizes');
}
{
  const p = normalizePreset({
    id: 'm', name: 'M', family: 'F', attentionType: 'mla',
    parameters: 671, layers: 61, hiddenSize: 7168, qHeads: 128, kvHeads: 128,
    mlaDc: 512, mlaDr: 0, headDim: 128,
  });
  ok(p.mlaDc === 512 && p.mlaDr === 0 && p.headDim === 128, 'optional fields kept (including 0)');
}

console.log('\n=== normalizePreset: validation throws ===\n');

expectThrow(
  () => normalizePreset({ id: 'x', name: 'X', family: 'F', attentionType: 'standard',
    parameters: 0, layers: 32, hiddenSize: 4096, qHeads: 32, kvHeads: 8 }),
  'parameters = 0 throws',
);
expectThrow(
  () => normalizePreset({ id: 'x', name: 'X', family: 'F', attentionType: 'standard',
    parameters: 8, layers: -1, hiddenSize: 4096, qHeads: 32, kvHeads: 8 }),
  'negative layers throws',
);
expectThrow(
  () => normalizePreset({ id: 'x', name: 'X', family: 'F', attentionType: 'standard',
    parameters: 8, layers: 32, hiddenSize: 4096, qHeads: 32 }),
  'missing kvHeads throws',
);
expectThrow(
  () => normalizePreset({ name: 'X', family: 'F', attentionType: 'standard',
    parameters: 8, layers: 32, hiddenSize: 4096, qHeads: 32, kvHeads: 8 }),
  'missing id throws',
);
expectThrow(
  () => normalizePreset({ id: 'x', name: 'X', family: 'F', attentionType: 'bogus',
    parameters: 8, layers: 32, hiddenSize: 4096, qHeads: 32, kvHeads: 8 }),
  'invalid attentionType throws',
);

// ── dedupeById ──────────────────────────────────────────────────────────────

console.log('\n=== dedupeById ===\n');

{
  const a = mkPreset({ id: 'dup', name: 'First' });
  const b = mkPreset({ id: 'dup', name: 'Second' });
  const c = mkPreset({ id: 'uniq', name: 'Only' });
  let dupReported = 0;
  const result = dedupeById([a, b, c], () => { dupReported++; });
  ok(result.length === 2, `keeps 2 of 3 (got ${result.length})`);
  ok(result.find((p) => p.id === 'dup')?.name === 'First', 'keeps first occurrence');
  ok(dupReported === 1, 'onDuplicate fired once');
}

// ── presetFromState round-trip ──────────────────────────────────────────────

console.log('\n=== presetFromState round-trip ===\n');

{
  // Standard
  const p = presetFromState('My 8B', baseState());
  const n = normalizePreset(p);
  ok(n.attentionType === 'standard' && n.layers === 32 && n.kvHeads === 8, 'standard: arch preserved');
  ok(n.headDim === undefined, 'standard: headDim=0 omitted');
  ok(n.isMoe === undefined, 'standard: isMoe omitted (not MoE)');
  ok(n.mlaDc === undefined, 'standard: mlaDc omitted');
  ok(n.id.startsWith('user-my-8b-'), 'standard: id slugified');
}
{
  // MLA — headDim must NOT be carried
  const s = baseState({ attentionType: 'mla', headDim: 128, mlaDc: 512, mlaDr: 64, kvHeads: 128 });
  const n = normalizePreset(presetFromState('DS', s));
  ok(n.attentionType === 'mla' && n.mlaDc === 512 && n.mlaDr === 64, 'mla: dc/dr preserved');
  ok(n.headDim === undefined, 'mla: headDim dropped');
}
{
  // Hybrid
  const s = baseState({ attentionType: 'hybrid', fullAttnLayers: 15, layers: 60 });
  const n = normalizePreset(presetFromState('Qwen Hybrid', s));
  ok(n.fullAttnLayers === 15, 'hybrid: fullAttnLayers preserved');
}
{
  // MoE
  const s = baseState({ isMoe: true, numExperts: 256, moeInterSize: 2048, moeLayers: 58 });
  const n = normalizePreset(presetFromState('MoE', s));
  ok(n.isMoe === true && n.numExperts === 256 && n.moeLayers === 58, 'moe: fields preserved');
}

// ── groupPresets ordering ───────────────────────────────────────────────────

console.log('\n=== groupPresets ordering ===\n');

{
  const custom = mkPreset({ id: 'custom', family: 'Custom' });
  const builtins = [
    mkPreset({ id: 'llama', name: 'LLaMA 3', family: 'LLaMA' }),
    mkPreset({ id: 'ds', name: 'DeepSeek V3', family: 'DeepSeek' }),
  ];
  const users = [
    mkPreset({ id: 'u1', name: 'Zeta', family: 'My Presets' }),
    mkPreset({ id: 'u2', name: 'Alpha', family: 'My Presets' }),
  ];
  const g = groupPresets(users, builtins, custom);
  const families = Object.keys(g);
  ok(families[0] === 'Custom', `Custom first (got ${families[0]})`);
  ok(families[1] === 'My Presets', `My Presets second (got ${families[1]})`);
  ok(g['My Presets'][0].name === 'Alpha', 'user presets sorted by name (Alpha first)');
  ok(families.includes('DeepSeek') && families.includes('LLaMA'), 'builtin families present');
  ok(families.indexOf('Custom') < families.indexOf('DeepSeek'), 'Custom before builtins');
}

// ── summary ─────────────────────────────────────────────────────────────────

console.log(`\n=== RESULTS: ${pass} passed, ${fail} failed ===`);
process.exit(fail > 0 ? 1 : 0);
