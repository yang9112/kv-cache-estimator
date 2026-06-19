// Model preset schema, default template, and normalization.
//
// Built-in (default) presets live as individual JSON files in ./models/ and are
// discovered at build time via Vite's import.meta.glob (see ./builtins.ts).
// Adding or removing a file there automatically adds or removes a preset — no
// central registry to edit. The UI can also export the current configuration
// as the same JSON shape (see ../lib/presets.ts). All produce the shape below.

export type AttentionType = 'standard' | 'mla' | 'hybrid';

export const ATTENTION_TYPES: AttentionType[] = ['standard', 'mla', 'hybrid'];

export interface ModelPreset {
  id: string;
  name: string;
  family: string;
  attentionType: AttentionType;
  parameters: number;       // total params in billions (B)
  layers: number;           // num_hidden_layers
  hiddenSize: number;       // hidden_size
  qHeads: number;           // num_attention_heads
  kvHeads: number;          // num_key_value_heads
  headDim?: number;         // explicit head_dim; if omitted, derived as hiddenSize / qHeads
  mlaDc?: number;           // MLA latent dim (kv_lora_rank / shared head_dim)
  mlaDr?: number;           // MLA RoPE dim (qk_rope_head_dim)
  fullAttnLayers?: number;  // hybrid: number of full-attention layers
  // MoE fields (verified against HF config.json where possible)
  isMoe?: boolean;
  numExperts?: number;      // n_routed_experts / num_local_experts
  moeInterSize?: number;    // moe_intermediate_size (per expert)
  moeLayers?: number;       // number of MoE layers (= total_layers - first_k_dense_replace)
}

// Default config template. Used by the "Custom" pseudo-preset and as the
// documented baseline a preset file is merged over. Optional architecture
// fields are intentionally absent so the calculator falls back to its live
// state when a preset does not specify them.
export const DEFAULT_PRESET_TEMPLATE: ModelPreset = {
  id: 'custom',
  name: 'Custom',
  family: 'Custom',
  attentionType: 'standard',
  parameters: 7,
  layers: 32,
  hiddenSize: 4096,
  qHeads: 32,
  kvHeads: 8,
};

// Display order of model families in the preset dropdown. Families not listed
// here are appended alphabetically; user presets always render in their own group.
export const FAMILY_ORDER: string[] = [
  'Custom',
  'DeepSeek',
  'GLM',
  'Moonshot (Kimi)',
  'MiniMax',
  'Qwen',
  'LLaMA',
];

// Family name used for user-saved presets (kept here so both the hook and the UI agree).
export const USER_PRESETS_FAMILY = 'My Presets';

const toNum = (v: unknown, fallback = 0): number => {
  const n = typeof v === 'string' ? parseFloat(v) : (v as number);
  return Number.isFinite(n) ? n : fallback;
};

// Validate a required numeric field: must be present, finite, and > 0.
// Catches typos / omitted fields in hand-edited preset JSON before they
// silently produce zero-byte results downstream.
const requirePositiveNum = (id: string, field: string, v: unknown): number => {
  if (v === undefined || v === null) {
    throw new Error(`Preset "${id}" is missing required field: ${field}`);
  }
  const n = typeof v === 'string' ? parseFloat(v) : (v as number);
  if (!Number.isFinite(n)) {
    throw new Error(`Preset "${id}" has non-numeric ${field}: ${String(v)}`);
  }
  if (n <= 0) {
    throw new Error(`Preset "${id}" has non-positive ${field}: ${n}`);
  }
  return n;
};
const toBool = (v: unknown): boolean | undefined =>
  typeof v === 'boolean' ? v : undefined;

/**
 * Normalize a raw preset object (from a JSON file) into a
 * validated ModelPreset. Required fields are coerced; optional fields are passed
 * through only when present, so the calculator's `?? prev` fallbacks keep
 * working when a preset omits an architecture field. Throws on missing required
 * fields or an invalid attentionType.
 */
export function normalizePreset(raw: unknown): ModelPreset {
  if (!raw || typeof raw !== 'object') {
    throw new Error('Preset must be a JSON object');
  }
  const r = raw as Record<string, unknown>;

  const id = typeof r.id === 'string' ? r.id : undefined;
  const name = typeof r.name === 'string' ? r.name : undefined;
  const family = typeof r.family === 'string' ? r.family : undefined;
  const attentionType =
    typeof r.attentionType === 'string' ? (r.attentionType as AttentionType) : undefined;

  if (!id || !name || !family) {
    const missing = [
      !id && 'id',
      !name && 'name',
      !family && 'family',
    ].filter(Boolean).join(', ');
    throw new Error(`Preset is missing required field(s): ${missing}`);
  }
  if (!attentionType || !ATTENTION_TYPES.includes(attentionType)) {
    throw new Error(
      `Preset "${id}" has invalid attentionType "${String(r.attentionType)}" (expected one of ${ATTENTION_TYPES.join(', ')})`,
    );
  }

  const preset: ModelPreset = {
    id,
    name,
    family,
    attentionType,
    parameters: requirePositiveNum(id, 'parameters', r.parameters),
    layers: requirePositiveNum(id, 'layers', r.layers),
    hiddenSize: requirePositiveNum(id, 'hiddenSize', r.hiddenSize),
    qHeads: requirePositiveNum(id, 'qHeads', r.qHeads),
    kvHeads: requirePositiveNum(id, 'kvHeads', r.kvHeads),
  };

  // Optional architecture fields — only set when the source provides them.
  if (r.headDim !== undefined && r.headDim !== null) preset.headDim = toNum(r.headDim);
  if (r.mlaDc !== undefined && r.mlaDc !== null) preset.mlaDc = toNum(r.mlaDc);
  if (r.mlaDr !== undefined && r.mlaDr !== null) preset.mlaDr = toNum(r.mlaDr);
  if (r.fullAttnLayers !== undefined && r.fullAttnLayers !== null) preset.fullAttnLayers = toNum(r.fullAttnLayers);

  // MoE fields
  const isMoe = toBool(r.isMoe);
  if (isMoe !== undefined) preset.isMoe = isMoe;
  if (r.numExperts !== undefined && r.numExperts !== null) preset.numExperts = toNum(r.numExperts);
  if (r.moeInterSize !== undefined && r.moeInterSize !== null) preset.moeInterSize = toNum(r.moeInterSize);
  if (r.moeLayers !== undefined && r.moeLayers !== null) preset.moeLayers = toNum(r.moeLayers);

  return preset;
}

/**
 * Drop presets with duplicate ids, keeping the first occurrence. Without this,
 * two files sharing an id would render duplicate <option> values in the
 * dropdown (React key warnings + the second option being unselectable).
 * Pure so it can be unit-tested under plain Node.
 */
export function dedupeById(
  presets: ModelPreset[],
  onDuplicate?: (kept: ModelPreset, dropped: ModelPreset) => void,
): ModelPreset[] {
  const seen = new Map<string, ModelPreset>();
  const result: ModelPreset[] = [];
  for (const p of presets) {
    const existing = seen.get(p.id);
    if (existing) {
      onDuplicate?.(existing, p);
    } else {
      seen.set(p.id, p);
      result.push(p);
    }
  }
  return result;
}
