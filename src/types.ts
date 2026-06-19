export type Precision = {
  label: string;
  bytes: number;
};

export const PRECISIONS: Precision[] = [
  { label: 'FP32 (32-bit float)', bytes: 4 },
  { label: 'FP16 / BF16 (16-bit float)', bytes: 2 },
  { label: 'INT8 / FP8 (8-bit int)', bytes: 1 },
  { label: 'INT4 (4-bit int)', bytes: 0.5 },
];

export type KVCacheDType = {
  label: string;
  key: string;
  bytesPerParam: number; // -1 means "same as model precision"
  hasPerTokenHeadScales: boolean;
};

// Based on vLLM CacheConfig.cache_dtype and KVQuantMode
export const KV_CACHE_DTYPES: KVCacheDType[] = [
  { label: 'Auto (same as model)', key: 'auto', bytesPerParam: -1, hasPerTokenHeadScales: false },
  { label: 'FP8 E4M3 (per-tensor scale)', key: 'fp8_e4m3', bytesPerParam: 1, hasPerTokenHeadScales: false },
  { label: 'FP8 (per-token-head scale)', key: 'fp8_per_token_head', bytesPerParam: 1, hasPerTokenHeadScales: true },
  { label: 'INT8 (per-token-head scale)', key: 'int8_per_token_head', bytesPerParam: 1, hasPerTokenHeadScales: true },
];

export type ModelPreset = {
  id: string;
  name: string;
  family: string;
  attentionType: 'standard' | 'mla' | 'hybrid';
  parameters: number;
  layers: number;
  hiddenSize: number;
  qHeads: number;
  kvHeads: number;
  headDim?: number;       // explicit head dim; if omitted, derived as hiddenSize / qHeads
  mlaDc?: number;
  mlaDr?: number;
  fullAttnLayers?: number;
  // MoE fields (verified against HF config.json where possible)
  isMoe?: boolean;
  numExperts?: number;      // n_routed_experts
  moeInterSize?: number;    // moe_intermediate_size (per expert)
  moeLayers?: number;       // number of MoE layers (= total_layers - first_k_dense_replace)
};

export const PRESETS: ModelPreset[] = [
  {
    id: 'custom',
    name: 'Custom',
    family: 'Custom',
    attentionType: 'standard',
    parameters: 7,
    layers: 32,
    hiddenSize: 4096,
    qHeads: 32,
    kvHeads: 8,
  },
  // ── DeepSeek ────────────────────────────────────────────────────────────
  // V4 uses a new DSA attention: num_kv_heads=1 with a SHARED K==V tensor of
  // head_dim (config.head_dim, e.g. 512). KV cache per layer = head_dim bytes.
  // Represented here as attentionType 'mla' with mlaDc=head_dim, mlaDr=0 so the
  // existing (dc+dr) formula yields head_dim without a schema change.
  // Source: huggingface transformers deepseek_v4/modeling_deepseek_v4.py.
  {
    id: 'deepseek-v4-pro',
    name: 'DeepSeek V4 Pro (1.6T)',
    family: 'DeepSeek',
    attentionType: 'mla',
    parameters: 1600,      // 1.6T total (49B activated) — official V4 README
    layers: 61,
    hiddenSize: 7168,
    qHeads: 128,
    kvHeads: 1,
    mlaDc: 512,            // = config.head_dim (shared K==V dim)
    mlaDr: 0,              // V4 has no separate RoPE slice in the KV path
    isMoe: true,
    numExperts: 384,       // n_routed_experts
    moeInterSize: 3072,    // moe_intermediate_size
    moeLayers: 61,         // all layers are MoE (no first_k_dense_replace)
  },
  {
    id: 'deepseek-v4-flash',
    name: 'DeepSeek V4 Flash (284B)',
    family: 'DeepSeek',
    attentionType: 'mla',
    parameters: 284,       // 284B total (13B activated) — official V4 README
    layers: 43,
    hiddenSize: 4096,
    qHeads: 64,
    kvHeads: 1,
    mlaDc: 512,            // = config.head_dim
    mlaDr: 0,
    isMoe: true,
    numExperts: 256,
    moeInterSize: 2048,
    moeLayers: 43,         // all layers are MoE
  },
  {
    id: 'deepseek-v3',
    name: 'DeepSeek V3 / R1 (671B)',
    family: 'DeepSeek',
    attentionType: 'mla',
    parameters: 671,
    layers: 61,
    hiddenSize: 7168,
    qHeads: 128,
    kvHeads: 128,
    mlaDc: 512,            // kv_lora_rank
    mlaDr: 64,             // qk_rope_head_dim
    isMoe: true,
    numExperts: 256,       // n_routed_experts
    moeInterSize: 2048,    // moe_intermediate_size
    moeLayers: 58,         // 61 - first_k_dense_replace(3)
  },
  {
    id: 'deepseek-v2',
    name: 'DeepSeek V2 (236B)',
    family: 'DeepSeek',
    attentionType: 'mla',
    parameters: 236,
    layers: 60,
    hiddenSize: 5120,
    qHeads: 128,
    kvHeads: 128,
    mlaDc: 512,            // kv_lora_rank
    mlaDr: 64,             // qk_rope_head_dim
    isMoe: true,
    numExperts: 160,       // n_routed_experts
    moeInterSize: 1536,    // moe_intermediate_size
    moeLayers: 59,         // 60 - first_k_dense_replace(1)
  },
  // ── GLM ──────────────────────────────────────────────────────────────────
  {
    id: 'glm-5',
    name: 'GLM 5 / 5.1 (754B)',
    family: 'GLM',
    attentionType: 'mla',
    parameters: 754,       // 753.86B — HF safetensors.total
    layers: 78,
    hiddenSize: 6144,
    qHeads: 64,
    kvHeads: 64,
    mlaDc: 512,            // kv_lora_rank
    mlaDr: 64,             // qk_rope_head_dim
    isMoe: true,
    numExperts: 256,       // n_routed_experts
    moeInterSize: 2048,    // moe_intermediate_size
    moeLayers: 75,         // 78 - first_k_dense_replace(3)
  },
  {
    id: 'glm-4-flash',
    name: 'GLM 4.7 Flash (30B-A3B)',
    family: 'GLM',
    attentionType: 'mla',
    parameters: 30,        // 31.22B total — HF safetensors.total (3B activated)
    layers: 47,
    hiddenSize: 2048,
    qHeads: 20,
    kvHeads: 20,
    mlaDc: 512,            // kv_lora_rank
    mlaDr: 64,             // qk_rope_head_dim
    isMoe: true,
    numExperts: 64,        // n_routed_experts
    moeInterSize: 1536,    // moe_intermediate_size
    moeLayers: 46,         // 47 - first_k_dense_replace(1)
  },
  {
    id: 'glm-4-9b',
    name: 'GLM 4 (9B)',
    family: 'GLM',
    attentionType: 'standard',
    parameters: 9,
    layers: 40,
    hiddenSize: 4096,
    qHeads: 32,
    kvHeads: 2,            // multi_query_group_num (MQA-style)
  },
  // ── Moonshot (Kimi) ──────────────────────────────────────────────────────
  // K2.5 reuses the DeepSeek-V3 text arch (MLA + MoE) under a multimodal head;
  // all architecture numbers below come from text_config of Kimi-K2.5.
  {
    id: 'kimi-k2.5',
    name: 'Kimi K2.5 / K2 (1T)',
    family: 'Moonshot (Kimi)',
    attentionType: 'mla',
    parameters: 1060,      // ~1.06T total (32B activated) — HF safetensors.total=1058.6B
    layers: 61,
    hiddenSize: 7168,
    qHeads: 64,
    kvHeads: 64,
    mlaDc: 512,            // kv_lora_rank
    mlaDr: 64,             // qk_rope_head_dim
    isMoe: true,
    numExperts: 384,       // n_routed_experts
    moeInterSize: 2048,    // moe_intermediate_size
    moeLayers: 60,         // 61 - first_k_dense_replace(1)
  },
  // ── MiniMax ──────────────────────────────────────────────────────────────
  // M2/M2.5: GQA attention with EXPLICIT head_dim=128 (not hidden/qHeads=64).
  // All 62 layers are full attention (attn_type_list all = 1), MoE every layer.
  {
    id: 'minimax-m2.5',
    name: 'MiniMax M2.5 / M2 (~230B)',
    family: 'MiniMax',
    attentionType: 'standard',
    parameters: 230,       // 228.7B — HF safetensors.total
    layers: 62,
    hiddenSize: 3072,
    qHeads: 48,
    kvHeads: 8,
    headDim: 128,          // config.head_dim (≠ 3072/48=64)
    isMoe: true,
    numExperts: 256,       // num_local_experts
    moeInterSize: 1536,    // intermediate_size
    moeLayers: 62,         // all layers MoE (no dense replace)
  },
  // ── Qwen ─────────────────────────────────────────────────────────────────
  // Qwen3.5-397B: hybrid (linear + full attention) AND MoE. 60 layers,
  // full_attention_interval=4 → 15 full-attn layers (indices 3,7,...,59).
  // MoE every layer (no dense replace). head_dim=256 (explicit).
  {
    id: 'qwen-3.5-397b',
    name: 'Qwen 3.5 (397B-A17B Hybrid)',
    family: 'Qwen',
    attentionType: 'hybrid',
    parameters: 397,       // 397B total (17B activated) — official README
    layers: 60,
    fullAttnLayers: 15,    // 60 / full_attention_interval(4) = 15
    hiddenSize: 4096,
    qHeads: 32,
    kvHeads: 2,
    headDim: 256,          // config.head_dim (≠ 4096/32=128)
    isMoe: true,
    numExperts: 512,       // num_experts
    moeInterSize: 1024,    // moe_intermediate_size
    moeLayers: 60,         // all layers MoE
  },
  {
    id: 'qwen-3-32b',
    name: 'Qwen 3 (32B GQA)',
    family: 'Qwen',
    attentionType: 'standard',
    parameters: 32,
    layers: 64,
    hiddenSize: 5120,
    qHeads: 64,            // config.num_attention_heads (was wrongly 40)
    kvHeads: 8,
    headDim: 128,          // config.head_dim (= 5120/64, now explicit)
  },
  {
    id: 'qwen-2-72b',
    name: 'Qwen 2/2.5 (72B)',
    family: 'Qwen',
    attentionType: 'standard',
    parameters: 72,
    layers: 80,
    hiddenSize: 8192,
    qHeads: 64,
    kvHeads: 8,
  },
  // ── LLaMA ────────────────────────────────────────────────────────────────
  {
    id: 'llama-3-8b',
    name: 'LLaMA 3/3.1 (8B)',
    family: 'LLaMA',
    attentionType: 'standard',
    parameters: 8,
    layers: 32,
    hiddenSize: 4096,
    qHeads: 32,
    kvHeads: 8,
  },
  {
    id: 'llama-3-70b',
    name: 'LLaMA 3/3.1 (70B)',
    family: 'LLaMA',
    attentionType: 'standard',
    parameters: 70,
    layers: 80,
    hiddenSize: 8192,
    qHeads: 64,
    kvHeads: 8,
  },
  {
    id: 'llama-2-7b',
    name: 'LLaMA 2 (7B)',
    family: 'LLaMA',
    attentionType: 'standard',
    parameters: 7,
    layers: 32,
    hiddenSize: 4096,
    qHeads: 32,
    kvHeads: 32,           // MHA (no GQA in Llama 2)
  },
];

export interface CalculatorState {
  // Model architecture
  attentionType: 'standard' | 'mla' | 'hybrid';
  parameters: number;
  layers: number;
  hiddenSize: number;
  qHeads: number;
  kvHeads: number;
  headDim: number;        // 0 = derive from hiddenSize / qHeads
  mlaDc: number;
  mlaDr: number;
  fullAttnLayers: number;
  // Inference config
  seqLength: number;
  batchSize: number;
  precision: number;
  presetId: string;
  maxNumBatchedTokens: number;
  // MoE (Mixture of Experts)
  isMoe: boolean;
  numExperts: number;      // n_routed_experts
  moeInterSize: number;    // moe_intermediate_size (per expert)
  moeLayers: number;       // number of MoE layers
  enableExpertParallel: boolean;
  // Parallelism & deployment
  dp: number;
  tp: number;
  pp: number;
  gpuMemory: number;
  gpuUtilization: number;
  // vLLM-specific parameters
  kvCacheDtype: string;
  blockSize: number;
  maxModelLen: number;
  enforceEager: boolean;
  enablePrefixCaching: boolean;
}
