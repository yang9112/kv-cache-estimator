import { CalculatorState, KV_CACHE_DTYPES } from '../types';

export const formatBytes = (bytes: number) => {
  if (bytes === 0) return '0 B';
  if (bytes < 0) return 'N/A';
  const k = 1024;
  const sizes = ['B', 'KB', 'MB', 'GB', 'TB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
};

/**
 * Compute routed expert params (in number of parameters).
 *
 * Per expert: gate (H→I) + up (H→I) + down (I→H) = 3 × H × I
 * Total routed experts = num_experts × per_expert_params × moe_layers
 *
 * Source: vLLM FusedMoE gate/up/down = 3 matrices per expert.
 * Verified against DeepSeek V3 config: 256 × 3 × 7168 × 2048 × 58 = ~654B (matches 671B total).
 */
export const computeExpertParams = (state: CalculatorState): number => {
  if (!state.isMoe || state.numExperts <= 0 || state.moeInterSize <= 0 || state.moeLayers <= 0) {
    return 0;
  }
  return state.numExperts * 3 * state.hiddenSize * state.moeInterSize * state.moeLayers;
};

export const calculateKV = (state: CalculatorState) => {
  // ── KV cache dtype resolution ──
  const kvDtype = KV_CACHE_DTYPES.find(d => d.key === state.kvCacheDtype) || KV_CACHE_DTYPES[0];
  const kvBytesPerParam = kvDtype.bytesPerParam === -1 ? state.precision : kvDtype.bytesPerParam;
  const hasPerTokenHeadScales = kvDtype.hasPerTokenHeadScales;

  // ── Head dimension ──
  // head_dim: explicit (config.head_dim) takes priority over hidden/qHeads.
  // Many newer models (MiniMax M2, Qwen3.5, DeepSeek V4 DSA) set head_dim
  // independently of hidden_size/num_attention_heads.
  const headDim = state.attentionType === 'mla'
    ? 0
    : (state.headDim > 0 ? state.headDim : state.hiddenSize / state.qHeads);

  // ── Per-token per-layer KV cache size ──
  let sizePerTokenPerLayer = 0;
  if (state.attentionType === 'mla') {
    sizePerTokenPerLayer = (state.mlaDc + state.mlaDr) * kvBytesPerParam;
  } else {
    sizePerTokenPerLayer = 2 * state.kvHeads * headDim * kvBytesPerParam;
  }

  // ── Effective layers ──
  const effectiveLayers = state.attentionType === 'hybrid' ? state.fullAttnLayers : state.layers;
  const sizePerTokenTotal = sizePerTokenPerLayer * effectiveLayers;

  // ── Block-level calculations (vLLM paged attention) ──
  const blockSize = state.blockSize || 16;
  let pageSizeBytesPerLayer = sizePerTokenPerLayer * blockSize;
  if (hasPerTokenHeadScales && state.attentionType !== 'mla') {
    pageSizeBytesPerLayer += 2 * blockSize * state.kvHeads * 4;
  }
  const pageSizeBytesAllLayers = pageSizeBytesPerLayer * effectiveLayers;

  // ── Total memory for batch and sequence ──
  const totalMemory = sizePerTokenTotal * state.seqLength * state.batchSize;
  const tokensPerGB = sizePerTokenTotal > 0 ? (1024 * 1024 * 1024) / sizePerTokenTotal : 0;

  // ── Parallelism ──
  // TP × PP: shards dense weights, KV cache, and attention.
  // DP: replicates dense model, BUT shards MoE expert weights (via flatten_tp).
  // EP: boolean toggle, does NOT change per-GPU expert weight amount.
  //     EP group size = TP × DP (derived, not independent).
  // Source: vllm/config/parallel.py:125-127, fused_moe/config.py make()
  const tpPpParallelism = (state.tp || 1) * (state.pp || 1);
  const totalGpus = tpPpParallelism * (state.dp || 1);

  // ── Weight memory: split dense vs expert (MoE) ──
  // Dense (attention, dense FFN, shared expert, embeddings): / (TP × PP)
  // Expert (routed MoE): / (TP × DP × PP) — DP participates via flatten_tp
  // Source: vllm/model_executor/layers/fused_moe/config.py
  const weightTotal = (state.parameters || 0) * 1e9 * state.precision;
  const expertParams = computeExpertParams(state);
  const expertWeightTotal = expertParams * state.precision;
  const denseWeightTotal = weightTotal - expertWeightTotal;

  const denseWeightPerGPU = denseWeightTotal / tpPpParallelism;
  // For MoE models, DP shards expert weights (flatten_tp = TP×DP×PCP).
  // For dense models, expertWeightTotal = 0 so this is irrelevant.
  const expertWeightPerGPU = state.isMoe && state.dp > 1
    ? expertWeightTotal / (tpPpParallelism * (state.dp || 1))
    : expertWeightTotal / tpPpParallelism;
  const weightPerGPU = denseWeightPerGPU + expertWeightPerGPU;

  // ── Framework overhead estimation ──
  let overheadPerGPU = 0;

  // Activation memory: profile_run() processes max_num_batched_tokens tokens
  // through the full forward pass. Peak activation ≈ tokens × layers × hiddenSize × precision × K.
  // K ≈ 0.5 empirical (accounts for FFN intermediates + attention outputs + allocator overhead).
  // Source: vllm/v1/worker/gpu_worker.py:387 profile_run() → gpu_model_runner.py:6288 _dummy_run(max_num_tokens)
  const maxBatchTokens = state.maxNumBatchedTokens || 0;
  const activationMemory = maxBatchTokens > 0
    ? maxBatchTokens * state.layers * state.hiddenSize * state.precision * 0.5
    : 0;
  overheadPerGPU += activationMemory; // activation peak from profile_run
  overheadPerGPU += 0.5 * 1024 * 1024 * 1024; // torch allocator fragmentation (~0.5 GB)
  if (state.tp > 1) {
    const ncclBaseGB = 0.5;
    const ncclPerTpGB = 0.3;
    overheadPerGPU += (ncclBaseGB + ncclPerTpGB * Math.log2(state.tp)) * 1024 * 1024 * 1024;
  }
  // EP AllToAll communication buffers (DeepEP / nixl / allgather-reducescatter)
  // Source: vllm/config/parallel.py:186-198 — DeepEP backend requires extra buffer
  if (state.enableExpertParallel && state.isMoe && (state.dp > 1 || state.tp > 1)) {
    overheadPerGPU += 1.0 * 1024 * 1024 * 1024;
  }
  if (!state.enforceEager) {
    overheadPerGPU += 1.0 * 1024 * 1024 * 1024; // CUDA graph
  }

  // ── KV cache per GPU (sharded across TP × PP) ──
  const kvPerGPU = totalMemory / tpPpParallelism;
  // Total per-GPU VRAM includes KV cache so it tracks seq/batch changes.
  const totalPerGPU = weightPerGPU + overheadPerGPU + kvPerGPU;

  // ── vLLM KV cache budget ──
  const totalUsablePerGPU = (state.gpuMemory || 80) * 1024 * 1024 * 1024 * (state.gpuUtilization || 0.9);
  let vllmKvBudgetPerGPU = totalUsablePerGPU - weightPerGPU - overheadPerGPU;
  if (vllmKvBudgetPerGPU < 0) vllmKvBudgetPerGPU = 0;

  // ── Block-level vLLM calculations ──
  const numBlocks = pageSizeBytesAllLayers > 0
    ? Math.max(0, Math.floor(vllmKvBudgetPerGPU / pageSizeBytesAllLayers))
    : 0;
  const kvCacheTokensPerGPU = numBlocks * blockSize;
  const maxModelLen = state.maxModelLen || state.seqLength;
  const maxConcurrency = maxModelLen > 0 ? kvCacheTokensPerGPU / maxModelLen : 0;

  // Legacy linear calculation
  const tokenSizePerGPU = sizePerTokenTotal / tpPpParallelism;
  const maxTokensPerGPU = tokenSizePerGPU > 0 ? vllmKvBudgetPerGPU / tokenSizePerGPU : 0;

  // ── Total cluster capacity ──
  const totalClusterKVTokens = kvCacheTokensPerGPU * (state.dp || 1);

  // ── EP display values ──
  // EP group size = TP × DP (not an independent dimension).
  // Source: vllm/distributed/parallel_state.py EP group init
  const epSize = state.enableExpertParallel ? (state.tp || 1) * (state.dp || 1) : 1;
  const localNumExperts = epSize > 1 && state.numExperts > 0
    ? Math.ceil(state.numExperts / epSize) // average: some ranks get +1 from remainder
    : state.numExperts;

  // ── KV per GPU ──

  return {
    // KV cache dtype info
    kvBytesPerParam,
    hasPerTokenHeadScales,
    // Architecture
    headDim,
    sizePerTokenPerLayer,
    sizePerTokenTotal,
    effectiveLayers,
    // Block-level
    blockSize,
    pageSizeBytesPerLayer,
    pageSizeBytesAllLayers,
    numBlocks,
    kvCacheTokensPerGPU,
    maxConcurrency,
    // Total memory
    totalMemory,
    tokensPerGB,
    // Per-GPU breakdown
    kvPerGPU,
    denseWeightTotal,
    expertWeightTotal,
    denseWeightPerGPU,
    expertWeightPerGPU,
    weightTotal,
    weightPerGPU,
    activationMemory,
    overheadPerGPU,
    totalPerGPU,
    // vLLM budget
    totalUsablePerGPU,
    vllmKvBudgetPerGPU,
    maxTokensPerGPU,
    // Parallelism
    tpPpParallelism,
    totalGpus,
    // Cluster
    totalClusterKVTokens,
    // EP
    epSize,
    localNumExperts,
    expertParams,
  };
};
