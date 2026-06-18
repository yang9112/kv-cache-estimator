import { CalculatorState, PRECISIONS, PRESETS } from '../types';

export const formatBytes = (bytes: number) => {
  if (bytes === 0) return '0 B';
  const k = 1024;
  const sizes = ['B', 'KB', 'MB', 'GB', 'TB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
};

export const calculateKV = (state: CalculatorState) => {
  // Head Dimension (d) = Hidden Size (h) / Number of Query Heads (a)
  const headDim = state.attentionType === 'mla' ? 0 : state.hiddenSize / state.qHeads;
  
  let sizePerTokenPerLayer = 0;
  if (state.attentionType === 'mla') {
    // MLA caches compressed latent vector and decoupled RoPE keys
    sizePerTokenPerLayer = (state.mlaDc + state.mlaDr) * state.precision;
  } else {
    // Standard size per token per layer = 2 (for K and V) * num_kv_heads * head_dim * bytes_per_param
    sizePerTokenPerLayer = 2 * state.kvHeads * headDim * state.precision;
  }
  
  // Total Token Size (across all layers)
  // For hybrid models, only a subset of layers use full KV cache (linear attention uses constant memory)
  const effectiveLayers = state.attentionType === 'hybrid' ? state.fullAttnLayers : state.layers;
  const sizePerTokenTotal = sizePerTokenPerLayer * effectiveLayers;
  
  // Total Memory for Batch and Seq = Total Token Size * SeqLength * BatchSize
  const totalMemory = sizePerTokenTotal * state.seqLength * state.batchSize;
  
  // Tokens per GB = (1 GB in bytes) / sizePerTokenTotal
  const tokensPerGB = (1024 * 1024 * 1024) / sizePerTokenTotal;

  // GPU Memory Estimations
  const parallelism = (state.tp || 1) * (state.pp || 1);
  const kvPerGPU = totalMemory / parallelism;
  
  // 1 Billion = 10^9
  const weightTotal = (state.parameters || 0) * 1e9 * state.precision;
  const weightPerGPU = weightTotal / parallelism;
  
  const overheadPerGPU = 5 * 1024 * 1024 * 1024; // 5 GB for Workspace, CUDA graph, Comm Buffer etc
  const totalPerGPU = kvPerGPU + weightPerGPU + overheadPerGPU;

  // vLLM calculations
  const totalUsablePerGPU = (state.gpuMemory || 80) * 1024 * 1024 * 1024 * (state.gpuUtilization || 0.9);
  let vllmKvBudgetPerGPU = totalUsablePerGPU - weightPerGPU - overheadPerGPU;
  if (vllmKvBudgetPerGPU < 0) vllmKvBudgetPerGPU = 0;
  
  const tokenSizePerGPU = sizePerTokenTotal / parallelism;
  const maxTokensPerGPU = tokenSizePerGPU > 0 ? vllmKvBudgetPerGPU / tokenSizePerGPU : 0;

  return {
    headDim,
    sizePerTokenPerLayer,
    sizePerTokenTotal,
    totalMemory,
    tokensPerGB,
    kvPerGPU,
    weightTotal,
    weightPerGPU,
    overheadPerGPU,
    totalPerGPU,
    vllmKvBudgetPerGPU,
    maxTokensPerGPU
  };
};
