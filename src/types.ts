// Re-export the model preset type from the config layer so existing type
// imports (`import { ModelPreset } from '../types'`) keep working. The actual
// preset data lives as JSON files in ./configs/models/ (see ./configs).
export type { ModelPreset, AttentionType } from './configs/schema';

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
