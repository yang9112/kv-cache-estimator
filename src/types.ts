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

export type ModelPreset = {
  id: string;
  name: string;
  attentionType: 'standard' | 'mla' | 'hybrid';
  parameters: number;
  layers: number;
  hiddenSize: number;
  qHeads: number;
  kvHeads: number;
  mlaDc?: number;
  mlaDr?: number;
  fullAttnLayers?: number;
};

export const PRESETS: ModelPreset[] = [
  {
    id: 'custom',
    name: 'Custom',
    attentionType: 'standard',
    parameters: 7,
    layers: 32,
    hiddenSize: 4096,
    qHeads: 32,
    kvHeads: 8,
  },
  {
    id: 'deepseek-v4-pro',
    name: 'DeepSeek V4 Pro',
    attentionType: 'mla',
    parameters: 1000,
    layers: 61,
    hiddenSize: 7168,
    qHeads: 128,
    kvHeads: 128,
    mlaDc: 512,
    mlaDr: 64,
  },
  {
    id: 'deepseek-v4-flash',
    name: 'DeepSeek V4 Flash',
    attentionType: 'mla',
    parameters: 236,
    layers: 60,
    hiddenSize: 5120,
    qHeads: 128,
    kvHeads: 128,
    mlaDc: 512,
    mlaDr: 64,
  },
  {
    id: 'deepseek-v3',
    name: 'DeepSeek V3 / R1 (671B)',
    attentionType: 'mla',
    parameters: 671,
    layers: 61,
    hiddenSize: 7168,
    qHeads: 128,
    kvHeads: 128,
    mlaDc: 512,
    mlaDr: 64,
  },
  {
    id: 'deepseek-v2',
    name: 'DeepSeek V2 (236B)',
    attentionType: 'mla',
    parameters: 236,
    layers: 60,
    hiddenSize: 5120,
    qHeads: 128,
    kvHeads: 128,
    mlaDc: 512,
    mlaDr: 64,
  },
  {
    id: 'glm-5',
    name: 'GLM 5 / 5.1',
    attentionType: 'mla',
    parameters: 150,
    layers: 78,
    hiddenSize: 6144,
    qHeads: 64,
    kvHeads: 64,
    mlaDc: 512,
    mlaDr: 64,
  },
  {
    id: 'glm-4-flash',
    name: 'GLM 4.7 Flash',
    attentionType: 'mla',
    parameters: 12,
    layers: 47,
    hiddenSize: 2048,
    qHeads: 20,
    kvHeads: 20,
    mlaDc: 512,
    mlaDr: 64,
  },
  {
    id: 'kimi-k2.5',
    name: 'Kimi K2.5 / K2',
    attentionType: 'mla',
    parameters: 200,
    layers: 61,
    hiddenSize: 7168,
    qHeads: 64,
    kvHeads: 64,
    mlaDc: 512,
    mlaDr: 64,
  },
  {
    id: 'minimax-m2.5',
    name: 'MiniMax M2.5 / M2',
    attentionType: 'standard',
    parameters: 100,
    layers: 62,
    hiddenSize: 3072,
    qHeads: 48,
    kvHeads: 8,
  },
  {
    id: 'glm-4-9b',
    name: 'GLM 4 (9B)',
    attentionType: 'standard',
    parameters: 9,
    layers: 40,
    hiddenSize: 4096,
    qHeads: 32,
    kvHeads: 2,
  },
  {
    id: 'qwen-3.5-397b',
    name: 'Qwen 3.5 / 3.6 (397B Hybrid)',
    attentionType: 'hybrid',
    parameters: 397,
    layers: 60,
    fullAttnLayers: 20,
    hiddenSize: 8192,
    qHeads: 64,
    kvHeads: 2,
  },
  {
    id: 'qwen-3-32b',
    name: 'Qwen 3 (32B GQA)',
    attentionType: 'standard',
    parameters: 32,
    layers: 64,
    hiddenSize: 5120,
    qHeads: 40,
    kvHeads: 8,
  },
  {
    id: 'llama-3-8b',
    name: 'LLaMA 3/3.1 (8B)',
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
    attentionType: 'standard',
    parameters: 70,
    layers: 80,
    hiddenSize: 8192,
    qHeads: 64,
    kvHeads: 8,
  },
  {
    id: 'qwen-2-72b',
    name: 'Qwen 2/2.5 (72B)',
    attentionType: 'standard',
    parameters: 72,
    layers: 80,
    hiddenSize: 8192,
    qHeads: 64,
    kvHeads: 8,
  },
  {
    id: 'llama-2-7b',
    name: 'LLaMA 2 (7B)',
    attentionType: 'standard',
    parameters: 7,
    layers: 32,
    hiddenSize: 4096,
    qHeads: 32,
    kvHeads: 32,
  },
  {
    id: 'mixtral-8x7b',
    name: 'Mixtral 8x7B',
    attentionType: 'standard',
    parameters: 47,
    layers: 32,
    hiddenSize: 4096,
    qHeads: 32,
    kvHeads: 8,
  }
];

export interface CalculatorState {
  attentionType: 'standard' | 'mla' | 'hybrid';
  parameters: number;
  layers: number;
  hiddenSize: number;
  qHeads: number;
  kvHeads: number;
  mlaDc: number;
  mlaDr: number;
  fullAttnLayers: number;
  seqLength: number;
  batchSize: number;
  precision: number;
  presetId: string;
  tp: number;
  pp: number;
}
