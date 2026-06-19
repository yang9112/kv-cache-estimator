import { CalculatorState } from '../types';

export function parseConfigJson(jsonStr: string, currentState: CalculatorState): CalculatorState {
  try {
    const config = JSON.parse(jsonStr);

    // Parse standard params with fallbacks to hf typical names
    const layers = config.num_hidden_layers ?? config.n_layer ?? currentState.layers;
    const hiddenSize = config.hidden_size ?? config.n_embd ?? currentState.hiddenSize;
    const qHeads = config.num_attention_heads ?? config.n_head ?? config.num_heads ?? currentState.qHeads;
    const parameters = currentState.parameters;

    // GQA/MQA fallback to MHA
    const kvHeads = config.num_key_value_heads ?? config.multi_query_group_num ?? qHeads;

    // max_model_len from config if available
    const maxModelLen = config.max_position_embeddings ?? config.max_sequence_length ?? currentState.maxModelLen;

    // attention_type detection
    let attentionType: 'standard' | 'mla' | 'hybrid' = 'standard';
    let mlaDc = currentState.mlaDc;
    let mlaDr = currentState.mlaDr;
    let fullAttnLayers = currentState.fullAttnLayers;
    // head_dim may be explicit in config (e.g. MiniMax M2, Qwen3.5, DeepSeek V4);
    // otherwise derived later as hidden_size / num_attention_heads.
    const headDim = config.head_dim ?? 0;

    // DeepSeek MLA architecture detection (kv_lora_rank is the key marker)
    if (config.kv_lora_rank !== undefined) {
      attentionType = 'mla';
      mlaDc = config.kv_lora_rank;
      mlaDr = config.qk_rope_head_dim ?? config.rope_dim ?? 64;
    }

    // Hybrid model detection: some architectures declare a mix of full/linear attention layers
    // e.g. Qwen3.5 Hybrid exposes layer_types or a hybrid config
    if (config.attention_type === 'hybrid' || config.hybrid_attention_ratio !== undefined) {
      attentionType = 'hybrid';
      // Try to infer full attention layer count
      if (config.full_attention_layers !== undefined) {
        fullAttnLayers = config.full_attention_layers;
      } else if (config.hybrid_attention_ratio !== undefined && layers) {
        fullAttnLayers = Math.round(Number(layers) * config.hybrid_attention_ratio);
      }
    }

    // ── MoE detection ──
    // n_routed_experts: DeepSeek style
    // num_local_experts: Mixtral style
    const numExperts = config.n_routed_experts ?? config.num_local_experts ?? currentState.numExperts;
    const isMoe = numExperts > 0;
    const moeInterSize = config.moe_intermediate_size ?? currentState.moeInterSize;
    // first_k_dense_replace: first N layers are dense (DeepSeek); Mixtral = 0 (all MoE)
    const firstK = config.first_k_dense_replace ?? 0;
    const moeLayers = isMoe ? Number(layers) - Number(firstK) : 0;

    return {
      ...currentState,
      presetId: 'custom',
      attentionType,
      layers: Number(layers),
      hiddenSize: Number(hiddenSize),
      qHeads: Number(qHeads),
      kvHeads: Number(kvHeads),
      headDim: Number(headDim),
      mlaDc: Number(mlaDc),
      mlaDr: Number(mlaDr),
      fullAttnLayers: Number(fullAttnLayers),
      maxModelLen: Number(maxModelLen),
      // Keep seqLength in sync with maxModelLen so the context box never exceeds max_len
      seqLength: Math.min(Number(currentState.seqLength), Number(maxModelLen)),
      parameters,
      isMoe,
      numExperts: Number(numExperts),
      moeInterSize: Number(moeInterSize),
      moeLayers: Number(moeLayers),
    };
  } catch (e) {
    console.error("Failed to parse config.json", e);
    throw new Error("Invalid config.json format");
  }
}
