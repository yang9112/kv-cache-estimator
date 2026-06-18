import { CalculatorState } from '../types';

export function parseConfigJson(jsonStr: string, currentState: CalculatorState): CalculatorState {
  try {
    const config = JSON.parse(jsonStr);
    
    // Parse standard params with fallbacks to hf typical names
    const layers = config.num_hidden_layers ?? config.n_layer ?? currentState.layers;
    const hiddenSize = config.hidden_size ?? config.n_embd ?? currentState.hiddenSize;
    const qHeads = config.num_attention_heads ?? config.n_head ?? config.num_heads ?? currentState.qHeads;
    const parameters = currentState.parameters; // config json doesn't usually contain parameter count natively

    
    // GQA/MQA fallback to MHA
    const kvHeads = config.num_key_value_heads ?? config.multi_query_group_num ?? qHeads;
    
    let attentionType: 'standard' | 'mla' = 'standard';
    let mlaDc = currentState.mlaDc;
    let mlaDr = currentState.mlaDr;

    // DeepSeek MLA architecture detection
    if (config.kv_lora_rank !== undefined) {
      attentionType = 'mla';
      mlaDc = config.kv_lora_rank;
      mlaDr = config.qk_rope_head_dim ?? config.rope_dim ?? 64; 
    }

    return {
      ...currentState,
      presetId: 'custom',
      attentionType,
      layers: Number(layers),
      hiddenSize: Number(hiddenSize),
      qHeads: Number(qHeads),
      kvHeads: Number(kvHeads),
      mlaDc: Number(mlaDc),
      mlaDr: Number(mlaDr),
      parameters
    };
  } catch (e) {
    console.error("Failed to parse config.json", e);
    throw new Error("Invalid config.json format");
  }
}
