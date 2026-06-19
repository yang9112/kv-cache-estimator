import React, { createContext, useContext, useState, ReactNode, useEffect } from 'react';

export type Language = 'en' | 'zh';

const translations = {
  en: {
    title: "KV Cache Calculator",
    subtitle: "Calculate memory requirements for LLM inference KV Cache based on model architecture, context window, and precision.",
    modelArch: "Model Architecture",
    modelPreset: "Model Preset",
    layers: "Number of Layers (L)",
    hiddenSize: "Hidden Size (H)",
    attentionType: "Attention Type",
    qHeads: "Query Heads (Q)",
    kvHeads: "KV Heads",
    mlaDc: "MLA Latent Dim (dc)",
    mlaDr: "MLA RoPE Dim (dr)",
    inferenceConfig: "Inference Configuration",
    dataType: "Data Type (Precision)",
    seqLength: "Context Length (K)",
    seqLengthHelp: "Input + generation length. 1K = 1024 tokens.",
    batchSize: "Batch Size",
    totalMemory: "Total KV Cache Memory",
    gigabytes: "Gigabytes",
    tokensPerGb: "Tokens per 1 GB",
    headDim: "Head Dimension (d)",
    bytesPerToken1L: "Bytes per Token (1 Layer)",
    bytesPerTokenAll: "Bytes per Token (All Layers)",
    formula: "Formula",
    formulaStandard: "2 × Layers × SeqLen × BatchSize × KV_Heads × (HiddenSize / Q_Heads) × Precision_Bytes",
    formulaMla: "Layers × SeqLen × BatchSize × (dc + dr) × Precision_Bytes",
    formulaHybrid: "2 × Full Attention Layers × SeqLen × BatchSize × KV_Heads × (HiddenSize / Q_Heads) × Precision_Bytes",
    kvHeadsHelp: "Determines MHA vs GQA vs MQA",
    custom: "Custom",
    langSwitchTo: "中文",
    standardAttr: "Standard (MHA/GQA/MQA)",
    mlaAttr: "MLA / DSA (DeepSeek)",
    hybridAttr: "Hybrid (Sparse/Linear/Full)",
    fullAttnLayers: "Full Attention Layers",
    notApplicable: "N/A",
    importConfig: "Import config.json",
    importError: "Failed to parse config file.",
    deploymentConfig: "Deployment Configuration",
    parameters: "Parameters (Billion)",
    tensorParallel: "Tensor Parallel (TP)",
    pipelineParallel: "Pipeline Parallel (PP)",
    vramPerGPU: "VRAM Required GPU",
    kvPerGPU: "KV Cache / GPU",
    weightPerGPU: "Weights / GPU",
    overheadVram: "CUDA/Framework Overhead",
    weightMemory: "Total Weights Memory",
    explanationTitle: "Engineered VRAM Estimation Guide",
    formulaDesc1: "Total VRAM = Weights + KV Cache + Activations + Framework Overhead + Comm Buffer",
    formulaDesc2: "For inference deployment, managing VRAM accurately avoids Out Of Memory (OOM) errors. Different parallelism strategies (TP/PP/DP/EP) affect memory distribution differently — especially for MoE models.",
    expWeight: "Model Weights:",
    expWeightDesc: "Calculated as Parameters × Bytes_per_Precision. For MoE models, weights split into dense (attention/FFN/embedding) and expert (routed MoE) parts, sharded differently.",
    expKV: "KV Cache:",
    expKVDesc: "Scales linearly with sequence length and batch size. Most critical factor for long-context windows. Divided by (TP × PP). Not affected by DP or EP.",
    expAct: "Activations & Workspace:",
    expActDesc: "Typically requires 2-5GB for CUDA workspace (FlashAttention, Graph) and communication buffers limit (AllReduce).",
    expTpPp: "TP & PP Impact:",
    expTpPpDesc: "Dense weights and KV cache are divided by (TP × PP). E.g. A 140GB dense model with TP=8 needs 17.5GB per GPU.",
    expDp: "DP (Data Parallelism):",
    expDpDesc: "Replicates dense weights/KV across GPUs (no per-GPU reduction). BUT for MoE models, DP participates in sharding expert weights: expert / (TP × DP × PP). See EP below.",
    expEp: "EP (Expert Parallelism):",
    expEpDesc: "enable_expert_parallel is a boolean flag (not a separate dimension). EP group size = TP × DP. When EP is on, experts are partitioned whole-expert rather than TP-sharded, and uses AllToAll communication. EP does NOT change per-GPU expert weight amount — that is controlled by DP (flatten_tp = TP × DP). EP only changes partitioning granularity and communication overhead.",
    expEnd: "Estimation methodology based on typical vLLM / TensorRT-LLM allocations.",
    gpuMemory: "GPU Memory Cap (GB per GPU)",
    gpuUtilization: "GPU Mem Utilization",
    vllmBudgetHeading: "vLLM KV Cache Budget",
    maxTokensPerGpu: "Max Concurrent Tokens (per GPU)",
    vllmHelpText: "Calculated based on usable GPU memory derived from specified utilization.",
    maxUsableKvBudget: "Max Usable KV Budget (per GPU)",
    dataParallel: "Data Parallel (DP)",
    dpHelp: "Dense model: replicates (no per-GPU reduction). MoE model: also shards expert weights (via vLLM flatten_tp).",
    vllmConfig: "vLLM Advanced Config",
    kvCacheDtype: "KV Cache Dtype",
    kvCacheDtypeHelp: "Quantize KV cache independently of model weights.",
    blockSize: "Block Size (tokens)",
    blockSizeHelp: "vLLM paged-attention block size (default 16).",
    maxModelLen: "Max Model Length (K)",
    maxModelLenHelp: "vLLM max_model_len; used for concurrency calc. 1K = 1024 tokens.",
    enforceEager: "Enforce Eager",
    enforceEagerHelp: "Disable CUDA graphs (saves ~1GB, slower decode).",
    enablePrefixCaching: "Prefix Caching",
    enablePrefixCachingHelp: "Share KV blocks across requests with same prefix.",
    maxNumBatchedTokens: "Max Batched Tokens",
    maxNumBatchedTokensHelp: "max_num_batched_tokens: max tokens per scheduling step. Larger values increase activation memory (profile_run peak). Default 8192.",
    numBlocks: "Num KV Blocks (per GPU)",
    kvCacheTokens: "KV Cache Capacity (tokens/GPU)",
    maxConcurrency: "Max Concurrency (reqs @ max_len)",
    totalGpus: "Total GPUs",
    totalClusterKVTokens: "Cluster KV Capacity (tokens)",
    vllmBlockMetrics: "vLLM Block-Level Metrics",
    enableExpertParallel: "Expert Parallel (EP)",
    epHelp: "Use expert parallelism instead of tensor parallelism for MoE layers (vLLM enable_expert_parallel). EP group = TP×DP.",
    expertParams: "Expert Params",
    denseParams: "Dense Params",
    epSize: "EP Size",
    localExperts: "Experts/GPU",
    isMoe: "MoE Model",
    numExperts: "Routed Experts (num_experts)",
    moeInterSize: "Expert Intermediate (moe_intermediate_size)",
    moeLayers: "MoE Layers"
  },
  zh: {
    title: "KV Cache 计算器",
    subtitle: "根据模型架构、上下文窗口和精度计算大模型推理的 KV Cache 显存需求。",
    modelArch: "模型架构",
    modelPreset: "模型预设",
    layers: "层数 (L, Layers)",
    hiddenSize: "隐藏层维度 (H, Hidden Size)",
    attentionType: "注意力类型 (Attention)",
    qHeads: "查询头数量 (Q Heads)",
    kvHeads: "KV头数量 (KV Heads)",
    mlaDc: "MLA 隐空间维度 (dc)",
    mlaDr: "MLA RoPE 维度 (dr)",
    inferenceConfig: "推理配置",
    dataType: "数据类型 (精度)",
    seqLength: "上下文长度 (K)",
    seqLengthHelp: "输入 + 生成长度，1K = 1024 tokens",
    batchSize: "批次大小 (Batch Size)",
    totalMemory: "总 KV Cache 显存",
    gigabytes: "GB",
    tokensPerGb: "1GB 可用 Token 数",
    headDim: "注意力头维度 (d)",
    bytesPerToken1L: "单 Token 占用 (单层)",
    bytesPerTokenAll: "单 Token 占用 (所有层)",
    formula: "计算公式",
    formulaStandard: "2 × 层数 × 序列长度 × 批次 × KV头数 × (隐藏层维度 / Q头数) × 精度字节数",
    formulaMla: "层数 × 序列长度 × 批次 × (dc + dr) × 精度字节数",
    formulaHybrid: "2 × 全注意力层数 × 序列长度 × 批次 × KV头数 × (隐藏层维度 / Q头数) × 精度",
    kvHeadsHelp: "决定 MHA / GQA / MQA",
    custom: "自定义",
    langSwitchTo: "English",
    standardAttr: "标准 (MHA/GQA/MQA)",
    mlaAttr: "MLA / DSA架构 (DeepSeek)",
    hybridAttr: "混合架构(Qwen3.5/Gemma)",
    fullAttnLayers: "全注意力层数",
    notApplicable: "无",
    importConfig: "导入 config.json",
    importError: "解析配置文件失败",
    deploymentConfig: "部署与并行配置",
    parameters: "模型参数量 (十亿/B)",
    tensorParallel: "张量并行 (TP)",
    pipelineParallel: "流水线并行 (PP)",
    vramPerGPU: "单卡显存需求",
    kvPerGPU: "单卡 KV Cache",
    weightPerGPU: "单卡权重显存",
    overheadVram: "框架与通信开销",
    weightMemory: "总权重显存",
    explanationTitle: "工程化显存估算指南",
    formulaDesc1: "模型显存占用 = 权重 + KV Cache + 激活值 + 框架开销 + 通信Buffer",
    formulaDesc2: "在推理部署中，不同并行策略 (DP/TP/PP/EP) 对各部分显存的影响截然不同，MoE 模型尤其需要注意。真正决定是否 OOM 的往往是 KV Cache。",
    expWeight: "模型权重 (Weight):",
    expWeightDesc: "参数量 × 每参数字节数。MoE 模型需拆分为 dense (attention/FFN/embedding) 与 expert (路由专家) 两部分，切分方式不同。",
    expKV: "KV Cache:",
    expKVDesc: "随上下文长度 (Seq) 和并发批次 (Batch) 线性增加，长文本时占用最大。它被 (TP × PP) 整除切分，不受 DP 或 EP 影响。",
    expAct: "激活值及框架预留 (Activations/Workspace):",
    expActDesc: "推理激活值较小。一般需为 CUDA Workspace (如 FlashAttention)、通信 Buffer (TP时的AllReduce) 等预留 2~5GB 空间。",
    expTpPp: "TP 与 PP并行的影响:",
    expTpPpDesc: "Dense 权重和 KV cache 都除以 (TP × PP)。例如 140GB dense 模型 TP=8 时单卡 17.5GB。",
    expDp: "数据并行 (DP) 的影响:",
    expDpDesc: "Dense 权重和 KV cache 复制不切分。但对 MoE 模型，DP 参与专家权重切分：expert / (TP × DP × PP)（vLLM flatten_tp）。详见 EP。",
    expEp: "专家并行 (EP) 的影响:",
    expEpDesc: "enable_expert_parallel 是布尔开关，不是独立维度。EP 组大小 = TP × DP。EP 开启后专家按整体均分（而非 TP 切分），使用 AllToAll 通信。但 EP 不改变单卡专家权重总量（由 DP flatten_tp=TP×DP 控制），只改变切分粒度和通信 overhead。",
    expEnd: "以上规则适用于 vLLM、SGLang、TensorRT-LLM 等主流大模型推理引擎。",
    gpuMemory: "单卡显存上限 (GB)",
    gpuUtilization: "显存占用率 (gpu_memory_utilization)",
    vllmBudgetHeading: "vLLM KV 显存预算",
    maxTokensPerGpu: "最大并发 Tokens 数 (单卡)",
    vllmHelpText: "根据显存占用率和模型权重等开销，推算出的 KV Cache 可用上限。",
    maxUsableKvBudget: "最大可用 KV 显存 (单卡)",
    dataParallel: "数据并行 (DP)",
    dpHelp: "Dense 模型：复制不减少单卡显存。MoE 模型：同时切分专家权重 (vLLM flatten_tp)。",
    vllmConfig: "vLLM 高级配置",
    kvCacheDtype: "KV Cache 精度",
    kvCacheDtypeHelp: "独立于模型权重的 KV cache 量化。",
    blockSize: "Block 大小 (tokens)",
    blockSizeHelp: "vLLM paged-attention 块大小 (默认 16)。",
    maxModelLen: "最大模型长度 (K)",
    maxModelLenHelp: "vLLM max_model_len，用于并发数计算，1K = 1024 tokens",
    enforceEager: "禁用 CUDA Graph",
    enforceEagerHelp: "关闭 CUDA graph (节省 ~1GB，decode 变慢)。",
    enablePrefixCaching: "前缀缓存",
    enablePrefixCachingHelp: "相同前缀的请求共享 KV 块。",
    maxNumBatchedTokens: "单步最大 Token 数",
    maxNumBatchedTokensHelp: "max_num_batched_tokens：单次调度迭代最大 token 数。值越大，激活值峰值越高（profile_run 实测）。默认 8192。",
    numBlocks: "KV Block 数量 (单卡)",
    kvCacheTokens: "KV Cache 容量 (tokens/单卡)",
    maxConcurrency: "最大并发 (按 max_len)",
    totalGpus: "总 GPU 数",
    totalClusterKVTokens: "集群 KV 总容量 (tokens)",
    vllmBlockMetrics: "vLLM Block 级指标",
    enableExpertParallel: "专家并行 (EP)",
    epHelp: "对 MoE 层使用专家并行替代张量并行 (vLLM enable_expert_parallel)。EP 组大小 = TP×DP。",
    expertParams: "专家参数量",
    denseParams: "Dense 参数量",
    epSize: "EP 组大小",
    localExperts: "单卡专家数",
    isMoe: "MoE 模型",
    numExperts: "路由专家数 (num_experts)",
    moeInterSize: "专家中间维度 (moe_intermediate_size)",
    moeLayers: "MoE 层数"
  }
};

type TransKey = keyof typeof translations.en;

interface I18nContextType {
  lang: Language;
  t: (key: TransKey) => string;
  toggleLang: () => void;
}

const I18nContext = createContext<I18nContextType | null>(null);

export function LanguageProvider({ children }: { children: ReactNode }) {
  const [lang, setLang] = useState<Language>('zh');

  useEffect(() => {
    const userLang = navigator.language.toLowerCase();
    if (userLang.startsWith('zh')) {
      setLang('zh');
    } else {
      setLang('en');
    }
  }, []);

  const t = (key: TransKey) => translations[lang][key];

  const toggleLang = () => {
    setLang(l => l === 'en' ? 'zh' : 'en');
  };

  return (
    <I18nContext.Provider value={{ lang, t, toggleLang }}>
      {children}
    </I18nContext.Provider>
  );
}

export function useI18n() {
  const context = useContext(I18nContext);
  if (!context) {
    throw new Error('useI18n must be used within LanguageProvider');
  }
  return context;
}
