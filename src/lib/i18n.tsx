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
    seqLength: "Sequence Length",
    seqLengthHelp: "Context + Generation length",
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
    formulaDesc2: "For inference deployment, managing VRAM accurately avoids Out Of Memory (OOM) errors. Different parallelism strategies (TP/PP/DP) affect memory distribution differently.",
    expWeight: "Model Weights:",
    expWeightDesc: "Calculated as Parameters × Bytes_per_Precision. TP (Tensor) and PP (Pipeline) parallelism divide the weights evenly across GPUs. (E.g. A 70B FP16 model = 140GB).",
    expKV: "KV Cache:",
    expKVDesc: "Scales linearly with sequence length and batch size. Most critical factor for long-context windows. Divided by both TP and PP.",
    expAct: "Activations & Workspace:",
    expActDesc: "Typically requires 2-5GB for CUDA workspace (FlashAttention, Graph) and communication buffers limit (AllReduce).",
    expTpPp: "TP & PP Impact:",
    expTpPpDesc: "Weights and KV Cache are divided by (TP × PP). E.g. A 140GB model with TP=8 requires 17.5GB per GPU.",
    expDp: "DP (Data Parallelism):",
    expDpDesc: "Does not reduce memory per GPU. It replicates the model across GPUs to increase throughput.",
    expEnd: "Estimation methodology based on typical vLLM / TensorRT-LLM allocations."
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
    seqLength: "上下文长度 (Seq Length)",
    seqLengthHelp: "输入 + 生成长度",
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
    formulaDesc2: "在推理部署中，不同并行策略 (DP/TP/PP) 对各部分显存的影响截然不同。真正决定是否 OOM 的往往是 KV Cache。",
    expWeight: "模型权重 (Weight):",
    expWeightDesc: "参数量 × 每参数字节数。TP 和 PP 都会切分权重。例如 70B FP16 (140GB) 在 TP=8 时，单卡仅占 17.5GB。",
    expKV: "KV Cache:",
    expKVDesc: "随上下文长度 (Seq) 和并发批次 (Batch) 线性增加，长文本时占用最大。它会被 TP 和 PP 同时整除切分。",
    expAct: "激活值及框架预留 (Activations/Workspace):",
    expActDesc: "推理激活值较小。一般需为 CUDA Workspace (如 FlashAttention)、通信 Buffer (TP时的AllReduce) 等预留 2~5GB 空间。",
    expTpPp: "TP 与 PP并行的影响:",
    expTpPpDesc: "单卡分担的显存将同时除以 (TP × PP)。",
    expDp: "数据并行 (DP) 的影响:",
    expDpDesc: "DP 不切分模型，每个 Replica 都有完整模型和 KV。所以 DP 仅用于提升吞吐并发，完全不会减少单卡显存要求。",
    expEnd: "以上规则适用于 vLLM、SGLang、TensorRT-LLM 等主流大模型推理引擎。"
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
