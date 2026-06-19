# Changelog

## [1.1.0] - 2025-06-18

### Added
- GPU memory capacity and utilization inputs for deployment-aware estimation
- vLLM KV cache budget calculation per GPU
- Max concurrent tokens estimation based on available KV budget
- Model presets grouped by family in dropdown (DeepSeek, GLM, Qwen, LLaMA, MiniMax, Moonshot)

## [1.0.0] - 2025-06-18

### Added
- Interactive KV cache memory calculator with real-time estimation
- Standard (MHA/GQA/MQA), MLA (DeepSeek-style), and Hybrid attention support
- Model presets: DeepSeek V3/R1, GLM 5, Qwen 3.5, LLaMA 3, Kimi K2.5, MiniMax M2.5, and more
- config.json auto-import from HuggingFace model repos
- Tensor parallelism (TP) and pipeline parallelism (PP) for multi-GPU estimation
- Precision options: FP32, FP16/BF16, INT8/FP8, INT4
- Bilingual UI (English / Chinese)
- Responsive layout with Tailwind CSS 4
