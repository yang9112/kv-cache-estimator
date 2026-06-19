# Changelog

## [1.3.0] - 2026-06-20

### Changed
- Rewrote README with professional structure: Table of Contents, Supported Models table, Calculation Methodology section, Development commands
- Project title renamed from "Calculator" to "Estimator" to match repository name
- Formatted formulas with proper Unicode operators (`×`, `÷`) instead of ASCII `x`/`/`
- Added bilingual "Supported Models" section listing all 17 built-in presets by family
- Consolidated tech stack into a compact single-line layout

## [1.2.0] - 2025-06-20

### Added
- Step-by-step formula tooltips on all result metrics — hover any metric to see a description and numbered derivation with live values (e.g. ① per-token/layer → ② per-token/all-layers → ③ total → ④ per-GPU)
- 16 tooltip description keys (`tt_*`) and 11 formula label keys (`fl_*`) in both English and Chinese
- DeepSeek V3.2-Exp preset (same MLA+MoE backbone as V3 with DSA index attention)
- HuggingFace source provenance in all 16 model preset descriptions (e.g. `Source: HF deepseek-ai/DeepSeek-V3 config.json`)

### Changed
- **weightPerGPU** value no longer shows inline Dense/Expert breakdown — moved into tooltip formula with numbered steps
- Renamed "KV Cache / GPU" → "Active KV Cache / GPU" / "单卡 KV Cache 实际占用"
- Removed "Save as Preset" (localStorage CRUD) feature — now export-only
- Simplified `usePresets` hook to built-ins only (no user preset state)
- Tooltip popover aligned left to avoid right-edge clipping, responsive width (w-64 mobile / w-80 desktop)

### Fixed
- Qwen3-32B description: `head_dim=128` is explicit in config, not derived from `hidden/qHeads` (5120/64=80)
- DeepSeek V4 Flash/Pro descriptions: corrected "DSA attention" → "V4 hybrid CSA+HCA attention" (official terminology); clarified `rope_head_dim=64` is a slice within the 512-dim cache
- Kimi K2.5 description: removed inaccurate "under a multimodal head" and "from text_config" claims; added missing `moe_intermediate_size=2048`
- Decorative glow overflow no longer bleeds outside the rounded results card

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
