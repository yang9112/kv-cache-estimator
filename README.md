# KV Cache Estimator

[![CI & Deploy](https://github.com/yang9112/kv-cache-estimator/actions/workflows/main.yml/badge.svg)](https://github.com/yang9112/kv-cache-estimator/actions/workflows/main.yml)
[![GitHub Pages](https://img.shields.io/badge/Demo-GitHub%20Pages-brightgreen)](https://yang9112.github.io/kv-cache-estimator/)
[![Release](https://img.shields.io/github/v/release/yang9112/kv-cache-estimator)](https://github.com/yang9112/kv-cache-estimator/releases)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

**English | [中文](#中文)**

A lightweight, interactive web tool for estimating GPU memory requirements of KV cache in large language model inference.

**Live Demo: https://yang9112.github.io/kv-cache-estimator/**

---

## Table of Contents

- [Features](#features)
- [Supported Models](#supported-models)
- [Calculation Methodology](#calculation-methodology)
- [Quick Start](#quick-start)
- [Development](#development)
- [Tech Stack](#tech-stack)
- [Contributing](#contributing)
- [License](#license)

## Features

- **Multi-architecture support** — Standard (MHA / GQA / MQA), Multi-Latent Attention (MLA), and Hybrid attention
- **Model presets by family** — Built-in presets organized by model family; file-based configuration (`src/configs/models/*.json`) makes it easy to add or remove models
- **vLLM KV budget estimation** — Calculate maximum usable KV cache and concurrent tokens per GPU based on VRAM capacity and utilization
- **HuggingFace config import** — Drop in a `config.json` to auto-fill architecture parameters, including MoE fields
- **Step-by-step formula tooltips** — Hover any result metric to see a description and numbered derivation with live values (e.g. ① per-token/layer → ② per-token/all-layers → ③ total → ④ per-GPU)
- **Parallelism awareness** — Tensor (TP), Pipeline (PP), Data (DP), and Expert (EP) parallelism with correct MoE weight sharding based on [vLLM](https://github.com/vllm-project/vllm)
- **Precision options** — FP32, FP16 / BF16, INT8 / FP8, INT4
- **Bilingual UI** — English and Chinese

## Supported Models

| Family | Models |
|---|---|
| DeepSeek | V2, V3, V3.2-Exp, V4 Flash, V4 Pro |
| GLM | 4-9B, 4-Flash, 5 |
| Qwen | 2-72B, 3-32B, 3.5-397B |
| LLaMA | 2-7B, 3-8B, 3-70B |
| Kimi | K2.5 |
| MiniMax | M2.5 |

Built-in presets are sourced from each model's official HuggingFace `config.json`. See [`src/configs/models/README.md`](src/configs/models/README.md) for how to add custom presets.

## Calculation Methodology

### KV Cache Memory

| Architecture | Formula |
|---|---|
| **Standard** (MHA / GQA / MQA) | `2 × kv_heads × head_dim × precision × layers × seq_len × batch_size` |
| **MLA** (DeepSeek) | `(d_c + d_r) × precision × layers × seq_len × batch_size` |
| **Hybrid** (Qwen 3.5) | Full KV on attention layers only; linear layers use constant memory |

### MoE Weight Sharding

Based on [vLLM source](https://github.com/vllm-project/vllm):

| Weight Type | Sharding |
|---|---|
| Dense (attention / FFN / embedding) | `÷ (TP × PP)` |
| Expert (routed MoE) | `÷ (TP × DP × PP)` — DP participates via vLLM `flatten_tp` |
| KV Cache | `÷ (TP × PP)` — not affected by DP or EP |

### vLLM KV Budget

```
Usable VRAM   = GPU Memory × Utilization
KV Budget     = Usable VRAM − (Dense Weights + Expert Weights) − Overhead
Max Tokens    = KV Budget ÷ Token Size per GPU
```

## Quick Start

```bash
npm install
npm run dev
```

Open http://localhost:3000 in your browser.

## Development

```bash
npm run build    # Type-check and production build
npm run preview  # Preview production build locally
npm run lint     # Type-check only
npm run test     # Validate model presets
npm run verify   # Verify preset correctness
npm run clean    # Remove build artifacts
```

## Tech Stack

React 19 · TypeScript · Vite 6 · Tailwind CSS 4 · Lucide Icons · Framer Motion

## Contributing

Contributions are welcome! Feel free to open issues or submit pull requests.

## License

[MIT](LICENSE)

---

## 中文

一个轻量级的交互式 Web 工具，用于估算大语言模型推理时 KV Cache 的 GPU 显存占用。

**在线体验: https://yang9112.github.io/kv-cache-estimator/**

### 功能特性

- **多架构支持** — 标准注意力 (MHA / GQA / MQA)、多潜空间注意力 (MLA)、混合架构 (Hybrid)
- **模型预设按系列分组** — 内置预设按模型系列组织；基于文件配置 (`src/configs/models/*.json`)，增删文件即可调整默认配置
- **vLLM KV 预算估算** — 根据单卡显存容量和占用率，计算可用 KV Cache 上限和最大并发 Token 数
- **HuggingFace 配置导入** — 拖入 `config.json` 自动填充架构参数（含 MoE 字段）
- **悬停公式推导** — 鼠标移至任一结果指标，展示说明与分步推导公式（如 ① 单Token/单层 → ② 单Token/全层 → ③ 总量 → ④ 单卡）
- **并行策略** — 张量并行 (TP)、流水线并行 (PP)、数据并行 (DP)、专家并行 (EP)，MoE 权重按 [vLLM](https://github.com/vllm-project/vllm) 源码正确切分
- **精度选项** — FP32、FP16 / BF16、INT8 / FP8、INT4
- **中英双语** — 界面支持中文和英文

### 支持模型

| 系列 | 模型 |
|---|---|
| DeepSeek | V2, V3, V3.2-Exp, V4 Flash, V4 Pro |
| GLM | 4-9B, 4-Flash, 5 |
| Qwen | 2-72B, 3-32B, 3.5-397B |
| LLaMA | 2-7B, 3-8B, 3-70B |
| Kimi | K2.5 |
| MiniMax | M2.5 |

内置预设均来自各模型官方 HuggingFace `config.json`。添加自定义预设详见 [`src/configs/models/README.md`](src/configs/models/README.md)。

### 计算原理

#### KV Cache 显存

| 架构 | 公式 |
|---|---|
| **标准** (MHA / GQA / MQA) | `2 × KV头数 × 头维度 × 精度字节数 × 层数 × 序列长度 × 批次大小` |
| **MLA** (DeepSeek) | `(dc + dr) × 精度字节数 × 层数 × 序列长度 × 批次大小` |
| **混合** (Qwen 3.5) | 仅全注意力层使用完整 KV Cache；线性注意力层为常数内存 |

#### MoE 权重切分

基于 [vLLM 源码](https://github.com/vllm-project/vllm)：

| 权重类型 | 切分方式 |
|---|---|
| Dense (attention / FFN / embedding) | `÷ (TP × PP)` |
| Expert (路由 MoE) | `÷ (TP × DP × PP)` — DP 参与切分（vLLM `flatten_tp`） |
| KV Cache | `÷ (TP × PP)` — 不受 DP 或 EP 影响 |

#### vLLM 预算

```
可用显存  = 单卡显存 × 占用率
KV 预算   = 可用显存 − (Dense 权重 + Expert 权重) − 框架开销
最大 Token = KV 预算 ÷ 单 Token 显存占用
```

### 快速开始

```bash
npm install
npm run dev
```

打开 http://localhost:3000 即可使用。

### 开发

```bash
npm run build    # 类型检查并构建生产版本
npm run preview  # 本地预览生产构建
npm run lint     # 仅类型检查
npm run test     # 验证模型预设
npm run verify   # 校验预设正确性
npm run clean    # 清除构建产物
```

### 技术栈

React 19 · TypeScript · Vite 6 · Tailwind CSS 4 · Lucide Icons · Framer Motion

### 开源协议

[MIT](LICENSE)
