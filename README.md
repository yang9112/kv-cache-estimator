# KV Cache Calculator

A lightweight, interactive web tool for estimating GPU memory requirements of KV cache in large language models.

## Features

- **Multi-architecture support** — Standard (GQA/MQA), Multi-Latent Attention (MLA), and Hybrid attention
- **Model presets** — DeepSeek V3/R1, GLM 5, Qwen 3.5, LLaMA 3, Kimi K2.5, MiniMax M2.5, Mixtral, and more
- **Custom config import** — Drop in a `config.json` from HuggingFace to auto-fill architecture parameters
- **Parallelism** — Tensor parallelism (TP) and pipeline parallelism (PP) for multi-GPU memory estimation
- **Precision options** — FP32, FP16/BF16, INT8/FP8, INT4
- **Bilingual UI** — English and Chinese

## Quick Start

```bash
npm install
npm run dev
```

Open http://localhost:3000 in your browser.

## How It Works

The calculator estimates KV cache memory based on:

- **Standard attention**: `2 x kv_heads x head_dim x precision x layers x seq_len x batch_size`
- **MLA** (DeepSeek-style): `(d_c + d_r) x precision x layers x seq_len x batch_size`
- **Hybrid** (e.g. Qwen 3.5): full KV cache only on full-attention layers; linear-attention layers use constant memory

## Tech Stack

- React 19 + TypeScript
- Vite 6
- Tailwind CSS 4
- Lucide Icons
- Framer Motion

## License

MIT
