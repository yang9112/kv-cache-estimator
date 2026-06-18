# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

KV Cache Estimator — an interactive web tool for estimating GPU memory requirements of KV cache in LLM inference. Supports Standard (MHA/GQA/MQA), MLA (DeepSeek-style), and Hybrid attention architectures. Bilingual UI (English/Chinese). Deployed to GitHub Pages.

## Commands

```bash
npm run dev       # Start dev server at http://localhost:3000
npm run build     # Type-check (tsc --noEmit) then production build via Vite
npm run lint      # Type-check only (tsc --noEmit) — no ESLint/Prettier configured
npm run preview   # Preview production build
npm run clean     # Remove dist/
```

No test framework is configured. There are no tests.

## Architecture

**Stack**: React 19 + TypeScript (strict) + Vite 6 + Tailwind CSS 4 + Framer Motion + Lucide React

**Source structure** (7 files total, intentionally compact):

- `src/main.tsx` — Entry point, renders `<App />` into `#root`
- `src/App.tsx` — Root component; wraps in `LanguageProvider`, renders header + `<Calculator />`
- `src/types.ts` — `Precision`, `ModelPreset`, `CalculatorState` types; `PRESETS` array (hardcoded model presets); `PRECISIONS` array
- `src/components/Calculator.tsx` — Monolithic ~550-line component with all UI state and layout. Contains inline `InputGroup` and `ResultRow` subcomponents. Two-column layout: inputs left, results right
- `src/lib/calc.ts` — Core calculation logic (`calculateKV`, `computeExpertParams`, `formatBytes`). Three KV formulas:
  - Standard: `2 × kv_heads × head_dim × precision × layers × seq_len × batch_size`
  - MLA: `(d_c + d_r) × precision × layers × seq_len × batch_size`
  - Hybrid: Standard formula for full-attention layers only
  - MoE expert params: `num_experts × 3 × hidden_size × moe_intermediate_size × moe_layers`
  - Weight split: dense / (TP×PP) + expert / (TP×DP×PP) — DP shards MoE experts (vLLM flatten_tp)
  - vLLM budget: `(Usable VRAM - Weights - Overhead) / token size per GPU`
- `src/lib/configParser.ts` — Parses HuggingFace `config.json` to auto-fill model parameters
- `src/lib/i18n.tsx` — Custom `LanguageProvider` context + `useI18n` hook with inline en/zh translation dictionaries

**Path alias**: `@/*` → project root (configured in both `tsconfig.json` and `vite.config.ts`)

**Deployment**: GitHub Pages with base path `/kv-cache-estimator/` (set in `vite.config.ts`); dev server uses `/`

## CI

`.github/workflows/main.yml` — On push/PR to `main`: type-check + build + deploy to GitHub Pages (deploy only on push, not PRs). Node 20.

## Conventions

- 2-space indentation, LF line endings (`.editorconfig`)
- Tailwind CSS 4 (imported via `@import "tailwindcss"` in `index.css`)
- No ESLint or Prettier — `npm run lint` is purely `tsc --noEmit`
- Default UI language auto-detects from browser, falls back to Chinese (`zh`)
