# vLLM 影响显存的配置参数

基于 vLLM 源码（`vllm/config/cache.py`, `vllm/config/parallel.py`, `vllm/v1/kv_cache_interface.py`）梳理。

## 1. 直接影响 KV Cache 显存的参数

### `gpu_memory_utilization` (float, 默认 0.92)
- 决定 vLLM 可以使用的 GPU 显存比例
- `requested_memory = total_gpu_memory × gpu_memory_utilization`
- 源码：`vllm/config/cache.py` CacheConfig

### `kv_cache_memory_bytes` (int | None, 默认 None)
- 直接指定 KV cache 的显存大小（bytes），优先级高于 `gpu_memory_utilization`
- 设置后跳过 profiling，直接使用该值
- 源码：`vllm/config/cache.py` CacheConfig

### `block_size` (int, 默认 16)
- KV cache 的页大小（token 数）
- 影响 `page_size_bytes` 和 `num_blocks` 的计算
- 更小的 block_size 减少内部碎片，但增加 block table 管理 overhead
- 源码：`vllm/config/cache.py` CacheConfig.DEFAULT_BLOCK_SIZE

### `cache_dtype` (str, 默认 "auto")
- KV cache 的数据类型，可以与模型权重精度不同
- 支持的值：
  - `"auto"`: 与模型权重相同
  - `"fp8"` / `"fp8_e4m3"` / `"fp8_e5m2"`: FP8 量化（每 tensor 一个 scale）
  - `"fp8_per_token_head"`: FP8 + per-token-head scale
  - `"int8_per_token_head"`: INT8 + per-token-head scale
  - `"nvfp4"`: FP4 packed + FP8 block scales
- 量化会显著减少 KV cache 显存，但 per-token-head 模式有额外 scale 开销
- 源码：`vllm/config/cache.py` CacheConfig, `vllm/v1/kv_cache_interface.py` KVQuantMode

### `num_gpu_blocks_override` (int | None, 默认 None)
- 强制指定 GPU block 数量，覆盖 profiling 结果
- 源码：`vllm/config/cache.py` CacheConfig

### `max_model_len` (int)
- 模型最大上下文长度
- 直接影响 KV cache 的最大显存需求：`max_memory = ceil(max_model_len / block_size) × page_size_bytes × num_layers`
- 设为 -1 时 vLLM 会自动适配到最大可用长度
- 源码：`vllm/model_config.py`

### `kv_offloading_size` (float | None, 默认 None)
- KV cache offloading 到 CPU 的 buffer 大小（GiB）
- 开启后部分 KV 可以 swap 到 CPU，减少 GPU KV cache 压力
- 源码：`vllm/config/cache.py` CacheConfig

## 2. 间接影响显存的参数

### `enable_prefix_caching` (bool, 默认 True)
- 启用前缀缓存，相同前缀的请求共享 KV cache blocks
- 不直接减少显存，但提高 KV cache 利用率（相同前缀只存一份）
- 源码：`vllm/config/cache.py` CacheConfig

### `enforce_eager` (bool, 默认 False)
- 禁用 CUDA graph 捕获
- CUDA graph 会额外占用显存（每个 graph 副本约几十 MB ~ 几百 MB）
- 设置 `enforce_eager=True` 可以节省这部分显存，但降低 decode 速度
- 源码：`vllm/config/compilation.py` CUDAGraphMode

### `tensor_parallel_size` (TP, int, 默认 1)
- 权重和 KV cache 都除以 TP

### `pipeline_parallel_size` (PP, int, 默认 1)
- 权重和 KV cache 都除以 PP

### `data_parallel_size` (DP, int, 默认 1)
- 对 dense 模型: 不减少单卡显存，增加总 GPU 数
- 对 MoE 模型: DP **参与专家权重切分** (TP × DP，via `flatten_tp_across_dp_and_pcp`)
- 源码：`vllm/config/parallel.py:125-127`
- 详细机制见 [05-expert-parallelism-memory.md](./05-expert-parallelism-memory.md)

### `decode_context_parallel_size` (DCP, int, 默认 1)
- Decode Context Parallel，与 TP 共用 GPU
- DCP 将 KV cache 按序列长度切分到多个 DCP rank
- `max_model_len_local = max_model_len / dcp_world_size`
- 源码：`vllm/v1/kv_cache_interface.py` FullAttentionSpec.max_memory_usage_bytes

### `prefill_context_parallel_size` (PCP, int, 默认 1)
- Prefill Context Parallel，与 DCP 类似


### `enable_expert_parallel` (bool, 默认 False)
- MoE 专家并行开关。开启后专家按 TP × DP 均分（而非 TP 列切分）
- 不改变单卡专家权重总量（仍是 total / (TP×DP×PP)），只改变切分方式和通信模式
- EP group size = TP × DP（非独立维度）
- 源码：`vllm/config/parallel.py:162-163`

### `enable_ep_weight_filter` (bool, 默认 False)
- EP 开启时，每卡只从磁盘加载本地专家权重，节省 I/O
- 对运行时显存无影响
- 源码：`vllm/config/parallel.py:165-170`

### `enable_eplb` (bool, 默认 False) + `eplb_config`
- Expert Parallelism Load Balancing
- 支持 redundant experts（额外专家副本以提升负载均衡）
- redundant experts 会增加专家权重显存
- 源码：`vllm/config/parallel.py:57-107`

### `all2all_backend` (str, 默认 "allgather_reducescatter")
- EP 通信后端，影响通信 buffer 大小
- 选项：allgather_reducescatter / deepep_high_throughput / deepep_low_latency / nixl_ep 等
- DeepEP 后端需要额外通信 buffer（~1-3 GB）
- 源码：`vllm/config/parallel.py:186-198`
## 3. 非显存参数但影响 KV cache 行为

### `swap_space` (float, 默认 4.0)
- CPU swap 空间大小（GB）——注意：这是 v0 引擎的参数，v1 引擎用 `kv_offloading_size` 替代

### `max_num_batched_tokens` (int, 默认 2048)
- 单步最大 batch token 数：单次调度迭代中能处理的最大 token 数
- **直接影响显存**：`profile_run()` 用该值作为 dummy batch 大小跑一次前向传播，测量激活值峰值（`torch_peak_increase`），该峰值从 KV cache 预算中扣除
- 激活值估算公式：`actMem ≈ max_num_batched_tokens × layers × hidden_size × precision × K`（K ≈ 0.5 经验值，含 FFN 中间结果 + 注意力输出 + 分配器开销）
- vLLM 默认值：2048（测试用）；实际部署通常设为 8192 或更大
- 源码链路：`gpu_worker.py:387 profile_run()` → `gpu_model_runner.py:6288 _dummy_run(max_num_tokens)` → `gpu_model_runner.py:467 max_num_tokens = scheduler_config.max_num_batched_tokens`
- 与 `max_num_seqs` 的区别：`max_num_batched_tokens` 限总 token 数，`max_num_seqs` 限序列数。一个 step 里可能有 1 条长 prefill + N 条各 1 token decode

### `max_num_seqs` (int)
- 最大并发请求数
- 影响 CUDA graph 捕获的 batch 大小

## 4. 权重显存计算

```
weight_memory = parameters × bytes_per_param / (TP × PP)
```

| 精度 | bytes_per_param |
|-----|----------------|
| FP32 | 4 |
| FP16 / BF16 | 2 |
| INT8 / FP8 | 1 |
| INT4 | 0.5 |

注意：实际加载的权重大小可能因 quantization method 不同而有差异（GPTQ, AWQ 等有 group scales 和 zero points）。

## 5. 框架开销估算

vLLM 的非权重、非 KV cache 显存开销包括：

| 开销项 | 典型大小 | 说明 |
|-------|---------|------|
| NCCL buffer | 0.5-2 GB | 取决于 TP size，TP=1 时几乎为 0 |
| CUDA graph | 0-2 GB | 取决于 graph 数量和 batch 大小，enforce_eager 时为 0 |
| Activation workspace | 1-4 GB | FlashAttention 等临时 buffer |
| Torch allocator fragmentation | ~10% | PyTorch CUDA caching allocator 的碎片 |

在当前估算器中硬编码为 5GB。更精确的做法是：
- TP=1 且 enforce_eager 时：~2 GB
- TP>1 时：~3-5 GB
- CUDA graph 开启时：+1-2 GB

---

## 源码引用

| 配置 | 文件 |
|-----|------|
| CacheConfig | `vllm/config/cache.py` |
| ParallelConfig | `vllm/config/parallel.py` |
| CompilationConfig (CUDAGraphMode) | `vllm/config/compilation.py` |
| KVQuantMode | `vllm/v1/kv_cache_interface.py` ~40 |
