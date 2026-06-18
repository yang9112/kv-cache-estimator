# vLLM GPU 显存分配流水线

## 总览

vLLM 的 GPU 显存分配分为以下步骤（代码路径：`v1/engine/core.py` `_initialize_kv_caches`）：

```
1. 初始化设备 & 分布式环境
2. 加载模型权重
3. 采集初始显存快照 (init_snapshot)
4. 计算 requested_memory = total_gpu_memory × gpu_memory_utilization
5. 执行 profile_run（用 dummy 数据跑一次前向）
6. 采集 profiling 后的显存快照
7. 计算 available_kv_cache_memory
8. 生成 KVCacheConfig (num_blocks)
9. 分配 KV cache tensors
10. Warmup kernels
```

## 核心公式

### 1. requested_memory

```python
# vllm/v1/worker/utils.py :: request_memory()
requested_memory = ceil(total_gpu_memory × gpu_memory_utilization)
```

- `total_gpu_memory` = `torch.cuda.mem_get_info()[1]`，即 GPU 总显存
- `gpu_memory_utilization` 默认 **0.92**（vLLM CacheConfig）

### 2. available_kv_cache_memory

```python
# vllm/v1/worker/gpu_worker.py :: determine_available_memory()
available_kv_cache_memory = requested_memory - non_kv_cache_memory - cudagraph_memory
```

其中：

```python
non_kv_cache_memory = non_torch_memory + torch_peak_increase + weights_memory
```

- **weights_memory**: 模型权重大小 = 参数量 × bytes_per_param
- **torch_peak_increase**: profile_run 中 PyTorch 分配的峰值增量（激活值）
- **non_torch_memory**: 非 PyTorch 的 CUDA 显存（NCCL buffer 等）
- **cudagraph_memory**: CUDA graph 捕获的显存估算（可通过 `VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS` 环境变量启用）

### 3. num_blocks 计算

```python
# vllm/v1/core/kv_cache_utils.py :: get_num_blocks()
num_blocks = available_memory // page_size // num_layers
```

- `page_size` = 单个 block 的字节数（见下文 `page_size_bytes`）
- `num_layers` = 模型层数（对于 hybrid 模型，是 group_size，即各组层数的最大值）

### 4. vLLM 也可通过 `kv_cache_memory_bytes` 直接指定 KV cache 显存

```python
# 如果指定了 kv_cache_memory_bytes，则跳过 profiling 直接使用该值
if kv_cache_memory_bytes := self.cache_config.kv_cache_memory_bytes:
    return kv_cache_memory_bytes
```

此参数优先级高于 `gpu_memory_utilization`。

## 显存构成图

```
┌─────────────────────────────────────────────────────┐
│                  GPU Total Memory                    │
├─────────────────────────────────────────────────────┤
│  Non-vLLM Processes (1 - gpu_memory_utilization)    │
├─────────────────────────────────────────────────────┤
│  vLLM Requested Memory (gpu_memory_utilization)     │
│  ┌───────────────────────────────────────────────┐  │
│  │  Model Weights (参数量 × bytes_per_param)      │  │
│  ├───────────────────────────────────────────────┤  │
│  │  Activation Peak (profile_run 测量)            │  │
│  ├───────────────────────────────────────────────┤  │
│  │  Non-Torch Memory (NCCL, etc.)                │  │
│  ├───────────────────────────────────────────────┤  │
│  │  CUDA Graph Memory (可选)                     │  │
│  ├───────────────────────────────────────────────┤  │
│  │  ★ KV Cache (available_kv_cache_memory) ★     │  │
│  └───────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────┘
```

## 源码引用

| 关键函数 | 文件 | 行号 |
|---------|------|------|
| `request_memory()` | `vllm/v1/worker/utils.py` | 405 |
| `determine_available_memory()` | `vllm/v1/worker/gpu_worker.py` | ~240 |
| `get_num_blocks()` | `vllm/v1/core/kv_cache_utils.py` | 952 |
| `get_kv_cache_configs()` | `vllm/v1/core/kv_cache_utils.py` | 1937 |
| `_initialize_kv_caches()` | `vllm/v1/engine/core.py` | ~260 |
