# DP / TP / PP / EP 对显存的影响

## 核心结论

| 并行策略 | Dense 权重 | MoE Expert 权重 | KV Cache | GPU 总数 |
|---------|----------|---------------|---------|---------|
| TP | ÷ TP | ÷ TP (EP关) / 不切分(EP开) | ÷ TP | TP × PP × DP |
| PP | ÷ PP | ÷ PP | ÷ PP | TP × PP × DP |
| DP | 不变 | ÷ DP (MoE 专家权重) | 不变 | TP × PP × DP |
| EP 开关 | 不影响 | 改变切分方式，总量不变 | 不影响 | 不增加 |

## 1. Tensor Parallelism (TP)

### 权重
TP 将模型权重按列/行切分到多个 GPU。每个 GPU 只保存 1/TP 的权重。

```python
weight_per_gpu = total_weights / TP
```

### KV Cache
TP 将 KV heads 切分到不同 GPU。每个 GPU 只缓存 1/TP 的 KV heads。

```python
kv_heads_per_gpu = num_kv_heads / TP  # 必须整除
kv_cache_per_gpu = total_kv_cache / TP
```

**注意**：对于 GQA 模型，`num_kv_heads` 必须能被 TP 整除。例如 LLaMA 3 70B（kv_heads=8, TP=8），每个 GPU 只有 1 个 KV head。

### 激活值
TP 使用 AllReduce 同步中间结果，激活值也被切分。

```python
activation_per_gpu ≈ total_activation / TP
```

### NCCL 通信开销
TP 需要 AllReduce 通信，NCCL 会额外占用显存（通常 0.5-2GB，取决于 TP size 和消息大小）。

## 2. Pipeline Parallelism (PP)

### 权重
PP 将模型按层切分到不同 GPU。每个 GPU 只保存 1/PP 的层（以及对应的权重）。

```python
weight_per_gpu = total_weights / PP
```

### KV Cache
PP 将层分配到不同 stage。每个 GPU 只需要存储自己 stage 内层的 KV cache。

```python
layers_per_gpu = total_layers / PP
kv_cache_per_gpu = total_kv_cache / PP
```

**vLLM 源码中的体现**：不同 PP stage 的 worker 有不同的 `kv_cache_specs`（只包含自己 stage 的层），但最终 `num_blocks` 取所有 worker 的最小值以保证一致性。

```python
# vllm/v1/core/kv_cache_utils.py :: get_kv_cache_configs()
min_num_blocks = min(cfg.num_blocks for cfg in kv_cache_configs)
```

### 激活值
PP 的每个 stage 只计算部分层，激活值也被切分。

## 3. Data Parallelism (DP)

### 权重（Dense 模型）
DP **不切分** dense 模型权重。每个 DP rank 都有完整的模型副本。

```python
weight_per_gpu = total_weights  # 不变！（仅 dense 模型）
```

### KV Cache
DP **不切分** KV cache。每个 DP rank 独立处理不同的请求，拥有完整的 KV cache 容量。

```python
kv_cache_per_gpu = total_kv_cache  # 不变！
```

DP 的作用是**提升吞吐量**：多个副本并行处理不同请求，而非减少单卡显存。

### MoE 模型中的特殊情况

在 vLLM 中，**MoE 专家权重**的切分机制与 dense 权重不同:

```python
# vllm/config/parallel.py:125-127
# "MoE layers will be sharded according to the product of
#  the tensor parallel size and data parallel size."
# → MoE 专家权重的分母是 TP × DP, 而非仅 TP
```

vLLM 的 `flatten_tp_across_dp_and_pcp` 将 DP 纳入 MoE 专家切分：
- **Dense 部分**（attention/shared expert/embedding/dense FFN）：TP × PP 切分，DP 复制
- **Expert 部分**（routed MoE experts）：TP × DP × PP 切分
- **EP 开关**：`enable_expert_parallel` 是布尔值，改变切分方式（按专家 vs 按张量）但不改变单卡总量

**EP group size** = TP × DP（非独立维度，由 TP 和 DP 推导）。

详细机制见 [05-expert-parallelism-memory.md](./05-expert-parallelism-memory.md)。

## 4. 组合并行 (TP × PP × DP)

```python
# Dense 模型
weight_per_gpu = total_weights / (TP × PP)
kv_cache_per_gpu = total_kv_cache / (TP × PP)

# MoE 模型
dense_weight_per_gpu  = dense_params × precision / (TP × PP)
expert_weight_per_gpu = expert_params × precision / (TP × DP × PP)
weight_per_gpu = dense_weight_per_gpu + expert_weight_per_gpu
kv_per_gpu = total_kv / (TP × PP)  # EP 不影响 KV cache

# 通用
overhead_per_gpu ≈ base_overhead + NCCL_buffer(TP)
total_gpus = TP × PP × DP
total_cluster_kv = kv_per_gpu × total_gpus = total_kv × DP
```

DP 增加了集群总 KV cache 容量（×DP），但不改变单卡容量。对 MoE 模型，DP 同时减少单卡专家权重显存。

## 5. vLLM 源码中的 DP 处理

### Worker 初始化时的 GPU 分配

```python
# vllm/v1/worker/gpu_worker.py :: init_device()
dp_local_rank = self.parallel_config.data_parallel_rank_local
tp_pp_world_size = PP × TP
self.local_rank += dp_local_rank × tp_pp_world_size
```

每个 DP rank 分配到不同的 GPU，各自独立运行一个完整的模型副本。

### DP 间的协调

```python
# vllm/v1/worker/dp_utils.py :: coordinate_batch_across_dp()
# DP rank 之间通过 AllReduce 同步 batch 大小
# 仅在使用 CUDA graph 时需要 padding 到相同大小
```

DP rank 之间**不需要**交换模型数据或 KV cache，仅协调调度。

## 6. 对估算器的影响

对于 KV Cache 估算器：

- **TP/PP**: Dense 权重和 KV cache 除以 `TP × PP`（正确）
- **DP**: 对 dense 模型**不影响单卡显存**；但对 **MoE 模型，DP 参与专家权重切分**
- **EP 开关**: 不影响单卡专家权重总量，改变切分方式和通信 overhead

MoE 模型需要将权重拆分为 dense/expert 两部分:

```
dense_weight_per_gpu  = dense_params × precision / (TP × PP)
expert_weight_per_gpu = expert_params × precision / (TP × DP × PP)
weight_per_gpu = dense_weight_per_gpu + expert_weight_per_gpu
kv_per_gpu = total_kv / (TP × PP)   # EP 不影响 KV cache
total_gpus = TP × PP × DP
```

---

## 源码引用

| 关键代码 | 文件 |
|---------|------|
| DP worker 的 GPU 分配 | `vllm/v1/worker/gpu_worker.py` ~init_device |
| DP batch 协调 | `vllm/v1/worker/dp_utils.py` |
| PP 层切分 | `vllm/v1/core/kv_cache_utils.py` ~get_kv_cache_configs |
| MoE DP+TP 切分 | `vllm/config/parallel.py:125-127` |
| EP group = TP × DP | `vllm/distributed/parallel_state.py:1866-1874` |
| EP/TP 切分方式 | `vllm/model_executor/layers/fused_moe/config.py` `make()` |
| 专家均分公式 | `vllm/model_executor/layers/fused_moe/expert_map_manager.py:62-69` |
| `min_num_blocks` 一致性 | `vllm/v1/core/kv_cache_utils.py` ~2045 |
