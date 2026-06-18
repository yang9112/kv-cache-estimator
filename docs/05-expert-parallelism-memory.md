# Expert Parallelism (EP) 对显存的影响

## 核心结论

| 场景 | Dense 权重(Attention/SharedExpert/Embedding) | MoE Expert 权重 | KV Cache | 总 GPU 数 |
|-----|----|---|---|---|
| EP 关闭 (MoE 模型) | ÷ (TP × PP) | ÷ (TP × DP × PP) | ÷ (TP × PP) | TP × PP × DP |
| EP 开启 (MoE 模型) | ÷ (TP × PP) | ÷ (TP × DP × PP) | ÷ (TP × PP) | TP × PP × DP |

**关键结论: EP 开关不影响单卡专家权重总量,只改变切分方式。** DP 无论 EP 开关都会参与 MoE 专家权重切分。

---

## 1. EP 的本质

### `enable_expert_parallel` 是布尔开关,不是独立并行维度

```python
# vllm/config/parallel.py:162-163
enable_expert_parallel: bool = False
"""Use expert parallelism instead of tensor parallelism for MoE layers."""
```

EP group size 由 TP 和 DP 推导而来,不是独立的用户输入:

```python
# vllm/distributed/parallel_state.py ~line 1866-1874
# EP group size = DP × PCP × TP (PCP 默认 1, 即 TP × DP)
group_ranks = (
    all_ranks.transpose(1, 2)
    .reshape(-1, data_parallel_size * prefill_context_model_parallel_size * tensor_model_parallel_size)
    .unbind(0)
)
```

---

## 2. MoE 专家权重的切分机制

### `FusedMoEParallelConfig.make()` 的完整逻辑

源码: `vllm/model_executor/layers/fused_moe/config.py` `FusedMoEParallelConfig.make()`

**无论 EP 开关**,MoE 专家权重始终按 `flatten_tp_size = TP × DP × PCP` 切分:

```python
# 步骤 1: 计算 flatten_tp
tp_size, tp_rank = FusedMoEParallelConfig.flatten_tp_across_dp_and_pcp(
    tp_size_, dp_size_, dp_rank, pcp_size_, pcp_rank
)
# 结果: flatten_tp_size = dp_size × pcp_size × tp_size

# 步骤 2: 根据 EP 开关决定切分方式
use_ep = (dp_size_ * pcp_size_ * tp_size_ > 1) and vllm_parallel_config.enable_expert_parallel

if not use_ep:
    # EP 关闭: 专家按 TP 列切分 (flatten_tp), 每张卡持有所有专家的 1/N 片段
    # ep_size = 1, tp_size = flatten_tp_size
    ...
else:
    # EP 开启: 专家按数量均分, 每张卡持有 ~N/ep_size 个完整专家
    # ep_size = flatten_tp_size, tp_size = 1
    ep_size = tp_size   # = flatten_tp_size
    ep_rank = tp_rank
```

### 两种切分方式对比

| | EP 关闭 (TP-style) | EP 开启 (expert-level) |
|---|---|---|
| 切分粒度 | 每个专家的权重矩阵被 TP 列切分 | 专家整体按数量均分 |
| 每卡持有 | 所有专家,每专家 1/N 权重 | ~N/ep_size 个完整专家 |
| 通信方式 | AllReduce | AllToAll (DeepEP/nixl/etc.) |
| **单卡专家权重** | **expert_weight / (TP×DP×PP)** | **expert_weight / (TP×DP×PP)** |

**单卡权重总量两种方式完全相同。**

---

## 3. DP 对 MoE 的特殊行为

vLLM 文档:

```python
# vllm/config/parallel.py:125-127
data_parallel_size: int = Field(default=1, ge=1)
"""Number of data parallel groups. MoE layers will be sharded according to
the product of the tensor parallel size and data parallel size."""
```

**对 MoE 模型**: DP 不是简单地复制模型,而是参与专家权重切分。`flatten_tp_across_dp_and_pcp` 将 DP rank 也纳入 tensor parallel 的行列切分中。

**对 Dense 模型**: DP 仍然是纯复制,不参与权重切分。

因此对 MoE 模型:
- **Dense 部分**(attention/shared expert/embedding/dense FFN): 仍由 TP × PP 切分,DP 复制
- **Expert 部分**(routed MoE experts): 由 TP × DP × PP 切分

---

## 4. 专家均分公式

```python
# vllm/model_executor/layers/fused_moe/expert_map_manager.py:62-69
def determine_expert_map(ep_size, ep_rank, global_num_experts, ...):
    base_experts = global_num_experts // ep_size
    remainder = global_num_experts % ep_size
    local_num_experts = base_experts + 1 if ep_rank < remainder else base_experts
```

例: 256 experts, ep_size=32 → 每卡 8 experts。ep_size=64 → 每卡 4 experts。

---

## 5. 权重拆分公式

MoE 模型的总参数量 = dense + expert:

```
expert_params = num_moe_layers × num_experts × 3 × hidden_size × moe_intermediate_size
dense_params  = total_params - expert_params
```

示例 (DeepSeek V3, 671B):
- num_moe_layers = 58 (61 - first_k_dense_replace=3)
- num_experts = 256, moe_intermediate_size = 2048, hidden_size = 7168
- expert_params = 58 × 256 × 3 × 7168 × 2048 ≈ 654B
- dense_params = 671B - 654B = 17B

---

## 6. KV Cache 不受 EP 影响

KV cache 由 attention 层产生,与 MoE FFN 层无关。无论 EP 开关:
- KV cache 单卡 = total_kv / (TP × PP)
- Attention 权重单卡 = attention_weight / (TP × PP)

---

## 7. 对 KV Cache 估算器的启示

### 当前实现的问题

```typescript
// src/lib/calc.ts (当前)
const tpPpParallelism = (state.tp || 1) * (state.pp || 1);
const weightPerGPU = weightTotal / tpPpParallelism;  // 所有权重都除以 TP×PP
```

**对 MoE 模型这是错的** — 专家权重应该除以 `TP × DP × PP` 而非 `TP × PP`。DP 对 MoE 专家权重有切分作用,当前代码完全忽略了。

### 正确实现

```typescript
// 需要区分 dense 和 expert 权重
const denseWeightPerGPU = denseWeightTotal / (tp * pp);
const expertWeightPerGPU = expertWeightTotal / (tp * dp * pp);  // DP 参与切分专家
const weightPerGPU = denseWeightPerGPU + expertWeightPerGPU;
```

---

## 8. EP 对框架 overhead 的影响

EP 开启时使用 AllToAll 通信代替 AllReduce,需要额外的通信 buffer:
- DeepEP 高吞吐: 额外 ~1-3 GB 通信 buffer
- DeepEP 低延迟: buffer 较小
- enable_ep_weight_filter: EP 开启时只加载本地专家权重,节省加载内存

这些 overhead 差异通过 profiling 反映在 `non_kv_cache_memory` 中,不单独估算。

---

## 源码引用

| 关键代码 | 文件 |
|---------|------|
| `enable_expert_parallel` 定义 | `vllm/config/parallel.py:162-163` |
| DP 参与 MoE 切分 | `vllm/config/parallel.py:125-127` |
| EP group = TP × DP | `vllm/distributed/parallel_state.py:1866-1874` |
| `flatten_tp_across_dp_and_pcp` | `vllm/model_executor/layers/fused_moe/config.py` `FusedMoEParallelConfig` |
| `make()` EP vs TP 切分 | `vllm/model_executor/layers/fused_moe/config.py` `FusedMoEParallelConfig.make()` |
| 专家均分公式 | `vllm/model_executor/layers/fused_moe/expert_map_manager.py:62-69` |
| EP weight filter | `vllm/config/parallel.py:165-170` |
| All2All backend 选项 | `vllm/config/parallel.py:186-198` |
