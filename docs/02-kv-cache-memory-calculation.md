# KV Cache 内存计算详解

## 1. 标准 MHA/GQA/MQA 注意力

### 单 Token 单层显存

```
size_per_token_per_layer = 2 × num_kv_heads × head_dim × bytes_per_param
```

- `2` = Key 和 Value 各一份
- `head_dim = hidden_size / num_q_heads`
- GQA 时 `num_kv_heads < num_q_heads`，MQA 时 `num_kv_heads = 1`

### vLLM 的 block 级计算

vLLM 使用 paged attention，以 block 为单位分配显存：

```python
# vllm/v1/kv_cache_interface.py :: AttentionSpec.real_page_size_bytes
real_page_size_bytes = 2 × block_size × num_kv_heads × head_size × dtype_size
```

- `block_size` = 默认 **16** tokens（CacheConfig.DEFAULT_BLOCK_SIZE）
- `head_size` = 与上面 `head_dim` 相同
- `dtype_size` = KV cache 数据类型的字节数（可能不同于模型权重精度）

### 单层最大显存

```python
# vllm/v1/kv_cache_interface.py :: FullAttentionSpec.max_memory_usage_bytes
max_memory_per_layer = ceil(max_model_len / block_size) × page_size_bytes
```

### 所有层总显存

```
total_kv_memory = num_blocks × page_size × num_layers
```

或等效地：

```
total_kv_memory = size_per_token_per_layer × max_model_len × num_layers
```

---

## 2. MLA (Multi-head Latent Attention) — DeepSeek 系列

MLA 将 KV 压缩到低维 latent 空间，只缓存压缩后的向量 + 解耦的 RoPE Key。

### 单 Token 单层显存

```
size_per_token_per_layer = (dc + dr) × bytes_per_param
```

- `dc` (kv_lora_rank) = 压缩后的 latent 维度（DeepSeek V3: 512）
- `dr` (qk_rope_head_dim) = 解耦 RoPE 维度（DeepSeek V3: 64）
- 注意：**没有** `2 ×` 前缀，因为 MLA 只有压缩后的 latent 向量（不是分开的 K 和 V）

### vLLM 中的 MLA page size

```python
# vllm/v1/kv_cache_interface.py :: MLAAttentionSpec.real_page_size_bytes
real_page_size_bytes = storage_block_size × num_kv_heads × head_size × dtype_size
```

对于 DeepSeek V3.2 `fp8_ds_mla`：
```python
real_page_size_bytes = block_size × 656  # 自定义布局
```

对于 DeepSeek V4 `fp8_ds_mla`：
```python
real_page_size_bytes = storage_block_size × 584  # 448B NoPE + 128B RoPE + 8B fp8 scale
```

---

## 3. Hybrid 模型（混合全注意力/线性注意力）

如 Qwen3.5，部分层用全 KV cache，部分层用线性注意力（常数内存）。

```
effective_layers = full_attn_layers  # 仅全注意力层需要 KV cache
total_kv_memory = size_per_token_per_layer × effective_layers × seq_len × batch_size
```

在 vLLM 中，hybrid 模型会创建多个 KV cache group，每组有不同的 page_size。

### vLLM 中的 group-based 计算

```python
# vllm/v1/core/kv_cache_utils.py :: get_kv_cache_config_from_groups()
group_size = max(len(group.layer_names) for group in kv_cache_groups)
page_size = get_uniform_page_size([group.kv_cache_spec for group in kv_cache_groups])
num_blocks = available_memory // page_size // group_size
```

---

## 4. KV Cache 量化

vLLM 支持 KV cache 使用比模型权重更低的精度：

| `cache_dtype` | 实际效果 | 额外开销 |
|---------------|---------|---------|
| `"auto"` | 与模型权重同精度 | 无 |
| `"fp8"` / `"fp8_e4m3"` | FP8 per-tensor scale | 1 个 scale 值 |
| `"int8_per_token_head"` | INT8 + per-token-head FP32 scale | `2 × block_size × num_kv_heads × 4` bytes/page |
| `"fp8_per_token_head"` | FP8 + per-token-head FP32 scale | `2 × block_size × num_kv_heads × 4` bytes/page |
| `"nvfp4"` | FP4 packed + FP8 block scales | 特殊布局 |

### 量化对 page_size 的影响

```python
# vllm/v1/kv_cache_interface.py :: AttentionSpec.page_size_bytes
if kv_quant_mode.is_per_token_head:
    page_size = real_page_size + 2 × block_size × num_kv_heads × 4  # FP32 scales
```

---

## 5. 源码引用

| 关键代码 | 文件 |
|---------|------|
| `AttentionSpec.real_page_size_bytes` | `vllm/v1/kv_cache_interface.py` ~170 |
| `FullAttentionSpec.max_memory_usage_bytes` | `vllm/v1/kv_cache_interface.py` ~280 |
| `MLAAttentionSpec.real_page_size_bytes` | `vllm/v1/kv_cache_interface.py` ~382 |
| `AttentionSpec.page_size_bytes` (含量化) | `vllm/v1/kv_cache_interface.py` ~155 |
| `get_num_blocks()` | `vllm/v1/core/kv_cache_utils.py` ~952 |
