# Qwen3.6-27B（`qwen3_5_text`）拓扑与 kernel 对照表

来源：`/models/Qwen3.6-27B-MXFP4-HMT-AutoRound/config.json` + 实测 trace
`Qwen3.6-27B-MXFP4-HMT-AutoRound-all-xpugraph-in3300-out20-bs1-tp1-260917-005123`。

## 层结构

```
num_hidden_layers = 64,  full_attention_interval = 4
layer_types       = [linear, linear, linear, full] x 16
                  = 48 x GDN(linear attention) + 16 x full attention
hidden = 5120, intermediate = 17408, vocab = 248320, dtype = bf16
```

| 全注意力层 | 值 |
|---|---|
| `num_attention_heads` | 24 |
| `num_key_value_heads` | 4（GQA 6:1） |
| `head_dim` | 256 |
| `attn_output_gate` | **true** —— q 侧输出翻倍 |
| rope | mrope interleaved，`partial_rotary_factor = 0.25` |

| GDN 层 | 值 |
|---|---|
| `linear_num_key_heads` x `linear_key_head_dim` | 16 x 128 |
| `linear_num_value_heads` x `linear_value_head_dim` | 48 x 128 |
| `linear_conv_kernel_dim` | 4 |
| `mamba_ssm_dtype` | float32 |
| recurrent state | 48 x 128 x 128 x 4 B = **3.15 MB / 层** |
| conv state | (2·16·128 + 48·128) x 4 x 2 B = **81.9 KB / 层** |

## 每层 Linear 的 shape

| linear | 所在层 | K | N | 量化 | 权重字节(MXFP4) |
|---|---|---|---|---|---|
| `qkv_proj` | full | 5120 | 14336 = 2·24·256 + 2·4·256 | MXFP4 | 39.03 MB |
| `o_proj` | full | 6144 = 24·256 | 5120 | MXFP4 | 16.73 MB |
| `in_proj_qkvz` | gdn | 5120 | 16384 = 2·16·128 + 48·128 + 48·128 | MXFP4 | 44.60 MB |
| `in_proj_ba` | gdn | 5120 | 96 = 2·48 | MXFP4 | 0.26 MB |
| `gdn_out_proj` | gdn | 6144 = 48·128 | 5120 | MXFP4 | 16.73 MB |
| `gate_up` | 全部 | 5120 | 34816 | MXFP4 | 94.77 MB |
| `down_proj` | 全部 | 17408 | 5120 | MXFP4 | 47.37 MB |
| `lm_head` | head | 5120 | 248320 | **bf16** | 2543.30 MB |

一次 decode 需读取 **14.43 GiB**：MXFP4 12.06 GiB + bf16 `lm_head` 2.37 GiB。

checkpoint 里的权重名与 runtime 的融合关系（所有 `model.language_model.layers.*`
的 Linear 都是 `weight_packed` + `weight_scale`，`lm_head.weight` 是裸 bf16）：

| checkpoint | runtime |
|---|---|
| `self_attn.{q,k,v}_proj` | `qkv_proj` |
| `mlp.{gate,up}_proj` | `gate_up` |
| `linear_attn.in_proj_qkv` + `in_proj_z` | `in_proj_qkvz` |
| `linear_attn.in_proj_b` + `in_proj_a` | `in_proj_ba` |

## 一次前向的 kernel 计数（sanity check 的理论值）

| 量 | 值 |
|---|---|
| Dense-GEMM | 5·48 + 4·16 + 1 = **305** |
| Hadamard 旋转（prefill） | 19·48 + 16·16 = **1168** |
| Hadamard 旋转（decode） | 5·48 + 4·16 = **304** |
| `Quantize(mxfp4)` / `Quant-scale cast` | **304** |
| `Activation(SiLU)` / `Norm(RMS)` | 64 / 128 |
| `GDN-Norm/Gate` / `GDN-Attn(recurrent)` | 48 / 96 |
| `GDN-Attn(chunk)`（仅 prefill） | 7·48 = **336** |
| `KVCache-Write` / `FullAttn-OutGate` / FMHA | 16 / 16 / 16(+16 伴生) |
| decode 一步的 kernel 总数 | **1776** |
| prefill 的 kernel 总数（剔除图重放后） | **3319** |

## 层内 kernel 顺序（decode，M=1）

GDN 层：
```
[H rot, quant, e8m0 cast] x2        <- 同一个 hidden 被旋转+量化了两次（qkvz 与 ba 各一份）
triton_poi_fused_zeros_4            <- GDN state 初始化
gemm_kernel {32;1;1}{64;8;1}        <- in_proj_qkvz   134 us
gemm_kernel {1;1;1}{128;4;1}        <- in_proj_ba      20 us（单 work-group）
gdn::causal_conv1d_kernel                              4.8 us
gdn::gated_delta_rule_kernel                           7.5 us
triton_per_fused_..._silu_view_0    <- GDN 输出 RMSNorm + 门控  10.9 us（grid {2;1;1}）
[H rot, quant, cast] -> gemm        <- gdn_out_proj    79 us
triton_red_fused_..._mm_view_1      <- 残差 + post-attn norm
[H rot, quant, cast] -> gemm        <- gate_up        292 us
triton_poi_fused_mm_mul_silu_..._2  <- SwiGLU
[H rot, quant, cast] -> gemm        <- down_proj      214 us
triton_red_fused_..._rms_norm_3     <- 残差 + 下一层 input norm（下一层是 GDN）
```

全注意力层：
```
[H rot, quant, cast] -> gemm        <- qkv_proj       129 us
triton_poi_fused_4 / triton_red_fused_5              <- split + q/k RMSNorm
triton_poi_fused_..._select_split_where_6 / _7       <- mrope
vllm::reshape_and_cache_flash_strided_kernel
XeFMHAFwdSplitKVKernel  {1;1;128}                     21.6 us
ReduceSplitK            {1;24;1}                      20.5 us
triton_poi_fused_mm_mul_sigmoid_view_0               <- output gate（swish）
[H rot, quant, cast] -> gemm        <- o_proj         78 us
... 与 GDN 层相同的 MLP ...
triton_red_fused_..._rms_norm_mm_view_3  <- 残差 + 下一层 input norm（下一层是 full）
```

`lm_head` 段：
```
triton_red_fused_..._rms_norm_3     <- 模型最终 norm（不是层起点！）
[H rot, quant, cast] x2 + zeros_4   <- decode layer-0 的前导被挪到了这里
IndexKernelFunctor {5;1;1}          <- 取最后一个 token 的 hidden
gemm_kernel {32;1;1}{32;2;8}        <- lm_head, bf16, 2856 us
ArgMax reduce_kernel                <- sampler，窗口边界
```

## prefill 与 decode 的路径差异

| | prefill (T=3300) | decode (M=1) |
|---|---|---|
| GDN attention | chunk 路径，7 个 `gdn::Chunk*` / `tiled_kernel_launcher` | recurrent 路径，2 个 kernel |
| full attention | `XeFMHAFwdKernel`（+1 个 1.6 µs 伴生 kernel） | `XeFMHAFwdSplitKVKernel` + `ReduceSplitK` |
| Hadamard 旋转 | 每次 3 个（K=5120/6144）或 7 个（K=17408）kernel | 每次 1 个 |
| GEMM ND-range | `{32;1;1} {128;4;1}` | `{32;1;1} {64;8;1}` |
