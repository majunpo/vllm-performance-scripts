# 计量陷阱与分类口径（GDN + Full-Attention 混合模型）

每一条都在 `Qwen3.6-27B-MXFP4-...-260917-005123` 这条 trace 上实际踩到过，都会产生
"看起来合理但错误"的结果。

## 1. XPU-Graph 把整张图的 kernel 压到同一个时间戳

实测 **56.9 %** 的 kernel 与他人共享 `ts`，单个时间戳上最多 **1154** 个 kernel。

- **duration 有效，start time 无效。**
- union busy / GPU idle / bubble 全部失效，脚本不做这类分析。
- 一律用 **sum-of-durations**。
- 排序必须 `evs.sort(key=lambda e: e[0])`（只按 ts，稳定排序），否则同戳事件被重排，
  layer walker 的状态机立刻错乱。
- 要看时间线必须先 **reflow**：把每个同戳分组按采集顺序首尾相接重排
  （`make_hybrid_perfetto_trace.py`）。**游标必须 clamp 到上一组的结束时刻**，否则
  本 trace 那个总时长 33.9 ms、却被打在下一个时间戳前 22.3 ms 的分组会和后面重叠，
  slice 堆叠、step 边界错位。reflow 之后顺序和时长正确，绝对位置误差在一次图提交之内。

## 2. decode 的图重放被盖在 prefill 窗口里

这是本模型特有、且最容易出错的一条。

`cudagraph_mode = FULL_AND_PIECEWISE`：decode 走整图重放，提交时刻被打成整批 kernel
的时间戳。实测有一批 **1153 个 kernel（33.9 ms）** 的提交时间戳落在 prefill 执行区间
的中段（+166.185 ms，而 prefill 跨度 3.5 ms → 1809.8 ms），排序后整块插进了 prefill
的中间。

症状：prefill 窗口里出现 `causal_conv1d`（M=1 的 decode 形状）、`gemm_kernel[SIMD16
{1; 5; 1} ...]` 等只可能来自 decode 的 kernel。

处理：`split_graph_blocks()` 把窗口内"被 ≥64 个 kernel 共享的时间戳"整组剔除。剔除后
prefill 的 kernel 计数刚好落在理论值上（见 SKILL.md 的 sanity 表），是这个做法正确的
证据。不剔除则 prefill 虚高 33.9 ms（1.9 %），且分类计数全部对不上。

## 3. 两个 sampler 窗口会把一次图重放劈成两半

19 个 decode 窗口里，第 1、2 个窗口分别有 1775 / 622 个 kernel，其余 17 个都是整齐的
1776。原因同上：一次重放的 kernel 被时间戳拉到了 sampler 边界的另一侧。

处理：只取**众数尺寸**的窗口做统计（脚本会打印被跳过的窗口号）。对比分析一律用单步
精确表，不要用把所有窗口平均后的数字。

## 4. AutoRound 的在线 Hadamard 旋转也叫 `gemm_kernel`

`quantization_config.rotation_config.allow_online_rotation = true`。AutoRound 导出的权重
已经是 `W @ V`（离线折好），`V` 是 block_size=32、按 `1/sqrt(32)` 归一化的块对角
Hadamard；激活侧必须在线补上 `x @ V`。实现就是一个 matmul
（`vllm/model_executor/layers/quantization/inc/rotation.py::rotate_input`）：

```python
return (x.unflatten(-1, (-1, block_size)) @ matrix).flatten(-2, -1)
```

`[tokens, K]` → `[tokens·K/32, 32] @ [32, 32]`，bf16 `torch.matmul` → oneDNN →
在 trace 里同样叫 `gemm_kernel`。

- prefill：每个 GDN 层 19 个、每个 full 层 16 个，共 **1168** 次（39.8 ms）
- decode：每个量化 Linear 1 次，共 **304** 次（0.37 ms）

不区分的话 Dense-GEMM 计数会从 305 虚高到 1473（prefill）/ 609（decode），逐 shape 的
TFLOPS 全部作废。

判据是 **work-group 形状**，不是 duration：

| ND-range | 含义 |
|---|---|
| `{32;1;1} {128;4;1}` | prefill 的真实 Linear |
| `{1;1;1} {128;4;1}` | decode 的 `in_proj_ba`（N=96，单 work-group） |
| `{32;1;1} {64;8;1}` | decode 的真实 Linear |
| `{32;1;1} {32;2;8}` | `lm_head`（bf16 / oneDNN） |
| `{128;1;1} {32;4;1}`、`{58;1;1} {32;4;1}` | prefill 的 Hadamard 旋转 |
| `{1;n;1} {64;8;1}`，n = K/1024 | decode 的 Hadamard 旋转（n = (K/32)/32，K=5120→5，K=17408→17） |

## 5. `in_proj_ba` 会跑到 `in_proj_qkvz` 前面

GDN 层的两个输入投影没有顺序依赖，实测 48 层里有 1 层把 `in_proj_ba` 排在了
`in_proj_qkvz` 之前。纯按顺序归属会让 `in_proj_qkvz` 出现一个 19.8 µs 的离群值
（中位数 134 µs），`GB/s@min` 直接变成 2253 GB/s 这种明显不可能的数。

处理：decode 下 `in_proj_ba` 的 ND-range 是唯一的 `{1;1;1} {128;4;1}`，遇到就把这一对
交换回来。

## 6. 最后一个 `..._rms_norm_3` 是模型的 final norm，不是层起点

`..._rms_norm_*_3` 这个融合 kernel = 残差加 + **下一层**的 input norm，所以它出现在上一
层的末尾。一次前向里它出现 `num_layers` 次：`num_layers - 1` 次层起点，加上最后的
final norm。直接按 marker 切会得到 65 段，第 65 段会把 `lm_head` 当成
`in_proj_qkvz`。

处理：切完之后截断到 `num_layers` 段，剩下的归为 `head` 段，其中唯一的主 GEMM 就是
`lm_head`。

## 7. decode 的 layer-0 前导被排到了窗口末尾

decode 窗口开头是 embedding gather 之后**直接**跟 `in_proj_qkvz`，没有 Hadamard 和量化；
而窗口结尾在 final norm 之后多出 2 组 `Hadamard + quant + scale` 和一个
`triton_poi_fused_zeros_4`。那是 layer 0 的前导，被图划分挪到了末尾。

后果：整个前向的量化计数仍然是 `5G + 4F`，所以**不要**按"每层都在层内自带前导"去校验；
按整窗口总数校验。

## 8. trace 停止时仍在飞行的 kernel

本 trace 有 2 个 `gemm_kernel`（一个 qkvz、一个 ba）拿不到完成时间戳，unitrace 给出
`dur ≈ 48823 s`。不剔除的话单个事件就淹没整个 run。脚本用 `--max-kernel-s`（默认 10 s）
丢弃并告警。

## 9. prefill FMHA 的伴生 kernel 名字里带 `XeFMHAFwdSplitKVKernel`

prefill 每个 full-attention 层除了 `XeFMHAFwdKernel`（930 µs）还会发一个 1.6 µs 的退化
split-KV kernel。它的 mangled name 里含 `XeFMHAFwdSplitKVKernel`，按 decode FMHA 统计会
得到 8274 GB/s 这种荒谬值。判定 reduce / decode FMHA 时要先排除 `ReduceSplitK`，并按
phase 分开算。

---

# 分类口径

规则顺序很重要。

| 类别 | 匹配 |
|---|---|
| `FullAttn-SplitK-Reduce` | `ReduceSplitK` —— **必须排在 FMHA 前面** |
| `FullAttn-FMHA` | `XeFMHAFwdKernel`、`XeFMHAFwdSplitKVKernel` |
| `GDN-Attn(chunk)` | `gdn::Chunk*`、`gdn::chunk_update_states`、`gdn::tiled_kernel_launcher` |
| `GDN-Attn(recurrent)` | `gdn::causal_conv1d_kernel`、`gdn::gated_delta_rule_kernel` |
| `Dense-GEMM` / `Hadamard-Rotation` | `gemm_kernel`，按 work-group 形状二分（见第 4 条） |
| `KVCache-Write` | `reshape_and_cache` —— **必须排在 Quantize 前面**（名字里含 `Fp8KVCacheDataType`） |
| `Quantize(mxfp4)` | `per_token_group_quant_mxfp4`、`float_e2m1`、`fp4` |
| `Quant-scale cast` | `Float8_e8m0`（block scale 的独立 elementwise 拷贝） |
| `GDN-Norm/Gate` | `triton_per_fused__to_copy_add_mean_mm_mul_pow_rsqrt_silu_view_0` |
| `FullAttn-OutGate` | `triton_poi_fused_mm_mul_sigmoid_view_0`（`attn_output_gate`） |
| `Activation(SiLU)` | `triton_poi_fused_mm_mul_silu_slice_view_2` |
| `Norm(RMS)` | `..._rms_norm_3` / `..._rms_norm_mm_view_1` / `..._rms_norm_mm_view_3` |
| `QK-Norm/RoPE` | `triton_poi_fused_4`、`triton_red_fused_5`、`..._select_split_where_6`、`triton_poi_fused_7`、mrope 的 `cat_neg*` |
| `GDN-State-Init` | `triton_poi_fused_zeros_4` |

## 为什么这些 op 必须合并成一个桶

torch.compile 把多个 op 融进同一个 Triton kernel，device 上只有一次 launch、一个
duration，**物理上无法分开**：

- `triton_per_fused__to_copy_add_mean_mm_mul_pow_rsqrt_silu_view_0`
  = GDN 输出的 per-head RMSNorm + SiLU 门控 + 残差
- `triton_red_fused__to_copy_add_fused_add_rms_norm_mm_view_1`
  = attention 输出残差加 + post-attention RMSNorm
- `triton_poi_fused_mm_mul_silu_slice_view_2` = split + SiLU + 乘（SwiGLU）
- `triton_poi_fused_arange_bitwise_and_eq_index_lt_remainder_select_split_where_6`
  = mrope 的 interleaved section 选择

## GEMM 后端识别（最关键的一条判据）

| trace 里的名字 | 实际后端 |
|---|---|
| `cutlass::gemm::kernel::GemmUniversal<...MainloopIntelXeXMX16...>` | CUTLASS-SYCL，走 XMX |
| `gemm_kernel[SIMD16 {...} {...}]` | **oneDNN JIT**（名字里不带任何 dtype 模板参数） |

bf16 的 `lm_head` 走 `torch.nn.functional.linear` → PyTorch XPU → oneDNN，它在所有 XPU
trace 里都叫 `gemm_kernel`。因此**量化 Linear 也只出现 `gemm_kernel`、整条 trace 里
没有一个 `GemmUniversal`，就说明量化路径没有命中任何 XMX kernel**。

**但"没命中"不等于"回退"**，这两种情况要分清，结论和行动项完全不同：

| 情况 | 如何确认 | 行动项 |
|---|---|---|
| 有 XMX kernel 但没被选中 | `_POSSIBLE_*_KERNELS[PlatformEnum.XPU]` / `_LINEAR_BACKEND_KERNEL_MAP` 里存在 CUTLASS 候选 | 改配置 / 修选择条件 |
| 根本没有 XMX kernel | 该平台的候选列表里只有 oneDNN 那一个 | 需要新写 kernel |

MXFP4 属于后者。源码链（已核实）：

```
INCMxfp4LinearMethod                     vllm/model_executor/layers/quantization/inc/schemes/inc_mxfp4_linear.py
  -> XPUMxFp4LinearKernel                vllm/model_executor/kernels/linear/mxfp4/xpu.py
  -> torch.ops._xpu_C.fp4_gemm(...)      vllm_xpu_kernels
  -> oneDNN::dnnl_matmul_w4a4_fp4        csrc/xpu/onednn/fp4_gemm_w4a4.h
     joint_dtypes_t::mxfp4_bf16          E2M1 权重/激活 + E8M0 scale 作为 primitive attr，
                                         kernel 内解压，累加/输出 BF16
```

`_POSSIBLE_MXFP4_KERNELS[PlatformEnum.XPU]` 只有 `XPUMxFp4LinearKernel` 一个候选，
CUTLASS-SYCL 在这棵树里只用于 MoE 的 grouped GEMM。MXFP8 则另有
`XPUW8A8FP8LinearKernel` / `XPUFp8BlockScaledMMKernel` 和 CUTLASS 路径。

## 计算精度怎么确认

**kernel 名字里没有 dtype，trace 单独无法判定计算精度。** 四条互补的判据，按可信度排序：

1. **`ONEDNN_VERBOSE=1` 重抓（最直接，先做这个）**。每条 matmul 会打印实现名 +
   完整的内存描述符 + attr。实测这个模型只有两类：

   ```
   14592  jit:gemm:any | src:f4_e2m1 wei:f4_e2m1 dst:bf16
                       | attr-scales:src0:3:e8m0:1x32+wei:3:e8m0:32x1
   14616  jit:gemm:any | src:bf16     wei:bf16     dst:bf16
   ```

   这直接坐实了 **W4A4**（src 和 wei 都是 `f4_e2m1`，MX 的 e8m0 group scale）
   和 **BF16 输出**。计数也能闭合：14592 / 304 = 48 次前向，304 正是每次前向的量化
   Linear 数；bf16 组里除 `lm_head` 外全是 `<tokens·K/32>x32:32x32`，即 Hadamard 旋转。
   shape 字段还能直接看出哪些 Linear 被量化了（本例 `in_proj_ba` 的
   `1x5120:5120x96` 在 f4 组，`lm_head` 的 `1x5120:5120x248320` 在 bf16 组）。

   **注意两点**：（a）若 SYCL queue 没开 profiling，verbose 会打印
   `execution times will not be reported`，时间字段全是 0，耗时仍要看 unitrace；
   （b）**verbose 只证明数据格式，不证明 MAC 速率** —— 本例 f4 组和 bf16 组的 impl 名
   完全相同（都是通用的 `jit:gemm:any`），且全程没有 `attr-fpmath:`（默认 `strict`）。

2. **字节模型自洽性**：用日志里的 `Actual usage is X GiB for consumed memory` 反推权重
   到底是以几位存的。本 trace 18.67 GiB 与 MXFP4（0.53125 B/元素）吻合，与 bf16（会是
   ~48 GiB）完全不符 —— 证明权重确实是 4-bit 存储，解压发生在 kernel 内部。
   另一个反证：若按 bf16 计字节，`gate_up` 会算出 1221 GB/s，超过设备实测上限 1035 GB/s，
   物理上不可能。
3. **算力倒推**：拿同设备另一条 trace 的 MXFP8 CUTLASS 值当锚。若 fp8 = 500 TFLOPS，
   则 bf16 ≈ 250、原生 fp4 ≈ 1000。实测 116 TFLOPS 连 bf16 推算值的一半都不到，
   与"解压成 BF16 再算"一致，与"原生 FP4 MAC"矛盾。
4. **同 shape 微基准（唯一能拍死 MAC 速率的办法）**：
   [bench_mxfp4_vs_bf16.py](../scripts/bench_mxfp4_vs_bf16.py) 对同一组 M/K/N 分别跑
   `torch.ops._xpu_C.fp4_gemm` 和 bf16 `torch.matmul`。必须跑在**空闲卡**上
   （`ZE_AFFINITY_MASK=<n>`），否则数字不准。

   本模型实测结果（`Intel(R) Graphics [0x674f]`，峰值 HBM 1.2 TB/s，
   XMX BF16 299.5 / FP8 599 / FP4 1198 TFLOPS）：

   | 路径 | 硬件峰值 | 实测 | 效率 |
   |---|---:|---:|---:|
   | BF16 GEMM（oneDNN `jit:gemm`） | 299.5 TFLOPS | 236\u2013265 | **79\u201388 %** |
   | MXFP8 GEMM（CUTLASS-SYCL） | 599 TFLOPS | 500 | **83 %** |
   | **MXFP4 GEMM（oneDNN w4a4）** | **1198 TFLOPS** | **109\u2013122** | **9\u201310 %** |
   | BF16 GEMV（decode） | 1200 GB/s | 950\u20131045 | **79\u201387 %** |
   | **MXFP4 GEMV（decode）** | 1200 GB/s | **229\u2013327** | **19\u201327 %** |

   **MXFP4 比 BF16 还慢一倍多**（0.41\u20130.49×）—— 解压 4-bit nibble + 应用 E8M0 scale
   的开销超过了 MAC 本身：按实测 BF16 速率做完这 160.72 TFLOP 的 MAC 只需 630 ms，
   实测 1382 ms，多出的 752 ms 全是非 MAC 开销。
   微基准的 TFLOPS 与完整推理 trace 的差异 <1.7 %，反过来也验证了 trace 侧的
   GEMM 归属是对的。

   **这张表是判断"软件问题还是硬件上限"的标准做法**：同一块卡上有两条路径跑到了
   各自峰值的 83\u201388 %，只有一条停在 9\u201310 %，那就一定是软件问题。
   只有厂商峰值、没有同机实测参照时，不要下"已达硬件上限"的结论。
