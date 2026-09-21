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

## 10. prefill 里 `in_proj_qkvz` 和 `in_proj_ba` 会被 walker 并成一个桶

现象（`...-260920-092747`）：`analyze_hybrid_gemm.py` 报
`in_proj_qkvz  calls=96  med_us=2401  TFLOPS=230.6`，`in_proj_ba` 整行消失，
aggregate 变成 `187.14 TFLOP / 137.6 TFLOPS`。

判据：**96 = 48 + 48**，而且 `med_us` 恰好是两组的中点
（48 个 ≈75 µs 和 48 个 ≈4800 µs 混排，中位数 ≈2437 µs）。真实值是
`160.71 TFLOP / 118.2 TFLOPS`，**偏高整整一倍**。

修正：用 `hybrid_common.cluster_gemm_durations()` 按时长聚类（**按 ND-range 签名分组**，
因为一个 build 可能用不同形状的 kernel 服务不同 shape）。M=3300 时每个 linear 的耗时
离散度 < 4 %，304 个 prefill GEMM 精确落进 6 个簇、合计误差为 0：

| n | med ms | linear |
|---:|---:|---|
| 48 | 0.073 | `in_proj_ba` (N=96) |
| 64 | 1.677 | `gdn_out_proj`(48) + `o_proj`(16)，同 shape 6144×5120 |
| 16 | 3.954 | `qkv_proj` |
| 48 | 4.762 | `in_proj_qkvz` |
| 64 | 5.364 | `down_proj` |
| 64 | 9.540 | `gate_up` |

同 shape 的那一簇用 walker 的计数再拆。阀值是**相对的**
（`gap > max(50 µs, 8 % × 前一个值)`），所以不绑定 prompt 长度；绝对下限是为了不把
`in_proj_ba` 拆开（它的第一次调用是 40.7 µs 的 warm-up 离群点，其余 ≈73 µs）。

**只对 prefill 窗口有意义**：M=1 时各 shape 的耗时互相重叠（decode 下 48 个 `gdn_out_proj`、
16 个 `o_proj`、16 个 `qkv_proj`、48 个 `in_proj_qkvz` 会并成一个 128 的簇），脚本因此
在 decode 窗口不输出这一段。**任何 prefill GEMM 表在发布前都要跑一遍聚类对账**，
两边总和必须完全相等。

## 11. 一次 fusion 改名会把整个类别搬家

`...-260920-092747` 相对 `...-260917-005123` 发生了两处融合：

1. Hadamard 旋转 + 量化 + E8M0 scale 写出，从 3 个 kernel
   （`gemm_kernel[SIMD16 {128;1;1} {32;4;1}]` ×1168 +
   `per_token_group_quant_mxfp4_vec_kernel` ×304 +
   `CastScalarFunc<float, Float8_e8m0fnu>` ×304）融成单个
   **`ark::XpuMxfp4Hadamard::fwht_quant_per_item`** ×304。
2. GDN 的 gated-norm kernel 改名成
   `triton_per_fused__to_copy_add_**inc_ark_mxfp4_hadamard_quant**_mean_mul_pow_rsqrt_silu_view_0`
   ——它把下一个 linear 的 Hadamard/量化也融了进来，所以名字**同时**命中
   `RE_QUANT` 和 `RE_GDN_GATED_NORM`。

这两点都已经在代码里处理好了：`bucket()` 把 `RE_GDN_GATED_NORM` 排在 `RE_QUANT`
**之前**，sanity check 用 `fused_hadamard_quant()` 判断后把 `Quant-scale cast` 的
期望值改成 0。旧 build、新 build、BF16 三种 trace 现在都能全部 `OK`。

**下次再遇到类似情况的处理顺序**：

1. `grep -o '<前缀>[^"]*' trace.json | sort -u` 拿到新全名；
2. 看“少掉的计数”是不是等于“多出来的计数”——相等就是改名，不是丢了活；
3. 改 `bucket()` 里的正则顺序，**不要**在调用方后处理，否则各脚本和 sanity check 会不一致；
4. 把“旧名 → 新名”写进报告的 §3.5。

同一版本还有这些改名，对比算子时一起看：

| 旧（260917-005123） | 新（260920-092747） |
|---|---|
| `triton_poi_fused_**mm_**mul_silu_slice_**view_**2` | `triton_poi_fused_mul_silu_slice_2` |
| `triton_poi_fused_**mm_**mul_sigmoid_view_0` | `triton_poi_fused_mul_sigmoid_view_0` |
| `..._rms_norm_3`(gdn) + `..._rms_norm_**mm_view_**3`(full) | 合并成 `..._rms_norm_3`（64 次） |
| `QK-Norm/RoPE` = 70（含 6 个 cos/sin 准备） | 64 = 16×4（cos/sin 已被融合/外提） |

## 12. decode kernel 的 work-group 尺寸读出来是 `{0; 0; 0}`

图重放时 unitrace 拿不到 local size，decode 的 ND-range 只有 global grid 有效
（`[SIMD32 {4; 48; 1} {0; 0; 0}]`）。后果：

- `is_main_gemm()` 只能靠 grid 判别，不能靠 local size；
- 报告里要引用带完整 ND-range 的 kernel 名时，**优先引用 prefill 的，或者专门跑一条
  `out=1` trace**（见第 13 条），那里每个 kernel 的 local size 都是真的。

## 13. `out=1` 的纯 prefill trace 要用另一个脚本，而且它比 `out>1` 更可信

`analyze_hybrid_trace.py` 按 sampler kernel 切窗口，要求至少一个 prefill + 一个 decode，
`out=1` 只有一次前向，会直接退出：`need at least one prefill and one decode window`。
用 `analyze_prefill_only.py`。

这类 trace 值得专门采：**没有任何图重放，时间戳塌缩比例 0.0 %**，所以
wall span / busy 比例 / 逐间隙的 idle 归属全部有效 —— 这是 `out>1` trace 根本给不出的。
参考数据（`...-out1-...-260920-093215`）：wall 1730.52 ms、busy 99.5 %、idle 9.27 ms，
其中 62 % 集中在前向准备段（H2D metadata / `_zero_kv_blocks` / slot-mapping），
计算主干（GEMM + GDN + FMHA + 量化）的空隙合计只有 0.50 ms。

它同时是 `out>1` 报告的**方法学验证**：剔除误入的 911 个 decode 重放 kernel 之后，
两条 trace 的 prefill 逐类别差异都在 ±0.25 % 以内（总计 1721.25 vs 1718.27 ms，0.17 %）。
这条对比要写进报告。

## 14. prompt 超过 `max_num_batched_tokens` 时，prefill 会被切成多次前向

`step_windows()` 按 sampler kernel 切窗口，脚本假设 **window 0 = 整个 prefill**。
开了 chunked prefill 且 `prompt > max_num_batched_tokens` 时这个假设不成立：
prefill 被切成 N 个 chunk，每个 chunk 一次前向，window 0 只是第一个 chunk，
其余 chunk 会被当成 decode step 统计进去 —— prefill 偏小、decode/step 偏大，
两边同时错。

判据（脚本已内置 `prefill_window_indices()`）：**一个窗口里是否有完整的
`XeFMHAFwdKernel`**。decode 窗口只会有 split-KV 变体，所以这个判据是精确的。
实测参考（in3300 / budget 8192，单 chunk）：

| window | kernels | 有 prefill-FMHA | KV-write grid[0] |
|---:|---:|---|---|
| 0 | 2754 | **True** | 3300（和误入的 decode 重放的 1） |
| 1..19 | 1164 | False | 1 |

出现多个 True 时脚本会打印警告并说明结果不可用。处理方式：重新抓 trace 并把
`--max-num-batched-tokens` 调到 ≥ prompt 长度，或者用
`dump_kernels.py <trace> <window>` 逐 chunk 单独分析。

## 15. batch 不是 1 的时候 `--batch` 默认值会静默算错

`--batch` 曾经默认 1，喂一条 bs=4 的 trace 不会报任何错，只会让 tok/s 变成
真实值的 1/4，并让 decode FMHA / GDN state 的字节模型全部偏小。

现在 `detect_batch()` 从 **decode 窗口 KV-write 的 ND-range grid[0]** 自动读出
（decode 一步正好写 `batch` 个 token），并在 `workload:` 行回显。
**看报告前先核对这一行和目录名是否一致**，例如
`workload: prompt=3300 tokens, batch=1 (autodetected)` 对应 `...-in3300-...-bs1-...`。

同理 `detect_prompt_len()` 取的是 KV-write grid[0] 的**最大值**而不是第一个非 1 的值 ——
prefill 窗口里通常混着一次 decode 重放，bs>1 时那次重放的 KV-write grid 也 > 1，
取第一个会拿到 batch 而不是 prompt 长度。

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
| `GDN-Norm/Gate` | `rsqrt_silu(_t)?_view_0$` —— **必须排在 Quantize 前面**，新 build 把 `hadamard_quant` 融进了这个 kernel 名（见第 11 条） |
| `Quantize(mxfp4)` | `per_token_group_quant_mxfp4`、`float_e2m1`、`fp4`、`ark::XpuMxfp4Hadamard::fwht_quant_per_item` |
| `Quant-scale cast` | `Float8_e8m0`（block scale 的独立 elementwise 拷贝；新版已融合，计数为 0） |
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

5. **拿未量化版本的同负载 trace 做端到端对照（最终验证）**。孤立微基准会漏掉只有整模型
   才暴露的开销。实测 BF16 版 Qwen3.6-27B（同设备、同 in3300/out20/bs1）：

   | | MXFP4 | BF16 | |
   |---|---:|---:|---|
   | prefill GEMM | 116.3 TFLOPS | **229.8 TFLOPS（峰值 77 %）** | BF16 快 **1.98×** |
   | decode GEMM | 311.6 GB/s | **790.4 GB/s（峰值 66 %）** | |
   | decode / step | **53.64 ms** | 86.64 ms | MXFP4 快 1.61× |

   微基准预测的 2.0\u20132.4× 被整模型的 1.98× 证实。**但孤立微基准会高估整模型带宽**：
   单 shape 的 bf16 GEMV 有 950\u20131045 GB/s，整模型混合 shape 只有 790 GB/s。
   写"若达到 X GB/s 能省多少"时，基准要用**整模型实测**，不要用单 shape 峰值 ——
   本报告先前就因此把"换 BF16 后 decode 持平"算错了（实际慢 1.61×）。

   对照 trace 还会暴露只存在于某一条路径的 bug：BF16 版有个
   `triton_poi_fused_cat_4`，ND-range 在 prefill 和 decode 下完全相同
   （`{164800;1;1}`），耗时与 token 数无关，在 decode 里占 21.9 %；MXFP4 版没有。
   **不要把两条路径的差异一律归因于量化格式本身。**

## 一个 kernel 到底在处理多少数据：从 grid 反推 numel

Inductor 的 pointwise kernel（`triton_poi_*`）的 ND-range 里没有 shape，但可以反推。
bf16 下每个 work-item 处理 2 个元素，所以 **`numel = grid[0] × local[0] × 2`**。
先拿两个 shape 已知的 kernel 标定，再套到未知的那个：

| kernel | grid | 理论 numel | grid×512 |
|---|---:|---:|---:|
| prefill SwiGLU | 112200 | 3300×17408 = 57,446,400 | 57,446,400 ✓ |
| decode SwiGLU | 34 | 1×17408 = 17,408 | 17,408 ✓ |
| **`cat_4`（未知）** | **164800** | ? | **84,377,600** |

再去凑模型维度：84,377,600 ÷ 3300 tokens 不是整数（排除激活），而
`hidden × in_proj_N` = 5120 × 16480 = 84,377,600 **完全相等** ——
于是确认它搬的是 **`in_proj` 的权重**，不是激活。

结合 checkpoint 布局（BF16 版把 GDN 输入投影分存成 `in_proj_{qkv,z,b,a}`，
宽度 10240+6144+48+48 = 16480）可以断定：这是**加载期的权重拼接泄漏进了运行时图**，
每次前向每个 GDN 层重做一遍。MXFP4 路径在 `process_weights_after_loading` 里
用 `replace_parameter` 做掉了，所以没有这个 kernel。

**这个反推方法是通用的**：凡是遇到"开销与 batch/token 数无关"的 kernel，
先用 `numel = grid × local × (2 if bf16 else 1)` 反推数据量，再去和
`权重维度` / `KV cache 维度` / `state 维度` 对表，通常一次就能定位。
判据：**只要 numel 除以 token 数不是整数，它就不是激活**。

   注意两边的 Inductor kernel 名和 Linear 分解都会变，`hybrid_common.py` 的 marker 与
   `CFG` 需要各自核对：BF16 版只有 **257** 个 Dense-GEMM（把 `in_proj_ba` 融进了
   `in_proj`），且 GDN 与 full 层的 input-norm 合并成了同一个 `..._rms_norm_3`，
   layer walker 需要改用别的判据。

