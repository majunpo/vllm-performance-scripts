# 计量陷阱与分类口径

每一条都是实际踩过的坑，都会产生"看起来合理但错误"的结果。

## 1. XPU-Graph 把 kernel 起始时间压到同一戳

XPU-Graph 把整张捕获图作为一个 Level-Zero command list 提交，unitrace 给图内**每个
kernel 都打上图的提交时间戳**。实测 88.4% 的 kernel 与他人共享 ts，单戳最多 1433 个。

- **duration 有效，start time 无效。**
- 因此 union busy / GPU idle / bubble 这类依赖时间线的指标全部失效，必须抑制。
- 统一用 **sum-of-durations**。`analyze_trace.py` 检测到 >20% 共享戳时会自动打印警告
  并跳过 bubble 分析。
- `make_perfetto_trace.py` 默认做 **reflow**：按采集顺序把每组 kernel 首尾相接重排，
  恢复顺序和时长正确、绝对位置误差在一次图提交内的时间线。`--no-reflow` 可关闭。

## 2. 排序必须只按 ts，且要稳定

```python
evs.sort()                      # 错：按 (ts, dur, name) 排，打乱同戳事件的采集顺序
evs.sort(key=lambda e: e[0])    # 对：稳定排序保留采集顺序
```

同戳事件被重排后，"当前处于 prefill 还是 decode"的状态机会错乱，导致 SiLU 之类的
kernel 计数变成 1158/122 而不是 1215/65。

## 3. DeepGEMM 的 split-K reduce 与 GEMM 完全重叠

```
gemm   ts=...118261.35 dur=39.71 stream=25
reduce ts=...118262.66 dur=39.07 stream=25   gap=-38.40us
```

同一个 stream，reduce 在 GEMM 启动后 1.3 µs 就开始 —— 它是**自旋等待** partial 落地的，
duration 里绝大部分是等待而非工作。

- 直接求和会让 NV decode 每步虚高 **9.76 ms**（39.6 vs 实际 29.8）。
- 正确做法：按**区间并集**记独占时间。`analyze_nv_trace.py` 的 `walk()` 用一个游标
  `cursor` 实现：`excl = max(0, end - max(ts, cursor))`。修正后 reduce 净增量仅 0.51 µs，
  单步 TOTAL 与真实 step 周期吻合到 99.7%。

## 4. NV 的 annotation 只包住 model forward

`execute_context_<n>(<tokens>)_generation_<m>(<tokens>)` 只覆盖前向，`lm_head`
（`gemvx`，1.26 ms）和 sampler 跑在窗口**之外**。

- 必须把窗口外的 kernel 归属到"最近一个结束的窗口"所属 phase。
- 否则 decode 每步漏掉约 1.3 ms（4%）。

## 5. run 起点锚在第一个 KV-write，而它位于 layer 0 中间

`find_runs()` 用 `reshape_and_cache` 定位 run 起点，但该 kernel 在 layer 0 的
qkv_proj **之后**。被裁掉的头部：

| kernel | 类别 |
|---|---|
| `triton_red_fused__to_copy_embedding_rms_norm_0` | Norm/RoPE |
| `per_token_group_quant_8bit_vec_kernel` | Quantize |
| `cutlass GemmUniversal`（qkv_proj L0） | **Dense-GEMM** |
| `..._view_1` / `..._view_2` | Norm/RoPE ×2 |

表现为 prefill 的 GEMM 少 1、Quantize 少 1、Norm/RoPE 少 3。`forward_start()` 会往前
回退到上一次 sampling 之后，并用 `4 × num_layers` 上限防止吞掉前面的 warmup。

## 6. trace 停止时仍在飞行的 kernel

未完成的 kernel 拿不到完成时间戳，unitrace / profiler 会给出荒谬的 duration
（实测 99234 s、106488 s）。每个活跃队列一个。不剔除的话单个事件就能淹没整个 run。
两边脚本都用 `--max-kernel-s`（默认 10 s）丢弃并打印警告。

## 7. 聚合表的 `cnt/step` 是小数

聚合表用 `总数 ÷ 步数`，首尾步被截断所以除不尽（如 `5048/19 = 265.7`）。
**做对比分析时一律用单步精确统计表**，其窗口以 sampler kernel 为边界，恰好一次完整
forward，计数全为整数。

## 8. 自动检测层数不可靠

`detect_num_layers()` 靠 KV-write 的周期性推断，reflow 之后会把 64 误判成 192
（周期的整数倍同样得分）。**始终显式传 `--num-layers`。**

---

# 分类口径

两套脚本使用同一组 bucket，规则顺序很重要。

| 类别 | XPU 匹配 | NV 匹配 |
|---|---|---|
| `Dense-GEMM` | `gemm_kernel`、`GemmUniversal`、`matmul` | `deep_gemm::`、`split_k_reduce`、`gemvx`、`cutlass`、`nvjet` |
| `Attention(FMHA/FA)` | `cutlass::fmha::`、`_ZTSN6compat` 前缀、`fmha` | `flash_fwd`、`flash::` |
| `KVCache-Write` | `reshape_and_cache` | 同 |
| `Quantize(fp8/fp4)` | `_quant_`、`quantize`、`Float8_e4m3/e5m2`、`float_e2m1`、`fp4` | 同 |
| `Quant-scale cast` | `e8m0`（block scale 的独立 elementwise 拷贝） | —（已融合） |
| `Norm/RoPE` | `rotary\|rope\|cat_index_select`，再 `rms_norm\|layer_norm` | 同 + 裸 `triton_(red\|poi)_fused_\d+$` |
| `Activation(SiLU)` | `act_and_mul\|silu\|gelu\|swiglu` | 同 |
| `Sampling` | `top_k_top_p\|Sampler\|argmax\|sample_kernel\|_bias_kernel` | `gumbel\|_bias_kernel\|argmax\|sample` |
| `Sched/Prep` | `slot_mapping\|block_table\|_prepare_\|_post_update\|_combine_sampled\|...` | 同 |
| `MemCopy` | `MemoryCopy\|Memcpy\|memcpy` | 同 + `memcpy32_post` |
| `Elementwise/Layout` | `Transpose\|fill\|copy_\|CopyScalarFunc\|elementwise_kernel\|IndexKernelFunctor\|scatter_gather\|reduce_kernel` | 同 |

## 必须遵守的规则顺序

1. **Attention 在 Dense-GEMM 之前** —— FlashAttention 的模板里含 `cutlass`，否则会被
   当成 GEMM（实测导致 NV decode 的 Dense-GEMM 虚高到 4608 次）。
2. **KVCache-Write 在 Quantize 之前** —— `reshape_and_cache<..., Fp8KVCacheDataType>`
   含 `Fp8`，否则被误判为量化 kernel。
3. **RoPE 在 Norm 之前** —— 融合 kernel 名里同时有 rope 和 `rms_norm` 的 op。

## 为什么 RoPE 和 Norm 必须合并

torch.compile (Inductor) 把 RoPE 和 q/k RMSNorm 融进了同一个 Triton kernel：

```
triton_per_fused_add_cat_index_select_mul_rms_norm_split_split_with_sizes_sub_unsqueeze_view_3
             └─ add, cat, index_select, mul, rms_norm, split, split_with_sizes, sub, unsqueeze, view
```

- `index_select` → 按 position_ids 取 cos/sin
- `split`/`cat` → qkv 拆分 + rotate_half
- `mul`/`sub`/`add` → `x·cos ± rotate_half(x)·sin`
- `rms_norm` → q_norm / k_norm

device 上只有一次 launch、一个 duration，**物理上无法分开**。`_view_3`(q 路) 与
`_view_4`(k 路) 的 grid 比恒为 `H_q/H_kv`，可用来验证归属。

命名前缀含义：`triton_per_` = persistent reduction，`triton_red_` = looped reduction，
`triton_poi_` = pointwise。

## GEMM 后端识别

| trace 里的名字 | 实际后端 |
|---|---|
| `deep_gemm::sm120_fp8_fp4_gemm_1d1d_impl<...>` | DeepGEMM（DeepSeek，CUDA 专属） |
| `internal::gemvx::kernel<...>` | cuBLAS GEMV（bf16 lm_head） |
| `cutlass::gemm::kernel::GemmUniversal<...MainloopIntelXeXMX16Block...>` | CUTLASS-SYCL（Intel Xe XMX） |
| `gemm_kernel[SIMD16 {...} {...}]` | **oneDNN** JIT GEMM |

判据：bf16 的 `lm_head` 走 `torch.nn.functional.linear` → PyTorch XPU → oneDNN，
它在所有 XPU trace 里都叫 `gemm_kernel`。因此**量化 Linear 也出现 `gemm_kernel` =
回退到了 oneDNN 路径**。实测 bs8 时 decode 带宽从 829 GB/s 掉到 80 GB/s，根因正是
M=8 未命中 CUTLASS 路径，而非 kernel tile 配置问题。
