# Kernel Benchmark Scripts

单算子（kernel 级）性能测试脚本，用于和 Intel XPU 侧
`vllm-xpu-kernels/benchmark/` 中的同名脚本做对照。

与 `profile-scripts/` 的区别：这里**不跑模型**，直接调用算子，输出确定性的
latency / 带宽数字，便于定位某个算子在两个平台上的差距。

## 目录

| 目录 | 平台 | 说明 |
| --- | --- | --- |
| [`nv/`](nv/) | NVIDIA CUDA | vLLM 自带的 CUDA / Triton 算子 |

XPU 侧不在本仓库，直接用 `vllm-xpu-kernels` repo 里的 `benchmark/` 目录。

## nv/benchmark_causal_conv1d.py

GDN / Mamba 线性注意力里的 depthwise causal conv1d：

- prefill（varlen）→ `causal_conv1d_fn` → `_causal_conv1d_fwd_kernel`
- decode（每序列 1 token）→ `causal_conv1d_update` → `_causal_conv1d_update_kernel`

```bash
# 默认 Qwen3-Next / Qwen3.x GDN 形状（conv_dim=8192, width=4），bf16
python kernel-bench-scripts/nv/benchmark_causal_conv1d.py

# 只跑 decode，并导出 CSV
python kernel-bench-scripts/nv/benchmark_causal_conv1d.py \
	--filter Decode --csv conv1d-nv.csv

# 换模型形状 / TP
python kernel-bench-scripts/nv/benchmark_causal_conv1d.py \
	--num-k-heads 16 --num-v-heads 32 --head-k-dim 128 --head-v-dim 128 \
	--tp-size 2
```

Workload 列表与 XPU 侧脚本保持一致：

| 类别 | 示例 | 说明 |
| --- | --- | --- |
| 单序列 prefill | `Prefill [2048]` … `[65536]` | 延迟 vs. 总 token 数 |
| 非均匀 prefill | `Prefill [6144,2048]`、`[4096,4096]` | 总量都是 8192，验证 varlen 切分是否影响性能 |
| 均匀多序列 prefill | `Prefill [4096]x8` | batch 维扩展 |
| decode | `Decode B=1, T=1` … `B=512` | 每序列 1 个新 token |

计时口径：`--warmup`（默认 5）次预热后，用 CUDA Event 计时 `--iters`（默认 25）
次迭代取均值，与 XPU 脚本的 `iterations=30`、前 5 次 warmup 完全对应。

### Shape 记法怎么读

方括号里是**这一次 kernel 调用中、每条序列各自的 token 数**，就是一个 list。

conv1d 是 varlen（continuous batching）算子：一个 batch 里多条序列的 token 被
首尾拼接成一个扁平张量，靠 `query_start_loc`（cu_seqlens，累积和）划分边界。
所以 shape 记法描述的就是这个"怎么拼"的信息。

以 `Prefill [6144,2048]` 为例（默认 `conv_dim=8192`）：

| 项 | 值 |
| --- | --- |
| batch | 2 条序列 |
| 每条长度 | 第 0 条 6144 token，第 1 条 2048 token |
| 总 token | 8192 |
| `x` shape | `(conv_dim, 8192)` = `(8192, 8192)` |
| `query_start_loc` | `[0, 6144, 8192]` |

三种记法的关系：

| 记法 | `seqlens` | batch | 总 token |
| --- | --- | --- | --- |
| `Prefill [8192]` | `(8192,)` | 1 | 8192 |
| `Prefill [6144,2048]` | `(6144, 2048)` | 2 | 8192 |
| `Prefill [1024]x8` | `(1024,)*8` | 8 | 8192 |

三者**总 token 数完全相同、`x` 张量形状也完全相同**，唯一的区别是
`query_start_loc` 不同 —— 也就是 kernel 看到的序列边界不同。

`Decode B=32, T=1` 则是 `seqlens = (1,)*32`：32 条序列、每条只推进 1 个 token，
走 `causal_conv1d_update` 的状态递推路径，不经过 varlen 拼接。

### 为什么要专门测非均匀 prefill

这组是**控制变量实验**：固定总 token = 8192，只改变序列长度分布
（`6144+2048` / `4096+4096` / `2048+6144` / `1024+7168`），用来回答一个问题：

> conv1d 的耗时只取决于总 token 数，还是也受序列切分方式影响？

两种可能的 kernel 实现会给出完全不同的答案：

- 按**固定 chunk**（如 `BLOCK_M=8`）切 token 分配 workgroup
	→ 切法不影响性能，四组数字应该一样
- 按**每条序列一个 workgroup** 分配
	→ `[1024,7168]` 这种长短悬殊的会因负载不均衡而明显变慢

NV 平台实测（bf16，默认 GDN 形状）：

| Shape | latency (µs) |
| --- | --- |
| `Prefill [8192]` | 114.973 |
| `Prefill [6144,2048]` | 116.346 |
| `Prefill [4096,4096]` | 116.737 |
| `Prefill [2048,6144]` | 116.004 |
| `Prefill [1024,7168]` | 116.293 |

四组基本一致（差异 < 1 %），和单序列 `[8192]` 也几乎相同，说明 vLLM 的 Triton
conv1d 是 chunk 级负载均衡，**对 varlen 切分不敏感**。

这个结论在真实服务里有意义：线上请求长度参差不齐，该实验证明不会因为长短
请求混排而掉性能，可以放心做 continuous batching。

### 新手上手建议

1. 先只跑一小部分确认环境没问题：
	 `--filter "Prefill [2048]"` 或 `--filter Decode`
2. 看懂三列指标的含义：
	 - `latency(us)`：算子单次调用耗时，最直接的对比口径
	 - `GB/s`：按理论最小访存量估算的带宽，conv1d 是访存瓶颈算子，这个值
		 越接近硬件峰值说明 kernel 写得越好
	 - `TFLOPS`：depthwise conv 计算量极小，这个数字必然很低，**不要**
		 拿它和 GEMM 的 TFLOPS 比
3. 先扫单序列 prefill（`[2048]`→`[65536]`）看延迟是否随 token 数线性增长，
	 再看 decode 随 batch 的变化，最后才看非均匀 prefill 这类对照实验

### 对比时的注意事项

两边的算子**边界不一样**，不能直接拿两个数字相除：

- XPU 的 `torch.ops._xpu_C.causal_conv1d` 是融合算子：拆
	`projected_states_qkvz/ba` + conv + SiLU + 更新 `conv_state` +
	写出 `{q,k,v,b,a}` 和 `z`。
- CUDA 的 `causal_conv1d_fn` 只做 conv + 激活，输入已经是切好的
	`x: (dim, tokens)`；拆分部分在 NV 上是独立的 `_fused_post_conv*` kernel。

做 XPU vs NV 对比时，应先从真实 trace 里把 `_causal_conv1d_fwd_kernel` 和
`_fused_post_conv*` 的时间合并，再与 XPU 的融合算子对照。
