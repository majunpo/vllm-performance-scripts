---
name: prefill-kernel-flow
description: 'Turn a single-forward (out=1) unitrace of an LLM prefill into a beginner-readable prefill-flow.md that walks the GPU kernels in execution order and explains what each one does. Use when asked to: 讲清楚一次 prefill 是怎么执行的、按执行顺序列出 kernel、每个 kernel 的作用是什么、画一张 prefill 的 flow 图、把重复的层折叠成"同上"、解释某个 kernel 属于哪一层哪一步、新手想看懂一次前向的全过程、把 ND-range 反查成张量宽度、搞清楚量化/反量化和 attention 计算分别发生在哪个 kernel。Complements qwen36-hybrid-perf-analysis (which answers "何处最慢"); this one answers "到底在干什么".'
argument-hint: '<path to an out=1 unitrace python.<pid>.json> [--weight-dtype mxfp8]'
---

# 一次 PREFILL 的 kernel 执行流程文档

把一条 `output=1` 的 unitrace 变成一份 `prefill-flow.md`：**按执行顺序**走完一次前向，
每个 kernel 说清楚"它在干什么、处理多宽的张量、属于哪一层的哪一步"，
重复的层折叠成"同上"。目标读者是**第一次看这个模型 trace 的人**。

## 与 qwen36-hybrid-perf-analysis 的分工

| | 这个 skill | [qwen36-hybrid-perf-analysis](../qwen36-hybrid-perf-analysis/SKILL.md) |
|---|---|---|
| 回答 | **在干什么**（顺序、语义、数据流） | **哪里慢**（分类耗时、TFLOPS、GB/s、瓶颈） |
| 保留 | 执行顺序，丢掉重复 | 聚合总量，丢掉顺序 |
| 输出 | `prefill-flow.md` | `perf-report.md` |

两者共用 `hybrid_common.py` 的分类与层切分，**类别名必须一致**，这样同一个 kernel
在两份文档里叫同一个名字。本 skill 的脚本直接 import 它，不要复制一份。

## 必须用 `out=1` 的 trace

`out>1` 的 trace 里 XPU-Graph 会把整张图的 kernel 盖上同一个提交时间戳，
**重放内部的 capture order 不等于执行顺序**，按它写出来的"流程"是错的。
`out=1` 只有一次前向、没有重放、时间戳 0 % 塌缩，顺序才是可信的。

脚本用"恰好一个 embedding-norm kernel"来校验这一点，不满足就直接退出。

## Step 1 — 生成原始流水

```bash
S=<repo>/.github/skills/prefill-kernel-flow/scripts
D=<trace dir>                       # 放着 out=1 的 python.<pid>.json
T=$(ls $D/python.*.json)

pip install ijson
python $S/dump_prefill_flow.py $T --weight-dtype mxfp8 --max-name 40 \
       > $D/prefill-kernel-flow.txt
```

输出四段：

| 段 | 内容 |
|---|---|
| `== 结构 ==` | prologue / 各类型层数 × 每层 kernel 数 / head 的时间占比 |
| `== 折叠依据 ==` | **每层的 kernel 序列签名分组**，证明"同上"是成立的 |
| `A. PROLOGUE` | 调度与准备（连续相同的 kernel 折成 `xN`） |
| `B./C.` 每种层 | 代表层的逐 kernel 流水 + 作用 + 张量宽度 |
| `D. HEAD + SAMPLER` | lm_head 与采样 |

## Step 2 — 先看折叠依据，再写文档

**不要凭印象说"其余层同上"。** 脚本把每层的 kernel 序列压成
`(category, kernel base name, ND-range)` 的元组做分组，只有签名逐字节相同才算同一种。
典型输出：

```
  gdn: 2 种不同签名
    x 47 层   23 kernels  层号 [1, 2, 4, 5]...
    x  1 层   23 kernels  层号 [0]        <- 第 0 层不一样，必须单独写
  full: 1 种不同签名
    x 16 层   21 kernels  层号 [3, 7, 11, 15]...
```

- 出现多个签名组 → 文档里**每组都要展开**，并说明差在哪（第 0 层通常是
  embedding norm 代替了输入 norm、state 初始化被提前）。
- 只有一组 → 可以放心写"其余 N 层同上"。

## Step 3 — ND-range 怎么反查成张量宽度

这是文档里对新手最有价值的一列，脚本已经自动算好了，但要知道它的依据，
因为**换个模型就要更新 `known_widths()`**：

| kernel 类型 | ND-range 的含义 | 反查方式 |
|---|---|---|
| GEMM（`gemm_kernel` / `GemmUniversal`） | grid 是 tile 或 N/16，**不是**每 token | 直接用 walker 的 linear 名 + config 的 `derive()` 给出 `M/K/N` |
| `triton_red_*` / `triton_per_*`（带 reduction） | `grid[0]` = reduction 行数，`local[0]` = 组内线程 | 报"N 行 = k 行/token"，**不猜宽度** |
| pointwise（`triton_poi_*`、`at::native` elementwise） | `grid[0]*local[0]` = work-item 数，每个处理 `v` 个元素 | `v` 在 `vectorized_elementwise_kernel<v,...>` 里是显式的；Triton 的取 2 的幂里最小能命中已知宽度的那个 |
| `gdn::` / `cutlass::` / memcpy | 按 chunk / head / tile 切分 | 不反查，靠 kernel 名本身说明 |

**宽度表必须按层类型分开**（`known_widths(cfg, section)`）。
同一个数字在两种层里意思完全不同 —— 本模型的 `6144` 既是 full attention 的
`Hq·D`（q / attention 输出 / gate），又是 GDN 的 `Vh·Dv`（v / z 门 / attention 输出）。
用一张全局表会把一半 kernel 标错。

## Step 4 — 写 `prefill-flow.md`

放在 trace 同目录，跟着 [references/flow-doc-template.md](./references/flow-doc-template.md)。
**结构要点**：

1. **先给全局图**（Mermaid）：prologue → 64 层（`GGGF` × 16）→ head，
   标出每段的耗时占比。让读者 10 秒内知道时间花在哪。
2. **再给"一层里在干什么"的分类表**：把层内 kernel 归成
   *输入归一化 → 量化 → 投影(GEMM) → token 混合(attention/GDN) → 输出投影 → MLP*
   这几档，说明每一档的作用与共性，再进入逐 kernel 表。
3. **逐 kernel 表**：`# | ms | 类别 | 作用 | ND-range | 张量宽度`。
   ND-range 原样保留 —— 它是唯一能把 kernel 反查回 shape 的线索。
4. **两种层并排对照**：GDN 层与 full 层共享同一个 MLP，差别只在 token-mixing。
   明确写出"共享部分"和"独有部分"，这是混合模型最容易讲清楚的切入点。
5. **数据流的精度标注**：哪一步是 BF16、哪一步是 FP8、反量化在哪里发生。
   新手最常见的误解就是"激活一直是 FP8"，必须用 kernel 模板参数正面回答（见下）。

## 必须在文档里讲清楚的几个点

这些是实践中反复被问、且**只能从 kernel 模板参数里读出来**的：

| 问题 | 证据在哪 |
|---|---|
| 激活到底是 FP8 还是 BF16？ | 量化 kernel 的模板 `<c10::BFloat16, c10::Float8_e4m3fn, …>` = 输入 BF16；GEMM epilogue 的 `XeLinCombPerColBiasEltAct<…, cutlass::bfloat16_t, …>` = **输出写回 BF16**。FP8 只活在"量化输出 → GEMM 输入"这一小段，**没有独立的反量化 kernel**。 |
| 量化要做几次？ | `input_activations.dynamic=true` → 每个 linear 之前都要重做。计数 `4G+4F`（`in_proj_qkvz`/`in_proj_ba` 读同一份 hidden，共用一次）。 |
| attention 用什么精度？ | FMHA 的 MMA atom `XE_DPAS_TT<8, float, cutlass::bfloat16_t, cutlass::bfloat16_t, float>` = BF16 输入 / fp32 累加。 |
| KV cache 转不转 FP8？ | `reshape_and_cache_flash_strided_kernel<bf16, bf16, (vllm::Fp8KVCacheDataType)0>`，第三个模板参数 0 = 不转。 |
| 哪些 linear 没被量化？ | `config.json` 的 `quantization_config.ignore`。本模型里 `in_proj_a`/`in_proj_b`（48 层）和 `lm_head` 在列表里 → **`in_proj_ba` 是 BF16**，这也解释了它为什么排在量化 kernel *之前*。 |
| 为什么 prefill 里还有一个 decode 的 attention kernel？ | unified attention 后端两条分支都会 launch；没有 decode 序列时 `XeFMHAFwdSplitKVKernel` 的 ND-range 退化成 `{1;1;4}`、1.8 µs，是空跑。 |

## 常见坑

| 坑 | 后果 |
|---|---|
| 用 `out>1` 的 trace 写顺序 | 图重放内部顺序不是执行顺序，流程图是错的 |
| 按 `(ts, dur, name)` 排序 | 破坏 capture order；只能按 `ts` 稳定排序（`hybrid_common.load` 已处理） |
| 把 prologue 当成第 0 层 | 前 ~56 个 kernel 是 KV block 清零 / H2D 元数据搬运，与模型无关；真正的起点是唯一那个 `_add_embedding_rms_norm_` |
| 认为第 0 层和其它层一样 | 第 0 层的输入 norm 融进了 embedding 查表，且 GDN state 初始化被提前，签名不同 |
| 用全局宽度表反查 ND-range | full 的 `Hq·D` 和 GDN 的 `Vh·Dv` 都是 6144，会标错一半 |
| 把 GEMM 的 grid 当成 per-token | oneDNN 是 `N/16`，CUTLASS 是 tile 数；要从 config 取 `M/K/N` |
| 漏掉 `xN` 折叠后的计数 | prologue 折叠后每行要带 `xN`，总数必须还能加回去 |

## 换一个模型要改什么

1. `hybrid_common.py` 的 `CFG`（维度）与 marker 正则 —— 与 qwen36 skill 共用，改一处即可。
2. 本 skill 的 `known_widths()`：按层类型列出该模型所有可能的 per-token 宽度。
3. 本 skill 的 `NOTES` / `SECTION_NOTES`：kernel 名 → 中文作用。torch.compile 会给
   融合 kernel 重新编号，`triton_poi_fused_7` 这种名字**每次编译都可能变**，
   所以写文档时要以 ND-range 和位置为准，名字只作检索用。
4. 非混合模型（纯 dense）没有 GDN 段，`B.` 只会有一种层，签名分组会自动退化。
