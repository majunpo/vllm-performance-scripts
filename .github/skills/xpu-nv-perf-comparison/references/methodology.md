# 对比方法论

## 数据来源：只用单步精确统计

聚合表（`category split: DECODE (... , N steps)`）用 `总数 ÷ 步数` 计算，首尾步被截断，
所以 `cnt/step` 是小数（如 `5048/19 = 265.7`）。**不能用于逐行对比。**

单步表（`--- single DECODE step #3 (exact counts, ...) ---`）的窗口以 sampler kernel
为边界，恰好一次完整 forward，计数全为整数。两个平台用同一套窗口定义，可逐行比较。

## 可比性前置条件

| 条件 | 要求 | 不满足时 |
|---|---|---|
| 模型 / 层数 / 头数 | 必须相同 | 不可比 |
| prompt 长度 | 必须相同 | attention 部分不可比（KV 长度不同） |
| batch | 必须相同 | 单步指标不可比 |
| TP | 必须相同 | 需换算或放弃 |
| output 长度 | 可不同 | 不影响 per-step 指标 |
| 量化 group size | 可不同 | **必须在报告里写明**，它改变每权重字节数 |
| 图执行模式 | 可不同 | 写明，决定哪些计量陷阱适用 |

量化粒度的影响可以量化：MXFP8 group 32 → 1+1/32 = 1.03125 B/元素；
FP8 group 128 → 1+1/128 = 1.0078。前者每步多读 **2.3%** 字节。这属于设计差异，
不是实现问题，但在带宽对比里要扣掉。

## 比值约定

- **加速比 = 慢方 ÷ 快方**，写成 `NV 加速` 列时即 `XPU_ms / NV_ms`。
- `> 1` = NV 更快；`< 1` = XPU 更快。**`< 1` 的行要显式指出是哪一方领先**，
  否则读者容易默认列名那一方总是更快。
- 表格行按其中一方耗时降序，两张表（prefill / decode）用同一个排序基准。

## gap 分解

```
delta        = A_total − B_total
该类别贡献    = (A_cat − B_cat) / delta × 100%
```

- 单项占比**可以超过 100%**，说明它被反向的类别抵消了。例如 prefill 里
  Dense-GEMM 给 XPU 带来 140 ms 领先（+137%），但 XPU 在 Quantize / Norm / SiLU
  上又还回去 47 ms。这是真实信息，不是计算错误。
- 结论要落在"差距的 X% 来自 Y"这种可执行的表述上。实测例：
  decode 差 14.19 ms，其中 **Dense-GEMM 占 87.3%**，Attention 只占 2.4%。

## 根因分类

对每个 gap 必须给出归因，三类：

### 软件问题（可执行）
判据：**同平台内部不一致**。
- 某个 shape 是离群点，兄弟 shape 正常 → tile / 配置问题
  （实测：XPU `gate_up` 726 GB/s，自己的 `down_proj` 1034 GB/s）
- 某个 batch 下换了 kernel 家族 → dispatch 回退
  （实测：bs8 从 CUTLASS `GemmUniversal` 退到 oneDNN `gemm_kernel`，带宽 829 → 80 GB/s）

### 硬件规格差异（不是 bug）
判据：**所有 shape 等比例下降**，且对照硬件参数能解释。
- 实测：NV prefill GEMM 361–379 TFLOPS 全线低于 XPU 的 468–525，但 PRO 5000 只有
  110 SM（约同代旗舰一半），且已达自身 FP8 峰值的 75–80%。这要明确写成规格差异。

### 实现成熟度（可改善但有上限）
判据：**全面偏低但一致**。用"相对本平台自身 GEMM 算力的百分比"衡量。
- 实测：prefill FMHA 只发挥了同平台 GEMM 算力的 35%(XPU) / 42%(NV)，是两边共性问题。

## 归一化原则

**先和自己比，再和对方比。**

| 指标 | 归一化基准 |
|---|---|
| prefill GEMM | 同平台四个 linear 的 TFLOPS 一致性（应在窄带内） |
| decode GEMM | 同平台四个 linear 的 GB/s 一致性 |
| attention | 相对同平台 GEMM 的算力/带宽百分比 |
| 整体带宽 | 相对该设备标称峰值带宽 |

跨厂商直接比原始 TFLOPS 只能说明"谁的卡更大"，比自身效率才能说明"谁的软件更好"。

## 优化收益估算

必须从实测数据推导，禁止拍脑袋。模板：

```
1. 取同平台最好的同类值作为目标（不要用对方平台的值，那可能受硬件限制）
2. 按比例换算单 kernel 耗时
3. 乘以每步调用次数
4. 折算成每步节省和百分比
5. 汇总多项后给出"差距从 A× 收窄到 B×"
```

实测例：
```
gate_up 726.4 -> 1034.2 GB/s（同平台 down_proj 的水平）
372.3 µs × 726.4/1034.2 = 261.5 µs
(372.3 - 261.5) × 64 = 7.09 ms/step
45.31 -> 38.19 ms/step (-15.7%)

量化融合：768 -> 256 次 launch/step，省 1.20 ms/step
合计 45.31 -> 37.02 ms/step，与 NV 差距 1.46× -> 1.19×
```

难度也要标注：tile 配置属"中"，需要硬件带宽支撑的属"高"，涉及精度验证的要注明。

## 交叉验证要写进报告

这些是结论可信的依据，不写读者无法判断：

- 四个 linear 的 TFLOPS / GB/s 落在同一窄带 → GEMM 归属正确
- 单步 TOTAL ≈ 聚合 ms/step（1% 内）→ 窗口划分正确
- NV 单步 TOTAL ≈ step 周期（start-to-start）→ split-K 重叠已正确处理
- prefill 与 decode 的 per-layer 类别计数相等 → 窗口未被截断

## 计量陷阱

两个平台各有坑，详见
[qwen3-perf-analysis/references/pitfalls.md](../../qwen3-perf-analysis/references/pitfalls.md)。
对比场景下最容易出错的三个：

1. **XPU-Graph 时间戳塌缩** → XPU 侧只能用 sum-of-durations，不能和 NV 的 union busy 比。
2. **DeepGEMM split-K reduce 与 GEMM 重叠** → NV 侧不做区间并集会虚高 ~30%，
   直接对比会得出"NV 更慢"的错误结论。
3. **NV annotation 不含 lm_head / sampler** → 不补进来会让 NV 每步少 1.3 ms，
   同样导致错误结论。
