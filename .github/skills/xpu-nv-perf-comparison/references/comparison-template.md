# xpu-vs-nv-comparison.md 结构模板

放在两个平台目录的共同父级（例如 `profile-scripts/`）。
§1–§4 由 `compare_perf.py` 自动生成，其余需人工补充。

````markdown
# <模型> 推理性能对比：<平台A> vs <平台B>

**日期**：YYYY-MM-DD

## 0. 对比基准

| | <平台A> | <平台B> |
|---|---|---|
| 设备 | <型号，核心数，架构> | |
| Trace | <文件名>（<采集工具>） | |
| 目录 | <相对路径> | |
| 模型 | <hidden / inter / layers / heads / vocab> | |
| 权重量化 | <格式，group size> | |
| GEMM 后端 | <库名 + kernel 家族> | |
| Attention | <实现> | |
| 图执行 | <XPU-Graph / CUDA Graph / eager> | |
| 负载 | in<N> / bs<N> / tp<N> / out<N> | |

**可比性**：<逐条说明哪些条件相同、哪些不同、不同项的影响>

**数据来源**：两边都取"单步精确统计"窗口（以 sampler kernel 为边界，恰好一次完整
forward），计数为整数。

## 1. 总体
<compare_perf.py 生成：prefill / decode 耗时与吞吐 + 加速比>
<补一行 GPU 利用率>

> 一句话结论：<谁赢在哪个阶段，为什么>

## 2. PREFILL 单步分类对照
<compare_perf.py 生成>

### 2.1 prefill GEMM 逐 shape（TFLOPS）
| linear | K | N | A µs | A TFLOPS | B µs | B TFLOPS | 加速 |
<来自两边的 PREFILL GEMM 表>
<一句话：各自是否落在窄带内>

### 2.2 prefill attention
| | A | B |
| kernel | | |
| µs / 层 | | |
| TFLOPS | | |
| 相对本平台 GEMM 算力 | | |

## 3. DECODE 单步分类对照
<compare_perf.py 生成>

### 3.1 decode GEMM 逐 shape（GB/s）
| linear | K | N | A µs | A GB/s | B µs | B GB/s | 加速 |
<标出离群 shape>
<一行：每步各自读多少 GiB，量化粒度差异带来多少字节差>

### 3.2 decode attention
| | A | B |
| main µs / 层 | | |
| reduce / combine µs / 层 | | |
| 有效带宽 GB/s | | |
| reduce/combine 占比 | | |

## 4. gap 分解
<compare_perf.py 生成两张表>

### Gap 1：<最大项> —— 占 decode 总差距的 X%
<拆解到具体来源，区分有效带宽差与量化粒度差>
<指出最明确的短板 shape>

### Gap 2：<次大项>
<根因：软件问题 / 硬件规格 / 实现成熟度>

### Gap 3：小算子开销
| | A | B | 比值 |
| prefill 非 GEMM 非 attention | | | |
| decode 非 GEMM 非 attention | | | |
| 其中量化相关 | <ms/step，launch 次数> | | |
<根因：融合程度差异>

### Gap 4：两边共同短板
<例如 attention 相对自身 GEMM 算力都偏低>

## 5. <落后方> 优化优先级

| # | 优化项 | 预期收益 | 难度 |
|---|---|---|---|
| 1 | <具体到 shape / kernel> | <−X ms/step（−Y%）> | 中 |

<若 1+2 完成，差距从 A× 收窄到 B×>

## 6. 其他已知问题
<例如某个 batch 下的回退路径>

## 7. 复现方式
```bash
# 单平台分析
...
# 生成对比
python <repo>/.github/skills/xpu-nv-perf-comparison/scripts/compare_perf.py \
    --xpu <...>/analyze_trace.txt --nv <...>/analysis.txt \
    --xpu-name "..." --nv-name "..." --prompt-len N --batch N \
    -o xpu-vs-nv-comparison.md
```

### 各平台详细报告
- <平台A>：[perf-report.md](<相对路径>)
- <平台B>：[perf-report.md](<相对路径>)

### 计量注意事项
<列出本次命中的陷阱及处理方式>
````

## 写作要求

- **类别名沿用分析脚本的输出**，不要自创或套用别处的分类法。
- **kernel 列必须填完整可搜索的名字**（XPU 带 ND-range，NV 带模板参数）。
  这是全表最有价值的一列 —— 它让人能回到原始 trace 定位到具体行。
- **加速比 < 1 的行要显式说明是哪一方领先**。
- **每个优化建议都要带量化收益**，用实测数据推导（见 methodology.md）。
- **交叉验证写进报告**：四个 shape 的一致性、单步 TOTAL 与周期吻合，这些是可信度依据。
- **区分软件问题与硬件规格差异**，不要把 SM 数量差异说成 bug。
- **小类别只报比值没有意义**：0.2 ms 上的 15× 是噪声，要么不提，要么连绝对值一起给。
