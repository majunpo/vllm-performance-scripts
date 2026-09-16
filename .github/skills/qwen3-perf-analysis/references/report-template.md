# perf-report.md 结构模板

放在 trace 同目录。所有数字直接取自脚本输出，不要手算或估算。

## 单平台报告

````markdown
# <模型>-<量化> 性能分析报告（<设备> / bs<N>）

**Trace**：`<文件名>`（<大小>，<kernel 事件数>）
**目录**：`<目录名>`
**日期**：YYYY-MM-DD

## 1. 配置
<表格：设备 / 模型维度 / attention 头数 / vocab / 权重量化 / 激活量化 /
       GEMM 后端 / Attention 实现 / 图执行模式 / 负载>
> 权重总量：按 <w> B/元素计，每次前向需读取约 <X> GiB。

## 2. 总览
<表格：prefill 时间与吞吐 / decode 每步与吞吐 / GPU 利用率 / step 数 / bubble>

### 计量说明
<列出本 trace 命中的陷阱，见 references/pitfalls.md>

## 3. 分类时间统计
### 3.1 PREFILL（<T> ms，<N> tokens；sampler-window aggregate）
<表格：类别 | cnt | total_ms | %>
### 3.2 DECODE（<T> ms / <N> complete steps；sampler-window aggregate）
<表格：类别 | cnt | total_ms | % | ms/step | cnt/step>
### 3.3 单个 prefill step 精确统计（标题写 TOTAL ms 和 kernel 总数）
### 3.4 单个 decode step 精确统计（step #3，标题写 TOTAL ms 和 kernel 总数）
<表格：类别 | cnt | ms | % | kernel（可直接在 trace 中搜索）>
<每个类别下用 ├ └ 展开到具体 kernel>
### 3.5 分类说明
<说明融合情况：哪些 op 被融进了同一个 kernel>

## 4. Dense-GEMM 逐 shape 分析
<说明归属方法和字节/FLOPs 公式>
### 4.1 PREFILL GEMM（M = <T>，compute-bound）
<表格：linear | M | K | N | calls | med_us | GFLOP | MB | TFLOPS | GB/s>
### 4.2 DECODE GEMM（M = <B>，memory-bound）
<同上 + p10_us>
**每个 decode step**：GEMM <T> ms，读取 <X> GiB → **<BW> GB/s**，占整步 <P>%
### 4.3 使用的 kernel
<表格：phase | kernel 模板 | 说明>

## 5. Attention 分析
<公式 + 表格：kernel | calls | med_us | ctx/T | GFLOP/call | MB/call | TFLOPS | GB/s>

## 6. 结论
1. 健康度判断（利用率、是否接近硬件上限）
2. 主要时间占比
3. 可优化点（按收益排序，每条带量化收益估算）

## 7. 与 <另一平台> 的对比
<指向独立对比文档的链接 + 一句话结论>

## 8. 复现方式
<命令 + 产物清单 + 注意事项>
````

## 跨平台对比报告

单独成文，放在两个平台目录的共同父级。

````markdown
# <模型> 推理性能对比：<平台A> vs <平台B>

## 0. 对比基准
<表格：两边的设备 / trace / 目录 / 模型 / 量化 / GEMM 后端 / 图执行 / 负载>
**可比性**：<说明哪些条件相同、哪些不同>
**数据来源**：两边都取"单步精确统计"窗口（以 sampler kernel 为边界），计数为整数。

## 1. 总体
<表格：指标 | A | B | 加速比>

## 2. PREFILL 单步分类对照
| 类别 | A cnt | A ms | A % | A kernel | B cnt | B ms | B % | B kernel | 加速 |
<行按 A 耗时降序；类别名沿用脚本口径，不要自创>
### 2.1 prefill GEMM 逐 shape（TFLOPS）
### 2.2 prefill attention

## 3. DECODE 单步分类对照
<同上结构>
### 3.1 decode GEMM 逐 shape（GB/s）
### 3.2 decode attention

## 4. 主要 gap 定位
<表格：差值来源 | ms | 占总差距 %>
<每个 gap 单独小节，给出量化拆解>

## 5. <落后方> 优化优先级
<表格：# | 优化项 | 预期收益 | 难度>
<给出"若完成 X+Y，差距从 A× 收窄到 B×"的估算>

## 6. 复现方式 + 计量注意事项
````

## 写作要求

- **类别名沿用脚本输出**，不要自创或套用别处的分类。
- **精确单步表必须完整展开脚本输出的 kernel 子项**，不能只写“代表 kernel”。同名
  kernel 可以聚合成一行，但必须保留 count、ms、%、可搜索名称和 ND-range。
- **聚合 phase split 必须使用 sampler-delimited complete windows**。若只能回退到
  KV-write marker，必须标注 layer-0 边界近似，不得把聚合计数当作精确计数。
- **对比表的行按其中一方耗时降序**，两张表保持同样的排序基准。
- **kernel 列填完整可搜索的名字**（XPU 带 ND-range，NV 带模板参数），这是这张表最大的价值。
- **加速比 = 慢方 ÷ 快方**，>1 表示后者快。<1 的行要显式标出是哪一方领先。
- **每个优化建议都要带量化收益**，用实测数据推导（例如"gate_up 拉到自身 down_proj 的
  1034 GB/s，单层 372.3 → 261.5 µs，每步省 7.09 ms"）。
- **交叉验证要写进报告**：例如四个 shape 的 TFLOPS 落在同一窄带、单步 TOTAL 与 step
  周期吻合，这些是结论可信的依据。
- 结论里区分**软件问题**和**硬件规格差异**，不要把 SM 数量差异说成软件 bug。
