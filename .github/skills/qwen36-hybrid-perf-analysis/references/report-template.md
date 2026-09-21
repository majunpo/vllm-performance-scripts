# perf-report.md 结构模板（GDN + Full-Attention 混合模型）

放在 trace 同目录。所有数字直接取自脚本输出，不要手算或估算。

````markdown
# <模型>-<量化> 性能分析报告（<设备> / bs<N> / in<T>-out<O>）

**Trace**：`<文件名>`（<大小>，<kernel 事件数>）
**目录**：`<目录名>`   **日期**：YYYY-MM-DD

## 1. 配置
### 1.1 硬件与软件
<表格：设备 / 显存 / vLLM 版本 / 图执行模式 / chunked prefill / 负载>
### 1.2 模型拓扑
<表格：层数拆分（GDN / full，interval）/ hidden / intermediate / vocab>
<表格：full attention 的 q/kv 头数、head_dim、是否 output-gated、rope 形式>
<表格：GDN 的 k/v 头数与维度、conv kernel、state dtype>
### 1.3 量化
<权重/激活的位宽、group size、字节/元素；是否有在线 Hadamard 旋转；lm_head 是否量化>
> 每步需读取的权重总量：<X> GiB（MXFP4 <A> GiB + bf16 lm_head <B> GiB）

## 2. 总览
<表格：prefill 时间与吞吐 / decode 每步与吞吐 / GEMM 占比 / 实测带宽 / 实测 TFLOPS>
### 2.1 计量说明
<列出本 trace 命中的陷阱，见 references/pitfalls.md；必须写明图重放剔除了多少 kernel>
### 2.2 计数自检
<贴脚本的 sanity check 表，这是全部结论可信的前提>

## 3. 分类时间统计
### 3.1 PREFILL 分类（<T> ms，<N> tokens）
### 3.2 DECODE 分类（<N> 个完整 step 的 ms/step）
### 3.3 单个 PREFILL step 精确统计（<T> ms，<N> 个 kernel）
### 3.4 单个 DECODE step 精确统计（step #k，<T> ms，<N> 个 kernel）
<表格：类别 | cnt | ms | % | kernel（可直接在 trace 中搜索，**完整名字 + ND-range**）>
<Dense-GEMM 下先按 linear 名展开（用 ├/└），再按 kernel 签名展开>
<数据源只能是 kernel-detail-*.txt（dump_kernels.py），不能用被截断的分析输出>
### 3.5 分类说明（融合情况 / 与上一版的算子改名对照）
<表格：kernel | 融合了哪些 op | 本报告分类>
<若存在同模型的旧报告，再加一张“旧 kernel 名 → 新 kernel 名”对照表>
<并给出每次前向的**恒定 kernel 计数**，供下一条 trace 直接校验>

## 4. GDN 层 vs Full-Attention 层
### 4.1 每层耗时对照
<表格：层类型 | 层数 | kernel 数 | ms | % | ms/层>
### 4.2 去掉共享 MLP 之后的 "attention core" 对照
<表格：GDN core 的各 op 与 full attention core 的各 op，逐项 µs>
### 4.3 GDN 的收支平衡点
<FMHA 随 L 线性增长、GDN 恒定，但 GDN 的 in_proj 更大；给出 break-even 的 L>

## 5. Dense-GEMM 逐 shape 分析
<说明归属方法（work-group 形状 + 层内顺序）和字节/FLOPs 公式>
### 5.1 PREFILL GEMM（M = <T>，compute-bound）
<表格：linear | M | K | N | calls | med_us | GFLOP | MB | TFLOPS | ms/step>
### 5.2 DECODE GEMM（M = <B>，memory-bound）
<表格：linear | K | N | calls | med_us | min_us | MB_w | GB/s | GB/s@min | ms/step>
### 5.3 与同设备参考值的对照
<lm_head 作为本 trace 自带的带宽基准；另一条 trace 的最佳值作为外部基准>
### 5.4 在线 Hadamard 旋转与量化开销

## 6. Attention 分析
### 6.1 Full attention（FMHA）
<表格：kernel | calls | us/call | ctx/T | GFLOP/call | MB/call | TFLOPS | GB/s>
<split-K reduce 的额外开销单独列>
### 6.2 GDN prefill（chunk 路径）
<表格：kernel | calls | ms | us/call | % | 说明>
### 6.3 GDN decode（recurrent 路径）
<state 字节数与实测带宽>

## 7. 主要问题与 gap 定位
<表格：问题 | 证据 | 影响（ms 与 %）>
<每个问题一个小节，给出量化拆解和根因判断>

## 8. 优化方向（按收益排序）
### 8.1 DECODE
<表格：# | 优化项 | 依据 | 预期收益（ms/step 与 %） | 难度>
### 8.2 PREFILL
<同上>
### 8.3 汇总
<"若完成 #1~#k，decode 从 A ms/step 降到 B ms/step，吞吐 X → Y tok/s">

## 9. 复现方式
<命令 + 产物清单 + 注意事项>
````

## 纯 prefill（out=1）trace 的变体

只有一次前向，没有 decode，所以 §3.2 / §3.4 / §5.2 / §6.3 去掉，并补上两节：

````markdown
### 2.1 与 out=20 trace 的交叉验证（方法学验证）
<表格：逐类别对比 out=1 与 out>1 报告的 prefill 窗口，给出差异百分比>
<out=1 时间戳不塔缩，可以反过来证明 out>1 报告里“剔除图重放 + sum-of-durations”的处理正确>

## 7. 调度与空隙分析（这条 trace 独有）
<wall span / busy 百分比 / idle 总量 / 间隙数与中位数>
### 7.1 空隙归属（按“间隙之后的那个 kernel”分类）
### 7.2 最大的 10 个间隙
<结论必须回答：瓶颈是 kernel 本身还是 launch/调度？计算主干（GEMM/GDN/FMHA/量化）
 的空隙总和是多少？>
````

## 写作要求

- **类别名沿用脚本输出**，不要自创。
- **精确单步表（§3.3 / §3.4）是整份报告最有价值的部分，不允许省略或用概述表代替**。
  它必须完整展开每个类别的 kernel 子项，保留 count、ms、%、**可搜索的完整 kernel 名
  （带 ND-range）**。ND-range 是这张表最大的价值：它是唯一能把 kernel 反查回具体 shape
  的线索。数据源只能是 `dump_kernels.py` 的 `kernel-detail-*.txt`，
  `analyze_hybrid_trace.py` 的名字被截断过，不能用。
- **看不出用途的 kernel 要加一句中文注解**（`UT-transform 求逆`、
  `K=17408（down_proj 的输入）`、`单 work-group`），否则表只对写报告的人有用。
- **长尾用一行「其余 N 个」收拢，不能默默丢掉**，count 和 ms 必须仍然加得回类别总数。
- **GDN 与 full attention 必须分开报**，这是混合模型报告的核心，不能只给一个
  "Attention" 总数。
- **每个优化建议都要带量化收益**，用实测数据推导（例如"MXFP4 linear 从 281.7 GB/s 拉到
  同设备实测的 1035 GB/s，45.91 → 12.44 ms，每步省 33.5 ms"）。
- **区分软件问题和硬件规格差异**。判断"是否已到硬件上限"要用**同一台设备上另一条
  trace 的最佳实测值**作参照，不要用厂商标称峰值。
- **交叉验证要写进报告**：sanity check 全 OK、prefill 各 shape 的 TFLOPS 落在同一窄带、
  单步 TOTAL 与分类求和一致 —— 这些是结论可信的依据。
- 聚合统计只用**众数尺寸的完整 decode 窗口**，并写明跳过了哪些窗口。
- **总览里 kernel 时间和 wall 时间都要给**。sum-of-durations 是 kernel 时间，reflow 后的
  slice 跨度才是 wall 时间，两者之差是设备空隙的**下界**，必须单列成一条问题。
