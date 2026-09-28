# prefill-flow.md 结构模板

放在 trace 同目录。所有 kernel 名、ms、ND-range 直接取自
`dump_prefill_flow.py` 的输出，不要手抄或估算。

````markdown
# <模型> 一次 PREFILL 的完整执行流程（<设备> / bs<N> / in<T>）

**Trace**：`<文件名>`（<kernel 数>，<总 ms>，时间戳 0 % 塌缩）
**产物**：`prefill-kernel-flow.txt`
> 为什么必须用 out=1 的 trace：没有图重放，start time 可信，顺序才是执行顺序。

## 0. 三十秒看懂
<Mermaid 全局图：prologue -> 64 层(GGGF x16) -> head，每段标 ms 与 %>
<一句话：时间的大头在哪>

## 1. 模型的层关系
<表格：层数拆分、interval、GGGF 的排布、两种层共享什么/独有什么>
<per-token 宽度表：hidden / inter / Hq·D / Kh·Dk / Vh·Dv / conv_dim …，
 并说明每个宽度来自 config 的哪个字段 —— 后面反查 ND-range 全靠它>

## 2. 一次前向的四个阶段
### 2.1 PROLOGUE（调度与准备）
### 2.2 GDN 层 x G
### 2.3 FULL-ATTENTION 层 x F
### 2.4 HEAD + SAMPLER
<每节：先一句话说这一段在干什么，再给逐 kernel 表>
<逐 kernel 表列：# | ms | 类别 | kernel（可搜索，带 ND-range） | 作用 | 张量宽度>

## 3. 一层里的工作分类
<把层内 kernel 归档成 6 类：输入归一化 / 量化 / 投影 GEMM / token 混合 /
 输出投影 / MLP，给每类的作用、在两种层里的差别、各自 ms>
<并排对照表：GDN 层 vs full 层，标出"共享 MLP"与"独有的 token-mixing">

## 4. 数据流与精度
<一张图：BF16 -> quantize -> FP8 -> GEMM(fp32 acc) -> epilogue -> BF16>
<说明反量化没有独立 kernel；attention 与 KV cache 都是 BF16；证据是模板参数>
<哪些 linear 没被量化，为什么它们排在量化 kernel 之前>

## 5. 折叠依据（为什么可以写"同上"）
<贴脚本的签名分组输出；有多个签名组时，逐组说明差异>

## 6. 复现
<命令 + 产物清单>
````

## 写作要求

- **顺序第一**。这份文档的价值就是顺序，任何"按耗时排序"的表都不要放进来
  （那是 `perf-report.md` 的事）。
- **每个 kernel 一句话作用**，写它在数学上做了什么、消费谁的输出、产出给谁，
  不要复述 kernel 名。
- **ND-range 原样保留**，并给出反查出来的张量宽度；这两列缺一不可。
- **类别名沿用 `hybrid_common.bucket()`**，与 `perf-report.md` 保持一致。
- **重复必须折叠，但要有依据**：先贴签名分组，再写"其余 N 层同上"。
- **区分"模型的一层"和"trace 的一段"**：第 0 层的输入 norm 融进了 embedding，
  prologue 不属于任何一层，最后一个 `rms_norm_3` 是模型的 final norm。
- 面向新手：出现 GQA、SwiGLU、RoPE、delta rule、paged KV cache 这些概念时，
  用一句话解释，不要假设读者知道。
