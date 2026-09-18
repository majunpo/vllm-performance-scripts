# vLLM Profiling 脚本使用指南

本目录用于重复采集 vLLM 的 prefill、decode 和端到端 trace，主要面向 Intel XPU，同时保留 NVIDIA CUDA 的 torch profiler 支持。

以下命令均从仓库根目录执行。

## 1. 脚本选择

| 脚本 | 平台 | 用途 |
| --- | --- | --- |
| [`xpu/run_torch_profile.sh`](xpu/run_torch_profile.sh) | Intel XPU | 推荐的 torch profiler 入口，支持 `all`、`prefill`、`decode-only` |
| [`xpu/run_unitrace.sh`](xpu/run_unitrace.sh) | Intel XPU | unitrace 入口，适合查看 SYCL、Level Zero 和 kernel 时间线；同样支持三种 phase |
| [`xpu/run_auto_model_xpu-bs-decode-only.py`](xpu/run_auto_model_xpu-bs-decode-only.py) | Intel XPU | Python 底层入口；支持 torch profiler 和 unitrace 控制 |
| [`nv/run_torch_profile_nv.sh`](nv/run_torch_profile_nv.sh) | NVIDIA CUDA | NV torch profiler 入口，默认 CUDA graph |
| [`nv/run_auto_model_nv-bs-decode-only.py`](nv/run_auto_model_nv-bs-decode-only.py) | NVIDIA CUDA | NV Python 底层入口；不支持 unitrace |

XPU 使用建议：

- 日常算子分析、shape 和调用栈：优先使用 `xpu/run_torch_profile.sh`。
- 分析 SYCL / Level Zero / GPU kernel 时间线：使用 unitrace。
- 采集严格固定 batch 的纯 decode：使用 `--phase decode-only`。
- 第一次换模型或环境时，先运行 `--check`。

三个 shell launcher 的选项名称完全一致，公共参数和默认值见第 4 节。每个选项都可以用同名大写环境变量覆盖，例如 `MODEL`、`TP`、`BS`、`PHASE`、`PROFILE_STEPS`、`LOG_DIR`、`TRACE_ROOT`；命令行参数优先级更高。三者都支持 `--check`（只做环境自检）和 `-h` / `--help`。

### 1.1 `xpu/run_torch_profile.sh`

XPU torch profiler 入口，以 `--profiler torch` 调用 [`run_auto_model_xpu-bs-decode-only.py`](xpu/run_auto_model_xpu-bs-decode-only.py)。

```bash
[环境变量] bash profile-scripts/xpu/run_torch_profile.sh [选项]
```

| 环境变量 | 默认值 | 说明 |
| --- | --- | --- |
| `ZE_AFFINITY_MASK` | `0` | 暴露的 XPU，数量不能少于 `--tp` |
| `PYTHON_BIN` | `python` / `python3` | 未激活 venv 或默认解释器不对时必须指定 |

脚本启动 Python 前会自动导出 `NEOReadDebugKeys=1` 和 `EnableImplicitConvertionToCounterBasedEvents=0`。

输出（`--trace-root` 和 `--log-dir` 可改）：

```text
profile-scripts/xpu/torch-profile-trace/<tag>/
profile-scripts/xpu/torch-profile-log/torch-profile-log-<tag>.log
```

用法示例见 2.2、2.3 和 2.4 节。

### 1.2 `xpu/run_unitrace.sh`

XPU unitrace 入口。与 1.1 使用同一个 Python 脚本，区别是外面套了一层 unitrace。

```bash
[环境变量] bash profile-scripts/xpu/run_unitrace.sh [选项]
```

| 环境变量 | 默认值 | 说明 |
| --- | --- | --- |
| `ZE_AFFINITY_MASK` | `0` | 暴露的 XPU，数量不能少于 `--tp` |
| `UNITRACE_BIN` | `PATH` 中的 `unitrace` | 找不到时脚本直接报错退出 |
| `PYTHON_BIN` | `python` / `python3` | unitrace 用 `execvp` 启动目标程序，解释器不在 `PATH` 时必须指定 |

比 1.1 多一个选项：

| 选项 | 默认值 | 说明 |
| --- | --- | --- |
| `--profiler <torch\|xpu>` | `xpu` | 传给 Python 的 profiler 类型；unitrace 采集时保持 `xpu` |

固定使用的 unitrace 参数：`--chrome-itt-logging`、`--chrome-sycl-logging`、`--chrome-call-logging`、`--chrome-kernel-logging`、`--output-dir-path`、`--start-paused`。

输出：

```text
profile-scripts/xpu/unitrace-trace/<tag>/
profile-scripts/xpu/unitrace-log/unitrace-log-<tag>.log
```

用法示例见第 3 节。

### 1.3 `nv/run_torch_profile_nv.sh`

NVIDIA torch profiler 入口，调用 [`run_auto_model_nv-bs-decode-only.py`](nv/run_auto_model_nv-bs-decode-only.py)，不支持 unitrace，因此没有 `--profiler`。

```bash
[环境变量] bash profile-scripts/nv/run_torch_profile_nv.sh [选项]
```

| 环境变量 | 默认值 | 说明 |
| --- | --- | --- |
| `CUDA_VISIBLE_DEVICES` | `0` | 暴露的 GPU，数量不能少于 `--tp` |
| `PYTHON_BIN` | `python` / `python3` | 未激活 venv 或默认解释器不对时必须指定 |

`--enforce-eager` 在这里关闭的是 CUDA graph，对应 tag 中的 `cudagraph` / `eager`。

输出：

```text
profile-scripts/nv/torch-profile-trace/<tag>/
profile-scripts/nv/torch-profile-log/torch-profile-log-<tag>.log
```

用法示例见第 8 节。

## 2. XPU 快速开始

### 2.1 环境自检

单卡：

```bash
ZE_AFFINITY_MASK=0 \
MODEL=/models/Qwen3-32B \
bash profile-scripts/xpu/run_torch_profile.sh --check
```

4 卡：

```bash
ZE_AFFINITY_MASK=4,5,6,7 \
MODEL=/models/Qwen3-32B \
TP=4 \
bash profile-scripts/xpu/run_torch_profile.sh --check
```

unitrace 自检：

```bash
ZE_AFFINITY_MASK=0 \
MODEL=/models/Qwen3-32B \
UNITRACE_BIN=/opt/install/unitrace/bin/unitrace \
bash profile-scripts/xpu/run_unitrace.sh --check
```

若默认 Python 不正确，设置：

```bash
PYTHON_BIN=/path/to/venv/bin/python
```

### 2.2 最常用：纯 decode torch profile

下面命令先排空所有 prefill，再只采集 20 个 batch size 为 32 的 decode step：

```bash
ZE_AFFINITY_MASK=4,5,6,7 \
bash profile-scripts/xpu/run_torch_profile.sh \
  --model /models/Qwen3-32B \
  --tp 4 \
  --max-model-len 8192 \
  --max-num-batched-tokens 8192 \
  --max-num-seqs 32 \
  --input-len 2048 \
  --bs 32 \
  --phase decode-only \
  --profile-steps 20 \
  --settle-steps 2
```

默认输出位置：

```text
profile-scripts/xpu/torch-profile-trace/<tag>/
profile-scripts/xpu/torch-profile-log/torch-profile-log-<tag>.log
```

`<tag>` 形如 `<model>-<phase>-<xpugraph|eager>-in<N>-out<N>-bs<N>-tp<N>[-kv<dtype>]-<timestamp>`。

### 2.3 Prefill-only torch profile

Prefill-only 建议关闭 chunked prefill，使 prompt 尽量在一个 scheduler step 内完成：

```bash
ZE_AFFINITY_MASK=0 \
bash profile-scripts/xpu/run_torch_profile.sh \
  --model /models/Qwen3-32B \
  --tp 1 \
  --max-model-len 8192 \
  --max-num-batched-tokens 8192 \
  --max-num-seqs 32 \
  --no-chunked-prefill \
  --input-len 4096 \
  --output-len 1 \
  --bs 1 \
  --phase prefill
```

### 2.4 Prefill + decode 混合 trace

`all` 是 launcher 的默认 phase，会采集一次完整的 `llm.generate()`：

```bash
ZE_AFFINITY_MASK=0 \
bash profile-scripts/xpu/run_torch_profile.sh \
  --model /models/Qwen3-32B \
  --tp 1 \
  --max-model-len 8192 \
  --max-num-batched-tokens 8192 \
  --input-len 3500 \
  --output-len 20 \
  --bs 1 \
  --phase all
```

该 trace 同时包含 prefill 和 decode，不适合直接计算固定 batch 的 TPOT。

## 3. XPU Unitrace

### 3.1 使用 launcher 采集

`run_unitrace.sh` 默认采集 `all` phase。下面用 `--phase prefill` 采集 prefill：

```bash
ZE_AFFINITY_MASK=4,5,6,7 \
UNITRACE_BIN=/opt/install/unitrace/bin/unitrace \
bash profile-scripts/xpu/run_unitrace.sh \
  --model /models/Qwen3-32B \
  --tp 4 \
  --max-model-len 8192 \
  --max-num-batched-tokens 8192 \
  --max-num-seqs 32 \
  --no-chunked-prefill \
  --input-len 4096 \
  --output-len 1 \
  --bs 1 \
  --phase prefill \
  --profiler xpu
```

默认输出位置：

```text
profile-scripts/xpu/unitrace-trace/<tag>/
profile-scripts/xpu/unitrace-log/unitrace-log-<tag>.log
```

`<tag>` 命名规则与 torch launcher 一致，同样包含 phase 和 graph 模式。

launcher 已自动设置：

```text
NEOReadDebugKeys=1
EnableImplicitConvertionToCounterBasedEvents=0
unitrace --start-paused
```

`--start-paused` 会让 unitrace 等到 Python 调用 `start_profile()` 后才开始采集，从而跳过模型加载和 warmup。

### 3.2 纯 decode unitrace

launcher 已暴露 `--phase`、`--profile-steps` 和 `--settle-steps`，直接使用即可：

```bash
ZE_AFFINITY_MASK=4,5,6,7 \
UNITRACE_BIN=/opt/install/unitrace/bin/unitrace \
bash profile-scripts/xpu/run_unitrace.sh \
  --model /models/Qwen3-32B \
  --tp 4 \
  --max-model-len 8192 \
  --max-num-batched-tokens 8192 \
  --max-num-seqs 32 \
  --enable-chunked-prefill \
  --input-len 2048 \
  --bs 32 \
  --phase decode-only \
  --profile-steps 10 \
  --settle-steps 2 \
  --profiler xpu \
  --shutdown-timeout 120
```

若需要修改 `--unitrace-final-flush-seconds` 等 launcher 未暴露的参数，可直接包装 Python 入口：

```bash
cd profile-scripts/xpu

ZE_AFFINITY_MASK=4,5,6,7 \
NEOReadDebugKeys=1 \
EnableImplicitConvertionToCounterBasedEvents=0 \
/opt/install/unitrace/bin/unitrace \
  --chrome-itt-logging \
  --chrome-sycl-logging \
  --chrome-call-logging \
  --chrome-kernel-logging \
  --output-dir-path ./unitrace-trace/decode-only-bs32 \
  --start-paused \
  python run_auto_model_xpu-bs-decode-only.py \
    --model /models/Qwen3-32B \
    --tp 4 \
    --max-model-len 8192 \
    --max-num-batched-tokens 8192 \
    --max-num-seqs 32 \
    --enable-chunked-prefill \
    --input-len 2048 \
    --bs 32 \
    --phase decode-only \
    --profile-steps 10 \
    --settle-steps 2 \
    --profiler xpu \
    --shutdown-timeout 120 \
    --unitrace-final-flush-seconds 20
```

unitrace 文件较大，建议先用 `--profile-steps 5` 或 `10`。如果 JSON 未完整写入，提高 `--shutdown-timeout` 和 `--unitrace-final-flush-seconds`。

### 3.3 XPU graph 与 eager

XPU Python 脚本默认设置：

```text
VLLM_XPU_ENABLE_XPU_GRAPH=1
```

torch profiler 通常保持 graph 开启。若 unitrace 中 graph replay 的 kernel 出现异常 device timestamp，可关闭 graph 后对照采集（两个 XPU launcher 都支持 `--enforce-eager`，trace 目录名会带 `eager`）：

```bash
ZE_AFFINITY_MASK=0 \
bash profile-scripts/xpu/run_unitrace.sh \
  --model /models/Qwen3-32B \
  --enforce-eager
```

## 4. Phase 与关键参数

| `--phase` | 采集窗口 | 适用场景 |
| --- | --- | --- |
| `all` | 完整 `llm.generate()` | 端到端 prefill + decode |
| `prefill` | 强制只生成 1 个 token | Prefill / TTFT |
| `decode-only` | 所有请求完成 prefill 后的固定步数 | 纯 decode、固定 batch、TPOT |

Shell launcher 常用默认值：

| 参数 | 默认值 | 说明 |
| --- | ---: | --- |
| `--tp` | 1 | tensor parallel size |
| `--input-len` | 3500 | prompt token 数 |
| `--output-len` | 10 | `all` 模式生成长度 |
| `--bs` | 1 | batch size |
| `--max-model-len` | 8192 | 最大上下文长度 |
| `--max-num-batched-tokens` | 8192 | 每个 scheduler step 的 token 预算 |
| `--max-num-seqs` | 32 | 最大并发序列数，必须不小于 `bs` |
| `--kv-cache-dtype` | `auto` | 可选 `fp8`、`fp8_e4m3`、`fp8_e5m2` |
| `CHUNKED_PREFILL` | 1 | launcher 默认开启 |
| `--phase` | `all` | 三个 launcher 均支持 |
| `--profile-steps` | 20 | `decode-only` 采集步数，必须为正整数 |
| `--settle-steps` | 2 | profiler 开启前的稳定步数，非负整数 |
| `--enforce-eager` | 关闭 | 关闭 XPU / CUDA graph capture |
| `--ep` | 关闭 | MoE 模型的 expert parallel |
| `--shutdown-timeout` | 120 秒 | 等待 worker 退出和 trace 落盘 |

注意：`decode-only` 会自动计算生成 token 预算，因此忽略 `--output-len`。

三个 launcher 的选项名称一致，且都会在启动 Python 前校验 `--phase` 取值、步数以及 `CHUNKED_PREFILL` / `ENFORCE_EAGER` / `ENABLE_EP` 开关。

## 5. Decode-only 原理与约束

`llm.generate()` 无法在调用中途精确开启 profiler。`decode-only` 会直接驱动 engine：

1. 加入 `bs` 条请求；
2. 循环 `engine.step()`，直到所有请求都完成 prefill；
3. 再执行 `settle_steps`；
4. 只采集 `profile_steps` 个纯 decode step；
5. 结束后 abort 请求，避免 batch 在窗口中缩小。

预算公式：

$$
\text{chunks}=\left\lceil\frac{\text{input\_len}}{\text{max\_num\_batched\_tokens}}\right\rceil
$$

$$
\text{max\_drain\_steps}=bs\times\text{chunks}+bs
$$

$$
\text{total\_tokens}=\text{max\_drain\_steps}+\text{settle\_steps}+\text{profile\_steps}+16
$$

必须满足：

$$
\text{max\_model\_len}\geq\text{input\_len}+\text{total\_tokens}
$$

另外还有以下约束：

1. `max_num_seqs >= bs`；
2. 关闭 chunked prefill 时，`max_num_batched_tokens >= max_model_len`；
3. `ZE_AFFINITY_MASK` 暴露的 XPU 数量不能少于 `tp`；
4. `bs * input_len` 必须能放入 KV cache，否则可能发生 preemption。

## 6. XPU 长序列示例

长序列 decode 应开启 chunked prefill。128K 输入、bs=8、TP=4：

```bash
ZE_AFFINITY_MASK=4,5,6,7 \
bash profile-scripts/xpu/run_torch_profile.sh \
  --model /models/Qwen3-32B \
  --tp 4 \
  --max-model-len 135168 \
  --max-num-batched-tokens 16384 \
  --max-num-seqs 8 \
  --enable-chunked-prefill \
  --input-len 131072 \
  --bs 8 \
  --phase decode-only \
  --profile-steps 20
```

常用参考：

| 输入长度 | 建议 `max_model_len` | `max_num_batched_tokens` | Chunked prefill |
| ---: | ---: | ---: | --- |
| 4K | 8192 | 8192 | 可选 |
| 16K | 20480 | 8192 或 16384 | 开启 |
| 64K | 69632 | 16384 | 开启 |
| 128K | 135168 | 16384 | 开启 |

最终以脚本根据 batch、chunk 数和 profile step 给出的最小长度提示为准。

## 7. 指标与 trace

`decode-only` 会输出：

```text
decode step latency : 42.13 ms
per-token latency   : 42.13 ms (TPOT, bs=32)
decode throughput   : 759.6 tok/s
```

- `decode step latency`：固定 batch 的单个 decode step 时延。
- `TPOT`：单序列每 token 时延；同步 decode 下与 step latency 相同。
- `decode throughput`：整机吞吐，等于 `bs / step_latency`。

若出现以下警告，说明窗口内 batch 不恒定，trace 不适合做严格对比：

```text
batch was not constant (preemption?), trace may be impure
```

torch trace 可使用 Perfetto 或 `chrome://tracing` 打开。

### 7.1 Trace 分析脚本

采集完之后的分析脚本按模型族分成两个目录，里面都是指向
`.github/skills/<skill>/scripts/` 的软链接，改动只需在 skill 目录里做一次。
每个 skill 的 `SKILL.md` 和 `references/` 里有完整的用法、计量陷阱和报告模板。

| 目录 | 适用模型 | 对应 skill |
| --- | --- | --- |
| [`qwen3-perf-analysis/`](qwen3-perf-analysis) | Qwen3 等**稠密**模型（XPU unitrace + NV torch profiler） | `qwen3-perf-analysis`、`xpu-nv-perf-comparison` |
| [`qwen36-hybrid-perf-analysis/`](qwen36-hybrid-perf-analysis) | Qwen3.5 / Qwen3.6 等 **GDN 线性注意力 + 全注意力混合**模型（XPU unitrace） | `qwen36-hybrid-perf-analysis` |

判据很简单：**trace 里出现 `gdn::` kernel 就用 hybrid 那套**，否则用稠密那套。

```bash
# 稠密模型
python profile-scripts/qwen3-perf-analysis/analyze_trace.py <trace>.json --num-layers 64 --batch 1
python profile-scripts/qwen3-perf-analysis/analyze_gemm_shapes.py <trace>.json --batch 1 --prompt-len 3500
python profile-scripts/qwen3-perf-analysis/make_perfetto_trace.py <trace>.json --num-layers 64

# 混合注意力模型（GDN + full attention）—— Intel XPU unitrace
A=profile-scripts/qwen36-hybrid-perf-analysis
python $A/analyze_hybrid_trace.py <trace>.json                     # 分类 / 逐层 / 单步精确表
python $A/analyze_hybrid_gemm.py  <trace>.json --batch 1 --prompt-len 3300 \
       --ref-tflops 500 --ref-bw 1035                              # 逐 shape TFLOPS / 带宽
python $A/make_hybrid_perfetto_trace.py <trace>.json               # reflow 后的时间线

# 混合注意力模型（GDN + full attention）—— NVIDIA torch profiler
python $A/analyze_nv_hybrid_trace.py rank0.*.pt.trace.json.gz --batch 1
```

依赖：`pip install ijson`。`hybrid_common.py` 是前三个脚本共用的模型拓扑与
kernel 分类，换模型时只改它里面的 `CFG`。

## 8. NVIDIA CUDA 使用

NV 目录只支持 torch profiler，不使用 unitrace。默认开启 CUDA graph，只有传入 `--enforce-eager` 才关闭。

纯 decode、TP=4：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
bash profile-scripts/nv/run_torch_profile_nv.sh \
  --model /models/Qwen3-32B \
  --tp 4 \
  --max-model-len 8192 \
  --max-num-batched-tokens 8192 \
  --max-num-seqs 32 \
  --input-len 2048 \
  --bs 32 \
  --phase decode-only \
  --profile-steps 20
```

Eager 对照：

```bash
CUDA_VISIBLE_DEVICES=0 \
bash profile-scripts/nv/run_torch_profile_nv.sh \
  --model /models/Qwen3-32B \
  --enforce-eager
```

NV trace 默认写入：

```text
profile-scripts/nv/torch-profile-trace/<tag>/
```

tag 中包含 `cudagraph` 或 `eager`。

## 9. 常见问题

### 找不到 Python、vLLM 或 unitrace

```bash
PYTHON_BIN=/path/to/python
UNITRACE_BIN=/path/to/unitrace
```

运行对应 launcher 的 `--check` 查看实际解释器、torch/vLLM 版本和设备数量。

### `ZE_AFFINITY_MASK exposes N device(s) but --tp is M`

可见 XPU 数量小于 TP。扩大 `ZE_AFFINITY_MASK` 或降低 `--tp`。

### `--max-num-batched-tokens must be >= --max-model-len`

关闭 chunked prefill 时，单步预算必须不小于最大模型长度。提高预算，或传入 `--enable-chunked-prefill`。

### `decode-only needs max_model_len >= input_len + N`

提高 `--max-model-len`，或降低 `--bs` / `--profile-steps`。

### `prefill did not drain ... KV cache is probably too small`

降低 `--bs` 或 `--input-len`。该错误通常表示 KV cache 无法同时容纳当前请求。

### `decode batch would shrink during profiling`

提高 `--max-num-batched-tokens` 以缩短 prefill 排空过程，或降低 `--profile-steps`。

### `--phase 只能是 all|prefill|decode-only`

launcher 在启动 Python 前会校验 phase 取值，检查拼写。同理 `--profile-steps` 必须是正整数，`--settle-steps` 必须是非负整数。

### unitrace trace 里仍然混着 prefill

确认传入了 `--phase decode-only`。只用 `--start-paused` 只能跳过模型加载和 warmup，`all` phase 仍会把 prefill 采进去。

### Unitrace JSON 不完整

提高 `--shutdown-timeout` 和 Python 参数 `--unitrace-final-flush-seconds`，并确保进程执行到正常 shutdown。

## 10. 查看完整参数

```bash
bash profile-scripts/xpu/run_torch_profile.sh --help
bash profile-scripts/xpu/run_unitrace.sh --help
bash profile-scripts/nv/run_torch_profile_nv.sh --help
```