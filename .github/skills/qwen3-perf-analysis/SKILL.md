---
name: qwen3-perf-analysis
description: 'Analyze vLLM LLM inference traces and produce a perf-report.md with prefill/decode breakdown. Use when given a unitrace Chrome-trace JSON from Intel XPU (python.*.json) or a PyTorch Profiler trace from NVIDIA (rank0.*.pt.trace.json.gz), or when asked to: break down GPU kernel time by category, compute per-shape GEMM TFLOPS and memory bandwidth, analyze attention (FMHA/FlashAttention) efficiency, compare a single prefill step against a single decode step, find why decode is slow, generate a Perfetto timeline, or compare XPU vs NVIDIA performance. Handles XPU-Graph/CUDA-Graph collapsed timestamps, DeepGEMM split-K overlap, and GEMM-to-linear-layer attribution.'
argument-hint: '<path to unitrace .json or torch profiler .pt.trace.json.gz> [--batch N] [--prompt-len N]'
---

# vLLM Inference Trace Analysis (XPU / NVIDIA)

Turns a raw device trace into a `perf-report.md` containing: phase-split category
timings, exact single-step breakdowns with kernel names, per-shape GEMM TFLOPS and
achieved bandwidth, and attention efficiency.

## When to Use

- A unitrace Chrome-trace JSON captured on Intel XPU (`python.<pid>.json`)
- A PyTorch Profiler trace captured on NVIDIA (`rank0.*.pt.trace.json[.gz]`)
- Questions like "decode 为什么慢"、"GEMM 带宽多少"、"prefill 和 decode 各占多少"、
  "XPU 和 NV 差距在哪"

## Step 0 — Identify the trace and gather inputs

| Trace | Platform | Tool |
|---|---|---|
| `python.<pid>.json`, events have `cat: "gpu_op"` | Intel XPU | [analyze_trace.py](./scripts/analyze_trace.py) |
| `rank0.*.pt.trace.json.gz`, has `deviceProperties` | NVIDIA | [analyze_nv_trace.py](./scripts/analyze_nv_trace.py) |

You **must** know three things before running anything. Ask the user if not obvious
from the folder name (they usually encode it, e.g. `...-in3500-out20-bs1-tp1`):

| Input | Why | How to get it |
|---|---|---|
| `--num-layers` | converts kernel counts to steps | model `config.json` → `num_hidden_layers` |
| `--batch` | separates decode from prefill forwards | `bs<N>` in the folder name |
| `--prompt-len` | KV length for the attention cost model | `in<N>` in the folder name |

**Always pass `--num-layers` explicitly.** Autodetection is unreliable (after
reflow it misreports 64 as 192).

Then read the model `config.json` and check the `CFG` dict at the top of
[analyze_gemm_shapes.py](./scripts/analyze_gemm_shapes.py) and
[analyze_nv_trace.py](./scripts/analyze_nv_trace.py) matches:
`hidden, heads, kv_heads, head_dim, inter, layers, vocab`. Also check the
quantization: `MXFP8_BYTES`/`FP8_BYTES` must match `quantization_config.group_size`
(XPU MXFP8 uses 32 → 1+1/32; NV DeepGEMM 1d1d uses 128 → 1+1/128).

Dependency: `pip install ijson` (XPU scripts stream large JSON).

## Step 1 — Run the scripts

### Intel XPU

```bash
# top-down + phase split + exact single prefill/decode step with kernel names
python analyze_trace.py <trace>.json --num-layers 64 --batch 1 \
    --expected-output-len 20 --top 20 --max-name 92 > analyze_trace.txt

# per-shape GEMM and attention: TFLOPS + achieved bandwidth
python analyze_gemm_shapes.py <trace>.json --batch 1 --prompt-len 3500

# Perfetto timeline -> drag the .json.gz into https://ui.perfetto.dev/
python make_perfetto_trace.py <trace>.json --num-layers 64
```

### NVIDIA

```bash
# everything in one pass: phase split, single-step, GEMM, attention
python analyze_nv_trace.py rank0.*.pt.trace.json.gz \
    --prompt-len 3500 --batch 1 > analysis.txt
```

Both sides share the same category taxonomy and the same "single step" window
definition, so their outputs can be compared row by row.

## Step 2 — Sanity-check before writing anything

Do not write the report until these pass. Each has caught a real bug before.

1. **Single-step counts must be whole numbers**, and prefill must equal decode for
   the per-layer categories: `Dense-GEMM = 4*layers + 1`, `Quantize = 4*layers`,
   `Norm/RoPE = 4*layers + 1`, `Activation = layers`, `KVCache-Write = layers`.
   A mismatch means the step window is clipped.
2. **Single-step TOTAL ≈ aggregate ms/step** (within ~1%).
3. **NV only**: single-step TOTAL ≈ the decode step period (start-to-start).
   If TOTAL is much larger, the split-K overlap is being double counted.
4. **GEMM rows must be self-consistent**: all four linears should land in a narrow
   TFLOPS band (prefill) or GB/s band (decode). A single outlier is a real finding;
   scattered values mean the attribution is wrong.

## Step 3 — Write `perf-report.md`

Put it next to the trace. Follow [the report template](./references/report-template.md).

## Critical Pitfalls

Read [references/pitfalls.md](./references/pitfalls.md) before interpreting any
number. Summary of the traps that produce plausible-but-wrong results:

| Trap | Effect if missed |
|---|---|
| XPU-Graph collapses kernel start times | union/idle/bubble metrics meaningless; must use sum-of-durations |
| Sorting events by `(ts, dur, name)` | destroys capture order under collapsed timestamps → wrong phase attribution |
| DeepGEMM split-K reduce runs concurrently with its GEMM | NV decode inflated by ~30% |
| NV `execute_context_*` annotation excludes lm_head + sampler | NV decode step under-counted by ~1.3 ms |
| Run start anchored on first KV-write (mid layer 0) | prefill loses one qkv GEMM + 3 norm kernels |
| Unterminated kernels at trace stop | one event with `dur` ≈ 10^5 s dwarfs the run |
| RoPE fused with q/k RMSNorm by torch.compile | cannot be reported separately; keep one `Norm/RoPE` bucket |

## How GEMM Attribution Works

Neither trace records M/N/K, so each GEMM launch is mapped to a linear layer:

- **NV prefill** — eager per-layer graphs, so DeepGEMM bakes N and K into the kernel
  name: `sm120_fp8_fp4_gemm_1d1d_impl<0u, N, K, ...>`. Direct lookup.
- **NV decode** — CUDA-Graph replay drops the host op and N becomes a runtime
  argument, so only K survives. K=8192→`o_proj`, K=25600→`down_proj`;
  K=hidden is ambiguous (qkv vs gate_up) and is resolved by the feeding kernel.
- **XPU** — the ND-range in the kernel name plus the fixed intra-layer order:
  `attn → o_proj → norm → gate_up → silu → down_proj → norm → qkv_proj`.
  The quantization/scale-copy kernels between them are skipped when looking back.

Cost models (GQA-aware, `H_q`/`H_kv` = query/kv heads, `D` = head_dim, `L` = KV length):

```
GEMM    FLOPs = 2·M·N·K
        Bytes = K·N·w + M·K·w + M·N·2      w = 1 + 1/group   (bf16 → w = 2)
decode  FLOPs = 4·B·H_q·D·L                Bytes = 4·B·H_kv·D·L + 4·B·H_q·D
prefill FLOPs = 2·H_q·D·T²  (causal)       Bytes = 4·T·H_q·D + 4·T·H_kv·D
```

## Adapting to Another Model

1. Update `CFG` in both `analyze_gemm_shapes.py` and `analyze_nv_trace.py`.
2. Update `MXFP8_BYTES` / `FP8_BYTES` for the quantization group size.
3. Check `classify()` still matches the layer order — MoE models add routing and
   grouped-GEMM kernels that need new buckets in `bucket()`.
4. Re-run Step 2's sanity checks; the counts change with layer count.

## Comparing Two Platforms

Use the [xpu-nv-perf-comparison](../xpu-nv-perf-comparison/SKILL.md) skill — it
consumes this skill's output and generates the comparison tables automatically.

Key rule: build the comparison from the **single-step** tables only (never the
aggregate averages — their counts are fractional).
