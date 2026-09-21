---
name: xpu-nv-perf-comparison
description: 'Compare LLM inference performance between Intel XPU and NVIDIA GPU and produce a xpu-vs-nv-comparison.md. Use when two perf-report.md / trace analyses already exist and the user asks to: compare XPU vs NV, cross-platform performance comparison, 对比分析两个平台, find where the performance gap is, decompose the prefill/decode gap by kernel category, build a side-by-side category table with kernel names, compare GEMM TFLOPS and memory bandwidth across platforms, or decide what to optimize first on the slower platform. Depends on the per-platform analyses produced by the qwen3-perf-analysis skill (dense models) or the qwen36-hybrid-perf-analysis skill (GDN hybrids, which has its own compare_hybrid_perf.py).'
argument-hint: '<xpu analyze_trace.txt> <nv analysis.txt> [--prompt-len N] [--batch N]'
---

# XPU vs NVIDIA Performance Comparison

Turns two per-platform trace analyses into a single `xpu-vs-nv-comparison-<YYMMDD-HHMMSS>.md`
with side-by-side category tables, gap decomposition, and a prioritized
optimization list for the slower platform.

## When to Use

- Two `perf-report.md` (or two analyzer outputs) already exist and need comparing
- "XPU 和 NV 差距在哪"、"gap 主要来自哪里"、"先优化什么"
- Any cross-platform LLM inference breakdown comparison

**Prerequisite**: run the [qwen3-perf-analysis](../qwen3-perf-analysis/SKILL.md)
skill on each platform first. This skill consumes its output.

> **GDN hybrid models (Qwen3.5 / 3.6 / Qwen3-Next) use a different script.**
> `compare_perf.py` below only parses the **dense** `analyze_trace.py` format
> (`--- single PREFILL step (exact counts, ...)`). If the traces contain `gdn::`
> kernels, generate the scaffold with
> [qwen36-hybrid-perf-analysis/scripts/compare_hybrid_perf.py](../qwen36-hybrid-perf-analysis/scripts/compare_hybrid_perf.py)
> instead, then keep using Step 2 / Step 4 and the templates in this skill:
>
> ```bash
> python ../qwen36-hybrid-perf-analysis/scripts/compare_hybrid_perf.py \
>     --xpu <xpu dir>/analyze_hybrid_trace.txt \
>     --nv  <nv dir>/analyze_nv_hybrid_trace.txt \
>     --xpu-name "Intel XPU" --nv-name "NVIDIA RTX PRO 5000" \
>     -o xpu-vs-nv-comparison-<YYMMDD-HHMMSS>.md
> ```

## Step 1 — Produce the two inputs

Both sides must be analyzed with `--batch` set, so the single-step sections exist.

```bash
S=<repo>/.github/skills/qwen3-perf-analysis/scripts

# Intel XPU (unitrace)
python $S/analyze_trace.py python.<pid>.json \
    --num-layers 64 --batch 1 --top 20 --max-name 92 > analyze_trace.txt
python $S/analyze_gemm_shapes.py python.<pid>.json \
    --batch 1 --prompt-len 3500 > gemm_shapes.txt

# NVIDIA (torch profiler) -- GEMM and attention tables are in the same output
python $S/analyze_nv_trace.py rank0.*.pt.trace.json.gz \
    --prompt-len 3500 --batch 1 > analysis.txt
```

## Step 2 — Verify the two runs are comparable

Do this before generating anything. Record the answers; they go into §0 of the report.

| Check | Must be | If it differs |
|---|---|---|
| 模型与层数 | identical | not comparable, stop |
| prompt 长度 (`in<N>`) | identical | not comparable for attention |
| batch (`bs<N>`) | identical | per-step numbers not comparable |
| TP | identical | scale or stop |
| `out<N>` | may differ | fine — per-step metrics are unaffected |
| 量化格式 / group size | may differ | **record it**; it changes bytes-per-weight |
| 图执行模式 | may differ | record it; affects which pitfalls apply |

## Step 3 — Generate the scaffold

```bash
python ./scripts/compare_perf.py \
    --xpu <path>/analyze_trace.txt \
    --nv  <path>/analysis.txt \
    --xpu-name "Intel Xe3" --nv-name "NVIDIA RTX PRO 5000" \
    --prompt-len 3500 --batch 1 \
    -o xpu-vs-nv-comparison-<YYMMDD-HHMMSS>.md
```

**Never overwrite an existing comparison report.** Always suffix the output with the
XPU trace's timestamp (the `-<YYMMDD-HHMMSS>` tail of its trace directory, e.g.
`xpu-vs-nv-comparison-260915-083118.md`). Earlier reports are kept as-is so two runs
can be diffed. The script's default `-o` is unsuffixed — always pass `-o` explicitly.

In this repo all five scripts are also symlinked into `profile-scripts/qwen3-perf-analysis/`,
so they can be invoked from one place without knowing the skill layout.

It emits §1 总体, §2 PREFILL 对照, §3 DECODE 对照, §4 gap 分解, and a §5 checklist
of what still needs human analysis.

**It also runs sanity checks.** If it prints `WARNING: step windows look clipped`,
the per-layer kernel counts disagree between prefill and decode on one platform —
the analyzer run is wrong, fix it before continuing. Only `split_k_reduce` /
`ReduceSplitK` / `splitkv_combine` are allowed to inflate a decode category.

## Step 4 — Fill in what the script cannot

The script handles the mechanical tables. These require reading both reports:

1. **§0 对比基准** — device, quantization, GEMM backend, attention impl, graph mode,
   workload for both sides, plus an explicit 可比性 statement.
2. **逐 shape GEMM 对照** — prefill TFLOPS and decode GB/s per linear, from the two
   GEMM tables. This is where single-shape outliers show up.
3. **attention 对照** — TFLOPS, effective bandwidth, and the reduce/combine share.
4. **根因判断** for each gap — see below.
5. **优化优先级表** — each row needs a quantified benefit.

Follow [references/comparison-template.md](./references/comparison-template.md)
for the full structure, and [references/methodology.md](./references/methodology.md)
for the conventions (ratio direction, what counts as a real finding, how to size
an optimization).

## Interpreting the Gap

The §4 table tells you *where* the time goes; you must say *why*. Three buckets:

| Root cause | Evidence | How to phrase it |
|---|---|---|
| **软件问题** | one shape is an outlier while its siblings are fine; or one platform falls back to a different kernel family | actionable, give a fix and a number |
| **硬件规格差异** | all shapes degrade proportionally; SM/Xe-core count or peak bandwidth differs | not a bug — say so explicitly |
| **实现成熟度** | consistent but lower efficiency across the board (e.g. FMHA at 35% vs 42% of the platform's own GEMM throughput) | improvable, but bounded |

Normalize against each platform's **own** peak before calling something slow.
Comparing attention TFLOPS to the same platform's GEMM TFLOPS is far more
informative than comparing raw numbers across vendors.

## Sizing an Optimization

Always derive the benefit from measured data, never a guess:

```
目标带宽取自同一平台自己最好的 shape
gate_up: 726.4 GB/s -> down_proj 的 1034.2 GB/s
单层 372.3 µs × 726.4/1034.2 = 261.5 µs
每步省 (372.3-261.5) × 64 layers = 7.09 ms
decode 45.31 -> 38.19 ms/step (-15.7%)
```

Then state the combined effect: "若 1+2 完成，差距从 1.46× 收窄到 1.19×".

## Common Mistakes

- **用聚合表做对比** — its `cnt/step` is fractional because the first and last step
  are clipped. Always use the single-step tables.
- **照搬别处的分类名** — keep the analyzer's own categories so rows stay traceable
  back to the trace.
- **kernel 列留空** — the searchable kernel name is the most valuable column; it is
  what lets someone re-open the trace and find the row.
- **把 SM 数量差异说成软件 bug** — check the hardware spec before blaming software.
- **只报比值不报绝对值** — a 15× ratio on a 0.2 ms category is noise.
