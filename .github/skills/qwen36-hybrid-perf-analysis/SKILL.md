---
name: qwen36-hybrid-perf-analysis
description: 'Analyze vLLM inference traces of Qwen3.5/Qwen3.6 hybrid-attention models (GDN gated-delta-net linear attention interleaved with full attention) on Intel XPU, and produce a perf-report.md with a per-operator prefill/decode breakdown down to individual kernel names and ND-ranges. Use when given a unitrace Chrome-trace JSON (python.<pid>.json) from a Qwen3.6-27B / Qwen3.5 / Qwen3-Next style run, or when asked to: break down GPU kernel time for a hybrid GDN + full-attention model, separate GDN layers from full-attention layers, compute per-shape GEMM TFLOPS and achieved bandwidth for MXFP4/MXFP8 linears, judge whether the quantized GEMM path has an XMX kernel at all or is running oneDNN weight decompression, analyze the GDN chunk prefill kernels or the recurrent decode kernels, find the exact hot kernel names, diff two runs operator by operator, analyze a prefill-only (out=1) trace including wall/idle/gap analysis, find why decode is slow, or estimate the optimization headroom. Handles XPU-Graph collapsed timestamps, graph replays that land inside the prefill window, AutoRound online Hadamard rotations (including the fused fwht_quant variant), and the in_proj_qkvz / in_proj_ba pair.'
argument-hint: '<path to unitrace python.<pid>.json> [--batch N] [--prompt-len N]'
---

# Qwen3.5/3.6 Hybrid-Attention Trace Analysis (Intel XPU)

Turns a unitrace device trace of a **GDN + full-attention hybrid** model into a
`perf-report.md` with: phase-split category timings, exact single-prefill and
single-decode kernel breakdowns **with full kernel names and ND-ranges**,
per-linear GEMM TFLOPS / achieved bandwidth, separate GDN and full-attention
cost models, and a quantified optimization list.

| script | purpose |
|---|---|
| `analyze_hybrid_trace.py` | phase split, category totals, sanity check, per-layer-type time |
| `analyze_hybrid_gemm.py` | per-shape GEMM TFLOPS / GB/s, GDN and FMHA efficiency, roll-up |
| **`dump_kernels.py`** | **per-kernel dump with untruncated names + GEMM attribution + duration clustering** — the source for the report's exact single-step tables |
| **`analyze_prefill_only.py`** | **`out=1` traces: categories + real wall / busy / idle / gap attribution** |
| **`compare_hybrid_perf.py`** | **XPU vs NV comparison report** from the two analyses above (the dense `compare_perf.py` cannot parse the hybrid format) |
| `make_hybrid_perfetto_trace.py` | reflowed Perfetto timeline (wall-clock step time) |
| `analyze_nv_hybrid_trace.py` | the NVIDIA counterpart of the first two |
| `bench_mxfp4_vs_bf16.py` | standalone microbenchmark to confirm a slow-GEMM claim |

This is the hybrid sibling of [qwen3-perf-analysis](../qwen3-perf-analysis/SKILL.md),
which covers dense Qwen3 models. Use this one whenever the trace contains
`gdn::` kernels.

## When to Use

- The trace contains `gdn::gated_delta_rule_kernel`, `gdn::causal_conv1d_kernel`
  or `gdn::Chunk*Kernel` → hybrid model → **this skill**
- The model config has `layer_types` mixing `linear_attention` and
  `full_attention`, or a `full_attention_interval`
- Questions like "GDN 和 full attention 各占多少"、"decode 为什么只有 18 tok/s"、
  "MXFP4 有没有生效"、"prefill 的 GDN chunk 路径贵在哪"

## Quick start

```bash
pip install ijson                        # the only dependency
S=<repo>/.github/skills/qwen36-hybrid-perf-analysis/scripts
D=<trace dir>                            # holds python.<pid>.json
T=$(ls $D/python.*.json)

python $S/analyze_hybrid_trace.py $T > $D/analyze_hybrid_trace.txt
head -40 $D/analyze_hybrid_trace.txt     # STOP unless the sanity check is all OK

python $S/analyze_hybrid_gemm.py  $T > $D/analyze_hybrid_gemm.txt
python $S/dump_kernels.py $T prefill > $D/kernel-detail-prefill.txt
python $S/dump_kernels.py $T <k>     > $D/kernel-detail-decode-step<k>.txt
python $S/make_hybrid_perfetto_trace.py $T
```

`<k>` is the decode step `analyze_hybrid_trace.py` reports as
`SINGLE DECODE STEP #k`. **Prompt length and batch size are auto-detected** from
the KV-write ND-range and echoed as `workload: prompt=... batch=...` — check
that line matches the directory name before reading anything else. Override
with `--prompt-len` / `--batch` only if detection fails. For an `out=1` trace
use `analyze_prefill_only.py` instead (see Step 1).

**On a new environment, three things are not portable and must be checked
before the numbers mean anything** — the skill tells you when each is wrong:

| what | how it fails | where to fix |
|---|---|---|
| `CFG` (model dims) | sanity check layer counts / Dense-GEMM go `BAD` | Step 0 |
| `gemm_kernel` work-group shapes (device + driver) | `Dense-GEMM` count `BAD`; the sanity check then prints the full ND-range histogram so they can be re-derived | Step 2 |
| device peak / reference numbers | nothing fails — the headroom estimate is silently wrong | Step 1, `--ref-*` |

Changing **input / output length or batch size needs no code change**, with one
exception: if the prompt exceeds `max_num_batched_tokens`, chunked prefill
splits it across several forward passes and the "window 0 = prefill" model
breaks. Both scripts detect that (more than one window contains the full
prefill FMHA kernel) and refuse to let the result pass silently.

### Cross-platform comparison

```bash
python $S/analyze_nv_hybrid_trace.py $N/rank0.*.pt.trace.json.gz > $N/analyze_nv_hybrid_trace.txt

python $S/compare_hybrid_perf.py \
    --xpu $D/analyze_hybrid_trace.txt \
    --nv  $N/analyze_nv_hybrid_trace.txt \
    --xpu-name "Intel XPU (MXFP4)" --nv-name "NVIDIA RTX PRO 5000 (NVFP4)" \
    -o xpu-vs-nv-comparison-<YYMMDD-HHMMSS>.md
```

It reads only the `.txt` outputs (no trace, no dependencies, <1 s) and writes
§1 总体 / §2 PREFILL 对照 / §2.1 逐 shape GEMM / §3 DECODE 对照 / §3.1 逐 shape GEMM +
**自我参照表** / §3.3 按层类型 / §4 gap 分解, plus a checklist of the parts that need
a human (§0 对比基准、attention 与 GDN chunk 逐 kernel 对照、根因判定、优化优先级).
`analyze_hybrid_gemm.txt` is picked up automatically from the XPU trace's
directory; pass `--xpu-gemm` if it lives elsewhere.

What it does for you beyond the tables:

- **refuses to overwrite** an existing comparison — always suffix `-o` with the
  XPU trace's timestamp so two runs can be diffed;
- **warns when the two runs are not comparable** (prompt length or batch differ);
- **flags the inflated `in_proj_qkvz` row** (⚠) and uses the corrected aggregate;
- emits the **自我参照 table** — each platform's quantised-GEMM bandwidth as a
  percentage of *its own* best shape. That ratio, not the cross-vendor one, is
  what separates "software problem" from "hardware spec difference".

> The `xpu-nv-perf-comparison` skill's `compare_perf.py` only parses the **dense**
> `analyze_trace.py` format (`--- single PREFILL step (exact counts, ...)`) and
> cannot read the hybrid analyses. Use `compare_hybrid_perf.py` for GDN hybrids,
> and follow that skill's `references/comparison-template.md` when filling in the
> human sections.

## Step 0 — Gather inputs

Read the model `config.json` (`text_config`) and make the `CFG` dict at the top
of [hybrid_common.py](./scripts/hybrid_common.py) match:

| CFG key | config.json field |
|---|---|
| `hidden`, `layers`, `inter`, `vocab` | `hidden_size`, `num_hidden_layers`, `intermediate_size`, `vocab_size` |
| `full_attention_interval` | `full_attention_interval` (cross-check `layer_types`) |
| `heads`, `kv_heads`, `head_dim` | `num_attention_heads`, `num_key_value_heads`, `head_dim` |
| `attn_output_gate` | `attn_output_gate` — **doubles the q half of `qkv_proj`** |
| `gdn_k_heads`, `gdn_k_dim` | `linear_num_key_heads`, `linear_key_head_dim` |
| `gdn_v_heads`, `gdn_v_dim` | `linear_num_value_heads`, `linear_value_head_dim` |
| `gdn_conv_kernel` | `linear_conv_kernel_dim` |
| `gdn_state_dtype_bytes` | `mamba_ssm_dtype` (float32 → 4) |

Also check `quantization_config`: `MXFP4_BYTES` must be `bits/8 + 1/group_size`
(4-bit, group 32 → `0.5 + 1/32`). If `rotation_config.allow_online_rotation` is
true, expect a chain of extra `gemm_kernel` launches in front of every
quantized linear — the scripts bucket those as `Hadamard-Rotation`.

`--batch` and `--prompt-len` are usually encoded in the directory name
(`...-in3300-out20-bs1-tp1`). Dependency: `pip install ijson`.

## Step 1 — Run the scripts

```bash
D=<trace dir>
python analyze_hybrid_trace.py $D/python.*.json > $D/analyze_hybrid_trace.txt
python analyze_hybrid_gemm.py  $D/python.*.json --batch 1 --prompt-len 3300 \
       --ref-tflops 500 --ref-bw 1035 > $D/analyze_hybrid_gemm.txt

# timeline -> drag the .json.gz into https://ui.perfetto.dev/
python make_hybrid_perfetto_trace.py $D/python.*.json

# per-kernel detail with FULL kernel names -- the data source for the report's
# §3.3 / §3.4 exact single-step tables. NOT optional, see Step 3.
python dump_kernels.py $D/python.*.json prefill > $D/kernel-detail-prefill.txt
python dump_kernels.py $D/python.*.json 4       > $D/kernel-detail-decode-step4.txt
```

`analyze_hybrid_trace.py` truncates kernel names to ~118 characters so its
tables stay readable. That truncation destroys exactly what the report needs:
a name that can be pasted back into the trace, and the ND-range that ties a
kernel to a shape. **Always also run `dump_kernels.py`** and build §3.3 / §3.4
from its output. Pass the decode step index that `analyze_hybrid_trace.py`
chose (it prints `DECODE step #k`) so the two agree.

### Prefill-only traces (`out=1`)

A run with `output=1` has a single forward pass, so `analyze_hybrid_trace.py`
exits with `need at least one prefill and one decode window`. Use:

```bash
python analyze_prefill_only.py $D/python.*.json --prompt-len 3300 > $D/analyze_prefill_only.txt
python dump_kernels.py        $D/python.*.json all               > $D/kernel-detail-prefill.txt
```

These traces are worth collecting deliberately: **with no graph replay the start
timestamps are not collapsed**, so wall span, device-busy ratio and per-gap idle
attribution are all valid — numbers an `out>1` trace cannot produce at all.
On the reference pair they also cross-validate the `out=20` report: after the
misplaced replay is removed the two prefill breakdowns agree to **0.17 %**.
Always state that comparison in the report.

For an **unquantized BF16 checkpoint** add `--weight-dtype bf16` to all analysis
scripts. It switches the byte model to 2 B/element, drops the Hadamard and
quantize sections, and uses the 4-linear GDN decomposition (a BF16 build fuses
the GDN input projection, giving `4G + 4F + 1 = 257` Dense-GEMMs per forward
instead of `5G + 4F + 1 = 305`). Running the same workload in both precisions is
the strongest available check on any "the quantized path is slow" claim -- see
pitfall 5.

In this repo every script is also symlinked into
`profile-scripts/qwen36-hybrid-perf-analysis/`, so they can be run straight from
the repo root without touching `.github/`. `hybrid_common.py` must stay next to
the others — they import it from their own directory. **Run the scripts from
there; do not copy them into the trace directory.**

### NVIDIA

```bash
python analyze_nv_hybrid_trace.py $D/rank0.*.pt.trace.json.gz --batch 1
```

One script covers everything the two XPU scripts do. Differences to know about:

- **Phases come from annotations**, not from a sampler kernel:
  `execute_context_<n>(<ctx>)_generation_<m>(<gen>)`. `ctx > 0` is the prefill.
- **`cat=overhead` must be excluded.** `Command Buffer Full` shows up in the
  profiler summary as "CUDA total 397.9 ms / 34.6 %" but is a host-side CUPTI
  marker that overlaps real kernels; counting it double-counts the run. Only
  `kernel` / `gpu_memset` / `gpu_memcpy` are summed.
- **There is no `--weight-dtype`.** An NVIDIA build serves each linear from a
  different backend and the backend name encodes the precision, so the byte
  model is derived per kernel:

  | kernel | precision | B/element |
  |---|---|---|
  | `marlin::Marlin<...>` | NVFP4, **W4A16** | 0.5625 |
  | `cudnn_..._matMul_pointwise` | FP8 | 1.0 |
  | `sm89_xmma_gemm_e4m3bf16_...` | FP8 (e4m3) | 1.0 |
  | `internal::gemvx::kernel` / `cutlass_..._bf16_...gemm` | BF16 | 2.0 |

  **Confirm this against the checkpoint's `quantization_config` rather than
  trusting the mapping blindly** — a ModelOpt export states it outright, and
  these exports are usually `"quant_algo": "MIXED_PRECISION"` rather than one
  format throughout. For `nvidia/Qwen3.6-27B-NVFP4`:

  | config entry | meaning |
  |---|---|
  | `config_groups.group_0`: `input_activations` **and** `weights` = `num_bits: 8, type: float` | W8A8 FP8, static scales — GDN `in_proj_qkv`/`in_proj_z`/`out_proj`, full-attn `q/k/v/o_proj` |
  | `config_groups.group_1`: `weights` = `num_bits: 4, group_size: 16`, **no `input_activations`** | weight-only; the per-layer list spells it `"quant_algo": "W4A16_NVFP4"` — MLP + `lm_head` |
  | a linear absent from every group | stays BF16 (here: `in_proj_b` / `in_proj_a`) |
  | `kv_cache_scheme`: `num_bits: 8, type: float` | FP8 KV cache, static scales |

  Then sanity-check the result as always: the decode GB/s of every shape must
  land in one narrow band. If a precision guess is wrong, that shape either
  exceeds the physical bandwidth or falls far outside the band.
- **`W4A16` means 4-bit buys bandwidth but not FLOPs** — the weights are
  decompressed to FP16 inside the kernel. The script's per-backend TFLOPS table
  makes this visible directly: on the reference trace the FP8 path reaches
  405.7 TFLOPS and Marlin only 189.0, a 2.15x ratio that matches the FP8:BF16
  tensor-core ratio and the config's literal `W4A16` label.
- **Timestamps are trustworthy.** CUPTI times every kernel individually even
  under CUDA Graph, so unlike the XPU side there is no reflow step and idle /
  bubble analysis is valid.

`--ref-tflops` / `--ref-bw` turn the report's headroom estimate from a guess
into a measurement, and they are **per device** — carrying the old server's
numbers to a new one is the easiest way to publish a wrong conclusion. They
default to 0 (section omitted). In order of preference, take them from:

1. another trace of a **healthy** kernel on the same device (e.g. the Qwen3-32B
   MXFP8 CUTLASS-SYCL run gave 500 TFLOPS prefill / 1035 GB/s decode on
   `Intel(R) Graphics [0x674f]`);
2. `bench_mxfp4_vs_bf16.py` on the same shapes;
3. the trace's own `lm_head` — it is bf16 and goes through oneDNN in every XPU
   trace, so its decode GB/s is a **lower bound** on what the device delivers,
   and it needs no external input at all. Always report this one regardless.

Vendor peak numbers (HBM GB/s, XMX TFLOPS per dtype) belong in the report's
§1.1 hardware table, but never use them as the efficiency yardstick — quote
"% of the best measured kernel on this device" first, "% of vendor peak"
second.

### The timeline is not optional here

With more than half the kernels sharing a start stamp, the raw trace is
unreadable in any viewer. [make_hybrid_perfetto_trace.py](./scripts/make_hybrid_perfetto_trace.py)
**reflows** each collapsed group end-to-end, monotonically (the cursor never
moves backwards — one group here sums to 33.9 ms while the next distinct stamp
is only 22.3 ms later, so a naive reflow would overlap). It emits four tracks:

| track | content |
|---|---|
| `PHASE` | PREFILL / DECODE, plus an explicit slice marking the **misplaced graph replay** |
| `forward passes` | one slice per sampler window, flagged complete / INCOMPLETE |
| `decoder layers` | `L0 gdn` … `L63 full` + `lm_head + sampler` — the fastest way to see GDN vs full-attention cost |
| `GPU kernels` | every kernel, with its category in `args.bucket` |

Use it to read off the **wall-clock** step time, which the sum-of-durations
analysis cannot give: reflow preserves inter-group gaps, so the difference
between a step's slice duration and its kernel time is a **lower bound** on
device idle. Do not read absolute positions inside a graph replay — they are
only accurate to within one submission.

## Step 2 — Sanity check before writing anything

`analyze_hybrid_trace.py` prints the check itself. Every line must say `OK`;
these counts are exact because a sampler-delimited window holds exactly one
forward pass. With `L` layers, `F = L / full_attention_interval` full layers
and `G = L - F` GDN layers:

| quantity | expected |
|---|---|
| gdn layers / full layers | `G` / `F` |
| `Dense-GEMM` | `5G + 4F + 1` quantized, `4G + 4F + 1` for BF16 (fused GDN in_proj) |
| `Quantize(mxfp4)`, `Quant-scale cast` | `5G + 4F`, or `0` for BF16 |
| `Activation(SiLU)` | `L` |
| `KVCache-Write`, `FullAttn-OutGate` | `F` |
| `GDN-Norm/Gate` | `G` |
| unattributed GEMM | `0` |

`Quant-scale cast` is expected to be `0` on a build that fuses rotation +
quantise + scale write into one `ark::XpuMxfp4Hadamard::fwht_quant_per_item`
(`fused_hadamard_quant()` detects this); older builds emit `5G + 4F` separate
`Float8_e8m0` casts. Both forms pass.

**A `BAD` line is usually a renamed kernel, not a clipped window.** The markers
are literal Inductor / backend kernel names and a newer build fuses more into
each one, which silently moves a whole category. The taxonomy already handles
the two known cases — `fwht_quant_per_item`, and the GDN gated-norm kernel that
absorbed the Hadamard/quantise (`RE_GDN_GATED_NORM` is matched **before**
`RE_QUANT` for exactly this reason). If a new one appears:

1. `grep -o '<prefix>[^"]*' trace.json | sort -u` to get the new full name;
2. check whether the count that went missing equals the count that grew — if so
   it is a rename, not lost work;
3. fix the regex order in `bucket()` rather than post-processing in a script,
   so every script and the sanity check agree;
4. record the old → new name in the report's §3.5 table.

Only treat `BAD` as a data problem once the total kernel count also disagrees.

The NVIDIA script checks the same invariants plus `GDN-Attn(recurrent) = 2G`
(decode) and `GDN-Attn(chunk)` divisible by `G` (prefill).

The NVIDIA script checks the same invariants plus `GDN-Attn(recurrent) = 2G`
(decode) and `GDN-Attn(chunk)` divisible by `G` (prefill).

Then cross-check the numbers themselves:

1. **Prefill GEMM TFLOPS must land in a narrow band** across all shapes.
   Scattered values mean the attribution is wrong, a single outlier is a real
   finding.
2. **Decode `ms/step` summed over the categories must equal the single-step
   TOTAL** to within a few µs.
3. **`lm_head` is the built-in bandwidth reference**: it is bf16 and goes
   through oneDNN in every XPU trace, so its GB/s is what the device actually
   delivers. Every quantized linear should be compared against it.

## Step 3 — Write `perf-report.md`

Put it next to the trace, following
[references/report-template.md](./references/report-template.md). Mandatory
content beyond the dense-model report:

- the **exact single-step tables** (`§3.3` prefill, `§3.4` decode). These are
  the whole point of the report: they are what lets a reader find a hot kernel
  by name and diff two runs operator by operator. Build them from
  `kernel-detail-*.txt`, **never** from the truncated `analyze_hybrid_trace.txt`
  tables. Requirements:
  - one row per kernel, with `cnt | ms | % | full kernel name including the
    ND-range`; `Dense-GEMM` expands by linear first, then by kernel signature
  - annotate what a kernel *is* when the name does not say so
    (`UT-transform 求逆`, `chunked causal conv1d`, `K=17408（down_proj 的输入）`)
  - collapse the long tail into one `其余 N 个` row, never drop it silently —
    the counts must still add up to the category total
- a **§3.5 fusion / kernel-rename table**: which ops are fused into which
  kernel, and — whenever an earlier report of the same model exists — a
  side-by-side of the renamed kernels. Without it the two reports look like
  they measured different models.
- a **GDN layer vs full-attention layer** table (ms/layer, and the attention
  core cost with the shared MLP removed)
- the **GDN break-even context length**: GDN's recurrent cost is O(1) while FMHA
  is O(L), but GDN's `in_proj_qkvz` is larger than `qkv_proj`. Report the `L` at
  which the hybrid actually starts paying off.
- the **GDN chunk-prefill table**, which is usually dominated by
  `ChunkInverseKernel`
- for an `out=1` trace: the **wall / busy / idle** table and the **idle
  attributed by following kernel**, plus the cross-validation against the
  `out>1` report's prefill window

Keep the generated `.txt` artifacts next to the report and list them in the
reproduction section — the report's tables must be re-derivable from them.

## Critical Pitfalls

Read [references/pitfalls.md](./references/pitfalls.md) before interpreting any
number.

| Trap | Effect if missed |
|---|---|
| XPU-Graph collapses kernel start times (57 % of kernels here) | union/idle/bubble metrics meaningless; use sum-of-durations only, and reflow before looking at any timeline |
| Reflowing without clamping the cursor to the previous group's end | a 33.9 ms group stamped 22.3 ms before the next one overlaps it; slices stack and step boundaries move |
| A decode graph replay is stamped *inside* the prefill window | prefill inflated by one decode step; 1153 kernels / 33.9 ms in the reference trace |
| Sorting events by `(ts, dur, name)` instead of `ts` alone | destroys capture order under collapsed timestamps → layer walker desyncs |
| Two sampler windows straddle one graph replay | those decode steps have fractional counts; use only the modal-size windows |
| AutoRound Hadamard matmuls are also called `gemm_kernel` | Dense-GEMM count inflated ~4x; discriminate on the work-group shape |
| `in_proj_ba` is sometimes scheduled before `in_proj_qkvz` | one GDN layer's two projections swap; detect `ba` by its single-work-group ND-range |
| The last `..._rms_norm_3` marker is the model's final norm, not a layer start | one phantom 65th layer that swallows `lm_head` |
| Decode layer 0's Hadamard/quant prologue is emitted at the *end* of the window | layer 0 looks like it has no quantization |
| Unterminated kernels at trace stop | two `gemm_kernel` events with `dur` ≈ 4.9·10^4 s dwarf the run |
| In **prefill** the walker merges `in_proj_qkvz` + `in_proj_ba` into one 96-call bucket | that row's GFLOP and the aggregate TFLOPS **double** (137.4 instead of 118.2); fix with `cluster_gemm_durations` |
| A fusion renames a marker kernel (`..._inc_ark_mxfp4_hadamard_quant_...`) | a whole category reads 0 and another doubles — fix the regex order in `bucket()`, not in the caller |
| Reading ND-range work-group size from a **decode** kernel | it is `{0; 0; 0}` inside a graph replay; only the global grid is valid. Quote prefill ND-ranges, or an `out=1` trace |
| Duration clustering applied to a **decode** window | at M=1 different shapes overlap in duration and merge into meaningless clusters; it is a prefill-only tool |
| Prompt longer than `max_num_batched_tokens` | chunked prefill emits several prefill forward passes; window 0 is one chunk and the rest are counted as decode steps. Detected via the prefill-FMHA kernel |
| Assuming `--batch 1` | scales tok/s and every attention byte count; now auto-detected from the decode KV-write ND-range |
| `out=1` trace fed to `analyze_hybrid_trace.py` | exits with "need at least one prefill and one decode window"; use `analyze_prefill_only.py` |

## How GEMM Attribution Works

unitrace records the kernel name and the ND-range, never M/N/K. Attribution is
therefore structural:

1. **Hadamard vs real linear** — a real linear has work-group shape
   `{128;4;1}` (prefill or `in_proj_ba`), `{32;2;8}` (`lm_head`), or
   `{64;8;1}` with grid[0] != 1 (decode). Everything else called `gemm_kernel`
   is an online rotation. Verified: 305 main GEMMs and 1168 (prefill) / 304
   (decode) rotations per forward. A BF16 checkpoint has no rotation at all, so
   under `--weight-dtype bf16` every `gemm_kernel` is a linear.
2. **Which linear** -- from the kernel that *consumes* the GEMM's output, not
   from its position in the layer: `SiLU` -> `gate_up`, input-norm ->
   `down_proj`, post-attn norm / gate -> `o_proj` or `gdn_out_proj`, a GDN
   kernel -> `in_proj*`, qk-norm/RoPE/KV-write -> `qkv_proj`, sampler ->
   `lm_head`. Positional assignment breaks whenever the graph partitioner
   rotates a forward pass, which it does: the BF16 decode window starts in the
   middle of layer 0's MLP and layer 0's attention block lands after the final
   norm.
3. **GDN vs full layer** -- from the segment's contents (`gdn::` kernels vs
   FMHA / KV-write). The MXFP4 build emits two distinct input-norm kernels for
   the two layer types (`..._rms_norm_3` / `..._rms_norm_mm_view_3`) but the
   BF16 build collapses them into one, so the marker name cannot be trusted.
4. **Duration clustering is the independent cross-check.** At `M = prompt_len`
   every prefill linear has a distinct and very tight duration (spread < 4 %),
   so `cluster_gemm_durations()` recovers the per-linear totals from the
   durations alone, per ND-range signature. Always run it (both
   `dump_kernels.py` and `analyze_prefill_only.py` print it) and reconcile:
   - clusters that the walker got right → the attribution is confirmed
   - `in_proj_qkvz` / `in_proj_ba` → the walker merges them, the clusters do
     not; **take the split from the clusters** and recompute TFLOPS
   - `gdn_out_proj` + `o_proj` share a shape (6144x5120) so they land in one
     cluster; split that one with the walker's counts
   - the cluster sums must add up to the `Dense-GEMM` total with zero error;
     if they do not, the attribution is broken and nothing below can be trusted
   - **prefill only.** At M=1 the shapes' durations overlap and the clusters
     are meaningless, so the scripts do not print them for a decode window.

Cost models (`Hq`/`Hkv` query/kv heads, `D` head_dim, `L` KV length, `w` bytes
per weight element = `1/2 + 1/32` for MXFP4, `2` for bf16):

```
GEMM         FLOPs = 2·M·N·K              Bytes = K·N·w + M·K·w + M·N·2
prefill FMHA FLOPs = 2·Hq·D·T² (causal)   Bytes = 4·T·Hq·D + 8·T·Hkv·D
decode  FMHA FLOPs = 4·B·Hq·D·L           Bytes = 4·B·Hkv·D·L
GDN decode   Bytes = 2·(Vh·Dv·Dk·4)       recurrent state, read + write, fp32
GDN conv     Bytes = 2·(conv_dim·K·2)     conv_dim = 2·Kh·Dk + Vh·Dv
```

## Adapting to Another Model, Device or Driver

1. Update `CFG` and `MXFP4_BYTES` in `hybrid_common.py`.
2. Re-check the Inductor marker names (`RE_NORM_IN`, `RE_GDN_GATED_NORM`, ...):
   torch.compile renumbers fused kernels when the graph changes. The reliable
   way to re-derive them is to dump one decode window's kernel sequence
   (`dump_kernels.py <trace> <k>`) and find the kernels whose per-forward counts
   are `G`, `F` or `L`.
3. **Re-derive the GEMM work-group shapes.** `MAIN_GEMM_LOCAL` /
   `DECODE_MAIN_LOCAL` separate a real linear from an AutoRound Hadamard
   rotation, and they are a property of the *device and driver*, not of the
   model. When they are stale the `Dense-GEMM` check fails and the sanity
   check prints the ND-range histogram for you:

   ```
   [BAD] Dense-GEMM             got=0      expected=305
   !! Dense-GEMM is off: the gemm_kernel work-group shapes ... do not hold
        local WG         grid[0]==1     cnt         ms
        (128, 4, 1)           False     304   1356.826
        (32, 2, 8)            False       1      2.939
   ```

   Read it as: the shapes whose counts are combinations of `G`, `F`, `L` and 1
   (here 304 = 5G+4F and 1 = `lm_head`) are the linears → put them in
   `MAIN_GEMM_LOCAL`; a shape launched thousands of times is the rotation.
   A decode-only shape that also appears with `grid[0]==1` (the tiny
   `in_proj_ba`) goes in `DECODE_MAIN_LOCAL`.
4. Re-run Step 2. Every line must be `OK` before any number is quoted.
5. Re-measure `--ref-tflops` / `--ref-bw` on the new device; never carry them
   over.
6. MoE hybrids additionally need routing / grouped-GEMM buckets in `bucket()`.
