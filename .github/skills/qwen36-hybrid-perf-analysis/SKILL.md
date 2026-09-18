---
name: qwen36-hybrid-perf-analysis
description: 'Analyze vLLM inference traces of Qwen3.5/Qwen3.6 hybrid-attention models (GDN gated-delta-net linear attention interleaved with full attention) on Intel XPU, and produce a perf-report.md with a per-operator prefill/decode breakdown. Use when given a unitrace Chrome-trace JSON (python.<pid>.json) from a Qwen3.6-27B / Qwen3.5 / Qwen3-Next style run, or when asked to: break down GPU kernel time for a hybrid GDN + full-attention model, separate GDN layers from full-attention layers, compute per-shape GEMM TFLOPS and achieved bandwidth for MXFP4/MXFP8 linears, judge whether the quantized GEMM path has an XMX kernel at all or is running oneDNN weight decompression, analyze the GDN chunk prefill kernels or the recurrent decode kernels, find why decode is slow, or estimate the optimization headroom. Handles XPU-Graph collapsed timestamps, graph replays that land inside the prefill window, AutoRound online Hadamard rotations, and the in_proj_qkvz / in_proj_ba pair.'
argument-hint: '<path to unitrace python.<pid>.json> [--batch N] [--prompt-len N]'
---

# Qwen3.5/3.6 Hybrid-Attention Trace Analysis (Intel XPU)

Turns a unitrace device trace of a **GDN + full-attention hybrid** model into a
`perf-report.md` with: phase-split category timings, exact single-prefill and
single-decode kernel breakdowns, per-linear GEMM TFLOPS / achieved bandwidth,
separate GDN and full-attention cost models, and a quantified optimization list.

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
```

For an **unquantized BF16 checkpoint** add `--weight-dtype bf16` to both analysis
scripts. It switches the byte model to 2 B/element, drops the Hadamard and
quantize sections, and uses the 4-linear GDN decomposition (a BF16 build fuses
the GDN input projection, giving `4G + 4F + 1 = 257` Dense-GEMMs per forward
instead of `5G + 4F + 1 = 305`). Running the same workload in both precisions is
the strongest available check on any "the quantized path is slow" claim -- see
pitfall 5.

In this repo all four scripts are also symlinked into
`profile-scripts/qwen36-hybrid-perf-analysis/`, so they can be run straight from
the repo root without touching `.github/`. `hybrid_common.py` must stay next to
the other three — they import it from their own directory.

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

  Sanity-check the result the same way as always: the decode GB/s of every
  shape must land in one narrow band. If a precision guess is wrong, that shape
  either exceeds the physical bandwidth or falls far outside the band.
- **Marlin is W4A16**, so 4-bit buys bandwidth but not FLOPs. The script's
  per-backend TFLOPS table makes this visible directly: on the reference trace
  the FP8 path reaches 405.7 TFLOPS and Marlin only 189.0, a 2.15x ratio that
  matches the FP8:BF16 tensor-core ratio.
- **Timestamps are trustworthy.** CUPTI times every kernel individually even
  under CUDA Graph, so unlike the XPU side there is no reflow step and idle /
  bubble analysis is valid.

`--ref-tflops` / `--ref-bw` are the best numbers *measured on the same device*
by another trace (for `Intel(R) Graphics [0x674f]`: 500 TFLOPS prefill and
1035 GB/s decode, from the Qwen3-32B MXFP8 CUTLASS-SYCL run). They turn the
report's headroom estimate into a measurement instead of a guess.

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

- a **GDN layer vs full-attention layer** table (ms/layer, and the attention
  core cost with the shared MLP removed)
- the **GDN break-even context length**: GDN's recurrent cost is O(1) while FMHA
  is O(L), but GDN's `in_proj_qkvz` is larger than `qkv_proj`. Report the `L` at
  which the hybrid actually starts paying off.
- the **GDN chunk-prefill table**, which is usually dominated by
  `ChunkInverseKernel`

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

Cost models (`Hq`/`Hkv` query/kv heads, `D` head_dim, `L` KV length, `w` bytes
per weight element = `1/2 + 1/32` for MXFP4, `2` for bf16):

```
GEMM         FLOPs = 2·M·N·K              Bytes = K·N·w + M·K·w + M·N·2
prefill FMHA FLOPs = 2·Hq·D·T² (causal)   Bytes = 4·T·Hq·D + 8·T·Hkv·D
decode  FMHA FLOPs = 4·B·Hq·D·L           Bytes = 4·B·Hkv·D·L
GDN decode   Bytes = 2·(Vh·Dv·Dk·4)       recurrent state, read + write, fp32
GDN conv     Bytes = 2·(conv_dim·K·2)     conv_dim = 2·Kh·Dk + Vh·Dv
```

## Adapting to Another Hybrid Model

1. Update `CFG` and `MXFP4_BYTES` in `hybrid_common.py`.
2. Re-check the Inductor marker names (`M_NORM_IN_GDN`, `M_NORM_IN_FULL`, ...):
   torch.compile renumbers fused kernels when the graph changes. The reliable
   way to re-derive them is to dump one decode window's kernel sequence and
   find the two `..._rms_norm_*` variants whose counts are `G` and `F`.
3. Re-run Step 2. If a count is off by exactly the layer count, a marker moved.
4. MoE hybrids additionally need routing / grouped-GEMM buckets in `bucket()`.
