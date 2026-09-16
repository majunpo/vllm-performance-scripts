#!/usr/bin/env python3
"""Per-shape bandwidth / TFLOPS breakdown of the decode-phase dense GEMMs.

unitrace only records the SYCL kernel name plus its ND-range, never the GEMM
problem size, so each `gemm_kernel[...]` launch is attributed to a linear layer
by its position inside the decoder layer: every decode layer emits the fixed
sequence attn -> o_proj -> norm -> gate_up -> silu -> down_proj -> norm ->
qkv_proj, and the quantisation kernel that precedes each GEMM is skipped. The
shapes then come from the model config, not from the trace.
"""

import argparse
import re
import statistics
import sys
from collections import defaultdict

import ijson

DECODE_KV_WRITE = "vllm::reshape_and_cache_flash_strided_kernel"
# quantisation and its scale copies sit between a norm and the GEMM it feeds
GLUE = ("vllm::per_token_group_quant_8bit", "CopyScalarFunc",
    "elementwise_global_range_kernel", "gemm_zero_fill")
# first dimension of the ND-range, e.g. "...[SIMD32 {3500; 1; 1} {128; 1; 1}]"
RE_GRID0 = re.compile(r"\[SIMD\d+ \{(\d+);")


def is_gemm(name):
    return name.startswith("gemm_kernel") or "GemmUniversal" in name


def attn_kind(name):
    """Decode and prefill use different cutlass FMHA kernels; the class name is
    the reliable discriminator, a prefill forward can still launch a decode one."""
    if "ReduceSplitK" in name:
        return "reduce"
    if "XeFMHAFwdSplitKVKernel" in name or "DecodeProblemShape" in name:
        return "decode"
    if "XeFMHAFwdKernel" in name or "FMHA" in name or "fmha" in name:
        return "prefill"
    return None

# hidden, heads, kv_heads, head_dim, intermediate, layers, vocab -- Qwen3-32B
CFG = dict(hidden=5120, heads=64, kv_heads=8, head_dim=128,
           inter=25600, layers=64, vocab=151936)

MXFP8_BYTES = 1.0 + 1.0 / 32          # fp8 element + one uint8 E8M0 scale per 32
BF16_BYTES = 2.0


def layer_gemms(cfg, quantized=True):
    h, inter = cfg["hidden"], cfg["inter"]
    qkv_n = cfg["heads"] * cfg["head_dim"] + 2 * cfg["kv_heads"] * cfg["head_dim"]
    return {
        "qkv_proj":  (h, qkv_n, quantized),
        "o_proj":    (cfg["heads"] * cfg["head_dim"], h, quantized),
        "gate_up":   (h, 2 * inter, quantized),
        "down_proj": (inter, h, quantized),
        "lm_head":   (h, cfg["vocab"], False),
    }


def classify(prev_name):
    """Which linear a GEMM belongs to, from the kernel that fed it."""
    if "ReduceSplitK" in prev_name or "FMHA" in prev_name or "fmha" in prev_name:
        return "o_proj"
    if "rms_norm_0" in prev_name:
        return "gate_up"
    if "mul_silu" in prev_name:
        return "down_proj"
    if "rms_norm_2" in prev_name:
        return "qkv_proj"
    if "gather" in prev_name or "IndexKernelFunctor" in prev_name:
        return "lm_head"
    return None


def load(path, max_kernel_s):
    evs = []
    with open(path, "rb") as f:
        for ev in ijson.items(f, "traceEvents.item"):
            if ev.get("ph") != "X" or ev.get("cat") != "gpu_op":
                continue
            name = ev.get("name", "")
            dur = float(ev.get("dur", 0) or 0)
            if dur > max_kernel_s * 1e6 or name.startswith("zeCommandListAppendMemoryCopy"):
                continue
            evs.append((float(ev.get("ts", 0) or 0), dur, name))
    if not evs:
        sys.exit("no gpu_op events found")
    evs.sort(key=lambda e: e[0])
    return evs


def collect(evs, batch):
    """Walk the trace, binning decode GEMMs per linear and attention per phase.

    The KV-write kernel's ND-range gives the number of tokens in the forward
    that is currently running, which separates decode from prefill and, for
    prefill, gives the chunk length.
    """
    in_decode = False
    tokens = 0
    prev = ""
    out = defaultdict(list)
    pre = defaultdict(list)
    sigs = defaultdict(lambda: defaultdict(int))
    attn = []
    kv_writes = 0
    all_dur = 0.0
    for _, dur, name in evs:
        if name.startswith(DECODE_KV_WRITE):
            g = RE_GRID0.search(name)
            tokens = int(g.group(1)) if g else 0
            in_decode = tokens == batch
            kv_writes += in_decode
        if in_decode:
            all_dur += dur
        kind = attn_kind(name)
        if kind:
            attn.append((kind, in_decode, tokens, kv_writes, dur))
        if is_gemm(name):
            tag = classify(prev)
            if tag and in_decode:
                out[tag].append(dur)
                sigs[tag][name] += 1
            elif tag:
                pre[(tokens, tag)].append(dur)
        if not any(g in name for g in GLUE):
            prev = name
    return out, sigs, kv_writes, all_dur, attn, pre


def gemm_row(tag, m, k, n, quant, durs):
    wb = MXFP8_BYTES if quant else BF16_BYTES
    flops = 2.0 * m * n * k
    byts = k * n * wb + m * k * wb + m * n * BF16_BYTES
    med = statistics.median(durs)
    return (tag, m, k, n, len(durs), med, flops, byts)


def print_gemm_table(title, rows):
    print(f"\n--- {title} ---")
    hdr = (f"{'linear':<10} {'M':>6} {'K':>6} {'N':>7} {'calls':>6} "
           f"{'med_us':>9} {'GFLOP':>9} {'MB':>9} {'TFLOPS':>8} {'GB/s':>8}")
    print(hdr)
    print("-" * len(hdr))
    for tag, m, k, n, c, med, flops, byts in rows:
        print(f"{tag:<10} {m:>6} {k:>6} {n:>7} {c:>6} {med:>9.1f} "
              f"{flops/1e9:>9.3f} {byts/2**20:>9.1f} "
              f"{flops/med/1e6:>8.2f} {byts/med/1e3:>8.1f}")


def report_attention(attn, batch, prompt_len, layers, cfg):
    """FMHA cost model. GQA means K/V are read once per kv head, not per q head."""
    hq, hkv, d = cfg["heads"], cfg["kv_heads"], cfg["head_dim"]
    rows = []

    dec = [(kv, dur) for kind, in_dec, _, kv, dur in attn
           if kind == "decode" and in_dec]
    if dec:
        flops = byts = time = 0.0
        lens = []
        for kv_writes, dur in dec:
            step = (kv_writes - 1) // layers
            ctx = prompt_len + 1 + step        # keys visible to this step
            lens.append(ctx)
            flops += 4.0 * batch * hq * d * ctx
            byts += 4.0 * batch * hkv * d * ctx + 4.0 * batch * hq * d
            time += dur
        rows.append(("decode FMHA", len(dec), statistics.median(
            [dur for _, dur in dec]), statistics.fmean(lens), flops, byts, time))

    pre = [(t, dur) for kind, _, t, _, dur in attn if kind == "prefill"]
    chunks = sorted({t for t, _ in pre})
    for t in chunks:
        sel = [dur for tt, dur in pre if tt == t]
        # causal, and assumed to start at context 0 -- wrong under chunked prefill
        flops = 2.0 * hq * d * t * t * len(sel)
        byts = (2.0 * t * hq * d + 4.0 * t * hkv * d + 2.0 * t * hq * d) * len(sel)
        rows.append((f"prefill FMHA T={t}", len(sel), statistics.median(sel),
                     float(t), flops, byts, sum(sel)))

    if not rows:
        return
    print("\n--- attention (FMHA) ---")
    hdr = (f"{'kernel':<20} {'calls':>6} {'med_us':>9} {'ctx/T':>8} "
           f"{'GFLOP/call':>11} {'MB/call':>9} {'TFLOPS':>7} {'GB/s':>8}")
    print(hdr)
    print("-" * len(hdr))
    for label, n, avg, ctx, flops, byts, time in rows:
        print(f"{label:<20} {n:>6} {avg:>9.1f} {ctx:>8.0f} "
              f"{flops/n/1e9:>11.3f} {byts/n/2**20:>9.1f} "
              f"{flops/time/1e6:>7.1f} {byts/time/1e3:>8.1f}")

    red = [dur for kind, in_dec, _, _, dur in attn if kind == "reduce" and in_dec]
    if red:
        print(f"{'(split-K reduce)':<20} {len(red):>6} "
              f"{statistics.median(red):>9.1f} {'-':>8} {'-':>11} {'-':>9} "
              f"{'-':>7} {'-':>8}")
    if len(chunks) > 1:
        print("NOTE: several prefill chunk sizes seen -> chunked prefill; each "
              "chunk also attends to\n      earlier context, so the prefill "
              "FLOPs/bytes above are lower bounds.")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("trace")
    p.add_argument("--batch", type=int, default=8, help="decode batch size (M)")
    p.add_argument("--prompt-len", type=int, default=0,
                   help="prompt tokens per sequence; sets the KV length used "
                        "for the decode attention cost model")
    p.add_argument("--max-kernel-s", type=float, default=10.0)
    p.add_argument("--peak-bw", type=float, default=0.0,
                   help="device peak HBM bandwidth in GB/s, to print efficiency")
    p.add_argument("--weight-dtype", choices=("mxfp8", "bf16"), default="mxfp8",
                   help="linear weight dtype used by the bandwidth model")
    args = p.parse_args()

    evs = load(args.trace, args.max_kernel_s)
    runs, sigs, kv_writes, decode_dur, attn, pre = collect(evs, args.batch)
    if not runs:
        sys.exit("no decode GEMMs matched -- wrong --batch?")
    steps = kv_writes / CFG["layers"]

    quantized = args.weight_dtype == "mxfp8"
    shapes = layer_gemms(CFG, quantized)
    m = args.batch
    order = ["qkv_proj", "o_proj", "gate_up", "down_proj", "lm_head"]

    weights = ("MXFP8(e4m3, group=32, uint8 scale)" if quantized
               else "BF16")
    print(f"model: hidden={CFG['hidden']} inter={CFG['inter']} layers={CFG['layers']} "
          f"vocab={CFG['vocab']}  weights={weights}")
    print(f"decode M = {m}")

    for t in sorted({tk for tk, _ in pre if tk > 0}):
        rows = []
        for tag in order:
            if tag == "lm_head":
                continue        # logits are only computed for the last token
            durs = [d for d in pre.get((t, tag), []) if d > 0]
            if durs:
                k, n, quant = shapes[tag]
                rows.append(gemm_row(tag, t, k, n, quant, durs))
        if rows:
            print_gemm_table(f"PREFILL GEMM, chunk T={t}", rows)

    print("\n--- DECODE GEMM ---")
    hdr = (f"{'linear':<10} {'K':>6} {'N':>7} {'calls':>6} {'med_us':>9} "
           f"{'p10_us':>9} {'min_us':>8} {'GFLOP':>7} {'MB':>8} "
           f"{'TFLOPS':>7} {'GB/s':>8} {'GB/s@min':>9}")
    print(hdr)
    print("-" * len(hdr))

    step_time = 0.0
    step_bytes = 0.0
    step_flops = 0.0
    for tag in order:
        if tag not in runs:
            continue
        durs = sorted(d for d in runs[tag] if d > 0)
        if not durs:
            continue
        k, n, quant = shapes[tag]
        wb = BF16_BYTES if not quant else MXFP8_BYTES
        ab = BF16_BYTES if not quant else MXFP8_BYTES
        flops = 2.0 * m * n * k
        byts = k * n * wb + m * k * ab + m * n * BF16_BYTES
        med = statistics.median(durs)
        p10 = durs[len(durs) // 10]
        per_layer = CFG["layers"] if tag != "lm_head" else 1
        step_time += med * per_layer
        step_bytes += byts * per_layer
        step_flops += flops * per_layer
        print(f"{tag:<10} {k:>6} {n:>7} {len(durs):>6} {med:>9.1f} {p10:>9.1f} "
              f"{durs[0]:>8.1f} {flops/1e9:>7.3f} {byts/2**20:>8.1f} "
              f"{flops/med/1e6:>7.3f} {byts/med/1e3:>8.1f} "
              f"{byts/durs[0]/1e3:>9.1f}")

    print(f"\nper decode step (median kernel times x {CFG['layers']} layers + lm_head):")
    print(f"  GEMM time      : {step_time/1000:.1f} ms")
    print(f"  weights+act read: {step_bytes/2**30:.2f} GiB")
    print(f"  achieved BW    : {step_bytes/step_time/1e3:.1f} GB/s")
    print(f"  achieved FLOPS : {step_flops/step_time/1e6:.3f} TFLOPS")
    if args.peak_bw:
        print(f"  BW efficiency  : {step_bytes/step_time/1e3/args.peak_bw*100:.1f} % "
              f"of {args.peak_bw:g} GB/s")
    if steps:
        tot = decode_dur / steps
        print(f"  all decode kernels: {tot/1000:.1f} ms over {steps:.0f} steps "
              f"(GEMM = {step_time/tot*100:.0f} %)")

    if args.prompt_len:
        report_attention(attn, args.batch, args.prompt_len, CFG["layers"], CFG)
    else:
        print("\n--- attention (FMHA): skipped, pass --prompt-len ---")

    print("\nND-ranges seen per linear (unitrace kernel signature):")
    for tag in order:
        for s, c in sorted(sigs.get(tag, {}).items(), key=lambda x: -x[1]):
            nd = s[s.rfind("["):] if s.rfind("[") > 0 else s
            print(f"  {tag:<10} x{c:<6} {s[:40]:<40} {nd}")


if __name__ == "__main__":
    main()
