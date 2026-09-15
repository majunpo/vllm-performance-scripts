#!/usr/bin/env python3
"""Convert a unitrace Chrome-trace into a compact, phase-annotated trace for ui.perfetto.dev.

Detects vLLM serving phases (weight load / init / idle / prefill / decode) from
device kernel activity and emits them as named slices on dedicated tracks,
alongside the GPU kernel events and derived counter tracks.
"""

import argparse
import gzip
import json
import os
import sys
from collections import Counter, defaultdict

import ijson

# --- kernel name signatures used to recognise what the device is doing -------
SIG_WEIGHT_COPY = "zeCommandListAppendMemoryCopy(M2D)"
SIG_PREFILL_ATTN = "cutlass::fmha::kernel::"
SIG_DECODE_ATTN = "_ZTSN6compat"
SIG_STEP_MARKER = "vllm::reshape_and_cache"
SIG_SAMPLER = "TopkToppSampler"
SIG_MOE_GEMM = "MainloopMoE"
SIG_LAYOUT = "BatchTransposeFunctor"


def bucket(name):
    if SIG_MOE_GEMM in name:
        return "MoE-GEMM"
    if SIG_PREFILL_ATTN in name or SIG_DECODE_ATTN in name or "fmha" in name:
        return "Attention"
    if "moe::" in name or "TopKGating" in name or "MoeGather" in name \
            or "RemapHiddenStates" in name or "RowsPerExpertCount" in name:
        return "MoE-Routing"
    if "gemm_kernel" in name or "GemmUniversal" in name:
        return "Dense-GEMM"
    if "rms_norm" in name or "RMSNorm" in name:
        return "Norm/RoPE"
    if SIG_SAMPLER in name:
        return "Sampling"
    if "MemoryCopy" in name or "Memcpy" in name:
        return "MemCopy"
    if "act_and_mul" in name:
        return "Activation"
    if "reshape_and_cache" in name:
        return "KVCache"
    return "Other"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("trace", help="input unitrace chrome trace json")
    p.add_argument("-o", "--output", help="output .json.gz (default: <trace>.perfetto.json.gz)")
    p.add_argument("--phase-gap-sec", type=float, default=2.0,
                   help="idle gap (seconds) that separates phases (default: 2.0)")
    p.add_argument("--weight-copy-min-bytes", type=int, default=1 << 20,
                   help="M2D copies at least this large are treated as weight loading")
    p.add_argument("--bin-ms", type=float, default=200.0,
                   help="counter track resolution in milliseconds (default: 200)")
    p.add_argument("--max-name-len", type=int, default=100,
                   help="truncate kernel names to keep the trace small")
    p.add_argument("--num-layers", type=int, default=0,
                   help="decoder layers; decode steps are derived from per-layer "
                        "KV writes. Autodetected from the trace when omitted")
    p.add_argument("--bubble-min-us", type=float, default=1000.0,
                   help="idle gaps at least this long get their own slice (default: 1000)")
    p.add_argument("--no-kernels", action="store_true",
                   help="emit only phase/counter tracks, drop individual kernels")
    p.add_argument("--no-reflow", action="store_true",
                   help="keep XPU-Graph's collapsed start times instead of "
                        "laying each graph's kernels out end-to-end")
    p.add_argument("--max-kernel-s", type=float, default=10.0,
                   help="drop device events longer than this; they are kernels "
                        "left unterminated when tracing stopped (default: 10)")
    return p.parse_args()


def copy_bytes(name):
    if not name.startswith("zeCommandListAppendMemoryCopy"):
        return 0
    lb, rb = name.rfind("["), name.rfind("]")
    if lb == -1 or rb < lb:
        return 0
    try:
        return int(name[lb + 1:rb])
    except ValueError:
        return 0


def reflow(evs, max_kernel_us):
    """Rebuild start times for kernels that XPU-Graph collapsed onto one stamp.

    A replayed graph is one Level-Zero command list, so unitrace gives every
    kernel in it the graph's submit time. The kernels do run back-to-back on
    the queue, so laying each group out end-to-end in capture order recovers a
    timeline that is correct in order and duration, and correct in absolute
    position to within one graph submission.
    """
    out = []
    dropped = 0
    i = 0
    n = len(evs)
    while i < n:
        j = i
        while j < n and evs[j][0] == evs[i][0]:
            j += 1
        cursor = evs[i][0]
        for ts, dur, name, is_w in evs[i:j]:
            if dur > max_kernel_us:
                dropped += 1
                continue
            out.append((cursor, dur, name, is_w))
            cursor += dur
        i = j
    return out, dropped


def load(path, weight_min, max_kernel_s, do_reflow):
    """Return (events, t0). events = list of (ts, dur, name, is_weight_copy)."""
    evs = []
    with open(path, "rb") as f:
        for ev in ijson.items(f, "traceEvents.item"):
            if ev.get("ph") != "X" or ev.get("cat") != "gpu_op":
                continue
            name = ev.get("name", "")
            evs.append((float(ev.get("ts", 0) or 0),
                        float(ev.get("dur", 0) or 0),
                        name,
                        name.startswith(SIG_WEIGHT_COPY)
                        and copy_bytes(name) >= weight_min))
    if not evs:
        sys.exit("no gpu_op events found -- is this a unitrace device trace?")
    evs.sort(key=lambda e: e[0])

    stamps = Counter(e[0] for e in evs)
    shared = sum(c for c in stamps.values() if c > 1) / len(evs)
    if shared > 0.2:
        print(f"XPU-Graph detected: {shared*100:.1f}% of kernels share a start "
              f"time (max {stamps.most_common(1)[0][1]} on one stamp).")
        if do_reflow:
            evs, dropped = reflow(evs, max_kernel_s * 1e6)
            print(f"  reflowed onto sequential start times"
                  + (f", dropped {dropped} unterminated kernel(s)" if dropped else ""))
        else:
            print("  --no-reflow given: timeline will show stacked slices.")

    return evs, evs[0][0]


def split_phases(evs, gap_us):
    """Split the timeline into contiguous activity segments separated by idle gaps."""
    segs = []
    cur_start = evs[0][0]
    cur = []
    prev_end = evs[0][0]
    for ts, dur, name, is_w in evs:
        if ts - prev_end > gap_us:
            segs.append((cur_start, prev_end, cur))
            cur_start, cur = ts, []
        cur.append((ts, dur, name, is_w))
        prev_end = max(prev_end, ts + dur)
    segs.append((cur_start, prev_end, cur))
    return segs


def label_segment(seg_evs, num_layers):
    """Classify an activity segment and return (label, detail dict)."""
    n_weight = sum(1 for e in seg_evs if e[3])
    weight_bytes = sum(copy_bytes(e[2]) for e in seg_evs if e[3])
    n_prefill = sum(1 for e in seg_evs if SIG_PREFILL_ATTN in e[2])
    n_steps = sum(1 for e in seg_evs if e[2].startswith(SIG_STEP_MARKER))
    n_layout = sum(1 for e in seg_evs if SIG_LAYOUT in e[2])
    n_moe = sum(1 for e in seg_evs if SIG_MOE_GEMM in e[2])

    if n_weight > len(seg_evs) * 0.3:
        return "Model weight load (H2D)", {
            "copies": n_weight, "bytes_GiB": round(weight_bytes / 2**30, 2)}
    if n_steps == 0 and n_moe == 0:
        return "Device init / allocation", {"events": len(seg_evs)}
    # capture/warmup replays many dummy forwards but never runs a real prefill
    if n_prefill == 0 and n_layout > 0:
        return "Warmup: weight relayout + shape probe + XPU-Graph capture", {
            "relayout_kernels": n_layout, "dummy_forwards": n_steps // num_layers,
            "moe_gemm": n_moe}
    return "Inference (serving requests)", {
        "prefill_attn_kernels": n_prefill, "decode_steps": n_steps // num_layers}


def sub_phases(seg_evs, t0):
    """Inside an inference segment, emit prefill / decode sub-slices."""
    marks = []
    for ts, dur, name, _ in seg_evs:
        if SIG_PREFILL_ATTN in name:
            marks.append((ts, ts + dur, "prefill"))
        elif name.startswith(SIG_STEP_MARKER):
            marks.append((ts, ts + dur, "decode"))
    if not marks:
        return []
    out = []
    kind = marks[0][2]
    start = marks[0][0]
    end = marks[0][1]
    count = 1
    for s, e, k in marks[1:]:
        # a new run starts when the kind flips or a >0.5s hole appears
        if k != kind or s - end > 500_000:
            out.append((start, end, kind, count))
            kind, start, count = k, s, 0
        end = max(end, e)
        count += 1
    out.append((start, end, kind, count))
    merged = []
    for start, end, kind, count in out:
        if merged and merged[-1][2] == kind and start - merged[-1][1] < 500_000:
            p = merged[-1]
            merged[-1] = (p[0], max(p[1], end), kind, p[3] + count)
        else:
            merged.append((start, end, kind, count))
    return merged


def bubbles(seg_evs, min_us):
    """Idle gaps inside an activity segment, as (start, end) pairs."""
    ivals = sorted((ts, ts + dur) for ts, dur, _, _ in seg_evs)
    gaps = []
    cur_end = ivals[0][1]
    for s, e in ivals[1:]:
        if s - cur_end >= min_us:
            gaps.append((cur_end, s))
        cur_end = max(cur_end, e)
    return gaps


def decode_steps(seg_evs, num_layers):
    """Per-decode-step boundaries derived from per-layer KV-write kernels."""
    marks = sorted(ts for ts, _, name, _ in seg_evs
                   if name.startswith(SIG_STEP_MARKER))
    starts = marks[::num_layers]
    return [(starts[i], starts[i + 1]) for i in range(len(starts) - 1)]


def detect_num_layers(evs, lo=2, hi=200):
    """Recover the decoder layer count from the KV-write period.

    Every forward emits one reshape_and_cache per layer, and the step boundary
    adds the lm_head GEMM and the sampler, so the gap sequence has a spike
    every num_layers entries. Multiples of the true period score just as well,
    hence the smallest candidate within 10% of the best.
    """
    ts = sorted(t for t, _, n, _ in evs if n.startswith(SIG_STEP_MARKER))
    gaps = [b - a for a, b in zip(ts, ts[1:])]
    if len(gaps) < 4 * lo:
        return None
    scores = {}
    for period in range(lo, min(hi, len(gaps) // 4) + 1):
        sums = [0.0] * period
        for i, g in enumerate(gaps):
            sums[i % period] += g
        total = sum(sums)
        if total <= 0:
            return None
        scores[period] = max(sums) / total * period
    if not scores or max(scores.values()) < 2.0:
        return None
    best = max(scores.values())
    return min(p for p, s in scores.items() if s >= 0.9 * best)


def main():
    args = parse_args()
    out_path = args.output or (os.path.splitext(args.trace)[0] + ".perfetto.json.gz")
    gap_us = args.phase_gap_sec * 1e6
    bin_us = args.bin_ms * 1000.0

    evs, t0 = load(args.trace, args.weight_copy_min_bytes, args.max_kernel_s,
                   not args.no_reflow)
    if not args.num_layers:
        args.num_layers = detect_num_layers(evs)
        if args.num_layers:
            print(f"decoder layers: {args.num_layers} (autodetected)")
        else:
            args.num_layers = 48
            print(f"decoder layers: {args.num_layers} (fallback; pass --num-layers)",
                  file=sys.stderr)
    segs = split_phases(evs, gap_us)

    PID = 1
    TID_PHASE, TID_SUB, TID_KERNEL = 10, 11, 12
    TID_BUBBLE, TID_STEP = 13, 14
    out = []

    def meta(name, key, val, tid=None):
        e = {"ph": "M", "pid": PID, "name": key, "args": {"name": val}}
        if tid is not None:
            e["tid"] = tid
        out.append(e)

    meta(None, "process_name", "vLLM XPU device timeline")
    meta(None, "thread_name", "PHASES", TID_PHASE)
    meta(None, "thread_name", "sub-phase (prefill/decode)", TID_SUB)
    meta(None, "thread_name", f"decode steps", TID_STEP)
    meta(None, "thread_name",
         f"GPU bubbles (idle >= {args.bubble_min_us/1000:g} ms)", TID_BUBBLE)
    meta(None, "thread_name", "GPU kernels", TID_KERNEL)

    print(f"{'phase':>6}  {'start(s)':>9} {'dur(s)':>8}  label")
    prev_end = None
    for i, (s, e, seg_evs) in enumerate(segs):
        if prev_end is not None and s - prev_end > gap_us:
            out.append({"ph": "X", "pid": PID, "tid": TID_PHASE,
                        "name": f"IDLE {(s - prev_end)/1e6:.1f}s "
                                f"(GPU completely idle)",
                        "ts": round(prev_end - t0, 3),
                        "dur": round(s - prev_end, 3),
                        "args": {"reason": "no device activity -- host-side work "
                                           "or waiting for client requests"}})
            print(f"{'gap':>6}  {(prev_end-t0)/1e6:9.2f} {(s-prev_end)/1e6:8.2f}"
                  f"  *** IDLE ***")
        label, detail = label_segment(seg_evs, args.num_layers)
        busy = sum(d for _, d, _, _ in seg_evs)
        detail["gpu_busy_s"] = round(busy / 1e6, 3)
        detail["util_pct"] = round(busy / max(e - s, 1) * 100, 2)
        out.append({"ph": "X", "pid": PID, "tid": TID_PHASE, "name": label,
                    "ts": round(s - t0, 3), "dur": round(e - s, 3),
                    "args": detail})
        print(f"{i:>6}  {(s-t0)/1e6:9.2f} {(e-s)/1e6:8.2f}  {label}  {detail}")

        for ss, se, kind, cnt in sub_phases(seg_evs, t0):
            nl = args.num_layers
            n = (f"prefill x{cnt//nl or 1} fwd" if kind == "prefill"
                 else f"decode {cnt//nl} steps")
            out.append({"ph": "X", "pid": PID, "tid": TID_SUB, "name": n,
                        "ts": round(ss - t0, 3), "dur": round(se - ss, 3),
                        "args": {"kernel_hits": cnt,
                                 "ms_per_step": round((se - ss) / max(cnt // nl, 1) / 1000, 3)}})

        # explicit idle slices make it obvious where the device is starved
        gaps = bubbles(seg_evs, args.bubble_min_us)
        for gs, ge in gaps:
            out.append({"ph": "X", "pid": PID, "tid": TID_BUBBLE,
                        "name": f"bubble {(ge-gs)/1000:.2f} ms",
                        "ts": round(gs - t0, 3), "dur": round(ge - gs, 3)})
        if gaps:
            print(f"{'':>6}  {'':>9} {'':>8}  bubbles >= "
                  f"{args.bubble_min_us/1000:g}ms: {len(gaps)}, "
                  f"{sum(ge-gs for gs, ge in gaps)/1e6:.3f}s")

        # one slice per decode step + a step-time counter
        for j, (ps, pe) in enumerate(decode_steps(seg_evs, args.num_layers)):
            ms = (pe - ps) / 1000.0
            out.append({"ph": "X", "pid": PID, "tid": TID_STEP,
                        "name": f"step {j} ({ms:.1f} ms)",
                        "ts": round(ps - t0, 3), "dur": round(pe - ps, 3),
                        "args": {"step": j, "ms": round(ms, 3)}})
            out.append({"ph": "C", "pid": PID, "name": "decode step time (ms)",
                        "ts": round(ps - t0, 3), "args": {"ms": round(ms, 3)}})
        prev_end = e

    # counter tracks: GPU busy % and per-bucket busy time
    bins = defaultdict(lambda: defaultdict(float))
    for ts, dur, name, _ in evs:
        bins[int((ts - t0) // bin_us)][bucket(name)] += dur
    for b in sorted(bins):
        ts = round(b * bin_us, 3)
        busy = sum(bins[b].values())
        out.append({"ph": "C", "pid": PID, "name": "GPU busy %",
                    "ts": ts, "args": {"busy": round(min(busy / bin_us, 1.0) * 100, 2)}})
        out.append({"ph": "C", "pid": PID, "name": "GPU time by category (ms)",
                    "ts": ts,
                    "args": {k: round(v / 1000, 3) for k, v in bins[b].items()}})

    if not args.no_kernels:
        for ts, dur, name, _ in evs:
            out.append({"ph": "X", "pid": PID, "tid": TID_KERNEL,
                        "name": name[:args.max_name_len],
                        "ts": round(ts - t0, 3), "dur": round(dur, 3),
                        "args": {"bucket": bucket(name)}})

    with gzip.open(out_path, "wt", compresslevel=6) as f:
        f.write('{"traceEvents":[\n')
        f.write(",\n".join(json.dumps(e) for e in out))
        f.write("\n]}\n")

    print(f"\nwrote {out_path} "
          f"({os.path.getsize(out_path)/2**20:.1f} MiB, {len(out)} events)")
    print("open https://ui.perfetto.dev/ and drag the file in")


if __name__ == "__main__":
    main()
