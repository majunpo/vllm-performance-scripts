#!/usr/bin/env python3
"""Top-down + operator-level breakdown and device bubble analysis of a unitrace
vLLM device trace.

Splits the timeline into phases (weight load / graph capture / inference), then
reports, for each inference run separately, the model-level category mix, the
hottest kernels, and the distribution of GPU idle gaps.
"""

import argparse
import os
import re
import sys
from collections import Counter, defaultdict

import ijson

SIG_WEIGHT_COPY = "zeCommandListAppendMemoryCopy(M2D)"
SIG_PREFILL_ATTN = "cutlass::fmha::kernel::"
SIG_STEP_MARKER = "vllm::reshape_and_cache"
SIG_LAYOUT = "BatchTransposeFunctor"
SIG_MOE_GEMM = "MainloopMoE"

# activation quantisation: the op name is generic, the narrow dtype is the tell
RE_QUANT = re.compile(r"_quant_|quantize|Float8_e4m3|Float8_e5m2|float_e2m1|fp4",
                      re.IGNORECASE)
# the block scales are cast/copied in a separate elementwise launch
RE_SCALE = re.compile(r"e8m0")
# torch.compile fuses RoPE with the q/k RMSNorm into one kernel, so norm and
# rope cannot be separated on the device and share a bucket
RE_ROPE = re.compile(r"rotary|rope|cat_index_select")
RE_NORM = re.compile(r"rms_norm|RMSNorm|layer_norm|LayerNorm")
RE_ACT = re.compile(r"act_and_mul|silu|gelu|swiglu")
RE_SAMPLE = re.compile(r"top_k_top_p|Sampler|argmax|sample_kernel|_bias_kernel")
RE_SCHED = re.compile(r"slot_mapping|block_table|_prepare_|_post_update|"
                      r"_combine_sampled|_get_num_sampled|_apply_write")
RE_ELTWISE = re.compile(r"Transpose|fill|Fill|copy_|CopyScalarFunc|"
                        r"element_?wise_kernel|IndexKernelFunctor|scatter_gather|"
                        r"reduce_kernel|gather_kernel")
# every forward ends by sampling a token, which makes an exact step delimiter
RE_STEP_END = re.compile(r"_gumbel_sample_kernel|argmax|top_k_top_p")
# first dimension of the ND-range, e.g. "...[SIMD32 {3500; 1; 1} {128; 1; 1}]"
RE_GRID0 = re.compile(r"\[SIMD\d+ \{(\d+);")


def bucket(name):
    if SIG_MOE_GEMM in name:
        return "MoE-GEMM(cutlass)"
    if SIG_PREFILL_ATTN in name or name.startswith("_ZTSN6compat") or "fmha" in name:
        return "Attention(FMHA)"
    if "moe::" in name or "TopKGating" in name or "MoeGather" in name \
            or "RemapHiddenStates" in name or "RowsPerExpertCount" in name:
        return "MoE-Routing/Aux"
    if "gemm_kernel" in name or "GemmUniversal" in name or "matmul" in name:
        return "Dense-GEMM"
    if "reshape_and_cache" in name:
        return "KVCache-Write"
    if RE_QUANT.search(name):
        return "Quantize(fp8/fp4)"
    if RE_SCALE.search(name):
        return "Quant-scale cast"
    if RE_ROPE.search(name) or RE_NORM.search(name):
        return "Norm/RoPE"
    if RE_ACT.search(name):
        return "Activation(SiLU)"
    if RE_SAMPLE.search(name):
        return "Sampling"
    if RE_SCHED.search(name):
        return "Sched/Prep"
    if "MemoryCopy" in name or "Memcpy" in name or "memcpy" in name:
        return "MemCopy"
    if RE_ELTWISE.search(name):
        return "Elementwise/Layout"
    return "Other"


def copy_bytes(name):
    lb, rb = name.rfind("["), name.rfind("]")
    if lb == -1 or rb < lb:
        return 0
    try:
        return int(name[lb + 1:rb])
    except ValueError:
        return 0


def load(path, weight_min, max_kernel_s):
    evs = []
    dropped = []
    max_kernel_us = max_kernel_s * 1e6
    with open(path, "rb") as f:
        for ev in ijson.items(f, "traceEvents.item"):
            if ev.get("ph") != "X" or ev.get("cat") != "gpu_op":
                continue
            name = ev.get("name", "")
            if name.startswith(SIG_WEIGHT_COPY) and copy_bytes(name) >= weight_min:
                continue  # model weight upload: one-off startup cost
            ts = float(ev.get("ts", 0) or 0)
            dur = float(ev.get("dur", 0) or 0)
            # Kernels still in flight when tracing stops never get a completion
            # timestamp, so unitrace emits one absurd duration per active L0
            # queue. Left in, a single such event dwarfs the whole run.
            if dur > max_kernel_us:
                dropped.append((ts, dur, name))
                continue
            evs.append((ts, dur, name))
    if not evs:
        sys.exit("no gpu_op events found")
    if dropped:
        print(f"WARNING: dropped {len(dropped)} unterminated kernel(s) "
              f"with dur > {max_kernel_s:g}s (in flight at trace stop):")
        for ts, dur, name in dropped[:3]:
            print(f"  dur={dur/1e6:.1f}s  {name[:70]}")
        if len(dropped) > 3:
            print(f"  ... and {len(dropped) - 3} more")
    # sort on ts alone: under XPU-Graph whole graphs share one timestamp and a
    # stable sort is what keeps them in capture order
    evs.sort(key=lambda e: e[0])
    return evs


def detect_num_layers(evs, lo=2, hi=200):
    """Recover the decoder layer count from the KV-write period.

    Every forward emits one reshape_and_cache per layer, and the step boundary
    adds the lm_head GEMM and the sampler, so the gap sequence has a spike
    every num_layers entries. Score each candidate by how much of the total gap
    time it concentrates on a single residue class; multiples of the true
    period score just as well, so take the smallest that comes close.
    """
    ts = sorted(ts for ts, _, n in evs if n.startswith(SIG_STEP_MARKER))
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
    if not scores:
        return None
    best = max(scores.values())
    if best < 2.0:                      # no periodic structure worth trusting
        return None
    return min(p for p, s in scores.items() if s >= 0.9 * best)


def find_runs(evs, num_layers):
    """Group events into inference runs.

    A run starts when a batch is prefilled and ends when the next prefill
    begins, so warmup and the measured benchmark are reported separately.
    """
    marks = [(ts, "p" if SIG_PREFILL_ATTN in n else "d")
             for ts, _, n in evs
             if n.startswith(SIG_STEP_MARKER) or SIG_PREFILL_ATTN in n]
    runs = []
    cur = None
    for ts, kind in marks:
        new_batch = kind == "p" and cur is not None and cur[2] > num_layers
        if cur is None or new_batch or ts - cur[1] > 500_000:
            if cur:
                runs.append(cur)
            cur = [ts, ts, 0, 0]
        cur[1] = ts
        cur[2 if kind == "d" else 3] += 1
    if cur:
        runs.append(cur)
    # prefill forwards also emit a KV-write per layer, so discount them
    out = [(s, e, d // num_layers - p // num_layers) for s, e, d, p in runs]
    return [r for r in out if r[2] > 0]


def bubbles(evlist):
    iv = sorted((ts, ts + dur) for ts, dur, _ in evlist)
    merged = []
    cs, ce = iv[0]
    for s, e in iv[1:]:
        if s <= ce:
            ce = max(ce, e)
        else:
            merged.append((cs, ce))
            cs, ce = s, e
    merged.append((cs, ce))
    busy = sum(e - s for s, e in merged)
    span = merged[-1][1] - merged[0][0]
    gaps = sorted(((merged[i + 1][0] - merged[i][1], merged[i][1])
                   for i in range(len(merged) - 1)), reverse=True)
    return span, busy, gaps


def ts_collapsed(evlist):
    """Fraction of events whose start time is shared with another event.

    Under XPU-Graph the whole captured graph is submitted as a single L0
    command list, and unitrace stamps every kernel inside it with the graph's
    submit time. Durations stay correct but start times do not, so any
    timeline-derived metric (union busy, idle, bubbles) becomes meaningless.
    """
    cnt = Counter(ts for ts, _, _ in evlist)
    shared = sum(c for c in cnt.values() if c > 1)
    return shared / len(evlist), cnt.most_common(1)[0][1]


def report_run(evlist, label, steps, topn, t0):
    print(f"\n{'='*78}\n{label}\n{'='*78}")
    buckets = defaultdict(lambda: [0, 0.0])
    kern = defaultdict(lambda: [0, 0.0, float("inf"), 0.0])
    for ts, dur, name in evlist:
        b = buckets[bucket(name)]
        b[0] += 1
        b[1] += dur
        k = kern[name]
        k[0] += 1
        k[1] += dur
        k[2] = min(k[2], dur)
        k[3] = max(k[3], dur)

    total = sum(v[1] for v in buckets.values())
    span, busy, gaps = bubbles(evlist)
    frac, worst = ts_collapsed(evlist)
    graphed = frac > 0.2

    print(f"wall span        : {span/1e6:.3f} s")
    if graphed:
        print(f"!!! {frac*100:.1f}% of kernels share a start time with another "
              f"(up to {worst} on one timestamp).")
        print("!!! XPU-Graph collapses every kernel in a replayed graph onto the "
              "graph's submit\n!!! timestamp. Durations are valid, start times "
              "are not, so union/idle/bubble\n!!! numbers below are meaningless "
              "and are suppressed. Use sum-of-durations.")
        print(f"GPU busy (sum)   : {total/1e6:.3f} s")
        print(f"GPU utilisation  : {total/span*100:.2f} % (sum/span; >100% means "
              "real overlap)")
    else:
        print(f"GPU busy (union) : {busy/1e6:.3f} s")
        print(f"GPU idle         : {(span-busy)/1e6:.3f} s")
        print(f"GPU utilisation  : {busy/span*100:.2f} %")
    if steps:
        busy_step = total if graphed else busy
        print(f"decode steps     : {steps}  -> {span/steps/1000:.3f} ms/step "
              f"wall, {busy_step/steps/1000:.3f} ms/step busy")

    print(f"\n--- model level (kernel time {total/1e6:.3f}s) ---")
    for b, (c, d) in sorted(buckets.items(), key=lambda x: -x[1][1]):
        print(f"{b:<22} cnt={c:>9} total_s={d/1e6:>9.3f} pct={d/total*100:6.2f}%")

    print(f"\n--- operator level: top {topn} kernels ---")
    for name, (c, d, mn, mx) in sorted(kern.items(), key=lambda x: -x[1][1])[:topn]:
        print(f"[{bucket(name):<18}] cnt={c:>7} tot_s={d/1e6:>8.3f} "
              f"avg_us={d/c:>9.3f} min={mn:>8.3f} max={mx:>9.3f}")
        print(f"    {name[:150]}")

    if graphed:
        print("\n--- device bubbles: SUPPRESSED (start times collapsed by "
              "XPU-Graph) ---")
        return

    print("\n--- device bubbles ---")
    print(f"total gaps: {len(gaps)}  total idle: {sum(g for g, _ in gaps)/1e6:.3f}s")
    for thr in (1000, 100, 10, 0):
        sel = [g for g, _ in gaps if g >= thr]
        per_step = f"  ({len(sel)/steps:.2f}/step)" if steps else ""
        print(f"  gaps >= {thr:>5}us : count={len(sel):>8} "
              f"total={sum(sel)/1e6:>8.3f}s{per_step}")
    print("  top 10 gaps (us):")
    for g, at in gaps[:10]:
        print(f"    {g:12.2f}  at +{(at-t0)/1e6:.3f}s")


def short_name(name, width):
    """Keep the head, the tail and the ND-range: sibling cutlass instantiations
    only differ in the last template arguments and in the launch geometry."""
    nd = ""
    lb = name.rfind("[")
    if lb > 0 and name.endswith("]"):
        nd, name = name[lb:], name[:lb]
    room = width - len(nd)
    if len(name) <= room:
        return name + nd
    head = max(20, room - 33)
    return f"{name[:head]}...{name[-30:]}{nd}"


def forward_start(evlist, run_ts, num_layers):
    """Rewind a run's start to the first kernel of its forward pass.

    find_runs anchors a run on the first KV write, but that sits in the middle
    of layer 0 -- the embedding norm, layer 0's qkv GEMM and its RoPE already
    ran. Walking back to just after the previous sampling recovers them. The
    cap keeps a preceding warmup phase from being swallowed when that run was
    never sampled.
    """
    i = next((k for k, ev in enumerate(evlist) if ev[0] >= run_ts), 0)
    cap = max(0, i - 4 * num_layers)
    j = i
    while j > cap and not RE_STEP_END.search(evlist[j - 1][2]):
        j -= 1
    # j == 0 means the trace itself starts here, which is a valid forward start
    if j == 0 or (j > cap and RE_STEP_END.search(evlist[j - 1][2])):
        return evlist[j][0]
    return run_ts


def step_windows(evlist):
    """Split the run into forward passes, one per sampling call.

    Every step ends by sampling a token, so the sampler kernel is an exact
    step delimiter: window 0 is the prefill, window k+1 is decode step k.
    """
    ends = [i for i, (_, _, n) in enumerate(evlist) if RE_STEP_END.search(n)]
    wins = []
    prev = 0
    for e in ends:
        wins.append((prev, e + 1))
        prev = e + 1
    return wins


def report_step_detail(evlist, lo, hi, label, maxname):
    """Exact per-kernel breakdown of one forward pass, with kernel names.

    One sampler-delimited window holds exactly one forward, so every count here
    is a whole number. The aggregate phase split uses the same windows.
    """
    cats = defaultdict(lambda: [0, 0.0])
    kern = defaultdict(lambda: [0, 0.0])
    for _, dur, name in evlist[lo:hi]:
        c = bucket(name)
        cats[c][0] += 1
        cats[c][1] += dur
        kern[(c, name)][0] += 1
        kern[(c, name)][1] += dur
    total = sum(v[1] for v in cats.values())
    if total <= 0:
        return

    print(f"\n--- {label} (exact counts, {total/1e3:.3f} ms) ---")
    print(f"{'category':<22} {'cnt':>5} {'ms':>9} {'pct':>7}   kernel")
    for cat, (c, d) in sorted(cats.items(), key=lambda x: -x[1][1]):
        print(f"{cat:<22} {c:>5} {d/1e3:>9.4f} {d/total*100:>6.2f}%")
        for (_, name), (kc, kd) in sorted(
                [kv for kv in kern.items() if kv[0][0] == cat],
                key=lambda x: -x[1][1]):
            print(f"{'':<22} {kc:>5} {kd/1e3:>9.4f} {kd/total*100:>6.2f}%   "
                  f"{short_name(name, maxname)}")
    print(f"{'TOTAL':<22} {sum(v[0] for v in cats.values()):>5} "
          f"{total/1e3:>9.4f} {100.0:>6.2f}%")


def report_single_step(evlist, batch, num_layers, want, maxname):
    wins = step_windows(evlist)
    if not wins:
        print("\n--- single-step detail: no sampler kernel found ---")
        return
    report_step_detail(evlist, *wins[0], "single PREFILL step", maxname)
    if len(wins) <= want + 1:
        print(f"\n--- single decode step: need >= {want + 1} decode steps, "
              f"found {len(wins) - 1} ---")
        return
    report_step_detail(evlist, *wins[want + 1],
                       f"single DECODE step #{want}", maxname)


def report_phase_split(evlist, batch, num_layers):
    """Category mix for prefill and decode separately.

    Prefer sampler-delimited windows, which contain complete forwards. Falling
    back to the KV-write marker is less exact because that kernel occurs in the
    middle of layer 0 and assigns the decode prefix to prefill.
    """
    phases = {"prefill": defaultdict(lambda: [0, 0.0]),
              "decode": defaultdict(lambda: [0, 0.0])}
    wins = step_windows(evlist)
    if len(wins) > 1:
        for step, (lo, hi) in enumerate(wins):
            phase = "prefill" if step == 0 else "decode"
            for _, dur, name in evlist[lo:hi]:
                slot = phases[phase][bucket(name)]
                slot[0] += 1
                slot[1] += dur
        steps = len(wins) - 1
        print(f"\nphase split      : sampler windows "
              f"(1 prefill + {steps} complete decode steps)")
    else:
        in_decode = False
        kv_writes = 0
        for _, dur, name in evlist:
            if name.startswith(SIG_STEP_MARKER):
                g = RE_GRID0.search(name)
                in_decode = bool(g) and int(g.group(1)) == batch
                kv_writes += in_decode
            slot = phases["decode" if in_decode else "prefill"][bucket(name)]
            slot[0] += 1
            slot[1] += dur
        steps = kv_writes // num_layers
        print("\nphase split      : KV-write fallback (layer-0 boundary is approximate)")

    for ph in ("prefill", "decode"):
        b = phases[ph]
        total = sum(v[1] for v in b.values())
        ncall = sum(v[0] for v in b.values())
        if total <= 0:
            continue
        extra = f", {steps} steps" if ph == "decode" else ""
        print(f"\n--- category split: {ph.upper()} "
              f"({total/1e6:.3f} s{extra}) ---")
        per = steps if ph == "decode" and steps else 0
        head = f"{'category':<22} {'cnt':>9} {'total_ms':>10} {'pct':>7}"
        if per:
            head += f" {'ms/step':>9} {'cnt/step':>9}"
        print(head)
        for cat, (c, d) in sorted(b.items(), key=lambda x: -x[1][1]):
            line = (f"{cat:<22} {c:>9} {d/1e3:>10.3f} "
                    f"{d/total*100:>6.2f}%")
            if per:
                line += f" {d/per/1000:>9.3f} {c/per:>9.1f}"
            print(line)
        line = f"{'TOTAL':<22} {ncall:>9} {total/1e3:>10.3f} {100.0:>6.2f}%"
        if per:
            line += f" {total/per/1000:>9.3f} {ncall/per:>9.1f}"
        print(line)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("trace")
    p.add_argument("--num-layers", type=int, default=0,
                   help="decoder layers, used to convert kernel counts to steps; "
                        "autodetected from the trace when omitted")
    p.add_argument("--expected-output-len", type=int, default=0,
                   help="if set, flags runs that captured fewer decode steps "
                        "than expected (i.e. a truncated trace)")
    p.add_argument("--weight-copy-min-bytes", type=int, default=1 << 20)
    p.add_argument("--max-kernel-s", type=float, default=10.0,
                   help="drop device events longer than this; they are kernels "
                        "left unterminated when tracing stopped")
    p.add_argument("--top", type=int, default=25)
    p.add_argument("--batch", type=int, default=0,
                   help="decode batch size; enables the prefill/decode category "
                        "split and the per-decode-step averages")
    p.add_argument("--step-index", type=int, default=3,
                   help="which decode step to break down kernel by kernel, "
                        "counted from the first one (default: 3, i.e. after "
                        "3 warmup steps)")
    p.add_argument("--max-name", type=int, default=110,
                   help="truncate kernel names in the single-step table")
    p.add_argument("--all-runs", action="store_true",
                   help="report every inference run, not just the longest")
    args = p.parse_args()

    evs = load(args.trace, args.weight_copy_min_bytes, args.max_kernel_s)
    t0 = evs[0][0]
    num_layers = args.num_layers
    if not num_layers:
        num_layers = detect_num_layers(evs)
        if num_layers:
            print(f"decoder layers    : {num_layers} (autodetected)")
        else:
            num_layers = 48
            print(f"decoder layers    : {num_layers} (fallback; pass --num-layers)",
                  file=sys.stderr)
    runs = find_runs(evs, num_layers)

    print(f"trace: {args.trace} ({os.path.getsize(args.trace)/2**20:.1f} MiB)")
    print(f"gpu kernel events (weight upload excluded): {len(evs)}")
    print(f"\n--- inference runs detected ---")
    print(f"{'#':>3} {'start(s)':>9} {'dur(s)':>8} {'steps':>7}  status")
    for i, (s, e, steps) in enumerate(runs):
        status = ""
        if args.expected_output_len:
            want = args.expected_output_len - 1
            if steps < want * 0.98:
                status = (f"TRUNCATED? captured {steps}/{want} steps "
                          f"({steps/want*100:.1f}%)")
            else:
                status = "complete"
        print(f"{i:>3} {(s-t0)/1e6:9.2f} {(e-s)/1e6:8.2f} {steps:>7}  {status}")

    if not runs:
        sys.exit("no decode activity found")

    targets = runs if args.all_runs else [max(runs, key=lambda r: r[2])]
    for s, e, steps in targets:
        i = runs.index((s, e, steps))
        s0 = forward_start(evs, s, num_layers)
        sel = [ev for ev in evs if s0 <= ev[0] <= e + 100_000]
        report_run(sel, f"run #{i}  start=+{(s-t0)/1e6:.2f}s  steps={steps}",
                   steps, args.top, t0)
        if args.batch:
            report_phase_split(sel, args.batch, num_layers)
            report_single_step(sel, args.batch, num_layers, args.step_index,
                               args.max_name)


if __name__ == "__main__":
    main()
