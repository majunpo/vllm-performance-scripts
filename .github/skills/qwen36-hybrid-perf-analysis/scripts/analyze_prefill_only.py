#!/usr/bin/env python3
"""Breakdown of a prefill-only (out=1) Qwen3.5/3.6 hybrid unitrace.

`analyze_hybrid_trace.py` splits a run on the sampler kernel and needs at least
one prefill *and* one decode window; an `out=1` run has a single forward pass and
makes it exit with "need at least one prefill and one decode window".

Such a trace is worth analysing on its own: with no XPU-Graph replay the start
timestamps are **not** collapsed, so wall-clock span, device busy ratio and the
per-gap idle attribution are all valid -- none of which an out>1 trace can give.
Use it both as the prefill reference and as a cross-check of the out>1 report's
prefill window (the two agree to ~0.2 % once the misplaced replay is removed).
"""

import argparse
import os
import statistics
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hybrid_common import (CFG, bucket, cluster_gemm_durations, derive,
                           detect_prompt_len, load, ts_collapsed, walk_layers)


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("trace")
    p.add_argument("--prompt-len", type=int, default=0,
                   help="prompt tokens; autodetected from the prefill KV-write "
                        "ND-range when omitted")
    p.add_argument("--weight-dtype", choices=["mxfp4", "mxfp8", "bf16"],
                   default="mxfp4")
    p.add_argument("--max-kernel-s", type=float, default=10.0)
    p.add_argument("--top", type=int, default=12)
    p.add_argument("--max-name", type=int, default=110)
    return p.parse_args()


def gap_analysis(ev, weight_dtype):
    """Wall span, busy ratio and idle attributed to the kernel that follows."""
    evs = sorted(ev, key=lambda e: e[0])
    span = max(ts + d for ts, d, _ in evs) - evs[0][0]
    by_cat = defaultdict(lambda: [0, 0.0])
    gaps = []
    cur = evs[0][0] + evs[0][1]
    for ts, d, name in evs[1:]:
        if ts > cur:
            g = ts - cur
            gaps.append((g, name))
            c = bucket(name, weight_dtype)
            by_cat[c][0] += 1
            by_cat[c][1] += g
        cur = max(cur, ts + d)
    return span, gaps, by_cat


def main():
    a = parse_args()
    ev = load(a.trace, a.max_kernel_s)
    tot = sum(d for _, d, _ in ev)
    frac, mx = ts_collapsed(ev)
    tokens = a.prompt_len or detect_prompt_len(ev)
    if not tokens:
        sys.exit("could not detect the prompt length (no prefill KV-write "
                 "kernel in this trace) -- pass --prompt-len")

    print(f"trace  : {os.path.basename(a.trace)}")
    print(f"model  : {CFG['name']}  layers={CFG['layers']}  "
          f"weights={a.weight_dtype}  prompt={tokens} tokens"
          f"{'' if a.prompt_len else ' (autodetected)'}")
    print(f"kernels: {len(ev)}   sum-of-durations {tot/1e6:.3f} s  "
          f"-> {tokens/(tot/1e6):.0f} tok/s")
    print(f"ts collapse: {frac*100:.1f}% share a stamp (max {mx} on one)"
          f"  -> start times are {'NOT ' if frac > 0.05 else ''}usable")
    if frac > 0.05:
        print("!! this trace contains graph replays; the wall/gap numbers below "
              "are meaningless, use make_hybrid_perfetto_trace.py instead")

    span, gaps, gap_by_cat = gap_analysis(ev, a.weight_dtype)
    idle = sum(g for g, _ in gaps)
    print(f"\nwall span : {span/1e3:.3f} ms   busy {tot/1e3:.3f} ms "
          f"({tot/span*100:.1f}%)   idle {idle/1e3:.3f} ms ({idle/span*100:.1f}%)")
    if gaps:
        med = statistics.median([g for g, _ in gaps])
        print(f"gaps      : {len(gaps)}  median {med/1e3:.4f} ms")
        print("\n--- idle attributed to the kernel that follows the gap ---")
        for c, (k, s) in sorted(gap_by_cat.items(), key=lambda x: -x[1][1]):
            print(f"  {c:<26} n={k:>5}  {s/1e3:8.4f} ms  {s/idle*100:5.1f}%")
        print("\n--- top 10 gaps (ms, kernel that follows) ---")
        for g, name in sorted(gaps, reverse=True)[:10]:
            print(f"   {g/1e3:8.4f}  {name[:a.max_name]}")

    cats = defaultdict(lambda: [0, 0.0])
    kern = defaultdict(lambda: [0, 0.0])
    for _, d, name in ev:
        c = bucket(name, a.weight_dtype)
        cats[c][0] += 1
        cats[c][1] += d
        kern[(c, name)][0] += 1
        kern[(c, name)][1] += d

    print("\n--- PREFILL categories ---")
    print(f"{'category':<28} {'cnt':>6} {'ms':>11} {'pct':>8}")
    for c in sorted(cats, key=lambda x: -cats[x][1]):
        k, d = cats[c]
        print(f"{c:<28} {k:>6} {d/1e3:>11.3f} {d/tot*100:>7.2f}%")
    print(f"{'TOTAL':<28} {len(ev):>6} {tot/1e3:>11.3f} {100.0:>7.2f}%")

    print("\n--- per-kernel detail (names truncated; use dump_kernels.py for "
          "the full names) ---")
    for c in sorted(cats, key=lambda x: -cats[x][1]):
        k, d = cats[c]
        print(f"{c:<28} {k:>6} {d/1e3:>11.3f} {d/tot*100:>7.2f}%")
        rows = sorted(((kk[1], v[0], v[1]) for kk, v in kern.items()
                       if kk[0] == c), key=lambda r: -r[2])
        for name, kc, kd in rows[:a.top]:
            print(f"{'':<28} {kc:>6} {kd/1e3:>11.4f} {kd/tot*100:>7.2f}%   "
                  f"{name[:a.max_name]}")
        if len(rows) > a.top:
            rest = rows[a.top:]
            print(f"{'':<28} {sum(r[1] for r in rest):>6} "
                  f"{sum(r[2] for r in rest)/1e3:>11.4f} "
                  f"{sum(r[2] for r in rest)/tot*100:>7.2f}%   "
                  f"<{len(rest)} more kernels>")

    layers, tags = walk_layers(ev, weight_dtype=a.weight_dtype)
    bytype = defaultdict(lambda: [0, 0, 0.0])
    for kind, lo, hi in layers:
        bytype[kind][0] += 1
        bytype[kind][1] += hi - lo
        bytype[kind][2] += sum(d for _, d, _ in ev[lo:hi])
    print("\n--- time by layer type ---")
    print(f"{'kind':<10} {'layers':>7} {'kernels':>9} {'ms':>11} {'pct':>8} "
          f"{'ms/layer':>10}")
    for kind in ("gdn", "full", "head"):
        if kind in bytype:
            lc, kc, d = bytype[kind]
            print(f"{kind:<10} {lc:>7} {kc:>9} {d/1e3:>11.3f} "
                  f"{d/tot*100:>7.2f}% {d/lc/1e3:>10.4f}")
    kinds = [k for k, _, _ in layers]
    exp_g = CFG["layers"] - CFG["layers"] // CFG["full_attention_interval"]
    print(f"sanity: gdn={kinds.count('gdn')} full={kinds.count('full')} "
          f"expected {exp_g}/{CFG['layers'] - exp_g};  tagged GEMMs={len(tags)}")

    shapes = derive(CFG, a.weight_dtype)
    per = defaultdict(list)
    for i, tag in tags.items():
        per[tag].append(ev[i][1])
    print(f"\n--- Dense-GEMM per linear (walker attribution, M={tokens}) ---")
    print(f"{'linear':<16} {'K':>7} {'N':>8} {'calls':>6} {'med us':>9} "
          f"{'GFLOP/call':>11} {'TFLOPS':>8} {'ms':>10}")
    gtot = gflop = 0.0
    for tag, ds in sorted(per.items(), key=lambda x: -sum(x[1])):
        K, N, _ = shapes[tag]
        m = 1 if tag == "lm_head" else tokens
        f = 2.0 * m * K * N / 1e9
        med = statistics.median(ds)
        print(f"{tag:<16} {K:>7} {N:>8} {len(ds):>6} {med:>9.1f} {f:>11.1f} "
              f"{f/med*1e3:>8.1f} {sum(ds)/1e3:>10.3f}")
        gtot += sum(ds)
        gflop += f * len(ds)
    print(f"{'TOTAL':<16} {'':>7} {'':>8} {sum(len(v) for v in per.values()):>6} "
          f"{'':>9} {gflop:>11.1f} {gflop/gtot*1e3:>8.1f} {gtot/1e3:>10.3f}")
    print("NOTE: in prefill the walker merges in_proj_qkvz and in_proj_ba into a "
          "single bucket,\n      which doubles its GFLOP/TFLOPS and the TOTAL "
          "row. Use the clustering below.")

    clusters = cluster_gemm_durations(ev, a.weight_dtype)
    if clusters:
        print("\n--- duration clustering cross-check (one cluster per GEMM shape) ---")
        for nd, groups in clusters:
            print(f"  {nd}")
            for ds in groups:
                n = len(ds)
                print(f"    n={n:4d} sum={sum(ds)/1e3:9.3f} ms  "
                      f"min={ds[0]/1e3:7.3f} med={ds[n//2]/1e3:7.3f} "
                      f"max={ds[-1]/1e3:7.3f}")


if __name__ == "__main__":
    main()
