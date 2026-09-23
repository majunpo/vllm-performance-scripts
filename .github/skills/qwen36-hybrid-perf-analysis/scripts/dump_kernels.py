#!/usr/bin/env python3
"""Untruncated per-kernel dump for one forward pass of a Qwen3.5/3.6 hybrid trace.

`analyze_hybrid_trace.py` truncates kernel names to keep its tables readable,
which makes them useless for searching the trace or diffing two runs operator by
operator.  This script prints the same breakdown with **full** kernel names plus
the GEMM attribution, and is the data source for the report's
"single PREFILL / DECODE step exact breakdown" sections.

It also cross-checks the walker's GEMM attribution by clustering the main GEMM
ND-range on duration: in prefill the walker merges `in_proj_qkvz` and
`in_proj_ba` into one bucket, which doubles that row's GFLOP and TFLOPS.
"""

import argparse
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hybrid_common import (bucket, cluster_gemm_durations, load,
                           split_graph_blocks, step_windows, ts_collapsed,
                           walk_layers)


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("trace")
    p.add_argument("window", nargs="?", default="prefill",
                   help="'prefill' (graph replays removed), 'all' (whole trace, "
                        "for an out=1 prefill-only run), or a decode step index")
    p.add_argument("--weight-dtype", choices=["mxfp4", "mxfp8", "bf16"],
                   default="mxfp4")
    p.add_argument("--max-kernel-s", type=float, default=10.0)
    p.add_argument("--graph-block-min", type=int, default=64)
    p.add_argument("--top", type=int, default=14,
                   help="kernels listed per category before collapsing the tail")
    return p.parse_args()


def select(ev, which, block_min):
    wins = step_windows(ev)
    if which == "all":
        return ev, "all (whole trace)"
    if not wins:
        sys.exit("no sampler kernel found -- use the 'all' window")
    if which == "prefill":
        own, alien = split_graph_blocks(ev, *wins[0], min_block=block_min)
        if alien:
            print(f"# note: {len(alien)} kernels "
                  f"({sum(d for _, d, _ in alien)/1e3:.3f} ms) removed from the "
                  f"prefill window (misplaced graph replay)")
        return own, "prefill"
    i = int(which)
    if not 0 <= i < len(wins):
        if len(wins) == 1:
            sys.exit("this trace has a single forward pass (out=1): use the "
                     "'all' window and analyze_prefill_only.py")
        sys.exit(f"window {i} out of range: the trace has {len(wins)} "
                 f"(0 = prefill, 1..{len(wins)-1} = decode)")
    lo, hi = wins[i]
    return ev[lo:hi], f"decode step #{i}"


def main():
    a = parse_args()
    ev = load(a.trace, a.max_kernel_s)
    sub, label = select(ev, a.window, a.graph_block_min)
    tot = sum(d for _, d, _ in sub)
    frac, mx = ts_collapsed(sub)

    print(f"# {os.path.basename(a.trace)}  window={label}  "
          f"kernels={len(sub)}  total={tot/1e3:.4f} ms")
    print(f"# ts collapse {frac*100:.1f}% (max {mx} on one stamp) -> start times "
          f"{'NOT ' if frac > 0.05 else ''}usable; inside a graph replay the "
          f"local work-group size reads {{0; 0; 0}}")

    _, tags = walk_layers(sub, weight_dtype=a.weight_dtype)
    gemm = defaultdict(lambda: [0, 0.0])
    for i, t in tags.items():
        gemm[t][0] += 1
        gemm[t][1] += sub[i][1]
    print("\n## GEMM attribution (layer walker)")
    for t, (c, d) in sorted(gemm.items(), key=lambda x: -x[1][1]):
        print(f"{c:5d} {d/1e3:11.4f} {d/tot*100:7.2f}%  {t}")

    clusters = cluster_gemm_durations(sub, a.weight_dtype) \
        if a.window in ("prefill", "all") else []
    if clusters:
        print("\n## duration clustering cross-check (one cluster per GEMM shape)")
        print("# the walker merges in_proj_qkvz + in_proj_ba in prefill; the")
        print("# clusters give the true per-linear split, and their sums must")
        print("# add up to the Dense-GEMM total exactly")
        for nd, groups in clusters:
            print(f"  {nd}")
            for ds in groups:
                n = len(ds)
                print(f"    n={n:4d} sum={sum(ds)/1e3:10.4f} ms  "
                      f"min={ds[0]/1e3:8.4f} med={ds[n//2]/1e3:8.4f} "
                      f"max={ds[-1]/1e3:8.4f}")

    cats = defaultdict(lambda: [0, 0.0])
    kern = defaultdict(lambda: [0, 0.0])
    for _, d, name in sub:
        c = bucket(name, a.weight_dtype)
        cats[c][0] += 1
        cats[c][1] += d
        kern[(c, name)][0] += 1
        kern[(c, name)][1] += d

    print("\n## per-kernel, FULL names")
    for cat in sorted(cats, key=lambda x: -cats[x][1]):
        c, d = cats[cat]
        print(f"\n[{cat}]  cnt={c}  ms={d/1e3:.4f}  pct={d/tot*100:.2f}%")
        rows = sorted(((k[1], v[0], v[1]) for k, v in kern.items()
                       if k[0] == cat), key=lambda r: -r[2])
        for name, kc, kd in rows[:a.top]:
            print(f"  {kc:5d} {kd/1e3:11.4f} {kd/tot*100:7.2f}%  {name}")
        if len(rows) > a.top:
            rest = rows[a.top:]
            print(f"  {sum(r[1] for r in rest):5d} "
                  f"{sum(r[2] for r in rest)/1e3:11.4f} "
                  f"{sum(r[2] for r in rest)/tot*100:7.2f}%  "
                  f"<{len(rest)} more kernels>")


if __name__ == "__main__":
    main()
