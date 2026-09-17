#!/usr/bin/env python3
"""Build a Perfetto timeline from a Qwen3.5/3.6 hybrid-attention unitrace.

Recovers usable start times from XPU-Graph's collapsed timestamps, then emits
phase, decode-step, decoder-layer (GDN vs full attention), kernel and counter
tracks. Drag the resulting .json.gz into https://ui.perfetto.dev/.
"""

import argparse
import gzip
import json
import os
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hybrid_common import CFG, bucket, load, split_graph_blocks, step_windows, walk_layers

PID = 1
TID_PHASE, TID_STEP, TID_LAYER, TID_KERNEL = 10, 11, 12, 13


def reflow(evs):
    """Lay each collapsed-timestamp group out end-to-end, monotonically.

    A replayed graph is one Level-Zero command list, so unitrace stamps every
    kernel in it with the graph's submit time. The kernels do run back-to-back
    on the queue, so laying each group out sequentially recovers a timeline
    that is correct in order and duration. The cursor never moves backwards,
    which matters here: one group's kernels sum to 33.9 ms while the next
    distinct stamp is only 22.3 ms later, so a naive reflow would overlap.
    """
    out = []
    end = evs[0][0]
    i, n = 0, len(evs)
    while i < n:
        j = i
        while j < n and evs[j][0] == evs[i][0]:
            j += 1
        cursor = max(evs[i][0], end)
        for _, dur, name in evs[i:j]:
            out.append((cursor, dur, name))
            cursor += dur
        end = cursor
        i = j
    return out


def runs(indices):
    """Split a sorted index list into contiguous runs."""
    out = []
    for i in indices:
        if out and i == out[-1][1] + 1:
            out[-1][1] = i
        else:
            out.append([i, i])
    return out


def slice_ev(tid, name, lo, hi, evs, args=None):
    ts = evs[lo][0]
    end = max(evs[k][0] + evs[k][1] for k in range(lo, hi))
    e = {"ph": "X", "pid": PID, "tid": tid, "name": name,
         "ts": round(ts, 3), "dur": round(end - ts, 3)}
    if args:
        e["args"] = args
    return e


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("trace")
    p.add_argument("-o", "--output",
                   help="output .json.gz (default: <trace>.perfetto.json.gz)")
    p.add_argument("--max-kernel-s", type=float, default=10.0)
    p.add_argument("--graph-block-min", type=int, default=64)
    p.add_argument("--bin-ms", type=float, default=5.0,
                   help="counter track resolution (default: 5)")
    p.add_argument("--max-name-len", type=int, default=110)
    p.add_argument("--no-kernels", action="store_true",
                   help="emit only phase/step/layer/counter tracks")
    p.add_argument("--no-reflow", action="store_true",
                   help="keep the collapsed start times (slices will stack)")
    args = p.parse_args()

    out_path = args.output or (os.path.splitext(args.trace)[0] + ".perfetto.json.gz")
    cfg = CFG
    raw = load(args.trace, args.max_kernel_s)

    stamps = Counter(e[0] for e in raw)
    shared = sum(c for c in stamps.values() if c > 1) / len(raw)
    print(f"XPU-Graph: {shared*100:.1f}% of kernels share a start stamp "
          f"(max {stamps.most_common(1)[0][1]} on one).")

    wins = step_windows(raw)
    if len(wins) < 2:
        sys.exit("need a prefill and at least one decode window")
    # identify the misplaced replay before reflow destroys the shared stamps
    _, alien = split_graph_blocks(raw, *wins[0], args.graph_block_min)
    alien_ts = {e[0] for e in alien}
    alien_idx = [i for i in range(*wins[0]) if raw[i][0] in alien_ts]

    evs = raw if args.no_reflow else reflow(raw)
    if not args.no_reflow:
        print(f"  reflowed onto sequential start times "
              f"(span {(evs[-1][0]+evs[-1][1]-evs[0][0])/1e6:.3f} s)")
    t0 = evs[0][0]
    evs = [(ts - t0, dur, name) for ts, dur, name in evs]

    out = []

    def meta(key, val, tid=None):
        e = {"ph": "M", "pid": PID, "name": key, "args": {"name": val}}
        if tid is not None:
            e["tid"] = tid
        out.append(e)

    meta("process_name", f"{cfg['name']} XPU device timeline")
    meta("thread_name", "PHASE", TID_PHASE)
    meta("thread_name", "forward passes", TID_STEP)
    meta("thread_name", "decoder layers (GDN / full attention)", TID_LAYER)
    meta("thread_name", "GPU kernels", TID_KERNEL)

    sizes = defaultdict(list)
    for k, (lo, hi) in enumerate(wins[1:], start=1):
        sizes[hi - lo].append(k)
    modal = max(sizes, key=lambda s: len(sizes[s]))
    complete = set(sizes[modal])

    print(f"\n{'window':>7} {'start(ms)':>10} {'dur(ms)':>9}  label")
    for k, (lo, hi) in enumerate(wins):
        keep = [i for i in range(lo, hi) if i not in alien_idx]
        busy = sum(evs[i][1] for i in keep)
        if k == 0:
            label = f"PREFILL ({busy/1e3:.1f} ms)"
            detail = {"kernels": len(keep), "ms": round(busy / 1e3, 3)}
        else:
            ok = k in complete
            label = (f"decode step {k} ({busy/1e3:.2f} ms)" if ok
                     else f"decode step {k} INCOMPLETE ({busy/1e3:.2f} ms)")
            detail = {"kernels": len(keep), "ms": round(busy / 1e3, 3),
                      "complete": ok}
        out.append(slice_ev(TID_STEP, label, lo, hi, evs, detail))
        out.append(slice_ev(TID_PHASE, "PREFILL" if k == 0 else "DECODE",
                            lo, hi, evs))
        print(f"{k:>7} {evs[lo][0]/1e3:>10.2f} "
              f"{(evs[hi-1][0]+evs[hi-1][1]-evs[lo][0])/1e3:>9.2f}  {label}")

        sub = [evs[i] for i in keep]
        layers, _ = walk_layers(sub, cfg)
        for n, (kind, s, e) in enumerate(layers):
            if kind == "head":
                name, ln = "lm_head + sampler", None
            else:
                ln = n
                name = f"L{n} {kind}"
            ms = sum(sub[j][1] for j in range(s, e)) / 1e3
            for rs, re in runs([keep[j] for j in range(s, e)]):
                out.append(slice_ev(
                    TID_LAYER, f"{name} ({ms:.3f} ms)", rs, re + 1, evs,
                    {"layer": ln, "type": kind, "ms": round(ms, 4),
                     "kernels": e - s}))

    if alien_idx:
        ms = sum(evs[i][1] for i in alien_idx) / 1e3
        for rs, re in runs(alien_idx):
            out.append(slice_ev(
                TID_PHASE,
                f"misplaced decode graph replay ({ms:.1f} ms) -- NOT prefill",
                rs, re + 1, evs,
                {"kernels": len(alien_idx), "ms": round(ms, 3),
                 "why": "a full XPU-Graph replay whose submit timestamp landed "
                        "inside the prefill; excluded from every prefill number"}))
        print(f"\n!! {len(alien_idx)} kernels ({ms:.3f} ms) inside the prefill "
              f"window are a misplaced graph replay; marked on the PHASE track.")

    bin_us = args.bin_ms * 1000.0
    bins = defaultdict(lambda: defaultdict(float))
    for ts, dur, name in evs:
        bins[int(ts // bin_us)][bucket(name)] += dur
    for b in sorted(bins):
        ts = round(b * bin_us, 3)
        out.append({"ph": "C", "pid": PID, "name": "GPU busy %", "ts": ts,
                    "args": {"busy": round(min(sum(bins[b].values()) / bin_us, 1.0) * 100, 2)}})
        out.append({"ph": "C", "pid": PID, "name": "GPU time by category (ms)",
                    "ts": ts,
                    "args": {k: round(v / 1000, 4) for k, v in bins[b].items()}})

    if not args.no_kernels:
        for ts, dur, name in evs:
            out.append({"ph": "X", "pid": PID, "tid": TID_KERNEL,
                        "name": name[:args.max_name_len],
                        "ts": round(ts, 3), "dur": round(dur, 3),
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
