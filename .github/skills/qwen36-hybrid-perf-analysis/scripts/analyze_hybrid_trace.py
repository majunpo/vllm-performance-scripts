#!/usr/bin/env python3
"""Category and per-kernel breakdown of a Qwen3.5/3.6 hybrid-attention vLLM
unitrace on Intel XPU.

Splits the run into sampler-delimited forward passes, separates GDN
(linear-attention) layers from full-attention layers, and prints an exact
kernel-level breakdown of one prefill step and one decode step.
"""

import argparse
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hybrid_common import (CATEGORY_ORDER, CFG, GDN_LINEARS, bucket, is_gemm,
                           is_main_gemm, layer_kinds, load, split_graph_blocks,
                           step_windows, ts_collapsed, walk_layers)


def short(name, width):
    """Keep head, tail and ND-range: sibling instantiations differ only in the
    last template arguments and in the launch geometry."""
    nd = ""
    lb = name.rfind("[")
    if lb > 0 and name.endswith("]"):
        nd, name = name[lb:], name[:lb]
    room = width - len(nd)
    if len(name) <= room:
        return name + nd
    head = max(18, room - 33)
    return f"{name[:head]}...{name[-30:]}{nd}"


def tally(evlist, wd="mxfp4"):
    cats = defaultdict(lambda: [0, 0.0])
    kern = defaultdict(lambda: [0, 0.0])
    for _, dur, name in evlist:
        c = bucket(name, wd)
        cats[c][0] += 1
        cats[c][1] += dur
        kern[(c, name)][0] += 1
        kern[(c, name)][1] += dur
    return cats, kern


def print_categories(cats, title, total=None, per_step=0):
    total = total or sum(v[1] for v in cats.values())
    ncall = sum(v[0] for v in cats.values())
    print(f"\n--- {title} ---")
    head = f"{'category':<24} {'cnt':>7} {'ms':>10} {'pct':>7}"
    if per_step:
        head += f" {'ms/step':>9} {'cnt/step':>9}"
    print(head)
    order = [c for c in CATEGORY_ORDER if c in cats] + \
            [c for c in cats if c not in CATEGORY_ORDER]
    for c in sorted(order, key=lambda x: -cats[x][1]):
        n, d = cats[c]
        line = f"{c:<24} {n:>7} {d/1e3:>10.3f} {d/total*100:>6.2f}%"
        if per_step:
            line += f" {d/per_step/1e3:>9.3f} {n/per_step:>9.1f}"
        print(line)
    line = f"{'TOTAL':<24} {ncall:>7} {total/1e3:>10.3f} {100.0:>6.2f}%"
    if per_step:
        line += f" {total/per_step/1e3:>9.3f} {ncall/per_step:>9.1f}"
    print(line)


def print_step_detail(evlist, label, maxname, gemm_tags=None, wd="mxfp4"):
    cats, kern = tally(evlist, wd)
    total = sum(v[1] for v in cats.values())
    if total <= 0:
        return
    gemm_by_tag = defaultdict(lambda: [0, 0.0])
    if gemm_tags:
        for i, tag in gemm_tags.items():
            gemm_by_tag[tag][0] += 1
            gemm_by_tag[tag][1] += evlist[i][1]

    print(f"\n{'='*118}")
    print(f"{label}  --  {total/1e3:.3f} ms, {sum(v[0] for v in cats.values())} "
          f"kernel launches")
    print(f"{'='*118}")
    print(f"{'category':<24} {'cnt':>5} {'ms':>9} {'pct':>7}   kernel "
          f"(searchable in the trace)")
    order = [c for c in CATEGORY_ORDER if c in cats] + \
            [c for c in cats if c not in CATEGORY_ORDER]
    for c in sorted(order, key=lambda x: -cats[x][1]):
        n, d = cats[c]
        print(f"{c:<24} {n:>5} {d/1e3:>9.4f} {d/total*100:>6.2f}%")
        if c == "Dense-GEMM" and gemm_by_tag:
            for tag, (tn, td) in sorted(gemm_by_tag.items(), key=lambda x: -x[1][1]):
                print(f"{'':<24} {tn:>5} {td/1e3:>9.4f} {td/total*100:>6.2f}%"
                      f"   +- {tag}")
        for (_, name), (kn, kd) in sorted(
                [kv for kv in kern.items() if kv[0][0] == c],
                key=lambda x: -x[1][1]):
            print(f"{'':<24} {kn:>5} {kd/1e3:>9.4f} {kd/total*100:>6.2f}%"
                  f"   {short(name, maxname)}")
    print(f"{'TOTAL':<24} {sum(v[0] for v in cats.values()):>5} "
          f"{total/1e3:>9.4f} {100.0:>6.2f}%")


def print_layer_split(evlist, layers, label):
    """Time attributed to GDN layers vs full-attention layers vs the head."""
    agg = defaultdict(lambda: [0, 0, 0.0])
    for kind, lo, hi in layers:
        agg[kind][0] += 1
        agg[kind][1] += hi - lo
        agg[kind][2] += sum(e[1] for e in evlist[lo:hi])
    total = sum(v[2] for v in agg.values())
    print(f"\n--- {label}: time by layer type ---")
    print(f"{'layer type':<16} {'layers':>7} {'kernels':>8} {'ms':>10} "
          f"{'pct':>7} {'ms/layer':>10}")
    for kind in ("gdn", "full", "head"):
        if kind not in agg:
            continue
        n, k, d = agg[kind]
        print(f"{kind:<16} {n:>7} {k:>8} {d/1e3:>10.3f} {d/total*100:>6.2f}% "
              f"{d/n/1e3:>10.4f}")
    print(f"{'TOTAL':<16} {sum(v[0] for v in agg.values()):>7} "
          f"{sum(v[1] for v in agg.values()):>8} {total/1e3:>10.3f} "
          f"{100.0:>6.2f}%")


def sanity(evlist, layers, tags, phase, cfg, wd="mxfp4"):
    """Counts that must be exact; a mismatch means the window is clipped."""
    n_full = sum(1 for k, _, _ in layers if k == "full")
    n_gdn = sum(1 for k, _, _ in layers if k == "gdn")
    cats, _ = tally(evlist, wd)
    want_full = cfg["layers"] // cfg["full_attention_interval"]
    want_gdn = cfg["layers"] - want_full
    n_gdn_lin = len(GDN_LINEARS[wd])
    n_quant = 0 if wd == "bf16" else n_gdn_lin * want_gdn + 4 * want_full
    checks = [
        ("gdn layers", n_gdn, want_gdn),
        ("full layers", n_full, want_full),
        ("Dense-GEMM", cats["Dense-GEMM"][0],
         n_gdn_lin * want_gdn + 4 * want_full + 1),
        ("Quantize(mxfp4)", cats["Quantize(mxfp4)"][0], n_quant),
        ("Quant-scale cast", cats["Quant-scale cast"][0], n_quant),
        ("Activation(SiLU)", cats["Activation(SiLU)"][0], cfg["layers"]),
        ("KVCache-Write", cats["KVCache-Write"][0], want_full),
        ("GDN-Norm/Gate", cats["GDN-Norm/Gate"][0], want_gdn),
        ("FullAttn-OutGate", cats["FullAttn-OutGate"][0], want_full),
        ("unattributed GEMM", sum(1 for t in tags.values() if t == "unattributed"), 0),
    ]
    print(f"\n--- sanity check: {phase} ---")
    ok = True
    for name, got, want in checks:
        flag = "OK " if got == want else "BAD"
        ok &= got == want
        print(f"  [{flag}] {name:<22} got={got:<6} expected={want}")
    if not ok:
        print("  !! counts are off -- the step window is clipped or the kernel "
              "markers changed; do not trust the numbers below")
    return ok


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("trace")
    p.add_argument("--decode-step", type=int, default=3,
                   help="which complete decode step to break down (default 3)")
    p.add_argument("--max-kernel-s", type=float, default=10.0)
    p.add_argument("--graph-block-min", type=int, default=64,
                   help="a timestamp shared by at least this many kernels is a "
                        "collapsed XPU-Graph replay")
    p.add_argument("--weight-dtype", choices=("mxfp4", "bf16"), default="mxfp4",
                   help="checkpoint weight dtype; bf16 has no Hadamard rotation "
                        "or activation quantisation and fuses the GDN in_proj")
    p.add_argument("--max-name", type=int, default=96)
    p.add_argument("--no-detail", action="store_true")
    args = p.parse_args()

    cfg = CFG
    wd = args.weight_dtype
    evs = load(args.trace, args.max_kernel_s)
    total_dur = sum(e[1] for e in evs)
    frac, worst = ts_collapsed(evs)

    print(f"trace  : {args.trace} ({os.path.getsize(args.trace)/2**20:.1f} MiB)")
    print(f"model  : {cfg['name']}  layers={cfg['layers']} "
          f"(gdn={cfg['layers']-cfg['layers']//cfg['full_attention_interval']}, "
          f"full={cfg['layers']//cfg['full_attention_interval']}, "
          f"interval={cfg['full_attention_interval']})  weights={wd}")
    print(f"kernels: {len(evs)}   sum-of-durations {total_dur/1e6:.3f} s")
    print(f"!! {frac*100:.1f}% of kernels share a start timestamp with another "
          f"(up to {worst} on one stamp).")
    print("!! XPU-Graph stamps every kernel of a replayed graph with the graph's "
          "submit time.\n!! Durations are valid, start times are not: all totals "
          "below are sums of durations,\n!! and idle/bubble analysis is "
          "deliberately not attempted.")

    wins = step_windows(evs)
    if len(wins) < 2:
        sys.exit("need at least one prefill and one decode window")
    print(f"\nforward passes detected: {len(wins)} "
          f"(window 0 = prefill, 1..{len(wins)-1} = decode)")

    # ---- prefill -------------------------------------------------------
    plo, phi = wins[0]
    pre, alien = split_graph_blocks(evs, plo, phi, args.graph_block_min)
    if alien:
        print(f"\n!! {len(alien)} kernels ({sum(e[1] for e in alien)/1e3:.3f} ms) "
              f"inside the prefill window carry a collapsed graph timestamp and "
              f"belong to a\n!! decode replay that was submitted while the "
              f"prefill was still running. They are excluded from the prefill.")
    players, ptags = walk_layers(pre, cfg, wd)
    sanity(pre, players, ptags, "PREFILL", cfg, wd)

    # ---- decode --------------------------------------------------------
    # pick a complete decode window: the modal kernel count
    sizes = {}
    for k, (lo, hi) in enumerate(wins[1:], start=1):
        sizes.setdefault(hi - lo, []).append(k)
    modal = max(sizes, key=lambda s: len(sizes[s]))
    good = sizes[modal]
    print(f"\ncomplete decode steps: {len(good)} of {len(wins)-1} "
          f"({modal} kernels each); incomplete windows "
          f"{[k for k in range(1, len(wins)) if k not in good]} are skipped "
          f"(graph replay straddles the sampler boundary)")

    dk = good[min(args.decode_step, len(good) - 1)]
    dlo, dhi = wins[dk]
    dec = list(evs[dlo:dhi])
    dlayers, dtags = walk_layers(dec, cfg, wd)
    sanity(dec, dlayers, dtags, f"DECODE step #{dk}", cfg, wd)

    # ---- aggregate -----------------------------------------------------
    pcats, _ = tally(pre, wd)
    ptotal = sum(v[1] for v in pcats.values())
    dagg = defaultdict(lambda: [0, 0.0])
    dtotal = 0.0
    for k in good:
        lo, hi = wins[k]
        for _, dur, name in evs[lo:hi]:
            c = bucket(name, wd)
            dagg[c][0] += 1
            dagg[c][1] += dur
            dtotal += dur

    print(f"\n{'='*118}\nOVERVIEW\n{'='*118}")
    print(f"prefill              : {ptotal/1e3:10.3f} ms")
    print(f"decode               : {dtotal/1e3:10.3f} ms over {len(good)} steps "
          f"-> {dtotal/len(good)/1e3:.3f} ms/step "
          f"({1e6*len(good)/dtotal:.2f} tok/s at bs=1)")

    print_categories(pcats, "PREFILL categories", ptotal)
    print_layer_split(pre, players, "PREFILL")
    print_categories(dagg, f"DECODE categories ({len(good)} complete steps)",
                     dtotal, per_step=len(good))
    print_layer_split(dec, dlayers, f"DECODE step #{dk}")

    if not args.no_detail:
        print_step_detail(pre, "SINGLE PREFILL STEP (exact)", args.max_name,
                          ptags, wd)
        print_step_detail(dec, f"SINGLE DECODE STEP #{dk} (exact)",
                          args.max_name, dtags, wd)


if __name__ == "__main__":
    main()
