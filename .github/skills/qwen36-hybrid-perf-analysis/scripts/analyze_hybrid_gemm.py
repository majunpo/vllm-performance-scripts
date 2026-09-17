#!/usr/bin/env python3
"""Per-shape GEMM / attention efficiency for a Qwen3.5/3.6 hybrid-attention
vLLM unitrace on Intel XPU.

unitrace records only the SYCL kernel name and its ND-range, never the GEMM
problem size, so every `gemm_kernel` launch is attributed to a named linear by
walking the fixed intra-layer kernel order (see hybrid_common.walk_layers).
Shapes then come from the model config, and FLOPs / bytes from the cost models
below.
"""

import argparse
import os
import statistics
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hybrid_common import (BF16_BYTES, CFG, MXFP4_BYTES, RE_FMHA_DECODE,
                           RE_FMHA_PREFILL, RE_FMHA_REDUCE, RE_GDN_CHUNK,
                           RE_GDN_DECODE, derive, load, nd_range,
                           split_graph_blocks, step_windows, walk_layers)

PREFILL_ORDER = ["in_proj_qkvz", "qkv_proj", "gate_up", "down_proj",
                 "gdn_out_proj", "o_proj", "in_proj_ba", "lm_head"]


def gemm_cost(m, k, n, quant):
    w = MXFP4_BYTES if quant else BF16_BYTES
    a = MXFP4_BYTES if quant else BF16_BYTES
    flops = 2.0 * m * n * k
    byts = k * n * w + m * k * a + m * n * BF16_BYTES
    return flops, byts


def collect(evlist, cfg):
    layers, tags = walk_layers(evlist, cfg)
    durs = defaultdict(list)
    per_layer_kind = defaultdict(lambda: defaultdict(float))
    kind_of = {}
    for kind, lo, hi in layers:
        for i in range(lo, hi):
            kind_of[i] = kind
    for i, tag in tags.items():
        durs[tag].append(evlist[i][1])
        per_layer_kind[kind_of.get(i, "?")][tag] += evlist[i][1]
    return layers, tags, durs, per_layer_kind


def gemm_table(title, rows, decode=False):
    print(f"\n--- {title} ---")
    if decode:
        hdr = (f"{'linear':<14} {'K':>6} {'N':>7} {'calls':>6} {'med_us':>9} "
               f"{'min_us':>9} {'MB_w':>8} {'GB/s':>9} {'GB/s@min':>9} "
               f"{'ms/step':>8}")
    else:
        hdr = (f"{'linear':<14} {'M':>6} {'K':>6} {'N':>7} {'calls':>6} "
               f"{'med_us':>9} {'GFLOP':>9} {'MB':>8} {'TFLOPS':>8} {'GB/s':>9} "
               f"{'ms/step':>8}")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(r)
    print("-" * len(hdr))


def report_gemm(durs, cfg, shapes, m, phase, layers_of):
    rows = []
    tot_t = tot_f = tot_b = 0.0
    quant_t = quant_b = 0.0
    for tag in PREFILL_ORDER:
        if tag not in durs or not durs[tag]:
            continue
        k, n, quant = shapes[tag]
        mm = 1 if tag == "lm_head" else m
        flops, byts = gemm_cost(mm, k, n, quant)
        d = sorted(durs[tag])
        med = statistics.median(d)
        cnt = len(d)
        tot_t += sum(d)
        tot_f += flops * cnt
        tot_b += byts * cnt
        if quant:
            quant_t += sum(d)
            quant_b += byts * cnt
        if phase == "decode":
            rows.append(f"{tag:<14} {k:>6} {n:>7} {cnt:>6} {med:>9.2f} "
                        f"{d[0]:>9.2f} {byts/1e6:>8.2f} {byts/med/1e3:>9.1f} "
                        f"{byts/d[0]/1e3:>9.1f} {sum(d)/1e3:>8.3f}")
        else:
            rows.append(f"{tag:<14} {mm:>6} {k:>6} {n:>7} {cnt:>6} {med:>9.1f} "
                        f"{flops/1e9:>9.1f} {byts/1e6:>8.1f} "
                        f"{flops/med/1e6:>8.1f} {byts/med/1e3:>9.1f} "
                        f"{sum(d)/1e3:>8.3f}")
    gemm_table(f"{phase.upper()} Dense-GEMM per shape (M={m})", rows,
               decode=(phase == "decode"))
    print(f"aggregate       : {tot_t/1e3:.3f} ms, {tot_f/1e12:.2f} TFLOP, "
          f"{tot_b/2**30:.3f} GiB")
    if phase == "decode":
        print(f"                  {tot_b/tot_t/1e3:.1f} GB/s  (memory bound)")
        print(f"MXFP4 linears   : {quant_t/1e3:.3f} ms, {quant_b/2**30:.3f} GiB "
              f"-> {quant_b/quant_t/1e3:.1f} GB/s")
    else:
        print(f"                  {tot_f/tot_t/1e6:.1f} TFLOPS  (compute bound)")
        print(f"MXFP4 linears   : {quant_t/1e3:.3f} ms of {tot_t/1e3:.3f} ms")
    return tot_t, tot_f, tot_b


def report_hadamard(evlist, tags, cfg):
    from hybrid_common import is_gemm, is_main_gemm
    groups = defaultdict(lambda: [0, 0.0])
    for i, (_, dur, name) in enumerate(evlist):
        if is_gemm(name) and not is_main_gemm(name):
            nd = nd_range(name)
            groups[nd[0] if nd else None][0] += 1
            groups[nd[0] if nd else None][1] += dur
    tot = sum(v[1] for v in groups.values())
    cnt = sum(v[0] for v in groups.values())
    print(f"\n--- online Hadamard rotation (AutoRound `allow_online_rotation`) ---")
    print(f"{'nd-range grid':<20} {'calls':>7} {'ms':>9} {'us/call':>9}")
    for g, (c, d) in sorted(groups.items(), key=lambda x: -x[1][1]):
        print(f"{str(g):<20} {c:>7} {d/1e3:>9.4f} {d/c:>9.2f}")
    print(f"{'TOTAL':<20} {cnt:>7} {tot/1e3:>9.4f}")
    return tot


def report_gdn(evlist, cfg, batch, tokens, phase):
    """GDN recurrent-state cost model.

    decode : one step updates S[v_heads, v_dim, k_dim] in fp32 per layer, so the
             kernel must read and write the whole state.
    prefill: the chunked path materialises intermediates; reported as time only.
    """
    gv, gvd, gkd = cfg["gdn_v_heads"], cfg["gdn_v_dim"], cfg["gdn_k_dim"]
    sb = cfg["gdn_state_dtype_bytes"]
    state = gv * gvd * gkd * sb
    conv_state = (2 * cfg["gdn_k_heads"] * cfg["gdn_k_dim"] + gv * gvd) \
        * cfg["gdn_conv_kernel"] * BF16_BYTES
    rows = defaultdict(lambda: [0, 0.0])
    for _, dur, name in evlist:
        if RE_GDN_DECODE.search(name) or RE_GDN_CHUNK.search(name):
            base = name[:name.rfind("[")] if name.rfind("[") > 0 else name
            key = base.split("<")[0].split("(")[0]
            rows[key][0] += 1
            rows[key][1] += dur
    if not rows:
        return 0.0
    tot = sum(v[1] for v in rows.values())
    print(f"\n--- GDN (gated delta net) kernels, {phase} ---")
    print(f"{'kernel':<34} {'calls':>6} {'ms':>9} {'us/call':>9} {'pct':>7}   note")
    notes = {
        "gdn::gated_delta_rule_kernel":
            f"state {2*state/1e6:.2f} MB rd+wr/layer",
        "gdn::causal_conv1d_kernel":
            f"conv state {2*conv_state/1e3:.1f} KB rd+wr/layer",
        "gdn::ChunkInverseKernel": "UT-transform inverse (serial, 32-wide WG)",
        "gdn::ChunkComputeWUKernel": "W/U projection",
        "gdn::ChunkFwdOKernel": "intra+inter chunk output",
        "gdn::ChunkComputeAKernel": "A = decay-weighted k^T k",
        "gdn::tiled_kernel_launcher": "chunked causal conv1d",
        "gdn::ChunkPrepareKernel": "cumulative decay",
        "gdn::chunk_update_states_kernel": "carry state between chunks",
    }
    for k, (c, d) in sorted(rows.items(), key=lambda x: -x[1][1]):
        print(f"{k:<34} {c:>6} {d/1e3:>9.4f} {d/c:>9.2f} {d/tot*100:>6.2f}%   "
              f"{notes.get(k, '')}")
    print(f"{'TOTAL':<34} {sum(v[0] for v in rows.values()):>6} {tot/1e3:>9.4f}")
    if "gdn::gated_delta_rule_kernel" in rows:
        c, d = rows["gdn::gated_delta_rule_kernel"]
        print(f"  delta-rule achieved BW : {2*state*c/d/1e3:.1f} GB/s "
              f"(state = {state/1e6:.2f} MB/layer, fp32)")
    return tot


def report_attention(evlist, cfg, batch, tokens, ctx, phase):
    hq, hkv, d = cfg["heads"], cfg["kv_heads"], cfg["head_dim"]
    rd = [dur for _, dur, name in evlist if RE_FMHA_REDUCE.search(name)]
    fm = [(dur, name) for _, dur, name in evlist
          if not RE_FMHA_REDUCE.search(name)
          and (RE_FMHA_PREFILL.search(name) or RE_FMHA_DECODE.search(name))]
    if not fm:
        return
    print(f"\n--- full attention (cutlass FMHA), {phase} ---")
    hdr = (f"{'kernel':<22} {'calls':>6} {'us/call':>9} {'ctx/T':>7} "
           f"{'GFLOP/call':>11} {'MB/call':>9} {'TFLOPS':>8} {'GB/s':>9}")
    print(hdr)
    print("-" * len(hdr))
    pre = [dur for dur, n in fm if RE_FMHA_PREFILL.search(n)]
    dec = [dur for dur, n in fm if RE_FMHA_DECODE.search(n)]
    if pre:
        t = tokens
        flops = 2.0 * hq * d * t * t          # causal QK^T + PV
        byts = 2.0 * t * hq * d * 2 + 2.0 * 2 * t * hkv * d * 2
        tot = sum(pre)
        print(f"{'prefill FMHA':<22} {len(pre):>6} {tot/len(pre):>9.2f} "
              f"{t:>7} {flops/1e9:>11.2f} {byts/1e6:>9.2f} "
              f"{flops*len(pre)/tot/1e6:>8.1f} {byts*len(pre)/tot/1e3:>9.1f}")
    if dec:
        flops = 4.0 * batch * hq * d * ctx
        byts = 2.0 * 2 * batch * hkv * d * ctx      # K and V, bf16
        tot = sum(dec)
        print(f"{'decode FMHA split-KV':<22} {len(dec):>6} {tot/len(dec):>9.2f} "
              f"{ctx:>7} {flops/1e9:>11.3f} {byts/1e6:>9.2f} "
              f"{flops*len(dec)/tot/1e6:>8.2f} {byts*len(dec)/tot/1e3:>9.1f}")
    if rd:
        tot = sum(rd)
        print(f"{'  split-K reduce':<22} {len(rd):>6} {tot/len(rd):>9.2f} "
              f"{'-':>7} {'-':>11} {'-':>9} {'-':>8} {'-':>9}")
        if dec:
            print(f"  reduce overhead: +{tot/sum(dec)*100:.0f}% on top of the "
                  f"FMHA kernel -> effective "
                  f"{2.0*2*batch*hkv*d*ctx*len(dec)/(sum(dec)+tot)/1e3:.1f} GB/s")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("trace")
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--prompt-len", type=int, default=0,
                   help="prompt tokens; autodetected from the prefill KV-write "
                        "ND-range when omitted")
    p.add_argument("--decode-step", type=int, default=3)
    p.add_argument("--max-kernel-s", type=float, default=10.0)
    p.add_argument("--graph-block-min", type=int, default=64)
    p.add_argument("--ref-bw", type=float, default=0.0,
                   help="reference achievable HBM bandwidth in GB/s, e.g. the "
                        "best GEMM observed on the same device")
    p.add_argument("--ref-tflops", type=float, default=0.0,
                   help="reference achievable dense TFLOPS on the same device")
    args = p.parse_args()

    cfg = CFG
    shapes = derive(cfg)
    evs = load(args.trace, args.max_kernel_s, verbose=False)
    wins = step_windows(evs)
    if len(wins) < 2:
        sys.exit("need a prefill and at least one decode window")

    pre, alien = split_graph_blocks(evs, *wins[0], args.graph_block_min)
    tokens = args.prompt_len
    if not tokens:
        for _, _, n in pre:
            if "reshape_and_cache" in n:
                nd = nd_range(n)
                if nd:
                    tokens = nd[0][0]
                    break
    sizes = {}
    for k, (lo, hi) in enumerate(wins[1:], start=1):
        sizes.setdefault(hi - lo, []).append(k)
    modal = max(sizes, key=lambda s: len(sizes[s]))
    good = sizes[modal]
    dk = good[min(args.decode_step, len(good) - 1)]
    dec = list(evs[wins[dk][0]:wins[dk][1]])

    w = "MXFP4 (e2m1, group=32, uint8 E8M0 scale) weights + MXFP4 activations"
    print(f"model   : {cfg['name']}  hidden={cfg['hidden']} inter={cfg['inter']} "
          f"layers={cfg['layers']} vocab={cfg['vocab']}")
    print(f"          gdn: {cfg['gdn_v_heads']}v x {cfg['gdn_v_dim']} / "
          f"{cfg['gdn_k_heads']}k x {cfg['gdn_k_dim']}, conv={cfg['gdn_conv_kernel']}"
          f"   full: {cfg['heads']}q/{cfg['kv_heads']}kv x {cfg['head_dim']}"
          f"{', output-gated' if cfg['attn_output_gate'] else ''}")
    print(f"quant   : {w}  ({MXFP4_BYTES:.5f} B/element); lm_head kept in bf16")
    print(f"workload: prompt={tokens} tokens, batch={args.batch}, "
          f"decode step #{dk} of {len(good)} complete steps")
    if alien:
        print(f"note    : {len(alien)} kernels "
              f"({sum(e[1] for e in alien)/1e3:.3f} ms) removed from the prefill "
              f"window (collapsed graph replay)")

    _, ptags, pdurs, _ = collect(pre, cfg)
    _, dtags, ddurs, _ = collect(dec, cfg)

    pt, pf, pb = report_gemm(pdurs, cfg, shapes, tokens, "prefill", cfg["layers"])
    ph = report_hadamard(pre, ptags, cfg)
    print(f"  Hadamard + quantise overhead is {ph/pt*100:.1f} % of prefill GEMM time")

    dt, df, db = report_gemm(ddurs, cfg, shapes, args.batch, "decode", cfg["layers"])
    dh = report_hadamard(dec, dtags, cfg)

    report_gdn(pre, cfg, args.batch, tokens, "PREFILL")
    report_gdn(dec, cfg, args.batch, tokens, "DECODE (one step)")
    report_attention(pre, cfg, args.batch, tokens, tokens, "PREFILL")
    report_attention(dec, cfg, args.batch, tokens, tokens + 1 + dk,
                     f"DECODE step #{dk}")

    ptot = sum(e[1] for e in pre)
    dtot = sum(e[1] for e in dec)
    print(f"\n--- roll-up ---")
    print(f"prefill total       : {ptot/1e3:.3f} ms  "
          f"({tokens/ptot*1e6:.0f} tok/s), GEMM = {pt/ptot*100:.1f} %")
    print(f"decode step total   : {dtot/1e3:.3f} ms  "
          f"({args.batch*1e6/dtot:.2f} tok/s), GEMM = {dt/dtot*100:.1f} %")
    if args.ref_tflops:
        ideal = pf / (args.ref_tflops * 1e12) * 1e6
        print(f"prefill GEMM at {args.ref_tflops:g} TFLOPS would take "
              f"{ideal/1e3:.1f} ms (now {pt/1e3:.1f} ms, "
              f"{pt/ideal:.2f}x slower) -> prefill {(ptot-pt+ideal)/1e3:.0f} ms")
    if args.ref_bw:
        ideal = db / (args.ref_bw * 1e9) * 1e6
        print(f"decode GEMM at {args.ref_bw:g} GB/s would take "
              f"{ideal/1e3:.2f} ms (now {dt/1e3:.2f} ms, "
              f"{dt/ideal:.2f}x slower) -> step {(dtot-dt+ideal)/1e3:.2f} ms "
              f"= {args.batch*1e6/(dtot-dt+ideal):.1f} tok/s")


if __name__ == "__main__":
    main()
