#!/usr/bin/env python3
"""Phase-split breakdown of a vLLM torch-profiler trace captured on CUDA.

Counterpart of the unitrace scripts used for XPU, so the two platforms can be
compared row by row. Prefill and decode are separated with the
`execute_context_<n>(<tokens>)_generation_<m>(<tokens>)` annotations vLLM emits,
and every dense GEMM is attributed to a linear layer:

  * prefill runs eagerly, so DeepGEMM instantiates one kernel per shape and N/K
    are literal template arguments in the kernel name;
  * decode is replayed from a CUDA graph, which drops the host-side operator, so
    only K stays in the kernel name and the two K=hidden GEMMs (qkv and gate_up)
    are told apart by the kernel that feeds them.
"""

import argparse
import gzip
import json
import re
import statistics
import sys
from collections import defaultdict

# hidden, heads, kv_heads, head_dim, intermediate, layers, vocab -- Qwen3-32B
CFG = dict(hidden=5120, heads=64, kv_heads=8, head_dim=128,
           inter=25600, layers=64, vocab=151936)

BF16_BYTES = 2.0
FP8_GROUP = 128                       # DeepGEMM 1d1d block scaling
FP8_BYTES = 1.0 + 1.0 / FP8_GROUP

RE_PHASE = re.compile(r"execute_context_(\d+)\((\d+)\)_generation_(\d+)\((\d+)\)")
RE_DEEPGEMM = re.compile(r"sm120_fp8_fp4_gemm_1d1d_impl<\d+u, (\d+)u, (\d+)u")

RE_QUANT = re.compile(r"per_token_group_quant|quantize|_quant_kernel")
RE_ROPE_NORM = re.compile(r"rms_norm|layer_norm|rotary|rope|"
                          r"^triton_(red|poi)_fused_\d+$")
RE_ACT = re.compile(r"silu|gelu|act_and_mul|swiglu")
RE_ATTN = re.compile(r"flash_fwd|flash::|fmha|attention_kernel")
RE_SAMPLE = re.compile(r"gumbel|_bias_kernel|argmax|sample")
# every forward ends by sampling a token, which makes an exact step delimiter
RE_STEP_END = re.compile(r"_gumbel_sample_kernel|argmax|top_k_top_p")
RE_SCHED = re.compile(r"_apply_write|_prepare_|_combine_sampled|_gather_block_tables|"
                      r"_compute_slot_mappings|_post_update|block_table")
RE_ELTWISE = re.compile(r"elementwise|FillFunctor|index_|reduce_kernel|Transpose|"
                        r"CatArrayBatched|copy_")


def bucket(name):
    if RE_ATTN.search(name):
        return "Attention(FA)"
    if RE_DEEPGEMM.search(name) or "split_k_reduce" in name or "gemvx" in name \
            or "nvjet" in name or "cutlass" in name:
        return "Dense-GEMM"
    if "reshape_and_cache" in name:
        return "KVCache-Write"
    # the fused triton kernels do a norm or a silu *and* the quantisation, so the
    # heavier op wins the bucket and only standalone quant kernels are counted
    if RE_ACT.search(name):
        return "Activation(SiLU)"
    if RE_ROPE_NORM.search(name):
        return "Norm/RoPE"
    if RE_QUANT.search(name):
        return "Quantize(fp8)"
    if RE_SAMPLE.search(name):
        return "Sampling"
    if RE_SCHED.search(name):
        return "Sched/Prep"
    if "Memcpy" in name or "memcpy" in name or "Memset" in name:
        return "MemCopy"
    if RE_ELTWISE.search(name):
        return "Elementwise/Layout"
    return "Other"


def linears(cfg):
    h, inter = cfg["hidden"], cfg["inter"]
    qkv_n = cfg["heads"] * cfg["head_dim"] + 2 * cfg["kv_heads"] * cfg["head_dim"]
    return {
        "qkv_proj":  (h, qkv_n),
        "o_proj":    (cfg["heads"] * cfg["head_dim"], h),
        "gate_up":   (h, 2 * inter),
        "down_proj": (inter, h),
    }


def classify(prev_name):
    """Which linear a decode GEMM belongs to, from the kernel that feeds it."""
    if "flash_fwd" in prev_name or "flash::" in prev_name:
        return "o_proj"
    if RE_ACT.search(prev_name):
        return "down_proj"
    if prev_name.endswith("_0"):
        return "gate_up"
    if prev_name.endswith("_2") or "embedding" in prev_name:
        return "qkv_proj"
    return None


def load(path):
    op = gzip.open if path.endswith(".gz") else open
    with op(path, "rt") as f:
        d = json.load(f)
    evs = [e for e in d["traceEvents"]
           if e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset")]
    evs.sort(key=lambda e: e["ts"])
    phases = []
    for e in d["traceEvents"]:
        if e.get("cat") != "gpu_user_annotation":
            continue
        m = RE_PHASE.match(e["name"])
        if m:
            ctx_tok, gen_tok = int(m.group(2)), int(m.group(4))
            phases.append((e["ts"], e["ts"] + e["dur"],
                           "prefill" if ctx_tok else "decode",
                           ctx_tok or gen_tok))
    phases.sort()
    if not evs or not phases:
        sys.exit("no CUDA kernels or no execute_context annotations found")
    return d, evs, phases


def phase_of(ts, phases):
    for lo, hi, kind, tok in phases:
        if lo <= ts < hi:
            return kind, tok
    return None, 0


def walk(evs, phases):
    """Tag every kernel with its phase and attribute the dense GEMMs.

    The annotation only wraps the model forward; lm_head and the sampler run
    after it closes, so anything outside a window is charged to the phase whose
    window closed last, which makes the totals add up to the real step period.

    DeepGEMM launches its split-K reduce concurrently with the GEMM it reduces
    -- the reduce spins until the partials land -- so durations are charged
    exclusively against the union of what ran before, otherwise the reduce is
    counted twice and the totals exceed the wall time.
    """
    rows = []
    prev = ""
    cursor = None
    pi = 0
    last_kind = None
    for e in evs:
        name, dur, ts = e["name"], e["dur"], e["ts"]
        end = ts + dur
        excl = dur if cursor is None else max(0.0, end - max(ts, cursor))
        cursor = end if cursor is None else max(cursor, end)
        while pi < len(phases) and phases[pi][1] <= ts:
            last_kind = phases[pi][2]
            pi += 1
        if pi < len(phases) and phases[pi][0] <= ts:
            last_kind, tok = phases[pi][2], phases[pi][3]
        else:
            tok = 0
        kind = last_kind
        tag = None
        m = RE_DEEPGEMM.search(name)
        if m:
            n, k = int(m.group(1)), int(m.group(2))
            tag = ("gemm", n if n > 1 else 0, k)
        elif "split_k_reduce" in name:
            tag = ("reduce", 0, 0)
        rows.append((kind, tok, name, excl, tag, prev))
        if not RE_QUANT.search(name) and "memcpy32_post" not in name:
            prev = name
    return rows


def table(title, cats, total, steps=0):
    print(f"\n--- {title} ---")
    head = f"{'category':<22} {'cnt':>8} {'total_ms':>10} {'pct':>7}"
    if steps:
        head += f" {'ms/step':>9} {'cnt/step':>9}"
    print(head)
    ncall = sum(v[0] for v in cats.values())
    for cat, (c, d) in sorted(cats.items(), key=lambda x: -x[1][1]):
        line = f"{cat:<22} {c:>8} {d/1e3:>10.3f} {d/total*100:>6.2f}%"
        if steps:
            line += f" {d/steps/1000:>9.3f} {c/steps:>9.1f}"
        print(line)
    line = f"{'TOTAL':<22} {ncall:>8} {total/1e3:>10.3f} {100.0:>6.2f}%"
    if steps:
        line += f" {total/steps/1000:>9.3f} {ncall/steps:>9.1f}"
    print(line)


def gemm_table(title, entries, m, lm_m=1):
    """entries: {tag: (k, n, [durs], [reduce_durs])}"""
    print(f"\n--- {title} ---")
    hdr = (f"{'linear':<10} {'M':>6} {'K':>6} {'N':>7} {'calls':>6} "
           f"{'med_us':>9} {'+red':>6} {'GFLOP':>9} {'MB':>9} "
           f"{'TFLOPS':>8} {'GB/s':>8}")
    print(hdr)
    print("-" * len(hdr))
    for tag in ("qkv_proj", "o_proj", "gate_up", "down_proj", "lm_head"):
        if tag not in entries:
            continue
        k, n, durs, red = entries[tag]
        if not durs:
            continue
        wb = BF16_BYTES if tag == "lm_head" else FP8_BYTES
        mm = lm_m if tag == "lm_head" else m    # logits: only the last token
        flops = 2.0 * mm * n * k
        byts = k * n * wb + mm * k * wb + mm * n * BF16_BYTES
        med = statistics.median(durs)
        rmed = statistics.median(red) if red else 0.0
        tot = med + rmed
        print(f"{tag:<10} {mm:>6} {k:>6} {n:>7} {len(durs):>6} {med:>9.2f} "
              f"{rmed:>6.2f} {flops/1e9:>9.3f} {byts/2**20:>9.1f} "
              f"{flops/tot/1e6:>8.2f} {byts/tot/1e3:>8.1f}")


def short_name(name, width):
    """Keep the head and the tail: sibling DeepGEMM instantiations only differ
    in their template arguments, which sit at both ends of the name."""
    if len(name) <= width:
        return name
    head = max(20, width - 33)
    return f"{name[:head]}...{name[-30:]}"


def step_detail(rows, lo, hi, label, maxname):
    """Exact per-kernel breakdown of one forward pass, with kernel names.

    The aggregate tables divide by the step count, which yields fractional
    counts because the first and last step are clipped. One window holds
    exactly one forward, so every count here is a whole number.
    """
    cats = defaultdict(lambda: [0, 0.0])
    kern = defaultdict(lambda: [0, 0.0])
    for _, _, name, dur, _, _ in rows[lo:hi]:
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
    for cat, (c, dur) in sorted(cats.items(), key=lambda x: -x[1][1]):
        print(f"{cat:<22} {c:>5} {dur/1e3:>9.4f} {dur/total*100:>6.2f}%")
        for (_, name), (kc, kd) in sorted(
                [kv for kv in kern.items() if kv[0][0] == cat],
                key=lambda x: -x[1][1]):
            print(f"{'':<22} {kc:>5} {kd/1e3:>9.4f} {kd/total*100:>6.2f}%   "
                  f"{short_name(name, maxname)}")
    print(f"{'TOTAL':<22} {sum(v[0] for v in cats.values()):>5} "
          f"{total/1e3:>9.4f} {100.0:>6.2f}%")


def single_step(rows, want, maxname):
    ends = [i for i, r in enumerate(rows) if RE_STEP_END.search(r[2])]
    wins = []
    prev = 0
    for e in ends:
        wins.append((prev, e + 1))
        prev = e + 1
    if not wins:
        print("\n--- single-step detail: no sampler kernel found ---")
        return
    step_detail(rows, *wins[0], "single PREFILL step", maxname)
    if len(wins) <= want + 1:
        print(f"\n--- single decode step: need >= {want + 1} decode steps, "
              f"found {len(wins) - 1} ---")
        return
    step_detail(rows, *wins[want + 1], f"single DECODE step #{want}", maxname)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("trace", help="rank0.*.pt.trace.json[.gz]")
    p.add_argument("--prompt-len", type=int, default=0,
                   help="prompt tokens per sequence, for the attention cost model")
    p.add_argument("--batch", type=int, default=1, help="decode batch size (M)")
    p.add_argument("--step-index", type=int, default=3,
                   help="which decode step to break down kernel by kernel, "
                        "counted from the first one (default: 3, i.e. after "
                        "3 warmup steps)")
    p.add_argument("--max-name", type=int, default=96,
                   help="truncate kernel names in the single-step table")
    args = p.parse_args()

    d, evs, phases = load(args.trace)
    dev = (d.get("deviceProperties") or [{}])[0]
    rows = walk(evs, phases)

    pre_win = [p for p in phases if p[2] == "prefill"]
    dec_win = [p for p in phases if p[2] == "decode"]
    steps = len(dec_win)
    prompt = args.prompt_len or (pre_win[0][3] if pre_win else 0)

    print(f"device : {dev.get('name')}  SMs={dev.get('numSms')}  "
          f"CC {dev.get('computeMajor')}.{dev.get('computeMinor')}  "
          f"mem={dev.get('totalGlobalMem', 0)/2**30:.0f} GiB")
    print(f"model  : hidden={CFG['hidden']} inter={CFG['inter']} "
          f"layers={CFG['layers']} vocab={CFG['vocab']}  "
          f"weights=FP8(e4m3, 1d1d block scale, group={FP8_GROUP})")
    print(f"run    : prefill {prompt} tokens, {steps} decode steps, "
          f"decode batch {args.batch}")
    for lo, hi, kind, tok in phases[:1] + ([] if not dec_win else [dec_win[0]]):
        pass
    if pre_win:
        span = sum(hi - lo for lo, hi, _, _ in pre_win)
        print(f"prefill forward (annotation): {span/1e3:.3f} ms")
    if len(dec_win) > 1:
        fwd = sum(hi - lo for lo, hi, _, _ in dec_win) / steps
        starts = [lo for lo, _, _, _ in dec_win]
        period = statistics.median(b - a for a, b in zip(starts, starts[1:]))
        print(f"decode forward (annotation): {fwd/1e3:.3f} ms/step")
        print(f"decode step period (start-to-start): {period/1e3:.3f} ms/step "
              f"-> {1e6/period:.2f} tok/s")

    # ---- category split -------------------------------------------------
    cats = {"prefill": defaultdict(lambda: [0, 0.0]),
            "decode": defaultdict(lambda: [0, 0.0])}
    for kind, _, name, dur, _, _ in rows:
        if kind is None:
            continue
        slot = cats[kind][bucket(name)]
        slot[0] += 1
        slot[1] += dur
    for kind in ("prefill", "decode"):
        tot = sum(v[1] for v in cats[kind].values())
        if tot <= 0:
            continue
        extra = f", {steps} steps" if kind == "decode" else ""
        table(f"category split: {kind.upper()} ({tot/1e3:.3f} ms{extra})",
              cats[kind], tot, steps if kind == "decode" else 0)

    single_step(rows, args.step_index, args.max_name)

    # ---- GEMM -----------------------------------------------------------
    shapes = linears(CFG)
    by_nk = {(n, k): tag for tag, (k, n) in shapes.items()}
    pre = {t: [shapes[t][0], shapes[t][1], [], []] for t in shapes}
    dec = {t: [shapes[t][0], shapes[t][1], [], []] for t in shapes}
    for tbl in (pre, dec):                 # lm_head stays bf16 and runs once
        tbl["lm_head"] = [CFG["hidden"], CFG["vocab"], [], []]
    last = {"prefill": None, "decode": None}
    for kind, tok, name, dur, tag, prev in rows:
        if kind is None:
            continue
        dst = pre if kind == "prefill" else dec
        if "gemvx" in name or "gemv2" in name:
            dst["lm_head"][2].append(dur)
            last[kind] = None
            continue
        if tag is None:
            continue
        if tag[0] == "reduce":
            if last[kind]:
                dst[last[kind]][3].append(dur)
            continue
        n, k = tag[1], tag[2]
        who = by_nk.get((n, k)) if n else classify(prev)
        if who:
            dst[who][2].append(dur)
            last[kind] = who
        else:
            last[kind] = None

    if any(v[2] for v in pre.values()):
        gemm_table(f"PREFILL GEMM (M={prompt})", pre, prompt, args.batch)
    if any(v[2] for v in dec.values()):
        gemm_table(f"DECODE GEMM (M={args.batch})", dec, args.batch)
        per = [dec[t] for t in shapes if dec[t][2]]
        t = sum(statistics.median(v[2]) + (statistics.median(v[3]) if v[3] else 0)
                for v in per) * CFG["layers"]
        b = sum(v[0] * v[1] * FP8_BYTES + args.batch * v[0] * FP8_BYTES
                + args.batch * v[1] * BF16_BYTES
                for v in per) * CFG["layers"]
        print(f"\nper decode step (median x {CFG['layers']} layers, excl. lm_head):")
        print(f"  GEMM time       : {t/1000:.2f} ms")
        print(f"  weights+act read: {b/2**30:.2f} GiB")
        print(f"  achieved BW     : {b/t/1e3:.1f} GB/s")

    # ---- attention ------------------------------------------------------
    if prompt:
        hq, hkv, dh = CFG["heads"], CFG["kv_heads"], CFG["head_dim"]
        agg = defaultdict(lambda: [0, 0.0])
        for kind, _, name, dur, _, _ in rows:
            if kind is None or not RE_ATTN.search(name):
                continue
            key = (kind, "combine" if "combine" in name else "main")
            agg[key][0] += 1
            agg[key][1] += dur
        print("\n--- attention (FlashAttention) ---")
        hdr = (f"{'kernel':<22} {'calls':>6} {'avg_us':>9} {'ctx/T':>8} "
               f"{'GFLOP/call':>11} {'MB/call':>9} {'TFLOPS':>8} {'GB/s':>8}")
        print(hdr)
        print("-" * len(hdr))
        for (kind, part), (c, tot) in sorted(agg.items()):
            if part == "combine":
                print(f"{kind+' combine':<22} {c:>6} {tot/c:>9.2f} "
                      f"{'-':>8} {'-':>11} {'-':>9} {'-':>8} {'-':>8}")
                continue
            if kind == "prefill":
                fl = 2.0 * hq * dh * prompt * prompt
                by = 4.0 * prompt * hq * dh + 4.0 * prompt * hkv * dh
                ctx = prompt
            else:
                ctx = prompt + 1 + (steps - 1) / 2.0
                fl = 4.0 * args.batch * hq * dh * ctx
                by = 4.0 * args.batch * hkv * dh * ctx + 4.0 * args.batch * hq * dh
            n = c
            print(f"{kind+' FA':<22} {c:>6} {tot/c:>9.2f} {ctx:>8.0f} "
                  f"{fl/1e9:>11.3f} {by/2**20:>9.1f} "
                  f"{fl*n/tot/1e6:>8.2f} {by*n/tot/1e3:>8.1f}")
    else:
        print("\n--- attention: skipped, pass --prompt-len ---")


if __name__ == "__main__":
    main()
