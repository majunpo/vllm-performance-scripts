#!/usr/bin/env python3
"""Per-operator breakdown of a Qwen3.5/3.6 hybrid-attention PyTorch Profiler
trace captured on NVIDIA.

Counterpart of analyze_hybrid_trace.py / analyze_hybrid_gemm.py, which read
Intel unitrace. This reads `rank0.*.pt.trace.json[.gz]` and produces the same
category / layer-type / per-linear tables so the two platforms can be compared
row by row.

A ModelOpt NVFP4 export is usually `"quant_algo": "MIXED_PRECISION"`: each
linear is served from a different backend, and the backend name encodes the
weight precision, so the byte model is derived from the kernel name instead of
from a --weight-dtype flag. Verified against nvidia/Qwen3.6-27B-NVFP4's
quantization_config:

    marlin::Marlin<...>          W4A16_NVFP4 (group 16)  0.5625 B/element
    cudnn_..._matMul_pointwise   FP8 W8A8, static        1.0
    sm89_xmma_gemm_e4m3bf16_...  FP8 W8A8, static        1.0
    cublas gemvx / cutlass bf16  not quantized           2.0

W4A16 means the 4-bit weights are decompressed to FP16 before the MACs, so the
MLP gets 4-bit bandwidth but BF16 compute. The per-backend TFLOPS table exists
to make that visible.
"""

import argparse
import collections
import gzip
import json
import os
import re
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hybrid_common import CFG, layer_kinds

# --------------------------------------------------------------------------
# backend -> weight precision.  Marlin is W4A16: 4-bit weights, FP16/BF16 MACs.
# --------------------------------------------------------------------------
NVFP4_BYTES = 0.5 + 1.0 / 16        # e2m1 + one fp8 scale per 16 elements
FP8_BYTES = 1.0
FP8_BLOCK_BYTES = 1.0 + 4.0 / (128 * 128)   # e4m3 + one fp32 scale per 128x128
BF16_BYTES = 2.0

BACKENDS = [
    ("Marlin(NVFP4)", re.compile(r"marlin::Marlin"), NVFP4_BYTES),
    # must precede cutlass(bf16): the name also matches `cutlass\w*gemm`
    ("cutlass(fp8blk)", re.compile(r"gemm_fp8_blockwise|cutlass_3x_gemm_fp8"),
     FP8_BLOCK_BYTES),
    ("cuDNN(fp8)", re.compile(r"cudnn_generated\w*matMul"), FP8_BYTES),
    ("xmma(fp8)", re.compile(r"xmma_gemm_e4m3|xmma_gemm_\w*fp8"), FP8_BYTES),
    ("gemvx(bf16)", re.compile(r"internal::gemvx::kernel"), BF16_BYTES),
    ("cutlass(bf16)", re.compile(r"cutlass\w*gemm|cutlass::Kernel"), BF16_BYTES),
]


def backend_of(name):
    for label, rx, w in BACKENDS:
        if rx.search(name):
            return label, w
    return None, None


# --------------------------------------------------------------------------
# kernel markers.  Verified against the 260918 RTX PRO 5000 trace.
# --------------------------------------------------------------------------
RE_GDN_CHUNK = re.compile(
    r"chunk_fwd_kernel_o|chunk_gated_delta_rule_fwd|merge_\d+x\d+_to_\d+x\d+_inverse"
    r"|recompute_w_u|chunk_scaled_dot|_causal_conv1d_fwd|_fused_post_conv"
    r"|chunk_local_cumsum|solve_tril|prepare_wy")
RE_GDN_RECUR = re.compile(r"fused_recurrent_gated_delta_rule|_causal_conv1d_update")
RE_FMHA = re.compile(r"flashinfer::\w*AttentionKernel|BatchPrefillWithPagedKVCache"
                     r"|BatchDecodeWithPagedKVCache|flash_fwd")
RE_MERGE = re.compile(r"MergeStates|flash_fwd_splitkv_combine")
RE_KVWRITE = re.compile(r"reshape_and_cache")
RE_SILU = re.compile(r"silu_slice|silu_and_mul")
RE_GDN_NORM = re.compile(r"rsqrt_silu|mean_pow_view|layer_norm_fwd")
RE_OUT_GATE = re.compile(r"sigmoid_view")
# an FP8 build gives the input norm and the post-attention norm the same name,
# so the role comes from what the norm feeds (see norm_roles)
RE_NORM_POST = re.compile(r"rms_norm\w*marlin_gemm_view")
RE_NORM = re.compile(r"rms_norm")
RE_QUANT = re.compile(r"per_token_group_quant|per_block_quant_kernel|fp8_quant")
RE_GDN_STATE = re.compile(r"triton_poi_fused_zeros_\d+")
# only the three kernels that actually sit between qkv_proj and the KV write;
# a looser pattern also matches the GDN layer's in_proj split (triton_poi_fused_1)
# and silently re-tags every in_proj_qkvz as qkv_proj
RE_QK_ROPE = re.compile(r"triton_poi_fused_6$|triton_red_fused_7$"
                        r"|triton_poi_fused_8$|_fused_qk_rmsnorm_rope")
RE_SAMPLE = re.compile(r"ArgMaxOps|argmax|top_k_top_p|gumbel")
RE_SCHED = re.compile(r"slot_mapping|page_indices|block_table|_zero_kv_blocks"
                      r"|_prepare_|_apply_write|_combine_sampled|_post_update"
                      r"|_bias_kernel|_get_num_sampled|_scatter_num_accepted")
RE_MEMSET = re.compile(r"^Memset")
RE_MEMCPY = re.compile(r"^Memcpy|memcpy\d*_post")

CATEGORY_ORDER = [
    "Dense-GEMM", "GDN-Attn(chunk)", "GDN-Attn(recurrent)", "FullAttn-FMHA",
    "FullAttn-MergeStates", "Quantize(fp8)", "Norm(RMS)", "GDN-Norm/Gate",
    "QK-Norm/RoPE", "Activation(SiLU)", "FullAttn-OutGate", "KVCache-Write",
    "GDN-State-Init", "Sampling", "Sched/Prep", "Memset", "MemCopy",
    "Elementwise/Layout", "Other",
]


def bucket(name):
    if backend_of(name)[0]:
        return "Dense-GEMM"
    if RE_GDN_CHUNK.search(name):
        return "GDN-Attn(chunk)"
    if RE_GDN_RECUR.search(name):
        return "GDN-Attn(recurrent)"
    if RE_MERGE.search(name):
        return "FullAttn-MergeStates"
    if RE_FMHA.search(name):
        return "FullAttn-FMHA"
    if RE_KVWRITE.search(name):
        return "KVCache-Write"
    if RE_SILU.search(name):
        return "Activation(SiLU)"
    if RE_GDN_NORM.search(name):
        return "GDN-Norm/Gate"
    if RE_OUT_GATE.search(name):
        return "FullAttn-OutGate"
    if RE_NORM.search(name):
        return "Norm(RMS)"
    if RE_QUANT.search(name):
        return "Quantize(fp8)"
    if RE_GDN_STATE.search(name):
        return "GDN-State-Init"
    if RE_SAMPLE.search(name):
        return "Sampling"
    if RE_QK_ROPE.search(name):
        return "QK-Norm/RoPE"
    if RE_SCHED.search(name):
        return "Sched/Prep"
    if RE_MEMSET.search(name):
        return "Memset"
    if RE_MEMCPY.search(name):
        return "MemCopy"
    if "elementwise_kernel" in name or "index_" in name or "indexSelect" in name \
            or "gather_kernel" in name:
        return "Elementwise/Layout"
    return "Other"


# --------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------

DEVICE_EVENT_CATS = ("kernel", "gpu_memset", "gpu_memcpy")


def load(path):
    op = gzip.open if path.endswith(".gz") else open
    with op(path, "rt") as f:
        doc = json.load(f)
    ev = doc["traceEvents"]
    dev = (doc.get("deviceProperties") or [{}])[0]
    # cat=overhead holds host-side CUPTI markers such as "Command Buffer Full"
    # whose duration overlaps real kernels; counting them double-counts the run
    k = sorted((e for e in ev if e.get("cat") in DEVICE_EVENT_CATS and "dur" in e),
               key=lambda e: e["ts"])
    ann = sorted((e for e in ev if e.get("cat") == "gpu_user_annotation"
                  and "execute_context" in e.get("name", "")),
                 key=lambda e: e["ts"])
    return dev, k, ann


RE_CTX = re.compile(r"execute_context_\d+\((\d+)\)_generation_\d+\((\d+)\)")


def windows(kernels, ann):
    """(phase, annotation, [kernels], period_us) per annotated forward pass.

    The annotation name encodes how many context and generation tokens the
    forward ran, which is what separates prefill from decode.

    Kernels are assigned to the annotation that **most recently started**, not
    to the annotation's own span: `execute_context_*` closes before lm_head and
    the sampler run, so clipping on its `dur` drops the tail of every forward
    pass (on the reference FP8 trace that is 3 GEMMs, one SiLU, lm_head and the
    sampler -- enough to make every invariant count come out short).
    """
    out = []
    end = max(e["ts"] + e["dur"] for e in kernels) if kernels else 0
    bounds = [a["ts"] for a in ann] + [end + 1]
    for i, a in enumerate(ann):
        m = RE_CTX.search(a["name"])
        ctx = int(m.group(1)) if m else 0
        lo, hi = bounds[i], bounds[i + 1]
        sel = [e for e in kernels if lo <= e["ts"] < hi]
        out.append(("prefill" if ctx > 0 else "decode", a, sel, hi - lo))
    head = [e for e in kernels if e["ts"] < bounds[0]]
    return out, head


# --------------------------------------------------------------------------
# layer / linear attribution
# --------------------------------------------------------------------------

GDN_LINEARS = ["in_proj_qkvz", "in_proj_ba", "gdn_out_proj", "gate_up", "down_proj"]
FULL_LINEARS = ["qkv_proj", "o_proj", "gate_up", "down_proj"]


def shapes(cfg=CFG):
    h, inter = cfg["hidden"], cfg["inter"]
    hq, hkv, d = cfg["heads"], cfg["kv_heads"], cfg["head_dim"]
    gk, gkd = cfg["gdn_k_heads"], cfg["gdn_k_dim"]
    gv, gvd = cfg["gdn_v_heads"], cfg["gdn_v_dim"]
    return {
        "qkv_proj": (h, hq * d * (2 if cfg["attn_output_gate"] else 1) + 2 * hkv * d),
        "o_proj": (hq * d, h),
        "in_proj_qkvz": (h, 2 * gk * gkd + 2 * gv * gvd),
        "in_proj_ba": (h, 2 * gv),
        "gdn_out_proj": (gv * gvd, h),
        "gate_up": (h, 2 * inter),
        "down_proj": (inter, h),
        "lm_head": (h, cfg["vocab"]),
    }


def norm_roles(sel):
    """{index: 'in' | 'post'} for every RMS-norm kernel in the window.

    A Marlin build names the post-attention norm after the GEMM it feeds, but a
    blockwise-FP8 build gives both norms the same fused name, so the role has to
    come from what the norm feeds: the MLP (SiLU) for a post-attention norm, the
    attention block (GDN kernels / FMHA / KV write) for a layer's input norm.
    Getting this wrong splits every layer in two and parks its MLP under `head`.
    """
    roles = {}
    for i, e in enumerate(sel):
        nm = e["name"]
        if not RE_NORM.search(nm) or RE_GDN_NORM.search(nm):
            continue
        if RE_NORM_POST.search(nm):
            roles[i] = "post"
            continue
        role = "in"
        for j in range(i + 1, len(sel)):
            n2 = sel[j]["name"]
            if RE_SILU.search(n2):
                role = "post"
                break
            if (RE_GDN_CHUNK.search(n2) or RE_GDN_RECUR.search(n2)
                    or RE_FMHA.search(n2) or RE_KVWRITE.search(n2)
                    or (RE_NORM.search(n2) and not RE_GDN_NORM.search(n2))):
                break
        roles[i] = role
    return roles


def segment_layers(sel, cfg=CFG):
    """Split a forward into decoder layers, keyed on the input norm.

    The window opens with a scheduling prologue (slot mapping, page indices,
    embedding) that ends at the first input norm, so the first segment carries
    no GEMM and is not a layer.  Segments are therefore filtered to those that
    actually contain an attention block.
    """
    roles = norm_roles(sel)
    marks = [i for i, r in sorted(roles.items()) if r == "in"]
    if not marks:
        return []
    bounds = [(0, marks[0] + 1)]
    for j in range(len(marks) - 1):
        bounds.append((marks[j] + 1, marks[j + 1] + 1))
    bounds.append((marks[-1] + 1, len(sel)))

    kinds = []
    for lo, hi in bounds:
        kind = None
        for j in range(lo, hi):
            nm = sel[j]["name"]
            if RE_GDN_CHUNK.search(nm) or RE_GDN_RECUR.search(nm):
                kind = "gdn"
                break
            if RE_FMHA.search(nm) or RE_KVWRITE.search(nm):
                kind = "full"
                break
        kinds.append(kind)

    layers = []
    expected = layer_kinds(cfg)
    for i, ((lo, hi), kind) in enumerate(zip(bounds, kinds)):
        if kind is None:
            has_gemm = any(backend_of(sel[j]["name"])[0] for j in range(lo, hi))
            if not has_gemm:
                continue                    # scheduling prologue
            layers.append(("head", lo, hi))  # lm_head tail
            continue
        layers.append((kind, lo, hi))
    # anything past the last real decoder layer is the lm_head / sampling tail
    n = cfg["layers"]
    real = [l for l in layers if l[0] != "head"]
    if len(real) > n:
        real, extra = real[:n], real[n:]
        layers = real + [("head", extra[0][1], len(sel))]
    else:
        layers = real + [l for l in layers if l[0] == "head"]
    return layers


def tag_gemms(sel, layers, lookahead=40):
    """Name every GEMM from the kernel that consumes its output.

    At M=3300 Marlin splits one linear across several launches, so consecutive
    launches carrying the same tag are one logical linear; the caller sums them.
    """
    kind_at = {}
    for kind, lo, hi in layers:
        for i in range(lo, hi):
            kind_at[i] = kind
    roles = norm_roles(sel)
    tags = {}
    for i, e in enumerate(sel):
        label, _ = backend_of(e["name"])
        if not label:
            continue
        if label in ("gemvx(bf16)", "cutlass(bf16)"):
            # the bf16 backends serve only in_proj_ba and an unquantized lm_head
            tags[i] = "lm_head" if kind_at.get(i) == "head" else "in_proj_ba"
            continue
        kind = kind_at.get(i, "gdn")
        tag = None
        for j in range(i + 1, min(i + lookahead, len(sel))):
            nm = sel[j]["name"]
            if RE_SILU.search(nm):
                tag = "gate_up"
            elif RE_GDN_RECUR.search(nm) or RE_GDN_CHUNK.search(nm) \
                    or RE_GDN_NORM.search(nm) or RE_GDN_STATE.search(nm):
                tag = "in_proj_qkvz"
            elif RE_KVWRITE.search(nm) or RE_FMHA.search(nm) or RE_QK_ROPE.search(nm):
                tag = "qkv_proj"
            elif RE_SAMPLE.search(nm):
                tag = "lm_head"
            elif roles.get(j) == "post":
                tag = "gdn_out_proj" if kind == "gdn" else "o_proj"
            elif roles.get(j) == "in":
                tag = "down_proj"
            if tag:
                break
        tags[i] = tag or ("lm_head" if kind == "head" else "unattributed")
    return tags


# --------------------------------------------------------------------------
# reporting
# --------------------------------------------------------------------------

def tally(sel):
    cats = collections.defaultdict(lambda: [0, 0.0])
    kern = collections.defaultdict(lambda: [0, 0.0])
    for e in sel:
        c = bucket(e["name"])
        cats[c][0] += 1
        cats[c][1] += e["dur"]
        kern[(c, e["name"])][0] += 1
        kern[(c, e["name"])][1] += e["dur"]
    return cats, kern


def print_categories(cats, title, div=1):
    total = sum(v[1] for v in cats.values())
    print(f"\n--- {title} ---")
    head = f"{'category':<24} {'cnt':>8} {'ms':>10} {'pct':>7}"
    print(head)
    order = [c for c in CATEGORY_ORDER if c in cats] + \
            [c for c in cats if c not in CATEGORY_ORDER]
    for c in sorted(order, key=lambda x: -cats[x][1]):
        n, d = cats[c]
        print(f"{c:<24} {n/div:>8.1f} {d/div/1e3:>10.3f} {d/total*100:>6.2f}%")
    print(f"{'TOTAL':<24} {sum(v[0] for v in cats.values())/div:>8.1f} "
          f"{total/div/1e3:>10.3f} {100.0:>6.2f}%")


def print_kernels(kern, title, topn, maxname):
    total = sum(v[1] for v in kern.values())
    print(f"\n--- {title} ---")
    print(f"{'cat':<22} {'cnt':>6} {'ms':>9} {'us/call':>9} {'pct':>7}  kernel")
    for (c, n), (cnt, d) in sorted(kern.items(), key=lambda x: -x[1][1])[:topn]:
        print(f"{c:<22} {cnt:>6} {d/1e3:>9.3f} {d/cnt:>9.2f} "
              f"{d/total*100:>6.2f}%  {n[:maxname]}")


def print_gaps(sel, label, top=10, maxname=92):
    """Wall span, busy ratio and idle attributed to the kernel that follows.

    Valid on every NVIDIA trace: CUPTI times each kernel individually even under
    CUDA Graph, so unlike the XPU side the start timestamps can be trusted.
    """
    evs = sorted(sel, key=lambda e: e["ts"])
    if len(evs) < 2:
        return
    busy = sum(e["dur"] for e in evs)
    span = max(e["ts"] + e["dur"] for e in evs) - evs[0]["ts"]
    by_cat = collections.defaultdict(lambda: [0, 0.0])
    gaps = []
    cur = evs[0]["ts"] + evs[0]["dur"]
    for e in evs[1:]:
        if e["ts"] > cur:
            g = e["ts"] - cur
            gaps.append((g, e["name"]))
            c = bucket(e["name"])
            by_cat[c][0] += 1
            by_cat[c][1] += g
        cur = max(cur, e["ts"] + e["dur"])
    idle = sum(g for g, _ in gaps)
    print(f"\n--- {label}: wall / idle ---")
    print(f"wall span {span/1e3:.3f} ms   busy {busy/1e3:.3f} ms "
          f"({busy/span*100:.1f}%)   idle {idle/1e3:.3f} ms "
          f"({idle/span*100:.1f}%)   gaps {len(gaps)}")
    if not gaps:
        return
    print(f"{'idle before -> kernel category':<28} {'n':>6} {'ms':>9} {'pct':>7}")
    for c, (n, s) in sorted(by_cat.items(), key=lambda x: -x[1][1]):
        print(f"{c:<28} {n:>6} {s/1e3:>9.4f} {s/idle*100:>6.1f}%")
    print(f"top {top} gaps (ms, kernel that follows):")
    for g, n in sorted(gaps, reverse=True)[:top]:
        print(f"  {g/1e3:8.4f}  {n[:maxname]}")


def print_layer_split(sel, layers, label):
    agg = collections.defaultdict(lambda: [0, 0, 0.0])
    for kind, lo, hi in layers:
        agg[kind][0] += 1
        agg[kind][1] += hi - lo
        agg[kind][2] += sum(e["dur"] for e in sel[lo:hi])
    total = sum(v[2] for v in agg.values())
    print(f"\n--- {label}: time by layer type ---")
    print(f"{'layer type':<14} {'layers':>7} {'kernels':>8} {'ms':>10} {'pct':>7} "
          f"{'ms/layer':>10}")
    for kind in ("gdn", "full", "head"):
        if kind not in agg:
            continue
        n, k, d = agg[kind]
        print(f"{kind:<14} {n:>7} {k:>8} {d/1e3:>10.3f} {d/total*100:>6.2f}% "
              f"{d/n/1e3:>10.4f}")


def print_gemm(sel, tags, phase, m, cfg, div=1):
    shp = shapes(cfg)
    F = cfg["layers"] // cfg["full_attention_interval"]
    G = cfg["layers"] - F
    expect = {"in_proj_qkvz": G, "in_proj_ba": G, "gdn_out_proj": G,
              "qkv_proj": F, "o_proj": F, "gate_up": cfg["layers"],
              "down_proj": cfg["layers"], "lm_head": 1}
    agg = collections.defaultdict(lambda: [0, 0.0, set()])
    for i, tag in tags.items():
        a = agg[tag]
        a[0] += 1
        a[1] += sel[i]["dur"]
        a[2].add(backend_of(sel[i]["name"])[0])
    print(f"\n--- {phase.upper()} Dense-GEMM per linear (M={m}) ---")
    hdr = (f"{'linear':<14} {'backend':<15} {'K':>6} {'N':>7} {'launch':>7} "
           f"{'us/layer':>9} {'GFLOP':>9} {'MB_w':>8} {'TFLOPS':>8} {'GB/s':>8} "
           f"{'ms':>8}")
    print(hdr)
    print("-" * len(hdr))
    tot_t = tot_f = tot_b = 0.0
    for tag in ("in_proj_qkvz", "qkv_proj", "gate_up", "down_proj",
                "gdn_out_proj", "o_proj", "in_proj_ba", "lm_head",
                "unattributed"):
        if tag not in agg or tag not in shp:
            continue
        launches, dur, bks = agg[tag]
        k, n = shp[tag]
        w = max(BACKENDS[i][2] for i in range(len(BACKENDS))
                if BACKENDS[i][0] in bks) if bks else BF16_BYTES
        w = min(b[2] for b in BACKENDS if b[0] in bks)
        cnt = expect.get(tag, 1) * div
        mm = 1 if tag == "lm_head" else m
        flops = 2.0 * mm * n * k * cnt
        byts = k * n * w * cnt
        tot_t += dur
        tot_f += flops
        tot_b += byts
        per = dur / cnt
        print(f"{tag:<14} {'+'.join(sorted(bks)):<15} {k:>6} {n:>7} "
              f"{launches/div:>7.0f} {per:>9.2f} {flops/cnt/1e9:>9.2f} "
              f"{k*n*w/1e6:>8.2f} {flops/dur/1e6:>8.1f} {byts/dur/1e3:>8.1f} "
              f"{dur/div/1e3:>8.3f}")
    print("-" * len(hdr))
    print(f"aggregate      : {tot_t/div/1e3:.3f} ms, {tot_f/div/1e12:.2f} TFLOP, "
          f"{tot_b/div/1e9:.2f} GB")
    if phase == "prefill":
        print(f"                 {tot_f/tot_t/1e6:.1f} TFLOPS  (compute bound)")
    else:
        print(f"                 {tot_b/tot_t/1e3:.1f} GB/s  (memory bound)")
    bk = collections.defaultdict(lambda: [0, 0.0, 0.0])
    for i, tag in tags.items():
        label, w = backend_of(sel[i]["name"])
        k, n = shp.get(tag, (0, 0))
        mm = 1 if tag == "lm_head" else m
        bk[label][0] += 1
        bk[label][1] += sel[i]["dur"]
    print(f"\n{'backend':<16} {'launches':>9} {'ms':>10} {'pct of GEMM':>12}")
    for label, (c, d, _) in sorted(bk.items(), key=lambda x: -x[1][1]):
        print(f"{label:<16} {c/div:>9.0f} {d/div/1e3:>10.3f} {d/tot_t*100:>11.2f}%")
    return tot_t, tot_f, tot_b


def sanity(sel, layers, tags, phase, cfg):
    F = cfg["layers"] // cfg["full_attention_interval"]
    G = cfg["layers"] - F
    cats, _ = tally(sel)
    n_gdn = sum(1 for k, _, _ in layers if k == "gdn")
    n_full = sum(1 for k, _, _ in layers if k == "full")
    logical = collections.Counter(tags.values())
    checks = [
        ("gdn layers", n_gdn, G),
        ("full layers", n_full, F),
        ("FullAttn-FMHA", cats["FullAttn-FMHA"][0], F),
        ("KVCache-Write", cats["KVCache-Write"][0], F),
        ("Activation(SiLU)", cats["Activation(SiLU)"][0], cfg["layers"]),
        ("in_proj_ba launches", logical.get("in_proj_ba", 0), G),
        ("unattributed GEMM", logical.get("unattributed", 0), 0),
    ]
    if phase.lower().startswith("prefill"):
        checks.append(("GDN-Attn(chunk) per layer",
                       cats["GDN-Attn(chunk)"][0] % G, 0))
    else:
        checks.append(("GDN-Attn(recurrent)", cats["GDN-Attn(recurrent)"][0], 2 * G))
    print(f"\n--- sanity check: {phase} ---")
    ok = True
    for name, got, want in checks:
        flag = "OK " if got == want else "BAD"
        ok &= got == want
        print(f"  [{flag}] {name:<22} got={got:<6} expected={want}")
    if not ok:
        print("  !! counts are off -- markers may have changed with the vLLM "
              "version; do not trust the numbers below")
    return ok


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("trace", help="rank0.*.pt.trace.json[.gz]")
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--prompt-len", type=int, default=0,
                   help="autodetected from the prefill annotation when omitted")
    p.add_argument("--decode-step", type=int, default=5)
    p.add_argument("--top", type=int, default=30)
    p.add_argument("--max-name", type=int, default=92)
    p.add_argument("--no-detail", action="store_true")
    args = p.parse_args()

    cfg = CFG
    dev, kernels, ann = load(args.trace)
    if not ann:
        sys.exit("no execute_context_* annotations -- was the run profiled with "
                 "vLLM's detailed annotation enabled?")
    wins, tail = windows(kernels, ann)
    pre = [w for w in wins if w[0] == "prefill"]
    dec = [w for w in wins if w[0] == "decode"]
    if not pre and not dec:
        sys.exit("no annotated forward pass found")
    prefill_only = not dec
    decode_only = not pre
    if decode_only and not args.prompt_len:
        sys.exit("decode-only trace (no prefill annotation): pass --prompt-len "
                 "-- the KV context length is recorded nowhere else and it "
                 "scales every attention byte count")

    tokens = args.prompt_len or int(RE_CTX.search(pre[0][1]["name"]).group(1))
    print(f"trace  : {args.trace} "
          f"({os.path.getsize(args.trace)/2**20:.1f} MiB, {len(kernels)} device events)")
    print(f"device : {dev.get('name','?')}  sm_{dev.get('computeMajor','?')}"
          f"{dev.get('computeMinor','')}  {dev.get('numSms','?')} SMs  "
          f"{dev.get('totalGlobalMem',0)/2**30:.1f} GiB")
    print(f"model  : {cfg['name']}  layers={cfg['layers']} "
          f"(gdn={cfg['layers']-cfg['layers']//cfg['full_attention_interval']}, "
          f"full={cfg['layers']//cfg['full_attention_interval']})")
    print(f"load   : prompt={tokens}"
          f"{' (from --prompt-len)' if decode_only else ''}"
          f", batch={args.batch}, {len(dec)} decode windows"
          + ("  -> PREFILL-ONLY trace (out=1)" if prefill_only else "")
          + ("  -> DECODE-ONLY trace" if decode_only else ""))
    print("\n!! cat=overhead events (e.g. 'Command Buffer Full') are host-side "
          "CUPTI markers whose\n!! duration overlaps real kernels; they are "
          "excluded. Only kernel/gpu_memset/gpu_memcpy\n!! events are counted.")
    if tail:
        print(f"!! {len(tail)} kernels ({sum(e['dur'] for e in tail)/1e3:.3f} ms) "
              f"ran before the first annotation (warm-up) and are excluded.")

    counts = [len(w[2]) for w in dec]
    if not prefill_only:
        modal = collections.Counter(counts).most_common(1)[0][0]
        # a stray memcpy shifts the count by one; anything within 1% is a full step
        good = [w for w in dec if abs(len(w[2]) - modal) <= max(2, modal // 100)]
        print(f"\ncomplete decode windows: {len(good)}/{len(dec)} "
              f"({min(counts)}-{max(counts)} kernels each)")

    psel = players = ptags = None
    if not decode_only:
        psel = pre[0][2]
        players = segment_layers(psel, cfg)
        ptags = tag_gemms(psel, players)
        sanity(psel, players, ptags, "PREFILL", cfg)

    if not prefill_only:
        di = min(args.decode_step, len(good) - 1)
        dsel = good[di][2]
        dlayers = segment_layers(dsel, cfg)
        dtags = tag_gemms(dsel, dlayers)
        sanity(dsel, dlayers, dtags, f"DECODE step #{di}", cfg)

    # the step period is start-to-start: the annotation's own `dur` stops before
    # lm_head and the sampler, so it is not the step time
    dstarts = [w[1]["ts"] for w in dec]
    print(f"\n{'='*100}\nOVERVIEW\n{'='*100}")
    if not decode_only:
        pk = sum(e["dur"] for e in psel)
        if prefill_only:
            pw = max(e["ts"] + e["dur"] for e in psel) - pre[0][1]["ts"]
        else:
            pw = dstarts[0] - pre[0][1]["ts"]
        print(f"prefill        : period {pw/1e3:8.3f} ms   "
              f"kernels {pk/1e3:8.3f} ms "
              f"({pk/pw*100:.1f}% busy)  -> {tokens/pw*1e6:.0f} tok/s")
    if not prefill_only:
        periods = [b - a for a, b in zip(dstarts, dstarts[1:])]
        dwin = statistics.mean(periods) if periods \
            else statistics.mean(w[3] for w in good)
        dk = statistics.mean(sum(e["dur"] for e in w[2]) for w in good)
        print(f"decode / step  : period {dwin/1e3:8.3f} ms   "
              f"kernels {dk/1e3:8.3f} ms "
              f"({dk/dwin*100:.1f}% busy)  -> {args.batch*1e6/dwin:.2f} tok/s")
        if len(periods) > 1:
            # the mean is the headline, but a decode-only capture's first period
            # carries the profiler start-up, so always show the robust median
            med = statistics.median(periods)
            print(f"decode spread  : median {med/1e3:.3f} ms   "
                  f"min {min(periods)/1e3:.3f} / max {max(periods)/1e3:.3f} "
                  f"over {len(periods)} periods "
                  f"(stdev {statistics.stdev(periods)/1e3:.3f} ms)"
                  + (f"   !! mean is skewed by an outlier, quote the median "
                     f"-> {args.batch*1e6/med:.2f} tok/s"
                     if abs(med - dwin) > 0.01 * med else ""))
        if not decode_only:
            print(f"end-to-end     : {(pw + dwin*len(dec))/1e3:.1f} ms "
                  f"(1 prefill + {len(dec)} decode)")

    if not decode_only:
        pcats, pkern = tally(psel)
        print_categories(pcats, f"PREFILL categories ({pk/1e3:.3f} ms)")
        print_gaps(psel, "PREFILL", maxname=args.max_name)
        print_layer_split(psel, players, "PREFILL")
        print_gemm(psel, ptags, "prefill", tokens, cfg)

    if not prefill_only:
        dcats = collections.defaultdict(lambda: [0, 0.0])
        dkern = collections.defaultdict(lambda: [0, 0.0])
        for w in good:
            c1, k1 = tally(w[2])
            for k, v in c1.items():
                dcats[k][0] += v[0]
                dcats[k][1] += v[1]
            for k, v in k1.items():
                dkern[k][0] += v[0]
                dkern[k][1] += v[1]
        print_categories(dcats,
                         f"DECODE categories (per step, {len(good)} steps)",
                         div=len(good))
        print_gaps(dsel, f"DECODE step #{di}", maxname=args.max_name)
        print_layer_split(dsel, dlayers, f"DECODE step #{di}")
        print_gemm(dsel, dtags, "decode", args.batch, cfg)

    if not args.no_detail:
        if not decode_only:
            print_kernels(pkern, f"PREFILL top {args.top} kernels", args.top,
                          args.max_name)
        if not prefill_only:
            print_kernels({k: [v[0]/len(good), v[1]/len(good)]
                           for k, v in dkern.items()},
                          f"DECODE top {args.top} kernels (per step)", args.top,
                          args.max_name)


if __name__ == "__main__":
    main()
