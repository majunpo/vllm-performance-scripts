#!/usr/bin/env python3
"""Ordered kernel walkthrough of ONE prefill forward pass.

Every other script in this repo aggregates: it tells you *how much* each
category costs. This one keeps the **execution order** and throws the
repetition away instead, so a reader can follow a single forward pass from the
embedding lookup to the sampler and see what each kernel is for.

The collapse is not assumed. Every decoder layer's kernel sequence is reduced to
a signature (category, kernel base name, ND-range) and the signatures are
grouped; the report prints how many layers share each one and refuses to call a
layer "same as above" unless its signature is byte-identical to the
representative. A layer that differs is printed in full.

Requires a trace with exactly one forward pass (`output=1`): with `out>1` the
XPU-Graph replays collapse the timestamps and the capture order inside a replay
is no longer the execution order.
"""

import argparse
import os
import re
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_HYBRID = os.path.join(os.path.dirname(os.path.dirname(_HERE)),
                       "qwen36-hybrid-perf-analysis", "scripts")
sys.path.insert(0, _HYBRID)

from hybrid_common import (CFG, base_name, bucket, derive, detect_prompt_len,  # noqa: E402
                           layer_kinds, load, nd_range, walk_layers)

# the kernel that marks the model's real start: embedding lookup + first RMSNorm
RE_EMBED_NORM = re.compile(r"_add_embedding_rms_norm_")

# what each kernel is for.  (pattern, note) -- first match wins, so order
# matters.  `{tag}` is replaced with the GEMM's linear name when there is one.
NOTES = [
    (r"_zero_kv_blocks_kernel", "把本次请求分到的 KV cache block 清零"),
    (r"_compute_slot_mapping_kernel", "算出每个 token 写进 KV cache 的槽位"),
    (r"MemoryCopy\(H2D\)", "host->device 搬 block table / seqlen 等调度元数据"),
    (r"MemoryCopy\(D2D\)", "device 内部搬运调度元数据"),
    (r"MemoryCopy\(D2H\)", "把采样出的 token id 读回 host"),
    (r"_add_embedding_rms_norm_", "embedding 查表 + 第 0 层输入 RMSNorm（模型真正的起点）"),
    (r"per_token_group_quant_8bit_vec_kernel",
     "把 BF16 激活按 32 元素一组动态量化成 FP8 e4m3 + E8M0 scale，喂给下一个 GEMM"),
    (r"triton_poi_fused_zeros_", "GDN recurrent state 缓冲清零（prefill 从零状态开始）"),
    (r"gemm_kernel|GemmUniversal", "线性层 {tag}"),
    (r"gdn::tiled_kernel_launcher", "GDN 的因果深度可分离 1-D 卷积（q/k/v 三条流，kernel 宽度 4）"),
    (r"gdn::chunk_update_states_kernel", "把上一个 chunk 的 recurrent state 传给下一个 chunk"),
    (r"ChunkComputeAO2InvKernel",
     "GDN chunk 内的 A 矩阵 + UT-transform 求逆 + W/U 投影（四合一）"),
    (r"ChunkFwdOKernel", "GDN chunk 内 + chunk 间的输出合并"),
    (r"RecurrentGdnDecodeKernel", "GDN 逐 token 递推（decode 路径，prefill 不走）"),
    (r"rsqrt_silu.*_quantize_|rsqrt_silu(_t)?_view_0",
     "GDN 的 gated RMSNorm：用 z 门对 attention 输出做 SiLU 门控"),
    (r"_fused_add_rms_norm(_mm_view)?_1$", "残差相加 + post-attention RMSNorm（进 MLP 前）"),
    (r"_fused_add_rms_norm(_mm_view)?_3$", "残差相加 + **下一层**的输入 RMSNorm（层边界）"),
    (r"mul_silu_slice", "SwiGLU：把 gate_up 的输出劈成两半，SiLU(gate) * up"),
    (r"mul_sigmoid_view", "full attention 的 output gate：sigmoid(gate) 逐元素乘 attention 输出"),
    (r"triton_poi_fused_4$", "q/gate 段的布局拆分（pointwise）"),
    (r"triton_red_fused_5$", "q/k 的 RMSNorm（带 reduction）"),
    (r"arange_bitwise_and_eq_index_lt_remainder_select_split_where",
     "按 position 取 RoPE 的 cos/sin 表索引"),
    (r"triton_poi_fused_7$", "把 RoPE 旋转应用到 q 和 k 上"),
    (r"reshape_and_cache", "把 BF16 的 k/v 写进 paged KV cache（模板第 3 参 0 = 不转 FP8）"),
    (r"XeFMHAFwdKernel", "因果 FlashAttention（BF16 输入 / fp32 累加）"),
    (r"XeFMHAFwdSplitKVKernel", "unified attention 的 decode 分支（本次没有 decode 序列，空跑）"),
    (r"ReduceSplitK", "把 split-KV 的分块结果归约成一个输出"),
    (r"ArgMax", "采样：贪心取 argmax"),
    (r"IndexKernelFunctor.*IndexPut", "把采样结果写回输出张量"),
    (r"IndexKernelFunctor", "按索引 gather/scatter"),
    (r"FillFunctor", "张量填充（中间缓冲清零）"),
    (r"CopyScalarFunc|unrolled_elementwise_kernel|vectorized_elementwise_kernel",
     "小的 elementwise：切分 / 布局转换 / 标量运算"),
]

# notes that only make sense inside one section
SECTION_NOTES = {
    "head": [(r"IndexKernelFunctor(?!.*IndexPut)",
              "按索引取最后一个 token 的 hidden（只有它要过 lm_head）")],
    "prologue": [(r"IndexKernelFunctor(?!.*IndexPut)", "按索引整理调度元数据")],
    "gdn": [(r"FillFunctor<c10::BFloat16>",
             "把 in_proj 的输出切成独立的连续张量（q / k / v / b / a）")],
}


def note_for(name, tag, section=None):
    for pat, txt in SECTION_NOTES.get(section, []):
        if re.search(pat, name):
            return txt
    for pat, txt in NOTES:
        if re.search(pat, name):
            return txt.replace("{tag}", tag or "?")
    return ""


def known_widths(cfg, section=None):
    """Per-token tensor widths this model can touch -> human name.

    Keyed by section: 6144 is both `Hq*D` (full attention's q) and `Vh*Dv`
    (GDN's v), so a global table would label half the kernels wrong.
    """
    h, inter = cfg["hidden"], cfg["inter"]
    Hq, Hkv, D = cfg["heads"], cfg["kv_heads"], cfg["head_dim"]
    Kh, Dk = cfg["gdn_k_heads"], cfg["gdn_k_dim"]
    Vh, Dv = cfg["gdn_v_heads"], cfg["gdn_v_dim"]
    common = {
        h: "hidden",
        inter: "inter",
        2 * inter: "2*inter (gate_up 输出)",
        cfg["vocab"]: "vocab",
    }
    full = {
        Hq * D: "Hq*D (q 或 attention 输出 或 gate)",
        Hq * D * 2: "Hq*D*2 (q + output gate)",
        Hkv * D: "Hkv*D (k 或 v)",
        2 * Hkv * D: "2*Hkv*D (k+v)",
        Hq * D * 2 + 2 * Hkv * D: "qkv_proj 输出",
        Hq * D + Hkv * D: "q+k (RoPE 作用的范围)",
    }
    gdn = {
        Kh * Dk: "Kh*Dk (GDN 的 q 或 k)",
        Vh * Dv: "Vh*Dv (GDN 的 v 或 z 门 或 attention 输出)",
        2 * Kh * Dk + Vh * Dv: "conv_dim (q+k+v 过卷积)",
        2 * Kh * Dk + 2 * Vh * Dv: "in_proj_qkvz 输出",
        2 * Vh: "2*Vh (in_proj_ba 的 b 和 a)",
        Vh: "Vh (b 或 a)",
    }
    if section == "full":
        return {**gdn, **common, **full}      # full wins on a collision
    if section == "gdn":
        return {**full, **common, **gdn}      # gdn wins on a collision
    return {**full, **gdn, **common}


RE_VEC = re.compile(r"vectorized_elementwise_kernel<(\d+)")
RE_REDUCTION = re.compile(r"triton_(red|per)_fused")
# the ND-range of these is chunk- / head- / tile-based, not per token
RE_NO_HINT = re.compile(r"gdn::|cutlass::|gemm_kernel|GemmUniversal|"
                        r"MemoryCopy|reduce_kernel")


def width_hint(name, prompt_len, widths, tag, shapes):
    """Turn an ND-range into 'this kernel touches <W> values per token'.

    A kernel name never says which tensor it walks; the ND-range plus the
    model's own widths does.  Three cases, because the three backends lay out
    their launches differently:

      GEMM          -> take M/K/N from the config, the grid is tile-based
      Triton red/per-> grid[0] is the number of reduction rows
      pointwise     -> grid[0]*local[0] work-items, each handling `v` elements
                       (explicit in the at::native name, else the smallest
                       power of two that lands on a known width)
    """
    if tag and tag in shapes:
        K, N, _ = shapes[tag]
        M = 1 if tag == "lm_head" else prompt_len
        return f"M={M} K={K} N={N}"
    if RE_NO_HINT.search(name):
        return ""
    nd = nd_range(name)
    if not nd or not prompt_len:
        return ""
    if RE_REDUCTION.search(name):
        rows = nd[0][0]
        per = rows / prompt_len
        extra = f" = {per:g} 行/token" if abs(per - round(per, 3)) < 1e-9 else ""
        return f"reduction {rows} 行{extra}，组内 {nd[1][0]} 线程"
    items = nd[0][0] * (nd[1][0] or 1)
    per_tok = items / prompt_len
    if per_tok < 0.5:          # a scalar / metadata kernel, not a per-token one
        return ""
    explicit = RE_VEC.search(name)
    vecs = [int(explicit.group(1))] if explicit else [1, 2, 4, 8, 16, 32, 64]
    for v in vecs:
        for w in sorted(widths):
            if abs(per_tok * v - w) <= max(4, 0.03 * w):
                return f"{w} 值/token = {widths[w]}" + (f" (vec{v})" if v > 1 else "")
    return f"~{per_tok:.0f} item/token"


def nd_str(name):
    lb = name.rfind("[")
    return name[lb:] if lb > 0 and name.endswith("]") else ""


def signature(evs, lo, hi, wd):
    return tuple((bucket(n, wd), base_name(n), nd_str(n)) for _, _, n in evs[lo:hi])


def short(name, width):
    b = base_name(name)
    return b if len(b) <= width else b[:width - 1] + "\u2026"


def dump(evs, lo, hi, tags, wd, args, cfg, shapes, section=None, collapse=False):
    """One row per kernel; with `collapse`, a run of identical kernels becomes xN."""
    widths = known_widths(cfg, section)
    rows = []
    for i in range(lo, hi):
        _, d, name = evs[i]
        key = (bucket(name, wd), base_name(name), nd_str(name), tags.get(i, ""), name)
        if collapse and rows and rows[-1][0][:4] == key[:4]:
            rows[-1][1] += 1
            rows[-1][2] += d
        else:
            rows.append([key, 1, d])
    for k, (key, cnt, d) in enumerate(rows, start=1):
        cat, bname, nd, tag, full = key
        mult = f"x{cnt}" if cnt > 1 else ""
        hint = width_hint(full, args.prompt_len, widths, tag, shapes)
        print(f"  {k:3d}{mult:>5s}  {d/1e3:8.4f}  {cat:22s} {tag:13s} "
              f"{short(bname, args.max_name):{args.max_name}s} {nd:24s} "
              f"{note_for(bname, tag, section)}"
              + (f"  〔{hint}〕" if hint else ""))


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("trace", help="unitrace python.<pid>.json from an out=1 run")
    p.add_argument("--weight-dtype", choices=("mxfp4", "mxfp8", "bf16"),
                   default="mxfp8")
    p.add_argument("--max-name", type=int, default=62)
    p.add_argument("--max-kernel-s", type=float, default=10.0)
    p.add_argument("--show-all-variants", action="store_true",
                   help="print every signature group, not just the two biggest")
    p.add_argument("--no-collapse", action="store_true",
                   help="do not fold runs of identical prologue kernels into xN")
    p.add_argument("--prompt-len", type=int, default=None,
                   help="tokens in the prompt; autodetected from the KV-write "
                        "ND-range.  Used to turn an ND-range into a tensor width")
    args = p.parse_args()

    wd = args.weight_dtype
    cfg = CFG
    evs = load(args.trace, args.max_kernel_s)
    total = sum(e[1] for e in evs) / 1e3
    if args.prompt_len is None:
        args.prompt_len = detect_prompt_len(evs)
    shapes = derive(cfg, wd)

    embed = [i for i, (_, _, n) in enumerate(evs) if RE_EMBED_NORM.search(n)]
    if len(embed) != 1:
        sys.exit(f"expected exactly 1 embedding-norm kernel, found {len(embed)} "
                 f"-- this is not a single-forward (out=1) trace")
    prologue_end = embed[0]

    layers, tags = walk_layers(evs, cfg, wd)
    kinds = [k for k, _, _ in layers]
    n_gdn = kinds.count("gdn")
    n_full = kinds.count("full")

    print(f"# {os.path.basename(args.trace)}")
    print(f"# {len(evs)} kernels, {total:.3f} ms, weights={wd}, prompt={args.prompt_len} tokens")
    print(f"# {cfg['layers']} layers = {n_gdn} gdn + {n_full} full "
          f"(interval={cfg['full_attention_interval']}), "
          f"layer order = {''.join('F' if k == 'full' else 'G' for k in layer_kinds(cfg))[:16]}...")

    # layer 0 starts at the embedding norm, not at the segment boundary
    bounds = []
    for n, (kind, lo, hi) in enumerate(layers):
        bounds.append((kind, prologue_end if n == 0 else lo, hi))

    print("\n== 结构 ==")
    pro_ms = sum(e[1] for e in evs[:prologue_end]) / 1e3
    print(f"  prologue            [{0:5d}:{prologue_end:5d}]  "
          f"{prologue_end:4d} kernels  {pro_ms:8.3f} ms")
    for kind in ("gdn", "full"):
        sel = [(lo, hi) for k, lo, hi in bounds if k == kind]
        ms = sum(sum(e[1] for e in evs[lo:hi]) for lo, hi in sel) / 1e3
        print(f"  {kind:-<20s}{len(sel):3d} 层 x ~{(sel[1][1]-sel[1][0]) if len(sel)>1 else (sel[0][1]-sel[0][0]):2d} kernels"
              f"  {ms:8.3f} ms  ({ms/len(sel):.4f} ms/层)")
    hk, hlo, hhi = bounds[-1]
    print(f"  head + sampler      [{hlo:5d}:{hhi:5d}]  {hhi-hlo:4d} kernels  "
          f"{sum(e[1] for e in evs[hlo:hhi])/1e3:8.3f} ms")

    print("\n== 折叠依据：逐层 kernel 序列签名 ==")
    reps = {}
    for kind in ("gdn", "full"):
        groups = {}
        for n, (k, lo, hi) in enumerate(bounds):
            if k != kind:
                continue
            groups.setdefault(signature(evs, lo, hi, wd), []).append((n, lo, hi))
        ordered = sorted(groups.items(), key=lambda kv: -len(kv[1]))
        print(f"  {kind}: {len(ordered)} 种不同签名")
        for sig, members in ordered:
            idx = [n for n, _, _ in members]
            head = f"{idx[:4]}{'...' if len(idx) > 4 else ''}"
            print(f"    x{len(members):3d} 层  {len(sig):3d} kernels  层号 {head}")
        reps[kind] = ordered

    print("\n" + "=" * 100)
    print("A. PROLOGUE — 调度与准备（每次前向一次，与模型层无关）")
    print("=" * 100)
    dump(evs, 0, prologue_end, tags, wd, args, cfg, shapes, section="prologue",
         collapse=not args.no_collapse)

    labels = {"gdn": "B. GDN 层（linear attention）", "full": "C. FULL-ATTENTION 层"}
    for kind in ("gdn", "full"):
        for gi, (sig, members) in enumerate(reps[kind]):
            if not args.show_all_variants and gi > 1:
                print(f"\n  （还有 {len(reps[kind]) - 2} 种签名，用 --show-all-variants 展开）")
                break
            n, lo, hi = members[0]
            print("\n" + "=" * 100)
            suffix = "" if gi == 0 else f"  [变体 {gi}]"
            print(f"{labels[kind]} — 代表：第 {n} 层，共 {len(members)} 层同样的序列"
                  f"（{hi-lo} kernels，{sum(e[1] for e in evs[lo:hi])/1e3:.3f} ms）{suffix}")
            print("=" * 100)
            dump(evs, lo, hi, tags, wd, args, cfg, shapes, section=kind)

    print("\n" + "=" * 100)
    print("D. HEAD + SAMPLER")
    print("=" * 100)
    dump(evs, hlo, hhi, tags, wd, args, cfg, shapes, section="head")


if __name__ == "__main__":
    main()
