#!/usr/bin/env python3
"""Shared model topology, kernel taxonomy and trace walker for Qwen3.5/3.6
hybrid (GDN linear-attention + full-attention) unitrace analysis on Intel XPU.

The two analysis scripts import from here so that the category taxonomy, the
layer segmentation and the GEMM attribution can never drift apart.
"""

import re
import sys
from collections import Counter

try:
    import ijson
except ImportError:                                       # pragma: no cover
    sys.exit("pip install ijson")


# --------------------------------------------------------------------------
# model configuration (Qwen3.6-27B / qwen3_5_text)
# --------------------------------------------------------------------------

CFG = dict(
    name="Qwen3.6-27B",
    hidden=5120,
    layers=64,
    full_attention_interval=4,      # layer_types: 3 x linear_attention + 1 x full
    inter=17408,
    vocab=248320,
    # full attention
    heads=24,
    kv_heads=4,
    head_dim=256,
    attn_output_gate=True,          # q_proj also emits a gate of the same width
    # gated delta net (linear attention)
    gdn_k_heads=16,
    gdn_k_dim=128,
    gdn_v_heads=48,
    gdn_v_dim=128,
    gdn_conv_kernel=4,
    gdn_state_dtype_bytes=4,        # mamba_ssm_dtype = float32
)

MXFP4_BYTES = 0.5 + 1.0 / 32        # 4-bit element + one uint8 E8M0 per 32
BF16_BYTES = 2.0


def derive(cfg=CFG):
    """Per-layer linear shapes (K, N, quantized) for both layer types."""
    h, inter = cfg["hidden"], cfg["inter"]
    hq, hkv, d = cfg["heads"], cfg["kv_heads"], cfg["head_dim"]
    q_out = hq * d * (2 if cfg["attn_output_gate"] else 1)
    kv_out = 2 * hkv * d
    gk, gkd = cfg["gdn_k_heads"], cfg["gdn_k_dim"]
    gv, gvd = cfg["gdn_v_heads"], cfg["gdn_v_dim"]
    return {
        # full-attention layer
        "qkv_proj":  (h, q_out + kv_out, True),
        "o_proj":    (hq * d, h, True),
        # gdn (linear-attention) layer; the checkpoint stores in_proj_qkv/_z and
        # in_proj_b/_a separately, vLLM fuses each pair at load time
        "in_proj_qkvz": (h, 2 * gk * gkd + gv * gvd + gv * gvd, True),
        "in_proj_ba":   (h, 2 * gv, True),
        "gdn_out_proj": (gv * gvd, h, True),
        # shared MLP
        "gate_up":   (h, 2 * inter, True),
        "down_proj": (inter, h, True),
        # head
        "lm_head":   (h, cfg["vocab"], False),
    }


def derive(cfg=CFG, weight_dtype="mxfp4"):
    """Per-layer linear shapes (K, N, quantized) for both layer types.

    The BF16 checkpoint keeps the GDN input projection as a single fused linear,
    so it has 4 linears per GDN layer where the quantized one has 5.
    """
    q = weight_dtype != "bf16"
    h, inter = cfg["hidden"], cfg["inter"]
    hq, hkv, d = cfg["heads"], cfg["kv_heads"], cfg["head_dim"]
    q_out = hq * d * (2 if cfg["attn_output_gate"] else 1)
    kv_out = 2 * hkv * d
    gk, gkd = cfg["gdn_k_heads"], cfg["gdn_k_dim"]
    gv, gvd = cfg["gdn_v_heads"], cfg["gdn_v_dim"]
    qkvz_n = 2 * gk * gkd + 2 * gv * gvd
    ba_n = 2 * gv
    shapes = {
        # full-attention layer
        "qkv_proj":  (h, q_out + kv_out, q),
        "o_proj":    (hq * d, h, q),
        "gdn_out_proj": (gv * gvd, h, q),
        # shared MLP
        "gate_up":   (h, 2 * inter, q),
        "down_proj": (inter, h, q),
        # head -- never quantized in either checkpoint
        "lm_head":   (h, cfg["vocab"], False),
    }
    if q:
        shapes["in_proj_qkvz"] = (h, qkvz_n, True)
        shapes["in_proj_ba"] = (h, ba_n, True)
    else:
        shapes["in_proj"] = (h, qkvz_n + ba_n, False)
    return shapes


def layer_kinds(cfg=CFG):
    """['gdn', 'gdn', 'gdn', 'full', ...] following full_attention_interval."""
    k = cfg["full_attention_interval"]
    return ["full" if (i + 1) % k == 0 else "gdn" for i in range(cfg["layers"])]


GDN_LINEARS = {
    "mxfp4": ["in_proj_qkvz", "in_proj_ba", "gdn_out_proj", "gate_up", "down_proj"],
    "bf16": ["in_proj", "gdn_out_proj", "gate_up", "down_proj"],
}
FULL_LINEARS = ["qkv_proj", "o_proj", "gate_up", "down_proj"]


# --------------------------------------------------------------------------
# kernel name markers
# --------------------------------------------------------------------------

# torch.compile (Inductor) renumbers its fused kernels when the graph changes,
# so every marker below must match both the MXFP4 and the BF16 variant.  The
# reliable way to re-derive them is to dump one decode window and find the
# kernels whose per-forward counts are num_layers / G / F.
#   MXFP4: ..._rms_norm_3 (gdn) and ..._rms_norm_mm_view_3 (full) are distinct
#   BF16 : both collapse into ..._rms_norm_3, so the layer kind must come from
#          the segment's contents instead of the marker name
RE_NORM_IN = re.compile(r"_fused_add_rms_norm(_mm_view)?_3$")
RE_NORM_POST = re.compile(r"_fused_add_rms_norm(_mm_view)?_1$")
RE_GDN_GATED_NORM = re.compile(r"rsqrt_silu(_t)?_view_0$")
RE_FULL_OUT_GATE = re.compile(r"fused_(mm_)?mul_sigmoid_view_0$")
RE_SILU = re.compile(r"fused_(mm_)?mul_silu_slice(_view)?_2$")
RE_GDN_STATE = re.compile(r"triton_poi_fused_zeros_[0-9]+$")
RE_GDN_CAT = re.compile(r"triton_poi_fused_cat_[0-9]+$")

RE_QK_NORM_ROPE = re.compile(
    r"triton_poi_fused_4$|triton_red_fused_5$|"
    r"triton_poi_fused_arange_bitwise_and_eq_index_lt_remainder_select_split_where_6$|"
    r"triton_poi_fused_7$|"
    r"triton_poi_fused_0$|triton_red_fused_1$|triton_poi_fused_cat_neg|"
    r"triton_poi_fused__to_copy_cat_mul_5$")

RE_GDN_DECODE = re.compile(r"gdn::causal_conv1d_kernel|gdn::gated_delta_rule_kernel")
RE_GDN_CHUNK = re.compile(r"gdn::Chunk|gdn::chunk_update_states|gdn::tiled_kernel_launcher")
RE_FMHA_PREFILL = re.compile(r"XeFMHAFwdKernel")
RE_FMHA_DECODE = re.compile(r"XeFMHAFwdSplitKVKernel")
RE_FMHA_REDUCE = re.compile(r"ReduceSplitK")
RE_QUANT = re.compile(r"per_token_group_quant_mxfp4|_quant_|quantize|float_e2m1|fp4",
                      re.IGNORECASE)
RE_SCALE = re.compile(r"Float8_e8m0|e8m0")
RE_KVWRITE = re.compile(r"reshape_and_cache")
RE_SAMPLE = re.compile(r"ArgMax|argmax|top_k_top_p|_gumbel_sample|Sampler")
RE_SCHED = re.compile(r"slot_mapping|block_table|_prepare_|_post_update|"
                      r"_combine_sampled|_zero_kv_blocks|_apply_write")
RE_MEMCPY = re.compile(r"MemoryCopy|Memcpy|memcpy")
RE_ELTWISE = re.compile(r"elementwise_kernel|IndexKernelFunctor|IndexSelect|"
                        r"gather_kernel|FillFunctor|CopyScalarFunc|reduce_kernel|"
                        r"RoundedRangeKernel|Transpose")

RE_ND = re.compile(r"\[SIMD(\d+) \{(\d+); (\d+); (\d+)\} \{(\d+); (\d+); (\d+)\}\]")

# a gemm_kernel is one of the model's real linears (as opposed to the online
# Hadamard rotation that AutoRound inserts in front of every quantised linear)
# iff its work-group shape is one of these.  Verified: 305 main GEMMs and 1168
# (prefill) / 304 (decode) Hadamard launches per forward.
MAIN_GEMM_LOCAL = {(128, 4, 1), (32, 2, 8)}
DECODE_MAIN_LOCAL = (64, 8, 1)


def nd_range(name):
    m = RE_ND.search(name)
    if not m:
        return None
    g = tuple(int(m.group(i)) for i in (2, 3, 4))
    l = tuple(int(m.group(i)) for i in (5, 6, 7))
    return g, l


def is_gemm(name):
    return name.startswith("gemm_kernel")


def is_main_gemm(name, weight_dtype="mxfp4"):
    """Distinguish a model linear from a Hadamard-rotation matmul.

    A BF16 checkpoint has no rotation_config, so every gemm_kernel is a linear.
    """
    if weight_dtype == "bf16":
        return True
    nd = nd_range(name)
    if nd is None:
        return False
    grid, local = nd
    if local in MAIN_GEMM_LOCAL:
        return True
    return local == DECODE_MAIN_LOCAL and grid[0] != 1


def base_name(name):
    """Kernel name without its trailing ND-range."""
    lb = name.rfind("[")
    return name[:lb] if lb > 0 and name.endswith("]") else name


def is_tiny_gemm(name):
    """A decode GEMM launched with a single work-group: only in_proj_ba (N=96)."""
    nd = nd_range(name)
    return bool(nd) and nd[0] == (1, 1, 1) and nd[1] == (128, 4, 1)


def bucket(full, weight_dtype="mxfp4"):
    """Category taxonomy.  Order matters -- see references/pitfalls.md."""
    name = base_name(full)
    if RE_FMHA_REDUCE.search(name):
        return "FullAttn-SplitK-Reduce"
    if RE_FMHA_PREFILL.search(name) or RE_FMHA_DECODE.search(name):
        return "FullAttn-FMHA"
    if RE_GDN_CHUNK.search(name):
        return "GDN-Attn(chunk)"
    if RE_GDN_DECODE.search(name):
        return "GDN-Attn(recurrent)"
    if is_gemm(name):
        return "Dense-GEMM" if is_main_gemm(full, weight_dtype) else "Hadamard-Rotation"
    if RE_KVWRITE.search(name):
        return "KVCache-Write"
    if RE_QUANT.search(name):
        return "Quantize(mxfp4)"
    if RE_SCALE.search(name):
        return "Quant-scale cast"
    if RE_GDN_GATED_NORM.search(name):
        return "GDN-Norm/Gate"
    if RE_FULL_OUT_GATE.search(name):
        return "FullAttn-OutGate"
    if RE_SILU.search(name):
        return "Activation(SiLU)"
    if RE_NORM_IN.search(name) or RE_NORM_POST.search(name) or "rms_norm" in name:
        return "Norm(RMS)"
    if RE_QK_NORM_ROPE.search(name):
        return "QK-Norm/RoPE"
    if RE_GDN_STATE.search(name):
        return "GDN-State-Init"
    if RE_GDN_CAT.search(name):
        return "GDN-Concat(in_proj)"
    if RE_SAMPLE.search(name):
        return "Sampling"
    if RE_SCHED.search(name):
        return "Sched/Prep"
    if RE_MEMCPY.search(name):
        return "MemCopy"
    if RE_ELTWISE.search(name):
        return "Elementwise/Layout"
    return "Other"


CATEGORY_ORDER = [
    "Dense-GEMM", "GDN-Attn(chunk)", "GDN-Attn(recurrent)", "FullAttn-FMHA",
    "FullAttn-SplitK-Reduce", "Hadamard-Rotation", "Quantize(mxfp4)",
    "Quant-scale cast", "Norm(RMS)", "GDN-Norm/Gate", "QK-Norm/RoPE",
    "Activation(SiLU)", "FullAttn-OutGate", "KVCache-Write", "GDN-State-Init",
    "Sampling", "Sched/Prep", "MemCopy", "Elementwise/Layout", "Other",
]


# --------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------

def load(path, max_kernel_s=10.0, verbose=True):
    """Stream the Chrome trace and return [(ts, dur, name)] in capture order."""
    evs, dropped = [], []
    lim = max_kernel_s * 1e6
    with open(path, "rb") as f:
        for ev in ijson.items(f, "traceEvents.item"):
            if ev.get("ph") != "X" or ev.get("cat") != "gpu_op":
                continue
            dur = float(ev.get("dur", 0) or 0)
            name = ev.get("name", "")
            if dur > lim:
                dropped.append((dur, name))
                continue
            evs.append((float(ev.get("ts", 0) or 0), dur, name))
    if not evs:
        sys.exit("no gpu_op events found")
    # stable sort on ts only: XPU-Graph gives every kernel of a replayed graph
    # the graph's submit timestamp, and only a stable sort keeps capture order
    evs.sort(key=lambda e: e[0])
    if verbose and dropped:
        print(f"WARNING: dropped {len(dropped)} unterminated kernel(s) "
              f"(dur > {max_kernel_s:g}s, in flight when tracing stopped):")
        for dur, name in dropped[:4]:
            print(f"    dur={dur/1e6:.0f}s  {name[:78]}")
    return evs


def step_windows(evs):
    """Sampler-delimited forward passes.  Window 0 contains the prefill."""
    ends = [i for i, (_, _, n) in enumerate(evs) if RE_SAMPLE.search(n)]
    wins, prev = [], 0
    for e in ends:
        wins.append((prev, e + 1))
        prev = e + 1
    return wins


def split_graph_blocks(evs, lo, hi, min_block=64):
    """Separate a window into its own kernels and misplaced graph replays.

    A full XPU-Graph replay is submitted as one L0 command list, so unitrace
    stamps all of its kernels with the submit time.  When that submit time
    falls inside the (eagerly executed, correctly stamped) prefill, the whole
    replay sorts into the middle of the prefill window and silently inflates
    it.  Any timestamp shared by >= min_block kernels is such a replay.
    """
    cnt = Counter(ts for ts, _, _ in evs[lo:hi])
    blocks = {ts for ts, c in cnt.items() if c >= min_block}
    own = [e for e in evs[lo:hi] if e[0] not in blocks]
    alien = [e for e in evs[lo:hi] if e[0] in blocks]
    return own, alien


def ts_collapsed(evlist):
    cnt = Counter(ts for ts, _, _ in evlist)
    shared = sum(c for c in cnt.values() if c > 1)
    return shared / max(1, len(evlist)), (cnt.most_common(1)[0][1] if cnt else 0)


# --------------------------------------------------------------------------
# layer walker: assigns every main GEMM to a named linear
# --------------------------------------------------------------------------

def walk_layers(evlist, cfg=CFG, weight_dtype="mxfp4"):
    """Segment one forward pass into decoder layers and label its GEMMs.

    Returns (layers, gemm_tags) where
      layers    = [(kind, lo, hi)]  kind in {'gdn', 'full', 'head'}
      gemm_tags = {index_in_evlist: linear_name}

    The decoder emits a fixed kernel order which the Inductor fusion names make
    unambiguous:
      gdn  : [H,quant] -> in_proj(s) -> conv1d -> delta-rule -> gated-norm
             -> [H,quant] -> out_proj -> post-norm -> [H,quant] -> gate_up
             -> silu -> [H,quant] -> down_proj -> IN-NORM(next)
      full : [H,quant] -> qkv -> qk-norm/rope -> kv-write -> fmha -> out-gate
             -> [H,quant] -> o_proj -> post-norm -> ...same MLP... -> IN-NORM(next)

    The fused `..._rms_norm*_3` kernel is the residual add plus the *next*
    layer's input norm, so it terminates a layer.  It appears num_layers times
    per forward: num_layers-1 layer starts plus the model's final norm.

    The layer kind comes from the segment's *contents* (gdn:: kernels vs FMHA /
    KV-write), not from the marker name: the MXFP4 build emits two distinct norm
    kernels for the two layer types but the BF16 build collapses them into one.
    """
    marks = [i for i, (_, _, n) in enumerate(evlist)
             if RE_NORM_IN.search(base_name(n))]
    if not marks:
        return [], {}
    bounds = [(0, marks[0] + 1)]
    for j in range(len(marks) - 1):
        bounds.append((marks[j] + 1, marks[j + 1] + 1))
    bounds.append((marks[-1] + 1, len(evlist)))
    # the final norm produced one boundary too many: everything after the last
    # real decoder layer is the lm_head / sampling tail
    n_layers = cfg["layers"]
    tail = None
    if len(bounds) > n_layers:
        tail = (bounds[n_layers][0], len(evlist))
        bounds = bounds[:n_layers]

    expected = layer_kinds(cfg)
    layers = []
    for n, (lo, hi) in enumerate(bounds):
        kind = _segment_kind(evlist, lo, hi)
        if kind is None:                    # window started mid-layer
            kind = expected[n] if n < len(expected) else "gdn"
        layers.append((kind, lo, hi))
    if tail:
        layers.append(("head", tail[0], tail[1]))

    order = {"gdn": GDN_LINEARS[weight_dtype], "full": FULL_LINEARS,
             "head": ["lm_head"]}
    kind_at = {}
    for kind, lo, hi in layers:
        for i in range(lo, hi):
            kind_at[i] = kind
    tags = {}
    for i, (_, _, name) in enumerate(evlist):
        if not is_gemm(name) or not is_main_gemm(name, weight_dtype):
            continue
        tags[i] = _gemm_role(evlist, i, kind_at.get(i, "gdn"), weight_dtype,
                             order)
    return layers, tags


def _segment_kind(evlist, lo, hi):
    for i in range(lo, hi):
        n = base_name(evlist[i][2])
        if RE_GDN_DECODE.search(n) or RE_GDN_CHUNK.search(n) or RE_GDN_CAT.search(n):
            return "gdn"
        if RE_FMHA_PREFILL.search(n) or RE_FMHA_DECODE.search(n) \
                or RE_KVWRITE.search(n):
            return "full"
    return None


def _gemm_role(evlist, i, kind, weight_dtype, order, lookahead=48):
    """Name a linear from the op that consumes its output.

    Positional assignment inside a layer segment breaks whenever the graph
    partitioner rotates a forward pass, which it does: the BF16 decode window
    starts in the middle of layer 0's MLP and layer 0's attention block lands
    after the final norm.  Looking forward to the first role-defining kernel is
    immune to that, and gives the same answer for both checkpoints.
    """
    consumers = order.get(kind, [])
    for j in range(i + 1, min(i + lookahead, len(evlist))):
        n = base_name(evlist[j][2])
        if RE_SILU.search(n):
            return "gate_up"
        if RE_NORM_IN.search(n):
            return "down_proj"
        if RE_NORM_POST.search(n) or RE_FULL_OUT_GATE.search(n) \
                or RE_GDN_GATED_NORM.search(n):
            return "gdn_out_proj" if kind == "gdn" else "o_proj"
        if RE_GDN_DECODE.search(n) or RE_GDN_CHUNK.search(n) \
                or RE_GDN_CAT.search(n) or RE_GDN_STATE.search(n):
            if weight_dtype == "bf16":
                return "in_proj"
            return "in_proj_ba" if is_tiny_gemm(evlist[i][2]) else "in_proj_qkvz"
        if RE_QK_NORM_ROPE.search(n) or RE_KVWRITE.search(n) \
                or RE_FMHA_PREFILL.search(n) or RE_FMHA_DECODE.search(n):
            return "qkv_proj"
        if RE_SAMPLE.search(n):
            return "lm_head"
    return consumers[0] if consumers else "unattributed"
