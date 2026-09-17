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


def layer_kinds(cfg=CFG):
    """['gdn', 'gdn', 'gdn', 'full', ...] following full_attention_interval."""
    k = cfg["full_attention_interval"]
    return ["full" if (i + 1) % k == 0 else "gdn" for i in range(cfg["layers"])]


GDN_LINEARS = ["in_proj_qkvz", "in_proj_ba", "gdn_out_proj", "gate_up", "down_proj"]
FULL_LINEARS = ["qkv_proj", "o_proj", "gate_up", "down_proj"]


# --------------------------------------------------------------------------
# kernel name markers
# --------------------------------------------------------------------------

# torch.compile (Inductor) emits stable fused-kernel names; these are the
# anchors the layer walker relies on.  Verified against the 260917 trace.
M_NORM_IN_GDN = "triton_red_fused__to_copy_add_fused_add_rms_norm_3"
M_NORM_IN_FULL = "triton_red_fused__to_copy_add_fused_add_rms_norm_mm_view_3"
M_NORM_POST_ATTN = "triton_red_fused__to_copy_add_fused_add_rms_norm_mm_view_1"
M_GDN_GATED_NORM = "triton_per_fused__to_copy_add_mean_mm_mul_pow_rsqrt_silu_view_0"
M_FULL_OUT_GATE = "triton_poi_fused_mm_mul_sigmoid_view_0"
M_SILU = "triton_poi_fused_mm_mul_silu_slice_view_2"
M_GDN_ZEROS = "triton_poi_fused_zeros_4"

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


def is_main_gemm(name):
    """Distinguish a model linear from a Hadamard-rotation matmul."""
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


def bucket(full):
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
        return "Dense-GEMM" if is_main_gemm(full) else "Hadamard-Rotation"
    if RE_KVWRITE.search(name):
        return "KVCache-Write"
    if RE_QUANT.search(name):
        return "Quantize(mxfp4)"
    if RE_SCALE.search(name):
        return "Quant-scale cast"
    if M_GDN_GATED_NORM in name:
        return "GDN-Norm/Gate"
    if M_FULL_OUT_GATE in name:
        return "FullAttn-OutGate"
    if M_SILU in name:
        return "Activation(SiLU)"
    if name.startswith(M_NORM_IN_GDN) or name.startswith(M_NORM_IN_FULL) \
            or name.startswith(M_NORM_POST_ATTN) or "rms_norm" in name:
        return "Norm(RMS)"
    if RE_QK_NORM_ROPE.search(name):
        return "QK-Norm/RoPE"
    if M_GDN_ZEROS in name:
        return "GDN-State-Init"
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

def walk_layers(evlist, cfg=CFG):
    """Segment one forward pass into decoder layers and label its GEMMs.

    Returns (layers, gemm_tags) where
      layers    = [(kind, lo, hi)]  kind in {'gdn', 'full', 'head'}
      gemm_tags = {index_in_evlist: linear_name}

    The decoder emits a fixed kernel order which the Inductor fusion names make
    unambiguous:
      gdn  : [H,quant x2 -> zeros] -> qkvz, ba -> conv1d -> delta-rule
             -> gated-norm -> H,quant -> out_proj -> post-norm -> H,quant
             -> gate_up -> silu -> H,quant -> down_proj -> IN-NORM(next)
      full : [H,quant] -> qkv -> qk-norm/rope -> kv-write -> fmha -> out-gate
             -> H,quant -> o_proj -> post-norm -> ...same MLP... -> IN-NORM(next)

    The fused `..._rms_norm_3` / `..._rms_norm_mm_view_3` kernel is the residual
    add plus the *next* layer's input norm, so it terminates a layer and names
    the kind of the one that follows.  There are num_layers such markers per
    forward: num_layers-1 layer starts plus the model's final norm.
    """
    marks = []
    for i, (_, _, n) in enumerate(evlist):
        if n.startswith(M_NORM_IN_GDN):
            marks.append((i, "gdn"))
        elif n.startswith(M_NORM_IN_FULL):
            marks.append((i, "full"))
    if not marks:
        return [], {}
    kinds = layer_kinds(cfg)
    layers = [(kinds[0], 0, marks[0][0] + 1)]
    for j in range(len(marks) - 1):
        layers.append((marks[j][1], marks[j][0] + 1, marks[j + 1][0] + 1))
    layers.append((marks[-1][1], marks[-1][0] + 1, len(evlist)))
    # the final norm produced one marker too many: everything after the last
    # real decoder layer is the lm_head / sampling tail
    n_layers = cfg["layers"]
    if len(layers) > n_layers:
        tail_lo = layers[n_layers][1]
        layers = layers[:n_layers] + [("head", tail_lo, len(evlist))]

    order = {"gdn": GDN_LINEARS, "full": FULL_LINEARS, "head": ["lm_head"]}
    tags = {}
    for kind, lo, hi in layers:
        seq = list(order[kind])
        k = 0
        for i in range(lo, hi):
            name = evlist[i][2]
            if not is_gemm(name) or not is_main_gemm(name):
                continue
            if k < len(seq):
                # in_proj_ba is occasionally scheduled before in_proj_qkvz; its
                # N=96 output makes the launch geometry unmistakable at M=1
                if seq[k] == "in_proj_qkvz" and is_tiny_gemm(name):
                    seq[k], seq[k + 1] = seq[k + 1], seq[k]
                tags[i] = seq[k]
            else:
                tags[i] = "unattributed"
            k += 1
    return layers, tags
