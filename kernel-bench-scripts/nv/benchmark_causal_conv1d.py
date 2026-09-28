#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Benchmark vLLM's CUDA (Triton) causal_conv1d kernels on NVIDIA GPUs.

This is the NV counterpart of
`vllm-xpu-kernels/benchmark/benchmark_causal_conv1d.py`, which benchmarks
`torch.ops._xpu_C.causal_conv1d` on Intel XPU. The workload list and the
timing protocol (5 warmup iterations, then CUDA/XPU-event timing over N
iterations) are kept identical so the two result tables can be put side by
side.

Two kernels are covered, matching how vLLM dispatches GDN / Mamba linear
attention:

    prefill (varlen)  -> causal_conv1d_fn      -> _causal_conv1d_fwd_kernel
    decode  (1 token) -> causal_conv1d_update  -> _causal_conv1d_update_kernel

IMPORTANT - op scope differs from XPU:
    The XPU op `torch.ops._xpu_C.causal_conv1d` is fused: it splits
    projected_states_{qkvz,ba}, runs the depthwise conv + SiLU, updates
    conv_state AND writes out the {q, k, v, b, a} intermediates plus z.
    The CUDA path here only does depthwise conv + activation on an already
    split `x` of shape (dim, tokens); the split/post-conv reshape lives in a
    separate kernel (`_fused_post_conv*`). So a strict XPU-vs-NV ratio built
    from these two numbers alone slightly favours NV - add the post-conv
    kernel time from a real trace before drawing conclusions.

Usage:
    python kernel-bench-scripts/nv/benchmark_causal_conv1d.py
    python kernel-bench-scripts/nv/benchmark_causal_conv1d.py --csv out.csv
    python kernel-bench-scripts/nv/benchmark_causal_conv1d.py --filter Decode
"""

from __future__ import annotations

import argparse
import csv
import gc
import sys
from dataclasses import dataclass

import torch

try:
    from vllm.model_executor.layers.mamba.ops.causal_conv1d import (
        causal_conv1d_fn,
        causal_conv1d_update,
    )
except ImportError as exc:  # pragma: no cover
    sys.exit(f"Failed to import vLLM causal_conv1d ops: {exc}\n"
             "Run this on a machine with vLLM (>=0.29) installed.")

# Index 0 of the conv-state cache is the reserved null block.
NULL_BLOCK_ID = 0

DEVICE = "cuda"


# ---------------------------------------------------------------------------
# Model shape
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class GdnShape:
    """Linear-attention shape. Defaults match Qwen3-Next / Qwen3.x GDN."""

    num_k_heads: int = 16
    num_v_heads: int = 32
    head_k_dim: int = 128
    head_v_dim: int = 128
    conv_width: int = 4
    tp_size: int = 1

    @property
    def conv_dim(self) -> int:
        """mixed_qkv_size: the channel count the depthwise conv runs over."""
        nk = self.num_k_heads // self.tp_size
        return nk * (2 * self.head_k_dim
                     + self.head_v_dim * self.num_v_heads // self.num_k_heads)


# ---------------------------------------------------------------------------
# Workloads
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Workload:
    name: str
    mode: str                 # "prefill" | "decode"
    seqlens: tuple[int, ...]  # per-sequence token counts

    @property
    def num_tokens(self) -> int:
        return sum(self.seqlens)

    @property
    def batch(self) -> int:
        return len(self.seqlens)


def build_workloads() -> list[Workload]:
    wl: list[Workload] = []

    # Single-sequence prefill: latency vs. total tokens.
    for n in (2048, 4096, 8192, 16384, 32768, 65536):
        wl.append(Workload(f"Prefill [{n}]", "prefill", (n,)))

    # Same 8192 total tokens, different varlen splits: isolates the effect of
    # the sequence-length distribution from the effect of total token count.
    for lens in ((6144, 2048), (4096, 4096), (2048, 6144), (1024, 7168)):
        label = ",".join(str(x) for x in lens)
        wl.append(Workload(f"Prefill [{label}]", "prefill", lens))

    # Uniform multi-sequence prefill batches.
    for n, reps in ((1024, 4), (1024, 8), (1024, 16),
                    (2048, 4),
                    (4096, 4), (4096, 8), (4096, 16),
                    (8192, 4), (8192, 8), (8192, 16), (8192, 32)):
        wl.append(Workload(f"Prefill [{n}]x{reps}", "prefill", (n,) * reps))

    # Pure decode: one new token per sequence.
    for b in (1, 4, 8, 16, 32, 64, 128, 256, 512):
        wl.append(Workload(f"Decode B={b}, T=1", "decode", (1,) * b))

    return wl


# ---------------------------------------------------------------------------
# Input construction
# ---------------------------------------------------------------------------
def _cache_lines(batch: int) -> int:
    # +1 so state indices can start at 1 and leave block 0 as the null block.
    return max(256, batch * 2) + 1


def make_prefill_inputs(shape: GdnShape, wl: Workload, dtype: torch.dtype):
    dim = shape.conv_dim
    width = shape.conv_width
    n_tok = wl.num_tokens
    batch = wl.batch
    cache_lines = _cache_lines(batch)

    # The Triton kernel requires channel-last x: stride(0) == 1.
    x = torch.randn(n_tok, dim, device=DEVICE, dtype=dtype).transpose(0, 1)
    weight = torch.randn(dim, width, device=DEVICE, dtype=dtype)
    bias = torch.randn(dim, device=DEVICE, dtype=dtype)
    conv_states = torch.randn(cache_lines, width - 1, dim,
                              device=DEVICE, dtype=dtype).transpose(1, 2)

    qsl_cpu = torch.zeros(batch + 1, dtype=torch.int32)
    qsl_cpu[1:] = torch.tensor(wl.seqlens, dtype=torch.int32).cumsum(0)
    query_start_loc = qsl_cpu.to(DEVICE)

    cache_indices = torch.arange(1, batch + 1, dtype=torch.int32, device=DEVICE)
    has_initial_state = torch.ones(batch, dtype=torch.bool, device=DEVICE)

    metadata = _make_metadata(qsl_cpu)

    return dict(x=x, weight=weight, bias=bias, conv_states=conv_states,
                query_start_loc=query_start_loc, cache_indices=cache_indices,
                has_initial_state=has_initial_state, metadata=metadata)


def make_decode_inputs(shape: GdnShape, wl: Workload, dtype: torch.dtype):
    dim = shape.conv_dim
    width = shape.conv_width
    batch = wl.batch
    cache_lines = _cache_lines(batch)

    x = torch.randn(batch, dim, device=DEVICE, dtype=dtype)
    weight = torch.randn(dim, width, device=DEVICE, dtype=dtype)
    bias = torch.randn(dim, device=DEVICE, dtype=dtype)
    conv_state = torch.randn(cache_lines, width - 1, dim,
                             device=DEVICE, dtype=dtype).transpose(1, 2)
    conv_state_indices = torch.arange(1, batch + 1, dtype=torch.int32,
                                      device=DEVICE)
    out = torch.empty_like(x)

    return dict(x=x, weight=weight, bias=bias, conv_state=conv_state,
                conv_state_indices=conv_state_indices, out=out)


def _make_metadata(query_start_loc_cpu: torch.Tensor):
    """Precompute the kernel's launch metadata, like the GDN backend does.

    Without it causal_conv1d_fn redoes a CPU/numpy grid computation on every
    call, which would show up as host-side overhead in the measurement.
    """
    try:
        from types import SimpleNamespace

        from vllm.v1.attention.backends.utils import (
            compute_causal_conv1d_metadata,
        )
    except ImportError:
        return None

    nums_dict, batch_ptr, token_chunk_offset_ptr = (
        compute_causal_conv1d_metadata(query_start_loc_cpu,
                                       device=torch.device(DEVICE)))
    return SimpleNamespace(nums_dict=nums_dict, batch_ptr=batch_ptr,
                           token_chunk_offset_ptr=token_chunk_offset_ptr)


# ---------------------------------------------------------------------------
# Memory / FLOPs models (conv stage only - keep in sync with the XPU script)
# ---------------------------------------------------------------------------
def estimate_bytes_moved(shape: GdnShape, wl: Workload,
                         dtype: torch.dtype) -> int:
    bpe = torch.tensor([], dtype=dtype).element_size()
    dim = shape.conv_dim
    width = shape.conv_width
    n_tok = wl.num_tokens

    bytes_x = n_tok * dim * bpe
    bytes_out = n_tok * dim * bpe
    bytes_conv_state = wl.batch * (width - 1) * dim * bpe * 2  # read + write
    bytes_weight = dim * width * bpe
    bytes_bias = dim * bpe
    return bytes_x + bytes_out + bytes_conv_state + bytes_weight + bytes_bias


def estimate_flops(shape: GdnShape, wl: Workload) -> int:
    return 2 * wl.num_tokens * shape.conv_dim * shape.conv_width


# ---------------------------------------------------------------------------
# Timing
# ---------------------------------------------------------------------------
def time_us(fn, iters: int, warmup: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters * 1000.0


def run_workload(shape: GdnShape, wl: Workload, dtype: torch.dtype,
                 iters: int, warmup: int) -> float:
    if wl.mode == "prefill":
        kw = make_prefill_inputs(shape, wl, dtype)

        def _run():
            causal_conv1d_fn(kw["x"], kw["weight"], kw["bias"],
                             conv_states=kw["conv_states"],
                             query_start_loc=kw["query_start_loc"],
                             cache_indices=kw["cache_indices"],
                             has_initial_state=kw["has_initial_state"],
                             activation="silu",
                             metadata=kw["metadata"])
    else:
        kw = make_decode_inputs(shape, wl, dtype)

        def _run():
            causal_conv1d_update(kw["x"], kw["conv_state"], kw["weight"],
                                 kw["bias"], activation="silu",
                                 conv_state_indices=kw["conv_state_indices"],
                                 out=kw["out"])

    us = time_us(_run, iters, warmup)

    del kw
    torch.cuda.empty_cache()
    gc.collect()
    return us


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dtype", choices=["bf16", "fp16"], default="bf16")
    p.add_argument("--iters", type=int, default=25,
                   help="Timed iterations (XPU script uses 30-5=25)")
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--num-k-heads", type=int, default=16)
    p.add_argument("--num-v-heads", type=int, default=32)
    p.add_argument("--head-k-dim", type=int, default=128)
    p.add_argument("--head-v-dim", type=int, default=128)
    p.add_argument("--conv-width", type=int, default=4)
    p.add_argument("--tp-size", type=int, default=1)
    p.add_argument("--filter", default=None,
                   help="Only run workloads whose name contains this substring")
    p.add_argument("--csv", default=None, help="Write results to this CSV file")
    args = p.parse_args()

    if not torch.cuda.is_available():
        sys.exit("CUDA device not available.")

    torch.manual_seed(1234)
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    shape = GdnShape(args.num_k_heads, args.num_v_heads, args.head_k_dim,
                     args.head_v_dim, args.conv_width, args.tp_size)

    workloads = build_workloads()
    if args.filter:
        workloads = [w for w in workloads if args.filter in w.name]
    if not workloads:
        sys.exit("No workload matched --filter.")

    print(f"Device : {torch.cuda.get_device_name(0)}")
    print(f"dtype  : {args.dtype}   conv_dim={shape.conv_dim}   "
          f"width={shape.conv_width}   tp={shape.tp_size}")
    print(f"timing : {args.warmup} warmup + {args.iters} timed iterations\n")

    header = (f"{'Stage / Shape':<26}{'tokens':>8}{'batch':>7}"
              f"{'latency(us)':>13}{'GB/s':>10}{'TFLOPS':>9}")
    print(header)
    print("-" * len(header))

    rows = []
    for wl in workloads:
        us = run_workload(shape, wl, dtype, args.iters, args.warmup)
        gbs = estimate_bytes_moved(shape, wl, dtype) / 1e9 / (us / 1e6)
        tflops = estimate_flops(shape, wl) / (us / 1e6) / 1e12
        print(f"{wl.name:<26}{wl.num_tokens:>8}{wl.batch:>7}"
              f"{us:>13.3f}{gbs:>10.1f}{tflops:>9.2f}")
        rows.append({"stage_shape": wl.name, "mode": wl.mode,
                     "tokens": wl.num_tokens, "batch": wl.batch,
                     "latency_us": round(us, 3), "gbps": round(gbs, 1),
                     "tflops": round(tflops, 3)})

    if args.csv:
        with open(args.csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nSaved: {args.csv}")


if __name__ == "__main__":
    main()
