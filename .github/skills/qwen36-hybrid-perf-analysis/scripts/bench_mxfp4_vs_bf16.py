"""Measure whether the XPU MXFP4 (W4A4) GEMM actually delivers low-precision
compute throughput, by timing it against a plain BF16 matmul of the same shape.

The unitrace kernel name (`gemm_kernel`) and the oneDNN verbose line
(`src:f4_e2m1 wei:f4_e2m1 dst:bf16`, impl `jit:gemm:any`) both describe the data
format, not the MAC rate. This settles it: if MXFP4 is not meaningfully faster
than BF16 at a compute-bound shape, the 4-bit path buys memory only.

  python bench_mxfp4_vs_bf16.py [--device xpu:0] [--iters 20]
"""

import argparse
import time

import torch
import vllm_xpu_kernels._xpu_C  # noqa: F401  registers torch.ops._xpu_C
import vllm._xpu_ops  # noqa: F401  registers torch.ops.vllm.xpu_mxfp4_quantize

from vllm.model_executor.layers.quantization.utils.mxfp4_utils import (
    xpu_mxfp4_quantize as quant_mxfp4,
)

# (label, M, K, N) -- the real Qwen3.6-27B linears
SHAPES = [
    ("gate_up      prefill", 3300, 5120, 34816),
    ("down_proj    prefill", 3300, 17408, 5120),
    ("in_proj_qkvz prefill", 3300, 5120, 16384),
    ("qkv_proj     prefill", 3300, 5120, 14336),
    ("gate_up      decode ", 1, 5120, 34816),
    ("down_proj    decode ", 1, 17408, 5120),
]

MXFP4_BYTES = 0.5 + 1.0 / 32
BF16_BYTES = 2.0


def timeit(fn, iters, warmup=5):
    for _ in range(warmup):
        fn()
    torch.xpu.synchronize()
    t = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.xpu.synchronize()
    return (time.perf_counter() - t) / iters * 1e6


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--device", default="xpu:0")
    p.add_argument("--iters", type=int, default=20)
    args = p.parse_args()

    dev = torch.device(args.device)
    print(f"device : {torch.xpu.get_device_name(dev)}  torch {torch.__version__}")
    print(f"iters  : {args.iters}\n")
    hdr = (f"{'shape':<21} {'M':>5} {'K':>6} {'N':>6} "
           f"{'mxfp4_us':>10} {'bf16_us':>10} {'speedup':>8} "
           f"{'mxfp4_TF':>9} {'bf16_TF':>9} {'mxfp4_GB/s':>11}")
    print(hdr)
    print("-" * len(hdr))

    for label, m, k, n in SHAPES:
        x = torch.randn(m, k, dtype=torch.bfloat16, device=dev)
        w = torch.randn(n, k, dtype=torch.bfloat16, device=dev)

        x_fp4, x_scale = quant_mxfp4(x)
        w_fp4, w_scale = quant_mxfp4(w)
        # process_weights_after_loading: oneDNN wants [K, N] plus [N, K/32] scales
        w_fp4 = w_fp4.view(torch.float4_e2m1fn_x2).t()
        w_scale = w_scale.view(torch.float8_e8m0fnu).t().contiguous()

        def run_fp4():
            torch.ops._xpu_C.fp4_gemm(x_fp4, w_fp4, x_scale, w_scale,
                                      torch.bfloat16, None)

        wt = w.t().contiguous()

        def run_bf16():
            torch.matmul(x, wt)

        t4 = timeit(run_fp4, args.iters)
        tb = timeit(run_bf16, args.iters)
        flops = 2.0 * m * n * k
        wbytes = k * n * MXFP4_BYTES
        print(f"{label:<21} {m:>5} {k:>6} {n:>6} "
              f"{t4:>10.1f} {tb:>10.1f} {tb/t4:>8.2f}x "
              f"{flops/t4/1e6:>9.1f} {flops/tb/1e6:>9.1f} "
              f"{wbytes/t4/1e3:>11.1f}")

    print("\nspeedup = bf16_us / mxfp4_us.  >1 means MXFP4 is faster.")
    print("At the prefill shapes (compute bound) a native FP4 XMX path should be "
          "several x\nfaster than BF16; ~1x means the 4-bit inputs are being "
          "decompressed and the MACs\nrun at the BF16 rate, so 4-bit buys memory "
          "traffic only.")


if __name__ == "__main__":
    main()
