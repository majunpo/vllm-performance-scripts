import argparse
import gc
import math
import os
import sys
import time
from datetime import datetime

os.environ.setdefault("ZE_AFFINITY_MASK", "4,5,6,7")

# Set the following env vars to enable XPU graph capture and breakable graph support.
# os.environ.setdefault("VLLM_USE_BREAKABLE_CUDAGRAPH", "1")
os.environ.setdefault("VLLM_XPU_ENABLE_XPU_GRAPH", "1")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LOCAL_VLLM_SRC = os.path.join(SCRIPT_DIR, "vllm")
if os.path.isfile(os.path.join(LOCAL_VLLM_SRC, "vllm", "__init__.py")):
    sys.path.insert(0, LOCAL_VLLM_SRC)

from vllm import LLM, SamplingParams

MODEL = "/models/Qwen3-32B"
TP = 4
INPUT_LEN = 2048
OUTPUT_LEN = 256
BATCH_SIZE = 1
PROFILE_DIR = os.path.abspath(f"./profile_logs_{datetime.now().strftime('%m%d_%H%M')}")


def decode_only_budget(args):
    """Upper bound on prefill-drain steps, and the per-request token budget it implies.

    With chunked prefill a single request needs ceil(input_len / max_num_batched_tokens)
    steps, so the request that finishes prefill first keeps decoding for the whole drain
    and must be given enough tokens to survive the profiler window.
    """
    chunks = math.ceil(args.input_len / args.max_num_batched_tokens)
    max_drain_steps = args.bs * chunks + args.bs
    total_tokens = max_drain_steps + args.settle_steps + args.profile_steps + 16
    return max_drain_steps, total_tokens


def run_decode_only(llm, args):
    """Profile ONLY steps where all `bs` sequences are in the decode phase.

    llm.generate() hides the prefill ramp-up inside one blocking call, so the trace
    always mixes prefill + growing decode batches. Here we drive the engine step by
    step, wait until every request has emitted its first token (= prefill done), and
    only then open the profiler window. Requests are aborted afterwards so the decode
    batch size never shrinks during profiling.
    """
    engine = llm.llm_engine
    bs = args.bs
    max_drain_steps, total_tokens = decode_only_budget(args)

    params = SamplingParams(
        temperature=0.0,
        max_tokens=total_tokens,
        min_tokens=total_tokens,
        ignore_eos=True,
        detokenize=False,
    )
    request_ids = [f"decode-{i}" for i in range(bs)]
    for i, rid in enumerate(request_ids):
        engine.add_request(rid, {"prompt_token_ids": [2000 + i] * args.input_len}, params)

    print(f"4a. Draining prefill for {bs} requests...")
    generated = {rid: 0 for rid in request_ids}
    steps = 0
    while not all(generated.values()):
        for out in engine.step():
            if out.outputs:
                generated[out.request_id] += 1
        steps += 1
        if steps > max_drain_steps + 100:
            raise RuntimeError(
                f"prefill did not drain: {sum(1 for v in generated.values() if v)}/{bs} "
                f"requests decoding after {steps} steps; KV cache is probably too small to hold "
                f"{bs} x {args.input_len} tokens, lower --bs/--input-len or raise --gpu-memory-utilization"
            )
    lead = max(generated.values())
    print(f"    prefill done in {steps} steps, all {bs} seqs are decoding.")

    # The earliest request must not hit max_tokens inside the profiler window.
    if lead + args.settle_steps + args.profile_steps >= total_tokens:
        raise RuntimeError(
            f"decode batch would shrink during profiling (lead={lead}, budget={total_tokens}); "
            f"lower --profile-steps or raise --max-num-batched-tokens."
        )

    for _ in range(args.settle_steps):
        engine.step()

    print(f"4b. Profiling {args.profile_steps} pure decode steps (bs={bs})...")
    llm.start_profile()
    t0 = time.time()
    window_tokens = 0
    for _ in range(args.profile_steps):
        for out in engine.step():
            if out.outputs:
                window_tokens += 1
    t1 = time.time()
    llm.stop_profile()

    expected = bs * args.profile_steps
    if window_tokens != expected:
        print(f"    WARNING: {window_tokens} tokens decoded in window, expected {expected} "
              f"-- batch was not constant (preemption?), trace may be impure")

    dt = (t1 - t0) / args.profile_steps
    print(f"    decode step latency : {dt * 1000:.2f} ms")
    print(f"    per-token latency   : {dt * 1000:.2f} ms (TPOT, bs={bs})")
    print(f"    decode throughput   : {bs / dt:.1f} tok/s")

    try:
        engine.abort_request(request_ids)
    except Exception:
        for rid in request_ids:
            engine.abort_request(rid)
    return t0, t1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=MODEL)
    parser.add_argument("--tp", type=int, default=TP)
    parser.add_argument("--max-model-len", type=int, default=16384)
    parser.add_argument("--max-num-batched-tokens", type=int, default=16384)
    parser.add_argument("--max-num-seqs", type=int, default=128)
    parser.add_argument("--input-len", type=int, default=INPUT_LEN)
    parser.add_argument("--output-len", type=int, default=OUTPUT_LEN)
    parser.add_argument("--bs", type=int, default=BATCH_SIZE)
    parser.add_argument("--profile-dir", type=str, default=PROFILE_DIR)
    parser.add_argument("--profiler", type=str, default="torch", choices=["torch", "xpu"])
    parser.add_argument("--kv-cache-dtype", type=str, default="auto",
                        choices=["auto", "fp8", "fp8_e4m3", "fp8_e5m2"],
                        help="KV cache dtype, same as `vllm serve --kv-cache-dtype`.")
    parser.add_argument(
        "--shutdown-timeout", "--worker-shutdown-grace-period",
        dest="shutdown_timeout",
        type=float,
        default=120.0,
        help="Seconds to wait for vLLM workers to exit and flush unitrace files.",
    )
    parser.add_argument(
        "--unitrace-final-flush-seconds",
        type=float,
        default=20.0, #default=10.0,
        help="Seconds to keep the main process alive after workers exit so unitrace can finalize output.",
    )
    parser.add_argument("--phase", type=str, default="all",
                        choices=["all", "prefill", "decode-only"])
    parser.add_argument("--enable-chunked-prefill", action="store_true",
                        help="Off by default. Pass this flag to turn chunked prefill ON: matches the "
                             "`vllm serve` default and allows max_num_batched_tokens < max_model_len "
                             "(required for long-sequence runs).")
    parser.add_argument("--ep", action="store_true",
                        help="Enable expert parallelism (MoE models); combine with --tp for TP+EP runs.")
    parser.add_argument("--enforce-eager", action="store_true",
                        help="Disable XPU graph capture. Graph replay reuses Level Zero events, which can "
                             "make unitrace emit kernels with garbage device timestamps.")
    parser.add_argument("--profile-steps", type=int, default=20,
                        help="decode-only: number of pure-decode steps inside the profiler window")
    parser.add_argument("--settle-steps", type=int, default=2,
                        help="decode-only: extra steps to run after the last prefill, before profiling")
    args = parser.parse_args()

    visible = [d for d in os.environ.get("ZE_AFFINITY_MASK", "").split(",") if d.strip()]
    if visible and len(visible) < args.tp:
        parser.error(
            f"ZE_AFFINITY_MASK exposes {len(visible)} device(s) but --tp is {args.tp}"
        )
    if args.max_num_seqs < args.bs:
        parser.error(f"--max-num-seqs ({args.max_num_seqs}) must be >= --bs ({args.bs})")
    if not args.enable_chunked_prefill and args.max_num_batched_tokens < args.max_model_len:
        parser.error(
            f"--max-num-batched-tokens ({args.max_num_batched_tokens}) must be >= --max-model-len "
            f"({args.max_model_len}) when chunked prefill is off; raise it or pass --enable-chunked-prefill"
        )
    if args.phase == "decode-only":
        _, total_tokens = decode_only_budget(args)
        if args.input_len + total_tokens > args.max_model_len:
            parser.error(
                f"decode-only needs max_model_len >= input_len + {total_tokens} "
                f"(= {args.input_len + total_tokens}), got {args.max_model_len}; "
                f"raise --max-model-len or lower --profile-steps/--bs"
            )
    if args.shutdown_timeout < 0:
        parser.error("--shutdown-timeout must be >= 0")
    if args.unitrace_final_flush_seconds < 0:
        parser.error("--unitrace-final-flush-seconds must be >= 0")

    profile_dir = os.path.abspath(args.profile_dir)
    if args.profiler == "torch":
        os.makedirs(profile_dir, exist_ok=True)

    profiler_config = {"profiler": "xpu"}
    if args.profiler == "torch":
        profiler_config = {
            "profiler": "torch",
            "torch_profiler_dir": profile_dir,
            "torch_profiler_with_stack": True,
            "torch_profiler_with_flops": True,
            "torch_profiler_record_shapes": True,
        }

    print("1. Loading vLLM engine...")
    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tp,
        #load_format="dummy",
        load_format="auto",
        max_model_len=args.max_model_len,
        max_num_batched_tokens=args.max_num_batched_tokens,
        max_num_seqs=args.max_num_seqs,
        enable_chunked_prefill=args.enable_chunked_prefill,
        enable_expert_parallel=args.ep,
        enable_prefix_caching=False,
        trust_remote_code=True,
        kv_cache_dtype=args.kv_cache_dtype,
        enforce_eager=args.enforce_eager,
        gpu_memory_utilization=0.90,
        profiler_config=profiler_config,
    )

    print(f"2. Preparing dummy data (bs={args.bs}, in={args.input_len}, out={args.output_len}, "
          f"phase={args.phase}, kv_cache_dtype={args.kv_cache_dtype})...")
    warmup_inputs = [{"prompt_token_ids": [1000 + i] * args.input_len} for i in range(args.bs)]
    profile_inputs = [{"prompt_token_ids": [2000 + i] * args.input_len} for i in range(args.bs)]

    if args.phase == "prefill":
        sampling_params = SamplingParams(temperature=0.0, max_tokens=1)
    else:
        sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=args.output_len,
            min_tokens=args.output_len,
            ignore_eos=True
        )

    # decode-only ignores --output-len, so keep the warmup short but still exercise decode.
    warmup_params = sampling_params
    if args.phase == "decode-only":
        warmup_params = SamplingParams(
            temperature=0.0, max_tokens=4, min_tokens=4, ignore_eos=True
        )

    print("3. Warmup (1 round)...")
    llm.generate(warmup_inputs, warmup_params)
    print("   Warmup complete.")

    outputs = None
    if args.phase == "decode-only":
        t0, t1 = run_decode_only(llm, args)
    else:
        print("4. Profiling...")
        llm.start_profile()
        t0 = time.time()
        outputs = llm.generate(profile_inputs, sampling_params)
        t1 = time.time()
        llm.stop_profile()

    print("\n" + "=" * 60)
    label = "Profiled decode window" if args.phase == "decode-only" else "Generate time"
    print(f"{label}: {t1 - t0:.2f}s")
    print("=" * 60)

    if args.profiler == "torch":
        print(f"\n   Shutting down workers before flushing traces to {profile_dir}/ ...")
    else:
        print("\n   Shutting down workers so unitrace can flush their traces...")

    # LLM currently has no public close()/shutdown() API. Explicitly stop its
    # EngineCore instead of waiting for interpreter teardown, then release all
    # remaining references before the final main-process flush delay.
    try:
        llm.llm_engine.engine_core.shutdown(
            timeout=args.shutdown_timeout
        )
    except Exception as exc:
        print(f"   WARNING: explicit engine shutdown failed: {exc}")
    finally:
        del outputs
        del llm
        gc.collect()

    print(
        f"   Workers stopped. Waiting {args.unitrace_final_flush_seconds:g}s "
        "for final trace output..."
    )
    time.sleep(args.unitrace_final_flush_seconds)

    if args.profiler == "torch":
        print(f"\n   Traces saved to: {profile_dir}/")
    else:
        print("\n   XPU/unitrace traces are saved by unitrace --output-dir-path.")

if __name__ == "__main__":
    main()
