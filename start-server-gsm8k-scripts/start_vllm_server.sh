#!/bin/bash

set -euo pipefail

MODEL_PATH="${MODEL_PATH:-/models/Qwen3.6-27B}"
MODEL_NAME=""
SERVED_MODEL_NAME=""
PLATFORM="${PLATFORM:-xpu}"
HOST="0.0.0.0"
PORT="9001"
TENSOR_PARALLEL_SIZE=1
MAX_MODEL_LEN=8192
MAX_NUM_SEQS=128
GPU_MEMORY_UTILIZATION=0.9
KV_CACHE_DTYPE=""
XPU_ID="${XPU_ID:-${ZE_AFFINITY_MASK:-0}}"
CUDA_DEVICES="${CUDA_DEVICES:-${CUDA_VISIBLE_DEVICES:-0}}"
ENABLE_GRAPH=1
ENABLE_BREAKABLE=0
USE_V2_MODEL_RUNNER=0
ENFORCE_EAGER=0
CHUNKED_PREFILL=1
LOG_DIR="./server-logs"
LOG_TAG=""
DRY_RUN=0
EXTRA_ARGS=()

usage() {
    cat <<'EOF'
Usage:
    ./start_vllm_server.sh [options] [-- <extra vllm serve args>]

Model / platform:
    -m, --model-path            Model path or HF model id, default /models/Qwen3.6-27B
    --served-model-name         Optional API model name passed to vLLM
    --platform <xpu|cuda>       Runtime platform, default xpu

Server options:
    --model-name                Name used only in the log file (default: model basename)
  --host                      Bind host, default 0.0.0.0
  -p, --port                  Bind port, default 9001
  -t, --tensor-parallel-size  TP size, default 1
  --max-model-len             Max model len, default 8192
  --max-num-seqs              Max num seqs, default 128
    --gpu-memory-utilization    GPU memory utilization, default 0.9
  --kv-cache-dtype            auto/bfloat16/float16/fp8/fp8_e4m3/fp8_e5m2;
                              unset means vllm keeps its own default
    --enable-chunked-prefill    Enable chunked prefill (default)
    --no-chunked-prefill        Disable chunked prefill
    --enforce-eager             Disable CUDA/XPU graph through the vLLM CLI

Device / XPU graph options:
  -x, --xpu-id                XPU id(s) for ZE_AFFINITY_MASK, e.g. 0 or 4,5,6,7
                                                            (default: current ZE_AFFINITY_MASK or 0)
    --cuda-devices              CUDA_VISIBLE_DEVICES, e.g. 0 or 0,1,2,3
                                                            (default: current CUDA_VISIBLE_DEVICES or 0)
    --graph <0|1>               XPU: VLLM_XPU_ENABLE_XPU_GRAPH, default 1
    --breakable <0|1>           XPU: VLLM_USE_BREAKABLE_CUDAGRAPH, default 0
    --v2-model-runner <0|1>     XPU: VLLM_USE_V2_MODEL_RUNNER, default 0

Logging:
  --log-dir                   Log directory, default ./server-logs (auto-created)
  --log-tag                   Extra tag appended to the log file name
  --dry-run                   Print the command and exit

Examples:
  ./start_vllm_server.sh -m /models/Qwen3.6-27B -x 0 --graph 1
  ./start_vllm_server.sh -m /models/Qwen3.6-27B -t 4 -x 4,5,6,7 \
      --graph 1 --breakable 1 --kv-cache-dtype fp8
    ./start_vllm_server.sh --platform cuda -m /models/Qwen3-32B -t 4 \
            --cuda-devices 0,1,2,3 -- --quantization fp8
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help)
            usage
            exit 0
            ;;
        -m|--model-path)
            MODEL_PATH="$2"
            shift 2
            ;;
        --served-model-name)
            SERVED_MODEL_NAME="$2"
            shift 2
            ;;
        --platform)
            PLATFORM="$2"
            shift 2
            ;;
        --model-name)
            MODEL_NAME="$2"
            shift 2
            ;;
        --host)
            HOST="$2"
            shift 2
            ;;
        -p|--port)
            PORT="$2"
            shift 2
            ;;
        -t|--tensor-parallel-size)
            TENSOR_PARALLEL_SIZE="$2"
            shift 2
            ;;
        --max-model-len)
            MAX_MODEL_LEN="$2"
            shift 2
            ;;
        --max-num-seqs)
            MAX_NUM_SEQS="$2"
            shift 2
            ;;
        --gpu-memory-util|--gpu-memory-utilization)
            GPU_MEMORY_UTILIZATION="$2"
            shift 2
            ;;
        --kv-cache-dtype)
            KV_CACHE_DTYPE="$2"
            shift 2
            ;;
        -x|--xpu-id)
            XPU_ID="$2"
            shift 2
            ;;
        --cuda-devices)
            CUDA_DEVICES="$2"
            shift 2
            ;;
        --graph)
            ENABLE_GRAPH="$2"
            shift 2
            ;;
        --breakable)
            ENABLE_BREAKABLE="$2"
            shift 2
            ;;
        --v2-model-runner)
            USE_V2_MODEL_RUNNER="$2"
            shift 2
            ;;
        --enforce-eager)
            ENFORCE_EAGER=1
            shift
            ;;
        --enable-chunked-prefill)
            CHUNKED_PREFILL=1
            shift
            ;;
        --no-chunked-prefill)
            CHUNKED_PREFILL=0
            shift
            ;;
        --log-dir)
            LOG_DIR="$2"
            shift 2
            ;;
        --log-tag)
            LOG_TAG="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=1
            shift
            ;;
        --)
            shift
            EXTRA_ARGS=("$@")
            break
            ;;
        *)
            echo "Error: unknown argument: $1"
            usage
            exit 1
            ;;
    esac
done

case "$PLATFORM" in
    xpu|cuda) ;;
    *) echo "Error: --platform must be xpu or cuda. Got: $PLATFORM"; exit 1 ;;
esac

if ! [[ "$TENSOR_PARALLEL_SIZE" =~ ^[1-9][0-9]*$ ]]; then
    echo "Error: tensor parallel size must be a positive integer. Got: $TENSOR_PARALLEL_SIZE"
    exit 1
fi

for value_name in MAX_MODEL_LEN MAX_NUM_SEQS; do
    value="${!value_name}"
    if ! [[ "$value" =~ ^[1-9][0-9]*$ ]]; then
        echo "Error: ${value_name} must be a positive integer. Got: $value"
        exit 1
    fi
done

for flag_name in ENABLE_GRAPH ENABLE_BREAKABLE USE_V2_MODEL_RUNNER ENFORCE_EAGER CHUNKED_PREFILL; do
    flag_value="${!flag_name}"
    if [[ "$flag_value" != "0" && "$flag_value" != "1" ]]; then
        echo "Error: ${flag_name} must be 0 or 1. Got: ${flag_value}"
        exit 1
    fi
done

if [[ -n "$KV_CACHE_DTYPE" ]]; then
    case "$KV_CACHE_DTYPE" in
        auto|bfloat16|float16|fp8|fp8_e4m3|fp8_e5m2|fp8_inc) ;;
        *)
            echo "Error: unsupported kv cache dtype: ${KV_CACHE_DTYPE}."
            exit 1
            ;;
    esac
fi

if [[ -z "$MODEL_NAME" ]]; then
    MODEL_NAME="$(basename "${MODEL_PATH%/}")"
fi

if [[ "$PLATFORM" == "xpu" ]]; then
    export ZE_AFFINITY_MASK="$XPU_ID"
    export VLLM_XPU_ENABLE_XPU_GRAPH="$ENABLE_GRAPH"
    export VLLM_USE_BREAKABLE_CUDAGRAPH="$ENABLE_BREAKABLE"
    export VLLM_USE_V2_MODEL_RUNNER="$USE_V2_MODEL_RUNNER"
    DEVICE_LIST="$ZE_AFFINITY_MASK"
else
    export CUDA_VISIBLE_DEVICES="$CUDA_DEVICES"
    unset ZE_AFFINITY_MASK VLLM_XPU_ENABLE_XPU_GRAPH
    DEVICE_LIST="$CUDA_VISIBLE_DEVICES"
fi

IFS=',' read -r -a VISIBLE_DEVICES <<< "$DEVICE_LIST"
if (( ${#VISIBLE_DEVICES[@]} < TENSOR_PARALLEL_SIZE )); then
    echo "Error: $PLATFORM exposes ${#VISIBLE_DEVICES[@]} device(s), but TP is $TENSOR_PARALLEL_SIZE."
    exit 1
fi

sanitize() {
    printf '%s' "$1" | sed -E 's/[[:space:]]+/_/g; s/[^A-Za-z0-9_.-]+/-/g; s/-+/-/g; s/^-+|-+$//g'
}

TIMESTAMP="$(TZ='Asia/Shanghai' date +%F-%H-%M-%S)"
if [[ "$PLATFORM" == "xpu" ]]; then
    GRAPH_TAG="graph${ENABLE_GRAPH}-breakable${ENABLE_BREAKABLE}"
elif (( ENFORCE_EAGER )); then
    GRAPH_TAG="eager"
else
    GRAPH_TAG="cudagraph"
fi
LOG_NAME="server_$(sanitize "$MODEL_NAME")_${PLATFORM}_tp${TENSOR_PARALLEL_SIZE}_kv${KV_CACHE_DTYPE:-auto}_${GRAPH_TAG}"
if [[ -n "$LOG_TAG" ]]; then
    LOG_NAME="${LOG_NAME}_$(sanitize "$LOG_TAG")"
fi
LOG_NAME="${LOG_NAME}_${TIMESTAMP}.log"

LOG_PATH="${LOG_DIR%/}/${LOG_NAME}"

server_args=(
    --model "$MODEL_PATH"
    --host "$HOST"
    --port "$PORT"
    --trust-remote-code
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"
    --max-model-len "$MAX_MODEL_LEN"
    --max-num-seqs "$MAX_NUM_SEQS"
    --tensor-parallel-size "$TENSOR_PARALLEL_SIZE"
    --no-enable-prefix-caching
)

[[ -n "$SERVED_MODEL_NAME" ]] && server_args+=(--served-model-name "$SERVED_MODEL_NAME")
(( CHUNKED_PREFILL )) && server_args+=(--enable-chunked-prefill)
(( ! CHUNKED_PREFILL )) && server_args+=(--no-enable-chunked-prefill)
(( ENFORCE_EAGER )) && server_args+=(--enforce-eager)

if [[ -n "$KV_CACHE_DTYPE" ]]; then
    server_args+=(--kv-cache-dtype "$KV_CACHE_DTYPE")
fi

if (( ${#EXTRA_ARGS[@]} > 0 )); then
    server_args+=("${EXTRA_ARGS[@]}")
fi

VLLM_BIN="${VLLM_BIN:-$(command -v vllm || true)}"

print_summary() {
    echo "=============================================="
    echo "Start time        : $(TZ='Asia/Shanghai' date '+%F %T %Z')"
    echo "Model             : $MODEL_PATH"
    echo "Served model name : ${SERVED_MODEL_NAME:-<vllm default>}"
    echo "Platform          : $PLATFORM"
    echo "Tensor parallel   : $TENSOR_PARALLEL_SIZE"
    echo "Visible devices   : $DEVICE_LIST"
    echo "KV cache dtype    : ${KV_CACHE_DTYPE:-auto (vllm default)}"
    echo "Chunked prefill   : $CHUNKED_PREFILL"
    echo "Enforce eager     : $ENFORCE_EAGER"
    if [[ "$PLATFORM" == "xpu" ]]; then
        echo "XPU graph         : $VLLM_XPU_ENABLE_XPU_GRAPH"
        echo "Breakable graph   : $VLLM_USE_BREAKABLE_CUDAGRAPH"
        echo "V2 model runner   : $VLLM_USE_V2_MODEL_RUNNER"
    fi
    echo "Log file          : $LOG_PATH"
    printf 'Command           : '
    printf '%q ' "${VLLM_BIN:-vllm}" serve "${server_args[@]}"
    echo
    echo "=============================================="
}

if (( DRY_RUN == 1 )); then
    print_summary
    echo "Dry run requested; not starting the server."
    exit 0
fi

if [[ -z "$VLLM_BIN" ]]; then
    echo "Error: vllm is not available in the current environment. Set VLLM_BIN explicitly."
    exit 1
fi

mkdir -p "$LOG_DIR"
print_summary | tee -a "$LOG_PATH"
"$VLLM_BIN" serve "${server_args[@]}" 2>&1 | tee -a "$LOG_PATH"
