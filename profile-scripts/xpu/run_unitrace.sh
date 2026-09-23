#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")"

MODEL=${MODEL:-/models/Qwen3-32B}
TP=${TP:-1}
INPUT_LEN=${INPUT_LEN:-3500}
OUTPUT_LEN=${OUTPUT_LEN:-10}
BS=${BS:-1}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-8192}
MAX_NUM_BATCHED_TOKENS=${MAX_NUM_BATCHED_TOKENS:-8192}
MAX_NUM_SEQS=${MAX_NUM_SEQS:-32}
KV_CACHE_DTYPE=${KV_CACHE_DTYPE:-auto}
CHUNKED_PREFILL=${CHUNKED_PREFILL:-1}
PHASE=${PHASE:-all}
PROFILE_STEPS=${PROFILE_STEPS:-20}
SETTLE_STEPS=${SETTLE_STEPS:-2}
ENFORCE_EAGER=${ENFORCE_EAGER:-0}
ENABLE_EP=${ENABLE_EP:-0}
LANGUAGE_MODEL_ONLY=${LANGUAGE_MODEL_ONLY:-0}
SHUTDOWN_TIMEOUT=${SHUTDOWN_TIMEOUT:-120}
PROFILER=${PROFILER:-xpu}
LOG_DIR=${LOG_DIR:-./unitrace-log}
TRACE_ROOT=${TRACE_ROOT:-./unitrace-trace}

usage() {
  cat <<EOF
Usage: $0 [options]
  -m, --model <path>             model path                (default: $MODEL)
      --tp <n>                   tensor parallel size      (default: $TP)
  -i, --input-len <n>            input length              (default: $INPUT_LEN)
  -o, --output-len <n>           output length             (default: $OUTPUT_LEN)
  -b, --bs <n>                   decode batch size         (default: $BS)
      --max-model-len <n>                                  (default: $MAX_MODEL_LEN)
      --max-num-batched-tokens <n>                         (default: $MAX_NUM_BATCHED_TOKENS)
      --max-num-seqs <n>                                   (default: $MAX_NUM_SEQS)
      --kv-cache-dtype <auto|fp8|fp8_e4m3|fp8_e5m2>        (default: $KV_CACHE_DTYPE)
      --enable-chunked-prefill   打开 chunked prefill       (default: CHUNKED_PREFILL=$CHUNKED_PREFILL)
      --no-chunked-prefill       关闭 chunked prefill（需 max-num-batched-tokens >= max-model-len）
      --phase <all|prefill|decode-only>                    (default: $PHASE)
      --profile-steps <n>        decode-only 窗口步数      (default: $PROFILE_STEPS)
      --settle-steps <n>         decode-only 预热步数      (default: $SETTLE_STEPS)
      --enforce-eager            关闭 XPU graph capture   (default: ENFORCE_EAGER=$ENFORCE_EAGER)
      --ep                       开启 expert parallel       (default: ENABLE_EP=$ENABLE_EP)
      --language-model-only      只加载语言模型（跳过多模态模块）(default: LANGUAGE_MODEL_ONLY=$LANGUAGE_MODEL_ONLY)
      --shutdown-timeout <sec>   worker shutdown/trace flush timeout (default: $SHUTDOWN_TIMEOUT)
      --profiler <torch|xpu>                               (default: $PROFILER)
      --log-dir <path>           log directory             (default: $LOG_DIR)
      --trace-root <path>        unitrace output root      (default: $TRACE_ROOT)
      --check                    只做环境自检，不启动 profiling
  -h, --help
EOF
}

CHECK_ONLY=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    -m|--model) MODEL=$2; shift 2 ;;
    --tp) TP=$2; shift 2 ;;
    -i|--input-len) INPUT_LEN=$2; shift 2 ;;
    -o|--output-len) OUTPUT_LEN=$2; shift 2 ;;
    -b|--bs) BS=$2; shift 2 ;;
    --max-model-len) MAX_MODEL_LEN=$2; shift 2 ;;
    --max-num-batched-tokens) MAX_NUM_BATCHED_TOKENS=$2; shift 2 ;;
    --max-num-seqs) MAX_NUM_SEQS=$2; shift 2 ;;
    --kv-cache-dtype) KV_CACHE_DTYPE=$2; shift 2 ;;
    --enable-chunked-prefill) CHUNKED_PREFILL=1; shift ;;
    --no-chunked-prefill) CHUNKED_PREFILL=0; shift ;;
    --phase) PHASE=$2; shift 2 ;;
    --profile-steps) PROFILE_STEPS=$2; shift 2 ;;
    --settle-steps) SETTLE_STEPS=$2; shift 2 ;;
    --enforce-eager) ENFORCE_EAGER=1; shift ;;
    --ep) ENABLE_EP=1; shift ;;
    --language-model-only) LANGUAGE_MODEL_ONLY=1; shift ;;
    --shutdown-timeout) SHUTDOWN_TIMEOUT=$2; shift 2 ;;
    --profiler) PROFILER=$2; shift 2 ;;
    --log-dir) LOG_DIR=$2; shift 2 ;;
    --trace-root) TRACE_ROOT=$2; shift 2 ;;
    --check) CHECK_ONLY=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage; exit 1 ;;
  esac
done

# unitrace 用 execvp 启动目标程序，PATH 里找不到 python 时只会报 "Failed to launch target application"
PYTHON_BIN=${PYTHON_BIN:-$(command -v python || command -v python3 || true)}
if [[ -z $PYTHON_BIN || ! -x $PYTHON_BIN ]]; then
  echo "[ERROR] PATH 中找不到 python/python3，请设置 PYTHON_BIN=<解释器绝对路径>" >&2
  exit 1
fi

UNITRACE_BIN=${UNITRACE_BIN:-$(command -v unitrace || true)}
if [[ -z $UNITRACE_BIN ]]; then
  echo "[ERROR] PATH 中找不到 unitrace，请设置 UNITRACE_BIN=<unitrace 绝对路径>" >&2
  exit 1
fi

if [[ ! -d $MODEL ]]; then
  echo "[ERROR] 模型路径不存在: $MODEL" >&2
  exit 1
fi

if [[ $CHECK_ONLY -eq 1 ]]; then
  echo "python       : $PYTHON_BIN"
  echo "unitrace     : $UNITRACE_BIN"
  echo "VIRTUAL_ENV  : ${VIRTUAL_ENV:-<unset>}"
  echo "model        : $MODEL (ok)"
  "$PYTHON_BIN" -c "import sys, torch, vllm
print('sys.executable:', sys.executable)
print('sys.prefix    :', sys.prefix)
print('torch         :', torch.__version__)
print('vllm          :', vllm.__version__)
print('xpu available :', torch.xpu.is_available(), '| device_count:', torch.xpu.device_count())" 2>&1 | grep -v "^W[0-9]"
  exit 0
fi

if [[ $CHUNKED_PREFILL != 0 && $CHUNKED_PREFILL != 1 ]]; then
  echo "[ERROR] CHUNKED_PREFILL 只能是 0 或 1，当前为 '$CHUNKED_PREFILL'" >&2
  exit 1
fi
if [[ $ENFORCE_EAGER != 0 && $ENFORCE_EAGER != 1 ]]; then
  echo "[ERROR] ENFORCE_EAGER 只能是 0 或 1，当前为 '$ENFORCE_EAGER'" >&2
  exit 1
fi
if [[ $ENABLE_EP != 0 && $ENABLE_EP != 1 ]]; then
  echo "[ERROR] ENABLE_EP 只能是 0 或 1，当前为 '$ENABLE_EP'" >&2
  exit 1
fi
if [[ $LANGUAGE_MODEL_ONLY != 0 && $LANGUAGE_MODEL_ONLY != 1 ]]; then
  echo "[ERROR] LANGUAGE_MODEL_ONLY 只能是 0 或 1，当前为 '$LANGUAGE_MODEL_ONLY'" >&2
  exit 1
fi
if [[ $CHUNKED_PREFILL -eq 0 && $MAX_NUM_BATCHED_TOKENS -lt $MAX_MODEL_LEN ]]; then
  echo "[ERROR] 关闭 chunked prefill 时需要 --max-num-batched-tokens ($MAX_NUM_BATCHED_TOKENS) >= --max-model-len ($MAX_MODEL_LEN)" >&2
  exit 1
fi

case "$PHASE" in
  all|prefill|decode-only) ;;
  *) echo "[ERROR] --phase 只能是 all|prefill|decode-only，当前为 '$PHASE'" >&2; exit 1 ;;
esac
if ! [[ $PROFILE_STEPS =~ ^[1-9][0-9]*$ ]]; then
  echo "[ERROR] --profile-steps 必须是正整数，当前为 '$PROFILE_STEPS'" >&2
  exit 1
fi
if ! [[ $SETTLE_STEPS =~ ^[0-9]+$ ]]; then
  echo "[ERROR] --settle-steps 必须是非负整数，当前为 '$SETTLE_STEPS'" >&2
  exit 1
fi

EXTRA_ARGS=()
[[ $CHUNKED_PREFILL -eq 1 ]] && EXTRA_ARGS+=(--enable-chunked-prefill)
[[ $ENFORCE_EAGER -eq 1 ]] && EXTRA_ARGS+=(--enforce-eager)
[[ $ENABLE_EP -eq 1 ]] && EXTRA_ARGS+=(--ep)
[[ $LANGUAGE_MODEL_ONLY -eq 1 ]] && EXTRA_ARGS+=(--language-model-only)

MODEL_NAME=$(basename "${MODEL%/}")
KV_TAG=""
[[ $KV_CACHE_DTYPE != auto ]] && KV_TAG="-kv${KV_CACHE_DTYPE}"
GRAPH_TAG="xpugraph"
[[ $ENFORCE_EAGER -eq 1 ]] && GRAPH_TAG="eager"
TAG="${MODEL_NAME}-${PHASE}-${GRAPH_TAG}-in${INPUT_LEN}-out${OUTPUT_LEN}-bs${BS}-tp${TP}${KV_TAG}-$(date +%y%m%d-%H%M%S)"
OUT_DIR="$TRACE_ROOT/$TAG"
LOG_FILE="$LOG_DIR/unitrace-log-$TAG.log"

mkdir -p "$OUT_DIR" "$LOG_DIR"

echo "[INFO] model     : $MODEL"
echo "[INFO] in/out/bs : $INPUT_LEN / $OUTPUT_LEN / $BS (tp=$TP)"
echo "[INFO] phase     : $PHASE"
[[ $PHASE == decode-only ]] && echo "[INFO] steps     : profile=$PROFILE_STEPS settle=$SETTLE_STEPS"
echo "[INFO] kv dtype  : $KV_CACHE_DTYPE"
echo "[INFO] chunked   : $([[ $CHUNKED_PREFILL -eq 1 ]] && echo on || echo off)"
echo "[INFO] XPU graph : $([[ $ENFORCE_EAGER -eq 1 ]] && echo off || echo on)"
echo "[INFO] EP        : $([[ $ENABLE_EP -eq 1 ]] && echo on || echo off)"
echo "[INFO] lm only   : $([[ $LANGUAGE_MODEL_ONLY -eq 1 ]] && echo on || echo off)"
echo "[INFO] shutdown  : ${SHUTDOWN_TIMEOUT}s"
echo "[INFO] python    : $PYTHON_BIN"
echo "[INFO] unitrace  : $UNITRACE_BIN"
echo "[INFO] trace dir : $(realpath "$OUT_DIR")"
echo "[INFO] log file  : $(realpath -m "$LOG_FILE")"

export ZE_AFFINITY_MASK=${ZE_AFFINITY_MASK:-0}
export NEOReadDebugKeys=1
export EnableImplicitConvertionToCounterBasedEvents=0

"$UNITRACE_BIN" \
  --chrome-itt-logging \
  --chrome-sycl-logging \
  --chrome-call-logging \
  --chrome-kernel-logging \
  --output-dir-path "$OUT_DIR" \
  --start-paused \
  "$PYTHON_BIN" run_auto_model_xpu-bs-decode-only.py \
  --model "$MODEL" \
  --tp "$TP" \
  --max-model-len "$MAX_MODEL_LEN" \
  --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS" \
  "${EXTRA_ARGS[@]}" \
  --max-num-seqs "$MAX_NUM_SEQS" \
  --kv-cache-dtype "$KV_CACHE_DTYPE" \
  --input-len "$INPUT_LEN" \
  --output-len "$OUTPUT_LEN" \
  --bs "$BS" \
  --phase "$PHASE" \
  --profile-steps "$PROFILE_STEPS" \
  --settle-steps "$SETTLE_STEPS" \
  --profiler "$PROFILER" \
  --shutdown-timeout "$SHUTDOWN_TIMEOUT" \
  2>&1 | tee -a "$LOG_FILE"
