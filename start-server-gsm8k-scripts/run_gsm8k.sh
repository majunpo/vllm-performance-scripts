#!/bin/bash
# Chat mode requires the gsm8k_eval.py chat-completions patch:
#   patchs/0002-evals-gsm8k-Support-chat-completions-and-chat_templa.patch
# Completion mode works with the upstream evaluator.

set -euo pipefail

# Falls back to the first checkout that actually contains the evaluator.
if [[ -z "${VLLM_REPO:-}" ]]; then
    for _candidate in /home/junpo/vllm-xpu /home/junpo/vllm /home/junpo/applications.ai.gpu.vllm-xpu; do
        if [[ -f "${_candidate}/tests/evals/gsm8k/gsm8k_eval.py" ]]; then
            VLLM_REPO="$_candidate"
            break
        fi
    done
    VLLM_REPO="${VLLM_REPO:-/home/junpo/vllm-xpu}"
fi
HOST="${HOST:-http://127.0.0.1}"
PORT="${PORT:-9001}"
MODEL=""
MODE="chat"
THINKING=0
NUM_QUESTIONS=1319
NUM_SHOTS=5
MAX_TOKENS=""
MAX_CONCURRENCY=32
TEMPERATURE=0.0
SEED=42
RESULT_DIR="/home/junpo/gsm8k"
TAG=""
DRY_RUN=0
EXTRA_ARGS=()

usage() {
    cat <<'EOF'
Usage:
  ./run_gsm8k.sh [options] [-- <extra gsm8k_eval.py args>]

Server:
  --host                  Host URL, default http://127.0.0.1
  -p, --port              Port, default 9001
  -m, --model             Served model id; default: auto-detect from /v1/models
  --vllm-repo             vLLM checkout holding tests/evals/gsm8k/gsm8k_eval.py;
                          also settable via the VLLM_REPO env var

Evaluation:
  --mode <chat|completion>  chat applies the chat template (default),
                            completion posts the raw few-shot text
  --thinking <0|1>          enable_thinking for chat mode, default 0
  -n, --num-questions       Default 1319 (full test set); use 20 for a probe
  --num-shots               Default 5 (GSM8K standard, do not change casually)
  --max-tokens              Default 1024 when thinking=0, 4096 when thinking=1
  -c, --max-concurrency     Default 32; perf only, does not affect accuracy
  --temperature             Default 0.0
  --seed                    Default 42

Output:
  --result-dir            Default /home/junpo/gsm8k
  --tag                   Extra tag in the result file name
  --dry-run               Print the command and exit

Examples:
  ./run_gsm8k.sh -n 20                                  # quick probe
  ./run_gsm8k.sh --tag mxfp8-graph1
  ./run_gsm8k.sh --thinking 1 --tag qwen3.6-think
  ./run_gsm8k.sh --mode completion --tag baseline-raw
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help) usage; exit 0 ;;
        --host) HOST="$2"; shift 2 ;;
        -p|--port) PORT="$2"; shift 2 ;;
        -m|--model) MODEL="$2"; shift 2 ;;
        --vllm-repo) VLLM_REPO="$2"; shift 2 ;;
        --mode) MODE="$2"; shift 2 ;;
        --thinking) THINKING="$2"; shift 2 ;;
        -n|--num-questions) NUM_QUESTIONS="$2"; shift 2 ;;
        --num-shots) NUM_SHOTS="$2"; shift 2 ;;
        --max-tokens) MAX_TOKENS="$2"; shift 2 ;;
        -c|--max-concurrency) MAX_CONCURRENCY="$2"; shift 2 ;;
        --temperature) TEMPERATURE="$2"; shift 2 ;;
        --seed) SEED="$2"; shift 2 ;;
        --result-dir) RESULT_DIR="$2"; shift 2 ;;
        --tag) TAG="$2"; shift 2 ;;
        --dry-run) DRY_RUN=1; shift ;;
        --) shift; EXTRA_ARGS=("$@"); break ;;
        *) echo "Error: unknown option: $1"; usage; exit 1 ;;
    esac
done

HOST="${HOST%/}"

case "$MODE" in
    chat|completion) ;;
    *) echo "Error: --mode must be chat or completion, got: $MODE"; exit 1 ;;
esac

PYTHON_BIN="${PYTHON_BIN:-$(command -v python3 || command -v python || true)}"
EVAL_SCRIPT="${VLLM_REPO%/}/tests/evals/gsm8k/gsm8k_eval.py"
if (( ! DRY_RUN )) && [[ -z "$PYTHON_BIN" ]]; then
    echo "Error: python3/python is not available. Set PYTHON_BIN explicitly."
    exit 1
fi
if (( ! DRY_RUN )) && [[ ! -f "$EVAL_SCRIPT" ]]; then
    echo "Error: $EVAL_SCRIPT not found."
    echo "Set VLLM_REPO=<vllm checkout> or pass --vllm-repo <path>."
    exit 1
fi

if [[ -z "$MODEL" ]]; then
    if command -v curl >/dev/null 2>&1 && [[ -n "$PYTHON_BIN" ]]; then
        MODEL="$(curl -sf "${HOST}:${PORT}/v1/models" \
            | "$PYTHON_BIN" -c 'import sys, json; print(json.load(sys.stdin)["data"][0]["id"])' \
            2>/dev/null || true)"
    fi
    if [[ -z "$MODEL" ]] && (( DRY_RUN )); then
        MODEL="auto-detect-from-server"
    elif [[ -z "$MODEL" ]]; then
        echo "Error: cannot reach ${HOST}:${PORT}/v1/models. Is the server up?"
        exit 1
    fi
fi

if [[ -z "$MAX_TOKENS" ]]; then
    if [[ "$MODE" == "chat" && "$THINKING" == "0" ]]; then
        MAX_TOKENS=1024
    else
        MAX_TOKENS=4096
    fi
fi

sanitize() {
    printf '%s' "$1" | sed -E 's#^/+##; s#/+#-#g; s/[[:space:]]+/_/g; s/[^A-Za-z0-9_.-]+/-/g; s/-+/-/g; s/^-+|-+$//g'
}

if [[ "$MODE" == "chat" ]]; then
    MODE_TAG="chat-think${THINKING}"
else
    MODE_TAG="completion"
fi
NAME="gsm8k_$(sanitize "$MODEL")_${MODE_TAG}_n${NUM_QUESTIONS}_shot${NUM_SHOTS}_max${MAX_TOKENS}"
if [[ -n "$TAG" ]]; then
    NAME="${NAME}_$(sanitize "$TAG")"
fi
NAME="${NAME}_$(TZ='Asia/Shanghai' date +%F-%H-%M-%S)"

RESULT_PATH="${RESULT_DIR%/}/${NAME}.json"
LOG_PATH="${RESULT_DIR%/}/${NAME}.log"

eval_args=(
    --host "$HOST"
    --port "$PORT"
    --num-questions "$NUM_QUESTIONS"
    --num-shots "$NUM_SHOTS"
    --max-tokens "$MAX_TOKENS"
    --max-concurrency "$MAX_CONCURRENCY"
    --temperature "$TEMPERATURE"
    --seed "$SEED"
    --save-results "$RESULT_PATH"
)

if [[ "$MODE" == "chat" ]]; then
    if [[ "$THINKING" == "1" ]]; then
        THINK_JSON='{"enable_thinking": true}'
    else
        THINK_JSON='{"enable_thinking": false}'
    fi
    eval_args+=(--model "$MODEL" --use-chat-completions --chat-template-kwargs "$THINK_JSON")
fi

if (( ${#EXTRA_ARGS[@]} > 0 )); then
    eval_args+=("${EXTRA_ARGS[@]}")
fi

print_summary() {
    echo "=============================================="
    echo "Start time        : $(TZ='Asia/Shanghai' date '+%F %T %Z')"
    echo "Endpoint          : ${HOST}:${PORT}"
    echo "Model             : $MODEL"
    echo "Mode              : $MODE"
    echo "Thinking          : $([[ "$MODE" == chat ]] && echo "$THINKING" || echo 'n/a')"
    echo "Questions / shots : ${NUM_QUESTIONS} / ${NUM_SHOTS}"
    echo "Max tokens        : $MAX_TOKENS"
    echo "Max concurrency   : $MAX_CONCURRENCY"
    echo "Result file       : $RESULT_PATH"
    printf 'Command           : '
    printf '%q ' "${PYTHON_BIN:-python3}" "$EVAL_SCRIPT" "${eval_args[@]}"
    echo
    echo "=============================================="
}

if (( DRY_RUN )); then
    print_summary
    echo "Dry run, not executing."
    exit 0
fi

mkdir -p "$RESULT_DIR"
print_summary | tee "$LOG_PATH"
cd "$VLLM_REPO"
"$PYTHON_BIN" "$EVAL_SCRIPT" "${eval_args[@]}" 2>&1 | tee -a "$LOG_PATH"
