#!/bin/bash

set -euo pipefail

HOST="${HOST:-http://127.0.0.1}"
PORT="${PORT:-9001}"
MODEL="${MODEL:-}"
TASKS="${TASKS:-gsm8k}"
BATCH_SIZE="${BATCH_SIZE:-64}"
MAX_LENGTH="${MAX_LENGTH:-16384}"
MAX_GEN_TOKS="${MAX_GEN_TOKS:-2048}"
TRUST_REMOTE_CODE="${TRUST_REMOTE_CODE:-true}"
NUM_FEWSHOT="${NUM_FEWSHOT:-}"
LIMIT="${LIMIT:-}"
LOG_SAMPLES="${LOG_SAMPLES:-1}"
CONFIRM_UNSAFE_CODE="${CONFIRM_UNSAFE_CODE:-1}"
OUTPUT_PATH="${OUTPUT_PATH:-}"
TAG="${TAG:-}"
DRY_RUN=0
EXTRA_MODEL_ARGS=()
EXTRA_ARGS=()

usage() {
    cat <<'EOF'
Usage:
  ./run_gsm8k_lm_eval.sh [options] [-- <extra lm_eval args>]

Endpoint:
  --host                      Host URL, default http://127.0.0.1
  -p, --port                  Port, default 9001
  -m, --model                 Served model id/path; default: auto-detect from /v1/models

Evaluation:
  --tasks                     lm-eval task list, default gsm8k
  -b, --batch-size            lm-eval batch size, default 64
  --max-length                local-completions max_length, default 16384
  --max-gen-toks              local-completions max_gen_toks, default 2048
  --trust-remote-code <bool>  Model arg, default true
  --num-fewshot <n>           Override task few-shot count; default: task configuration
  --limit <n|fraction>        Evaluate a subset, e.g. 20 or 0.1
  --model-arg <key=value>     Append a local-completions model arg; repeatable
  --no-log-samples            Do not pass --log_samples
  --no-confirm-unsafe-code    Do not pass --confirm_run_unsafe_code

Output:
  -o, --output-path           Default ./lm_eval_output_<timestamp>[_tag]
  --tag                       Suffix for the generated output path
  --dry-run                   Print the command without requiring lm_eval or a live server
  -h, --help

Examples:
  ./run_gsm8k_lm_eval.sh --model /llm/models/Qwen3.6-35B-A3B-NVFP4 -p 30201
  ./run_gsm8k_lm_eval.sh --limit 20 --tag probe
  ./run_gsm8k_lm_eval.sh --num-fewshot 5 --batch-size 32
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help) usage; exit 0 ;;
        --host) HOST="$2"; shift 2 ;;
        -p|--port) PORT="$2"; shift 2 ;;
        -m|--model) MODEL="$2"; shift 2 ;;
        --tasks) TASKS="$2"; shift 2 ;;
        -b|--batch-size) BATCH_SIZE="$2"; shift 2 ;;
        --max-length) MAX_LENGTH="$2"; shift 2 ;;
        --max-gen-toks) MAX_GEN_TOKS="$2"; shift 2 ;;
        --trust-remote-code) TRUST_REMOTE_CODE="$2"; shift 2 ;;
        --num-fewshot) NUM_FEWSHOT="$2"; shift 2 ;;
        --limit) LIMIT="$2"; shift 2 ;;
        --model-arg) EXTRA_MODEL_ARGS+=("$2"); shift 2 ;;
        --no-log-samples) LOG_SAMPLES=0; shift ;;
        --no-confirm-unsafe-code) CONFIRM_UNSAFE_CODE=0; shift ;;
        -o|--output-path) OUTPUT_PATH="$2"; shift 2 ;;
        --tag) TAG="$2"; shift 2 ;;
        --dry-run) DRY_RUN=1; shift ;;
        --) shift; EXTRA_ARGS=("$@"); break ;;
        *) echo "Error: unknown option: $1" >&2; usage; exit 1 ;;
    esac
done

HOST="${HOST%/}"

for value_name in BATCH_SIZE MAX_LENGTH MAX_GEN_TOKS; do
    value="${!value_name}"
    if ! [[ "$value" =~ ^[1-9][0-9]*$ ]]; then
        echo "Error: ${value_name} must be a positive integer. Got: $value" >&2
        exit 1
    fi
done

case "$TRUST_REMOTE_CODE" in
    true|false) ;;
    *) echo "Error: --trust-remote-code must be true or false." >&2; exit 1 ;;
esac
for flag_name in LOG_SAMPLES CONFIRM_UNSAFE_CODE; do
    flag_value="${!flag_name}"
    if [[ "$flag_value" != "0" && "$flag_value" != "1" ]]; then
        echo "Error: ${flag_name} must be 0 or 1. Got: $flag_value" >&2
        exit 1
    fi
done

LM_EVAL_BIN="${LM_EVAL_BIN:-$(command -v lm_eval || true)}"
PYTHON_BIN="${PYTHON_BIN:-$(command -v python3 || command -v python || true)}"

if (( ! DRY_RUN )) && [[ -z "$LM_EVAL_BIN" ]]; then
    echo "Error: lm_eval is not available. Install lm-evaluation-harness or set LM_EVAL_BIN." >&2
    exit 1
fi

if [[ -z "$MODEL" ]] && command -v curl >/dev/null 2>&1 && [[ -n "$PYTHON_BIN" ]]; then
    MODEL="$(curl -sf "${HOST}:${PORT}/v1/models" \
        | "$PYTHON_BIN" -c 'import sys, json; print(json.load(sys.stdin)["data"][0]["id"])' \
        2>/dev/null || true)"
fi
if [[ -z "$MODEL" ]] && (( DRY_RUN )); then
    MODEL="auto-detect-from-server"
elif [[ -z "$MODEL" ]]; then
    echo "Error: cannot detect a model from ${HOST}:${PORT}/v1/models; pass --model explicitly." >&2
    exit 1
fi

sanitize() {
    printf '%s' "$1" | sed -E 's#^/+##; s#/+#-#g; s/[[:space:]]+/_/g; s/[^A-Za-z0-9_.-]+/-/g; s/-+/-/g; s/^-+|-+$//g'
}

if [[ -z "$OUTPUT_PATH" ]]; then
    TIMESTAMP="$(TZ='Asia/Shanghai' date +%F-%H-%M-%S)"
    OUTPUT_PATH="./lm_eval_output_${TIMESTAMP}"
    if [[ -n "$TAG" ]]; then
        OUTPUT_PATH="${OUTPUT_PATH}_$(sanitize "$TAG")"
    fi
fi

BASE_URL="${HOST}:${PORT}/v1/completions"
MODEL_ARGS="model=${MODEL},base_url=${BASE_URL},max_length=${MAX_LENGTH},max_gen_toks=${MAX_GEN_TOKS},trust_remote_code=${TRUST_REMOTE_CODE}"
for model_arg in "${EXTRA_MODEL_ARGS[@]}"; do
    MODEL_ARGS+=",${model_arg}"
done

command_args=(
    --model local-completions
    --tasks "$TASKS"
    --model_args "$MODEL_ARGS"
    --batch_size "$BATCH_SIZE"
)
[[ -n "$NUM_FEWSHOT" ]] && command_args+=(--num_fewshot "$NUM_FEWSHOT")
[[ -n "$LIMIT" ]] && command_args+=(--limit "$LIMIT")
(( LOG_SAMPLES )) && command_args+=(--log_samples)
command_args+=(--output_path "$OUTPUT_PATH")
(( CONFIRM_UNSAFE_CODE )) && command_args+=(--confirm_run_unsafe_code)
if (( ${#EXTRA_ARGS[@]} > 0 )); then
    command_args+=("${EXTRA_ARGS[@]}")
fi

echo "=============================================="
echo "Start time        : $(TZ='Asia/Shanghai' date '+%F %T %Z')"
echo "Endpoint          : $BASE_URL"
echo "Model             : $MODEL"
echo "Tasks             : $TASKS"
echo "Batch size        : $BATCH_SIZE"
echo "Max length / gen  : $MAX_LENGTH / $MAX_GEN_TOKS"
echo "Output path       : $OUTPUT_PATH"
printf 'Command           : '
printf '%q ' "${LM_EVAL_BIN:-lm_eval}" "${command_args[@]}"
echo
echo "=============================================="

if (( DRY_RUN )); then
    echo "Dry run, not executing."
    exit 0
fi

mkdir -p "$(dirname "$OUTPUT_PATH")"
LOG_PATH="${OUTPUT_PATH%/}.log"
"$LM_EVAL_BIN" "${command_args[@]}" 2>&1 | tee "$LOG_PATH"