#!/usr/bin/env bash
set -euo pipefail

######################################
# 使用说明
######################################
usage() {
  cat <<EOF
用法: $0 -c <config_file> [选项]

必选参数:
  -c, --config <file>       测试配置文件路径

可选参数（会覆盖配置文件中的同名变量）:
  -m, --model <path>        模型路径
  -H, --host <host>         服务地址 (默认: 127.0.0.1)
  -p, --port <port>         服务端口 (默认: 9001)
  -i, --input-len <len>     随机输入长度
  -o, --output-len <len>    随机输出长度
  -r, --range-ratio <r>     随机范围比例 (默认: 0.2)
  -l, --logdir <dir>        日志目录 (默认: ./benchmark_logs)
  -h, --help                显示帮助

示例:
  # 使用配置文件运行
  $0 -c configs/qwen3_235b_2k_2k.conf

  # 使用配置文件，覆盖端口
  $0 -c configs/qwen3_235b_2k_2k.conf -p 8080

  # 使用配置文件，覆盖 input/output 长度
  $0 -c configs/qwen3_235b_2k_2k.conf -i 4096 -o 2048

配置文件格式参见 configs/ 目录下的示例。
EOF
  exit "${1:-0}"
}

######################################
# 默认值
######################################
CONFIG_FILE=""
CLI_MODEL=""
CLI_HOST=""
CLI_PORT=""
CLI_INPUT_LEN=""
CLI_OUTPUT_LEN=""
CLI_RANGE_RATIO=""
CLI_LOGDIR=""

######################################
# 解析命令行参数
######################################
while [[ $# -gt 0 ]]; do
  case "$1" in
    -c|--config)      CONFIG_FILE="$2";      shift 2 ;;
    -m|--model)       CLI_MODEL="$2";        shift 2 ;;
    -H|--host)        CLI_HOST="$2";         shift 2 ;;
    -p|--port)        CLI_PORT="$2";         shift 2 ;;
    -i|--input-len)   CLI_INPUT_LEN="$2";    shift 2 ;;
    -o|--output-len)  CLI_OUTPUT_LEN="$2";   shift 2 ;;
    -r|--range-ratio) CLI_RANGE_RATIO="$2";  shift 2 ;;
    -l|--logdir)      CLI_LOGDIR="$2";       shift 2 ;;
    -h|--help)        usage 0 ;;
    *)                echo "未知参数: $1"; usage 1 ;;
  esac
done

if [[ -z "$CONFIG_FILE" ]]; then
  echo "❌ 错误: 必须通过 -c 指定配置文件"
  echo ""
  usage 1
fi

if [[ ! -f "$CONFIG_FILE" ]]; then
  echo "❌ 错误: 配置文件不存在: $CONFIG_FILE"
  exit 1
fi

######################################
# 加载配置文件
######################################
# 配置文件中定义的变量（默认值在 source 之前设定）
MODEL=""
HOST="127.0.0.1"
PORT="9001"
RANDOM_INPUT_LEN=""
RANDOM_OUTPUT_LEN=""
RANDOM_RANGE_RATIO="0.2"
LOGDIR="./benchmark_logs"
TEST_CONFIGS=()

# shellcheck source=/dev/null
source "$CONFIG_FILE"

######################################
# CLI 参数覆盖配置文件
######################################
[[ -n "$CLI_MODEL" ]]       && MODEL="$CLI_MODEL"
[[ -n "$CLI_HOST" ]]        && HOST="$CLI_HOST"
[[ -n "$CLI_PORT" ]]        && PORT="$CLI_PORT"
[[ -n "$CLI_INPUT_LEN" ]]   && RANDOM_INPUT_LEN="$CLI_INPUT_LEN"
[[ -n "$CLI_OUTPUT_LEN" ]]  && RANDOM_OUTPUT_LEN="$CLI_OUTPUT_LEN"
[[ -n "$CLI_RANGE_RATIO" ]] && RANDOM_RANGE_RATIO="$CLI_RANGE_RATIO"
[[ -n "$CLI_LOGDIR" ]]      && LOGDIR="$CLI_LOGDIR"

######################################
# 参数校验
######################################
ERRORS=()
[[ -z "$MODEL" ]]            && ERRORS+=("MODEL 未设置")
[[ -z "$RANDOM_INPUT_LEN" ]] && ERRORS+=("RANDOM_INPUT_LEN 未设置")
[[ -z "$RANDOM_OUTPUT_LEN" ]] && ERRORS+=("RANDOM_OUTPUT_LEN 未设置")
[[ ${#TEST_CONFIGS[@]} -eq 0 ]] && ERRORS+=("TEST_CONFIGS 未设置或为空")

if [[ ${#ERRORS[@]} -gt 0 ]]; then
  echo "❌ 配置错误:"
  for err in "${ERRORS[@]}"; do
    echo "   - $err"
  done
  echo ""
  echo "请检查配置文件: $CONFIG_FILE"
  exit 1
fi

######################################
# 生成日志路径
######################################
MODEL_NAME=$(basename "$MODEL")
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOGFILE="$LOGDIR/vllm_bench_${MODEL_NAME}_input_${RANDOM_INPUT_LEN}_output_${RANDOM_OUTPUT_LEN}_${TIMESTAMP}.log"

mkdir -p "$LOGDIR"

######################################
# 显示完整配置
######################################
echo ""
echo "====== 运行配置 ======"
echo "  配置文件:       $CONFIG_FILE"
echo "  模型:           $MODEL"
echo "  服务地址:       $HOST:$PORT"
echo "  Input/Output:   ${RANDOM_INPUT_LEN}/${RANDOM_OUTPUT_LEN}"
echo "  Range Ratio:    $RANDOM_RANGE_RATIO"
echo "  测试组数:       ${#TEST_CONFIGS[@]}"
echo "  日志文件:       $LOGFILE"
echo "  测试参数:       ${TEST_CONFIGS[*]}"
echo "======================"
echo ""

######################################
# 固定的 bench 参数
######################################
COMMON_ARGS=(
  --backend vllm
  --model "$MODEL"
  --trust-remote-code
  --host "$HOST"
  --port "$PORT"
  --dataset-name random
  --random-input-len "$RANDOM_INPUT_LEN"
  --random-output-len "$RANDOM_OUTPUT_LEN"
  --random-range-ratio "$RANDOM_RANGE_RATIO"
  --request-rate inf
  --seed 0
  --ignore-eos
  --num-warmup 10  # 预热请求数量
)

######################################
# 开始测试（日志输出格式不变）
######################################
echo "==================================================" | tee -a "$LOGFILE"
echo "VLLM BENCH MATRIX TEST START: $(date)" | tee -a "$LOGFILE"
echo "MODEL: $MODEL" | tee -a "$LOGFILE"
echo "RANDOM_INPUT_LEN: $RANDOM_INPUT_LEN" | tee -a "$LOGFILE"
echo "RANDOM_OUTPUT_LEN: $RANDOM_OUTPUT_LEN" | tee -a "$LOGFILE"
echo "RANDOM_RANGE_RATIO: $RANDOM_RANGE_RATIO" | tee -a "$LOGFILE"
echo "Total test configurations: ${#TEST_CONFIGS[@]}" | tee -a "$LOGFILE"
echo "==================================================" | tee -a "$LOGFILE"

TEST_COUNT=0
for CONFIG in "${TEST_CONFIGS[@]}"; do
  # 解析配置：num-prompts:max-concurrency
  IFS=':' read -r NUM_PROMPTS MAX_CONCURRENCY <<< "$CONFIG"

  TEST_COUNT=$((TEST_COUNT + 1))

  echo "" | tee -a "$LOGFILE"
  echo "==================================================" | tee -a "$LOGFILE"
  echo "TEST [$TEST_COUNT/${#TEST_CONFIGS[@]}] START: $(date)" | tee -a "$LOGFILE"
  echo "Configuration: num-prompts=$NUM_PROMPTS, max-concurrency=$MAX_CONCURRENCY" | tee -a "$LOGFILE"
  echo "==================================================" | tee -a "$LOGFILE"

  vllm bench serve \
    "${COMMON_ARGS[@]}" \
    --num-prompts "$NUM_PROMPTS" \
    --max-concurrency "$MAX_CONCURRENCY" \
    2>&1 | tee -a "$LOGFILE"

  echo "" | tee -a "$LOGFILE"
  echo "TEST [$TEST_COUNT/${#TEST_CONFIGS[@]}] END: $(date)" | tee -a "$LOGFILE"
  echo "==================================================" | tee -a "$LOGFILE"

  # 防止过快连续压服务
  sleep 5

done

echo "==================================================" | tee -a "$LOGFILE"
echo "ALL TESTS FINISHED: $(date)" | tee -a "$LOGFILE"
echo "==================================================" | tee -a "$LOGFILE"

echo ""
echo "✅ 全部测试完成，日志已保存到: $LOGFILE"
