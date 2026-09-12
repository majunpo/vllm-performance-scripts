# vLLM Performance Scripts

个人常用的 vLLM 测试与性能分析脚本集合，用于保存和复用服务启动、模型评测、吞吐测试及 profiler 采集流程。

本仓库以实际测试脚本为主，不提供统一安装流程。使用前请根据本机环境修改模型路径、设备编号、端口和并行参数。

## 目录说明

| 目录 | 用途 |
| --- | --- |
| [`profile-scripts/`](profile-scripts/README.md) | 在 Intel XPU 或 NVIDIA GPU 上采集 torch profiler、unitrace，以及 prefill/decode trace |
| [`start-server-gsm8k-scripts/`](start-server-gsm8k-scripts/README.md) | 启动 vLLM OpenAI 兼容服务，并通过 lm-eval 或 vLLM evaluator 运行 GSM8K |
| [`vllm-bench-scripts/`](vllm-bench-scripts/) | 批量运行 `vllm bench serve`，保存日志并将性能指标解析为 CSV |

## 环境要求

- Linux 与 Bash
- Python 3
- 已安装并可正常运行的 vLLM
- 对应平台的 PyTorch 与设备运行时（Intel XPU 或 NVIDIA CUDA）
- 可选工具：`lm-eval`、unitrace、Perfetto

具体依赖和参数以各目录中的 README 及脚本 `--help` 输出为准。

## 快速使用

从仓库根目录执行命令。

查看 profiling 脚本的环境自检：

```bash
MODEL=/models/Qwen3-32B \
bash profile-scripts/xpu/run_torch_profile.sh --check
```

启动 vLLM 服务：

```bash
bash start-server-gsm8k-scripts/start_vllm_server.sh \
	--model-path /models/Qwen3-32B \
	--xpu-id 0 \
	--port 9001
```

使用配置文件运行 serving benchmark：

```bash
bash vllm-bench-scripts/run_bench_test.sh \
	-c vllm-bench-scripts/configs/qwen3_235b_2k_2k.conf
```

将 benchmark 日志解析为 CSV：

```bash
python vllm-bench-scripts/parse_vllm_bench.py \
	./benchmark_logs \
	-o ./benchmark_results.csv
```

## 使用提示

- 仓库中的模型路径和测试参数均为示例，运行前应按实际环境调整。
- 大规模 benchmark 前，建议先用较小的 batch、并发数或题目数量验证环境。
- Profiler trace 和 benchmark 日志可能占用较多磁盘空间，请定期整理输出目录。
- 新测试场景可复制 [`configs/template.conf`](vllm-bench-scripts/configs/template.conf) 后修改。
