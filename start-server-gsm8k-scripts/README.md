# vLLM Server 与 GSM8K 快速使用指南

本目录提供一个 vLLM server 启动脚本和两种 GSM8K 评测方式。默认面向 Intel XPU，也支持 NVIDIA CUDA。

## 脚本说明

| 脚本 | 用途 |
| --- | --- |
| [`start_vllm_server.sh`](start_vllm_server.sh) | 启动 vLLM OpenAI 兼容服务；默认 XPU |
| [`run_gsm8k_lm_eval.sh`](run_gsm8k_lm_eval.sh) | 使用 `lm_eval --model local-completions` 评测 GSM8K |
| [`run_gsm8k.sh`](run_gsm8k.sh) | 使用 vLLM 仓库自带的 `gsm8k_eval.py` 评测 |

以下命令均从仓库根目录执行。

## 1. 启动 XPU 服务

单卡 XPU，默认开启 XPU graph：

```bash
bash start-server-gsm8k-scripts/start_vllm_server.sh \
  --model-path /models/Qwen3.6-27B \
  --xpu-id 0 \
  --port 9001
```

4 卡 TP：

```bash
bash start-server-gsm8k-scripts/start_vllm_server.sh \
  --model-path /models/Qwen3.6-27B \
  --xpu-id 4,5,6,7 \
  --tensor-parallel-size 4 \
  --max-model-len 16384 \
  --max-num-seqs 128 \
  --kv-cache-dtype fp8 \
  --port 9001
```

常用 XPU graph 选项：

```bash
--graph 1                 # 开启 XPU graph，默认值
--breakable 1             # 开启 breakable graph
--enforce-eager           # 强制 eager，关闭 graph/compile 路径
```

## 2. 启动 NVIDIA CUDA 服务

CUDA 默认使用 CUDA graph；只有传入 `--enforce-eager` 才会关闭。

```bash
bash start-server-gsm8k-scripts/start_vllm_server.sh \
  --platform cuda \
  --model-path /llm/models/Qwen3.6-35B-A3B-NVFP4 \
  --served-model-name Qwen3.6-35B-A3B-NVFP4 \
  --cuda-devices 0,1,2,3 \
  --tensor-parallel-size 4 \
  --max-model-len 16384 \
  --port 30201
```

启动前只查看命令，不要求当前环境已经安装 vLLM：

```bash
bash start-server-gsm8k-scripts/start_vllm_server.sh \
  --platform cuda \
  --cuda-devices 0,1,2,3 \
  --tensor-parallel-size 4 \
  --dry-run
```

## 3. 检查服务

```bash
curl http://127.0.0.1:9001/v1/models
```

评测脚本的 `--port` 必须与 server 的端口一致。若不传 `--model`，两个评测脚本会从 `/v1/models` 自动获取模型 ID。

## 4. 使用 lm-eval 评测

需要安装 `lm-evaluation-harness`，并确保 `lm_eval` 在 `PATH` 中。

先运行 20 题 probe：

```bash
bash start-server-gsm8k-scripts/run_gsm8k_lm_eval.sh \
  --model /llm/models/Qwen3.6-35B-A3B-NVFP4 \
  --port 30201 \
  --limit 20 \
  --tag probe
```

完整 GSM8K：

```bash
bash start-server-gsm8k-scripts/run_gsm8k_lm_eval.sh \
  --model /llm/models/Qwen3.6-35B-A3B-NVFP4 \
  --port 30201 \
  --batch-size 64 \
  --max-length 16384 \
  --max-gen-toks 2048
```

该命令等价于以下核心调用：

```bash
lm_eval \
  --model local-completions \
  --tasks gsm8k \
  --model_args "model=<model>,base_url=http://127.0.0.1:<port>/v1/completions,max_length=16384,max_gen_toks=2048,trust_remote_code=true" \
  --batch_size 64 \
  --log_samples \
  --output_path ./lm_eval_output_<timestamp> \
  --confirm_run_unsafe_code
```

## 5. 使用 vLLM 自带 evaluator

设置 `VLLM_REPO` 指向 vLLM 源码目录。默认 `chat` 模式依赖脚本头部注明的 chat-completions patch。

Chat 模式快速测试：

```bash
VLLM_REPO=/home/junpo/applications.ai.gpu.vllm-xpu \
bash start-server-gsm8k-scripts/run_gsm8k.sh \
  --model Qwen3.6-27B \
  --port 9001 \
  --num-questions 20 \
  --mode chat \
  --thinking 0 \
  --tag probe
```

使用上游兼容的 raw completion 模式：

```bash
VLLM_REPO=/home/junpo/applications.ai.gpu.vllm-xpu \
bash start-server-gsm8k-scripts/run_gsm8k.sh \
  --port 9001 \
  --mode completion \
  --num-questions 1319 \
  --tag completion
```

## 6. 输出位置

- Server 日志：默认写入当前目录的 `server-logs/`。
- lm-eval：默认写入 `./lm_eval_output_<timestamp>`，日志为同名 `.log` 文件。
- vLLM evaluator：默认写入 `/home/junpo/gsm8k/`，可用 `--result-dir` 修改。

使用 `--help` 查看全部参数：

```bash
bash start-server-gsm8k-scripts/start_vllm_server.sh --help
bash start-server-gsm8k-scripts/run_gsm8k_lm_eval.sh --help
bash start-server-gsm8k-scripts/run_gsm8k.sh --help
```
