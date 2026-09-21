# 运行与 API 指南

本文说明 Part 2 本地实现。假设 Linux/WSL2 与 Python 3.12；GPU 和 Docker 命令是可复现流程，不表示已在当前开发环境运行成功。

## 安装与预检

```bash
python3.12 -m venv /tmp/lora-pipeline-venv
source /tmp/lora-pipeline-venv/bin/activate
python -m pip install -U pip
python -m pip install -e '.[dev]'
python -m lora_pipeline preflight
```

真实 CUDA 路径需显式安装兼容的 PyTorch，再用 `preflight` 检查 `torch.cuda.is_available()` 和 `nvidia-smi`：

```bash
python -m pip install torch==2.8.0 torchvision==0.23.0 \
  --index-url https://download.pytorch.org/whl/cu128
```

SD 1.5、BLIP、CLIP 需要本地 Hugging Face cache；`--local-files-only` 明确禁止下载。CPU tiny 测试在本地初始化 Diffusers/PEFT 组件（随机权重），不下载预训练权重；它只验证技术链路，结果永远标记为 `test_only`。

通过 `LORA_CONFIG` 提供 JSON 配置，或使用 `.env.example` 中的环境覆盖。`LORA_API_KEYS` 是 token 到 owner ID 的 JSON 对象，API 没有 key 不会启动；`LORA_ADMIN_KEYS` 控制 `/metrics`。离线 CPU 测试需显式启用 test backend 和 fake slots：

```bash
export LORA_API_KEYS='{"demo-token":"owner-a"}'
export LORA_ADMIN_KEYS='["admin-token"]'
export LORA_TEST_BACKEND=1
export LORA_FAKE_SLOTS=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
python -m lora_pipeline preflight
```

默认数据根为 `./var`，含 `pipeline.sqlite3`、`objects/`、`artifacts/` 和 `locks/`。SQLite 使用 `DELETE` rollback journal 与 `synchronous=FULL`；停止服务时同时备份数据库和引用文件。

## CPU smoke

输出目录必须为空或不存在。命令生成 120 张合成图，执行准备/caption，tiny backend 运行 4 个 optimizer steps，在 step 2 中断并从完整 checkpoint 恢复，重载 adapter 后执行技术评估：

```bash
python -m lora_pipeline cpu-smoke --output var/smoke-cpu
python -m pytest
```

结果为 `test_only: true`、`quality_status: UNCALIBRATED`。它证明编排、梯度、checkpoint 恢复和 adapter 加载，不证明视觉质量或 GPU 性能。

## GPU smoke

在有 NVIDIA driver、允许的 GPU UUID 和本地模型权重的主机上：

```bash
python -m lora_pipeline preflight
python -m lora_pipeline gpu-smoke --output var/smoke-gpu \
  --model stable-diffusion-v1-5/stable-diffusion-v1-5 --local-files-only
```

真实 `sd15` backend 运行 10 steps，在 5 steps 后中断、恢复、导出/重载 adapter 并运行小评估。`--gpu-uuid` 选择 `nvidia-smi` 报告的 UUID，并使用与 worker 相同的设备锁。短 smoke 不是容量基准；只有实际运行后才可报告结果。

## CLI、API 与 worker

已实现命令为 `api`、`worker`、`worker-health`、`preflight`、`generate-data`、`prepare`、`caption`、`train`、`evaluate`、`compare`、`cpu-smoke`、`gpu-smoke`；精确 flags 使用 `python -m lora_pipeline <command> --help`。直接 CLI 流程使用仓库中的配置文件，并让评估生成配对的 baseline 与 adapter 输出：

```bash
python -m lora_pipeline generate-data --output var/generated --count 120
python -m lora_pipeline prepare --files var/generated/files.json --output var/prepared
python -m lora_pipeline caption --prepared var/prepared/prepared.json --output var/input \
  --mode template --trigger-token mystyle
python -m lora_pipeline train --input var/input/training-input.json \
  --config configs/sd15-smoke.json --output var/training
python -m lora_pipeline evaluate --training-result var/training/training-result.json \
  --input var/input/training-input.json --output var/evaluation \
  --config configs/evaluation-smoke.json
```

API 与 worker 使用同一配置和数据根，并应在两个终端启动：

```bash
export LORA_API_KEYS='{"demo-token":"owner-a"}'
export LORA_ADMIN_KEYS='["admin-token"]'
# 离线 CPU 测试时，两个终端都设置以下变量
export LORA_TEST_BACKEND=1
export LORA_FAKE_SLOTS=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
# 终端 1
python -m lora_pipeline api --host 127.0.0.1 --port 8000
# 终端 2：重复以上 export 后运行
python -m lora_pipeline worker
```

所有资源调用使用 `Authorization: Bearer demo-token`，变更请求必须带稳定的 `Idempotency-Key`。主要路由如下：

| 方法与路径 | 用途 |
|---|---|
| `POST /v1/datasets` | 提交 `name`、`size_bytes`、`sha256`、`mime_type` 和可选 `caption`。 |
| `PUT /v1/datasets/{dataset_id}/files/{file_id}` | 流式上传声明的文件字节。 |
| `POST /v1/datasets/{dataset_id}/complete` | 冻结上传并排队验证。 |
| `GET /v1/datasets/{dataset_id}` | 查询 owner 范围 dataset。 |
| `GET /v1/training-profiles` | 查询真实 profile；test backend 开启时另有 `tiny-test`。 |
| `POST /v1/training-jobs` | 用 `dataset_id`、`profile_revision_id`、`trigger_token` 创建 job。 |
| `GET /v1/training-jobs/{job_id}` / `POST /v1/training-jobs/{job_id}/cancel` | 查询或取消。 |
| `GET /v1/training-jobs/{job_id}/evaluation` | 查询评估报告。 |
| `GET /v1/models/{model_id}` / `GET .../download` | 查询或下载 adapter；未验证下载需 `?allow_unverified=true`。 |
| `GET /health/live`, `/health/ready`, `/metrics` | 存活、就绪和 admin 指标。 |

先创建 dataset、逐个 PUT 文件并 complete，等待状态为 `COMPLETED`，再创建并轮询 job。`COMPLETED_UNVERIFIED` 是默认质量 policy 下的正常技术结果，下载需显式同意；只有校准 policy PASS 才能 `READY`。跨 owner 资源故意返回 `404`。错误包含 `code`、`message`、`retryable`、`details`、`request_id`。仓库中的 `scripts/api_demo.py` 可作为真实 HTTP 上传、轮询和下载示例（需先启动 API 与 worker）。

## Docker/Compose

`Dockerfile`、`compose.yaml`、`compose.gpu.yaml` 和 `.env.example` 提供本地打包路径。默认 Compose 使用 CPU test backend；GPU overlay 要求 Docker daemon、NVIDIA Container Toolkit、driver 和模型权重：

```bash
docker compose -f compose.yaml up --build
docker compose -f compose.yaml -f compose.gpu.yaml up --build
```

持久化 Compose 的 `pipeline-data` 和 `model-cache` volumes，并在环境文件中设置真实 key。GPU/Docker 命令是目标主机流程；本次会话不声称 Docker build 或 GPU 已成功运行。

## 开发检查范围

`python -m ruff check src tests scripts/api_demo.py` 检查仓库源码；`python -m mypy` 按 `pyproject.toml` 当前配置只检查 `src/lora_pipeline/common.py` 和 `src/lora_pipeline/config.py`，不是全源码类型检查。Docker Compose 与真实 GPU smoke 需要目标环境，验证记录应分别报告其是否实际执行。
