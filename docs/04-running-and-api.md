# 运行、验收与 API 指南

使用网站完成手动四步流程，见 [浏览器工作台部署与操作指南](06-web-studio.md)。下文保留 CLI/API 自动流程及 GPU 验收说明。

本文是实现边界和验收证据的索引。完整 GPU Docker runbook（IBean 准备、`dc run` heredoc、真实训练、评估和 benchmark 命令）在 [Docker CLI 训练与 GPU 验收指南](07-docker-cli-training.md)；这里保留入口、判据和 API 合约。更完整的作业导航见 [00-assignment-guide.md](00-assignment-guide.md)。

## 先区分三种证据

- **CPU tiny** 是离线回归后端，输入是 `generate-data` 生成的合成图，权重随机初始化，制品永远是 `test_only`。它可证明编排、梯度、checkpoint/resume、adapter 加载和 HTTP 合约，不能证明 SD 1.5 视觉质量、显存、吞吐或 GPU 扩展性。
- **10-step GPU smoke** 是目标 NVIDIA 主机功能验收：真实 SD 1.5、5 步中断后恢复、adapter 重载和小评估。它不是容量 benchmark，也不是质量阈值。
- **100-step GPU performance** 是固定真实输入和 pinned 基座上的性能实验，要求一次 warm-up 加至少三次完整运行；它与质量评估是不同证据。质量只来自固定 prompt/seed 的 paired base/adapter 评估和已校准 policy。

推荐真实验收数据是 IBean 本地副本：`datasets/ibean/images/` 中 999 张图（每类 333 张）和 `captions.json`，不是合成数据，也不是校准的风格质量集。下载、许可证、SHA-256 和整理方式以 [Docker CLI 指南的 IBean 小节](07-docker-cli-training.md#推荐的本地验证数据集ibean) 为准。

## 本地安装与 CPU 回归

```bash
python3.12 -m venv /tmp/lora-pipeline-venv
source /tmp/lora-pipeline-venv/bin/activate
python -m pip install -U pip
python -m pip install -e '.[dev]'
python -m lora_pipeline preflight
```

离线 tiny 路径必须显式打开测试后端；不要把这些变量带到真实 GPU 验收：

```bash
export LORA_API_KEYS='{"demo-token":"owner-a"}'
export LORA_ADMIN_KEYS='["admin-token"]'
export LORA_TEST_BACKEND=1
export LORA_FAKE_SLOTS=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
python -m lora_pipeline cpu-smoke --output var/smoke-cpu
python -m pytest -q
```

CPU smoke 默认生成 120 张合成图，在 step 2 中断、从完整 checkpoint 恢复到 step 4，再加载 adapter 并做技术评估。预期 `technical_pass=true`、`quality_status=UNCALIBRATED`、`test_only=true`；不要把旧耗时或“全通过”文字当作本次结果。`VALIDATION.md` 顶部是 2026-09-21 历史记录；当前 collection 可能已变化，不能把历史 63 tests 当最新总数。交付时记录本次日期和实际命令，未重跑写 `not_run`。

关键回归范围：

| 风险 | 测试/证据 | 判据 |
|---|---|---|
| 图片边界、精确/近重复、group split | `tests/test_data.py`、`tests/test_image_boundaries.py` | 坏图、超限、泄漏被拒；split 可复现 |
| caption 优先级和 BLIP 失败 | `tests/test_captions.py` | 用户 caption 优先；缺模型是明确错误，不退回 template |
| LoRA、梯度、checkpoint 兼容性 | `tests/test_training.py` | 恢复 cursor/adapter 正确；checksum/config 不匹配失败 |
| paired A/B、指标算术 | `tests/test_evaluation.py` | base/adapter 条件相同；技术成功与质量 policy 分开 |
| owner、幂等、HTTP、未验证下载 | `tests/test_service_contracts.py` | 跨 owner 为 404；相同 body 重放，改 body 冲突 |
| fencing、取消、重试、slot/预算 | `tests/test_scheduler_recovery.py`、`tests/test_service_review_fixes.py` | 旧 attempt 不能发布；OOM/预算/取消有终态 |

`ruff` 和当前 `mypy` 范围也要单独记录；mypy 只检查 `common.py` 和 `config.py`，不是全源码类型检查。Docker daemon、Compose 实际启动和真实 GPU 不能由 Python 测试代替。

## GPU Docker 主流程

目标主机需要 Linux/WSL2、Docker Engine/Compose v2、NVIDIA driver 和 Container Toolkit。GPU overlay 关闭 test backend/fake slots，只给 worker 请求 GPU；API 不使用 CUDA。Compose 的 `pipeline-data` 保存 SQLite、上传、manifest、checkpoint、adapter 和报告，`model-cache` 保存权重。

根目录 helper 必须在**每个新 shell**重新定义；`dc` 是 Bash 函数，不是 Docker 子命令：

```bash
dc() { docker compose -f compose.yaml -f compose.gpu.yaml "$@"; }
dc config --quiet
docker buildx version
dc build --progress=plain
dc run --rm --no-deps worker lora-pipeline preflight
```

没有 socket 权限时把函数体改成 `sudo docker compose ...`；不要执行 `sudo dc ...`。PowerShell 使用 `function dc { docker compose -f compose.yaml -f compose.gpu.yaml @args }`。确认 `preflight` 的 `cuda_available`、允许的 GPU UUID 和显存正确后再继续。

Docker CLI 指南的模块 1–4 是唯一 GPU 主 runbook。其 heredoc 容器命令使用 `-i -T`：`-i` 传 stdin，`-T` 禁用伪终端，兼容 SSH/CI/重定向；缺少 `-T` 会出现 `the input device is not a TTY`，不要手工伪造 manifest。`/data` 是容器内的 Compose named volume 路径，**不是宿主机 `/data`**；一次性容器 `--rm` 后文件仍在 `pipeline-data`，宿主机需用 `dc run ... cat` 或 `dc cp` 导出。

### 模型缓存、训练和进度

首次训练联网预热 `model-cache`，解析 immutable commit，并生成 `/data/sd15-smoke-pinned.json`；之后训练/恢复使用 pinned 配置和 `local_files_only=true`，下载时间不计入性能。详见 [Docker CLI 指南的 Training 模块](07-docker-cli-training.md#模块-2training)。

基座下载失败归类为 `BASE_MODEL_UNAVAILABLE`，不是 `CHECKPOINT_CORRUPT`、`CHECKPOINT_INCOMPATIBLE` 或 OOM；错误 details 只暴露脱敏模型/revision、缓存/offline 等受控分类，不承诺原始 root cause。先修复网络、认证、缓存或磁盘，再离线检查，不能反复启动训练代替预热。frozen input checksum 不一致时保留 path/expected/actual，重新 build 后完整重跑 prepare+caption 生成新 manifest；不要编辑旧 manifest 或跳过校验。规范化图片以最终 PNG 字节 SHA-256 内容寻址，新 encoder 不覆盖仍被旧 manifest 引用的文件。

直接 CLI 是**一个前台 terminal**；另开 terminal 只用于可选 `nvidia-smi`。CLI 在 stderr 输出 lifecycle/进度，stdout 只输出最终 JSON，失败返回非零：

```bash
python -m lora_pipeline train --input /path/training-input.json \
  --config /path/sd15-smoke-pinned.json --output /path/training
python -m lora_pipeline evaluate --training-result /path/training/training-result.json \
  --input /path/training-input.json --output /path/evaluation --config configs/evaluation-smoke.json
```

训练 progress 有 `global_step`、loss、累计 `samples_processed`、本次 `samples_processed_this_run`、吞吐、checkpoint 时间、RSS、CUDA allocated/reserved；API 查询位置是 `stages.TRAIN.progress`。评估 progress 有 technical smoke、model loading、generation、CLIP scoring、report writing。实现中的 evaluation `elapsed_seconds` 在 input validation 之后开始，并在 `report_writing`/HTML 与最终 manifest 写入之前计算，因此不覆盖验证和报告写入；checkpoint restore 也没有独立自动计时字段。没有外部 timer 时不要宣称精确完整 eval 或 restore 周期。

API + worker 是**两个进程/两个 terminal**；Compose 用 `dc up -d api worker` 后台运行同一拓扑。常驻 worker 启动后 `/health/ready` 才会从短暂 503 变 200。模块 1–3 一次性 GPU 容器不要与 worker 争用 GPU。

## REST API 合约和最小示例

```bash
export LORA_API_KEYS='{"demo-token":"owner-a"}'
export LORA_ADMIN_KEYS='["admin-token"]'
python -m lora_pipeline api --host 127.0.0.1 --port 8000   # terminal 1
python -m lora_pipeline worker                             # terminal 2
```

真实 profile 是 `local-sd15-v1`（`style-lora`）；仅显式 `LORA_TEST_BACKEND=1` 时，profiles 才额外返回 `local-tiny-v1`（`tiny-test`）。客户端不能提交 backend/device/precision 或任意 server path；overrides 仅允许 bounded knobs。

每个变更 POST（创建 dataset、complete、创建 job、cancel）需要 `Idempotency-Key`。键按 `owner_id + concrete route (method/path) + key` 作用域并绑定 body hash：相同请求重放原 status/body，改 body 返回 `IDEMPOTENCY_KEY_REUSED`。文件 PUT 是按 file ID、声明大小和 SHA-256 校验的流式上传，不接受任意路径；`scripts/api_demo.py` 的 POST helper 会设置幂等键。

```bash
curl -sS -H 'Authorization: Bearer demo-token' \
  http://127.0.0.1:8000/v1/training-profiles
curl -sS -X POST http://127.0.0.1:8000/v1/training-jobs \
  -H 'Authorization: Bearer demo-token' -H 'Content-Type: application/json' \
  -H 'Idempotency-Key: job-001' \
  -d '{"dataset_id":"<completed-dataset-id>","profile_revision_id":"local-sd15-v1","trigger_token":"mystyle"}'
curl -sS -H 'Authorization: Bearer demo-token' \
  'http://127.0.0.1:8000/v1/training-jobs/<job-id>'
```

job 创建是 `202 ACCEPTED`，初始 `state=ACCEPTED,current_stage=PREPARE`；终态包括 `READY`、`COMPLETED_UNVERIFIED`、`FAILED`、`QUALITY_REJECTED`、`CANCELLED`。默认 `quality_policy=null` 时技术成功是 `COMPLETED_UNVERIFIED`，非 test-only 且 policy `PASS` 才是 `READY`。非 READY 模型下载必须 `allow_unverified=true`。

由应用主动抛出的 `PipelineError` 才使用统一 HTTP JSON：`error.code/message/retryable/details/request_id`，并带 `X-Request-ID`；典型状态为 401、404、409、429、503 或 422。FastAPI/Pydantic 的请求体校验在路由函数之前执行，`RequestValidationError` 默认仍是 HTTP 422 的 `{"detail":[...]}`，不是上述 `error` envelope。CLI 则把 `{"error":{"code":"...","message":"...","details":{}}}` 写 stderr 并返回 1，stdout 保持最终成功 JSON；两者不要混用。

`scripts/api_demo.py` 是 **synthetic PNG + tiny 回归示例**：硬编码 `image/png`、优先 tiny，不能作为混合 MIME 的 IBean GPU 验收、benchmark 或质量证据。真实客户端须保留源 MIME、选择 `local-sd15-v1`，并分别设置 dataset/job timeout。

### API 验收表

| 阶段 | 证据 | 通过条件 |
|---|---|---|
| health/auth | live、ready、admin metrics | live 200；ready 的 database/storage/worker 正常；非 admin metrics 为 401 |
| dataset | 创建、每个 PUT 204、complete 后 GET | size/SHA/MIME 校验；最终 `COMPLETED`，否则 `INVALID` 有原因 |
| training | job、`stages.TRAIN.progress`、result/checkpoint | 终态明确；adapter、配置、基座 revision/fingerprint、input hash 有记录 |
| evaluation/quality | `/evaluation`、paired outputs、report | `technical_pass=true`；policy 未校准保持 `UNVERIFIED` |
| artifact | model 查询和下载文件 | adapter/manifest/report checksum 可重算；UNVERIFIED 需 opt-in |
| isolation/replay | 两 owner、相同/变更 body 重试 | 跨 owner 404；相同幂等请求只产生一个资源；变更 body 冲突 |

质量阈值、CLIP 局限和发布边界见 [技术规格](02-technical-specification.md) 与 [设计取舍](03-implementation-tradeoffs.md)。提交附 commit、命令日志、配置、manifest/model revision、checkpoint/quality/artifact 证据和 `VALIDATION.md` 本次日期；未执行 Docker/GPU/benchmark 写 `not_measured`/`null`。

## Compose 生命周期与资源约束

```bash
dc up -d api worker
dc ps
dc logs --tail=200 api worker
dc down                 # 保留 named volumes
# dc down -v 会删除数据库、checkpoint、adapter、模型缓存；导出后才可用
```

每个物理 GPU UUID 一个 slot；CPU fake slot 只做接纳/公平回归。owner/global queue limit、stage/job deadline、attempt/retry 和累计 GPU 秒预算均是 scheduler 约束。GPU OOM 是明确失败，不自动偷偷降低参数；改配置后重新测量并记录。SQLite + named volumes 是单机共同故障域，多机生产需要共享 DB/对象存储/执行器。

2–4 GPU、多 owner 的 fairness、queue wait、OOM/retry 成本和 occupancy 方案见 [性能基准指南](05-performance-benchmarks.md)。
