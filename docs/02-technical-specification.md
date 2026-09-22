# Part 1/2：技术规格、契约与验收

本文只写当前源码中存在的契约。字段实现主要见 [api.py](../src/lora_pipeline/api.py)、[store.py](../src/lora_pipeline/store.py)、[config.py](../src/lora_pipeline/config.py)、[common.py](../src/lora_pipeline/common.py)；单元/集成覆盖见 [tests/](../tests/)。

## 1. 组件、持久化与制品链

浏览器工作台及新增接口见 [Web Studio 指南](06-web-studio.md)。默认 `execution_mode=auto` 保留下文自动编排；`manual` 在冻结输入、训练、评估之后进入 `WAITING_FOR_USER`，由带幂等键的 `advance` 请求启动 `waiting_for_stage`。基础 profile 始终冻结，手动任务的训练/评估 overrides 在相应阶段首次入队时分别冻结。

| 组件 | 当前职责 | 关键证据 |
|---|---|---|
| Web | 英文四步向导、caption 编辑、有界上传、参数表单、状态/历史、产物预览下载；Nginx 同源代理 | [frontend](../frontend/) |
| API | Bearer owner、有界流式上传、校验请求、必要 mutation 幂等、状态/取消/下载、health、admin metrics | [api.py](../src/lora_pipeline/api.py) |
| Store | SQLite schema/短事务、配额、队列、attempt/lease/fencing、owner 隔离、唯一 model | [store.py](../src/lora_pipeline/store.py) |
| Worker | singleton supervisor、VERIFY、stage 子进程、heartbeat/watchdog、取消、恢复、资源调度 | [worker.py](../src/lora_pipeline/worker.py) |
| Data/Captions | 图片解码与规范化、去重/group split、user/template/BLIP caption | [data.py](../src/lora_pipeline/data.py)、[captions.py](../src/lora_pipeline/captions.py) |
| Training/Evaluation | SD15 LoRA、tiny offline graph、完整 checkpoint/resume、adapter reload、CLIP 诊断和 A/B | [training.py](../src/lora_pipeline/training.py)、[evaluation.py](../src/lora_pipeline/evaluation.py) |
| Common/Config | canonical JSON、SHA-256、atomic write、结构化 PipelineError、冻结配置 | [common.py](../src/lora_pipeline/common.py)、[config.py](../src/lora_pipeline/config.py) |

SQLite 是单主机队列：每个进程独立连接，BEGIN IMMEDIATE、foreign keys、短事务和 synchronous=FULL；模型执行和大文件 IO 不在 DB transaction 内。主要表为 datasets/dataset_files、jobs/stage_tasks/task_attempts、models、idempotency、scheduler_state。公开资源 ID 是 UUID；时间是 Unix UTC 秒；摘要是 SHA-256。写入制品后才把引用写入状态记录，旧 attempt 受 fencing token 保护。

持久化关系的最小评审 schema 如下（其余时间/错误字段省略）。这些是内部 SQLite 表，不是对外 API schema；外部 dataset 的 files 数组由 store.dataset() 组装后返回，object_key、fencing token 等内部字段不会作为公共契约暴露：

| 表 | 关键字段与关系 | 约束/用途 |
|---|---|---|
| datasets | id、owner_id、state、declared_count、frozen_manifest_path/sha256 | owner 作用域；dataset_files.dataset_id 外键；上传冻结/验证状态 |
| dataset_files（内部上传元数据） | id、dataset_id、object_key、size_bytes、sha256、uploaded、verification_status | object_key UNIQUE；dataset 删除级联；字节和原始摘要绑定 |
| jobs | id、owner_id、dataset_id、state、current_stage、profile_json、training_overrides_json、model_id | dataset 外键；job profile/input/training/evaluation JSON 是冻结快照 |
| stage_tasks | id、job_id、candidate_index、stage、state、active_token、lease_expires_at、output_json | UNIQUE(job_id,candidate_index,stage)；每个 successor 只能出现一次 |
| task_attempts | id、task_id、attempt_no、fencing_token、gpu_slot、pid/process identity、progress_json | UNIQUE(task_id,attempt_no) 与 UNIQUE(task_id,fencing_token)；旧 token 不能写回 |
| models | id、job_id、owner_id、state、adapter_path、adapter_sha256、manifest_path/sha256、report_json、test_only | UNIQUE(job_id)；PUBLISH 只登记 checksum 已核验的 model |

idempotency 以 (owner_id, route, key) 为主键并保存 request_sha256/response；scheduler_state 保存 cursor、owner rotation、leader epoch 和 worker heartbeat。

## 2. 数据与 manifest schema

服务入口的 dataset file metadata 是：

    {
      "name": "leaf-001.png",
      "size_bytes": 123456,
      "sha256": "64 lowercase hex chars",
      "mime_type": "image/png",
      "caption": "optional printable text <= 512 chars"
    }

POST /v1/datasets 接受 {"files": [...]}，数量 100–1,000；单文件 20 MiB、总量 2 GiB、像素 40 MP、最短边 256 是当前 DataConfig 默认服务限制。只接受 JPEG、PNG、static WebP；动画、坏图、尺寸/摘要错误会拒绝或记录 per-file rejection。prepare_dataset() 的 prepared.json 记录：

- schema_version、preprocessing_version、grouping_version、dataset_id、完整 config；
- train[] / validation[] 每项的 id/name/path/sha256/group_id 和 warning；
- statistics（submitted、accepted_unique、train/validation、groups、rejected、exact_duplicates、total_source_bytes）；
- rejected[] 与 duplicates[]。

归一化 PNG 路径是 pixel_hash_prefix-encoded_png_sha256.png；最终 PNG 字节摘要会写入 entry，重复 prepare 不覆盖已发布的 image artifact。training-input.json 还保存 prepared_manifest_path/sha256、captioning mode/trigger/source counts 和训练/验证项。训练、评估、checkpoint 都校验这一冻结链；图片或 manifest 漂移会给出 CHECKSUM_MISMATCH/INPUT_INCOMPATIBLE，details 包括 path、expected_sha256、actual_sha256（如可得）。

数据默认约 90/10 group-exclusive split，要求至少 80 train、10 validation、每边至少 2 groups；这是可配置约束和方向，不是声称任何输入都能满足。blur/brightness/contrast 只是 warning；训练默认 aspect-preserving resize + center crop，random crop/horizontal flip 必须显式打开，validation 不做随机增强。

## 3. 配置、profile 与训练/评估制品

当前 profile ID 是：

- local-sd15-v1 / key style-lora：真实 SD 1.5，默认 512、rank/alpha 4/4、batch 1、accumulation 4、learning rate 1e-4；配置文件 smoke 为 10 steps（configs/sd15-smoke.json），fp16 CUDA；服务 TrainConfig 默认 max_steps 是 500，不应把二者混为实测质量阈值。
- local-tiny-v1 / key tiny-test：仅 enable_test_backend=true 暴露，CPU FP32、16px tiny Diffusers/PEFT 随机图，制品永远 test-only。

profile 的 config/data/evaluation/caption/limits 会在 admission 时冻结到 job。用户可覆盖的 training fields 只有 max_steps、learning_rate、rank、lora_alpha、checkpoint_every、seed、batch_size、gradient_accumulation_steps；不可从 API 选择 base model、device、precision、路径、远程 URL 或任意 checkpoint。

训练结果至少含 kind=training-result、state、adapter_path/adapter_sha256、manifest_path/manifest_sha256、input_manifest_sha256、base model revision/fingerprint、config、global_step、elapsed_seconds、test_only；checkpoint 包含 adapter、optimizer、LR scheduler、AMP scaler（如适用）、RNG、sample cursor 和兼容性 key。只在配置的 checkpoint boundary 保存；损坏、部分或输入/config/base 不匹配是 CHECKPOINT_CORRUPT/CHECKPOINT_INCOMPATIBLE。

冻结 revision 契约：SD15 训练必须解析到 immutable snapshot revision，并把 revision 与 base fingerprint 写进 training-result/checkpoint；评估只允许同一 revision/fingerprint，snapshot 内容漂移或未 pin 是 BASE_SNAPSHOT_MISMATCH/BASE_REVISION_UNPINNED。tiny 测试使用明确的 local identity，不能伪称 SD15 revision。

基座解析失败的诊断会保留可操作字段但脱敏 model/revision/cache URL 与 token；不会把 HF token 或原始异常 URL 写入错误 details。见 [test_training.py](../tests/test_training.py) 的 base-model diagnostics cases。

评估 report 含 technical_pass、quality_status、quality_failures、metrics、test_only、adapter/input/base identity、config、paired_outputs、report_path/report_sha256。SD15 metrics 为 CLIP prompt score、heldout similarity、diversity、max train similarity、baseline 和 paired count；tiny metrics 明确 unavailable。A/B 复用相同 prompt/seed/inference settings，并拒绝不同 base revision、resolution 或 input manifest 的 adapter。

质量策略的 schema 和方向是实现事实，不是默认业务门槛：

    {
      "version": "operator-defined",
      "calibration_reference": "operator-supplied evidence identifier",
      "bounds": {
        "clip_prompt_score": {"min": 0.0},
        "heldout_similarity": {"min": 0.0},
        "diversity": {"min": 0.0},
        "max_train_similarity": {"max": 1.0}
      }
    }

上面的 0.0/1.0 仅是字段形状示意，不是默认或测得 threshold；运行配置必须提供 finite number。代码只检查 reference 字段非空、bounds 字段/数字/min≤max 和运行值方向；**不会读取、验证或证明 calibration_reference 指向的外部证据真实存在，也没有仓库内校准数据或测得 threshold**。默认 null → UNCALIBRATED；技术成功且非 test-only、policy PASS 才能得到 READY；PASS 但 test-only 仍为 UNCALIBRATED；policy FAIL 在 EVALUATE 后进入 QUALITY_REJECTED，不排 PUBLISH。已完整 policy 中指标不可用会按缺失 metric 进入 FAIL；只有 policy 缺失/不完整才是 UNCALIBRATED。

## 4. 状态机与 store 实际检查

| 业务/工作项 | 状态或顺序 | worker/store 的实际检查 |
|---|---|---|
| dataset VERIFY | UPLOADING → VERIFYING → COMPLETED/INVALID | complete_dataset 要求全部 file uploaded=1；worker 校验对象存在、大小、原始 SHA；finish_verification 写 valid/invalid counts、error、frozen manifest。VERIFY 是 dataset verification work，**不是 stage task**。 |
| job stages | PREPARE → CAPTION → TRAIN → EVALUATE → PUBLISH | claim_task 只领取 job ACCEPTED/RUNNING 且 task PENDING/RETRY_WAIT；每次 task 有 attempt token/lease。 |
| PREPARE | task PENDING/RUNNING/SUCCEEDED... | 从 store object keys 构造 trusted entries，写 prepared manifest；若 train captions 已齐可同次生成 input，否则创建 CAPTION successor。 |
| CAPTION | 同上 | 从 job frozen profile 读取 mode/device/model/revision；user 优先，BLIP 失败不会静默 template fallback。 |
| TRAIN | 同上 | 只读 job frozen input/profile/overrides；GPU slot 时改为 assigned cuda:0，test slot 强制 tiny CPU；保存 checkpoint 和 progress。 |
| EVALUATE | 同上 | 必须已有 training/input；重新校验 adapter/input/base，quality FAIL 时直接写 QUALITY_REJECTED，不创建 PUBLISH。 |
| PUBLISH | 独立 successor task | 只接受 active token、未过期 lease、存在且 checksum 匹配的 adapter/manifest/report/input binding；每 job models.job_id UNIQUE，终态为 READY 或 COMPLETED_UNVERIFIED。 |

job 终态为 READY、FAILED、QUALITY_REJECTED、CANCELLED、COMPLETED_UNVERIFIED；取消先持久化 CANCEL_REQUESTED，待无 running executor 才 CANCELLED。lease 到期先检查 PID/start time/PGID 对应进程是否已退出；旧 fencing token 无法更新 canonical progress/output。

## 5. REST API、幂等性与 schema

所有资源端点用 Authorization: Bearer <key>；跨 owner 查询统一 404 NOT_FOUND。当前端点：

| 方法 | 路径 | 输入/输出要点 |
|---|---|---|
| POST | /v1/datasets | files[] metadata；201 返回 dataset/file IDs。**需要 Idempotency-Key**。 |
| PUT | /v1/datasets/{dataset_id}/files/{file_id} | 原始 bytes，streaming size/hash 校验，204。按 file ID 和 checksum 重试安全；这是上传幂等逻辑，**不是通用 idem 表 mutation，不要求 Idempotency-Key**。 |
| POST | /v1/datasets/{dataset_id}/complete | 空 JSON；冻结并排 VERIFY，202 + Location。**需要 key**。 |
| GET | /v1/datasets/{id} | state、计数、错误、可选 files。 |
| GET | /v1/training-profiles | 实际 profile IDs、冻结 config 和 limits。 |
| POST | /v1/training-jobs | dataset_id/profile_revision_id/trigger_token/training_overrides；202 + status URL。**需要 key**。 |
| GET | /v1/training-jobs/{id} | job/stages/attempt/progress/error/model。 |
| POST | /v1/training-jobs/{id}/cancel | 空 JSON；202 CANCEL_REQUESTED。**需要 key**。 |
| GET | /v1/training-jobs/{id}/evaluation | report；未生成返回 conflict。 |
| GET | /v1/models/{id} | owner-scoped model/report/provenance/state。 |
| GET | /v1/models/{id}/download?allow_unverified=true | adapter bytes；UNVERIFIED 未显式 opt-in 时 409。 |
| GET | /health/live、/health/ready | live；ready 同时检查 DB、storage、worker heartbeat，未 ready 返回 503。 |
| GET | /metrics | admin key 的 Prometheus text；非 admin 401。 |

需要 idem 的 POST mutation 将 (owner, concrete route, Idempotency-Key) 与请求 hash 存入 idempotency；同 key 同 body replay 原响应，不同 body 是 409 IDEMPOTENCY_KEY_REUSED。读取、health、metrics 不需要；PUT 依赖 file object 的 byte identity/frozen-state 检查。API 错误固定为：

    {"error":{"code":"...","message":"...","retryable":false,"details":{},"request_id":"uuid"}}

`api.py` 的 PipelineError 由 middleware/exception handler 包装为上述 error 对象，并按 code 映射 HTTP 状态：`UNAUTHORIZED` → 401，`NOT_FOUND` → 404，列举的状态/幂等冲突 → 409，`ADMISSION_LIMIT` → 429，`GPU_UNAVAILABLE`/`WORKER_ALREADY_RUNNING` → 503；其余 `PipelineError`（包括存储配额超限）默认 → 422。Pydantic/FastAPI 请求校验错误（例如 body 字段类型不符）走默认 RequestValidationError 响应，形状是 422 和 detail 数组，不是统一 error wrapper。`/health/ready` 另按依赖状态返回 200/503。OpenAPI 是 FastAPI 运行时 schema。

## 6. CLI 错误、进度、日志与验收

CLI 成功时 stdout 是最终 JSON；train/evaluate/compare 进度逐行写 stderr（phase、current/total、百分比），失败返回 exit code 1，并在 stderr 输出：

    {"error":{"code":"CHECKSUM_MISMATCH","message":"...","details":{}}}

CLI 错误没有 API 的 HTTP status/retryable/request_id 包装；worker 则记录结构化 stage_started/stage_completed/stage_failed，attempt progress 写入 SQLite。训练 progress 字段含 global_step、loss、samples_processed、samples_per_second、checkpoint_seconds、CPU RSS，以及 CUDA allocated/reserved（不可得时应保持 null/not_measured）。[test_cli.py](../tests/test_cli.py) 和 [test_observability.py](../tests/test_observability.py) 验证输出边界。

验收必须区分三类结果：离线 pytest/tiny 只证明逻辑、LoRA/resume/契约；Docker Compose 只证明镜像/服务配置；真实 SD15 + 2–4 物理 GPU、CLIP 和性能 benchmark 必须在具备硬件的目标机运行，未运行字段写 not_measured，不能用 CPU tiny 数字冒充。入口和成功判据见[运行/API 指南](04-running-and-api.md)与[性能定义](05-performance-benchmarks.md)。
