# 自动 LoRA 训练流水线——实现规格

## 1. 组件与持久化

包 `lora_pipeline` 提供 FastAPI 应用、独立 worker 命令和 operator CLI，目标是 Linux/WSL2、Python 3.12。`api` 负责 bearer 认证、owner 授权、流式上传、幂等和状态操作；`store` 负责 SQLite 事务、配额、持久队列、attempt、调度 cursor 和 registry；`worker` 负责单例 supervisor、CPU/GPU 分发、watchdog、恢复和取消；`data`/`captions` 负责解码、去重、划分和 caption；`training` 支持真实 SD 1.5 LoRA 与 tiny 离线后端；`evaluation` 负责 adapter 重载、CLIP 诊断、policy 和报告；`common`/`config` 提供规范 JSON、SHA-256、原子写入和校验配置。

SQLite 每次使用独立连接，短事务执行 `BEGIN IMMEDIATE`、外键、busy timeout、`journal_mode=DELETE` 和 `synchronous=FULL`。它是单主机持久队列，不是 PostgreSQL 行锁的模拟。图像和权重保存在 `objects/`、`artifacts/`，临时写入后原子发布。记录包括 dataset/file、job、stage task、attempt、model、幂等和 scheduler state；UUID 标识公开资源，哈希为 SHA-256。

## 2. 数据准备与 caption

默认限制为声明 100–1,000 张、清理后至少 100 张有效且唯一图像、单文件 20 MiB、dataset 2 GiB、单图 40,000,000 像素、短边 256。允许 JPEG、PNG、静态 WebP；解码失败、动画和超限会被拒绝。规范化 EXIF 方向并转换 RGB，透明通道合成白底；按规范化像素做完全去重。近重复使用 64-bit perceptual hash，默认 Hamming 距离不超过 4，连通分组必须留在同一 split。

确定性 seed 产生约 90/10 的 group-wise 划分，并要求训练至少 80 张、验证至少 10 张、每侧至少两个独立组。无法分组返回 `DATASET_UNSUITABLE`，有效唯一图像不足返回 `DATASET_TOO_SMALL`。模糊、亮度和对比度是默认 warning，不会静默删除艺术性图像。训练默认保持比例 resize 后 center crop；random crop、horizontal flip 是显式开关，验证无随机增强。

用户 caption 优先；缺失项使用 generic template 或可选 BLIP。最终 caption 会附加配置的 style trigger；文本必须非空、有长度上限且无控制字符。BLIP 只在需要时加载，失败不会静默退回 template。prepared/training-input manifest 保存 split、文件身份、checksum、预处理版本、seed、caption 来源及父 manifest digest。

## 3. 训练、检查点与评估

真实后端是 SD 1.5：VAE、text encoder 和 UNet 基础参数冻结，只优化 attention LoRA。默认分辨率 512、rank/alpha `4/4`、batch 1、累积 4、AdamW `1e-4`、500 optimizer steps、CUDA FP16、gradient checkpointing、梯度裁剪 1、seed 42；每 50 步及结束时检查点。CPU tiny profile 使用 Diffusers 的 AutoencoderKL、UNet2DConditionModel、DDPMScheduler 和 PEFT LoRA 组成的随机初始化小图，分辨率 16/32、FP32，不下载预训练权重；其产物始终是 test-only。OOM 会给出失败，不会自动改变分辨率或优化器。

检查点只在 optimizer 边界提交，包含 adapter、optimizer、scheduler、适用的 AMP scaler、RNG、global step 和 sample position。compatibility key 绑定冻结输入、caption、模型 revision、配置、precision、累积、增强和实现版本；损坏或不兼容检查点拒绝。API 不接收任意序列化 checkpoint，可信 operator CLI 可 resume 本地流水线产生的检查点。

评估重新加载准确 adapter，并在相同设置下为冻结基础模型和 LoRA 生成成对套件。默认每个 variant 使用 20 prompts × 2 seeds；smoke 每个 variant 使用 2 prompts × 1 seed。CLIP 图文对齐、留出相似度、多样性及训练图像最大相似度都是诊断维度，不是独立质量证明。默认 policy 是 `UNCALIBRATED`；技术成功注册 `UNVERIFIED` 并以 `COMPLETED_UNVERIFIED` 结束。只有技术成功加版本化校准 policy PASS 才能 `READY`；test-only 永远不能 READY。comparison 在相同 prompts、seeds 和推理设置下生成成对结果。

## 4. API 契约与状态

资源端点需要 `Authorization: Bearer <key>`；key 映射 owner，跨 owner 访问返回 `404`。dataset 状态为 `UPLOADING → VERIFYING → COMPLETED | INVALID`；job 状态包括 `ACCEPTED`、`RUNNING`、`CANCEL_REQUESTED`、`CANCELLED`、`FAILED`、`QUALITY_REJECTED`、`COMPLETED_UNVERIFIED`、`READY`。stage 为 `PREPARE`、`CAPTION`、`TRAIN`、`EVALUATE`，发布在评估完成的事务中执行。

| 方法与路径 | 行为 |
|---|---|
| `POST /v1/datasets` | 提交 `name`、`size_bytes`、`sha256`、`mime_type` 及可选 `caption`。 |
| `PUT /v1/datasets/{id}/files/{file_id}` | 流式上传原始字节并校验大小/checksum。 |
| `POST /v1/datasets/{id}/complete` | 冻结并排入验证，返回 `202`。 |
| `GET /v1/datasets/{id}` | owner 范围的 dataset 状态。 |
| `GET /v1/training-profiles` | 真实 profile；启用 test backend 时额外返回 `tiny-test`。 |
| `POST /v1/training-jobs` | 使用 `dataset_id`、`profile_revision_id`、`trigger_token` 创建 job。 |
| `GET /v1/training-jobs/{id}` / `POST .../cancel` | 查询或请求取消。 |
| `GET /v1/training-jobs/{id}/evaluation` | 获取评估报告。 |
| `GET /v1/models/{id}` / `GET .../download` | 获取模型信息或 adapter；未验证下载需 `allow_unverified=true`。 |
| `GET /health/live`, `/health/ready`, `/metrics` | 存活、依赖就绪和 admin 认证的 Prometheus 风格指标。 |

变更请求使用按 owner、route 和 key 作用域的 `Idempotency-Key`；相同请求重放原结果，不同 body 冲突。错误包含 `code`、`message`、`retryable`、`details`、`request_id`。

## 5. 调度、恢复与验收

GPU 按 UUID 发现，每张物理 GPU 使用共享 OS 锁；supervisor 使用单例文件锁。类别循环为 `EVALUATE, TRAIN, CAPTION, TRAIN`，类别内 owner 轮转和 FIFO；CPU 有独立有限池。默认 heartbeat 10 秒、lease 60 秒、stage 上限 3,600 秒、GPU stage 累计 7,200 秒、基础设施 attempt 最多 3 次。取消、截止时间或控制 heartbeat 丢失会终止准确的进程组；设备复用要求确认进程退出。

关键测试覆盖图像限制、完全/近重复和 group 泄漏、确定性划分、caption 优先级、LoRA 梯度、完整 checkpoint 恢复、指标与 A/B 配对、owner 隔离、幂等、配额、公平 slot、过期结果、取消、恢复、唯一发布和显式未验证下载。CPU 使用合成图像和本地 tiny 组件，不下载预训练权重；RTX 4060 Ti 用户 smoke 是 10 steps、5 steps 中断恢复、adapter 重载和小评估。GPU 与 Docker 结果只有实际执行后才能报告。
