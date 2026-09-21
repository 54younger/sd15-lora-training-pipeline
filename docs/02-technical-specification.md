# 自动风格 LoRA 训练流水线——技术规格

## 1. 目的与范围

本服务将 tenant 所有的图像数据集转换为现有图像生成推理消费者可以加载的、带版本的**风格 LoRA**。一次提交包含 100–1,000 张图像。服务会验证并清理图像，可选地生成面向内容的 captions，在冻结基础模型的同时训练 LoRA，依据固定的 quality policy 进行评估，并且只逻辑注册通过的制品。服务不设计或运营常驻模型服务平台；这里的“部署”指向现有推理消费者完成经过验证的交接。

设计有意针对由两到四张 GPU 组成的资源池。系统可以接收并发 job，但每个独占 GPU 槽位最多运行一个获授权的 GPU 阶段。所有 GPU 工作——可选 caption、训练、评估生成以及已保存 adapter 的加载冒烟测试——都使用同一个资源池。一个 job 最多有一个当前获授权的 GPU attempt；v1 不将一个 job 拆到多个 GPU，也不在一张 GPU 上并置独立 job。主机发生分区后，旧的 fenced attempt 可能在 host watchdog 终止它之前，与另一健康槽位上的重试发生物理重叠；这不是物理意义上的 exactly-once 执行。未确认前绝不复用原设备，两个 attempt 都计入保守预算和 owner eligibility。

逻辑组件可以作为少量可部署进程运行，不必拆成独立微服务：

| 组件 | 职责 |
| --- | --- |
| API/auth service | tenant 授权、dataset/job/model API、幂等性、接纳配额检查、预签名对象存储上传授权。 |
| PostgreSQL | 权威元数据、持久化 stage-task 队列、lease、公平性 cursor、状态转换、幂等记录和 registry 记录。 |
| Scheduler/reconciler | active/standby controller，分配兼容的 GPU 阶段，检测过期 lease，驱动重试/取消，并协调被中断的发布。 |
| CPU workers | 解码/验证/去重/划分图像，验证制品，发布 registry 记录，并删除过期的孤立制品。 |
| GPU host agent | 在本地锁定的设备上启动隔离且固定环境的阶段容器；执行截止时间、报告 heartbeat，并隔离不健康设备。 |
| Object storage | 兼容 S3 的存储，用于保存上传内容、清理后的 manifest、captions、检查点、attempt 输出、不可变制品 manifest、评估报告和 adapter。 |
| Registry/inference handoff | 由数据库和对象存储支持的逻辑 model registry。只有固定推理环境的加载测试成功后，才暴露 `READY` 模型。 |
| Observability | 结构化日志、指标、trace/attempt 审计事件、告警以及备份/恢复监控。 |

轮询 job API 是权威方式。可选 event 或 webhook 只能作为唤醒提示；由于通知可能重复、延迟或丢失，客户端必须重新读取 job。

## 2. 工作流、状态模型与组件行为

### 提交与 CPU 准备

`POST /v1/datasets` 记录上传 manifest，并返回限定 tenant 的预签名 PUT URL。客户端只上传列出的对象，然后调用 `complete`。上传 bucket 必须支持对象版本控制：完成操作解析准确的对象版本 ID，然后原子持久化该快照并加入验证队列；服务不信任文件扩展名或用户提供的 caption。此前发出的 PUT URL 可以创建更新的对象版本，但不能改变验证或训练读取的固定版本。冻结快照不能新增对象，也不能接收可写 URL 替换。验证器检查对象存在性、字节大小、checksum、MIME 声明和任何已上传的 caption，然后将 dataset 转为 `COMPLETED` 或 `INVALID`。

创建 job 会在 `SERIALIZABLE` 隔离级别执行一个短 PostgreSQL 事务：验证 ownership、dataset 已完成、所选 profile 受支持且已发布、存在经过校准的 quality policy，并确认持久化的用户级/全局接纳配额允许创建；插入 job、candidate-0 记录和第一个 `PREPARE` task；保存幂等结果。Candidate 0 起初没有准备好的 training input；只有 PREPARE/CAPTION 绑定该不可变输入后，才能将 TRAIN 加入队列。序列化失败会在同一个幂等 key 下进行有界内部重试，防止并发请求同时消耗最后一个配额位置。因此，API 确认和队列插入之间不会丢失已返回的 job。由于 100–1,000 张图像的深度检查可能耗时，该检查保持异步。

`PREPARE` 在 CPU 上并行执行。它在严格的文件、像素、总大小和解压限制下解码；通过 MIME sniffing 验证，而不是信任 manifest；移除损坏、重复和 policy 禁止的图像；写入清理后的不可变 manifest；并创建约 90/10 的**按组（group-wise）**训练/留出划分。近重复图像以及属于同一检测分组的全部图像必须留在同一侧，以防止验证集泄漏。受保护的完成事务创建不可变的 `prepared_dataset_version`，其中包含清理/训练/留出 manifest checksum、预处理器版本、划分 seed/分组和产生它的 attempt，然后将该准确版本绑定到 job。模型相关处理或使用缓存前，划分 manifest、分组和 seed 都不可变。清理后且划分前，job 至少需要 100 张有效且不重复的图像；还必须满足 profile 的适用性边界，包括足够的独立分组以及留出主题/内容覆盖，使划分具有意义，否则失败并返回 `DATASET_UNSUITABLE`。数量不足则失败并返回 `DATASET_TOO_SMALL`；两种情况都不消耗 GPU 容量。

`CAPTION` 是可选的，由 profile 控制。上传的 caption 经过内容/安全验证后可以复用。如果训练划分中的每个文件已有 caption，PREPARE 会在其受保护的完成事务中派生最终 `training_input_version`，并直接将 `TRAIN` 加入队列。否则必须启用 captioning，否则 PREPARE 失败并返回 `DATASET_CAPTIONS_REQUIRED`。CAPTION 生成缺失的、面向内容的描述并派生最终版本，帮助区分风格和图中主体。该版本包含 caption 制品/checksum/version，以及指向 prepared version 的外键；它不能修改留出集成员关系。如果 CAPTION 在 GPU 上运行，它与训练和评估使用完全相同的 scheduler 和预算。TRAIN、检查点、EVALUATE 和 PUBLISH 都固定并验证这条 manifest 链。只有影响中间产物的所有输入都匹配时，冻结的中间产物才能按内容寻址并复用：规范化图像字节、不可变划分成员关系、预处理和增强等价性、captioner 版本/设置、profile 字段，以及影响输出的任何 seed 或 trigger token。输入派生的缓存命名空间按 tenant 隔离；公共基础模型制品可以共享。

### 训练、检查点与评估

每个不可变的 training-profile revision 固定基础模型版本、训练容器 digest、分辨率、LoRA 方法/rank、optimizer 设置、learning rate、steps、batch size、precision、预计设备显存以及资源/时间预算。它还固定 caption 和 evaluator 版本，或声明关闭 captioning。profile 会针对受支持的 GPU 槽位类别预先验证；不支持的组合在接纳前拒绝。profile 可以定义一个预先验证的 OOM fallback 和一个不可变的质量补救 variant。job API 不提供自由形式的 override。

`TRAIN` 加载 candidate 对应的准确 profile revision，冻结所有基础模型参数，只更新 LoRA adapter 权重。默认至少每五分钟写入一次检查点，检查点包含 adapter、optimizer state、scheduler state、RNG state 和数据位置。只有不可变且带 checksum 的 manifest 提交后，检查点才可用于恢复。恢复只允许用于相同 dataset snapshot、candidate、profile revision、容器 digest 和兼容检查点；否则 candidate 必须重新开始。特别是，optimizer 配置变化会重新开始训练，除非 profile 包含单独验证过的状态转换流程。这遵循可靠恢复时保存完整训练状态而非只有模型权重的实际要求（[Accelerate 检查点指南](https://huggingface.co/docs/accelerate/usage_guides/checkpoint)）。训练方法采用 [Diffusers LoRA 训练文档](https://huggingface.co/docs/diffusers/training/lora)所述的标准参数高效 LoRA 适配。

`EVALUATE` 在 profile 固定的生成和推理环境中运行。它使用不可变的留出参考集、带版本的固定 prompt 套件（请求的主体不同于训练图像）和固定 seed。示例默认值是 20 个 prompt × 2 个 seed（40 个输出），可在 quality policy 中配置；这是设计默认值，不是吞吐量或质量基准。预算感知的筛选可以提前停止明显失败的 candidate，但不能发布模型；必须通过完整的 policy 评估。随后，评估在固定推理环境中加载保存的 adapter 并执行一次冒烟生成，在发布前捕获打包和兼容性缺陷。

质量是多维的：

| 维度 | 证据与门禁 |
| --- | --- |
| 风格一致性 | 带版本的风格导向 evaluator 与留出参考进行比较，并用人工标注校准。图像-图像 CLIP 相似度可以作为诊断代理，但绝不能是唯一门禁，因为内容会造成混淆。 |
| 文本遵循度 | 在未见过的主体和场景上进行 prompt/图像评分与 evaluator 检查。 |
| 多样性与记忆防护 | 输出多样性统计，加上与训练图像的相似度检索，用于发现坍缩或过度复制。 |
| 技术有效性 | 图像生成成功；adapter checksum、元数据以及固定环境加载均有效。 |

所选 metric model 版本、prompt 套件、聚合规则、不确定性处理以及每个维度的下限/边界都属于不可变的 `quality_policy`。本规格有意不臆造数值阈值：启用 profile 前，必须针对具有代表性的数据集和人工标注进行线下校准。如果没有固定的已校准 policy，创建 job 返回 `409 PROFILE_NOT_READY`。训练 loss 是健康度信号，不是接收标准。风格评估应保留人工校准组件；[StyleDrop](https://research.google/blog/styledrop-text-to-image-generation-in-any-style/)等已发表工作也将自动相似度指标视为人工评估的补充。

失败的 candidate 最多获得一次质量补救 attempt：candidate 0 只能使用 profile 中不可变且受支持的 variant ID 生成 candidate 1，该 ID 会记录在 candidate 上。policy 和阈值不会放宽。Candidate-1 variant 属于独立的检查点兼容域；只能恢复该 variant 有效的检查点，optimizer 变化则按照上文重新开始。质量补救不同于基础设施重试，使用相同的 evaluation policy 和剩余累计 GPU 预算。最终质量失败返回 `QUALITY_REJECTED`，报告说明失败维度和不确定性。

### 状态、重试、发布与取消

job 状态为 `ACCEPTED`、`RUNNING`、`CANCEL_REQUESTED`、`READY`、`FAILED`、`QUALITY_REJECTED` 或 `CANCELLED` 之一。`current_stage` 独立取值为 `PREPARE`、`CAPTION`、`TRAIN`、`EVALUATE` 或 `PUBLISH`。每个 candidate/stage 的 task 状态为 `PENDING`、`RUNNING`、`RETRY_WAIT`、`SUCCEEDED`、`FAILED` 或 `CANCELLED`。同时返回两种状态，可以让客户端区分排队中的 GPU 阶段和实际执行中的阶段。

基础设施错误的每个 task 最多执行三次 attempt（包含第一次），采用有界指数退避和 jitter。reconciler 将损坏输入、不支持的配置和 policy 失败归类为不可重试；临时 worker、主机和对象存储故障可以重试。每个 attempt——包括评估和重试——都计入 profile 的累计 GPU 预算。超出预算时 job 失败并返回 `GPU_BUDGET_EXCEEDED`；任何重试或质量补救都不能绕过该限制。

包括 CPU 阶段在内的每个阶段，都通过一个受保护事务完成：检查当前 attempt 的 fencing token、未过期 lease、task 状态 `RUNNING` 以及 job 状态 `ACCEPTED` 或 `RUNNING`；记录并验证不可变的输出 manifest 引用；将当前 task 标记为 `SUCCEEDED`；并插入唯一的后继 task 或提交终态结果。TRAIN/EVALUATE/PUBLISH 要求 candidate 已绑定非空且不可变的 training-input 和兼容性信息。这样可以防止 worker 在制品已存储但下一条队列记录尚未写入时崩溃而丢失状态转换。过期、失效、已取消或属于终态 job 的 attempt 不能绑定 prepared/training/evaluation manifest，也不能推进工作流。

发布是**effectively once**，而不是执行 exactly once。publish worker 先写入以 attempt 为范围的制品，验证 checksum、针对准确 adapter checksum 的完整通过报告，以及固定环境的加载/冒烟结果，然后写入不可变 model manifest。受保护的数据库事务创建或查找该 job 的唯一模型，完成 task，并将 job 和模型转为 `READY`。事务要求当前且未过期的 fencing token，且没有取消请求。只有事务完成后 registry 才能暴露 READY 模型；部分对象存储写入或未提交的 registry 行不可公开。对象存储与数据库不是原子操作，因此未被引用的部分写入会在 TTL 后清理；reconciliation 会找到已提交的唯一模型，而不是创建重复模型。失败的 candidate 永远不会被发现为 READY。

取消是持久化的。请求会将符合条件的 job 改为 `CANCEL_REQUESTED`；worker 在安全点停止，reconciler 只有在每个 executor 都确认进程终止后才将其改为 `CANCELLED`。fencing 和隔离可以阻止旧 executor 提交，但单独不足以使取消进入终态。对于永久无法访问的主机，operator 记录经过审计的终止证明，该证明由准确 host instance/attempt 的 instance-control-plane 关机/删除证据或 BMC 断电证据支持。该证据可以替代 agent 确认；经过的时间或手工状态覆盖不可以。没有证据时 job 保持 `CANCEL_REQUESTED`，保留配额/受保护制品，并向 operator 告警。退役设备在替换并通过健康检查前保持不可用。取消不声称 GPU 能够即时终止。带锁的 job 状态转换解决与发布的竞争：job 已经 `READY` 后发起请求会得到 `409 JOB_ALREADY_READY`。

## 3. GPU 调度与分布式正确性

每个 GPU 设备由一个独占的 `gpu_slot` 表示，具有经过 profile 的能力类别以及 `HEALTHY`、`BUSY`、`DRAINING` 或 `QUARANTINED` 状态。host agent 获取设备级独占锁，启动隔离的阶段容器；只有进程终止且设备健康检查通过后，才将 GPU 返回为 `HEALTHY`。网络分区或 lease 过期本身不会释放本地设备：设备会被隔离，直到 agent/watchdog 确认安全。重试可以运行在另一兼容且健康的槽位上。这一设计有意优先保证安全，而不是立即恢复可用性。

 scheduler/reconciler 采用 active/standby。单个持久化的 `scheduler_guard` 行保存 leader epoch 和公平性 cursor。每个短 GPU dispatch 事务使用 `FOR UPDATE` 锁定/检查它，然后串行化兼容性选择、owner eligibility/活跃计数、槽位申领、task 申领、预算预留和 cursor 更新；模型训练期间不会持有该锁。接管操作会递增 epoch，所有 dispatch/start 更新都要求该 epoch；即使两个 controller 会选择不同槽位，也会拒绝 split-brain controller。candidate task 先按兼容性选择，然后遵循持久化的加权阶段循环：

`EVALUATE → TRAIN → CAPTION → TRAIN`

空的或不符合条件的类别会被跳过。在每个类别内，scheduler 在 owner 之间使用公平 round-robin，在同一 owner 的可执行 task 之间使用 FIFO。当另一个 owner 有资格执行时，每个 owner 最多拥有一个活跃或保守意义上未确认的 GPU task；没有其他符合条件的 owner 时，可以借用空闲容量。另一个 owner 到达后，已借用的工作不会被抢占；每个 owner 的上限只适用于另一个 owner 有资格时的新授权。持久化的循环 cursor 和最早符合条件的选择，使 failover 对审计足够确定。阶段截止时间以及队列年龄指标/告警可以暴露饥饿；在有限的已接纳积压下，该 policy 确保符合条件的类别会被再次访问，而不会永久将全部容量让给训练。没有 GPU 被永久预留给评估。

在受保护的 dispatch 事务内，`FOR UPDATE SKIP LOCKED` 申领符合条件的 task/slot 行；无关的已锁工作会被跳过，不会延迟恢复或 reconciliation（[PostgreSQL 文档](https://www.postgresql.org/docs/current/sql-select.html)）。单例 guard 有意串行化新的 GPU 授权，以保证公平性和预算记账正确。接纳上限通过事务保证，设计默认值是每个用户 5 个非终态 job、全局 100 个；超出请求返回 `429` 和 `Retry-After`。这些是可调整的运维默认值，不是容量基准。

容量规划可以采用以下保守的示例估算：

`admitted GPU seconds per period ≤ healthy slots × usable seconds per slot × target utilization`

左侧包含预留的 train、caption、evaluation、smoke-test、retry 和 remediation 预算。它用于队列/接纳决策；实际训练时长和利用率必须针对每个已部署 profile 测量，不能从公式推断。

每个运行中的 attempt 都有基于 DB 时钟的 lease 过期时间和单调递增的 fencing token。agent 每 10 秒发送 heartbeat，标准 lease 为 60 秒。任何状态写入、检查点提交、heartbeat 或 registry 发布都携带该 token；过期或不匹配的 token 会被拒绝。过期后，controller 撤销旧 epoch，并且只在剩余预算内将工作重新入队。lease 丢失或达到截止时间后，agent watchdog 停止工作。在确认终止前，保守记账会保留未确认 attempt 的预算暴露。

如果 PostgreSQL 不可用，controller 不启动新工作，worker 也不发布任何内容。agent 以 fail-closed 方式运行，并在 lease 过期前停止。这会暂时降低吞吐量，但可以避免重复发布。对象存储故障同样会阻止阶段提交输出。两到四张 GPU 的部署可能共享一台主机或一个故障域，因此主机故障容忍度受实际部署位置限制；副本/备份可以改善元数据恢复，但不会虚构不存在的 GPU 冗余。

## 4. 外部 API

所有 endpoint 都要求 `Authorization: Bearer <token>`，并强制执行 tenant ownership。修改类 endpoint 要求 `Idempotency-Key`；其记录范围为 `(owner_id, route, key)`，其中 `route` 包含 HTTP method 和具体资源路径，请求 hash 覆盖该目标及规范化 body。使用相同 key 的完全重放返回原始 status/body；相同 key 搭配不同 body 返回 `409 IDEMPOTENCY_KEY_REUSED`。

### Dataset API

`POST /v1/datasets` 只接受 manifest；禁止任意外部 URL。

```json
{
  "files": [
    {"name":"001.jpg","size_bytes":1830421,"sha256":"d2d2d2d2d2d2d2d2d2d2d2d2d2d2d2d2d2d2d2d2d2d2d2d2d2d2d2d2d2d2d2d2","mime_type":"image/jpeg","caption":"a red bicycle"}
  ]
}
```

`files` 包含 100–1,000 个不重复名称；每个 `name` 是相对对象 key，`size_bytes` 是服务限制内的正整数，`sha256` 是小写 64 位十六进制字符串，`mime_type` 必须是允许的图像 MIME 类型。可选 `caption` 是满足配置长度/安全限制的 UTF-8 纯文本；它会被验证、规范化并绑定到冻结的文件快照，不会被当作可执行 prompt 语法。上述小示例仅用于说明；有效请求至少包含 100 项。`201` 响应返回 UUID dataset ID、对象 key 和短期预签名 PUT URL。例如，`dataset_id` 是类似 `3fa85f64-5717-4562-b3fc-2c963f66afa6` 的字符串 UUID。

`POST /v1/datasets/{dataset_id}/complete` 的 body 为 `{}`。在其幂等事务中，服务冻结已上传的对象版本，将 `UPLOADING` 改为 `VERIFYING`，并插入持久化 verify task；返回带 `Location: /v1/datasets/{dataset_id}` 的 `202`。Dataset 状态严格为 `UPLOADING`、`VERIFYING`、`COMPLETED` 和 `INVALID`。`GET /v1/datasets/{dataset_id}` 是客户端轮询 API，返回 ID、状态、不可变 manifest version/checksum、已知时的接受/拒绝数量、时间戳，并在 `INVALID` 时返回可操作的 `verification_error`。只有 `COMPLETED` dataset 才能创建 job。

`GET /v1/training-profiles` 只列出与调用者方案兼容的已发布 profile。每项返回不可变 UUID `profile_revision_id`（例如 `8ac7a2f1-694c-4da4-8d6a-e40dc9b0f2fd`）、人类可读的 `profile_key`（如 `style-lora`）、revision number、display name、固定的基础模型版本、训练/环境版本、支持的输入边界、是否启用 captioning，以及 quality-policy readiness 标志；不会暴露可变的底层 override。Quality policy 使用相同模式：UUID `quality_policy_revision_id`，以及稳定的 policy key 和 revision number。

### Job API

`POST /v1/training-jobs`:

```json
{
  "dataset_id":"3fa85f64-5717-4562-b3fc-2c963f66afa6",
  "profile_revision_id":"8ac7a2f1-694c-4da4-8d6a-e40dc9b0f2fd",
  "trigger_token":"mystyle"
}
```

`dataset_id` 必须是调用者拥有且已完成的 dataset；`profile_revision_id` 必须已发布；`trigger_token` 是必填的非空、且对 profile 有效的风格 token。成功响应为 `202 Accepted`：

```json
{
  "job_id":"c0a8012e-7b6d-4b7d-a48f-8a10b2b5c0ea",
  "state":"ACCEPTED",
  "current_stage":"PREPARE",
  "status_url":"/v1/training-jobs/c0a8012e-7b6d-4b7d-a48f-8a10b2b5c0ea"
}
```

响应包含 `Location: /v1/training-jobs/{job_id}`。`GET /v1/training-jobs/{job_id}` 返回上述字段，以及 `candidate_index`、阶段 task 状态/attempt 数、提交/更新时间戳、非敏感进度、错误摘要、`READY` 时的 model ID，以及排队时的 `retry_after_seconds`。轮询结果示例：

```json
{
  "job_id":"c0a8012e-7b6d-4b7d-a48f-8a10b2b5c0ea",
  "state":"RUNNING",
  "current_stage":"TRAIN",
  "candidate_index":0,
  "stages":{"PREPARE":"SUCCEEDED","CAPTION":"SUCCEEDED","TRAIN":"RUNNING"}
}
```

`POST /v1/training-jobs/{job_id}/cancel` 接受 `{}`。相同幂等 key 会重放原始的 `202 CANCEL_REQUESTED` 响应。终态为 `CANCELLED` 后使用新 key 会返回包含既有终态结果的 `200`；如果发布已经成功则返回 `409 JOB_ALREADY_READY`，其他终态结果返回 `409 JOB_TERMINAL`。在报告存在前，`GET /v1/training-jobs/{job_id}/evaluation` 返回 `409 EVALUATION_NOT_READY`；之后返回带版本的报告，包括 policy revision ID、candidate index/profile revision ID、adapter checksum、prompt-suite/reference 制品、每个门禁的分数和 verdict、不确定性标志、失败原因以及报告制品引用。对于机密内容，绝不会泄露其他 tenant 的图像或 prompt。下面的示例指标值仅是诊断输出，不是通用阈值：

```json
{
  "policy_revision_id":"9b772267-96f5-4c87-97c5-48a8dbb426c1",
  "candidate_index":0,
  "profile_revision_id":"8ac7a2f1-694c-4da4-8d6a-e40dc9b0f2fd",
  "prepared_version_id":"2b81041a-7e12-4dc4-b92c-87a012867c32",
  "training_input_version_id":"2a91b5e1-5ee7-45c8-bb72-99b54aa91144",
  "training_input_manifest_sha256":"cdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcd",
  "adapter_sha256":"abababababababababababababababababababababababababababababababab",
  "prompt_suite_artifact":"s3://private/eval/prompt-suite-v7.json",
  "heldout_reference_artifact":"s3://private/eval/heldout-v4.json",
  "gates":{"style":{"value":0.71,"verdict":"PASS"},"text_adherence":{"value":0.83,"verdict":"PASS"},"diversity_memorization":{"value":0.18,"verdict":"PASS"},"technical_smoke":{"verdict":"PASS"}},
  "overall_verdict":"PASS",
  "report_artifact":"s3://private/reports/c0a8012e/candidate-0.json"
}
```

### Model API 与错误

`GET /v1/models/{model_id}` 返回调用者拥有的模型的 `READY` 状态、profile/基础模型兼容性、adapter checksum、manifest URI、创建时间和评估报告摘要。`GET /v1/models/{model_id}/download` 只授权 `READY` 模型，并返回不可变 manifest 和 adapter 的短期下载授权；其他情况返回 `409 MODEL_NOT_READY`。

所有错误使用以下格式：

```json
{
  "error":{"code":"DATASET_TOO_SMALL","message":"Fewer than 100 valid unique images remain after validation.","retryable":false,"details":{"valid_unique_images":83,"minimum":100},"request_id":"7c2d1cb8-8ba3-4fa0-9e80-8e2c5db281d0"}
}
```

格式错误请求使用 `400`，缺失/无效凭据使用 `401`，无权访问的资源使用 `404`，状态/幂等性/profile 冲突使用 `409`，有效但不可用的数据集或配置使用 `422`，接纳上限使用 `429`，临时控制面不可用使用 `503`。有意义的错误码包括 `PROFILE_NOT_READY`、`EVALUATION_NOT_READY`、`UNSUPPORTED_PROFILE`、`DATASET_INVALID`、`DATASET_UNSUITABLE`、`GPU_BUDGET_EXCEEDED`、`QUALITY_REJECTED` 和 `JOB_ALREADY_READY`。

## 5. 持久化数据模型与运维要求

该 schema 使用 `uuid` 资源主键（PK）、明确列出的复合/单例控制 key、UTC 时间的 `timestamptz`、计数/字节/秒数的 `bigint`、URI/枚举的 `text`、不可变结构化配置的 `jsonb`，以及基于 DB 时钟的 lease 比较。每个 SHA-256 字段，包括下方简写的 `/sha256` 字段，都是 `char(64)`，并验证为小写 64 位十六进制。除非标记为 nullable，字段均为 `NOT NULL`；tenant 拥有且可外部寻址的表包含 `owner_id uuid NOT NULL FK tenant(id)`。全局 profile 和 policy revision 由 operator 所有，因此不带该字段。Job、stage、task 和 device 的状态字段受第 2–3 节对应枚举约束。

| 表 | 类型化字段、key 与重要约束 |
| --- | --- |
| `dataset` | `id uuid PK`、`owner_id uuid FK`、`state text CHECK (UPLOADING, VERIFYING, COMPLETED, INVALID)`、`submitted_manifest_sha256 char(64)`、`frozen_manifest_uri text NULL`、`frozen_manifest_sha256 char(64) NULL`、`declared_count int CHECK (100..1000)`、`valid_count int NULL`、`invalid_count int NULL`、`verification_error jsonb NULL`、`created_at timestamptz`、`completed_at timestamptz NULL`；UPLOADING/VERIFYING 时 `completed_at IS NULL`；completed/invalid 要求该字段，completed 还要求 frozen manifest 字段。 |
| `dataset_file` / `dataset_verify_task` | File：`id uuid PK`、`dataset_id uuid FK dataset(id)`、`name text`、完成前为 `object_version text NULL`、`size_bytes bigint CHECK (>0)`、`sha256 char(64) CHECK (sha256 ~ '^[0-9a-f]{64}$')`、`mime_type text`、`caption text NULL`、`caption_sha256 char(64) NULL CHECK (caption_sha256 ~ '^[0-9a-f]{64}$')`、`verification_status text`、`rejection_code text NULL`；`UNIQUE(dataset_id,name)`，且 caption 存在当且仅当 caption checksum 必须存在。Verify task：`id uuid PK`、`dataset_id uuid NOT NULL UNIQUE FK`、`state text`、`attempt_count smallint`、`ready_at/lease_expires_at timestamptz`、`fencing_token bigint NULL`；完成操作冻结对象版本时插入。 |
| `prepared_dataset_version` | `id uuid PK`、`job_id uuid NOT NULL UNIQUE FK training_job(id)`、`source_dataset_id uuid FK dataset(id)`、`cleaned_manifest_uri/sha256 text`、`train_manifest_uri/sha256 text`、`heldout_manifest_uri/sha256 text`、`preprocessor_version text`、`split_seed bigint`、`grouping_version text`、`producing_attempt_id uuid FK task_attempt(id)`、`created_at timestamptz`；所有 manifest 字段和 producing attempt 均非空且不可变。 |
| `training_input_version` | `id uuid PK`、`prepared_version_id uuid NOT NULL FK prepared_dataset_version(id)`、`caption_manifest_uri text NULL`、`caption_manifest_sha256 char(64) NOT NULL`（不存在 caption 制品时使用规范化空 caption manifest 的 digest）、`captioner_version text NULL`、`training_manifest_uri text`、`training_manifest_sha256 char(64)`、`trigger_token_sha256 char(64)`、`producing_attempt_id uuid FK task_attempt(id)`、`created_at timestamptz`；`UNIQUE(prepared_version_id, caption_manifest_sha256, trigger_token_sha256)`，任何字段都不能修改 prepared held-out manifest。 |
| `training_profile_revision` | `id uuid PK`、`profile_key text`、`revision int`、`status text`、`config jsonb`、`base_revision text`、`container_digest text`、`quality_policy_revision_id uuid FK quality_policy_revision(id)`、`gpu_class text`、`gpu_budget_seconds bigint CHECK (>0)`；`UNIQUE(profile_key,revision)`，一旦 `status=PUBLISHED` 即不可变。其 `config` 保存已验证的 OOM fallback 和 candidate-1 variant ID。 |
| `quality_policy_revision` | `id uuid PK`、`policy_key text`、`revision int`、`status text`、`definition jsonb`、`calibration_artifact_uri text NULL`、`created_at timestamptz`；`UNIQUE(policy_key,revision)`，`PUBLISHED` 要求存在 calibration artifact。 |
| `training_job` | `id uuid PK`、`owner_id uuid FK`、`dataset_id uuid FK`、`profile_revision_id uuid FK`、`quality_policy_revision_id uuid FK`、`prepared_version_id uuid NULL UNIQUE FK`、`training_input_version_id uuid NULL FK`、`state text`、`current_stage text NULL`、`candidate_index smallint CHECK (0..1)`、`trigger_token text`、`gpu_seconds_charged/reserved bigint CHECK (>=0)`、`cancel_requested_at timestamptz NULL`、`error_code text NULL`、`created_at/updated_at timestamptz`；`UNIQUE(owner_id,id)`。 |
| `job_candidate` | `job_id uuid FK`、`candidate_index smallint CHECK (0..1)`、`profile_revision_id uuid FK training_profile_revision(id)`、`training_input_version_id uuid NULL FK`、`adapter_sha256 char(64) NULL`、`checkpoint_compatibility_key text NULL`、`outcome text NULL`；复合 `PK(job_id,candidate_index)` 和 `UNIQUE(job_id,candidate_index,training_input_version_id)`。Input/compatibility 仅在 TRAIN 接纳前可以为空，绑定后不可变。Candidate 1 必须引用配置的不可变 variant 和同一个 quality-policy revision。 |
| `stage_task` / `task_attempt` | Task：`id uuid PK`、`job_id uuid FK`、`candidate_index smallint CHECK (0..1)`、`stage text`、`state text`、`ready_at timestamptz`、`attempt_count smallint CHECK (0..3)`、`active_token bigint NULL`、`lease_expires_at timestamptz NULL`、`deadline_at timestamptz NULL`、`checkpoint_uri text NULL`；`UNIQUE(job_id,candidate_index,stage)`，以及复合 `FK(job_id,candidate_index) REFERENCES job_candidate`。Attempt：`id uuid PK`、`task_id uuid FK`、`attempt_no smallint CHECK (1..3)`、`gpu_slot_id uuid NULL FK`、`fencing_token bigint`、`scheduler_epoch bigint`、`started_at/last_heartbeat_at/lease_expires_at/ended_at timestamptz NULL`、`charged_gpu_seconds bigint`、制品/日志 URI 和 `termination_evidence jsonb NULL`（来源、准确 host/attempt、观测时间、证据引用、记录 actor）；`UNIQUE(task_id,attempt_no)` 和 `UNIQUE(task_id,fencing_token)`。 |
| `gpu_slot` / `model_version` / `evaluation_report` | Slot：`id uuid PK`、`host_id/device_id text`、`capabilities jsonb`、`health text`、`current_attempt_id uuid NULL FK`、`quarantine_reason text NULL`；`UNIQUE(host_id,device_id)`。Model：`id uuid PK`、`job_id uuid NOT NULL UNIQUE FK`、`evaluation_report_id uuid NOT NULL UNIQUE FK evaluation_report(id)`、`adapter_sha256 char(64)`、`manifest_uri text`、`manifest_sha256 char(64)`、`state text`、`ready_at timestamptz NULL`。Report：`id uuid PK`、`job_id uuid`、`candidate_index smallint CHECK (0..1)`、`policy_revision_id uuid FK`、`prepared_version_id uuid FK`、`training_input_version_id uuid FK`、`training_input_manifest_sha256 char(64)`、`adapter_sha256 char(64)`、`full_pass boolean`、`smoke_pass boolean`、`report_uri text`、`report_sha256 char(64)`；复合 `FK(job_id,candidate_index,training_input_version_id) REFERENCES job_candidate` 和 `UNIQUE(job_id,candidate_index,adapter_sha256,policy_revision_id)`。 |
| `idempotency_request` / `scheduler_guard` | Idempotency：`owner_id uuid`、`route text`、`key text`、`request_sha256 char(64)`、`response_status int`、`response_body jsonb`、`expires_at timestamptz`，复合 `PK(owner_id,route,key)`。Guard：单例 `id smallint PK CHECK(id=1)`、`leader_epoch bigint`、`stage_cursor smallint CHECK (0..3)`（四项循环中的位置，用于区分其中两项 TRAIN）、按 `EVALUATE`、`TRAIN`、`CAPTION` 建 key 的 `owner_cursor_by_class jsonb`、`updated_at timestamptz`。 |

索引包括 `stage_task(state,ready_at)`、`training_job(owner_id,created_at)` 和 `task_attempt(gpu_slot_id,lease_expires_at)`。外键和受保护的状态转换谓词确保 READY 模型引用已完成的完整评估，且 `full_pass=true`、`smoke_pass=true`，并匹配准确的 adapter/training-input checksum；活跃且未过期的 attempt 拥有 task 更新权限；每个 job 只能存在一个模型。队列和配额检查与 job 插入在同一事务中完成。lease 续期及任何可变 attempt 转换都验证 fencing token 和 DB 时钟过期时间。

安全控制包括限定 tenant、短期有效的预签名上传/下载；传输中和静态数据加密；最小权限 worker 凭据；审计日志；速率限制；以及在受限 worker 中解码图像。服务拒绝压缩包、任意代码和不受信任的序列化 checkpoint/pickle 输入。服务限制文件数量、压缩和解码后的像素尺寸、总字节数及解码工作量，以降低解压炸弹风险；具体限制由 profile/service 配置决定，此处不硬编码。

运维默认保留原始上传 30 天、未引用的 checkpoint/attempt 制品 7 天，并保留 READY 模型制品直到 owner 删除；垃圾回收绝不能删除被非终态 job、未确认 executor 或 READY 模型引用的数据。即使已记录终态技术失败，含未确认 executor 的 job 仍保留接纳配额占用和受保护引用。指标包括按 stage/owner 统计的队列等待、接纳拒绝、GPU 槽位利用率和隔离情况、scheduler 公平性/年龄、OOM、重试、heartbeat/lease 过期、取消延迟、对象存储/数据库错误、每种结果消耗的 GPU 秒数，以及质量通过/失败维度。结构化日志关联 request、job、task 和 attempt ID，但不存储图像 payload。

建议的服务目标是待测量的目标，而非既定事实：持久化的已接纳 job 能在 controller failover 后存活；所有 READY 模型都具备通过的 policy report 和 smoke test；并且在队列年龄、GPU 预算超支或错误率目标影响用户体验前触发告警。PostgreSQL 时间点恢复、必需的上传对象版本控制、可选的存储复制、加密备份、恢复演练和 controller standby 用于保证可恢复性。但它们无法消除小型 GPU 资源池的单主机故障域。

## 6. 验收场景

| 场景 | 预期结果 |
| --- | --- |
| 数据集数量边界 | 创建时拒绝 99 文件和 1,001 文件的 manifest；100 和 1,000 文件可进入异步验证，但仍需通过后续有效性/适用性检查。 |
| 校准 profile 下已知通过的 240 图像测试夹具 | Job 经过所有阶段；PREPARE 绑定 manifest 链，完整评估和加载 smoke test 通过，并为 owner 生成且只生成一个 checksum 固定的 READY 模型。任意有效数据集仍可能质量失败。 |
| 130 次上传，清理后有 35 个损坏/重复项 | PREPARE 统计数量并失败为 `DATASET_TOO_SMALL`；不启动 GPU 阶段。 |
| 独立分组或留出覆盖不足 | PREPARE 失败为 `DATASET_UNSUITABLE`；名义上有效的数量不能绕过泄漏/适用性门禁。 |
| 两张 GPU 上有四个并发且符合条件的 job | 最多执行两个阶段；每个阶段使用一个兼容的独占槽位；scheduler 在 owner 之间 round-robin，评估获得加权调度机会，且没有永久预留 GPU。 |
| 评估排在持续训练之后 | 加权循环和队列年龄告警使符合条件的评估在有限积压下获得调度机会；已借用的运行阶段不会被抢占。 |
| 受支持 profile 发生 OOM | 预算足够时尝试已记录且预先验证的 fallback；不支持的 fallback/configuration 直接失败，不临时改用较低质量的运行配置。 |
| GPU worker 崩溃后旧 worker 重新连接 | lease 过期，旧 fencing token 不能续租、绑定 manifest、提交检查点引用或发布；设备保持隔离直到确认终止/健康；重试只使用在 attempt/预算限制内已提交且兼容的检查点。 |
| 并发 controller 分发工作或发生接管 | 单例 guard 串行化申领/预算/cursor 变更，接管后拒绝旧 leader epoch；任何 slot/task 都不会收到两个有效授权。 |
| 发布期间发生部分写入 | 没有受保护 READY 事务的制品保持不可公开，之后会清理；重试/reconciliation 为 job 创建或找到唯一模型。 |
| 质量筛选或完整评估失败 | 没有模型进入 READY。最多运行一次不可变的 candidate-1 补救；第二次补救被拒绝，最终失败为 `QUALITY_REJECTED` 并附带带版本的证据。 |
| 重试加评估超出 GPU 预算 | 每次计费的 attempt、评估和 smoke 阶段都计入；watchdog 将 job 结束为 `GPU_BUDGET_EXCEEDED`。 |
| 运行中的 attempt 遇到数据库故障 | 不启动新工作也不发布；agent 在 lease 过期前停止；数据库恢复后 reconciliation 安全恢复。 |
| 客户端重复创建请求 | 相同幂等 key 和 body 返回原 job；body 改变则返回 `409`；配额不会被重复消耗。 |
| 取消与发布发生竞争 | 受保护事务最终得到 READY 模型或持久化取消结果之一；READY 之后的延迟取消返回 `409`。 |
| 已取消 job 的 executor 永久无法访问 | 在有效 agent 确认或经过审计的 instance/BMC 终止证明到达前，job 保持 `CANCEL_REQUESTED`，继续保护配额/制品并触发告警；随后变为 `CANCELLED`。 |
| 跨 tenant 访问 dataset/model | 授权返回 `404`，不提供上传/下载授权，也不泄露 dataset、report 或 model 元数据。 |

这些场景直接对应评估标准：manifest 和评估链体现 ML pipeline architecture；lease、epoch、事务队列和 reconciliation 体现分布式系统正确性；配额、隔离、保留策略、可观测性和备份约束体现生产就绪度；共享资源池的加权 scheduler、经过校准的多门禁 quality policy 以及保守的发布协议，则在不声称未经测量的性能前提下，处理小型 GPU 资源约束。
