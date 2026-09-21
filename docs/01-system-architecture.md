# 交付物 1——系统架构

## 1. 设计摘要与边界

一个异步、质量门控的流水线将用户的图像集合转换为可供推理消费者加载的、带版本的风格 LoRA。稀缺资源是**共享的 2–4 张 GPU 资源池**，因此 CPU 准备、持久化编排和 GPU 执行分别承担不同职责。训练、GPU 图像描述生成和评估共同竞争同一个明确管理的资源池。

本文是系统设计提交物，描述可部署的架构和契约；不声称已经实现服务、训练出模型、测得吞吐量或校准过质量阈值。配套的[技术规格](02-technical-specification.md)定义组件、API、数据模式、故障处理和验收场景。

| 要求 | 设计边界 / 假设 |
|---|---|
| 输入 | 针对一个目标视觉风格的 100–1,000 张图像；清理后、训练/验证划分前，至少保留 100 张有效且不重复的图像。 |
| 输出 | 通过审批的适配器权重、不可变的基础模型版本、加载配置、制品哈希、评估报告，以及成功的推理加载冒烟测试。LoRA 不是独立的基础模型。 |
| 硬件 | 2–4 个已注册的 GPU 设备，可位于一台或多台主机上。只有经过单 GPU 配置验证的任务才可进入队列；每种支持的配置都必须实测显存和运行时可行性。 |
| 模型中立性 | 带版本的训练 profile 封装特定模型的训练、预处理、加载和资源需求。更换模型系列不改变编排契约。 |
| 并发 | 可以接收并排队多个请求；每个 GPU 最多运行一个阶段，并且每个任务最多有一个当前获授权的 GPU attempt。失联的 attempt 在被终止前，可能与另一设备上的恢复 attempt 在物理上重叠；其资源占用仍计入账本。系统设置有界接纳。 |
| 部署 | 注册可供推理的模型并交给现有推理消费者。常驻服务集群及其 SLA 不在本次训练资源预算范围内。 |
| 质量 | 只有满足已启用且经过校准的评估 policy 的候选模型才能进入 `READY`。系统不保证任意数据集都能产生合格模型。 |

## 2. 架构图

![Automatic LoRA training pipeline architecture](../diagrams/system-architecture.png)

[Scalable SVG](../diagrams/system-architecture.svg) · [Editable PlantUML source](../diagrams/system-architecture.puml)

API、scheduler、registry 和发布逻辑都是**逻辑组件**，不要求每个方框都部署成独立微服务。PostgreSQL 同时作为元数据权威存储和持久化 stage-task 队列。registry 由指向不可变对象存储制品的数据库记录组成，不是第二套元数据系统。

### 编号数据流

1. **上传与认证。** API 创建属于指定 owner 的 dataset 上传会话，并返回短期上传 URL。图像字节直接写入带版本的对象存储。完成上传后冻结对象版本并加入验证队列；当 dataset 处于 `VERIFYING` 时返回 `202`。只有 `COMPLETED` dataset 才能进入训练。
2. **原子接纳。** training request 引用已完成的 dataset 和启用的 training profile。数据库事务检查配额和幂等性，创建 job 并插入第一个 `PREPARE` task。客户端立即收到 job ID 和状态 URL。
3. **准备与调度。** CPU worker 检查文件、移除重复项，在约 90/10 划分前整理近重复组，并生成不可变的 prepared-data manifest。profile 指定最少训练图像数及留出图像/分组数；无法完成组互斥划分时，在训练前失败。缺少图像描述时，可选的 GPU `CAPTION` task 生成最终 training-input manifest，但不改变划分。每个阶段在当前 lease/token 下，通过一次原子操作绑定输出 manifest、完成当前 task 并创建唯一的后继 task。scheduler 按资源兼容性、阶段类别轮转和用户公平性选择可执行 task。
4. **在 lease 内执行。** host agent 在独占申领的 GPU 上启动隔离且固定版本的阶段容器。attempt 获得 fencing token、可续租的 lease 和硬截止时间。除了选定的 LoRA 参数外，模型权重保持冻结。进度和重试计数持久化。
5. **检查点与评估。** worker 写入不可变检查点和候选制品。`EVALUATE` 获得独立的 GPU 分配，运行固定的验证 prompt 套件，计算 quality policy 的各个维度，并重新加载准确保存的候选模型，以验证推理兼容性。最多允许一次、预算受限的质量补救训练。
6. **发布已验证版本。** CPU publisher 检查制品和评估哈希、当前 attempt 以及取消状态。一个事务将模型注册为 `READY`，将 job 标记为 `READY`，并完成发布。未通过的候选模型始终不可下载。
7. **交接给推理。** 已授权的消费者获取带版本的 adapter 和加载 manifest，其中包含兼容的基础模型版本。下载权限是短期且限定 tenant 的。发布可以重试，但不会产生重复版本。

## 3. 工作流与质量边界

![Job lifecycle with quality gates and bounded remediation](../diagrams/job-lifecycle.png)

[SVG](../diagrams/job-lifecycle.svg) · [PlantUML](../diagrams/job-lifecycle.puml)

job 生命周期和 stage 执行是两个不同维度。job 可以处于 `RUNNING`，其 `current_stage=EVALUATE`，而该 stage 仍在等待 GPU、处于 `PENDING`。API 同时暴露这两个维度，让用户区分排队和执行。

训练 loss 可以检测发散，但不能证明产品质量。发布门禁检查风格一致性、prompt 遵循度、多样性、对训练图像的过度复现以及技术可加载性。训练、检查点、评估和发布引用同一个不可变 prepared-data 版本及 training-input manifest 链；任何消费者都不会跟随未版本化的 “latest” 路径。评估引用使用留出数据；prompt 改变主题，以暴露内容泄漏。quality-policy 版本固定评估模型、prompt、seed、预处理和校准后的阈值。重试时不会静默降低阈值。

基于 CLIP 的图像相似度和图文相似度是有用的辅助指标，但图像相似度可能混淆风格和内容。建议的 policy 将经过校准的风格导向评估器与其他独立维度及线下人工判断结合起来。[StyleDrop 的评估](https://research.google/blog/styledrop-text-to-image-generation-in-any-style/)展示了图像/文本相似度指标和人工偏好评估；该文报告的分数不会直接移植为本系统的阈值。

## 4. 资源约束与设计取舍

| 决策 | 原因与限制 |
|---|---|
| PostgreSQL-backed stage queue | job 创建和 task 插入共用一个事务，避免数据库与 broker 双写不一致。当前规模不足以证明需要额外的队列服务。短暂的申领事务可以使用行锁；[PostgreSQL 文档说明了 `SKIP LOCKED` 在队列型消费者中的用法](https://www.postgresql.org/docs/current/sql-select.html)。轮询仍是权威机制。 |
| Single-GPU stage isolation | 资源归属可预测且故障域更小；不同 job 不会同时占用同一设备的显存。无法放入受支持单卡的 profile 会被拒绝。多 GPU job 是未来的调度扩展。 |
| Shared evaluation capacity | 不永久预留 GPU 做评估。循环阶段 `EVALUATE → TRAIN → CAPTION → TRAIN` 会跳过空的或不兼容的类别。每个类别内采用用户 round-robin 和 FIFO，避免某个用户的积压主导分发。运行中的任务不会被抢占。 |
| Work-conserving fairness | 分配新任务时，若其他用户有可执行工作，则每个用户最多一个活跃 GPU task；没有其他用户的可执行工作时，用户可借用空闲槽位。另一个用户到达时，已借用的工作不会被抢占，并在其阶段硬超时内完成。公平性表示获得调度机会，不承诺相同 GPU 秒数或立即启动。 |
| Bounded effort | 默认接纳上限：每个用户 5 个非终态 job，全局 100 个。每个 profile 明确累计 GPU 预算、阶段截止时间和评估预留。重试和补救也计入预算。 |
| Reuse and early rejection | 在训练前拒绝不合适的数据；缓存冻结基础模型制品和兼容的冻结预处理输出；在完整评估前筛除明显失败的候选。tenant 派生缓存相互隔离。筛选本身永远不能授权发布。 |

做一个粗略容量模型：令 `G` 为可用 GPU 槽位数量，`T` 为每个完成 job 的**总 GPU 小时**，包括 caption、评估、预热和重试。在利用率 `u` 下，吞吐量约为 `u × G / T` 个 job/小时。这是规划公式而不是基准测试；异构设备和失败 job 需要按 profile 分别测量。增加 API 副本不会提升 GPU 吞吐量。队列等待时间应与执行时间分开报告。

## 5. 可扩展性与容错

![Leases, device quarantine, recovery and stale-result rejection](../diagrams/lease-recovery.png)

[SVG](../diagrams/lease-recovery.svg) · [PlantUML](../diagrams/lease-recovery.puml)

**执行可以重复；发布必须唯一。** worker 可能在写入检查点或制品后失去 lease。以 attempt 为范围的对象路径、单调递增的 fencing token、受保护的状态转换，以及每个 job 唯一的已发布模型记录，可防止过期执行修改规范结果。对象存储制品在数据库事务暴露不可变 manifest 前，必须先写入并验证。

**lease 丢失不等于 GPU 已空闲。** 无法访问的主机会被隔离。只有进程终止且设备健康检查通过后，设备才会回到资源池。另一台健康设备可以在剩余的保守预算内重试。当前只有一个 attempt 获得提交授权，但未确认的旧进程可能在另一台设备恢复时发生物理重叠；两者都计入资源暴露。host watchdog 强制执行 lease 和执行截止时间；scheduler 不会在未确认的设备上启动第二个 task。取消会保持 `CANCEL_REQUESTED`，直到 agent 确认终止，或者对于不可恢复的主机，由经过审计的 control-plane/BMC 关机证据确认。永久退役的设备不会重新加入资源池。

**控制面故障会暂停安全进展。** 无状态 API 可以运行多个副本。scheduler 使用 active/standby leadership；每个短 dispatch 事务锁定持久化的 scheduler guard，并在检查 owner eligibility、申领 task/device、预留预算和推进 cursor 前验证 leader epoch。这可以防止并发或过期 controller 通过分开申领任务绕过用户公平性。元数据存储不可用时，新的启动和发布会停止；当 lease 续期不再安全时，运行中的 worker 也会停止。已完成对象和已提交的检查点在 worker 重启后仍保留。受管数据库的 failover/PITR 和对象存储持久性保护已提交状态，但恢复仍取决于已配置的备份保证。

**有意识地扩展瓶颈。** CPU worker 可以独立扩展；GPU host 注册更多经过 profile 的槽位，无需修改客户端 API。队列量更大时，可以由 broker 发送唤醒提示；若引入持久化外部事件，则配合 transactional outbox。数据库仍是状态权威。持续服务、多 GPU gang scheduling 和大规模超参数搜索需要单独的容量决策。

如果所有 GPU 位于同一主机，主机故障会移除全部训练能力，直到恢复。将可用 GPU 分布在多台主机上可以减小故障域；2–4 张 GPU 的预算本身不代表硬件冗余。应观测并告警积压、lease 丢失、GPU 利用率、OOM、预算消耗和质量拒绝。

## 6. 对评估标准的覆盖

| 评估标准 | 提交物中的证据 |
|---|---|
| ML pipeline architecture | 带版本的数据集和 profile；划分前去重；可选的面向内容的 caption；仅训练 LoRA；完整检查点；留出的多维评估；精确制品的推理冒烟测试。 |
| Distributed systems | 事务化接纳和阶段推进；持久化 task queue；lease 与 fencing；设备隔离；有界重试；幂等 API 和 effectively-once 模型发布；取消竞争处理。 |
| Production requirements | tenant 隔离；受范围限制的直接上传；schema 约束；接纳上限；审计和指标；备份与保留策略；可复现性；明确错误和运维验收场景。 |
| Creative use of limited resources | 共享训练/评估能力；公平且 work-conserving 的调度；有界补救；兼容缓存；提前拒绝数据并筛选评估；按 GPU 时间计费，而不是允许无限重试。 |

技术规格通过 API 示例、schema 契约和基于场景的验收标准，使这些机制具体化。图表仍可编辑，并可按照[提交指南](../README.md)中的说明在本地复现。
