# 交付物 1——系统架构

## 范围与实现边界

本实现把 owner 负责的 100–1,000 张图像转换为 SD 1.5 LoRA adapter、加载 manifest 和评估报告。它运行在单台 Linux/WSL2 主机上，使用 FastAPI、SQLite、本地持久化文件和独立 worker supervisor；一张物理 GPU 同时最多执行一个受管理阶段。默认配置使用一张设备，也可管理同一主机上的多张 GPU。CPU 测试使用显式的 tiny Diffusers/PEFT 图（随机初始化、离线运行）和 fake slots；它不伪造 CUDA 设备，也不能代表真实 SD 1.5 质量。

Part 1 曾提出 PostgreSQL、版本化对象存储和多主机 agent；Part 2 采用较小的实现边界。本文与三个图都描述当前边界；[实现取舍](03-implementation-tradeoffs.md)记录迁移影响。实现不声称具备分布式故障转移、经过校准的风格质量或已测得的 GPU 吞吐量。

## 架构与数据流

![单主机架构](../diagrams/system-architecture.png)

[SVG](../diagrams/system-architecture.svg) · [PlantUML 源文件](../diagrams/system-architecture.puml)

1. API 认证配置的 bearer API key，创建上传 manifest 并分配内部 file ID。客户端通过认证端点逐个上传文件；文件名只是元数据，不是路径。
2. 完成上传后冻结已记录对象并排入验证。job 接纳要求 dataset 已完成，并在同一事务中创建 job 和首个 stage task，同时检查幂等性及 owner/全局配额。
3. CPU 准备阶段验证解码格式和限制，规范化图像、移除完全重复项、聚合近重复项，并冻结约 90/10 的按组互斥划分。caption 优先使用用户文本，其次使用 template 或 BLIP。
4. 单例 supervisor 分发 stage 子进程。可选 BLIP、LoRA 训练和评估共享有限的物理 GPU 池；以 GPU UUID 为键的 OS 锁阻止阶段重叠。
5. 训练冻结基础参数，只优化 LoRA 参数并写入完整可恢复检查点。评估重新加载准确保存的 adapter，在相同条件下生成冻结基础模型/LoRA 成对图像套件并收集质量诊断。
6. 发布检查制品引用、checksum 和 attempt 所有权，在事务中注册每个 job 唯一的 model 并提交终态。没有校准证据的技术成功是 `COMPLETED_UNVERIFIED`，不是 `READY`。
7. owner 获取 manifest 和可供兼容推理消费者使用的 adapter。未验证模型必须显式同意下载；在线推理服务不在本作业范围内。

SQLite 保存元数据、队列、进度和逻辑 registry；图像和权重不进入数据库行。文件写入完成后引用才可见；事务短小且不包含模型执行。

## 生命周期与质量边界

![Job 生命周期](../diagrams/job-lifecycle.png)

[SVG](../diagrams/job-lifecycle.svg) · [PlantUML 源文件](../diagrams/job-lifecycle.puml)

Job 状态与 stage-task 状态分离：`RUNNING` job 可能有仍在 `PENDING` 的 `EVALUATE` task。有效检查点不等于完成的 adapter，可加载也不等于风格质量合格。

默认评估 policy 未校准。报告包含技术检查、CLIP 图文对齐、留出图像相似度、输出多样性和训练图像相似度；相似度只是诊断代理，不是独立风格评估器，也不能证明记忆。只有配置了带校准证据的版本化 policy 且所有门禁通过，才可授权 `READY`；test-only 制品永远不能 READY。

A/B 生成固定 prompts、seeds、scheduler、推理步数和 guidance。默认 comparison 是冻结基础模型与其 LoRA；CLI 也支持兼容 adapter。合成数据 smoke 只证明执行正确性，不证明产品质量。

## 资源管理与恢复

![本地恢复](../diagrams/lease-recovery.png)

[SVG](../diagrams/lease-recovery.svg) · [PlantUML 源文件](../diagrams/lease-recovery.puml)

GPU 调度使用 `EVALUATE → TRAIN → CAPTION → TRAIN` 权重循环、类别内 owner 轮转和 owner FIFO；空类别跳过。没有其他 owner 的可执行工作时可借用空闲容量，但不抢占运行中的 stage。CPU 使用独立的有限池。默认接纳上限是每个 owner 五个非终态 job、全局 100 个。

每个 stage 有 attempt token、heartbeat、deadline 和进程身份。lease 过期不会立即释放 GPU，必须确认终止和锁可用后才能复用。旧 attempt 的迟到输出不能改变规范状态。可恢复基础设施故障最多重试三次；OOM、不适用输入和质量失败显式结束，不会静默改超参数或自动质量重训。

API 和 worker 重启可在同一主机恢复持久状态；主机或磁盘丢失属于共同故障域。备份必须在停止服务时一致地包含数据库及引用文件。多主机可用性需要替换持久化和远程执行协调，不能通过把 SQLite 复制到网络文件系统获得。

## 评估标准覆盖

| 标准 | 当前实现证据 |
|---|---|
| ML 架构 | 冻结输入 manifest、按组划分、仅训练 LoRA、完整 resume、adapter 加载和配对评估。 |
| 代码架构 | `data`、`captions`、`training`、`evaluation`、`store`、`worker`、`api` 分离，含 CLI 和自动化测试。 |
| 生产考虑 | owner 隔离、流式上传、幂等、持久 task、取消、设备锁、过期 attempt 拒绝、health 和 metrics。 |
| 资源利用 | 单设备单 stage、共享评估容量、公平调度、明确预算、输入提前拒绝和可选 CPU caption。 |

[技术规格](02-technical-specification.md)定义实现契约；[运行/API 指南](04-running-and-api.md)区分 CPU 验证与用户 GPU 测试；[性能定义](05-performance-benchmarks.md)列出应采集的指标，不虚构已测数字。
