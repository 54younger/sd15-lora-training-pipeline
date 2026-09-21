# 实现取舍：可验证的单机交付与生产演进

本记录区分已实现机制和为 Part 1 保留的演进方向；不是把未来架构当成当前能力。源码索引见[系统架构](01-system-architecture.md)与[技术规格](02-technical-specification.md)。

## 1. 关键取舍

| 决策 | 得到什么 | 明确代价/迁移 |
|---|---|---|
| SQLite + 本地对象，而非 PostgreSQL + S3 | 无外部服务即可测试短事务、SHA-256、atomic publish、queue/recovery | 只有单主机 durable queue；数据库和对象同故障域。迁移要做 PostgreSQL schema/lease transaction、版本化 object key、scoped auth、部分上传恢复和集成测试。 |
| 一个 supervisor + UUID OS lock | 每张真实物理 GPU 一 slot，worker 与直接 CLI 共用锁，易验证 stale attempt/取消 | 无多主机 takeover；未来 agent 需要 lease、fencing、PID/termination evidence 和共享状态。 |
| 多 job 一卡一 stage，而非单 job DDP | 在 2–4 张物理卡上公平承载并发训练/评估，避免 16GB 卡的跨卡通信复杂度 | 单 job 不做 DDP/模型并行；一个 job 的 stage 仍逐步运行，吞吐要靠多 job 和多卡验收。 |
| batch 1 + accumulation 4 | 降低单卡峰值显存；每 4 个 microstep 才进行一个 optimizer/global step，checkpoint 在 optimizer boundary | wall time 更长，不能把 microstep 数当 optimizer steps；配置和报告分别记录 accumulation/global_step。 |
| FP16 + gradient checkpointing | 目标是降低 SD15 512px 的 CUDA 激活显存 | checkpointing 用额外计算换显存，FP16 需 scaler/数值监控；实际显存和能否运行必须由 GPU 测量，CPU 强制 FP32。OOM 报错不自动降低分辨率或偷偷改 optimizer。 |
| 冻结 SD15 基座，只训练 attention LoRA | adapter 小、可独立下载和 inference reload，训练参数/显存较少；rank/alpha 4/4 是当前配置起点 | 表达能力和风格质量需真实数据评估；不支持 SDXL、多 text encoder 或完整基座微调。 |
| template caption 默认，BLIP 可选 | 缺模型也能离线完成 manifest；BLIP 只在显式 profile 选择时占用资源，并记录 revision | template 只写安全 generic content，不保证主体描述；BLIP 失败显式报错，不 fallback 掩盖问题。CPU template 可释放 GPU 给 TRAIN/EVALUATE。 |
| CLIP + paired A/B，而非 FID/主观质量假设 | 固定 prompt/seed 的 base-vs-adapter 差值可复现，适合 smoke 和诊断 | CLIP 混合语义/风格，FID 未实现；没有仓库内校准数据和业务 threshold。 |
| 技术成功与质量 READY 分离 | 能发布并检查 loadable adapter，同时不把 tiny 或未校准结果冒充质量合格 | 默认 policy 为 null，job 是 COMPLETED_UNVERIFIED、model state 是 UNVERIFIED；calibration_reference 仅作 operator 证据标识，代码不读取/核验外部证据真实性。 |
| 仅基础设施 retry | 恢复 bounded、解释性强，避免 OOM/质量失败无限重训 | 需要换参数或重新质量候选时由 operator 新建 job；没有自动超参搜索。 |

## 2. inference-ready 交付边界

发布的可推理 artifact 不是一个裸 checkpoint：至少绑定 adapter.safetensors、其 SHA-256、base model 名称与 immutable revision/fingerprint、训练 input manifest digest、trigger token、inference/evaluation config，以及带 technical outcome/quality status 的 evaluation report。这些 metadata 分布在 training-input、training-result、evaluation report 和 model registry；download endpoint 本身只返回 adapter bytes。PUBLISH 逐项检查 adapter、manifest、report、input binding 和 checksum；model record 每 job 唯一。使用者可用该 adapter 加载兼容 SD15 基座并复现 trigger/inference 配置，但默认结果未质量认证，下载 UNVERIFIED 需要显式同意。在线 inference serving、autoscaling、TLS ingress、模型路由和实时请求 API 未实现。

## 3. 可靠性保留与明确缺口

保留下来的生产性质是 owner 隔离、大小/摘要限制、必要 mutation 幂等、冻结 manifest 链、attempt fencing、heartbeat/lease、GPU 资源公平、取消、唯一 publication、完整 resume 和结构化错误/metrics。资源调度是 weighted EVALUATE → TRAIN → CAPTION → TRAIN， 并在 owner 间 round-robin；它是多 job 的逻辑公平，不是 DDP。

仍不保证多主机故障转移、磁盘损坏零丢失、恶意代码 sandbox、内容审核、外部校准器、自动 retention/backup 或在线 serving。2–4 张 GPU、SD15/BLIP/CLIP 质量和 benchmark 必须在目标硬件实际运行；CPU tiny 和 fake slots 只能覆盖离线逻辑。所有未运行数字写 not_measured。

## 4. 证据与外部参考

实现证据包括 [config.py](../src/lora_pipeline/config.py) 的 batch/accumulation/precision/checkpoint 配置，[training.py](../src/lora_pipeline/training.py) 的冻结参数、resume 和进度，[store.py](../src/lora_pipeline/store.py) 的 GPU slot/fairness/attempt，[evaluation.py](../src/lora_pipeline/evaluation.py) 的 policy 方向，以及 [tests/test_training.py](../tests/test_training.py)、[tests/test_scheduler_recovery.py](../tests/test_scheduler_recovery.py) 和 [tests/test_evaluation.py](../tests/test_evaluation.py)。

有关 single-writer 与 WAL 的边界，见 [SQLite 文档](https://sqlite.org/wal.html)；本项目选择默认 rollback journal，不把网络文件系统当 HA 数据库。有关 attention LoRA 与 adapter saving，见 [Diffusers LoRA guide](https://huggingface.co/docs/diffusers/training/lora)；有关 optimizer/RNG/scaler checkpointing，见 [Accelerate checkpoint guide](https://huggingface.co/docs/accelerate/usage_guides/checkpoint)。
