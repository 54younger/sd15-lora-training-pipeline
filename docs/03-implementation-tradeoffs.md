# 决策记录——可测试的单主机流水线

状态：已接受。本文解释当前实现边界，不声称原始生产设计的所有机制都已构建。

| 决策 | 收益 | 成本与迁移 |
|---|---|---|
| SQLite 代替 PostgreSQL | 无需数据库服务，临时目录即可运行 CPU 集成测试。 | 只有单主机写入串行；没有 `SKIP LOCKED` 或分布式 HA。迁移需要新事务、迁移脚本和并发测试。 |
| 本地文件代替 S3 | 无 bucket/凭据，checksum 和原子写入容易测试。 | 上传经过 API，文件共享主机故障域；S3 迁移需版本固定、授权和部分写入恢复。 |
| 一个 supervisor 加设备锁 | 设备所有权清晰，本地进程恢复简单。 | 没有多主机接管；远程调度需 agent、lease、fencing 和可靠终止证据。 |
| SD 1.5 作为真实模型族 | 适合 16GB 消费 GPU 的训练和推理集成。 | 不声明 SDXL 或多模型族支持；新增 backend 必须实现并测试。 |
| 保守图像过滤 | 不会误删有意的模糊、低对比度和异常曝光。 | warning 不能保证数据适用；感知分组是启发式，可能过度分组。 |
| template 或 BLIP caption | 无额外权重即可离线运行；BLIP 可提供内容描述。 | template 不识别主体，BLIP 可能误述；失败不会隐藏。 |
| 技术成功与质量审批分离 | 合成数据可测试全链路而不伪称风格质量。 | 默认结果为 `UNVERIFIED`，下载需显式同意；`READY` 需要外部校准证据。 |
| CLIP 诊断，不默认 FID | 小规模作业适合配对图文比较。 | CLIP 混合内容和风格；不宣称阈值能独立验证风格。 |
| 仅基础设施重试 | 恢复可解释且资源有界。 | OOM 或质量失败需新配置；不自动搜索。 |
| CPU tiny 测试加本地 GPU smoke | tiny 使用 Diffusers/PEFT 的随机初始化 Autoencoder/UNet/scheduler 图，离线验证真实 LoRA 梯度、冻结参数、resume 和编排。 | 随机权重不验证 CUDA kernel、预训练输出或真实风格质量；真实 SD/BLIP/CLIP 另行运行，tiny 产物永远是 test-only。 |
| 只定义指标，不虚构基准 | 无 GPU 实测时仍提供可复现实验协议。 | 未运行值保持 `not_measured`。 |

## 保留的可靠性

实现保留不可变 manifest 链、有界上传、owner 授权、接纳配额、幂等 mutation、持久 task、唯一发布和完整 checkpoint。API 与 worker 是独立进程；新的 supervisor 不把过期 heartbeat 当作旧 GPU 进程已停止的证据。`COMPLETED_UNVERIFIED` 是技术成功，`READY` 仍是质量审批结果；test-only 结果始终保留标记。

## 不是生产保证

没有多主机故障转移、敌意代码沙箱、内容审核、校准风格评估器、磁盘故障零数据丢失或自动保留清理。API 不接收自定义 Python 或任意 checkpoint。公开服务还需要 TLS、限流和身份提供商。一个物理 GPU 不能被虚构为四个 CUDA slot；fake slots 只测试 admission/fairness。

## 参考来源

SQLite 的单写入和同主机约束：[SQLite 文档](https://sqlite.org/wal.html)。Diffusers 的 attention LoRA 和 adapter 保存：[Diffusers LoRA guide](https://huggingface.co/docs/diffusers/training/lora)。Accelerate 对 optimizer、RNG、scaler 和 data-loader checkpoint 的说明：[checkpoint guide](https://huggingface.co/docs/accelerate/usage_guides/checkpoint)。
