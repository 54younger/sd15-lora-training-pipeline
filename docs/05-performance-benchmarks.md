# 性能基准——定义与采集协议

本作业提供测量 schema 和运行时 instrumentation，不声称已测得 GPU 数字。CPU 测试耗时只验证开发测试，不是训练吞吐基准；GPU smoke 是短功能检查，也不是容量测量。

## 指标

| 指标 | 单位/聚合 | 采集与解释 |
|---|---|---|
| API latency | 毫秒，按 route 的 P50/P95 | 中间件或受控客户端；label 不含 payload 和资源 ID。 |
| API errors | 按 route/status 的数量/比例 | 区分认证/校验失败与服务端失败。 |
| Queue wait | 按 stage 的秒数 | dispatch 减 ready/enqueue；不并入执行时间。 |
| Data preparation throughput | accepted images/second | accepted unique 数除准备墙钟时间，同时报告提交数和尺寸。 |
| Optimizer step time | 秒/optimizer step | accumulation microstep 不是 optimizer step；单独报告 warm-up。 |
| Training progress | samples | `samples_processed` 是跨 resume 的累计 cursor；`samples_processed_this_run` 只统计当前 `train()` 调用新增的样本。二者都应与 global step 一起记录。 |
| Training throughput | samples/second | `samples_per_second` 使用当前调用的 `samples_processed_this_run` 作为分子，并除以当前调用 elapsed time；resume 后不能把历史累计样本放进本次吞吐分子。 |
| Host/CUDA memory | bytes，峰值 RSS/allocated/reserved | CUDA phase 重置 peak counter 并在边界同步；无 CUDA 用 `null`。 |
| Checkpoint save/restore | 秒和 bytes | 含完整发布/加载及完整性校验。 |
| Generation latency | 秒/图像 | 说明分辨率、步数和是否包含模型加载。 |
| Slot occupancy | 比例 | 已占用 slot 秒 / 配置 slot 秒，不等同硬件利用率。 |
| GPU budget use/reliability | 秒/job；次数/秒数 | 含失败 attempt、caption、评估、重试、OOM、过期结果和取消退出时间。 |

## 可复现实验协议

1. 记录 commit、依赖、OS、CPU/RAM、GPU UUID/显存、driver/CUDA、dataset manifest hash、模型 revision 和配置。
2. 明确 cold/warm cache，下载时间与计算分开；固定 prompts、seeds 和模型版本。
3. 使用相同数据规模和设置，重复 steady-state workload，并报告样本数和离散程度。
4. 并发时用两个 owner，在一个与多个真实物理 slot 间比较；CPU fake slots 只验证接纳/公平，不验证 GPU 扩展。
5. 分开报告吞吐与队列延迟，把失败 job 和重试纳入成本。
6. 未运行指标保持空值并给原因，不能把估计填入 measured 字段。

示例报告可使用 `measurement_status: "not_measured"`、`sample_count: 0`，指标值为 `null`，原因写明需在目标主机运行受控 workload。训练进度报告同时保留累计 cursor 与本次调用增量，避免 resume 后吞吐分母/分子混淆。容量规划关系为 `jobs/hour ≈ available_slots × utilization / total_GPU_hours_per_job`，只是规划公式。
