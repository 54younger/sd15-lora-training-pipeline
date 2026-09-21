# 性能基准、资源实验与提交证据

本文件定义如何报告数字，不制造数字。`VALIDATION.md` 的 2026-09-21 记录不是本次运行结果；Docker daemon、真实 SD 1.5 GPU、IBean 训练和 2–4 GPU 并发若本次没有实际执行，必须写 `measurement_status=not_measured`、`sample_count=0`，数值写 `null`。

## 四类结果必须分开

| 工作负载 | 用途 | 可以得出的结论 | 不能得出 |
|---|---|---|---|
| CPU tiny + pytest | 离线回归/服务合约 | 编排、checkpoint、错误/幂等、指标算术 | SD 1.5 质量、GPU 吞吐/显存、容量 |
| 10-step GPU smoke | 真实基座功能验收 | CUDA、缓存、5-step resume、adapter load、技术评估 | 稳态吞吐、容量、质量阈值 |
| 100-step GPU performance | 固定配置性能 | warm-up 后 step 时间、吞吐、峰值显存、checkpoint save 成本 | 视觉质量 |
| paired evaluation/A-B | 质量诊断/发布输入 | 固定 prompt/seed 的 base/adapter 对照、CLIP/held-out 报告 | 未校准 policy 下 PASS；CLIP 单指标不等于风格质量 |

真实训练固定 IBean-999（或明确的真实 manifest）、pinned SD 1.5 revision、resolution/batch/accumulation/rank/seed。合成图只用于 CPU tiny/API demo。

## 已有 instrumentation 与外部测量

运行时已经写入/返回 job/stage/attempt 状态、`global_step`、loss、累计 `samples_processed`、本次调用 `samples_processed_this_run`、`samples_per_second`、（发生 checkpoint 保存的 step 上）checkpoint 秒数、CPU RSS、CUDA allocated/reserved，以及 adapter/manifest/evaluation checksum。queue/lease 时间存在 SQLite 的 `stage_tasks`/`task_attempts` 内部记录，需要 operator 只读导出或受控采集；公开 `GET /v1/training-jobs/{id}` 只返回 job/stage state、attempt count 和 progress。API 另外提供按 route/status 的 Prometheus counter/histogram 和 job state gauge；`/metrics` 需要 admin token。这些是可采集字段，不表示已有 GPU 数字。

仍需目标主机外部受控测量或补充：GPU 型号/UUID/driver/CUDA、NVML utilization、wall time（含/不含模型加载）、queue P50/P95、2–4 GPU fairness/occupancy、失败/OOM/retry 成本、网络下载和磁盘。`managed slot occupancy = occupied slot seconds / configured slot seconds`，不是硬件 GPU utilization。

progress 新增 lifecycle 事件后不再是同构 step 表；下列名称只是实现中的示例，并非穷举。聚合 optimizer step 时**必须先过滤 `phase == "training"`**，再读取 `global_step`、`elapsed_seconds`、`samples_processed_this_run`；`training_started`、`model_loading`、`checkpoint_saving`、`adapter_saved`、`training_completed` 等事件只用于生命周期/加载/保存阶段追踪，否则会把 0% 加载事件误算进吞吐。`checkpoint_seconds` 只在实际保存 checkpoint 的 step 上出现；CPU/GPU memory 字段也可能因设备或采集条件缺省，汇总时按 `null` 处理，不能当作零。

## 100-step 性能协议

按 [README](../README.md) 的 Benchmarks 小节在停止常驻 worker、固定单一 GPU UUID 后执行：

1. 预热 `model-cache` 并写 `/data/sd15-smoke-pinned.json`，下载与计算分开。
2. 同一真实 `training-input.json`、model revision 和完整配置运行 **1 次 warm-up**。
3. 再运行 **至少 3 次完整 100 optimizer steps**；每次从 step 1 开始。稳态只从 optimizer step 11 开始，排除前 10 个 optimizer steps，不把 accumulation microsteps 当 optimizer steps。
4. 保存每次 raw progress、`training-result.json`、环境/preflight、配置和 manifest hash；任一失败/OOM/checksum 错误都令实验失败，不补造 summary。
5. 对三次完整 runs 报均值、stdev 和原始值；总 wall time 与 steady-state step time 分列。

聚合示例（不是替代 README Docker 命令）：

```python
steps = [p for p in progress if p.get("phase") == "training"]
warmup, steady = steps[:10], steps[10:]
b, last = steps[9], steps[-1]  # step 10 boundary through step 100
steady_seconds = last["elapsed_seconds"] - b["elapsed_seconds"]
steady_samples = last["samples_processed_this_run"] - b["samples_processed_this_run"]
```

输出至少含 `warmup_steps_excluded=10`、`steady_steps`、`steady_samples_per_second`、`wall_seconds_including_load`、`checkpoint_seconds_total`、`peak_gpu_memory_allocated/reserved`、`peak_cpu_rss_bytes`、GPU UUID、revision、commit、`sample_count=3`。Docker prepare 外层墙钟应标为 measured prepare wall time，不称纯算法吞吐。

## 2–4 GPU / 多 owner 实验

每个物理 GPU UUID 是一个 scheduler slot。固定数据、profile、100-step 设置和 pinned revision；分别运行 2、3、4 个实际 UUID（设备不足则只报告可用组），每组至少两个 owner、相同数量 bounded jobs，并做单 GPU/单 owner baseline。

每组记录 owner 的 enqueue→dispatch queue wait、完成时间、吞吐和 fairness（Jain 或明确 max/min+p95）；FIFO/owner rotation；attempt、lease、retry、取消、stale completion；每 UUID 的 slot occupied seconds、job GPU seconds、NVML utilization/显存；OOM、`GPU_BUSY`、`GPU_BUDGET_EXCEEDED`、`ADMISSION_LIMIT` 和最终状态；API P50/P95、error rate、worker backlog。CPU `fake_slots` 只能验证 admission/fairness，不能外推 GPU scaling、显存或 CUDA occupancy。异构 GPU 必须分 profile，未运行组保留 `null`。

## 指标定义

| 指标 | 定义 | 注意事项 |
|---|---|---|
| API latency/error | route template P50/P95；按 status 分认证/校验与 5xx | label 不含 token/payload/resource ID |
| Queue wait | dispatch - ready/enqueue，按 stage/owner | 不并入执行时间 |
| Prepare throughput | `accepted_unique / prepare_wall_seconds` | 同时报提交数、尺寸、warning/rejection |
| Optimizer step | steady elapsed / steady optimizer steps | 只用 `phase=training`，排除前 10 |
| Training throughput | `samples_processed_this_run / current invocation elapsed` | resume 的历史累计不可重复计入 |
| Memory | progress 中 CPU RSS、CUDA allocated/reserved 峰值 | 无 CUDA 或字段缺省为 `null`，记录 phase/进程范围 |
| Checkpoint | 完整保存 timer、bytes、checksum | 当前 runtime 有 save 时间；restore 时间需外部计时，不要伪称已有完整字段 |
| Adapter/artifact | adapter bytes、manifest/report SHA-256 | 与 optimizer checkpoint、base cache 分开 |
| Generation/evaluation | 秒/图像或总秒，注明 resolution/steps/模型加载 | generation、CLIP loading、scoring 分开；eval elapsed 不能自动解释成完整总周期 |
| Slot/budget | occupied slot seconds / configured slot seconds；GPU 秒含 caption/train/eval、失败/retry | 不是 NVML utilization |
| Reliability | OOM、retry、stale result、取消 request→exit、lease expiry | 失败成本进入 summary |

## 报告模板与提交清单

```json
{
  "measurement_status": "not_measured",
  "sample_count": 0,
  "commit": null,
  "environment": {"gpu_uuid": null, "driver": null, "cuda": null},
  "model": {"name": "stable-diffusion-v1-5/stable-diffusion-v1-5", "revision": null},
  "metrics": {
    "steady_samples_per_second_mean": null,
    "optimizer_step_seconds": null,
    "peak_gpu_memory_allocated": null,
    "api_latency_p95_ms": null,
    "managed_slot_occupancy": null
  },
  "reason": "Target-host Docker/GPU controlled workload was not run."
}
```

提交 evidence 至少包括：

- commit、依赖/driver/CUDA/OS/CPU/RAM、GPU UUID/显存和 cache cold/warm；
- IBean-999 或真实数据来源、frozen input manifest SHA-256、caption/split 统计；
- `/data/sd15-smoke-pinned.json` 的 revision、训练配置和 preflight；
- CPU pytest/ruff/mypy 与 10-step smoke 的实际日期/结果，明确 `test_only`/smoke；
- 1 warm-up + 3 full 100-step 的 raw progress/result、summary JSON/CSV、过滤规则和失败/retry/OOM；
- evaluation report、paired 条件、CLIP 限制和 quality policy 校准来源；
- API health/auth/Idempotency/owner isolation、checkpoint checksum、model artifact 下载证据；
- 2–4 GPU、多 owner 的 queue/fairness/occupancy 结果；未实测字段为 `null/not_measured`。

质量阈值和生产边界回链 [技术规格](02-technical-specification.md) 与 [设计取舍](03-implementation-tradeoffs.md)：未校准的 `UNCALIBRATED`/`COMPLETED_UNVERIFIED` 不能写成 PASS/READY，CLIP 分数不能单独成为风格质量证明。
