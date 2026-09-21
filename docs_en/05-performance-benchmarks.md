# Performance benchmarks, resource experiments, and submission evidence

This document defines how to report numbers; it does not invent them. `VALIDATION.md` is a historical record dated 2026-09-21. If Docker, real SD 1.5 GPU/IBean training, or 2–4 GPU concurrency was not actually run for this submission, use `measurement_status=not_measured`, `sample_count=0`, and `null` values.

## Keep four result classes separate

| Workload | Purpose | Supports | Does not support |
|---|---|---|---|
| CPU tiny + pytest | Offline regression/service contracts | Orchestration, checkpoint, errors/idempotency, metric arithmetic | SD 1.5 quality, GPU throughput/memory, capacity |
| 10-step GPU smoke | Real-base functional acceptance | CUDA, cache, 5-step resume, adapter load, technical evaluation | Steady throughput, capacity, quality threshold |
| 100-step GPU performance | Fixed-config performance | Warmed step time, throughput, peak memory, checkpoint-save cost | Visual quality |
| Paired evaluation/A-B | Quality diagnostics/publication input | Base/adapter comparison under fixed prompts/seeds, CLIP/held-out report | PASS without calibrated policy; CLIP alone is not style quality |

Fix IBean-999 (or another explicit real manifest), pinned SD 1.5 revision, resolution/batch/accumulation/rank/seed. Synthetic images are for CPU tiny/API demo only.

## Existing instrumentation versus external measurement

Runtime artifacts expose job/stage/attempt state, `global_step`, loss, cumulative `samples_processed`, per-invocation `samples_processed_this_run`, `samples_per_second`, checkpoint seconds (on steps that actually save a checkpoint), CPU RSS, CUDA allocated/reserved, and adapter/manifest/evaluation checksums. Queue/lease times live in SQLite's internal `stage_tasks`/`task_attempts` records and require operator read-only export or controlled collection; public `GET /v1/training-jobs/{id}` exposes job/stage state, attempt count, and progress only. The API separately exposes route/status Prometheus counters/histograms and job-state gauges; `/metrics` requires an admin token. These are fields available for collection, not claimed GPU results.

The target host still needs controlled external measurement or metadata for GPU model/UUID/driver/CUDA, NVML utilization, wall time with/without model loading, queue P50/P95, 2–4 GPU fairness/occupancy, failure/OOM/retry cost, network download, and disk. `managed slot occupancy = occupied slot seconds / configured slot seconds`; it is not hardware GPU utilization.

With lifecycle events, progress is no longer a homogeneous step table; the names below are implementation examples, not an exhaustive event list. **Filter `phase == "training"` before aggregating optimizer steps**, then read `global_step`, `elapsed_seconds`, and `samples_processed_this_run`. `training_started`, `model_loading`, `checkpoint_saving`, `adapter_saved`, and `training_completed` are lifecycle/load/save observations; including them would count a 0% load event as throughput. `checkpoint_seconds` appears only on steps that actually save a checkpoint; CPU/GPU memory fields may be absent when the device or collector cannot provide them, so aggregate missing values as `null`, never zero.

## 100-step performance protocol

Follow the benchmark section in [README.md](../README.md), stop the resident worker, and pin one physical GPU UUID:

1. Warm `model-cache` and write `/data/sd15-smoke-pinned.json`; separate download from compute.
2. Run **one warm-up** with the same real `training-input.json`, model revision, and complete configuration.
3. Run **at least three complete 100-optimizer-step runs**, each starting at step 1. Steady state starts at optimizer step 11, excluding the first 10 optimizer steps; accumulation microsteps are not optimizer steps.
4. Save raw progress, `training-result.json`, environment/preflight, config, and manifest hash for every run. Any failure/OOM/checksum error fails the experiment; do not fabricate a summary.
5. Report mean, stdev, and raw values for the three complete runs. Keep total wall time and steady-state step time separate.

Minimal aggregation (not a replacement for the README Docker command):

```python
steps = [p for p in progress if p.get("phase") == "training"]
warmup, steady = steps[:10], steps[10:]
b, last = steps[9], steps[-1]  # step-10 boundary through step 100
steady_seconds = last["elapsed_seconds"] - b["elapsed_seconds"]
steady_samples = last["samples_processed_this_run"] - b["samples_processed_this_run"]
```

At minimum report `warmup_steps_excluded=10`, `steady_steps`, `steady_samples_per_second`, `wall_seconds_including_load`, `checkpoint_seconds_total`, `peak_gpu_memory_allocated/reserved`, `peak_cpu_rss_bytes`, GPU UUID, revision, commit, and `sample_count=3`. If preparation timing is the outer Docker wall time, label it measured prepare wall time rather than pure algorithm throughput.

## 2–4 GPU / multi-owner experiment

Each physical GPU UUID is one scheduler slot. Keep data, profile, 100-step settings, and pinned revision fixed. Run separate groups with 2, 3, and 4 real UUIDs (report only available groups), at least two owners per group and equal bounded jobs, plus a one-GPU/one-owner baseline.

For each group record owner enqueue-to-dispatch queue wait, completion time, throughput, and fairness (Jain or explicit max/min plus p95); FIFO/owner rotation; attempts, leases, retries, cancellation, stale completions; per-UUID occupied slot seconds, job GPU seconds, NVML utilization/memory; OOM, `GPU_BUSY`, `GPU_BUDGET_EXCEEDED`, `ADMISSION_LIMIT`, and final states; API P50/P95, error rate, and worker backlog. CPU `fake_slots` validates admission/fairness only and cannot imply GPU scaling, memory, or CUDA occupancy. Separate heterogeneous GPUs by profile. Unrun groups remain `null`.

## Metric definitions

| Metric | Definition | Notes |
|---|---|---|
| API latency/error | Route-template P50/P95; split auth/validation from 5xx | Labels omit token/payload/resource IDs |
| Queue wait | dispatch minus ready/enqueue by stage/owner | Do not fold into execution time |
| Preparation throughput | `accepted_unique / prepare_wall_seconds` | Also report submitted count, dimensions, warnings/rejections |
| Optimizer step | steady elapsed / steady optimizer steps | `phase=training` only; exclude first 10 |
| Training throughput | `samples_processed_this_run / current invocation elapsed` | Do not count historical resume samples again |
| Memory | Peak CPU RSS and CUDA allocated/reserved from progress | `null` without CUDA or when fields are absent; state process/phase scope |
| Checkpoint | Complete save timer, bytes, checksum | Runtime has save time; restore needs an external timer and must not be claimed as an existing full-cycle field |
| Adapter/artifact | Adapter bytes and manifest/report SHA-256 | Separate from optimizer checkpoint and base cache |
| Generation/evaluation | Seconds/image or total, with resolution/steps/model-load scope | Separate generation, CLIP loading, and scoring; eval elapsed is not automatically a full-cycle duration |
| Slot/budget | Occupied slot seconds / configured slot seconds; GPU seconds include caption/train/eval and failed retries | Not NVML utilization |
| Reliability | OOM, retry, stale result, cancellation request-to-exit, lease expiry | Include failure cost in summary |

## Report template and submission checklist

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

Submit at least:

- commit, dependency/driver/CUDA/OS/CPU/RAM, GPU UUID/memory, and cold/warm cache state;
- IBean-999 or real-data provenance, frozen input SHA-256, caption/split statistics;
- `/data/sd15-smoke-pinned.json` revision, training config, and preflight;
- actual dated CPU pytest/Ruff/mypy and 10-step smoke results, explicitly marked `test_only`/smoke;
- raw progress/results and summary JSON/CSV for 1 warm-up + 3 full 100-step runs, filtering rule, and failure/retry/OOM record;
- evaluation report, paired conditions, CLIP limitations, and quality-policy calibration source;
- API health/auth/idempotency/owner-isolation, checkpoint checksum, and model-artifact download evidence;
- 2–4 GPU multi-owner queue/fairness/occupancy results; every unmeasured field remains `null/not_measured`.

Link threshold calibration and production boundaries to the [technical specification](02-technical-specification.md) and [trade-offs](03-implementation-tradeoffs.md): uncalibrated `UNCALIBRATED`/`COMPLETED_UNVERIFIED` must not be written as PASS/READY, and a CLIP score alone is not style-quality proof.
