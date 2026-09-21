# Performance benchmarks — definitions and collection protocol

This assignment delivers a measurement schema and runtime instrumentation. No GPU benchmark numbers are claimed. CPU unit-test elapsed time verifies development tests; it is not a training throughput benchmark. Real GPU smoke runs are short functional checks, not representative capacity measurements.

## Metrics

| Metric | Unit / aggregation | Collection / interpretation |
|---|---|---|
| API latency | milliseconds, P50/P95 by route template | Request middleware or a controlled client workload; exclude payload content and resource IDs from metric labels. |
| API errors | count / rate by route and status | Request completion; report authentication/validation failures separately from server failures. |
| Queue wait | seconds by stage | Stage dispatch time minus enqueue/ready time; do not fold into execution duration. |
| Data preparation throughput | accepted images/second | Accepted unique count divided by preparation wall time; also report submitted count and dimensions. |
| Rejection / duplication | count and fraction | Per-file preparation report; categories retain corrupt, size, duplicate and warning reasons. |
| Optimizer step time | seconds/optimizer step | Training progress; accumulation microsteps are not optimizer steps. Separate warm-up when measuring steady state. |
| Training progress | samples | `samples_processed` is the cumulative cursor across resume; `samples_processed_this_run` counts only samples added by the current `train()` invocation. Record both with the global step. |
| Training throughput | processed samples/second | `samples_per_second` uses `samples_processed_this_run` for the numerator and the current invocation's elapsed time for the denominator. Do not include historical cumulative samples after resume. |
| Host memory | bytes, peak process RSS | psutil process samples; child-process scopes must be stated. |
| CUDA memory | bytes, peak allocated and reserved | Reset PyTorch peak counters at phase start, synchronize at timing boundaries. `null` without CUDA. |
| Checkpoint save / restore | seconds and bytes | Timer around complete checkpoint publication/load, including integrity checks. |
| Adapter size | bytes | Final adapter file size, separately from optimizer checkpoints and base-model cache. |
| Generation latency | seconds/image | Full generation interval divided by output count; state resolution, steps and model-load inclusion. |
| Evaluation duration | seconds | Generation, embedding extraction and report phases; distinguish model loading from metric math. |
| Managed slot occupancy | fraction | Occupied slot seconds / available configured slot seconds. This is not hardware GPU compute utilization. |
| Hardware GPU utilization | percent | Optional external `nvidia-smi`/NVML sampling on the user's host; unsupported when unavailable. |
| GPU-stage budget use | seconds/job | Sum of managed GPU stage execution exposure, including failed attempts, caption and evaluation. |
| Reliability | counts / seconds | Retries, OOMs, stale completions rejected, cancellation request-to-confirmed-exit time. |

## Repeatable measurement protocol

1. Record commit/version, dependency versions, OS, CPU/RAM, GPU name/UUID/VRAM, driver/CUDA, dataset manifest hash, model revision and configuration.
2. Describe cold or warm cache explicitly. Keep model download time separate from compute. Pin prompts, seeds and model versions for comparisons.
3. Use the same dataset size/dimensions and training settings. Repeat a steady-state workload several times before reporting percentiles; include sample count and dispersion.
4. For concurrency, submit bounded work for two owners and compare one versus multiple actual physical slots. CPU fake slots validate admission/fairness, not GPU scaling.
5. Report throughput and queue latency independently. Include unsuccessful jobs and retries when estimating service cost.
6. Leave unsupported or unrun metrics empty with a reason. Do not insert estimates into measured fields.

Illustrative report schema, intentionally containing no measured values:

```json
{
  "measurement_status": "not_measured",
  "workload": "sd15-lora-512-batch1-accum4",
  "environment": {"gpu": "RTX 4060 Ti 16GB", "model_revision": null},
  "sample_count": 0,
  "metrics": {
    "api_latency_p95_ms": null,
    "prepare_images_per_second": null,
    "optimizer_step_seconds": null,
    "peak_cuda_allocated_bytes": null,
    "checkpoint_restore_seconds": null,
    "generation_seconds_per_image": null,
    "managed_slot_occupancy": null
  },
  "reason": "Run a controlled workload on the target host before filling measured values."
}
```

Progress reports retain both the cumulative cursor and the current-invocation increment so resumed runs cannot mix historical samples into a new throughput measurement. Capacity planning may use `jobs/hour ≈ available_slots × utilization / total_GPU_hours_per_job`. This is a planning relationship only. Caption, training, evaluation, retries and warm-up all contribute to total GPU time; heterogeneous devices require separate profiles.
