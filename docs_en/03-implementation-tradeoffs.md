# Decision record: a testable single-host delivery and production evolution

This record separates implemented mechanisms from the Part 1 evolution boundary; future architecture is not claimed as current capability. See the [system architecture](01-system-architecture.md) and [technical specification](02-technical-specification.md) for the source index.

## 1. Key decisions

| Decision | Benefit | Explicit cost / migration |
|---|---|---|
| SQLite + local objects instead of PostgreSQL + S3 | Short transactions, SHA-256, atomic publication and queue/recovery are testable without external services | The durable queue is single-host and database/object share a failure domain. Migration needs PostgreSQL schema/lease transactions, versioned object keys, scoped authorization, partial-upload recovery and integration tests. |
| One supervisor + UUID OS lock | One slot per real physical GPU; worker and direct CLI share the lock; stale attempts and cancellation are testable | No multi-host takeover. Future agents need leases, fencing, PID/termination evidence and shared state. |
| One stage per GPU per job instead of single-job DDP | Fair concurrent training/evaluation across 2–4 physical cards without cross-card communication complexity on a 16GB-class card | No DDP/model parallelism; a job still advances stage by stage. Throughput must be measured with multiple jobs and real cards. |
| Batch 1 + accumulation 4 | Lower peak memory; four microsteps produce one optimizer/global step, and checkpoints are at optimizer boundaries | Longer wall time; microstep count must not be reported as optimizer steps. Config and reports retain accumulation and global_step separately. |
| FP16 + gradient checkpointing | The goal is lower CUDA activation memory for SD15 512px | Checkpointing exchanges extra compute for memory and FP16 needs scaler/numerical monitoring; actual memory and whether it runs must be measured on a GPU, while CPU is forced to FP32. OOM does not silently lower resolution or change the optimizer. |
| Frozen SD15 base with attention LoRA only | Small independently downloadable/reloadable adapter and fewer trainable parameters; rank/alpha 4/4 is the current starting configuration | Capacity and style quality require real-data evaluation; SDXL, dual text encoders and full-base fine-tuning are unsupported. |
| Template captions by default, optional BLIP | Offline manifests work without another model; BLIP is selected explicitly, records its revision and consumes resources only when needed | The template is safe generic content, not subject recognition; BLIP failure is explicit, never hidden by fallback. CPU template work releases GPUs for TRAIN/EVALUATE. |
| CLIP + paired A/B instead of FID or subjective quality claims | Fixed prompt/seed base-vs-adapter differences are reproducible for smoke and diagnostics | CLIP mixes semantic/style signals; FID is not implemented and the repository has no calibrated business threshold. |
| Technical success separated from READY | A loadable adapter can be published without pretending that tiny or uncalibrated output is quality-approved | The default policy is null, so the job is COMPLETED_UNVERIFIED and the model state is UNVERIFIED; calibration_reference is only an operator evidence identifier and code neither reads nor verifies external evidence. |
| Infrastructure retry only | Bounded, explainable recovery without infinite OOM/quality retraining | A new configuration/job is required for changed parameters or another quality candidate; there is no automatic hyperparameter search. |

## 2. Inference-ready artifact boundary

The published inference artifact is not a bare checkpoint. It binds adapter.safetensors and its SHA-256, the base model name plus immutable revision/fingerprint, the training input-manifest digest, trigger token, inference/evaluation configuration, and an evaluation report with technical outcome and quality status. This metadata is distributed across training-input, training-result, evaluation report and the model registry; the download endpoint itself returns adapter bytes only. PUBLISH checks adapter, manifest, report, input binding and checksums; one model record exists per job. A consumer can load the adapter with the compatible SD15 base and reproduce the trigger/inference settings, but the default result is not quality-certified and downloading UNVERIFIED requires explicit consent. Online inference serving, autoscaling, TLS ingress, model routing and a realtime request API are not implemented.

## 3. Reliability retained and explicit gaps

Retained production properties are owner isolation, bounded size/digest checks, required mutation idempotency, frozen manifest lineage, attempt fencing, heartbeat/lease, fair GPU resources, cancellation, unique publication, complete resume and structured errors/metrics. Scheduling uses the weighted EVALUATE → TRAIN → CAPTION → TRAIN cycle and owner round-robin; it is logical fairness across jobs, not DDP.

There is no guarantee of multi-host failover, zero loss after disk failure, hostile-code sandboxing, content moderation, an external calibrator, automated retention/backups or online serving. 2–4 GPU execution, SD15/BLIP/CLIP quality and benchmarks must run on target hardware; CPU tiny and fake slots cover offline logic only. Every unrun number remains not_measured.

## 4. Evidence and external references

Implementation evidence includes batch/accumulation/precision/checkpoint configuration in [config.py](../src/lora_pipeline/config.py), frozen parameters/resume/progress in [training.py](../src/lora_pipeline/training.py), GPU slots/fairness/attempts in [store.py](../src/lora_pipeline/store.py), policy directions in [evaluation.py](../src/lora_pipeline/evaluation.py), and [test_training.py](../tests/test_training.py), [test_scheduler_recovery.py](../tests/test_scheduler_recovery.py), and [test_evaluation.py](../tests/test_evaluation.py).

For SQLite single-writer/WAL boundaries see [SQLite documentation](https://sqlite.org/wal.html); this project uses the default rollback journal and does not treat a network filesystem as HA. For attention LoRA and adapter saving see the [Diffusers LoRA guide](https://huggingface.co/docs/diffusers/training/lora); for optimizer/RNG/scaler checkpointing see the [Accelerate checkpoint guide](https://huggingface.co/docs/accelerate/usage_guides/checkpoint).
