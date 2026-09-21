# Deliverable 1 — System architecture

## 1. Scope and implementation boundary

The implementation turns an owner-scoped collection of 100–1,000 images into an SD 1.5 LoRA adapter, a loading manifest and an evaluation report. It runs on **one Linux/WSL2 host**, with FastAPI, SQLite, local persistent files and a separate worker supervisor. One physical GPU runs at most one managed stage. The local validation target is an RTX 4060 Ti 16GB; CPU tests use an explicit, randomly initialized Diffusers/PEFT tiny graph plus fake slots. They never pretend that a CUDA device exists and their artifacts are test-only, not evidence of SD 1.5 quality.

Part 1 originally proposed PostgreSQL, versioned object storage and multi-host agents. The user selected a smaller implementation boundary for Part 2. This document and all three diagrams now describe that boundary; [the decision record](03-implementation-tradeoffs.md) preserves the reasons and migration implications.

The scheduler can manage several physical GPUs on the same host, including the assignment's 2–4 GPU scenario. The default configuration uses one device. Multiple requests can be queued concurrently. The implementation does not claim distributed failover, calibrated style quality or measured GPU throughput.

## 2. Architecture and data flow

![Single-host architecture](../diagrams/system-architecture.png)

[SVG](../diagrams/system-architecture.svg) · [PlantUML source](../diagrams/system-architecture.puml)

1. The API authenticates a configured bearer API key, creates an upload manifest and assigns internal file IDs. Clients stream individual files through authenticated endpoints. Filenames are metadata, not filesystem paths.
2. Upload completion freezes the recorded objects and queues verification. Job admission requires a completed dataset and atomically creates the job and first stage task. Idempotency and owner/global quotas are checked in the same transaction.
3. CPU preparation validates decoded formats and limits, normalizes images, removes exact duplicates, groups near duplicates and freezes an approximately 90/10 group-exclusive split. Captions use supplied text first, then the configured template or BLIP backend.
4. A singleton supervisor dispatches stage subprocesses. GPU work shares a finite pool: optional BLIP, LoRA training and evaluation compete for the same physical devices. An OS lock keyed by GPU UUID prevents managed stage overlap.
5. Training freezes base parameters, optimizes only LoRA parameters and writes complete resumable checkpoints. Evaluation reloads the exact saved adapter, generates a paired frozen-base/LoRA suite under identical conditions, and collects quality diagnostics.
6. Publication verifies artifact references, checksums and attempt ownership. A transaction registers one model per job and commits its terminal status. Technical success without calibration becomes `COMPLETED_UNVERIFIED`, not `READY`.
7. The owner retrieves a manifest and adapter for a compatible inference consumer. Unverified models require explicit download opt-in. Online inference serving is outside this assignment.

SQLite stores metadata, queue state, progress and the logical registry. Images and weights never enter database rows. File writes complete before references become visible. Database transactions remain short and do not include model execution.

## 3. Lifecycle and quality boundary

![Job lifecycle](../diagrams/job-lifecycle.png)

[SVG](../diagrams/job-lifecycle.svg) · [PlantUML](../diagrams/job-lifecycle.puml)

Job state and stage-task state are separate: a `RUNNING` job may have an `EVALUATE` task waiting in `PENDING`. A valid checkpoint is not a finished adapter, and a loadable adapter is not proof of style quality.

The default evaluation policy is uncalibrated. Reports contain technical checks, CLIP text alignment, held-out image similarity, output diversity and similarity to training images. Image similarity is a diagnostic proxy, not an independent style evaluator or proof of memorization. No universal threshold is invented. Only an operator-configured versioned policy with calibration evidence and passing required gates can authorize `READY`. Test-only artifacts can never authorize READY.

A/B generation holds prompts, seeds, scheduler settings, inference steps and guidance constant. The default comparison is the frozen base against its LoRA; a CLI comparison also supports compatible adapters. Synthetic-data smoke tests establish execution correctness, not product quality.

## 4. Resource management and recovery

![Local recovery](../diagrams/lease-recovery.png)

[SVG](../diagrams/lease-recovery.svg) · [PlantUML](../diagrams/lease-recovery.puml)

GPU dispatch uses the weighted class cycle `EVALUATE → TRAIN → CAPTION → TRAIN`, owner round-robin within each class and owner-local FIFO. Empty classes are skipped. Idle capacity can be borrowed when another owner has no eligible work; running stages are not preempted for fairness. CPU work has a separate bounded pool. The default admission limits are five nonterminal jobs per owner and 100 globally.

The supervisor has a singleton file lock. Each stage has an attempt token, heartbeat, deadline and recorded process identity. A replacement supervisor checks PID, process start time and process group before acting on an orphan. Lease expiry does not release a GPU: termination and lock availability must be confirmed before reuse. The watchdog stops work on controller loss or deadline expiry. Late output from an old attempt cannot change canonical state.

Cancellation is durable and waits for confirmed executor termination. Recoverable infrastructure failures have at most three attempts and bounded backoff. OOM, unsuitable input and failed quality gates end explicitly; the implementation does not silently change hyperparameters or automatically retrain for quality.

API and worker restarts can recover persistent state on the same host. Host or disk loss is a shared failure domain. Backups must include both the database and referenced files in a consistent stopped-service snapshot. Multi-host availability requires replacing persistence and coordinating remote execution; it is not achieved by copying this SQLite database onto a network filesystem.

## 5. Evaluation criteria coverage

| Criterion | Concrete implementation evidence |
|---|---|
| ML architecture and concepts | Frozen input manifests; group-exclusive split; train-only augmentation; LoRA-only optimization; full-state resume; saved-adapter inference; paired evaluation. |
| Code quality and architecture | Typed configuration; separate data, caption, training, evaluation, store, worker and API modules; common artifact/error contracts; CLI and automated tests. |
| Production considerations | Owner isolation, bounded streaming uploads, idempotency, durable task transitions, cancellation, device locks, stale-attempt rejection, health checks, structured errors and metrics. |
| Resource problem solving | One stage per device, shared evaluation capacity, weighted fair scheduling, explicit budgets, early input rejection, optional CPU captions and sequential model loading. |

The [technical specification](02-technical-specification.md) defines the implemented contracts. The [run guide](04-running-and-api.md) separates offline CPU verification from user-run GPU tests. [Performance definitions](05-performance-benchmarks.md) provide measurements to collect without pretending that they have already been measured.
