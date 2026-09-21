# Automatic LoRA training pipeline — implementation specification

## 1. Components and persistence

The package `lora_pipeline` exposes a FastAPI application, an independent worker command and direct operator CLI commands. It targets Python 3.12 on Linux/WSL2. The runtime dependencies are pinned in `pyproject.toml`; no training code is fetched dynamically at runtime.

| Component | Responsibility |
|---|---|
| `api` | Bearer-key authentication, owner authorization, bounded uploads, validated requests, idempotency, status/cancel/download, health and metrics. |
| `store` | SQLite transactions, quotas, durable queue, attempts, immutable output references, scheduler cursor and registry uniqueness. |
| `worker` | Singleton supervisor, bounded CPU/GPU dispatch, stage subprocesses, watchdog, process identity checks, recovery and cancellation. |
| `data` / `captions` | Decode and normalize; exact deduplication; near-duplicate grouping; frozen split; provided/template/BLIP captions. |
| `training` | Real SD 1.5 LoRA loop and small offline test configuration, memory management, progress and complete checkpoints. |
| `evaluation` | Fresh adapter loading, paired generation, CLIP diagnostics, quality-policy evaluation and static comparison reports. |
| `common` / `config` | Canonical JSON/SHA-256, atomic file publication, structured errors and validated settings. |

SQLite uses short write transactions with `BEGIN IMMEDIATE`, foreign keys, a lock wait timeout and full synchronization. It is a single-host durable queue, not an emulation of PostgreSQL row locking. Each process opens its own connections. Only one supervisor dispatches tasks. Local files use internal keys, temporary writes and atomic publication; database records expose only completed artifact references.

Logical records include datasets/files, jobs, stage tasks, attempts, models, idempotency requests and scheduler state. UUIDs identify public resources; UTC timestamps describe events; hashes are SHA-256. A job stores its fully expanded training, caption, data and evaluation settings. Attempt records link a task to a token, process identity, device, progress, deadline and checkpoint/output references. One model record is permitted per job.

## 2. Data preparation and captions

Default limits: 100–1,000 declared images, at least 100 valid unique images after cleaning, 20 MiB per file, 2 GiB per dataset, 40 million decoded pixels per image and a shortest side of 256 pixels. These are configurable service limits, not measured capacity. A separate owner storage quota bounds retained uploads.

Allowed decoded formats are JPEG, PNG and static WebP. Decode failures, animated images and exceeded limits are rejected. Normalize EXIF orientation and convert to RGB; alpha composites onto white. Hash canonical dimensions and pixels for exact deduplication. Store normalized images under generated names and record per-file rejection, duplicate and warning reasons.

Near-duplicate grouping uses a 64-bit perceptual hash with default Hamming distance at most four; connected components stay on one side of the split. A deterministic seed controls group assignment towards 90/10. Require at least 80 training images, ten validation images and two independent groups on each side. Infeasible grouping yields `DATASET_UNSUITABLE`; insufficient valid unique images yields `DATASET_TOO_SMALL`.

Blur, brightness and contrast are warning diagnostics by default. Artistic blur or low contrast is not silently removed. Training defaults to aspect-preserving resize and center crop; random crop and horizontal flip are explicit switches, color jitter is disabled. Validation has no random augmentation. Per-sample/epoch seeds make augmentation reproducible after resume.

User captions take precedence. Missing captions use a generic image template or optional BLIP. The final caption associates the content with the configured style trigger. Captions must be nonempty bounded text without control characters; text is never executed. BLIP is loaded only when needed, with model provenance recorded. Explicit BLIP failure is not silently replaced by a template.

Prepared and training-input manifests preserve split membership, file identities, checksums, preprocessing version, seed, caption source and the parent manifest digest. Training, checkpoint compatibility and evaluation refer to this frozen chain.

## 3. Training and evaluation

SD 1.5 is the supported real model family. VAE, text encoder and UNet base weights remain frozen; attention LoRA weights alone are optimized. The loop encodes images to latents, samples noise and timesteps, constructs the configured scheduler target, predicts it and updates the adapter. Loss is a training-health signal, not a quality gate.

Defaults: resolution 512, rank/alpha 4/4, batch size one, four accumulation microsteps, AdamW at `1e-4`, 500 optimizer steps, FP16 on CUDA, gradient checkpointing, gradient norm clipping at one and seed 42. Checkpoint every 50 optimizer steps and at completion. CPU test profiles use a randomly initialized Diffusers `AutoencoderKL`/`UNet2DConditionModel`/`DDPMScheduler` graph with PEFT LoRA at 16/32 resolution and FP32; they do not download pretrained weights and their artifacts remain test-only. GPU OOM produces an actionable failure without altering resolution or optimizer.

Checkpoints are committed only at optimizer boundaries. They include adapter, optimizer, LR scheduler, applicable AMP scaler, RNG state, global step and sample position. A compatibility key binds frozen inputs/captions, model revision or content, configuration, precision, accumulation, augmentation and implementation version. Partial, corrupted or incompatible checkpoints are rejected. API users cannot upload arbitrary serialized checkpoints; the trusted operator CLI can resume local pipeline-produced checkpoints.

Evaluation reloads the exact saved adapter and generates a paired frozen-base/LoRA suite under identical settings. Default generation uses 20 prompts × two seeds per variant; smoke testing reduces this to two prompts × one seed per variant. CLIP text alignment uses normalized text/image features; held-out similarity, pairwise output diversity and maximum training-image similarity are diagnostic dimensions. The report records metric/model provenance, adapter/input hashes, sample counts, technical outcome and policy status. Small samples do not establish statistical significance.

The default policy is uncalibrated. Technical success registers an `UNVERIFIED` model and ends the job as `COMPLETED_UNVERIFIED`. `READY` requires technical success and a passing versioned calibrated policy with evidence. A failed calibrated policy yields `QUALITY_REJECTED`. Test-only models never become READY. A/B reports provide paired images and per-pair score differences; the comparison API is also available as an operator CLI for two compatible adapters.

## 4. API contract and state model

All resource endpoints require `Authorization: Bearer <key>`. Configured keys map to owner IDs; cross-owner requests return `404`. Health endpoints expose bounded operational status; `/metrics` requires an operator-authorized key. There is no account registration or external identity-provider integration.

Dataset states: `UPLOADING → VERIFYING → COMPLETED | INVALID`. Job states: `ACCEPTED`, `RUNNING`, `CANCEL_REQUESTED`, `CANCELLED`, `FAILED`, `QUALITY_REJECTED`, `COMPLETED_UNVERIFIED`, `READY`. Stages are `PREPARE`, `CAPTION`, `TRAIN`, `EVALUATE`, `PUBLISH`; verification is separate dataset work. Task states distinguish pending, running, retries and terminal completion.

| Method and path | Request and behavior |
|---|---|
| `POST /v1/datasets` | File metadata list: name, size_bytes, sha256, mime_type and optional caption. Returns dataset ID and file upload identifiers. |
| `PUT /v1/datasets/{id}/files/{file_id}` | Raw file bytes; streaming size/hash validation; repeated matching upload is safe; frozen data cannot be replaced. |
| `POST /v1/datasets/{id}/complete` | Empty object; freeze recorded objects and queue verification; return 202. |
| `GET /v1/datasets/{id}` | State, verification counters and actionable errors. |
| `GET /v1/training-profiles` | Available real or explicitly enabled test profile with allowed parameter overrides. |
| `POST /v1/training-jobs` | dataset_id, profile identifier, trigger_token and supported training overrides; atomic admission returns 202. |
| `GET /v1/training-jobs/{id}` | Job and stage state, attempts, progress, timestamps, error and model reference when available. |
| `POST /v1/training-jobs/{id}/cancel` | Persist cancellation; terminal publication cannot be undone by late cancel. |
| `GET /v1/training-jobs/{id}/evaluation` | Owner-scoped evaluation report; unavailable reports return conflict. |
| `GET /v1/models/{id}` | Model metadata, provenance, quality status and artifacts. |
| `GET /v1/models/{id}/download` | Owner-authorized artifact delivery; UNVERIFIED requires `allow_unverified=true`. |
| `GET /health/live`, `/health/ready` | Process liveness and dependency/worker readiness. |

Mutation endpoints use `Idempotency-Key`, scoped to owner, method and concrete route; identical requests replay their original result and changed bodies conflict. File-upload idempotency is tied to file ID/checksum. Upload names are never concatenated into paths. Requests cannot select arbitrary server paths, base-model code or remote URLs.

Errors have `error.code`, `message`, `retryable`, `details` and `request_id`. Typical categories: 401 authentication; 404 missing/unauthorized resource; 409 invalid state or idempotency conflict; 422 unusable configuration/data; 429 admission/storage quota; 503 unavailable dependency. FastAPI's generated OpenAPI is the executable schema; see the [run/API guide](04-running-and-api.md) for a reproducible client.

## 5. Scheduling, recovery and operational bounds

GPU devices are discovered by UUID. Configured UUIDs must be unique and available; CPU fake slots require an explicit test setting. Each physical GPU has one shared OS lock used by both worker and direct CLI GPU commands. The supervisor singleton lock prevents competing local dispatchers.

Dispatch uses the weighted `EVALUATE, TRAIN, CAPTION, TRAIN` cycle and persisted owner rotation. With other eligible owners, one owner cannot take all newly available slots. CPU stages use a separate bounded pool. No evaluation GPU is permanently reserved.

Default heartbeat is ten seconds, lease 60 seconds, per-stage limit 3,600 seconds and cumulative GPU-stage budget 7,200 seconds. These are configurable protective defaults, not throughput estimates. Each task has at most three infrastructure attempts. Cancellation, deadline or lost control heartbeat initiates termination of the exact stage process group. A process identity includes PID and start time to avoid killing a reused PID. Device reuse requires confirmed process exit and lock availability.

All canonical progress/output updates check the active attempt token and permissible job/task state. Completion atomically binds output and enqueues the unique successor. Publication checks the adapter, report and input binding and registers a unique model. Work may be retried; publication remains unique.

The API supports structured request/job logs and an authenticated metrics endpoint. GPU memory and training measurements live in progress/artifact reports; unmeasurable values are explicit rather than zero. No automatic garbage collection removes referenced objects. Retention and consistent backups are operator responsibilities in this minimal implementation; multi-host failover and a production calibrated style evaluator remain outside the implemented boundary.

## 6. Acceptance scenarios

Critical tests cover malformed/oversized images, exact duplicates and group leakage; deterministic augmentation and split; caption priority and explicit BLIP failures; frozen base weights and real LoRA gradients; complete checkpoint restoration and incompatibility rejection; metric arithmetic and A/B pairing; tenant isolation and idempotency; admission limits and fair logical slots; stale-result rejection, cancellation and worker recovery; unique publication and explicit unverified downloads.

Offline CPU tests use synthetic image fixtures and a locally initialized Diffusers/PEFT tiny graph with random weights. They do not download pretrained model weights, and the resulting artifacts are test-only. The user-run RTX 4060 Ti test performs ten real optimizer steps with interruption at five, resumes, exports/reloads an adapter, generates paired images and computes CLIP metrics. GPU outcomes remain unverified until that command is actually run. Container build/runtime status is reported separately when Docker daemon access is unavailable.
