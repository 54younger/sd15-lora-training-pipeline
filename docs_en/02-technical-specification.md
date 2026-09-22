# Part 1/2 — Technical specification, contracts and acceptance

This document describes contracts present in the source, not the README's aspirations. Field implementations are in [api.py](../src/lora_pipeline/api.py), [store.py](../src/lora_pipeline/store.py), [config.py](../src/lora_pipeline/config.py), and [common.py](../src/lora_pipeline/common.py); unit/integration coverage is in [tests/](../tests/).

## 1. Components, persistence and artifact lineage

See the [Web Studio guide](06-web-studio.md) for browser operation and new endpoints. Default `execution_mode=auto` preserves the automatic orchestration below. Manual jobs enter `WAITING_FOR_USER` after input preparation, training and evaluation; an idempotent `advance` starts `waiting_for_stage`. The base profile remains immutable; manual training/evaluation overrides freeze separately when their stage is first enqueued.

| Component | Current responsibility | Evidence |
|---|---|---|
| Web | English four-step workflow, caption editing, bounded uploads, parameter forms, status/history, artifact previews/downloads and same-origin Nginx proxy | [frontend](../frontend/) |
| API | Bearer owner auth, bounded streaming uploads, request validation, required mutation idempotency, status/cancel/download, health and admin metrics | [api.py](../src/lora_pipeline/api.py) |
| Store | SQLite schema/short transactions, quotas, queue, attempts/leases/fencing, owner isolation, unique model | [store.py](../src/lora_pipeline/store.py) |
| Worker | Singleton supervisor, dataset VERIFY, stage children, heartbeat/watchdog, cancellation, recovery and resource dispatch | [worker.py](../src/lora_pipeline/worker.py) |
| Data/Captions | Image decode/normalization, deduplication/group split, user/template/BLIP captions | [data.py](../src/lora_pipeline/data.py), [captions.py](../src/lora_pipeline/captions.py) |
| Training/Evaluation | SD15 LoRA, tiny offline graph, complete checkpoint/resume, adapter reload, CLIP diagnostics and A/B | [training.py](../src/lora_pipeline/training.py), [evaluation.py](../src/lora_pipeline/evaluation.py) |
| Common/Config | Canonical JSON, SHA-256, atomic writes, structured PipelineError and frozen settings | [common.py](../src/lora_pipeline/common.py), [config.py](../src/lora_pipeline/config.py) |

SQLite is a single-host queue: each process uses its own connection, BEGIN IMMEDIATE, foreign keys, short transactions and synchronous=FULL. Model execution and large-file IO are outside DB transactions. Main tables are datasets/dataset_files, jobs/stage_tasks/task_attempts, models, idempotency and scheduler_state. Public resources use UUIDs; times are Unix UTC seconds; digests are SHA-256. Artifact references become visible only after writes complete, and stale attempts are fenced.

The minimum persistence schema for review is below (other timestamps/error fields are omitted). These are internal SQLite tables, not the public API schema; the external dataset files array is assembled by store.dataset(), and object_key, fencing tokens and similar internal fields are not public contract fields:

| Table | Key fields and relation | Constraint / purpose |
|---|---|---|
| datasets | id, owner_id, state, declared_count, frozen_manifest_path/sha256 | Owner scoped; dataset_files.dataset_id foreign key; upload/verification state |
| dataset_files (internal upload metadata) | id, dataset_id, object_key, size_bytes, sha256, uploaded, verification_status | object_key UNIQUE; cascade from dataset; binds bytes and source digest |
| jobs | id, owner_id, dataset_id, state, current_stage, profile_json, training_overrides_json, model_id | Dataset foreign key; profile/input/training/evaluation JSON are frozen snapshots |
| stage_tasks | id, job_id, candidate_index, stage, state, active_token, lease_expires_at, output_json | UNIQUE(job_id,candidate_index,stage); one successor per stage |
| task_attempts | id, task_id, attempt_no, fencing_token, gpu_slot, process identity, progress_json | UNIQUE(task_id,attempt_no) and UNIQUE(task_id,fencing_token); stale tokens cannot write |
| models | id, job_id, owner_id, state, adapter_path, adapter_sha256, manifest_path/sha256, report_json, test_only | UNIQUE(job_id); PUBLISH registers only checksum-verified models |

idempotency uses (owner_id, route, key) as the primary key and stores request_sha256/response; scheduler_state stores cursors, owner rotation, leader epoch and worker heartbeat.

## 2. Data and manifest schema

The dataset-file metadata request is:

    {
      "name": "leaf-001.png",
      "size_bytes": 123456,
      "sha256": "64 lowercase hex chars",
      "mime_type": "image/png",
      "caption": "optional printable text <= 512 chars"
    }

POST /v1/datasets accepts {"files": [...]}. The count is 100–1,000; current DataConfig service defaults are 20 MiB per file, 2 GiB total, 40 MP and shortest side 256. Only JPEG, PNG and static WebP are accepted; animation, decode, size and digest failures are rejected or recorded as per-file rejection. prepare_dataset() writes prepared.json with:

- schema_version, preprocessing_version, grouping_version, dataset_id and complete config;
- train[] / validation[] entries with id/name/path/sha256/group_id and warnings;
- statistics: submitted, accepted_unique, train/validation, groups, rejected, exact_duplicates and total_source_bytes;
- rejected[] and duplicates[].

Normalized PNG paths contain a pixel-hash prefix and the final encoded-PNG-byte SHA-256; the digest is recorded in the entry, and a repeated prepare never overwrites a published image artifact. training-input.json also records prepared_manifest_path/sha256, captioning mode/trigger/source counts and train/validation entries. Training, evaluation and checkpoints validate this frozen chain; drift returns CHECKSUM_MISMATCH or INPUT_INCOMPATIBLE with path, expected_sha256 and actual_sha256 when available.

The default split is approximately 90/10 and group-exclusive, requiring at least 80 train images, 10 validation images and two groups per side. These are configurable constraints, not a promise that every input is feasible. Blur/brightness/contrast are warnings; training defaults to aspect-preserving resize plus center crop, random crop/horizontal flip must be explicit, and validation has no random augmentation.

## 3. Profiles, configuration and training/evaluation artifacts

The actual profile IDs are:

- local-sd15-v1 / key style-lora: real SD 1.5, default 512 resolution, rank/alpha 4/4, batch 1, accumulation 4, learning rate 1e-4. The smoke config uses 10 steps (configs/sd15-smoke.json), FP16 CUDA; service TrainConfig defaults max_steps to 500, not a measured quality threshold.
- local-tiny-v1 / key tiny-test: exposed only with enable_test_backend=true; CPU FP32, 16px randomly initialized Diffusers/PEFT graph, always test-only artifacts.

The expanded profile/data/evaluation/caption/limits snapshot is frozen at admission. API training overrides are limited to max_steps, learning_rate, rank, lora_alpha, checkpoint_every, seed, batch_size and gradient_accumulation_steps. The API cannot select a base model, device, precision, path, remote URL or arbitrary checkpoint.

A training result includes at least kind=training-result, state, adapter_path/adapter_sha256, manifest_path/manifest_sha256, input_manifest_sha256, base revision/fingerprint, config, global_step, elapsed_seconds and test_only. A checkpoint contains adapter, optimizer, LR scheduler, applicable AMP scaler, RNG, sample cursor and compatibility key. It is saved only at the configured checkpoint boundary; corrupt, partial or input/config/base mismatches are CHECKPOINT_CORRUPT or CHECKPOINT_INCOMPATIBLE.

Frozen revision contract: SD15 training must resolve an immutable snapshot revision and write revision plus base fingerprint into training-result/checkpoint; evaluation accepts only the same revision/fingerprint, while drift or an unpinned snapshot is BASE_SNAPSHOT_MISMATCH or BASE_REVISION_UNPINNED. Tiny tests use an explicit local identity and never pretend to be an SD15 revision.

Base-snapshot failure diagnostics retain actionable fields while redacting model/revision/cache URLs and tokens; HF tokens and raw exception URLs are not written into error details. See the base-model diagnostic cases in [test_training.py](../tests/test_training.py).

An evaluation report contains technical_pass, quality_status, quality_failures, metrics, test_only, adapter/input/base identity, config, paired_outputs and report_path/report_sha256. SD15 metrics are CLIP prompt score, held-out similarity, diversity, maximum training similarity, baseline and paired count; tiny metrics explicitly report unavailable. A/B uses identical prompt/seed/inference settings and rejects adapters with different base revision, resolution or input manifest.

The quality-policy shape and directions are implementation facts, not business defaults:

    {
      "version": "operator-defined",
      "calibration_reference": "operator-supplied evidence identifier",
      "bounds": {
        "clip_prompt_score": {"min": 0.0},
        "heldout_similarity": {"min": 0.0},
        "diversity": {"min": 0.0},
        "max_train_similarity": {"max": 1.0}
      }
    }

The example numbers above are placeholders only; runtime bounds must be finite numbers. Code checks that the reference is non-empty, bounds are present and numeric, min is no greater than max, and values obey the direction. It does **not** read, validate or prove that the external calibration_reference exists, and the repository contains no calibrated data or measured thresholds. Default null yields UNCALIBRATED; technical success plus a non-test backend and policy PASS yields READY; PASS for test-only remains UNCALIBRATED. A complete policy with an unavailable metric is FAIL; a missing/incomplete policy is UNCALIBRATED. FAIL after EVALUATE enters QUALITY_REJECTED and does not enqueue PUBLISH.

## 4. State machine and actual store checks

| Work item | State/order | Worker/store check |
|---|---|---|
| Dataset VERIFY | UPLOADING → VERIFYING → COMPLETED/INVALID | complete_dataset requires every file uploaded=1; worker checks object existence, size and source SHA; finish_verification writes counts, error and frozen manifest. VERIFY is dataset work, **not a stage task**. |
| Job stages | PREPARE → CAPTION → TRAIN → EVALUATE → PUBLISH | claim_task only claims job ACCEPTED/RUNNING and task PENDING/RETRY_WAIT; each task has an attempt token/lease. |
| PREPARE | task PENDING/RUNNING/SUCCEEDED... | Builds trusted entries from store object keys and writes the prepared manifest; if every train caption exists it may create input in the same completion, otherwise it creates CAPTION. |
| CAPTION | same task states | Reads mode/device/model/revision from the frozen job profile; user text wins and BLIP failure never silently falls back to a template. |
| TRAIN | same task states | Reads only frozen input/profile/overrides; a physical GPU uses assigned cuda:0, a test slot forces tiny CPU; saves checkpoints and progress. |
| EVALUATE | same task states | Requires training/input; rechecks adapter/input/base. Quality FAIL writes QUALITY_REJECTED and creates no PUBLISH task. |
| PUBLISH | independent successor task | Requires active token, live lease, existing checksum-valid adapter/manifest/report/input binding; models.job_id is UNIQUE and terminal state is READY or COMPLETED_UNVERIFIED. |

Terminal job states are READY, FAILED, QUALITY_REJECTED, CANCELLED and COMPLETED_UNVERIFIED. Cancellation persists CANCEL_REQUESTED and becomes CANCELLED only when no executor remains. Lease expiry first checks the PID/start-time/PGID process identity; an old fencing token cannot update canonical progress/output.

## 5. REST API, idempotency and schema

All resource endpoints use Authorization: Bearer <key>; cross-owner lookups are masked as 404 NOT_FOUND. Current endpoints:

| Method | Path | Input/output |
|---|---|---|
| POST | /v1/datasets | files[] metadata; 201 returns dataset/file IDs. **Idempotency-Key required.** |
| PUT | /v1/datasets/{dataset_id}/files/{file_id} | Raw bytes, streaming size/hash validation, 204. Retry is safe by file ID and checksum; it is upload identity logic, **not the generic idempotency table and does not require Idempotency-Key**. |
| POST | /v1/datasets/{dataset_id}/complete | Empty JSON; freeze and queue VERIFY, 202 + Location. **Key required.** |
| GET | /v1/datasets/{id} | State, counts, errors and optional files. |
| GET | /v1/training-profiles | Actual profile IDs, frozen config and limits. |
| POST | /v1/training-jobs | dataset_id/profile_revision_id/trigger_token/training_overrides; 202 + status URL. **Key required.** |
| GET | /v1/training-jobs/{id} | Job/stages/attempts/progress/error/model. |
| POST | /v1/training-jobs/{id}/cancel | Empty JSON; 202 CANCEL_REQUESTED. **Key required.** |
| GET | /v1/training-jobs/{id}/evaluation | Report; unavailable report is a conflict. |
| GET | /v1/models/{id} | Owner-scoped model/report/provenance/state. |
| GET | /v1/models/{id}/download?allow_unverified=true | Adapter bytes; an UNVERIFIED model without opt-in is 409. |
| GET | /health/live, /health/ready | Liveness; readiness checks DB, storage and worker heartbeat and returns 503 when not ready. |
| GET | /metrics | Admin-key Prometheus text; non-admin is 401. |

Required POST mutations store (owner, concrete route, Idempotency-Key) and request hash in idempotency. Same key and body replays the original response; a changed body is 409 IDEMPOTENCY_KEY_REUSED. Reads, health and metrics do not require it. PUT relies on file-object byte identity and frozen-state checks. API errors have this fixed shape:

    {"error":{"code":"...","message":"...","retryable":false,"details":{},"request_id":"uuid"}}

PipelineError exceptions are wrapped by api.py's middleware/exception handler into the error object above and mapped by code: 401 for UNAUTHORIZED, 404 for NOT_FOUND, 409 for listed state/idempotency conflicts, 429 for ADMISSION_LIMIT, 503 for GPU_UNAVAILABLE or WORKER_ALREADY_RUNNING, and 422 for other PipelineError values including storage quota. Pydantic/FastAPI request validation errors (for example, a wrong body field type) instead use the default RequestValidationError response: HTTP 422 with a detail array, not the unified error wrapper. FastAPI's generated OpenAPI is the executable schema; a future PostgreSQL API is not a current contract.

## 6. CLI errors, progress, logs and acceptance

On success CLI stdout is final JSON; train/evaluate/compare progress is line-oriented stderr (phase, current/total and percentage). Failure exits 1 and prints:

    {"error":{"code":"CHECKSUM_MISMATCH","message":"...","details":{}}}

CLI errors do not carry API HTTP status/retryable/request_id fields. The worker emits structured stage_started/stage_completed/stage_failed events and writes attempt progress to SQLite. Training progress includes global_step, loss, samples_processed, samples_per_second, checkpoint_seconds, CPU RSS and CUDA allocated/reserved when available; unavailable measurements stay null/not_measured. [test_cli.py](../tests/test_cli.py) and [test_observability.py](../tests/test_observability.py) verify the output boundary.

Acceptance must separate three results: offline pytest/tiny proves logic, LoRA/resume and contracts; Docker Compose proves image/service configuration; real SD15 with 2–4 physical GPUs, CLIP and performance benchmarks must run on target hardware. Unrun fields remain not_measured; CPU tiny values must not be presented as GPU quality or throughput. Entry points and success criteria are in the [run/API guide](04-running-and-api.md) and [performance definitions](05-performance-benchmarks.md).
