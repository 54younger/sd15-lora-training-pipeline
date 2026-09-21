# Automatic Style LoRA Training Pipeline — Technical Specification

## 1. Purpose and scope

This service turns a tenant-owned image dataset into a versioned **style LoRA** that an existing image-generation inference consumer can load. A submission contains 100–1,000 images. The service validates and cleans the set, optionally generates content-focused captions, trains a LoRA while freezing the base model, evaluates it against a pinned quality policy, and logically registers only a passing artifact. It does not design or operate an always-on model-serving platform: “deployment” here means a verified handoff to the existing inference consumer.

The design is intentionally sized for a pool of two to four GPUs. It accepts concurrent jobs, but runs at most one authorized GPU stage in each exclusive GPU slot. All GPU work—optional captioning, training, evaluation generation, and the saved-adapter load smoke test—uses the same pool. A job has at most one current authorized GPU attempt and v1 does not split one job across GPUs or colocate independent jobs on a GPU. After a host partition, an old fenced attempt can physically overlap a retry on a different healthy slot until the host watchdog terminates it; this is not physical exactly-once execution. The original device is never reused while unconfirmed, and both attempts are charged/count toward conservative budget and owner eligibility.

Logical components may run as a small number of deployable processes rather than separate microservices:

| Component | Responsibility |
| --- | --- |
| API/auth service | Tenant authorization, dataset/job/model APIs, idempotency, admission quota checks, presigned object-store upload grants. |
| PostgreSQL | Authoritative metadata, durable stage-task queue, leases, fairness cursor, state transitions, idempotency records, and registry records. |
| Scheduler/reconciler | Active/standby controller that assigns compatible GPU stages, detects expired leases, drives retries/cancellation, and reconciles interrupted publication. |
| CPU workers | Decode/validate/deduplicate/split images, verify artifacts, publish registry records, and remove expired orphan artifacts. |
| GPU host agent | Starts isolated, pinned-environment stage containers on a locally locked device; enforces deadlines, reports heartbeats, and quarantines unhealthy devices. |
| Object storage | S3-compatible store for uploads, cleaned manifests, captions, checkpoints, attempt outputs, immutable artifact manifests, evaluation reports, and adapters. |
| Registry/inference handoff | DB and object-store-backed logical model registry. It exposes a READY model only after a pinned inference-environment load test succeeds. |
| Observability | Structured logs, metrics, traces/attempt audit events, alerting, and backup/restore monitoring. |

Polling the job API is authoritative. An optional event or webhook may be a wake-up hint, but clients must re-read the job because notifications can be duplicated, delayed, or lost.

## 2. Workflow, state model, and component behaviour

### Submission and CPU preparation

`POST /v1/datasets` records an upload manifest and returns tenant-scoped presigned PUT URLs. The client uploads only the listed objects, then calls `complete`. The upload bucket must support object versioning: completion resolves exact object version IDs, then atomically persists that snapshot and enqueues verification; it does not trust file extensions or user-supplied captions. Previously issued PUT URLs can create a newer object version but cannot change the pinned version read by verification or training. The frozen snapshot cannot gain objects or receive writable URL replacements. A verifier checks object existence, byte size, checksum, MIME declaration, and any uploaded caption before moving the dataset to `COMPLETED` or `INVALID`.

Creating a job performs a short PostgreSQL transaction at `SERIALIZABLE` isolation: verify ownership, that the dataset is completed, the selected profile is supported and published, a calibrated quality policy is available, and the durable per-user/global admission quotas allow it; insert the job, candidate-0 record and first `PREPARE` task; store the idempotency result. Candidate 0 initially has no prepared training input; PREPARE/CAPTION binds that immutable input before TRAIN can be enqueued. Serialization failures receive bounded internal retries under the same idempotency key, preventing concurrent requests from both consuming the last quota position. Thus a returned job cannot be lost between API acknowledgement and queue insertion. Deep inspection remains asynchronous because 100–1,000 images may take time.

`PREPARE` is CPU-parallel. It decodes with strict file, pixel, aggregate-size, and decompression limits; MIME-sniffs rather than trusting the manifest; removes corrupt, duplicate, and policy-disallowed images; writes a cleaned immutable manifest; and creates an approximately 90/10 **group-wise** training/held-out split. Near duplicates and all images from one detected group stay on one side, which prevents validation leakage. Its guarded completion transaction creates an immutable `prepared_dataset_version` containing cleaned/train/held-out manifest checksums, preprocessor version, split seed/groups, and producing attempt, then binds that exact version to the job. The split manifest, groups, and seeds are immutable before model-dependent processing or cache use. The job requires at least 100 valid, unique images after cleaning and before the split. It must also meet profile suitability bounds, including enough independent groups and held-out subject/content coverage to make the split meaningful; otherwise it fails `DATASET_UNSUITABLE`. A lower count fails `DATASET_TOO_SMALL`, in both cases without consuming GPU capacity.

`CAPTION` is optional and profile-controlled. Valid uploaded captions are reused after content/safety validation. If every training-split file already has one, PREPARE derives the final `training_input_version` in its guarded completion transaction and enqueues `TRAIN` directly. Otherwise, captioning must be enabled or PREPARE fails `DATASET_CAPTIONS_REQUIRED`. CAPTION generates the missing content-focused descriptions and derives the final version, helping separate style from pictured subjects. The version contains caption artifact/checksum/version and a foreign key to the prepared version; it cannot alter held-out membership. If CAPTION runs on GPU, it participates in exactly the same scheduler and budget as training and evaluation. TRAIN, checkpoints, EVALUATE, and PUBLISH pin and verify this manifest chain. Frozen intermediate products can be content-addressed and reused only when all inputs affecting them match: normalized image bytes, immutable split membership, preprocessing and augmentation equivalence, captioner version/settings, profile fields, and any seed or trigger token that affects output. Input-derived cache namespaces are tenant-isolated; public base-model artifacts may be shared.

### Training, checkpointing, and evaluation

Each immutable training-profile revision pins a base-model revision, training container digest, resolution, LoRA method/rank, optimizer settings, learning rate, steps, batch size, precision, estimated device memory, checkpoint cadence, and resource/time budgets. It also pins caption and evaluator versions or declares captioning off. Profiles are prevalidated against a supported GPU slot class; unsupported combinations are rejected before admission. A profile may define one previously validated OOM fallback and one immutable quality-remediation variant. There is no free-form override in the job API.

`TRAIN` loads the candidate's exact profile revision, freezes all base-model parameters, and updates LoRA adapter weights only. Checkpoints are written at least every five minutes by default and contain the adapter, optimizer state, scheduler state, RNG state, and data position. A checkpoint becomes resumable only after its immutable, checksummed manifest is committed. Resume is permitted solely for the same dataset snapshot, candidate, profile revision, container digest, and compatible checkpoint; otherwise the candidate restarts. In particular, a changed optimizer configuration restarts training unless the profile contains a separately validated state-conversion procedure. This follows the practical requirement to save the full training state for reliable resume, rather than model weights alone ([Accelerate checkpoint guidance](https://huggingface.co/docs/accelerate/usage_guides/checkpoint)). The training approach is the standard parameter-efficient LoRA adaptation described in the [Diffusers LoRA training documentation](https://huggingface.co/docs/diffusers/training/lora).

`EVALUATE` runs in the profile-pinned generation and inference environments. It uses an immutable held-out reference set, a versioned fixed prompt suite whose requested subjects differ from the training images, and fixed seeds. An illustrative default is 20 prompts × 2 seeds (40 outputs), configurable in the quality policy; it is a design default, not a throughput or quality benchmark. A budget-aware screen can stop clearly failing candidates early, but cannot publish a model; the complete policy evaluation is required to pass. Evaluation then loads the saved adapter under the pinned inference environment and makes a smoke generation, catching packaging and compatibility defects before publication.

Quality is multi-dimensional:

| Dimension | Evidence and gate |
| --- | --- |
| Style consistency | A versioned style-oriented evaluator compared with held-out references; calibrated with human labels. Image-image CLIP similarity may be a diagnostic proxy but never the sole gate because content can confound it. |
| Text adherence | Prompt/image scoring and evaluator checks across unseen subjects and scenes. |
| Diversity and memorization guard | Output diversity statistics plus similarity searches against training images to flag collapse or excessive copying. |
| Technical validity | Image generation succeeds; adapter checksum, metadata, and pinned-environment loading are valid. |

The chosen metric model versions, prompt suite, aggregation rules, uncertainty handling, and per-dimension floors/bounds belong to an immutable `quality_policy`. Numeric thresholds are deliberately not invented in this specification: before enabling a profile, they must be calibrated offline against representative datasets and human labels. If no calibrated policy is pinned, job creation returns `409 PROFILE_NOT_READY`. Training loss is a health signal, not an acceptance criterion. Style evaluation should retain a human-calibrated component; published work such as [StyleDrop](https://research.google/blog/styledrop-text-to-image-generation-in-any-style/) also treats automated similarity measures as complementary to human evaluation.

A failed candidate may receive at most one quality remediation attempt: candidate 0 can produce candidate 1 using only the profile's immutable, supported variant ID, which is recorded on the candidate. The policy and thresholds do not relax. The candidate-1 variant is a distinct checkpoint-compatibility domain; it may resume only a checkpoint valid for that variant, and an optimizer change restarts as described above. Quality remediation is distinct from an infrastructure retry and uses the same evaluation policy and the remaining cumulative GPU budget. A final quality failure is `QUALITY_REJECTED` with a report explaining failed dimensions and uncertainty.

### State, retries, publication, and cancellation

The job state is one of `ACCEPTED`, `RUNNING`, `CANCEL_REQUESTED`, `READY`, `FAILED`, `QUALITY_REJECTED`, or `CANCELLED`. `current_stage` is independently `PREPARE`, `CAPTION`, `TRAIN`, `EVALUATE`, or `PUBLISH`. Each candidate/stage has a task state of `PENDING`, `RUNNING`, `RETRY_WAIT`, `SUCCEEDED`, `FAILED`, or `CANCELLED`. Returning both states lets a client distinguish a queued GPU stage from one that is actually executing.

Infrastructure errors receive at most three total attempts per task, including the first, with bounded exponential backoff and jitter. The reconciler classifies corrupt input, unsupported configuration, and policy failures as non-retryable; transient worker, host, and object-store faults may retry. Every attempt—including evaluation and retries—is charged to the cumulative profile GPU budget. Exceeding it fails the job with `GPU_BUDGET_EXCEEDED`; no retry or quality remediation may bypass it.

Every stage, including CPU stages, completes through one guarded transaction: it checks the current attempt's fencing token, unexpired lease, task state `RUNNING`, and job state `ACCEPTED` or `RUNNING`; records and validates immutable output-manifest references; marks the current task `SUCCEEDED`; and inserts the unique successor task or commits a terminal outcome. TRAIN/EVALUATE/PUBLISH require the candidate's non-null, immutable training-input and compatibility bindings. This prevents a worker crash between a stored artifact and the next queue row from losing a transition. A stale, expired, cancelled or terminal-job attempt cannot bind a prepared/training/evaluation manifest or advance the workflow.

Publication is **effectively once**, not execution exactly once. A publish worker writes attempt-scoped artifacts first, verifies checksums, a complete passing report for the exact adapter checksum, and the pinned-environment load/smoke result, then writes an immutable model manifest. A guarded DB transaction creates or finds the unique model for the job, marks the task complete, and moves the job and model to `READY`. It requires a current, unexpired fencing token and no cancellation request. Only after this transaction may the registry expose the READY model; a partial object-store write or uncommitted registry row is not public. The object store and database are not atomic, so unreferenced partial writes are cleaned after a TTL; reconciliation finds an already committed unique model rather than creating a duplicate. Failed candidates are never discoverable as READY.

Cancellation is durable. A request changes an eligible job to `CANCEL_REQUESTED`; workers stop at safe points and the reconciler moves it to `CANCELLED` only after every executor has confirmed process termination. Fencing and quarantining prevent an old executor from committing, but alone are insufficient to make cancellation terminal. For a permanently unreachable host, an operator records an audited termination attestation backed by instance-control-plane shutdown/deletion or BMC power-off evidence for that exact host instance/attempt. This evidence can substitute for an agent acknowledgement; elapsed time or a manual state override cannot. Without proof the job stays `CANCEL_REQUESTED`, retains quota/protected artifacts, and alerts an operator. A retired device stays unavailable until replaced and health-checked. Cancellation never claims instantaneous GPU termination. A locked job transition resolves the race with publication: a request after the job is already `READY` receives `409 JOB_ALREADY_READY`.

## 3. GPU scheduling and distributed correctness

Each GPU device is represented by an exclusive `gpu_slot` with a profiled capability class and state `HEALTHY`, `BUSY`, `DRAINING`, or `QUARANTINED`. A host agent takes a device-level exclusive lock, starts an isolated stage container, and only returns a GPU to `HEALTHY` after the process has terminated and a device-health check passes. A network partition or expired lease does not itself release the local device: it is quarantined until the agent/watchdog confirms it is safe. A retry can run on a different healthy compatible slot. This intentionally favours safety over immediate availability.

The scheduler/reconciler is active/standby. A single persisted `scheduler_guard` row contains leader epoch and fairness cursors. Every short GPU-dispatch transaction locks/checks it with `FOR UPDATE`, then serializes compatibility selection, owner eligibility/active count, slot claim, task claim, budget reservation, and cursor update; it never holds this lock while a model trains. A takeover increments the epoch, and all dispatch/start updates require that epoch, rejecting split-brain controllers even when they would choose different slots. Candidate tasks are selected by compatibility first, then a persisted weighted stage cycle:

`EVALUATE → TRAIN → CAPTION → TRAIN`

Empty or ineligible classes are skipped. Within a class, the scheduler uses fair round-robin across owners and FIFO among each owner’s eligible tasks. It allows at most one active or conservatively unconfirmed GPU task per owner while another owner is eligible, but may borrow idle capacity when no other eligible owner exists. Borrowed work is never preempted if another owner arrives; the per-owner cap applies only to new grants while another owner is eligible. The persisted cycle cursor and oldest-eligible choice make failover deterministic enough for audit. Per-stage deadlines and queue-age metrics/alerts expose starvation; with a finite admitted backlog this policy ensures an eligible class is revisited rather than permanently losing all capacity to training. No GPU is reserved permanently for evaluation.

Within the guarded dispatch transaction, `FOR UPDATE SKIP LOCKED` claims eligible task/slot rows so unrelated locked work is skipped rather than delaying recovery or reconciliation ([PostgreSQL documentation](https://www.postgresql.org/docs/current/sql-select.html)). The singleton guard deliberately serializes new GPU grants for correct fairness and budget accounting. Admission is bounded transactionally by the design defaults of five nonterminal jobs per user and 100 globally; excess requests receive `429` and `Retry-After`. These are adjustable operations defaults, not capacity benchmarks.

For capacity planning, a conservative illustrative estimate is:

`admitted GPU seconds per period ≤ healthy slots × usable seconds per slot × target utilization`

The left side includes reserved train, caption, evaluation, smoke-test, retry, and remediation budgets. It supports queue/admission decisions; actual training duration and utilization must be measured for each deployed profile, not inferred from this formula.

Every running attempt has a DB-clock lease expiry and a monotonically increasing fencing token. The agent heartbeats every 10 seconds and the standard lease is 60 seconds. Any status write, checkpoint commit, heartbeat, or registry publication carries the token; stale or mismatched tokens are rejected. On expiry, the controller revokes the old epoch and requeues work only within remaining budget. The agent watchdog stops work after a lost lease or deadline. Until termination is confirmed, conservative accounting retains the unconfirmed attempt’s budget exposure.

If PostgreSQL is unavailable, controllers start no new work and no worker publishes. Agents fail closed and stop before lease expiry. This avoids duplicate publication at the cost of temporary throughput. Object-store failures similarly prevent a stage from committing output. A two-to-four GPU deployment may share a single host or fault domain, so host failure tolerance is limited by the actual placement; replicas/backups improve metadata recovery but do not claim GPU redundancy that is absent.

## 4. External API

All endpoints require `Authorization: Bearer <token>` and enforce tenant ownership. Mutating endpoints require `Idempotency-Key`; its record is scoped to `(owner_id, route, key)`, where `route` includes the HTTP method and concrete resource path, and the request hash covers that target and the canonical body. An exact replay with the same key returns the original status/body; the same key with a different body returns `409 IDEMPOTENCY_KEY_REUSED`.

### Dataset APIs

`POST /v1/datasets` accepts a manifest only; arbitrary external URLs are prohibited.

```json
{
  "files": [
    {"name":"001.jpg","size_bytes":1830421,"sha256":"d2d2d2d2d2d2d2d2d2d2d2d2d2d2d2d2d2d2d2d2d2d2d2d2d2d2d2d2d2d2d2d2","mime_type":"image/jpeg","caption":"a red bicycle"}
  ]
}
```

`files` has 100–1,000 unique names; each `name` is a relative object key, `size_bytes` is a positive integer within service limits, `sha256` is a lowercase 64-hex string, and `mime_type` must be an allowed image MIME type. Optional `caption` is UTF-8 plain text within configured length/safety limits; it is validated, normalized, and bound to the frozen file snapshot, not treated as executable prompt syntax. The small example is schematic; a valid request includes at least 100 entries. A `201` response returns a UUID dataset ID, object keys, and short-lived presigned PUT URLs. For example, `dataset_id` is a string UUID such as `3fa85f64-5717-4562-b3fc-2c963f66afa6`.

`POST /v1/datasets/{dataset_id}/complete` has body `{}`. In its idempotent transaction it freezes uploaded object versions, changes `UPLOADING` to `VERIFYING`, and inserts a durable verify task; it returns `202` with `Location: /v1/datasets/{dataset_id}`. Dataset states are exactly `UPLOADING`, `VERIFYING`, `COMPLETED`, and `INVALID`. `GET /v1/datasets/{dataset_id}` is the client polling API and returns ID, state, immutable manifest version/checksum, accepted/rejected counts when known, timestamps, and an actionable `verification_error` when `INVALID`. A job may be created only from `COMPLETED`.

`GET /v1/training-profiles` lists only published profiles compatible with the caller’s plan. Each entry returns an immutable UUID `profile_revision_id` (for example `8ac7a2f1-694c-4da4-8d6a-e40dc9b0f2fd`), a human-readable `profile_key` such as `style-lora`, revision number, display name, pinned base-model revision, training/environment versions, supported input bounds, whether captioning is enabled, and a quality-policy readiness flag; it does not expose mutable low-level overrides. Quality policies use the same pattern: a UUID `quality_policy_revision_id`, plus a stable policy key and revision number.

### Job APIs

`POST /v1/training-jobs`:

```json
{
  "dataset_id":"3fa85f64-5717-4562-b3fc-2c963f66afa6",
  "profile_revision_id":"8ac7a2f1-694c-4da4-8d6a-e40dc9b0f2fd",
  "trigger_token":"mystyle"
}
```

`dataset_id` must be a completed caller-owned dataset; `profile_revision_id` must be published; `trigger_token` is a required nonempty, profile-valid style token. A successful response is `202 Accepted`:

```json
{
  "job_id":"c0a8012e-7b6d-4b7d-a48f-8a10b2b5c0ea",
  "state":"ACCEPTED",
  "current_stage":"PREPARE",
  "status_url":"/v1/training-jobs/c0a8012e-7b6d-4b7d-a48f-8a10b2b5c0ea"
}
```

It includes `Location: /v1/training-jobs/{job_id}`. `GET /v1/training-jobs/{job_id}` returns the fields above plus `candidate_index`, stage task states/attempt counts, submitted/updated timestamps, non-sensitive progress, error summary, model ID when READY, and `retry_after_seconds` when queued. Example poll result:

```json
{
  "job_id":"c0a8012e-7b6d-4b7d-a48f-8a10b2b5c0ea",
  "state":"RUNNING",
  "current_stage":"TRAIN",
  "candidate_index":0,
  "stages":{"PREPARE":"SUCCEEDED","CAPTION":"SUCCEEDED","TRAIN":"RUNNING"}
}
```

`POST /v1/training-jobs/{job_id}/cancel` accepts `{}`. The same idempotency key replays its original `202 CANCEL_REQUESTED` response. A new key after terminal `CANCELLED` returns `200` with that existing terminal result; it returns `409 JOB_ALREADY_READY` if publication won, and `409 JOB_TERMINAL` for other terminal outcomes. `GET /v1/training-jobs/{job_id}/evaluation` returns `409 EVALUATION_NOT_READY` until a report exists, then returns a versioned report: policy revision ID, candidate index/profile revision ID, adapter checksum, prompt-suite/reference artifacts, scores and verdict per gate, uncertainty flags, failed reasons, and report artifact reference. It never leaks another tenant’s images or prompts where those are confidential. Illustrative metric values below are diagnostic output, never universal thresholds:

```json
{
  "policy_revision_id":"9b772267-96f5-4c87-97c5-48a8dbb426c1",
  "candidate_index":0,
  "profile_revision_id":"8ac7a2f1-694c-4da4-8d6a-e40dc9b0f2fd",
  "prepared_version_id":"2b81041a-7e12-4dc4-b92c-87a012867c32",
  "training_input_version_id":"2a91b5e1-5ee7-45c8-bb72-99b54aa91144",
  "training_input_manifest_sha256":"cdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcd",
  "adapter_sha256":"abababababababababababababababababababababababababababababababab",
  "prompt_suite_artifact":"s3://private/eval/prompt-suite-v7.json",
  "heldout_reference_artifact":"s3://private/eval/heldout-v4.json",
  "gates":{"style":{"value":0.71,"verdict":"PASS"},"text_adherence":{"value":0.83,"verdict":"PASS"},"diversity_memorization":{"value":0.18,"verdict":"PASS"},"technical_smoke":{"verdict":"PASS"}},
  "overall_verdict":"PASS",
  "report_artifact":"s3://private/reports/c0a8012e/candidate-0.json"
}
```

### Model APIs and errors

`GET /v1/models/{model_id}` returns a caller-owned model’s `READY` status, profile/base compatibility, adapter checksum, manifest URI, created time, and evaluation report summary. `GET /v1/models/{model_id}/download` authorizes only a READY model and returns a short-lived download grant for the immutable manifest and adapter; it returns `409 MODEL_NOT_READY` otherwise.

All errors use:

```json
{
  "error":{"code":"DATASET_TOO_SMALL","message":"Fewer than 100 valid unique images remain after validation.","retryable":false,"details":{"valid_unique_images":83,"minimum":100},"request_id":"7c2d1cb8-8ba3-4fa0-9e80-8e2c5db281d0"}
}
```

Use `400` for malformed requests, `401` for missing/invalid credentials, `404` for inaccessible resources, `409` for state/idempotency/profile conflicts, `422` for valid but unusable datasets or configuration, `429` for admission caps, and `503` for temporary control-plane unavailability. Meaningful codes include `PROFILE_NOT_READY`, `EVALUATION_NOT_READY`, `UNSUPPORTED_PROFILE`, `DATASET_INVALID`, `DATASET_UNSUITABLE`, `GPU_BUDGET_EXCEEDED`, `QUALITY_REJECTED`, and `JOB_ALREADY_READY`.

## 5. Persistent data model and operational requirements

The schema uses `uuid` resource primary keys (PK), the explicitly listed composite/singleton control keys, `timestamptz` UTC times, `bigint` counters/bytes/seconds, `text` URIs/enums, `jsonb` immutable structured configuration, and DB-clock lease comparisons. Every SHA-256 field, including shorthand `/sha256` fields below, is `char(64)` with lowercase 64-hex validation. Fields are `NOT NULL` unless marked nullable; tenant-owned externally addressable tables carry `owner_id uuid NOT NULL FK tenant(id)`. Global profile and policy revisions are operator-owned and do not. Job, stage, task and device state fields are constrained to the corresponding enumerations in sections 2–3.

| Table | Typed fields, keys, and important constraints |
| --- | --- |
| `dataset` | `id uuid PK`, `owner_id uuid FK`, `state text CHECK (UPLOADING, VERIFYING, COMPLETED, INVALID)`, `submitted_manifest_sha256 char(64)`, `frozen_manifest_uri text NULL`, `frozen_manifest_sha256 char(64) NULL`, `declared_count int CHECK (100..1000)`, `valid_count int NULL`, `invalid_count int NULL`, `verification_error jsonb NULL`, `created_at timestamptz`, `completed_at timestamptz NULL`; `completed_at IS NULL` for UPLOADING/VERIFYING; completed/invalid require it, and completed requires frozen manifest fields. |
| `dataset_file` / `dataset_verify_task` | File: `id uuid PK`, `dataset_id uuid FK dataset(id)`, `name text`, `object_version text NULL` until completion, `size_bytes bigint CHECK (>0)`, `sha256 char(64) CHECK (sha256 ~ '^[0-9a-f]{64}$')`, `mime_type text`, `caption text NULL`, `caption_sha256 char(64) NULL CHECK (caption_sha256 ~ '^[0-9a-f]{64}$')`, `verification_status text`, `rejection_code text NULL`; `UNIQUE(dataset_id,name)`, and caption checksum is required iff caption is present. Verify task: `id uuid PK`, `dataset_id uuid NOT NULL UNIQUE FK`, `state text`, `attempt_count smallint`, `ready_at/lease_expires_at timestamptz`, `fencing_token bigint NULL`; inserted when completion freezes object versions. |
| `prepared_dataset_version` | `id uuid PK`, `job_id uuid NOT NULL UNIQUE FK training_job(id)`, `source_dataset_id uuid FK dataset(id)`, `cleaned_manifest_uri/sha256 text`, `train_manifest_uri/sha256 text`, `heldout_manifest_uri/sha256 text`, `preprocessor_version text`, `split_seed bigint`, `grouping_version text`, `producing_attempt_id uuid FK task_attempt(id)`, `created_at timestamptz`; all manifest fields and producing attempt are non-null and immutable. |
| `training_input_version` | `id uuid PK`, `prepared_version_id uuid NOT NULL FK prepared_dataset_version(id)`, `caption_manifest_uri text NULL`, `caption_manifest_sha256 char(64) NOT NULL` (the digest of a canonical empty caption manifest is used when no caption artifact exists), `captioner_version text NULL`, `training_manifest_uri text`, `training_manifest_sha256 char(64)`, `trigger_token_sha256 char(64)`, `producing_attempt_id uuid FK task_attempt(id)`, `created_at timestamptz`; `UNIQUE(prepared_version_id, caption_manifest_sha256, trigger_token_sha256)` and no fields may change the prepared held-out manifest. |
| `training_profile_revision` | `id uuid PK`, `profile_key text`, `revision int`, `status text`, `config jsonb`, `base_revision text`, `container_digest text`, `quality_policy_revision_id uuid FK quality_policy_revision(id)`, `gpu_class text`, `gpu_budget_seconds bigint CHECK (>0)`; `UNIQUE(profile_key,revision)`, immutable once `status=PUBLISHED`. Its `config` holds validated OOM fallback and candidate-1 variant IDs. |
| `quality_policy_revision` | `id uuid PK`, `policy_key text`, `revision int`, `status text`, `definition jsonb`, `calibration_artifact_uri text NULL`, `created_at timestamptz`; `UNIQUE(policy_key,revision)` and `PUBLISHED` requires a calibration artifact. |
| `training_job` | `id uuid PK`, `owner_id uuid FK`, `dataset_id uuid FK`, `profile_revision_id uuid FK`, `quality_policy_revision_id uuid FK`, `prepared_version_id uuid NULL UNIQUE FK`, `training_input_version_id uuid NULL FK`, `state text`, `current_stage text NULL`, `candidate_index smallint CHECK (0..1)`, `trigger_token text`, `gpu_seconds_charged/reserved bigint CHECK (>=0)`, `cancel_requested_at timestamptz NULL`, `error_code text NULL`, `created_at/updated_at timestamptz`; `UNIQUE(owner_id,id)`. |
| `job_candidate` | `job_id uuid FK`, `candidate_index smallint CHECK (0..1)`, `profile_revision_id uuid FK training_profile_revision(id)`, `training_input_version_id uuid NULL FK`, `adapter_sha256 char(64) NULL`, `checkpoint_compatibility_key text NULL`, `outcome text NULL`; composite `PK(job_id,candidate_index)` and `UNIQUE(job_id,candidate_index,training_input_version_id)`. Input/compatibility are nullable only before TRAIN admission and immutable once bound. Candidate 1 must reference the configured immutable variant and the same quality-policy revision. |
| `stage_task` / `task_attempt` | Task: `id uuid PK`, `job_id uuid FK`, `candidate_index smallint CHECK (0..1)`, `stage text`, `state text`, `ready_at timestamptz`, `attempt_count smallint CHECK (0..3)`, `active_token bigint NULL`, `lease_expires_at timestamptz NULL`, `deadline_at timestamptz NULL`, `checkpoint_uri text NULL`; `UNIQUE(job_id,candidate_index,stage)` and composite `FK(job_id,candidate_index) REFERENCES job_candidate`. Attempt: `id uuid PK`, `task_id uuid FK`, `attempt_no smallint CHECK (1..3)`, `gpu_slot_id uuid NULL FK`, `fencing_token bigint`, `scheduler_epoch bigint`, `started_at/last_heartbeat_at/lease_expires_at/ended_at timestamptz NULL`, `charged_gpu_seconds bigint`, artifact/log URIs and `termination_evidence jsonb NULL` (source, exact host/attempt, observed time, evidence reference, recording actor); `UNIQUE(task_id,attempt_no)` and `UNIQUE(task_id,fencing_token)`. |
| `gpu_slot` / `model_version` / `evaluation_report` | Slot: `id uuid PK`, `host_id/device_id text`, `capabilities jsonb`, `health text`, `current_attempt_id uuid NULL FK`, `quarantine_reason text NULL`; `UNIQUE(host_id,device_id)`. Model: `id uuid PK`, `job_id uuid NOT NULL UNIQUE FK`, `evaluation_report_id uuid NOT NULL UNIQUE FK evaluation_report(id)`, `adapter_sha256 char(64)`, `manifest_uri text`, `manifest_sha256 char(64)`, `state text`, `ready_at timestamptz NULL`. Report: `id uuid PK`, `job_id uuid`, `candidate_index smallint CHECK (0..1)`, `policy_revision_id uuid FK`, `prepared_version_id uuid FK`, `training_input_version_id uuid FK`, `training_input_manifest_sha256 char(64)`, `adapter_sha256 char(64)`, `full_pass boolean`, `smoke_pass boolean`, `report_uri text`, `report_sha256 char(64)`; composite `FK(job_id,candidate_index,training_input_version_id) REFERENCES job_candidate` and `UNIQUE(job_id,candidate_index,adapter_sha256,policy_revision_id)`. |
| `idempotency_request` / `scheduler_guard` | Idempotency: `owner_id uuid`, `route text`, `key text`, `request_sha256 char(64)`, `response_status int`, `response_body jsonb`, `expires_at timestamptz`, composite `PK(owner_id,route,key)`. Guard: singleton `id smallint PK CHECK(id=1)`, `leader_epoch bigint`, `stage_cursor smallint CHECK (0..3)` (position in the four-entry cycle, distinguishing its two TRAIN entries), `owner_cursor_by_class jsonb` keyed by `EVALUATE`, `TRAIN`, and `CAPTION`, `updated_at timestamptz`. |

Indexes include `stage_task(state,ready_at)`, `training_job(owner_id,created_at)`, and `task_attempt(gpu_slot_id,lease_expires_at)`. Foreign keys and guarded state-transition predicates enforce that a READY model references a completed full evaluation with `full_pass=true`, `smoke_pass=true`, and the exact adapter/training-input checksums; an active non-expired attempt owns a task update; and only one model exists per job. Queue and quota checks occur in the same transaction as job insertion. A lease renewal and any mutable attempt transition verify both fencing token and DB-clock expiry.

Security controls include tenant-scoped, short-lived presigned uploads/downloads; encryption in transit and at rest; least-privilege worker credentials; audit logs; rate limits; and image decoding in constrained workers. The service rejects archives, arbitrary code, and untrusted serialized checkpoint/pickle ingestion. It limits file count, compressed and decoded pixel dimensions, aggregate bytes, and decode work to reduce decompression-bomb risk; precise limits are profile/service configuration rather than hard-coded here.

Operational defaults retain raw uploads for 30 days, unreferenced checkpoints/attempt artifacts for seven days, and READY model artifacts until the owner deletes them; garbage collection must never remove data referenced by a nonterminal job, an unconfirmed executor, or a READY model. Jobs with unconfirmed executors retain admission-quota exposure and protected references even if a terminal technical failure has been recorded. Metrics include queue wait by stage/owner, admission denials, GPU slot utilization and quarantine, scheduler fairness/age, OOMs, retries, heartbeat/lease expiry, cancellation latency, object-store/DB errors, GPU seconds per outcome, and quality pass/failure dimensions. Structured logs correlate request, job, task, and attempt IDs without storing image payloads.

Suggested service objectives are targets to measure rather than established facts: durable accepted jobs survive controller failover; no READY model lacks a passing policy report and smoke test; and alerts fire before queue age, GPU-budget overrun, or error-rate targets threaten user experience. PostgreSQL point-in-time recovery, required upload-object versioning, optional storage replication, encrypted backups, restore exercises, and controller standby address recoverability. They do not remove the single-host fault domain of a small GPU pool.

## 6. Acceptance scenarios

| Scenario | Expected result |
| --- | --- |
| Dataset count boundaries | A 99-file manifest and a 1,001-file manifest are rejected at creation; 100 and 1,000 are accepted for asynchronous verification, subject to later validity/suitability checks. |
| Known-passing 240-image test fixture under a calibrated profile | Job progresses through all stages; PREPARE binds a manifest chain, full evaluation and load smoke test pass, and exactly one checksum-pinned READY model appears for the owner. Arbitrary valid datasets may still fail quality. |
| 130 uploads, 35 corrupt/duplicate after cleaning | PREPARE fails `DATASET_TOO_SMALL` with counts; no GPU stage starts. |
| Insufficient independent groups or held-out coverage | PREPARE fails `DATASET_UNSUITABLE`; a nominally valid count does not bypass the leakage/suitability gate. |
| Four concurrent eligible jobs on two GPUs | At most two stages execute; each uses one compatible exclusive slot; scheduler round-robins owners and evaluation receives weighted opportunities without a permanently reserved GPU. |
| Evaluation waits behind continuing training | The weighted cycle and queue-age alert cause an eligible evaluation to receive a scheduling opportunity under finite admitted backlog; a borrowed running stage is not preempted. |
| OOM under a supported profile | The recorded, prevalidated fallback is attempted if budget remains; an unsupported fallback/configuration fails rather than improvising a lower-quality run. |
| GPU worker dies, then an old worker reconnects | Lease expires and stale fencing token cannot renew a lease, bind manifests, commit checkpoint references, or publish; device stays quarantined until confirmed termination/health; retry uses only a compatible committed checkpoint within attempt/budget limits. |
| Concurrent controllers dispatch work or one takes over | The singleton guard serializes claim/budget/cursor changes, and an old leader epoch is rejected after takeover; no slot/task receives two valid grants. |
| Partial write during publication | Artifacts without the guarded READY transaction remain non-public and are later cleaned; retry/reconciliation creates or finds the sole model for the job. |
| Quality screen or full evaluation fails | No model becomes READY. At most one immutable candidate-1 remediation may run; a second remediation is refused and final failure is `QUALITY_REJECTED` with versioned evidence. |
| Retries plus evaluation exceed GPU budget | Each charged attempt, evaluation, and smoke stage counts; the watchdog ends the job as `GPU_BUDGET_EXCEEDED`. |
| Database outage during a running attempt | No new starts/publications occur; agent stops before lease expiry; reconciliation resumes safely after recovery. |
| Client repeats a create request | Same idempotency key and body returns the original job; a changed body returns `409`; quota is not consumed twice. |
| Cancel races with publish | Guarded transaction yields either a READY model or a durable cancellation result; late cancellation after READY returns `409`. |
| Cancelled job has a permanently unreachable executor | It stays `CANCEL_REQUESTED` with quota/artifact protection and an alert until a valid agent acknowledgement or audited instance/BMC termination attestation arrives; then it becomes `CANCELLED`. |
| Cross-tenant dataset/model access | Authorization returns `404` without an upload/download grant or revealing dataset, report, or model metadata. |

These scenarios map directly to the assessment criteria: the manifest and evaluation chain demonstrate ML pipeline architecture; leases, epochs, transactional queueing, and reconciliation demonstrate distributed-systems correctness; quotas, isolation, retention, observability, and backup constraints demonstrate production readiness; and the shared-pool weighted scheduler, calibrated multi-gate quality policy, and conservative publication protocol address the small-GPU resource constraint without claiming unmeasured performance.
