# Deliverable 1 — System Architecture

## 1. Design summary and boundaries

An asynchronous, quality-gated pipeline converts a user's image collection into a versioned style LoRA that an inference consumer can load. The scarce resource is the **shared pool of 2–4 GPUs**, so CPU preparation, durable orchestration, and GPU execution are separate responsibilities. Training, GPU captioning, and evaluation compete for the same explicitly managed pool.

This is a system-design submission. It specifies a deployable architecture and contracts; it does not claim an implemented service, a trained model, measured throughput, or calibrated quality thresholds. The companion [technical specification](02-technical-specification.md) defines components, APIs, schemas, failure handling, and acceptance scenarios.

| Requirement | Design boundary / assumption |
|---|---|
| Input | 100–1,000 images for one intended visual style; at least 100 valid unique images must remain after cleaning, before the training/validation split. |
| Output | Approved adapter weights, immutable base-model revision, loading configuration, artifact hashes, evaluation report, and a successful inference-load smoke test. A LoRA is not a standalone base model. |
| Hardware | 2–4 registered GPU devices, possibly on one or multiple hosts. Only profiled single-GPU configurations are admitted; memory and runtime feasibility must be measured for each supported profile. |
| Model neutrality | A versioned training profile encapsulates model-specific training, preprocessing, loading and resource requirements. Changing model families does not change the orchestration contract. |
| Concurrency | Many requests may be accepted and queued; at most one stage runs on each GPU, and at most one current authorized GPU attempt per job. Lost attempts can physically overlap a recovery attempt on another device until termination; their resource exposure remains accounted for. Admission is bounded. |
| Deployment | Register an inference-ready model and hand it to an existing inference consumer. An always-on serving fleet and its SLA are outside this assignment's training-resource budget. |
| Quality | Only candidates that satisfy an enabled, calibrated evaluation policy become `READY`. Arbitrary datasets are not guaranteed to produce acceptable models. |

## 2. Architecture diagram

![Automatic LoRA training pipeline architecture](../diagrams/system-architecture.png)

[Scalable SVG](../diagrams/system-architecture.svg) · [Editable PlantUML source](../diagrams/system-architecture.puml)

The API, scheduler, registry and publishing logic are **logical components**, not a requirement for a separate microservice per box. PostgreSQL is both the metadata authority and the durable stage-task queue. The registry consists of database records pointing to immutable object-storage artifacts; it is not a second metadata system.

### Numbered data flow

1. **Upload and authenticate.** The API creates an owner-scoped dataset upload session and returns short-lived upload URLs. Image bytes go directly to versioned object storage. Completion freezes object versions and enqueues verification, returning `202` while the dataset is `VERIFYING`; only a `COMPLETED` dataset may be admitted to training.
2. **Admit atomically.** A training request references the completed dataset and an enabled training profile. A database transaction checks quotas and idempotency, creates the job, and inserts its first `PREPARE` task. The client immediately receives a job ID and status URL.
3. **Prepare and schedule.** CPU workers inspect files, remove duplicates, group near-duplicates before a roughly 90/10 split, and produce an immutable prepared-data manifest. Profiles specify minimum training and held-out image/group counts; an infeasible group-disjoint split fails before training. Missing training captions become an optional GPU `CAPTION` task that derives a final training-input manifest without changing the split. Every stage atomically binds its output manifest, completes its task, and creates its unique successor under the current lease/token. The scheduler selects eligible stage tasks by resource compatibility, stage-class rotation and user fairness.
4. **Execute within a lease.** A host agent starts an isolated, pinned stage container on an exclusively claimed GPU. The attempt receives a fencing token, renewable lease and hard deadline. Model weights remain frozen except for the selected LoRA parameters. Progress and retry accounting are durable.
5. **Checkpoint and evaluate.** Workers write immutable checkpoints and candidate artifacts. `EVALUATE` receives its own GPU allocation, renders a fixed validation suite, computes the policy's quality dimensions, and reloads the exact saved candidate to verify inference compatibility. At most one budgeted quality remediation is allowed.
6. **Publish a verified version.** CPU publishing checks artifact and evaluation hashes, the current attempt, and cancellation state. One transaction registers the model as `READY`, marks the job `READY`, and completes publication. Failed candidates remain unavailable for model download.
7. **Hand off for inference.** An authorized consumer receives the versioned adapter and loading manifest, including the compatible base-model revision. Download access is short-lived and tenant-scoped. Publication can be retried without producing duplicate versions.

## 3. Workflow and quality boundaries

![Job lifecycle with quality gates and bounded remediation](../diagrams/job-lifecycle.png)

[SVG](../diagrams/job-lifecycle.svg) · [PlantUML](../diagrams/job-lifecycle.puml)

The job lifecycle and stage execution are different dimensions. A job can be `RUNNING`, have `current_stage=EVALUATE`, and have that stage `PENDING` while waiting for a GPU. The API exposes both dimensions so users can distinguish queuing from execution.

Training loss detects divergence but does not establish product quality. The release gate checks style consistency, prompt adherence, diversity, excessive training-image reproduction, and technical loadability. Training, checkpoints, evaluation and publication reference the same immutable prepared-data version and training-input manifest chain; no consumer follows an unversioned "latest" path. Evaluation references are held out; prompts vary subject matter to expose content leakage. Quality-policy versions pin evaluator models, prompts, seeds, preprocessing and calibrated thresholds. No threshold is silently weakened on retry.

CLIP-based image and text similarities are useful supporting measurements, but image similarity may confound style and content. The proposed policy combines a calibrated style-oriented evaluator with other independent dimensions and offline human judgments. [StyleDrop's evaluation](https://research.google/blog/styledrop-text-to-image-generation-in-any-style/) illustrates both image/text similarity measurements and human preference evaluation; its reported scores are not transplanted as thresholds for this system.

## 4. Resource constraints and design tradeoffs

| Decision | Reason and limitation |
|---|---|
| PostgreSQL-backed stage queue | Job creation and task insertion share a transaction, avoiding a database/broker dual-write gap. This scale does not justify another queue service. A short claim transaction can use row locking; [PostgreSQL documents `SKIP LOCKED` for queue-like consumers](https://www.postgresql.org/docs/current/sql-select.html). Polling remains authoritative. |
| Single-GPU stage isolation | Predictable ownership and smaller failure domains; no concurrent jobs sharing one device's memory. Profiles that cannot fit a supported single device are rejected. Multi-GPU jobs are a future scheduling extension. |
| Shared evaluation capacity | No GPU permanently reserved for evaluation. The repeating class cycle `EVALUATE → TRAIN → CAPTION → TRAIN` skips empty or incompatible classes. User round-robin and FIFO eligible work within each class prevent a single user's backlog from dominating dispatch. Running work is not preempted. |
| Work-conserving fairness | When granting new work, one active GPU task per user while other users have eligible work; a user may borrow idle slots otherwise. Existing borrowed work is not preempted when another user arrives and completes within its hard stage timeout. Fairness is dispatch opportunity, not a promise of equal GPU-seconds or instant start. |
| Bounded effort | Default admission limits: 5 nonterminal jobs/user and 100 globally. Each profile defines an explicit cumulative GPU budget, stage deadlines and evaluation reservation. Retries and remediation are charged too. |
| Reuse and early rejection | Reject unsuitable data before training; cache frozen-base assets and compatible frozen preprocessing outputs; screen obviously failing candidates before a full evaluation. Tenant-derived caches remain isolated. A screening pass alone never authorizes release. |

For a rough capacity model, let `G` be the number of usable GPU slots and `T` the average **total GPU-hours per completed job**, including captioning, evaluation, warm-up and retries. At utilization `u`, throughput is approximately `u × G / T` jobs/hour. This is a planning formula, not a benchmark; heterogeneous devices and rejected jobs require per-profile measurements. Adding API replicas does not increase GPU throughput. Queue wait is reported separately from execution time.

## 5. Scalability and fault tolerance

![Leases, device quarantine, recovery and stale-result rejection](../diagrams/lease-recovery.png)

[SVG](../diagrams/lease-recovery.svg) · [PlantUML](../diagrams/lease-recovery.puml)

**Execution can repeat; publication must not.** A worker can lose its lease after writing a checkpoint or artifact. Attempt-scoped object paths, monotonically increasing fencing tokens, guarded state transitions, and a unique published-model record per job prevent stale execution from changing the canonical outcome. Storage objects are written and verified before a database transaction exposes their immutable manifest.

**A lost lease is not proof of an idle GPU.** An unreachable host is quarantined. Its device returns to the pool only after process termination and device health are confirmed. A different healthy device can retry within the remaining conservative budget. Only one attempt is currently authorized to commit, but an unconfirmed stale process may physically overlap recovery on that other device; both count toward resource exposure. Host watchdogs enforce lease and execution deadlines; the scheduler never launches a second task on an unconfirmed device. Cancellation remains `CANCEL_REQUESTED` until termination is confirmed by the agent or, for an unrecoverable host, by audited control-plane/BMC shutdown evidence. A permanently retired device does not rejoin the pool.

**Control-plane failure pauses safe progress.** Stateless APIs can have multiple replicas. The scheduler uses active/standby leadership; each short dispatch transaction locks the persisted scheduler guard and verifies its leader epoch before checking owner eligibility, claiming a task/device, reserving budget and advancing the cursor. This prevents concurrent or stale controllers from bypassing user fairness through separate task claims. If the metadata store is unavailable, new starts and publication stop; running workers stop when lease renewal is no longer safe. Completed objects and committed checkpoints survive a worker restart. Managed database failover/PITR and object-store durability protect committed state, but recovery still depends on configured backup guarantees.

**Scale the bottleneck deliberately.** CPU workers can scale independently; GPU hosts register additional profiled slots and require no client API change. At greater queue volume, a broker can deliver wake-up hints, with a transactional outbox if durable external events are introduced. The database remains the state authority. Continuous serving, multi-GPU gang scheduling and broad hyperparameter search require separate capacity decisions.

With all GPUs on one host, host failure removes all training capacity until recovery. Distributing available GPUs across hosts reduces that failure domain; a 2–4 GPU budget alone does not imply hardware redundancy. Backlog, lease loss, GPU utilization, OOMs, budget consumption and quality rejections are observable and alertable.

## 6. Evaluation-criteria coverage

| Evaluation criterion | Evidence in the submission |
|---|---|
| ML pipeline architecture | Versioned datasets and profiles; deduplication before splitting; optional content-focused captions; LoRA-only training; complete checkpoints; held-out multidimensional evaluation; exact-artifact inference smoke test. |
| Distributed systems | Transactional admission and stage progression; durable task queue; leases and fencing; device quarantine; bounded retries; idempotent API and effectively-once model publication; cancellation races. |
| Production requirements | Tenant isolation; scoped direct uploads; schema constraints; admission limits; audit and metrics; backups and retention; reproducibility; explicit errors and operational acceptance scenarios. |
| Creative use of limited resources | Shared training/evaluation capacity; fair, work-conserving scheduling; bounded remediation; compatible caches; early data rejection and evaluation screening; GPU-time budgets rather than unbounded retries. |

The technical specification makes these mechanisms concrete through API examples, schema contracts and scenario-based acceptance criteria. The diagrams remain editable and can be reproduced locally using the instructions in the [submission guide](../README.md).
