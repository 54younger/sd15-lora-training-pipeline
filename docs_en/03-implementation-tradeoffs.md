# Decision record — reducing the design to a testable single-host pipeline

Status: accepted by the user before implementation. This record explains the current boundary rather than claiming every original production mechanism was built.

| Decision | Benefit | Cost and future migration |
|---|---|---|
| SQLite instead of PostgreSQL | Durable local queue with no database service; CPU integration tests run in temporary directories. | Single-host writer serialization; no PostgreSQL `SKIP LOCKED` or distributed controller HA. A migration needs new transactions, migrations and concurrency tests, not only a connection string. |
| Local files instead of S3 | No credentials, buckets or storage emulator; checksums and atomic writes remain testable. | Uploads traverse the API, artifacts share the host failure domain. S3 migration needs object version pinning, scoped authorization, partial-write recovery and storage integration tests. |
| One supervisor with OS device locks | Clear device ownership and recoverable local process management. | No multi-host takeover. Remote scheduling needs agents, leases, fencing and trustworthy termination evidence. |
| SD 1.5 as the real training family | Smaller training and inference integration surface for a 16GB consumer GPU. | No SDXL conditioning, dual text encoders or multi-model-family claim. Model interfaces leave room for another backend, but that requires implementation and tests. |
| Conservative image filtering | Avoids rejecting intentional artistic blur, low contrast and unusual exposure. | Warning diagnostics cannot guarantee suitability. Exact duplicates are removed; perceptual groups are heuristics and may over-group images. |
| Optional template or BLIP captions | Offline workflow works without extra weights; BLIP supplies content descriptions when selected. | Generic templates do not identify subjects; BLIP can misdescribe images. Sources and versions are recorded; explicit BLIP failure is not hidden. |
| Technical success separate from quality approval | Synthetic data can test the full pipeline without pretending to establish style quality. | Default models remain UNVERIFIED and downloads require opt-in. Real READY approval requires externally calibrated evidence and fixed metric gates. |
| CLIP diagnostics instead of FID as the default | Paired prompt/image comparisons fit a small assignment evaluation workload. | CLIP mixes content and style. No claim that a similarity threshold independently validates style or memorization. FID is not implemented. |
| Infrastructure retry only | Predictable, explainable recovery and bounded resource consumption. | OOM and quality rejection require a new explicit configuration/job; no automatic search or second quality candidate. |
| CPU tiny tests plus local GPU smoke | The tiny path uses a Diffusers/PEFT Autoencoder/UNet/scheduler graph with random local weights, so offline tests verify real LoRA gradients, frozen weights, resume and orchestration. | Random weights do not verify CUDA kernels, pretrained output or style quality; real SD 1.5/BLIP/CLIP checks are separate user-run tests, and tiny artifacts are always test-only. |
| Metrics definitions without benchmark claims | Useful performance deliverable despite no measured GPU workload in the development session. | All unrun measurements remain `not_measured`; GPU smoke timings are not representative production throughput. |

## Reliability retained

The simplification retains immutable manifest chains, bounded uploads, owner authorization, admission quotas, idempotent mutation, durable tasks, unique publication and complete checkpoints. API and worker are separate processes so a long training call does not become an in-memory web background task. A new supervisor does not interpret an expired heartbeat as proof that an old GPU process has stopped.

The publication boundary has changed deliberately: `COMPLETED_UNVERIFIED` is a successful technical outcome, while READY remains a quality-approved outcome. UNVERIFIED results retain provenance and can be used for local inference only after explicit acceptance of their status. Test-only results always retain their marker.

## What is not a production guarantee

There is no claim of multi-host failover, tenant process sandboxing against hostile executable code, content moderation, a calibrated style evaluator, zero data loss after disk failure, or retention automation. The API does not accept custom Python code or arbitrary checkpoint uploads. Basic input validation, file limits and API-key ownership are implemented; a public service would additionally need an ingress/TLS/rate-limiting deployment and identity-provider integration.

The selected single-host implementation supports the assignment's resource contention through a configurable pool of physical GPUs. CPU tests exercise two and four logical test slots; one local physical GPU cannot be multiplied into four CUDA slots. Image count limits and fairness are preserved rather than bypassed to make the demo pass.

## Source guidance

SQLite documents single-writer concurrency and same-host constraints for WAL; this implementation uses short transactions and the default rollback journal rather than relying on a shared network database file. [SQLite documentation](https://sqlite.org/wal.html)

Diffusers describes attention LoRA injection, optimizing only adapter parameters and saving loadable adapters. The implementation pins package versions and owns the training loop. [Diffusers LoRA guide](https://huggingface.co/docs/diffusers/training/lora)

Accelerate's checkpointing guide explains optimizer, RNG, scaler and data-loader state; the pipeline also binds checkpoints to its frozen data/configuration lineage. [Accelerate checkpointing guide](https://huggingface.co/docs/accelerate/usage_guides/checkpoint)
