# Running, acceptance, and API guide

For the manual four-step browser workflow, see the [Web Studio deployment and operation guide](06-web-studio.md). The CLI/API automatic workflow and GPU acceptance guidance remain below.

This document indexes the implemented boundary and acceptance evidence. The complete GPU Docker runbook (IBean preparation, `dc run` heredocs, real training, evaluation, and benchmark commands) is the [Docker CLI training and GPU acceptance guide](07-docker-cli-training.md). Full submission navigation is in [00-assignment-guide.md](00-assignment-guide.md).

## Keep the three kinds of evidence separate

- **CPU tiny** is an offline regression backend. It uses synthetic images from `generate-data`, random weights, and artifacts permanently marked `test_only`. It proves orchestration, gradients, checkpoint/resume, adapter loading, and HTTP contracts; it does not prove SD 1.5 quality, GPU memory, throughput, or scaling.
- **10-step GPU smoke** is target-NVIDIA-host functional acceptance: real SD 1.5, interruption at step 5, resume, adapter reload, and a small evaluation. It is not a capacity benchmark or a quality threshold.
- **100-step GPU performance** is a fixed-real-input, pinned-base measurement requiring one warm-up and at least three complete runs. It is separate from quality evaluation. Quality evidence comes only from paired base/adapter evaluation under fixed prompts/seeds and a calibrated policy.

The recommended real acceptance data is the local IBean copy: 999 images (333 per class) under `datasets/ibean/images/` plus `captions.json`. It is neither synthetic nor a calibrated style-quality set. Use the [IBean section of the Docker CLI guide](07-docker-cli-training.md#ibean-999-acceptance-dataset) for download, license, SHA-256, and selection details.

## Install and CPU regression

```bash
python3.12 -m venv /tmp/lora-pipeline-venv
source /tmp/lora-pipeline-venv/bin/activate
python -m pip install -U pip
python -m pip install -e '.[dev]'
python -m lora_pipeline preflight
```

The offline tiny path must explicitly enable the test backend; do not carry these variables into real GPU acceptance:

```bash
export LORA_API_KEYS='{"demo-token":"owner-a"}'
export LORA_ADMIN_KEYS='["admin-token"]'
export LORA_TEST_BACKEND=1
export LORA_FAKE_SLOTS=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
python -m lora_pipeline cpu-smoke --output var/smoke-cpu
python -m pytest -q
```

CPU smoke creates 120 synthetic images, interrupts at step 2, resumes from a complete checkpoint to step 4, reloads the adapter, and runs a technical evaluation. Expect `technical_pass=true`, `quality_status=UNCALIBRATED`, and `test_only=true`; do not present an old elapsed time or “all passed” statement as this run's result. `VALIDATION.md` records a historical run dated 2026-09-21; the collection may have changed, so its historical 63-test total is not the current total. Record the date and actual commands for the submission; write `not_run` when a check was not rerun.

Critical regression scope:

| Risk | Tests/evidence | Acceptance signal |
|---|---|---|
| Image limits, exact/near duplicates, group split | `tests/test_data.py`, `tests/test_image_boundaries.py` | Invalid inputs rejected; split is reproducible and has no group leakage |
| Caption precedence and BLIP failure | `tests/test_captions.py` | User captions win; missing model is an explicit failure, not a silent template fallback |
| LoRA, gradients, checkpoint compatibility | `tests/test_training.py` | Resume restores cursor/adapter; checksum/config mismatches fail |
| Paired A/B and metric arithmetic | `tests/test_evaluation.py` | Base/adapter conditions match; technical success is separate from quality policy |
| Owner isolation, idempotency, HTTP, unverified download | `tests/test_service_contracts.py` | Cross-owner resources are 404; same body replays, changed body conflicts |
| Fencing, cancellation, retry, slots/budget | `tests/test_scheduler_recovery.py`, `tests/test_service_review_fixes.py` | Stale attempts cannot publish; OOM/budget/cancel have explicit outcomes |

Record Ruff and the current mypy scope separately; mypy checks only `common.py` and `config.py`, not the whole source tree. Docker, Compose startup, and real GPU execution cannot be substituted by Python tests.

## Main GPU Docker flow

The target host needs Linux/WSL2, Docker Engine/Compose v2, an NVIDIA driver, and the NVIDIA Container Toolkit. The GPU overlay disables the test backend/fake slots and requests GPUs only for the worker; the API does not use CUDA. Compose `pipeline-data` stores SQLite, uploads, manifests, checkpoints, adapters, and reports; `model-cache` stores model weights.

Define the helper again in **every new shell**; `dc` is a Bash function, not a Docker subcommand:

```bash
dc() { docker compose -f compose.yaml -f compose.gpu.yaml "$@"; }
dc config --quiet
docker buildx version
dc build --progress=plain
dc run --rm --no-deps worker lora-pipeline preflight
```

If the Docker socket requires elevated access, change the function body to `sudo docker compose ...`; do not run `sudo dc ...`, because sudo does not resolve the current shell function. In PowerShell use `function dc { docker compose -f compose.yaml -f compose.gpu.yaml @args }`. Stop if `preflight` does not show the expected `cuda_available`, allowed GPU UUIDs, and memory.

The Docker CLI guide's modules 1–4 are the canonical GPU runbook. Their heredoc container commands use `-i -T`: `-i` supplies script stdin and `-T` disables pseudo-TTY allocation for SSH/CI/redirection. Omitting `-T` causes `the input device is not a TTY`; do not hand-create missing manifests. `/data` is a container path backed by the Compose named volume, **not host `/data`**. Files survive a `--rm` one-shot container in `pipeline-data`; export them with `dc run ... cat` or `dc cp`.

### Model cache, training, and progress

Warm `model-cache` online once, resolve the immutable commit, and write `/data/sd15-smoke-pinned.json`. Training and resume then use that pinned configuration with `local_files_only=true`; download time is excluded from performance measurement. See the [Docker CLI guide's training module](07-docker-cli-training.md#2-pin-the-base-model-then-train-and-resume).

Classify base-model download failures as `BASE_MODEL_UNAVAILABLE`, not `CHECKPOINT_CORRUPT`, `CHECKPOINT_INCOMPATIBLE`, or OOM. The error details expose only a redacted model/revision, cache/offline state, and controlled category; they do not promise the raw root cause. Fix network, credentials, cache, or disk before the offline check. If a frozen input checksum differs, preserve path/expected/actual details, rebuild, and rerun prepare+caption to create a new manifest; never edit the old manifest or bypass verification. Normalized images are addressed by SHA-256 of final PNG bytes, so a new encoder cannot overwrite files referenced by an old frozen manifest.

Direct CLI execution is **one foreground terminal**; another terminal is optional for `nvidia-smi`. CLI lifecycle/progress goes to stderr, final JSON goes to stdout, and failures return non-zero:

```bash
python -m lora_pipeline train --input /path/training-input.json \
  --config /path/sd15-smoke-pinned.json --output /path/training
python -m lora_pipeline evaluate --training-result /path/training/training-result.json \
  --input /path/training-input.json --output /path/evaluation --config configs/evaluation-smoke.json
```

Training progress includes `global_step`, loss, cumulative `samples_processed`, per-invocation `samples_processed_this_run`, throughput, checkpoint time, RSS, and CUDA allocated/reserved; the API exposes it at `stages.TRAIN.progress`. Evaluation progress includes technical smoke, model loading, generation, CLIP scoring, and report writing. The implementation starts evaluation `elapsed_seconds` after input validation and computes it before `report_writing`/HTML and final-manifest writes, so it excludes validation and report writing; checkpoint restore has no independent automatic timer. Do not call these fields a precise full evaluation or restore cycle without an external timer covering that whole interval.

API plus worker are **two processes/two terminals** because long training is not an in-memory web background task. Compose runs the same topology in the background with `dc up -d api worker`. `/health/ready` may be 503 briefly until the worker heartbeat exists. Do not run one-shot modules 1–3 concurrently with the resident worker on the same GPU.

## REST contract and minimal example

```bash
export LORA_API_KEYS='{"demo-token":"owner-a"}'
export LORA_ADMIN_KEYS='["admin-token"]'
python -m lora_pipeline api --host 127.0.0.1 --port 8000   # terminal 1
python -m lora_pipeline worker                             # terminal 2
```

The real profile is `local-sd15-v1` (`style-lora`). Only with explicit `LORA_TEST_BACKEND=1` does `/v1/training-profiles` additionally return `local-tiny-v1` (`tiny-test`). Clients cannot submit backend/device/precision or arbitrary server paths; overrides are limited to the bounded knobs implemented by the store.

Every mutation POST (create dataset, complete, create job, cancel) requires `Idempotency-Key`. The key is scoped by `owner_id + concrete route (method/path) + key` and bound to the request body hash: an identical request replays its original status/body; a changed body returns `IDEMPOTENCY_KEY_REUSED`. File PUT is a streaming upload checked by file ID, declared size, and SHA-256; it accepts no arbitrary path. `scripts/api_demo.py` sets keys for its POST helpers.

```bash
curl -sS -H 'Authorization: Bearer demo-token' \
  http://127.0.0.1:8000/v1/training-profiles
curl -sS -X POST http://127.0.0.1:8000/v1/training-jobs \
  -H 'Authorization: Bearer demo-token' -H 'Content-Type: application/json' \
  -H 'Idempotency-Key: job-001' \
  -d '{"dataset_id":"<completed-dataset-id>","profile_revision_id":"local-sd15-v1","trigger_token":"mystyle"}'
curl -sS -H 'Authorization: Bearer demo-token' \
  'http://127.0.0.1:8000/v1/training-jobs/<job-id>'
```

Job creation returns `202 ACCEPTED` with initial `state=ACCEPTED,current_stage=PREPARE`. Terminal states include `READY`, `COMPLETED_UNVERIFIED`, `FAILED`, `QUALITY_REJECTED`, and `CANCELLED`. With `quality_policy=null`, technical success is `COMPLETED_UNVERIFIED`, not quality approval; a non-test-only result with policy `PASS` is required for `READY`. Non-READY downloads require `allow_unverified=true`.

Only application-raised `PipelineError` uses the uniform HTTP JSON envelope with `error.code/message/retryable/details/request_id` and `X-Request-ID`; common statuses are 401, 404, 409, 429, 503, and 422. FastAPI/Pydantic body validation runs before the route function, so its `RequestValidationError` remains the default HTTP 422 `{"detail":[...]}`, not the `error` envelope. CLI instead writes `{"error":{"code":"...","message":"...","details":{}}}` to stderr and returns 1 while keeping successful final JSON on stdout. Do not treat the CLI error object as the HTTP contract.

`scripts/api_demo.py` is a **synthetic PNG + tiny regression example**: it hard-codes `image/png` and prefers the tiny profile, so it is not IBean GPU acceptance, a benchmark, or quality evidence. A real client must preserve source MIME, select `local-sd15-v1`, and use separate dataset/job timeouts.

### API acceptance table

| Stage | Evidence | Acceptance |
|---|---|---|
| health/auth | live, ready, admin metrics | live 200; ready database/storage/worker healthy; metrics is 401 for non-admin |
| dataset | create response, every PUT 204, GET after complete | size/SHA/MIME verified; final `COMPLETED`, or `INVALID` with reason |
| training | job, `stages.TRAIN.progress`, result/checkpoint | explicit terminal state; adapter, config, base revision/fingerprint, input hash recorded |
| evaluation/quality | `/evaluation`, paired outputs, report | `technical_pass=true`; uncalibrated policy remains `UNVERIFIED` |
| artifact | model response and downloaded file | adapter/manifest/report checksums recompute; unverified requires opt-in |
| isolation/replay | two owners and same/changed-body retries | cross-owner 404; identical key creates one resource; changed body conflicts |

See the [technical specification](02-technical-specification.md) and [trade-offs](03-implementation-tradeoffs.md) for threshold calibration, CLIP limitations, and publication boundaries. Submit commit, command logs, config, manifest/model revision, checkpoint/quality/artifact evidence, and the date of the current validation. Use `not_measured`/`null` for unrun Docker/GPU/benchmark work.

## Compose lifecycle and resource constraints

```bash
dc up -d api worker
dc ps
dc logs --tail=200 api worker
dc down                 # preserves named volumes
# dc down -v deletes database, checkpoints, adapters, and model cache; export first
```

Each physical GPU UUID is one slot; CPU fake slots only test admission/fairness. Owner/global queue limits, stage/job deadlines, attempts/retries, and cumulative GPU-second budgets are scheduler constraints. GPU OOM is an explicit failure; parameters are not silently changed. Record a new configuration before rerunning. SQLite plus named volumes form a single-host failure domain; multi-host production needs shared DB/object storage/executors.

The [benchmark guide](05-performance-benchmarks.md) defines the 2–4 GPU, multi-owner fairness, queue wait, OOM/retry-cost, and occupancy experiment.
