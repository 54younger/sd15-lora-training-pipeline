# Running and API guide

This guide describes the implemented local Part 2 package. It assumes Linux/WSL2 and Python 3.12. GPU and Docker commands are reproducible procedures, not claims that they were run in this development environment.

## Install

```bash
python3.12 -m venv /tmp/lora-pipeline-venv
source /tmp/lora-pipeline-venv/bin/activate
python -m pip install -U pip
python -m pip install -e '.[dev]'
```

For CUDA, install the compatible PyTorch build explicitly, then run `python -m lora_pipeline preflight`:

```bash
python -m pip install torch==2.8.0 torchvision==0.23.0 \
  --index-url https://download.pytorch.org/whl/cu128
```

The real SD 1.5 path and BLIP/CLIP evaluation require model weights in the local Hugging Face cache. `--local-files-only` makes that requirement explicit. CPU tests initialize a randomly weighted Diffusers/PEFT tiny graph locally and do not download pretrained weights; their artifacts are always marked `test_only`.

## Configuration and preflight

Provide JSON configuration through `LORA_CONFIG`, or use the environment overrides in `.env.example`. `LORA_API_KEYS` is a JSON object mapping bearer tokens to owner IDs; the API refuses to start without one. `LORA_ADMIN_KEYS` controls `/metrics`. For an offline CPU test run, enable the explicit test backend and fake slots:

```bash
export LORA_API_KEYS='{"demo-token":"owner-a"}'
export LORA_ADMIN_KEYS='["admin-token"]'
export LORA_TEST_BACKEND=1
export LORA_FAKE_SLOTS=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
python -m lora_pipeline preflight
```

Replace example keys before use. The default data root is `./var`, containing `pipeline.sqlite3`, `objects/`, `artifacts/` and `locks/`. SQLite uses rollback journal `DELETE`, `synchronous=FULL`, and short `BEGIN IMMEDIATE` transactions. Back up the database and referenced files together while services are stopped.

## Offline CPU smoke

The output must be new or empty. This command generates 120 synthetic images, prepares/captions them, trains the `tiny` backend for four optimizer steps, interrupts at step two, resumes from a full checkpoint, reloads the adapter and runs technical evaluation:

```bash
python -m lora_pipeline cpu-smoke --output var/smoke-cpu
```

The result is `test_only: true` and `quality_status: UNCALIBRATED`. It proves orchestration, gradients, checkpoint restoration and adapter loading—not visual quality or GPU performance. Run tests with `python -m pytest`.

## GPU smoke

On a host with an NVIDIA driver, permitted GPU UUID and cached model weights:

```bash
python -m lora_pipeline preflight
python -m lora_pipeline gpu-smoke \
  --output var/smoke-gpu \
  --model stable-diffusion-v1-5/stable-diffusion-v1-5 \
  --local-files-only
```

The real `sd15` backend performs ten optimizer steps, interrupts after five, resumes, exports/reloads an adapter and runs a small evaluation. `--gpu-uuid` selects a UUID reported by `nvidia-smi`; the command uses the same per-device lock as the worker. This short workload is not a capacity benchmark, and results remain unverified until the command is actually run.

## Direct CLI

Implemented subcommands are `api`, `worker`, `worker-health`, `preflight`, `generate-data`, `prepare`, `caption`, `train`, `evaluate`, `compare`, `cpu-smoke` and `gpu-smoke`. Use `python -m lora_pipeline <command> --help` for exact flags. A local operator flow is:

```bash
python -m lora_pipeline generate-data --output var/generated --count 120
python -m lora_pipeline prepare --files var/generated/files.json --output var/prepared
python -m lora_pipeline caption --prepared var/prepared/prepared.json --output var/input --mode template --trigger-token mystyle
python -m lora_pipeline train --input var/input/training-input.json --config configs/sd15-smoke.json --output var/training
python -m lora_pipeline evaluate --training-result var/training/training-result.json --input var/input/training-input.json --output var/evaluation --config configs/evaluation-smoke.json
```

The example files under `configs/` are JSON representations of `TrainConfig` and `EvalConfig`. Evaluation always generates paired baseline and adapter outputs under identical prompts, seeds and inference settings. GPU commands acquire a device lock; `train --resume` accepts a pipeline-produced checkpoint. The API does not accept arbitrary serialized checkpoints or server paths.

## API and worker

Start the API and separate worker in two terminals, with the same configuration and data root. For an offline CPU test run, use the test flags shown below in both terminals:

```bash
export LORA_API_KEYS='{"demo-token":"owner-a"}'
export LORA_ADMIN_KEYS='["admin-token"]'
export LORA_TEST_BACKEND=1
export LORA_FAKE_SLOTS=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
# terminal 1
python -m lora_pipeline api --host 127.0.0.1 --port 8000
# terminal 2: repeat the exports above, then run
python -m lora_pipeline worker
```

Resource calls use `Authorization: Bearer demo-token`; mutation calls must include a stable `Idempotency-Key`. The main routes are:

| Method and path | Purpose |
|---|---|
| `POST /v1/datasets` | Declare file metadata (`name`, `size_bytes`, `sha256`, `mime_type`, optional `caption`). |
| `PUT /v1/datasets/{dataset_id}/files/{file_id}` | Stream raw bytes for one declared file. |
| `POST /v1/datasets/{dataset_id}/complete` | Freeze uploads and queue verification. |
| `GET /v1/datasets/{dataset_id}` | Read owner-scoped dataset state. |
| `GET /v1/training-profiles` | Read the real profile, plus `tiny-test` only when test mode is enabled. |
| `POST /v1/training-jobs` | Create a job using `dataset_id`, `profile_revision_id` and `trigger_token`. |
| `GET /v1/training-jobs/{job_id}` / `POST /v1/training-jobs/{job_id}/cancel` | Read status or request cancellation. |
| `GET /v1/training-jobs/{job_id}/evaluation` | Read the evaluation report. |
| `GET /v1/models/{model_id}` | Read model provenance and quality state. |
| `GET /v1/models/{model_id}/download` | Download; `COMPLETED_UNVERIFIED` requires `?allow_unverified=true`. |
| `GET /health/live`, `/health/ready` | Liveness and dependency/worker readiness. |

Use `curl -H 'Authorization: Bearer demo-token'` for authenticated requests. Dataset creation returns internal file IDs; upload bytes, complete the dataset, poll until `COMPLETED`, then create and poll a job. Cross-owner resources deliberately look like `404`. Errors contain `code`, `message`, `retryable`, `details` and `request_id`. `/metrics` requires an admin bearer token and returns Prometheus-style job gauges.

For a complete real HTTP upload/poll/download flow, run `scripts/api_demo.py` after starting API and worker. It discovers the enabled real or tiny profile and uses `LORA_DEMO_TOKEN` for authentication.

## Docker and Compose

`Dockerfile`, `compose.yaml`, `compose.gpu.yaml` and `.env.example` provide a local packaging path. The default Compose setup uses the CPU test backend; the GPU overlay requires a working Docker daemon, NVIDIA Container Toolkit, driver and model weights:

```bash
docker compose -f compose.yaml up --build
docker compose -f compose.yaml -f compose.gpu.yaml up --build
```

Persist the Compose `pipeline-data` and `model-cache` volumes and set real API/admin keys in an environment file. The GPU command is a target-host procedure; this session does not claim a successful Docker build or GPU run.

## Development-check scope

`python -m ruff check src tests scripts/api_demo.py` checks the repository source. With the current `pyproject.toml`, `python -m mypy` is intentionally limited to `src/lora_pipeline/common.py` and `src/lora_pipeline/config.py`; it is not a full-source type check. Docker Compose and the real GPU smoke require the target environment, so the validation record should report separately whether each was actually executed.
