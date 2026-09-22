# Automatic LoRA Training Pipeline

[中文说明](README_ZH.md) · [Detailed Docker CLI guide](docs_en/07-docker-cli-training.md) · [Web Studio guide](docs_en/06-web-studio.md)

Train a Stable Diffusion 1.5 attention LoRA from 100–1,000 images through a durable four-stage workflow: **prepare data → train → evaluate → publish and download**. The recommended entry point is the English browser workbench at `http://localhost:8080`.

## Quick start with Docker

Requirements: Docker Engine with Docker Compose v2. Real SD 1.5 training also needs a Linux/WSL2 NVIDIA GPU host, a compatible driver, and NVIDIA Container Toolkit. The CPU option below is a technical demo only.

Create a local configuration, then replace the sample token in `LORA_API_KEYS` with a long random value:

```bash
cp .env.example .env
openssl rand -hex 32
```

For example, set `LORA_API_KEYS={"your-random-token":"owner-a"}` in `.env`. The website login value is `your-random-token` — the **key** in that JSON mapping, not the complete JSON value. Do not commit `.env`.

If Docker socket access requires `sudo` (for example, you built the images with `sudo`), prefix **every** `docker compose` command below—including `up`, `ps`, `logs`, and `down`—with `sudo`. Use `sudo docker compose ...`; do not use `sudo dc`, because `dc` is only a shell function in the detailed CLI guide.

Start real GPU services (always use both Compose files for GPU commands):

```bash
docker compose -f compose.yaml -f compose.gpu.yaml up --build -d
```

If you already ran the build command, start the built images without rebuilding:

```bash
docker compose -f compose.yaml -f compose.gpu.yaml up -d
```

For the CPU technical demo instead:

```bash
docker compose up --build -d
```

Open `http://localhost:8080`, enter the token, and use the four guided screens:

1. **Prepare Dataset** — upload images, import/edit captions, verify and freeze the dataset.
2. **Train Model** — set bounded training controls and monitor real worker progress.
3. **Evaluate** — run fixed-condition checks and review generated comparisons when available.
4. **Publish & Download** — publish validated artifacts and download the adapter, reports, and manifests.

The web app, API, and worker start together. The API remains available at `http://localhost:8000`; it is normally proxied through the web app at `/api`.

## Everyday operations

For a GPU deployment, retain both `-f` options:

```bash
docker compose -f compose.yaml -f compose.gpu.yaml ps
docker compose -f compose.yaml -f compose.gpu.yaml logs --tail=200 web api worker
docker compose -f compose.yaml -f compose.gpu.yaml down
```

`down` preserves the named volumes. Do **not** use `down -v` for an ordinary restart: it deletes the database, uploaded data, checkpoints, adapters, reports, and model cache.

## Docker CLI training

The browser is the easiest way to operate the pipeline. The original command-line flow remains available for reproducible GPU acceptance, including IBean preparation/captioning, immutable model pinning, train/resume, evaluation, API checks, and benchmarks. Follow the [detailed Docker CLI guide](docs_en/07-docker-cli-training.md); it starts with GPU preflight, then runs `prepare`/`caption` before the one-shot worker-container training commands.

## Documentation and validation boundary

- [Web Studio guide](docs_en/06-web-studio.md): browser deployment, four-step behavior, recovery, and troubleshooting.
- [Detailed Docker CLI guide](docs_en/07-docker-cli-training.md): full real-GPU runbook, API, and benchmark procedure.
- [Running and API guide](docs_en/04-running-and-api.md): operational and contract reference.
- [Documentation index](docs_en/00-assignment-guide.md): architecture, technical specification, trade-offs, and evidence map.
- [Validation record](VALIDATION.md): checks actually run in this repository environment and their limits. Python/CPU checks do not establish real GPU, SD 1.5 quality, Docker runtime, or benchmark results.
