# LoRA Training Studio

The English workbench guides users through **Prepare Dataset → Train Model → Evaluate → Publish & Download**. Each completed stage waits for explicit confirmation. Closing the browser does not stop running work; reconnect with a token for the same owner to find it in Training Runs.

## Docker deployment

From the repository root, preserve an existing `.env` or create one:

```bash
test -f .env || cp .env.example .env
openssl rand -hex 32
```

Put the generated token in `LORA_API_KEYS`, such as `{"your-random-token":"owner-a"}`. Enter the token itself in the website, not the JSON. The website does not require an admin token. Never commit `.env`.

If Docker socket access requires `sudo` (for example, image builds use `sudo`), prefix every `docker compose` command in this guide—including `up`, `ps`, `logs`, and `down`—with `sudo`. Use `sudo docker compose ...`; never use `sudo dc`, because `dc` is only a shell function in the detailed CLI guide.

For real SD 1.5 training, install the NVIDIA driver and Container Toolkit on a GPU host:

```bash
docker compose -f compose.yaml -f compose.gpu.yaml up --build -d
```

For the CPU technical demo:

```bash
docker compose up --build -d
```

Open **http://localhost:8080** and enter your token. CPU runs are labeled Demo/Test Only and do not establish SD 1.5 quality or generate evaluation pairs.

Three services run together: `web` serves the application and proxies `/api`, `api` handles requests, and `worker` executes stages. The existing API remains on localhost:8000. `pipeline-data` stores the database, uploads and artifacts; `model-cache` stores model weights.

The first GPU run may download SD 1.5 and CLIP, plus BLIP when configured. Allow the worker to reach Hugging Face; optional `HF_TOKEN` in `.env` is passed only to the worker. Download time is not training throughput. Server profiles control the model, hardware, precision, caption policy and quality policy.

The website binds to loopback by default; change the port with `LORA_WEB_PORT`. Team access should use an HTTPS reverse proxy and adjust `LORA_WEB_BIND` as needed. Browser SHA-256 requires Web Crypto in an HTTPS or localhost secure context.

## Four steps

1. **Prepare Dataset:** Select a profile and trigger token, then add JPEG, PNG or static WebP images. Defaults are 100–1,000 files, 20 MiB each and 2 GiB total; actual limits come from the server profile. Import a `captions.json` filename-to-string mapping and edit captions before processing. Missing captions use the server generator. Processing uploads, verifies, deduplicates, splits and captions the data. Inspect statistics, previews and manifests before training. Submitted data is frozen; changes require a new dataset. Retry failed uploads, or after a refresh reselect matching original files to resume missing uploads with hash verification.
2. **Train Model:** Set steps and learning rate; advanced controls expose rank, alpha, batch size, gradient accumulation, checkpoint interval and seed. Starting the stage freezes overrides. The interface shows real stage/step progress, latest loss, throughput, memory and checkpoint information when available. Cancel running jobs if needed; worker retries and checkpoint recovery remain durable.
3. **Evaluate:** Configure prompts, seeds, inference steps and guidance scale, then start evaluation. GPU runs display base/LoRA pairs under identical conditions, metrics and technical/quality verdicts. An uncalibrated policy remains UNVERIFIED. Technical failures and quality rejection cannot be bypassed. CPU semantic metrics and generated pairs are explicitly unavailable.
4. **Publish & Download:** Inspect results and explicitly publish. Publication checks the adapter, manifests, report and checksums. Unverified model downloads require opt-in. Download the adapter, JSON report and associated manifests.

Example caption sidecar:

```json
{
  "photo-001.jpg": "a ceramic cup on a wooden table",
  "photo-002.png": "a blue ceramic bowl"
}
```

Files must have unique names. After invalid/duplicate images are removed, the dataset must still meet unique-image and group-split requirements. Failed/cancelled runs retain diagnostics; new runs can reuse verified datasets. Frozen stages are not rewound, and arbitrary server checkpoint paths are not accepted.

## Operations and compatibility

- `WAITING_FOR_USER` occupies no GPU and consumes no execution timeout/GPU budget, but counts toward active-job limits. Cancel abandoned runs to release that allowance.
- System Status reports API/database/storage/worker health. A reachable website does not prove GPU readiness.
- A 401 requires a valid token; tokens are session-scoped, logout clears caches, and history remains owner-scoped on the server.
- A 409 can mean another tab advanced/cancelled the run; refresh before continuing.
- A 422 identifies invalid fields or pipeline errors. Both pipeline and FastAPI validation errors are displayed.
- OOM and download failures remain explicit; the service does not silently lower parameters or substitute a CPU demo model.

```bash
docker compose ps
docker compose logs --tail=200 web api worker
docker compose down
```

For GPU deployments, also pass `-f compose.yaml -f compose.gpu.yaml` for lifecycle commands. `down` preserves volumes; `down -v` deletes the database, artifacts and cache and is not a restart command.

Old clients that omit `execution_mode` retain the automatic workflow. Website jobs use `manual`. `POST /v1/training-jobs/{id}/advance` requires Bearer authentication and `Idempotency-Key`; submit `stage` with optional `training_overrides` for TRAIN or `evaluation_overrides` for EVALUATE. Overrides freeze at first enqueue; replay does not duplicate work, and out-of-order/conflicting transitions are rejected.

Owner-scoped paginated datasets/jobs, upload status, lightweight job summaries and registered artifacts support the browser. Artifact IDs resolve server-side; arbitrary filesystem paths are rejected. Exact schemas are in runtime `/docs`.

SQLite upgrades are additive and preserve existing automatic jobs. Back up `pipeline-data`, then rebuild/restart API and worker together on the same version. See [VALIDATION.md](../VALIDATION.md) for measured verification and limitations.

## Frontend development and browser regression

Use Node 22.12+ and run `npm ci` and `npm run dev` inside `frontend`. The default API proxy targets port 8000. `npm run build` includes TypeScript checking; `npm test` runs unit tests.

The browser regression uses a real API and independent CPU worker:

```bash
python scripts/web_e2e_fixture.py --metadata-path /tmp/studio-fixture.json
```

Keep this process running. It generates 100 synthetic images and prints fixture paths and a test token. Start the frontend separately with `VITE_API_TARGET=http://127.0.0.1:8765 npm run dev`, then follow [frontend/README.md](../frontend/README.md) to supply `STUDIO_E2E_*` for Playwright. The fixture uses a new temporary directory and retains images, database and logs after shutdown for inspection.
