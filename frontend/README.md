# LoRA Training Studio frontend

The browser client guides a user through dataset processing, manual training confirmation, evaluation, and publishing. It uses a same-origin `/api` proxy in production; the bearer token stays in `sessionStorage` and never appears in artifact URLs.

## Local development

Node 22.12 or newer is required. Install and validate the client with:

```sh
npm ci
npm run build
npm test
npm run dev
```

The Vite dev server proxies `/api` to `http://localhost:8000`. Set `VITE_API_TARGET=http://127.0.0.1:8765` to use the real-backend fixture server.

The optional Playwright flow uses a real API and worker. Start `scripts/web_e2e_fixture.py` from the repository root, then supply its exported paths and token:

```sh
STUDIO_E2E_BASE_URL=http://127.0.0.1:5173 \
STUDIO_E2E_TOKEN=studio-test-token \
STUDIO_E2E_IMAGES_DIR=/tmp/... \
STUDIO_E2E_CAPTIONS_PATH=/tmp/.../captions.json \
npm run e2e
```

The production Docker image builds with Node 22 and serves the Vite bundle with Nginx. Nginx proxies `/api/` to the Compose `api` service, disables upload request buffering, allows requests up to 2 GiB, and falls back to `index.html` for client routes. The API still enforces the profile's per-file upload limit (20 MiB by default).

`src/App.tsx` contains route-level workflow UI, `src/ArtifactViews.tsx` provides paginated image previews and authenticated downloads, `src/api.ts` contains request/idempotency/upload helpers, and `src/App.module.css` defines the responsive visual system. The SHA-256 worker hashes one image at a time off the main thread.

Install a browser before the first Playwright run with `npx playwright install chromium`, or set `PLAYWRIGHT_CHROMIUM_EXECUTABLE` to an existing Chromium executable. The real-backend tests upload all 100 fixture PNGs, edit a caption, verify each manual gate and reload, download artifacts, check owner isolation, and recover an interrupted upload without creating another dataset. A separate browser test supplies an INVALID response to check form recovery. Test screenshots and failure traces are written to ignored `test-results/`.
