"""FastAPI surface for the local LoRA training pipeline."""

from __future__ import annotations

import os
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI, Header, Query, Request
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse, Response
from pydantic import BaseModel, ConfigDict, Field

from .common import PipelineError
from .config import Settings
from .observability import HttpMetrics, log_event
from .store import Store


class DatasetFileRequest(BaseModel):
    name: str
    size_bytes: int
    sha256: str
    mime_type: str
    caption: str | None = None


class DatasetRequest(BaseModel):
    files: list[DatasetFileRequest]


class TrainingJobRequest(BaseModel):
    dataset_id: str
    profile_revision_id: str = "local-sd15-v1"
    trigger_token: str
    training_overrides: dict[str, Any] = Field(default_factory=dict)
    execution_mode: Literal["auto", "manual"] = "auto"


class AdvanceJobRequest(BaseModel):
    """The explicit user confirmation that queues one manual stage."""

    model_config = ConfigDict(extra="forbid")

    stage: Literal["TRAIN", "EVALUATE", "PUBLISH"]
    training_overrides: dict[str, Any] = Field(default_factory=dict)
    evaluation_overrides: dict[str, Any] = Field(default_factory=dict)


def create_app(settings: Settings):
    """Create an app without starting a worker or importing ML dependencies."""
    store = Store(settings)
    app = FastAPI(title="Local LoRA Training Pipeline", version="0.1.0")
    app.state.store, app.state.settings = store, settings
    app.state.http_metrics = HttpMetrics()

    def response_error(exc: PipelineError, request: Request, status: int) -> JSONResponse:
        return JSONResponse(
            status_code=status,
            content={
                "error": {
                    "code": exc.code,
                    "message": exc.message,
                    "retryable": exc.retryable,
                    "details": exc.details,
                    "request_id": getattr(request.state, "request_id", None),
                }
            },
        )

    @app.middleware("http")
    async def request_identity(request: Request, call_next):
        request.state.request_id = str(uuid.uuid4())
        started = time.monotonic()
        response = None
        try:
            response = await call_next(request)
            response.headers["X-Request-ID"] = request.state.request_id
            return response
        except PipelineError as exc:
            status = error_status(exc)
            response = response_error(exc, request, status)
            return response
        finally:
            route = getattr(request.scope.get("route"), "path", "__unmatched__")
            status = response.status_code if response is not None else 500
            elapsed = time.monotonic() - started
            app.state.http_metrics.observe(request.method, route, status, elapsed)
            log_event(
                "request_completed",
                request_id=request.state.request_id,
                method=request.method,
                route=route,
                status=status,
                duration_seconds=round(elapsed, 6),
            )

    def error_status(exc: PipelineError) -> int:
        if exc.code == "UNAUTHORIZED":
            return 401
        if exc.code == "NOT_FOUND":
            return 404
        if exc.code in {
            "IDEMPOTENCY_KEY_REUSED",
            "JOB_ALREADY_READY",
            "JOB_TERMINAL",
            "EVALUATION_NOT_READY",
            "DATASET_TERMINAL",
            "UPLOAD_NOT_ALLOWED",
            "MODEL_NOT_READY",
            "ADVANCE_CONFLICT",
            "INVALID_STAGE",
        }:
            return 409
        if exc.code in {"ADMISSION_LIMIT"}:
            return 429
        if exc.code in {"GPU_UNAVAILABLE", "WORKER_ALREADY_RUNNING"}:
            return 503
        return 422

    def owner(authorization: str | None) -> str:
        if not authorization or not authorization.startswith("Bearer "):
            raise PipelineError("UNAUTHORIZED", "Bearer authentication is required")
        token = authorization.removeprefix("Bearer ")
        if token not in settings.api_keys:
            raise PipelineError("UNAUTHORIZED", "Bearer authentication is required")
        return settings.api_keys[token]

    def require_owner(authorization: str | None) -> str:
        return owner(authorization)

    async def body_object(request: Request) -> dict:
        try:
            payload = await request.json()
        except Exception as exc:
            raise PipelineError("INVALID_JSON", "Request body must be a JSON object") from exc
        if not isinstance(payload, dict):
            raise PipelineError("INVALID_JSON", "Request body must be a JSON object")
        return payload

    def idem(request: Request, route: str, owner_id: str, body: dict, operation):
        key = request.headers.get("Idempotency-Key")
        if not key:
            raise PipelineError("IDEMPOTENCY_KEY_REQUIRED", "Idempotency-Key is required for this mutation")
        return store.idempotent(owner_id, route, key, body, operation)

    @app.exception_handler(PipelineError)
    async def direct_pipeline_error(request: Request, exc: PipelineError):
        return response_error(exc, request, error_status(exc))

    @app.get("/health/live")
    def live():
        return {"status": "ok"}

    @app.get("/health/ready")
    def ready():
        database = False
        try:
            with store._connect() as c:
                c.execute("SELECT 1").fetchone()
            database = True
        except Exception:
            pass
        payload = {
            "status": "ok" if database and store.worker_healthy() else "degraded",
            "database": database,
            "storage": store.root.is_dir() and os.access(store.root, os.W_OK),
            "worker": store.worker_healthy(),
        }
        return JSONResponse(
            status_code=200 if payload["status"] == "ok" and payload["storage"] else 503, content=payload
        )

    @app.get("/metrics")
    def metrics(authorization: str | None = Header(default=None)):
        token = (
            authorization.removeprefix("Bearer ")
            if authorization and authorization.startswith("Bearer ")
            else ""
        )
        if token not in settings.admin_keys:
            raise PipelineError("UNAUTHORIZED", "Administrator authentication is required")
        with store._connect() as c:
            jobs = c.execute("SELECT state,COUNT(*) n FROM jobs GROUP BY state").fetchall()
        lines = ["# TYPE lora_jobs gauge"] + [f'lora_jobs{{state="{r["state"]}"}} {r["n"]}' for r in jobs]
        return PlainTextResponse(
            app.state.http_metrics.render() + "\n".join(lines) + "\n",
            media_type="text/plain; version=0.0.4",
        )

    @app.post("/v1/datasets", status_code=201)
    async def create_dataset(
        request: Request, payload: DatasetRequest, authorization: str | None = Header(default=None)
    ):
        actor, payload = require_owner(authorization), payload.model_dump()
        files = payload["files"]
        status, result = idem(
            request, "POST /v1/datasets", actor, payload, lambda: (201, store.create_dataset(actor, files))
        )
        return JSONResponse(status_code=status, content=result)

    @app.get("/v1/datasets")
    def list_datasets(
        limit: int = Query(default=50), offset: int = Query(default=0), authorization: str | None = Header(default=None)
    ):
        return store.datasets(require_owner(authorization), limit=limit, offset=offset)

    @app.put("/v1/datasets/{dataset_id}/files/{file_id}", status_code=204)
    async def upload_file(
        dataset_id: str, file_id: str, request: Request, authorization: str | None = Header(default=None)
    ):
        actor = require_owner(authorization)
        entry, target = store.upload_target(actor, dataset_id, file_id)
        fd, temporary = tempfile.mkstemp(prefix=".upload-", dir=store.objects)
        written = 0
        try:
            with os.fdopen(fd, "wb") as handle:
                async for chunk in request.stream():
                    written += len(chunk)
                    if written > entry["size_bytes"] or written > settings.data.max_file_bytes:
                        raise PipelineError("UPLOAD_TOO_LARGE", "Upload exceeds its declared size")
                    handle.write(chunk)
                handle.flush()
                os.fsync(handle.fileno())
            if written != entry["size_bytes"]:
                raise PipelineError("UPLOAD_SIZE_MISMATCH", "Upload size does not match manifest")
            store.finalize_upload(actor, dataset_id, file_id, Path(temporary))
        except Exception:
            Path(temporary).unlink(missing_ok=True)
            raise
        Path(temporary).unlink(missing_ok=True)
        return Response(status_code=204)

    @app.post("/v1/datasets/{dataset_id}/complete", status_code=202)
    async def complete_dataset(
        dataset_id: str, request: Request, authorization: str | None = Header(default=None)
    ):
        actor, payload = require_owner(authorization), await body_object(request)
        status, result = idem(
            request,
            f"POST /v1/datasets/{dataset_id}/complete",
            actor,
            payload,
            lambda: (202, store.complete_dataset(actor, dataset_id)),
        )
        return JSONResponse(
            status_code=status, content=result, headers={"Location": f"/v1/datasets/{dataset_id}"}
        )

    @app.get("/v1/datasets/{dataset_id}")
    def get_dataset(dataset_id: str, include_files: bool = False, authorization: str | None = Header(default=None)):
        return store.dataset(require_owner(authorization), dataset_id, include_files=include_files)

    @app.get("/v1/training-profiles")
    def profiles(authorization: str | None = Header(default=None)):
        require_owner(authorization)
        profile = store.profile()
        if settings.enable_test_backend:
            tiny = dict(profile)
            tiny["profile_revision_id"] = "local-tiny-v1"
            tiny["profile_key"] = "tiny-test"
            tiny["config"] = {
                **profile["config"],
                "backend": "tiny",
                "resolution": 16,
                "device": "cpu",
                "precision": "fp32",
            }
            return {"profiles": [profile, tiny]}
        return {"profiles": [profile]}

    @app.post("/v1/training-jobs", status_code=202)
    async def create_job(
        request: Request, payload: TrainingJobRequest, authorization: str | None = Header(default=None)
    ):
        actor, payload = require_owner(authorization), payload.model_dump()
        status, result = idem(
            request, "POST /v1/training-jobs", actor, payload, lambda: (202, store.create_job(actor, payload))
        )
        return JSONResponse(status_code=status, content=result, headers={"Location": result["status_url"]})

    @app.get("/v1/training-jobs")
    def list_jobs(
        limit: int = Query(default=50), offset: int = Query(default=0), state: str | None = None,
        authorization: str | None = Header(default=None),
    ):
        return store.jobs(require_owner(authorization), limit=limit, offset=offset, state=state)

    @app.get("/v1/training-jobs/{job_id}")
    def get_job(job_id: str, view: str | None = None, authorization: str | None = Header(default=None)):
        if view not in {None, "summary"}:
            raise PipelineError("INVALID_VIEW", "view must be summary")
        return store.job(require_owner(authorization), job_id, summary=view == "summary")

    @app.post("/v1/training-jobs/{job_id}/advance")
    async def advance_job(
        job_id: str,
        request: Request,
        payload: AdvanceJobRequest,
        authorization: str | None = Header(default=None),
    ):
        actor, payload = require_owner(authorization), payload.model_dump(exclude_unset=True)
        status, result = idem(
            request,
            f"POST /v1/training-jobs/{job_id}/advance",
            actor,
            payload,
            lambda: (202, store.advance_job(actor, job_id, payload)),
        )
        return JSONResponse(status_code=status, content=result, headers={"Location": result["status_url"]})

    @app.post("/v1/training-jobs/{job_id}/cancel")
    async def cancel_job(job_id: str, request: Request, authorization: str | None = Header(default=None)):
        actor, payload = require_owner(authorization), await body_object(request)
        status, result = idem(
            request,
            f"POST /v1/training-jobs/{job_id}/cancel",
            actor,
            payload,
            lambda: (202, store.cancel_job(actor, job_id)),
        )
        return JSONResponse(status_code=status, content=result)

    @app.get("/v1/training-jobs/{job_id}/evaluation")
    def get_evaluation(job_id: str, authorization: str | None = Header(default=None)):
        return store.evaluation(require_owner(authorization), job_id)

    @app.get("/v1/training-jobs/{job_id}/artifacts")
    def list_artifacts(job_id: str, authorization: str | None = Header(default=None)):
        actor = require_owner(authorization)
        artifacts = store.artifacts_for_job(actor, job_id)
        return {"artifacts": [{key: value for key, value in item.items() if key != "path"} | {"url": f"/v1/training-jobs/{job_id}/artifacts/{item['id']}"} for item in artifacts]}

    @app.get("/v1/training-jobs/{job_id}/artifacts/{artifact_id}")
    def download_artifact(job_id: str, artifact_id: str, authorization: str | None = Header(default=None)):
        artifact = store.artifact_for_job(require_owner(authorization), job_id, artifact_id)
        return FileResponse(artifact["path"], filename=Path(artifact["path"]).name, media_type=artifact["media_type"])

    @app.get("/v1/models/{model_id}")
    def get_model(model_id: str, authorization: str | None = Header(default=None)):
        return store.model(require_owner(authorization), model_id)

    @app.get("/v1/models/{model_id}/download")
    def download_model(
        model_id: str, allow_unverified: bool = False, authorization: str | None = Header(default=None)
    ):
        model = store.model(require_owner(authorization), model_id)
        if model["state"] != "READY" and not allow_unverified:
            raise PipelineError(
                "MODEL_NOT_READY", "Set allow_unverified=true to download an unverified model"
            )
        return FileResponse(
            model["adapter_path"], filename="adapter.safetensors", media_type="application/octet-stream"
        )

    return app
