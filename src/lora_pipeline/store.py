"""SQLite persistence for the single-host training service.

The store deliberately keeps database transactions short: model work and file IO
always happen outside a transaction.  Every operation opens its own connection so
the API and worker can safely run in different processes.
"""

from __future__ import annotations

import contextlib
import fcntl
import json
import os
import sqlite3
import time
import threading
import uuid
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Iterator

from .common import PipelineError, digest_json, sha256_file
from .config import Settings


TERMINAL_JOBS = {"READY", "FAILED", "QUALITY_REJECTED", "CANCELLED", "COMPLETED_UNVERIFIED"}
GPU_STAGES = {"CAPTION", "TRAIN", "EVALUATE"}
STAGE_CYCLE = ("EVALUATE", "TRAIN", "CAPTION", "TRAIN")


def _now() -> float:
    return time.time()


def discover_gpu_uuids() -> list[str]:
    """Return NVIDIA UUIDs without importing CUDA libraries.

    An empty list is a normal CPU-only result.  The command is intentionally
    best-effort because it is also used by preflight tooling on developer hosts.
    """
    import subprocess

    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return []
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def device_lock_path(settings: Settings, device_uuid: str) -> Path:
    """The shared lock name used by worker and direct CLI GPU operations."""
    safe = "".join(ch for ch in device_uuid if ch.isalnum() or ch in "-_")
    if not safe:
        raise ValueError("Invalid device UUID")
    return Path(settings.data_dir).resolve() / "locks" / f"gpu-{safe}.lock"


@contextlib.contextmanager
def device_lock(settings: Settings, device_uuid: str, *, blocking: bool = False):
    path = device_lock_path(settings, device_uuid)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as handle:
        flags = fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB)
        try:
            fcntl.flock(handle.fileno(), flags)
        except BlockingIOError as exc:
            raise PipelineError("GPU_BUSY", f"GPU {device_uuid} is already in use", retryable=True) from exc
        try:
            yield path
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _json(value: Any) -> str:
    if is_dataclass(value):
        value = asdict(value)
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def _obj(value: str | None, default: Any = None) -> Any:
    return default if value is None else json.loads(value)


class Store:
    """Durable metadata store and local-object layout.

    SQLite is intentionally a *single-host* queue. ``BEGIN IMMEDIATE`` serializes
    admission, dispatch, idempotency and state transitions without holding a lock
    while images are decoded or a model is trained.
    """

    def __init__(self, settings: Settings):
        self.settings = settings
        self.root = Path(settings.data_dir).resolve()
        self.objects = self.root / "objects"
        self.artifacts = self.root / "artifacts"
        self.locks = self.root / "locks"
        self._local = threading.local()
        for directory in (self.root, self.objects, self.artifacts, self.locks):
            directory.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _requires_gpu(self, stage: str, profile: dict | None = None) -> bool:
        # Template captions and CPU BLIP are ordinary bounded CPU work.  Training
        # and evaluation may run on explicit CPU test slots, but are scheduled
        # through a resource slot so production CUDA jobs remain exclusive.
        caption = (profile or {}).get("caption", {})
        mode = caption.get("mode", self.settings.caption_mode)
        device = caption.get("device", self.settings.caption_device)
        return stage in {"TRAIN", "EVALUATE"} or (
            stage == "CAPTION" and mode == "blip" and device.startswith("cuda")
        )

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.settings.db_path, timeout=10, isolation_level=None)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys=ON")
        # DELETE/FULL is the conservative choice for a durable, single-host
        # queue and avoids network/WAL-shm assumptions in mounted volumes.
        connection.execute("PRAGMA journal_mode=DELETE")
        connection.execute("PRAGMA synchronous=FULL")
        connection.execute("PRAGMA busy_timeout=10000")
        return connection

    @contextlib.contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        nested = getattr(self._local, "connection", None)
        if nested is not None:
            yield nested
            return
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            self._local.connection = connection
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            self._local.connection = None
            connection.close()

    @contextlib.contextmanager
    def reader(self) -> Iterator[sqlite3.Connection]:
        nested = getattr(self._local, "connection", None)
        if nested is not None:
            yield nested
            return
        connection = self._connect()
        try:
            yield connection
        finally:
            connection.close()

    def _init_db(self) -> None:
        with self.reader() as c:
            c.executescript(
                """
                CREATE TABLE IF NOT EXISTS datasets (
                  id TEXT PRIMARY KEY, owner_id TEXT NOT NULL, state TEXT NOT NULL,
                  declared_count INTEGER NOT NULL, submitted_manifest_sha256 TEXT NOT NULL,
                  frozen_manifest_path TEXT, frozen_manifest_sha256 TEXT,
                  valid_count INTEGER, invalid_count INTEGER, verification_error TEXT,
                  created_at REAL NOT NULL, updated_at REAL NOT NULL, completed_at REAL
                );
                CREATE TABLE IF NOT EXISTS dataset_files (
                  id TEXT PRIMARY KEY, dataset_id TEXT NOT NULL REFERENCES datasets(id) ON DELETE CASCADE,
                  original_name TEXT NOT NULL, object_key TEXT NOT NULL UNIQUE, size_bytes INTEGER NOT NULL,
                  sha256 TEXT NOT NULL, mime_type TEXT NOT NULL, caption TEXT,
                  uploaded INTEGER NOT NULL DEFAULT 0, verification_status TEXT,
                  rejection_code TEXT, created_at REAL NOT NULL,
                  UNIQUE(dataset_id, original_name)
                );
                CREATE TABLE IF NOT EXISTS jobs (
                  id TEXT PRIMARY KEY, owner_id TEXT NOT NULL, dataset_id TEXT NOT NULL REFERENCES datasets(id),
                  state TEXT NOT NULL, current_stage TEXT, candidate_index INTEGER NOT NULL DEFAULT 0,
                  trigger_token TEXT NOT NULL, profile_json TEXT NOT NULL, training_overrides_json TEXT NOT NULL,
                  prepared_json TEXT, input_json TEXT, training_json TEXT, evaluation_json TEXT,
                  error_code TEXT, error_message TEXT, cancel_requested_at REAL,
                  gpu_seconds_charged REAL NOT NULL DEFAULT 0, created_at REAL NOT NULL, updated_at REAL NOT NULL,
                  model_id TEXT
                );
                CREATE TABLE IF NOT EXISTS stage_tasks (
                  id TEXT PRIMARY KEY, job_id TEXT NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
                  candidate_index INTEGER NOT NULL DEFAULT 0, stage TEXT NOT NULL, state TEXT NOT NULL,
                  attempt_count INTEGER NOT NULL DEFAULT 0, ready_at REAL NOT NULL, active_token INTEGER,
                  lease_expires_at REAL, deadline_at REAL, checkpoint_path TEXT, checkpoint_sha256 TEXT,
                  output_json TEXT, created_at REAL NOT NULL, updated_at REAL NOT NULL,
                  UNIQUE(job_id, candidate_index, stage)
                );
                CREATE TABLE IF NOT EXISTS task_attempts (
                  id TEXT PRIMARY KEY, task_id TEXT NOT NULL REFERENCES stage_tasks(id) ON DELETE CASCADE,
                  attempt_no INTEGER NOT NULL, fencing_token INTEGER NOT NULL, gpu_slot TEXT,
                  pid INTEGER, process_start_time REAL, process_group_id INTEGER,
                  state TEXT NOT NULL, started_at REAL NOT NULL, last_heartbeat_at REAL NOT NULL,
                  lease_expires_at REAL NOT NULL, ended_at REAL, error_code TEXT, error_message TEXT,
                  progress_json TEXT, charged_seconds REAL NOT NULL DEFAULT 0,
                  UNIQUE(task_id, attempt_no), UNIQUE(task_id, fencing_token)
                );
                CREATE TABLE IF NOT EXISTS models (
                  id TEXT PRIMARY KEY, job_id TEXT NOT NULL UNIQUE REFERENCES jobs(id), owner_id TEXT NOT NULL,
                  state TEXT NOT NULL, adapter_path TEXT NOT NULL, adapter_sha256 TEXT NOT NULL,
                  manifest_path TEXT NOT NULL, manifest_sha256 TEXT NOT NULL, report_json TEXT,
                  test_only INTEGER NOT NULL, created_at REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS idempotency (
                  owner_id TEXT NOT NULL, route TEXT NOT NULL, key TEXT NOT NULL, request_sha256 TEXT NOT NULL,
                  status_code INTEGER NOT NULL, response_json TEXT NOT NULL, created_at REAL NOT NULL,
                  PRIMARY KEY(owner_id, route, key)
                );
                CREATE TABLE IF NOT EXISTS scheduler_state (
                  id INTEGER PRIMARY KEY CHECK(id=1), stage_cursor INTEGER NOT NULL, owner_cursors_json TEXT NOT NULL,
                  leader_epoch INTEGER NOT NULL, worker_heartbeat REAL
                );
                INSERT OR IGNORE INTO scheduler_state(id,stage_cursor,owner_cursors_json,leader_epoch) VALUES(1,0,'{}',1);
                CREATE INDEX IF NOT EXISTS task_ready ON stage_tasks(state, ready_at);
                CREATE INDEX IF NOT EXISTS jobs_owner ON jobs(owner_id, created_at);
                """
            )

    # ----- generic helpers -------------------------------------------------
    def _row(self, row: sqlite3.Row | None) -> dict | None:
        return dict(row) if row is not None else None

    def _owned(self, c: sqlite3.Connection, table: str, resource_id: str, owner: str) -> sqlite3.Row:
        row = c.execute(f"SELECT * FROM {table} WHERE id=? AND owner_id=?", (resource_id, owner)).fetchone()
        if row is None:
            raise PipelineError("NOT_FOUND", "Resource was not found")
        return row

    def idempotent(self, owner: str, route: str, key: str | None, body: dict, action):
        """Run an action once and persist the returned HTTP status/body."""
        if not key:
            return action()
        request_hash = digest_json(body)
        with self.transaction() as c:
            prior = c.execute(
                "SELECT * FROM idempotency WHERE owner_id=? AND route=? AND key=?", (owner, route, key)
            ).fetchone()
            if prior:
                if prior["request_sha256"] != request_hash:
                    raise PipelineError(
                        "IDEMPOTENCY_KEY_REUSED", "Idempotency key was reused with another request"
                    )
                return prior["status_code"], _obj(prior["response_json"])
            status, response = action()
            c.execute(
                "INSERT INTO idempotency VALUES(?,?,?,?,?,?,?)",
                (owner, route, key, request_hash, status, _json(response), _now()),
            )
            return status, response

    # ----- datasets and local object storage ------------------------------
    def create_dataset(self, owner: str, files: list[dict]) -> dict:
        if not self.settings.data.min_images <= len(files) <= self.settings.data.max_images:
            raise PipelineError(
                "INVALID_DATASET_SIZE",
                f"Dataset must declare {self.settings.data.min_images}-{self.settings.data.max_images} files",
            )
        names: set[str] = set()
        total = 0
        normalized: list[dict] = []
        for item in files:
            name, size, digest, mime = (
                item.get("name"),
                item.get("size_bytes"),
                item.get("sha256"),
                item.get("mime_type"),
            )
            if (
                not isinstance(name, str)
                or not name
                or name in names
                or "/" in name
                or "\\" in name
                or name in {".", ".."}
            ):
                raise PipelineError("INVALID_FILE_NAME", "File names must be unique plain names")
            if not isinstance(size, int) or not 0 < size <= self.settings.data.max_file_bytes:
                raise PipelineError("INVALID_FILE_SIZE", "Declared file size is outside the configured limit")
            if (
                not isinstance(digest, str)
                or len(digest) != 64
                or any(ch not in "0123456789abcdef" for ch in digest)
            ):
                raise PipelineError("INVALID_CHECKSUM", "sha256 must be 64 lower-case hexadecimal characters")
            if mime not in {"image/jpeg", "image/png", "image/webp"}:
                raise PipelineError("UNSUPPORTED_MEDIA_TYPE", "Only JPEG, PNG, and static WebP are accepted")
            caption = item.get("caption")
            if caption is not None and (
                not isinstance(caption, str)
                or not caption.strip()
                or len(caption) > 512
                or any(ord(ch) < 32 or ord(ch) == 127 for ch in caption)
            ):
                raise PipelineError(
                    "INVALID_CAPTION", "Caption must be printable text of at most 512 characters"
                )
            names.add(name)
            total += size
            normalized.append(
                {"name": name, "size_bytes": size, "sha256": digest, "mime_type": mime, "caption": caption}
            )
        if total > self.settings.data.max_total_bytes:
            raise PipelineError("DATASET_TOO_LARGE", "Dataset exceeds total upload limit")
        dataset_id, now = str(uuid.uuid4()), _now()
        manifest_hash = digest_json(normalized)
        with self.transaction() as c:
            used = c.execute(
                "SELECT COALESCE(SUM(size_bytes),0) FROM dataset_files f JOIN datasets d ON d.id=f.dataset_id WHERE d.owner_id=?",
                (owner,),
            ).fetchone()[0]
            if used + total > self.settings.storage_quota_bytes:
                raise PipelineError("STORAGE_QUOTA_EXCEEDED", "Storage quota would be exceeded")
            c.execute(
                "INSERT INTO datasets VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    dataset_id,
                    owner,
                    "UPLOADING",
                    len(files),
                    manifest_hash,
                    None,
                    None,
                    None,
                    None,
                    None,
                    now,
                    now,
                    None,
                ),
            )
            rows = []
            for item in normalized:
                file_id = str(uuid.uuid4())
                rows.append(
                    (
                        file_id,
                        dataset_id,
                        item["name"],
                        str(uuid.uuid4()),
                        item["size_bytes"],
                        item["sha256"],
                        item["mime_type"],
                        item["caption"],
                        0,
                        None,
                        None,
                        now,
                    )
                )
            c.executemany("INSERT INTO dataset_files VALUES(?,?,?,?,?,?,?,?,?,?,?,?)", rows)
        return self.dataset(owner, dataset_id, include_files=True)

    def dataset(self, owner: str, dataset_id: str, *, include_files: bool = False) -> dict:
        with self.reader() as c:
            row = self._owned(c, "datasets", dataset_id, owner)
            result = dict(row)
            result.pop("owner_id", None)
            if include_files:
                result["files"] = [
                    dict(x)
                    for x in c.execute(
                        "SELECT id,original_name AS name,size_bytes,sha256,mime_type,caption,uploaded,verification_status,rejection_code FROM dataset_files WHERE dataset_id=? ORDER BY original_name",
                        (dataset_id,),
                    )
                ]
            return result

    def upload_target(self, owner: str, dataset_id: str, file_id: str) -> tuple[dict, Path]:
        with self.reader() as c:
            self._owned(c, "datasets", dataset_id, owner)
            row = c.execute(
                "SELECT f.*,d.state FROM dataset_files f JOIN datasets d ON d.id=f.dataset_id WHERE f.id=? AND f.dataset_id=?",
                (file_id, dataset_id),
            ).fetchone()
            if row is None or row["state"] != "UPLOADING":
                raise PipelineError("UPLOAD_NOT_ALLOWED", "Dataset is not accepting uploads")
            return dict(row), self.objects / row["object_key"]

    def mark_uploaded(self, owner: str, dataset_id: str, file_id: str, path: Path) -> None:
        expected, target = self.upload_target(owner, dataset_id, file_id)
        if path.resolve() != target.resolve() or not target.is_file():
            raise PipelineError("UPLOAD_FAILED", "Upload object is missing")
        actual = sha256_file(target)
        if actual != expected["sha256"] or target.stat().st_size != expected["size_bytes"]:
            target.unlink(missing_ok=True)
            raise PipelineError("CHECKSUM_MISMATCH", "Uploaded bytes do not match the declared file")
        with self.transaction() as c:
            self._owned(c, "datasets", dataset_id, owner)
            c.execute(
                "UPDATE dataset_files SET uploaded=1 WHERE id=? AND dataset_id=?", (file_id, dataset_id)
            )

    def finalize_upload(self, owner: str, dataset_id: str, file_id: str, temporary: Path) -> None:
        """Validate then immutably link a streamed upload into its object key.

        A client retry may reuse an already validated object only when it is byte
        identical.  We never replace an object key, including during the small
        window between a caller's validation and ``complete`` freezing a dataset.
        """
        with self.transaction() as c:
            row = c.execute(
                """SELECT f.*,d.state,d.owner_id FROM dataset_files f
                JOIN datasets d ON d.id=f.dataset_id WHERE f.id=? AND f.dataset_id=?""",
                (file_id, dataset_id),
            ).fetchone()
            if not row or row["owner_id"] != owner:
                raise PipelineError("NOT_FOUND", "Resource was not found")
            if row["state"] != "UPLOADING":
                raise PipelineError("UPLOAD_NOT_ALLOWED", "Dataset is not accepting uploads")
            if (
                not temporary.is_file()
                or temporary.stat().st_size != row["size_bytes"]
                or sha256_file(temporary) != row["sha256"]
            ):
                raise PipelineError("CHECKSUM_MISMATCH", "Uploaded bytes do not match the declared file")
            target = self.objects / row["object_key"]
            try:
                os.link(temporary, target)
            except FileExistsError:
                if (
                    not target.is_file()
                    or target.stat().st_size != row["size_bytes"]
                    or sha256_file(target) != row["sha256"]
                ):
                    raise PipelineError("UPLOAD_CONFLICT", "An object key has unexpected bytes")
            c.execute("UPDATE dataset_files SET uploaded=1 WHERE id=?", (file_id,))

    def complete_dataset(self, owner: str, dataset_id: str) -> dict:
        with self.transaction() as c:
            row = self._owned(c, "datasets", dataset_id, owner)
            if row["state"] == "VERIFYING":
                return self._dataset_conn(c, dataset_id, False)
            if row["state"] != "UPLOADING":
                raise PipelineError("DATASET_TERMINAL", "Dataset cannot be completed in its current state")
            missing = c.execute(
                "SELECT COUNT(*) FROM dataset_files WHERE dataset_id=? AND uploaded=0", (dataset_id,)
            ).fetchone()[0]
            if missing:
                raise PipelineError(
                    "UPLOAD_INCOMPLETE", "All declared files must be uploaded", details={"missing": missing}
                )
            c.execute("UPDATE datasets SET state='VERIFYING',updated_at=? WHERE id=?", (_now(), dataset_id))
            return self._dataset_conn(c, dataset_id, False)

    def _dataset_conn(self, c: sqlite3.Connection, dataset_id: str, include_files: bool) -> dict:
        result = dict(c.execute("SELECT * FROM datasets WHERE id=?", (dataset_id,)).fetchone())
        result.pop("owner_id", None)
        if include_files:
            result["files"] = [
                dict(x) for x in c.execute("SELECT * FROM dataset_files WHERE dataset_id=?", (dataset_id,))
            ]
        return result

    def claim_verification(self) -> dict | None:
        with self.transaction() as c:
            row = c.execute(
                "SELECT * FROM datasets WHERE state='VERIFYING' ORDER BY created_at LIMIT 1"
            ).fetchone()
            if not row:
                return None
            c.execute("UPDATE datasets SET state='VERIFYING',updated_at=? WHERE id=?", (_now(), row["id"]))
            files = [
                dict(x)
                for x in c.execute(
                    "SELECT * FROM dataset_files WHERE dataset_id=? ORDER BY original_name", (row["id"],)
                )
            ]
            return {"dataset": dict(row), "files": files}

    def finish_verification(
        self,
        dataset_id: str,
        *,
        valid: bool,
        error: PipelineError | None = None,
        manifest: dict | None = None,
    ) -> None:
        with self.transaction() as c:
            state = "COMPLETED" if valid else "INVALID"
            now = _now()
            c.execute(
                "UPDATE datasets SET state=?,valid_count=?,invalid_count=?,verification_error=?,frozen_manifest_path=?,frozen_manifest_sha256=?,completed_at=?,updated_at=? WHERE id=?",
                (
                    state,
                    manifest.get("valid_count") if manifest else None,
                    manifest.get("invalid_count") if manifest else None,
                    _json({"code": error.code, "message": error.message, "details": error.details})
                    if error
                    else None,
                    manifest.get("manifest_path") if manifest else None,
                    manifest.get("manifest_sha256") if manifest else None,
                    now,
                    now,
                    dataset_id,
                ),
            )

    # ----- jobs and idempotent admission ----------------------------------
    def profile(self) -> dict:
        train = self.settings.train.snapshot()
        return {
            "profile_revision_id": "local-sd15-v1",
            "profile_key": "style-lora",
            "revision": 1,
            "display_name": "Local SD 1.5 LoRA",
            "base_model": train["model_name"],
            "config": train,
            "data": asdict(self.settings.data),
            "evaluation": asdict(self.settings.evaluation),
            "caption": {
                "mode": self.settings.caption_mode,
                "device": self.settings.caption_device,
                "model": self.settings.caption_model,
                "revision": self.settings.caption_revision,
            },
            "limits": {
                "max_stage_seconds": self.settings.max_stage_seconds,
                "max_job_gpu_seconds": self.settings.max_job_gpu_seconds,
            },
            "caption_enabled": True,
            "quality_policy_ready": bool(self.settings.evaluation.quality_policy),
        }

    def create_job(self, owner: str, body: dict) -> dict:
        dataset_id = body.get("dataset_id")
        if not isinstance(dataset_id, str):
            raise PipelineError("INVALID_REQUEST", "dataset_id is required")
        trigger = body.get("trigger_token")
        if not isinstance(trigger, str) or not trigger.strip() or len(trigger) > 128:
            raise PipelineError("INVALID_TRIGGER_TOKEN", "trigger_token must be 1-128 characters")
        profile_id = body.get("profile_revision_id", "local-sd15-v1")
        if profile_id not in {"local-sd15-v1", "local-tiny-v1"} or (
            profile_id == "local-tiny-v1" and not self.settings.enable_test_backend
        ):
            raise PipelineError("UNSUPPORTED_PROFILE", "Only the local profile is available")
        overrides = body.get("training_overrides", {})
        if not isinstance(overrides, dict):
            raise PipelineError("INVALID_OVERRIDES", "training_overrides must be an object")
        # Hardware/backend/precision are operator-controlled profile properties;
        # callers may tune only bounded training knobs.
        allowed = {
            "max_steps",
            "learning_rate",
            "rank",
            "lora_alpha",
            "checkpoint_every",
            "seed",
            "batch_size",
            "gradient_accumulation_steps",
        }
        unknown = set(overrides) - allowed
        if unknown:
            raise PipelineError(
                "INVALID_OVERRIDES", "Unsupported training override", details={"fields": sorted(unknown)}
            )
        config = self.settings.train.snapshot()
        if profile_id == "local-tiny-v1":
            config.update({"backend": "tiny", "resolution": 16, "device": "cpu", "precision": "fp32"})
        config.update(overrides)
        # Validate before persisting the frozen snapshot.
        from .config import TrainConfig

        try:
            TrainConfig(**config)
        except (TypeError, ValueError) as exc:
            raise PipelineError("INVALID_OVERRIDES", str(exc)) from exc
        job_id, now = str(uuid.uuid4()), _now()
        with self.transaction() as c:
            dataset = self._owned(c, "datasets", dataset_id, owner)
            if dataset["state"] != "COMPLETED":
                raise PipelineError("DATASET_INVALID", "Dataset must finish validation before training")
            owned = c.execute(
                "SELECT COUNT(*) FROM jobs WHERE owner_id=? AND state NOT IN ('READY','FAILED','QUALITY_REJECTED','CANCELLED','COMPLETED_UNVERIFIED')",
                (owner,),
            ).fetchone()[0]
            global_count = c.execute(
                "SELECT COUNT(*) FROM jobs WHERE state NOT IN ('READY','FAILED','QUALITY_REJECTED','CANCELLED','COMPLETED_UNVERIFIED')"
            ).fetchone()[0]
            if owned >= self.settings.owner_job_limit or global_count >= self.settings.global_job_limit:
                raise PipelineError(
                    "ADMISSION_LIMIT", "Training queue is at its configured limit", retryable=True
                )
            profile = self.profile()
            if profile_id == "local-tiny-v1":
                profile.update(
                    {"profile_revision_id": profile_id, "profile_key": "tiny-test", "config": config}
                )
            else:
                profile["config"] = config
            c.execute(
                """INSERT INTO jobs(id,owner_id,dataset_id,state,current_stage,candidate_index,trigger_token,
                profile_json,training_overrides_json,created_at,updated_at)
                VALUES(?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    job_id,
                    owner,
                    dataset_id,
                    "ACCEPTED",
                    "PREPARE",
                    0,
                    trigger.strip(),
                    _json(profile),
                    _json(overrides),
                    now,
                    now,
                ),
            )
            self._insert_task(c, job_id, "PREPARE", now)
        return {
            "job_id": job_id,
            "state": "ACCEPTED",
            "current_stage": "PREPARE",
            "status_url": f"/v1/training-jobs/{job_id}",
        }

    def _insert_task(
        self, c: sqlite3.Connection, job_id: str, stage: str, ready_at: float | None = None
    ) -> str:
        task_id, now = str(uuid.uuid4()), _now()
        c.execute(
            "INSERT INTO stage_tasks(id,job_id,stage,state,ready_at,created_at,updated_at) VALUES(?,?,?,'PENDING',?,?,?)",
            (task_id, job_id, stage, ready_at or now, now, now),
        )
        return task_id

    def job(self, owner: str, job_id: str) -> dict:
        with self.reader() as c:
            row = self._owned(c, "jobs", job_id, owner)
            result = dict(row)
            for field in (
                "profile_json",
                "training_overrides_json",
                "prepared_json",
                "input_json",
                "training_json",
                "evaluation_json",
            ):
                result[field.removesuffix("_json")] = _obj(result.pop(field), None)
            result.pop("owner_id", None)
            result["stages"] = {}
            for stage in c.execute(
                "SELECT id,stage,state,attempt_count FROM stage_tasks WHERE job_id=?", (job_id,)
            ):
                attempt = c.execute(
                    "SELECT progress_json FROM task_attempts WHERE task_id=? ORDER BY attempt_no DESC LIMIT 1",
                    (stage["id"],),
                ).fetchone()
                result["stages"][stage["stage"]] = {
                    "state": stage["state"],
                    "attempt_count": stage["attempt_count"],
                    "progress": _obj(attempt["progress_json"], {}) if attempt else {},
                }
            return result

    def task_context(self, task: dict) -> dict:
        """Internal immutable snapshot for a claimed task; never exposed by API."""
        with self.reader() as c:
            job = c.execute("SELECT * FROM jobs WHERE id=?", (task["job_id"],)).fetchone()
            if not job:
                raise PipelineError("NOT_FOUND", "Job was not found")
            dataset = c.execute("SELECT * FROM datasets WHERE id=?", (job["dataset_id"],)).fetchone()
            files = c.execute(
                "SELECT * FROM dataset_files WHERE dataset_id=? ORDER BY original_name", (job["dataset_id"],)
            ).fetchall()
            result = dict(job)
            for field in (
                "profile_json",
                "training_overrides_json",
                "prepared_json",
                "input_json",
                "training_json",
                "evaluation_json",
            ):
                result[field.removesuffix("_json")] = _obj(result.pop(field), None)
            return {"task": task, "job": result, "dataset": dict(dataset), "files": [dict(x) for x in files]}

    def cancel_job(self, owner: str, job_id: str) -> dict:
        with self.transaction() as c:
            job = self._owned(c, "jobs", job_id, owner)
            if job["state"] == "READY":
                raise PipelineError("JOB_ALREADY_READY", "Published jobs cannot be cancelled")
            if job["state"] in TERMINAL_JOBS:
                raise PipelineError("JOB_TERMINAL", "Job is already terminal")
            now = _now()
            c.execute(
                "UPDATE jobs SET state='CANCEL_REQUESTED',cancel_requested_at=?,updated_at=? WHERE id=?",
                (now, now, job_id),
            )
            c.execute(
                "UPDATE stage_tasks SET state='CANCELLED',updated_at=? WHERE job_id=? AND state IN ('PENDING','RETRY_WAIT')",
                (now, job_id),
            )
            return {"job_id": job_id, "state": "CANCEL_REQUESTED"}

    # ----- scheduling / attempts ------------------------------------------
    def heartbeat(self) -> None:
        with self.transaction() as c:
            c.execute("UPDATE scheduler_state SET worker_heartbeat=? WHERE id=1", (_now(),))

    def worker_ready(self) -> bool:
        with self.reader() as c:
            value = c.execute("SELECT worker_heartbeat FROM scheduler_state WHERE id=1").fetchone()[0]
            return value is not None and _now() - value <= self.settings.lease_seconds * 2

    worker_healthy = worker_ready

    def claim_task(self, slots: list[str]) -> dict | None:
        """Claim one runnable stage and mint its monotonic fencing token."""
        with self.transaction() as c:
            self._expire_conn(c)
            guard = c.execute("SELECT * FROM scheduler_state WHERE id=1").fetchone()
            pending = c.execute(
                """SELECT t.*,j.owner_id,j.state AS job_state,j.profile_json,j.gpu_seconds_charged
                FROM stage_tasks t JOIN jobs j ON j.id=t.job_id
                WHERE t.state IN ('PENDING','RETRY_WAIT') AND t.ready_at<=? AND j.state IN ('ACCEPTED','RUNNING')
                ORDER BY t.ready_at,t.created_at""",
                (_now(),),
            ).fetchall()
            if not pending:
                return None
            by_stage: dict[str, list[sqlite3.Row]] = {}
            for task in pending:
                by_stage.setdefault(task["stage"], []).append(task)
            cpu_running = c.execute(
                "SELECT COUNT(*) FROM task_attempts WHERE state='RUNNING' AND gpu_slot IS NULL"
            ).fetchone()[0]
            cpu_room = cpu_running < self.settings.cpu_slots
            chosen = (
                None
                if not cpu_room
                else next(
                    (
                        task
                        for task in pending
                        if not self._requires_gpu(task["stage"], _obj(task["profile_json"]))
                    ),
                    None,
                )
            )
            cursor = guard["stage_cursor"]
            if chosen is None:
                for offset in range(4):
                    index = (cursor + offset) % 4
                    stage = STAGE_CYCLE[index]
                    candidates = [
                        item
                        for item in by_stage.get(stage, [])
                        if (self._requires_gpu(stage, _obj(item["profile_json"])) and slots)
                        or (not self._requires_gpu(stage, _obj(item["profile_json"])) and cpu_room)
                    ]
                    if not candidates:
                        continue
                    if candidates:
                        # owner round-robin within a class; tasks of the owner stay FIFO.
                        cursors = _obj(guard["owner_cursors_json"], {})
                        owners = sorted({t["owner_id"] for t in candidates})
                        previous = cursors.get(stage)
                        owner = (
                            next((x for x in owners if x > previous), owners[0]) if previous else owners[0]
                        )
                        chosen = next(t for t in candidates if t["owner_id"] == owner)
                        cursors[stage] = owner
                        c.execute(
                            "UPDATE scheduler_state SET stage_cursor=?,owner_cursors_json=? WHERE id=1",
                            ((index + 1) % 4, _json(cursors)),
                        )
                        break
            if chosen is None:
                return None
            # Respect conservative per-owner GPU fairness when more than one owner is eligible.
            gpu_slot = None
            if self._requires_gpu(chosen["stage"], _obj(chosen["profile_json"])):
                limits = _obj(chosen["profile_json"]).get("limits", {})
                budget = limits.get("max_job_gpu_seconds", self.settings.max_job_gpu_seconds)
                remaining = budget - chosen["gpu_seconds_charged"]
                if remaining <= 0:
                    c.execute(
                        "UPDATE jobs SET state='FAILED',error_code='GPU_BUDGET_EXCEEDED',error_message='Cumulative GPU budget is exhausted',updated_at=? WHERE id=?",
                        (_now(), chosen["job_id"]),
                    )
                    c.execute(
                        "UPDATE stage_tasks SET state='FAILED',updated_at=? WHERE id=?",
                        (_now(), chosen["id"]),
                    )
                    return None
                eligible_owners = {
                    t["owner_id"] for t in pending if self._requires_gpu(t["stage"], _obj(t["profile_json"]))
                }
                active = c.execute("""SELECT j.owner_id,COUNT(*) n FROM stage_tasks t JOIN jobs j ON j.id=t.job_id
                    WHERE t.state='RUNNING' AND t.stage IN ('CAPTION','TRAIN','EVALUATE') GROUP BY j.owner_id""").fetchall()
                counts = {r["owner_id"]: r["n"] for r in active}
                if len(eligible_owners) > 1 and counts.get(chosen["owner_id"], 0):
                    return None
                occupied = {
                    row[0]
                    for row in c.execute(
                        "SELECT gpu_slot FROM task_attempts WHERE state='RUNNING' AND gpu_slot IS NOT NULL"
                    )
                }
                free_slots = [slot for slot in slots if slot not in occupied]
                if not free_slots:
                    return None
                gpu_slot = free_slots[0]
            # Tokens are never reset when a lease clears active_token; stale
            # executors must not regain authority on a retry.
            previous_token = c.execute(
                "SELECT COALESCE(MAX(fencing_token),0) FROM task_attempts WHERE task_id=?", (chosen["id"],)
            ).fetchone()[0]
            token = int(previous_token) + 1
            now = _now()
            lease = now + self.settings.lease_seconds
            limits = _obj(chosen["profile_json"]).get("limits", {})
            stage_limit = limits.get("max_stage_seconds", self.settings.max_stage_seconds)
            deadline_seconds = min(stage_limit, remaining) if gpu_slot else stage_limit
            c.execute(
                "UPDATE stage_tasks SET state='RUNNING',attempt_count=attempt_count+1,active_token=?,lease_expires_at=?,deadline_at=?,updated_at=? WHERE id=?",
                (token, lease, now + deadline_seconds, now, chosen["id"]),
            )
            attempt_id = str(uuid.uuid4())
            c.execute(
                "INSERT INTO task_attempts(id,task_id,attempt_no,fencing_token,gpu_slot,state,started_at,last_heartbeat_at,lease_expires_at) VALUES(?,?,?,?,?,'RUNNING',?,?,?)",
                (attempt_id, chosen["id"], chosen["attempt_count"] + 1, token, gpu_slot, now, now, lease),
            )
            c.execute(
                "UPDATE jobs SET state='RUNNING',current_stage=?,updated_at=? WHERE id=?",
                (chosen["stage"], now, chosen["job_id"]),
            )
            return {**dict(chosen), "attempt_id": attempt_id, "token": token, "gpu_slot": gpu_slot}

    def record_process(
        self, attempt_id: str, token: int, pid: int | None, start_time: float | None, pgid: int | None
    ) -> bool:
        with self.transaction() as c:
            row = c.execute(
                "SELECT a.*,t.active_token,t.lease_expires_at FROM task_attempts a JOIN stage_tasks t ON t.id=a.task_id WHERE a.id=?",
                (attempt_id,),
            ).fetchone()
            if (
                not row
                or row["fencing_token"] != token
                or row["active_token"] != token
                or row["lease_expires_at"] < _now()
            ):
                return False
            c.execute(
                "UPDATE task_attempts SET pid=?,process_start_time=?,process_group_id=? WHERE id=?",
                (pid, start_time, pgid, attempt_id),
            )
            return True

    def attempt_heartbeat(self, attempt_id: str, token: int, progress: dict | None = None) -> bool:
        with self.transaction() as c:
            row = c.execute(
                "SELECT a.*,t.active_token,t.state task_state,t.lease_expires_at task_lease_expires_at,t.deadline_at task_deadline,j.state job_state FROM task_attempts a JOIN stage_tasks t ON t.id=a.task_id JOIN jobs j ON j.id=t.job_id WHERE a.id=?",
                (attempt_id,),
            ).fetchone()
            if (
                not row
                or row["fencing_token"] != token
                or row["active_token"] != token
                or row["task_state"] != "RUNNING"
                or row["job_state"] not in {"ACCEPTED", "RUNNING"}
                or row["task_lease_expires_at"] < _now()
                or row["task_deadline"] < _now()
            ):
                return False
            now = _now()
            lease = now + self.settings.lease_seconds
            merged = {**_obj(row["progress_json"], {}), **(progress or {})}
            c.execute(
                "UPDATE task_attempts SET last_heartbeat_at=?,lease_expires_at=?,progress_json=? WHERE id=?",
                (now, lease, _json(merged), attempt_id),
            )
            checkpoint = merged.get("checkpoint_path")
            c.execute(
                "UPDATE stage_tasks SET lease_expires_at=?,checkpoint_path=COALESCE(?,checkpoint_path),updated_at=? WHERE id=?",
                (lease, checkpoint, now, row["task_id"]),
            )
            return True

    def complete_task(self, attempt_id: str, token: int, output: dict) -> bool:
        """Fence results and atomically add the workflow successor."""
        with self.transaction() as c:
            row = c.execute(
                """SELECT a.task_id,a.fencing_token,a.gpu_slot,a.started_at,a.progress_json,t.*,j.state job_state,j.id job_id
                FROM task_attempts a JOIN stage_tasks t ON t.id=a.task_id JOIN jobs j ON j.id=t.job_id WHERE a.id=?""",
                (attempt_id,),
            ).fetchone()
            if (
                not row
                or row["fencing_token"] != token
                or row["active_token"] != token
                or row["lease_expires_at"] < _now()
                or row["state"] != "RUNNING"
                or row["job_state"] not in {"ACCEPTED", "RUNNING"}
            ):
                return False
            stage, job_id, now = row["stage"], row["job_id"], _now()
            if row["gpu_slot"] and not str(row["gpu_slot"]).startswith("cpu-test-"):
                c.execute(
                    "UPDATE jobs SET gpu_seconds_charged=gpu_seconds_charged+? WHERE id=?",
                    (max(0, now - row["started_at"]), job_id),
                )
            c.execute(
                "UPDATE task_attempts SET state='SUCCEEDED',ended_at=?,progress_json=? WHERE id=?",
                (now, _json({**_obj(row["progress_json"], {}), **output.get("progress", {})}), attempt_id),
            )
            c.execute(
                "UPDATE stage_tasks SET state='SUCCEEDED',output_json=?,updated_at=? WHERE id=?",
                (_json(output), now, row["task_id"]),
            )
            column = {
                "PREPARE": "prepared_json",
                "CAPTION": "input_json",
                "TRAIN": "training_json",
                "EVALUATE": "evaluation_json",
            }.get(stage)
            if column:
                c.execute(f"UPDATE jobs SET {column}=?,updated_at=? WHERE id=?", (_json(output), now, job_id))
            next_stage = {
                "PREPARE": "CAPTION",
                "CAPTION": "TRAIN",
                "TRAIN": "EVALUATE",
                "EVALUATE": "PUBLISH",
            }.get(stage)
            if stage == "EVALUATE" and output.get("quality_status") == "FAIL":
                c.execute(
                    "UPDATE jobs SET state='QUALITY_REJECTED',error_code='QUALITY_REJECTED',error_message='Evaluation quality policy rejected the candidate',current_stage='EVALUATE',updated_at=? WHERE id=?",
                    (now, job_id),
                )
                next_stage = None
            if stage == "PREPARE" and output.get("_input"):
                c.execute(
                    "UPDATE jobs SET prepared_json=?,input_json=?,updated_at=? WHERE id=?",
                    (_json(output["prepared"]), _json(output["_input"]), now, job_id),
                )
                next_stage = "TRAIN"
            if next_stage:
                self._insert_task(c, job_id, next_stage, now)
                c.execute(
                    "UPDATE jobs SET current_stage=?,updated_at=? WHERE id=?", (next_stage, now, job_id)
                )
            return True

    def fail_task(self, attempt_id: str, token: int, error: PipelineError, *, retry: bool = False) -> bool:
        with self.transaction() as c:
            row = c.execute(
                "SELECT a.task_id,a.fencing_token,a.gpu_slot,a.started_at,t.*,j.id job_id,j.state job_state FROM task_attempts a JOIN stage_tasks t ON t.id=a.task_id JOIN jobs j ON j.id=t.job_id WHERE a.id=?",
                (attempt_id,),
            ).fetchone()
            if (
                not row
                or row["fencing_token"] != token
                or row["active_token"] != token
                or row["state"] != "RUNNING"
            ):
                return False
            now = _now()
            if row["gpu_slot"] and not str(row["gpu_slot"]).startswith("cpu-test-"):
                c.execute(
                    "UPDATE jobs SET gpu_seconds_charged=gpu_seconds_charged+? WHERE id=?",
                    (max(0, now - row["started_at"]), row["job_id"]),
                )
            cancelled = row["job_state"] == "CANCEL_REQUESTED"
            can_retry = (
                retry
                and error.retryable
                and row["attempt_count"] < self.settings.max_attempts
                and error.code != "OOM"
                and not cancelled
            )
            c.execute(
                "UPDATE task_attempts SET state=?,ended_at=?,error_code=?,error_message=? WHERE id=?",
                (
                    "RETRY_WAIT" if can_retry else ("CANCELLED" if cancelled else "FAILED"),
                    now,
                    error.code,
                    error.message,
                    attempt_id,
                ),
            )
            if can_retry:
                c.execute(
                    "UPDATE stage_tasks SET state='RETRY_WAIT',ready_at=?,active_token=NULL,lease_expires_at=NULL,updated_at=? WHERE id=?",
                    (now + min(60, 2 ** row["attempt_count"]), now, row["task_id"]),
                )
            else:
                job_state = (
                    "CANCELLED"
                    if cancelled
                    else ("QUALITY_REJECTED" if error.code == "QUALITY_REJECTED" else "FAILED")
                )
                c.execute(
                    "UPDATE stage_tasks SET state=?,updated_at=? WHERE id=?",
                    ("CANCELLED" if cancelled else "FAILED", now, row["task_id"]),
                )
                c.execute(
                    "UPDATE jobs SET state=?,error_code=?,error_message=?,current_stage=?,updated_at=? WHERE id=?",
                    (job_state, error.code, error.message, row["stage"], now, row["job_id"]),
                )
            return True

    def _expire_conn(self, c: sqlite3.Connection) -> None:
        now = _now()
        expired = c.execute(
            "SELECT t.*,a.id attempt_id,a.pid,a.gpu_slot,a.started_at,a.process_start_time,j.state job_state FROM stage_tasks t JOIN task_attempts a ON a.task_id=t.id AND a.fencing_token=t.active_token JOIN jobs j ON j.id=t.job_id WHERE t.state='RUNNING' AND (t.lease_expires_at<? OR t.deadline_at<?)",
            (now, now),
        ).fetchall()
        for task in expired:
            # A spawned executor gets a small persisted launch window to record
            # PID/start-time.  Without this, a deliberately short test lease can
            # re-dispatch the same task before the child reaches record_process.
            if task["pid"] is None and now - task["started_at"] < 5:
                continue
            if self._process_identity_alive(task["pid"], task["process_start_time"]):
                # The watchdog has requested termination but this device remains
                # conservatively occupied until its exact executor exits.
                continue
            if task["gpu_slot"] and not str(task["gpu_slot"]).startswith("cpu-test-"):
                c.execute(
                    "UPDATE jobs SET gpu_seconds_charged=gpu_seconds_charged+? WHERE id=?",
                    (max(0, now - task["started_at"]), task["job_id"]),
                )
            # Lease expiry only fences the old result.  The worker/process watchdog
            # owns physical process confirmation; until then this task stays retryable.
            terminal_state = "CANCELLED" if task["job_state"] == "CANCEL_REQUESTED" else "RETRY_WAIT"
            if terminal_state == "RETRY_WAIT" and task["attempt_count"] >= self.settings.max_attempts:
                terminal_state = "FAILED"
            c.execute(
                "UPDATE task_attempts SET state=?,ended_at=?,error_code=? WHERE id=?",
                (
                    terminal_state,
                    now,
                    "CANCELLED" if terminal_state == "CANCELLED" else "LEASE_EXPIRED",
                    task["attempt_id"],
                ),
            )
            c.execute(
                "UPDATE stage_tasks SET state=?,active_token=NULL,lease_expires_at=NULL,ready_at=?,updated_at=? WHERE id=?",
                (terminal_state, now + 1, now, task["id"]),
            )
            if terminal_state == "FAILED":
                c.execute(
                    "UPDATE jobs SET state='FAILED',error_code='LEASE_EXPIRED',error_message='Stage executor expired repeatedly',updated_at=? WHERE id=?",
                    (now, task["job_id"]),
                )

    @staticmethod
    def _process_identity_alive(pid: int | None, started_at: float | None) -> bool:
        if not pid or started_at is None:
            return False
        try:
            import psutil

            process = psutil.Process(pid)
            return (
                process.status() != psutil.STATUS_ZOMBIE
                and abs(float(process.create_time()) - float(started_at)) < 0.01
            )
        except Exception:
            return False

    def reconcile(self) -> None:
        with self.transaction() as c:
            self._expire_conn(c)
            # Cancellation is final only after no RUNNING executor remains.
            pending = c.execute("SELECT id FROM jobs WHERE state='CANCEL_REQUESTED'").fetchall()
            for job in pending:
                running = c.execute(
                    "SELECT COUNT(*) FROM stage_tasks WHERE job_id=? AND state='RUNNING'", (job["id"],)
                ).fetchone()[0]
                if not running:
                    c.execute(
                        "UPDATE jobs SET state='CANCELLED',updated_at=? WHERE id=?", (_now(), job["id"])
                    )

    def active_attempts(self) -> list[dict]:
        with self.reader() as c:
            return [
                dict(row)
                for row in c.execute("""SELECT a.*,t.deadline_at,t.active_token,t.state task_state,j.state job_state
                FROM task_attempts a JOIN stage_tasks t ON t.id=a.task_id JOIN jobs j ON j.id=t.job_id
                WHERE a.state='RUNNING' AND t.state='RUNNING'""")
            ]

    # ----- reports and model registration ---------------------------------
    def evaluation(self, owner: str, job_id: str) -> dict:
        job = self.job(owner, job_id)
        if not job.get("evaluation"):
            raise PipelineError("EVALUATION_NOT_READY", "Evaluation report is not available")
        return job["evaluation"]

    def publish(self, attempt_id: str, token: int, manifest: dict, report: dict) -> dict | None:
        with self.transaction() as c:
            row = c.execute(
                """SELECT a.task_id,a.fencing_token,t.job_id FROM task_attempts a
                JOIN stage_tasks t ON t.id=a.task_id WHERE a.id=?""",
                (attempt_id,),
            ).fetchone()
            if not row:
                return None
            job = c.execute("SELECT * FROM jobs WHERE id=?", (row["job_id"],)).fetchone()
            if job["state"] == "CANCEL_REQUESTED":
                return None
            task = c.execute("SELECT * FROM stage_tasks WHERE id=?", (row["task_id"],)).fetchone()
            if (
                task["state"] != "RUNNING"
                or task["active_token"] != token
                or task["lease_expires_at"] < _now()
            ):
                return None
            for key in ("adapter_path", "manifest_path"):
                if not manifest.get(key) or not Path(manifest[key]).is_file():
                    raise PipelineError("ARTIFACT_MISSING", f"{key} is missing")
            if sha256_file(manifest["adapter_path"]) != manifest.get("adapter_sha256") or sha256_file(
                manifest["manifest_path"]
            ) != manifest.get("manifest_sha256"):
                raise PipelineError("CHECKSUM_MISMATCH", "Model artifact checksum does not match")
            if not report.get("technical_pass"):
                raise PipelineError("TECHNICAL_EVALUATION_FAILED", "An adapter must pass its load smoke test")
            if report.get("adapter_sha256") != manifest.get("adapter_sha256"):
                raise PipelineError(
                    "INPUT_INCOMPATIBLE", "Evaluation adapter does not match training adapter"
                )
            if report.get("input_manifest_sha256") != manifest.get("input_manifest_sha256"):
                raise PipelineError("INPUT_INCOMPATIBLE", "Evaluation input does not match training input")
            report_path, report_sha = report.get("manifest_path"), report.get("manifest_sha256")
            if (
                not report_path
                or not report_sha
                or not Path(report_path).is_file()
                or sha256_file(report_path) != report_sha
            ):
                raise PipelineError("CHECKSUM_MISMATCH", "Evaluation report checksum does not match")
            model_id, now = str(uuid.uuid4()), _now()
            quality = report.get("quality_status", "UNCALIBRATED")
            if quality == "FAIL":
                raise PipelineError("QUALITY_REJECTED", "A quality-rejected adapter cannot be published")
            state = (
                "READY"
                if quality == "PASS" and report.get("technical_pass") and not manifest.get("test_only")
                else "UNVERIFIED"
            )
            c.execute(
                "INSERT OR IGNORE INTO models VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                (
                    model_id,
                    job["id"],
                    job["owner_id"],
                    state,
                    manifest["adapter_path"],
                    manifest["adapter_sha256"],
                    manifest["manifest_path"],
                    manifest["manifest_sha256"],
                    _json(report),
                    int(bool(manifest.get("test_only"))),
                    now,
                ),
            )
            existing = c.execute("SELECT * FROM models WHERE job_id=?", (job["id"],)).fetchone()
            model_id = existing["id"]
            final_state = "READY" if existing["state"] == "READY" else "COMPLETED_UNVERIFIED"
            c.execute("UPDATE task_attempts SET state='SUCCEEDED',ended_at=? WHERE id=?", (now, attempt_id))
            c.execute(
                "UPDATE stage_tasks SET state='SUCCEEDED',output_json=?,updated_at=? WHERE id=?",
                (_json(manifest), now, task["id"]),
            )
            c.execute(
                "UPDATE jobs SET state=?,current_stage='PUBLISH',model_id=?,updated_at=? WHERE id=?",
                (final_state, model_id, now, job["id"]),
            )
            return {"model_id": model_id, "state": existing["state"]}

    def model(self, owner: str, model_id: str) -> dict:
        with self.reader() as c:
            row = c.execute("SELECT * FROM models WHERE id=? AND owner_id=?", (model_id, owner)).fetchone()
            if not row:
                raise PipelineError("NOT_FOUND", "Resource was not found")
            result = dict(row)
            result.pop("owner_id", None)
            result["report"] = _obj(result.pop("report_json"))
            return result
