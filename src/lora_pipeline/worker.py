"""Single-host durable worker for verification, training and publication.

The worker is deliberately one supervisor per data directory.  A database lease
fences writes; the supervisor flock and physical-device flocks prevent local
processes from accidentally sharing a GPU.  This is not a multi-host scheduler.
"""

from __future__ import annotations

import contextlib
import fcntl
import multiprocessing
import os
import signal
import threading
import time
import traceback
from dataclasses import replace
from pathlib import Path

from .common import PipelineError, sha256_file, write_manifest
from .config import DataConfig, EvalConfig, Settings, TrainConfig
from .observability import configure_logging, log_event
from .store import Store, device_lock, discover_gpu_uuids


def _process_started_at(pid: int) -> float | None:
    try:
        import psutil

        return float(psutil.Process(pid).create_time())
    except Exception:
        return None


def _traceback_locations(exc: BaseException, *, limit: int = 12) -> list[dict[str, object]]:
    """Return useful call locations without copying exception text or locals to logs."""
    frames = traceback.extract_tb(exc.__traceback__)
    return [
        {"file": Path(frame.filename).name, "line": frame.lineno, "function": frame.name}
        for frame in frames[-limit:]
    ]


def _unexpected_worker_error(stage: str, exc: Exception) -> PipelineError:
    exception_type = type(exc).__name__
    return PipelineError(
        "WORKER_EXCEPTION",
        f"Unexpected {exception_type} while executing {stage}. Check worker logs for traceback locations.",
        retryable=True,
        details={"exception_type": exception_type},
    )


def _run_stage_child(settings: Settings, task: dict) -> None:
    """Forked stage executor. It owns only its exact attempt token."""
    os.setsid()
    # Spawn starts a fresh interpreter, so configure the child logger as well.
    configure_logging()
    if task.get("gpu_slot") and not str(task["gpu_slot"]).startswith("cpu-test-"):
        # UUID selection before lazy torch/diffusers imports makes the adapter
        # see a single physical device as cuda:0.
        os.environ["CUDA_VISIBLE_DEVICES"] = str(task["gpu_slot"])
    worker = Worker(settings)
    if not worker.store.record_process(
        task["attempt_id"], task["token"], os.getpid(), _process_started_at(os.getpid()), os.getpgid(0)
    ):
        return
    stopped = threading.Event()

    def renew() -> None:
        while not stopped.wait(max(1.0, settings.heartbeat_seconds / 2)):
            # A stage refuses to run indefinitely after its control supervisor
            # disappears.  Its own DB lease remains a fencing condition.
            if not worker.store.worker_healthy() or not worker.store.attempt_heartbeat(
                task["attempt_id"], task["token"], {"executor": "alive"}
            ):
                stopped.set()
                try:
                    os.kill(os.getpid(), signal.SIGTERM)
                except OSError:
                    pass
                return

    heartbeat = threading.Thread(target=renew, name="stage-heartbeat", daemon=True)
    heartbeat.start()
    try:
        worker._execute(task)
    finally:
        stopped.set()


class Worker:
    def __init__(self, settings: Settings):
        self.settings, self.store = settings, Store(settings)
        self._supervisor_path = self.store.locks / "supervisor.lock"
        self._children: dict[str, multiprocessing.Process] = {}

    @contextlib.contextmanager
    def _supervisor_lock(self):
        self._supervisor_path.parent.mkdir(parents=True, exist_ok=True)
        with self._supervisor_path.open("a+") as handle:
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise PipelineError(
                    "WORKER_ALREADY_RUNNING", "Another worker owns this data directory"
                ) from exc
            try:
                yield
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    def slots(self) -> list[str]:
        # Test mode is an explicit CPU-only execution domain.  It must never
        # opportunistically attach a developer's visible CUDA device.
        if self.settings.enable_test_backend:
            count = self.settings.fake_slots or self.settings.cpu_slots
            return [f"cpu-test-{index}" for index in range(max(1, count))]
        physical = discover_gpu_uuids()
        configured = self.settings.gpu_uuids or physical
        if configured:
            if len(set(configured)) != len(configured):
                raise PipelineError("INVALID_GPU_CONFIG", "GPU UUIDs must be unique")
            if any(item not in physical for item in configured):
                raise PipelineError(
                    "GPU_UNAVAILABLE", "A configured GPU UUID was not discovered", retryable=True
                )
            return configured
        return []

    def run(self, *, once: bool = False, poll_seconds: float = 0.25) -> None:
        configure_logging()
        with self._supervisor_lock():
            while True:
                worked = self.run_once()
                if once:
                    return
                if not worked:
                    time.sleep(poll_seconds)

    def run_once(self) -> bool:
        """Do one verification or pipeline stage. Returns whether work was found."""
        self.store.heartbeat()
        self._watchdog()
        self.store.reconcile()
        verification = self.store.claim_verification()
        if verification:
            log_event("verification_started", dataset_id=verification["dataset"]["id"])
            self._verify(verification)
            return True
        task = self.store.claim_task(self.slots())
        if not task:
            return False
        log_event(
            "stage_started",
            job_id=task["job_id"],
            task_id=task["id"],
            attempt_id=task["attempt_id"],
            stage=task["stage"],
        )
        # Stage work runs outside the supervisor.  The child starts a new session
        # and persists PID/start-time/PGID before touching artifacts, allowing a
        # successor to identify (rather than blindly kill) an orphan executor.
        context = multiprocessing.get_context("spawn")
        child = context.Process(target=_run_stage_child, args=(self.settings, task), daemon=False)
        child.start()
        self._children[task["attempt_id"]] = child
        return True

    def _same_process(self, attempt: dict) -> bool:
        pid = attempt.get("pid")
        if not pid:
            return False
        try:
            import psutil

            actual = float(psutil.Process(pid).create_time())
            expected = attempt.get("process_start_time")
            return expected is not None and abs(actual - float(expected)) < 0.01
        except Exception:
            return False

    def _terminate_attempt(self, attempt: dict) -> None:
        if not self._same_process(attempt):
            return
        try:
            os.killpg(int(attempt["process_group_id"]), signal.SIGTERM)
        except (ProcessLookupError, PermissionError, TypeError):
            return

    def _watchdog(self) -> None:
        """Terminate only the persisted process identity after cancel/deadline.

        Lease expiry alone never declares a device free: the task remains fenced
        until this watcher observes the exact child exit and reconciliation moves
        it to a retry or cancellation state.
        """
        now = time.time()
        for attempt in self.store.active_attempts():
            expired = attempt["lease_expires_at"] < now or attempt["deadline_at"] < now
            cancelled = attempt["job_state"] == "CANCEL_REQUESTED"
            if expired or cancelled:
                self._terminate_attempt(attempt)
        for key, child in list(self._children.items()):
            if not child.is_alive():
                child.join(timeout=0)
                self._children.pop(key, None)

    def _verify(self, item: dict) -> None:
        dataset, files = item["dataset"], item["files"]
        try:
            for entry in files:
                path = self.store.objects / entry["object_key"]
                if (
                    not path.is_file()
                    or path.stat().st_size != entry["size_bytes"]
                    or sha256_file(path) != entry["sha256"]
                ):
                    raise PipelineError(
                        "CHECKSUM_MISMATCH",
                        "Frozen upload is missing or corrupt",
                        details={"file_id": entry["id"]},
                    )
            # PREPARE performs costly image decoding, dedupe and split.  Verification
            # establishes the immutable upload snapshot only.
            manifest = write_manifest(
                self.store.artifacts / dataset["id"] / "upload-verified.json",
                {
                    "kind": "verified-upload",
                    "dataset_id": dataset["id"],
                    "valid_count": len(files),
                    "invalid_count": 0,
                    "files": [{"id": entry["id"], "sha256": entry["sha256"]} for entry in files],
                },
            )
            self.store.finish_verification(dataset["id"], valid=True, manifest=manifest)
        except PipelineError as exc:
            self.store.finish_verification(dataset["id"], valid=False, error=exc)

    def _progress(self, task: dict):
        def update(values: dict) -> None:
            if not self.store.attempt_heartbeat(task["attempt_id"], task["token"], values):
                raise PipelineError("LEASE_LOST", "Attempt lease was lost", retryable=True)

        return update

    def _execute(self, task: dict) -> None:
        try:
            context = self.store.task_context(task)
            stage = task["stage"]
            slot = task.get("gpu_slot")
            lock = (
                device_lock(self.settings, slot)
                if slot and not slot.startswith("cpu-test-")
                else contextlib.nullcontext()
            )
            with lock:
                output = self._stage(stage, context, task)
            if stage == "PUBLISH":
                result = self.store.publish(
                    task["attempt_id"], task["token"], output["model"], output["report"]
                )
                if result is None:
                    raise PipelineError("LEASE_LOST", "Attempt result was fenced", retryable=True)
            elif not self.store.complete_task(task["attempt_id"], task["token"], output):
                raise PipelineError("LEASE_LOST", "Attempt result was fenced", retryable=True)
            log_event(
                "stage_completed",
                job_id=task["job_id"],
                task_id=task["id"],
                attempt_id=task["attempt_id"],
                stage=stage,
            )
        except PipelineError as exc:
            self.store.fail_task(task["attempt_id"], task["token"], exc, retry=exc.retryable)
            log_event(
                "stage_failed",
                job_id=task["job_id"],
                task_id=task["id"],
                attempt_id=task["attempt_id"],
                stage=task["stage"],
                code=exc.code,
                retryable=exc.retryable,
            )
        except Exception as exc:  # unexpected worker/model failure is an infrastructure retry
            error = _unexpected_worker_error(task["stage"], exc)
            locations = _traceback_locations(exc)
            self.store.fail_task(
                task["attempt_id"],
                task["token"],
                error,
                retry=True,
            )
            log_event(
                "stage_failed",
                job_id=task["job_id"],
                task_id=task["id"],
                attempt_id=task["attempt_id"],
                stage=task["stage"],
                code=error.code,
                retryable=True,
                exception_type=type(exc).__name__,
                traceback_locations=locations,
            )

    def _stage(self, stage: str, context: dict, task: dict) -> dict:
        job, dataset = context["job"], context["dataset"]
        root = self.store.artifacts / job["id"] / f"attempt-{task['attempt_id']}"
        root.mkdir(parents=True, exist_ok=True)
        progress = self._progress(task)
        if stage == "PREPARE":
            from .data import prepare_dataset

            files = [
                {
                    "id": f["id"],
                    "name": f["original_name"],
                    "path": str(self.store.objects / f["object_key"]),
                    "caption": f["caption"],
                    "sha256": f["sha256"],
                }
                for f in context["files"]
            ]
            prepared = prepare_dataset(
                files, root, config=DataConfig(**job["profile"]["data"]), dataset_id=dataset["id"]
            )
            # If every training item already has a validated user caption, bind the
            # final input in the same PREPARE completion and avoid a needless task.
            if all(entry.get("caption") for entry in prepared.get("train", [])):
                from .captions import caption_dataset

                frozen_input = caption_dataset(
                    prepared,
                    root / "captions",
                    mode="template",
                    trigger_token=job["trigger_token"],
                    device="cpu",
                    model_name=job["profile"]["caption"]["model"],
                    revision=job["profile"]["caption"]["revision"],
                    local_files_only=True,
                )
                return {"prepared": prepared, "_input": frozen_input}
            return prepared
        if stage == "CAPTION":
            from .captions import caption_dataset

            prepared = job.get("prepared")
            if not prepared:
                raise PipelineError("MISSING_PREPARED_INPUT", "Caption task has no prepared manifest")
            return caption_dataset(
                prepared,
                root,
                mode=job["profile"]["caption"]["mode"],
                trigger_token=job["trigger_token"],
                device=job["profile"]["caption"]["device"],
                model_name=job["profile"]["caption"]["model"],
                revision=job["profile"]["caption"]["revision"],
                local_files_only=job["profile"]["config"]["local_files_only"],
            )
        if stage == "TRAIN":
            from .training import train

            input_manifest = job.get("input")
            if not input_manifest:
                raise PipelineError("MISSING_TRAINING_INPUT", "Train task has no caption manifest")
            config = dict(job["profile"]["config"])
            config.update(job.get("training_overrides") or {})
            # Test slots are always CPU, keeping fake concurrency separate from CUDA.
            if task.get("gpu_slot", "").startswith("cpu-test-"):
                config.update({"backend": "tiny", "device": "cpu", "precision": "fp32", "resolution": 16})
            elif task.get("gpu_slot"):
                config["device"] = "cuda:0"
            resume = Path(task["checkpoint_path"]) if task.get("checkpoint_path") else None
            return train(input_manifest, root, TrainConfig(**config), progress=progress, resume_from=resume)
        if stage == "EVALUATE":
            from .evaluation import evaluate

            if not job.get("training") or not job.get("input"):
                raise PipelineError("MISSING_TRAINING_INPUT", "Evaluation task has no frozen inputs")
            # Evaluation choices are frozen when a manual job enters EVALUATE;
            # the profile remains the operator-controlled immutable baseline.
            config = EvalConfig(**{**job["profile"]["evaluation"], **(job.get("evaluation_overrides") or {})})
            if task.get("gpu_slot", "").startswith("cpu-test-"):
                config = replace(config, device="cpu")
            return evaluate(job["training"], job["input"], root, config, progress=progress)
        if stage == "PUBLISH":
            training, report = job.get("training"), job.get("evaluation")
            if not training or not report:
                raise PipelineError("MISSING_EVALUATION", "Cannot publish without training and evaluation")
            return {"model": training, "report": report}
        raise PipelineError("INVALID_STAGE", f"Unknown stage {stage}")
