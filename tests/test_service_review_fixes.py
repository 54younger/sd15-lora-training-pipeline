"""Regression tests for supervisor fencing and frozen scheduling limits."""

from __future__ import annotations

import hashlib
import json
import time

import lora_pipeline.worker as worker_module
from lora_pipeline.common import PipelineError
from lora_pipeline.config import DataConfig, Settings, TrainConfig
from lora_pipeline.store import Store


def make_store(tmp_path, **extra) -> Store:
    return Store(
        Settings(
            data_dir=tmp_path,
            enable_test_backend=True,
            fake_slots=2,
            data=DataConfig(
                min_images=1,
                max_images=10,
                min_train_images=1,
                min_validation_images=1,
                min_groups_per_split=1,
                min_side=1,
            ),
            train=TrainConfig(backend="tiny", resolution=16, device="cpu", precision="fp32", max_steps=1),
            **extra,
        )
    )


def make_job(store: Store, owner: str = "owner", stage: str = "TRAIN") -> str:
    item = {
        "name": f"{owner}.png",
        "size_bytes": 1,
        "sha256": hashlib.sha256(owner.encode()).hexdigest(),
        "mime_type": "image/png",
    }
    dataset = store.create_dataset(owner, [item])
    store.finish_verification(dataset["id"], valid=True, manifest={"valid_count": 1, "invalid_count": 0})
    job = store.create_job(
        owner, {"dataset_id": dataset["id"], "profile_revision_id": "local-tiny-v1", "trigger_token": "style"}
    )
    with store.transaction() as connection:
        connection.execute("UPDATE stage_tasks SET stage=? WHERE job_id=?", (stage, job["job_id"]))
    return job["job_id"]


def test_stale_process_registration_is_fenced(tmp_path):
    store = make_store(tmp_path)
    make_job(store)
    task = store.claim_task(["cpu-test-0"])
    assert not store.record_process(task["attempt_id"], task["token"] + 1, 1234, 1.0, 1234)


def test_stage_child_does_not_execute_when_registration_is_fenced(monkeypatch):
    calls = {"execute": 0}

    class FakeStore:
        def record_process(self, *_args):
            return False

    class FakeWorker:
        def __init__(self, _settings):
            self.store = FakeStore()

        def _execute(self, _task):
            calls["execute"] += 1

    monkeypatch.setattr(worker_module, "Worker", FakeWorker)
    monkeypatch.setattr(worker_module.os, "setsid", lambda: None)
    monkeypatch.setattr(worker_module.os, "getpgid", lambda _pid: 1)
    monkeypatch.setattr(worker_module, "_process_started_at", lambda _pid: 1.0)
    worker_module._run_stage_child(object(), {"attempt_id": "a", "token": 1, "gpu_slot": None})
    assert calls["execute"] == 0


def test_mixed_caption_resources_respect_cpu_capacity(tmp_path):
    store = make_store(tmp_path, cpu_slots=1)
    cpu_job, gpu_job = make_job(store, "cpu", "CAPTION"), make_job(store, "gpu", "CAPTION")
    with store.transaction() as connection:
        row = connection.execute("SELECT profile_json FROM jobs WHERE id=?", (gpu_job,)).fetchone()
        profile = json.loads(row[0])
        profile["caption"].update({"mode": "blip", "device": "cuda:0"})
        connection.execute("UPDATE jobs SET profile_json=? WHERE id=?", (json.dumps(profile), gpu_job))
    first = store.claim_task([])
    assert first and first["job_id"] == cpu_job and first["gpu_slot"] is None
    second = store.claim_task(["GPU-test"])
    assert second and second["job_id"] == gpu_job and second["gpu_slot"] == "GPU-test"


def test_cancelled_stage_failure_cannot_overwrite_cancellation(tmp_path):
    store = make_store(tmp_path)
    job_id = make_job(store)
    task = store.claim_task(["cpu-test-0"])
    store.cancel_job("owner", job_id)
    assert store.fail_task(
        task["attempt_id"],
        task["token"],
        PipelineError("WORKER_EXCEPTION", "late", retryable=True),
        retry=True,
    )
    assert store.job("owner", job_id)["state"] == "CANCELLED"


def test_frozen_stage_limit_and_remaining_gpu_budget_drive_deadline(tmp_path):
    store = make_store(tmp_path, max_stage_seconds=3, max_job_gpu_seconds=4)
    job_id = make_job(store)
    store.settings.max_stage_seconds = 999  # admission snapshot must win
    with store.transaction() as connection:
        connection.execute("UPDATE jobs SET gpu_seconds_charged=2 WHERE id=?", (job_id,))
    before = time.time()
    task = store.claim_task(["GPU-test"])
    with store.reader() as connection:
        deadline = connection.execute(
            "SELECT deadline_at FROM stage_tasks WHERE id=?", (task["id"],)
        ).fetchone()[0]
    assert 1.0 <= deadline - before <= 2.5


def test_expired_physical_gpu_attempt_is_charged(tmp_path):
    store = make_store(tmp_path)
    job_id = make_job(store)
    task = store.claim_task(["GPU-test"])
    with store.transaction() as connection:
        connection.execute(
            "UPDATE stage_tasks SET lease_expires_at=? WHERE id=?", (time.time() - 1, task["id"])
        )
        connection.execute(
            "UPDATE task_attempts SET started_at=? WHERE id=?", (time.time() - 6, task["attempt_id"])
        )
    store.reconcile()
    assert store.job("owner", job_id)["gpu_seconds_charged"] >= 1


def test_physical_gpu_success_and_failure_are_charged(tmp_path):
    store = make_store(tmp_path)
    success_job = make_job(store, "success")
    success = store.claim_task(["GPU-success"])
    with store.transaction() as connection:
        connection.execute(
            "UPDATE task_attempts SET started_at=? WHERE id=?", (time.time() - 1, success["attempt_id"])
        )
    assert store.complete_task(success["attempt_id"], success["token"], {})
    assert store.job("success", success_job)["gpu_seconds_charged"] >= 0.9

    failed_job = make_job(store, "failed")
    failed = store.claim_task(["GPU-failure"])
    with store.transaction() as connection:
        connection.execute(
            "UPDATE task_attempts SET started_at=? WHERE id=?", (time.time() - 1, failed["attempt_id"])
        )
    assert store.fail_task(failed["attempt_id"], failed["token"], PipelineError("OOM", "test"))
    assert store.job("failed", failed_job)["gpu_seconds_charged"] >= 0.9
