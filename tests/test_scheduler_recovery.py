"""Scheduling invariants without model execution or fabricated GPU capacity."""

from __future__ import annotations

import hashlib
import os
import time
from dataclasses import asdict

import psutil
import pytest

from lora_pipeline.common import PipelineError
from lora_pipeline.config import DataConfig, Settings, TrainConfig
from lora_pipeline.store import Store


def _store(tmp_path, **overrides):
    return Store(
        Settings(
            data_dir=tmp_path,
            api_keys={"secret": "alice"},
            enable_test_backend=True,
            fake_slots=2,
            data=DataConfig(
                min_images=1,
                max_images=100,
                min_train_images=1,
                min_validation_images=1,
                min_groups_per_split=1,
                min_side=1,
            ),
            train=TrainConfig(backend="tiny", resolution=16, device="cpu", precision="fp32", max_steps=2),
            **overrides,
        )
    )


def _job(store, owner="alice", stage="TRAIN"):
    dataset = store.create_dataset(
        owner,
        [
            {
                "name": "a.png",
                "size_bytes": 1,
                "sha256": hashlib.sha256(b"a").hexdigest(),
                "mime_type": "image/png",
            }
        ],
    )
    store.finish_verification(dataset["id"], valid=True, manifest={"valid_count": 1, "invalid_count": 0})
    job = store.create_job(
        owner, {"dataset_id": dataset["id"], "profile_revision_id": "local-tiny-v1", "trigger_token": "style"}
    )
    with store.transaction() as connection:
        connection.execute("UPDATE stage_tasks SET stage=? WHERE job_id=?", (stage, job["job_id"]))
    return job["job_id"]


def _expire(store, attempt, *, live=False):
    with store.transaction() as connection:
        connection.execute(
            "UPDATE stage_tasks SET lease_expires_at=? WHERE id=?", (time.time() - 1, attempt["id"])
        )
        connection.execute(
            "UPDATE task_attempts SET started_at=?,pid=?,process_start_time=? WHERE id=?",
            (
                time.time() - 30,
                os.getpid() if live else None,
                psutil.Process().create_time() if live else None,
                attempt["attempt_id"],
            ),
        )


def _make_retry_ready(store):
    with store.transaction() as connection:
        connection.execute("UPDATE stage_tasks SET ready_at=0 WHERE state='RETRY_WAIT'")


@pytest.mark.parametrize("count", [2, 4])
def test_gpu_slots_are_reserved_distinctly_and_capacity_bounded(tmp_path, count):
    store = _store(tmp_path)
    slots = [f"cpu-test-{i}" for i in range(count)]
    for i in range(count + 1):
        _job(store, owner=f"owner-{i}")
    claims = [store.claim_task(slots) for _ in range(count)]
    assert all(claims)
    assert len({claim["gpu_slot"] for claim in claims}) == count
    assert store.claim_task(slots) is None


def test_cpu_stages_obey_separate_capacity(tmp_path):
    store = _store(tmp_path, cpu_slots=1)
    _job(store, "alice", "PREPARE")
    _job(store, "bob", "PREPARE")
    first = store.claim_task([])
    assert first is not None and first["gpu_slot"] is None
    assert store.claim_task([]) is None


def test_template_caption_does_not_need_gpu(tmp_path):
    store = _store(tmp_path)
    _job(store, stage="CAPTION")
    task = store.claim_task([])
    assert task is not None and task["gpu_slot"] is None


def test_expired_heartbeat_cannot_restore_authority(tmp_path):
    store = _store(tmp_path)
    _job(store)
    attempt = store.claim_task(["cpu-test-0"])
    _expire(store, attempt, live=True)
    assert store.attempt_heartbeat(attempt["attempt_id"], attempt["token"], {"loss": 1}) is False
    assert store.complete_task(attempt["attempt_id"], attempt["token"], {}) is False


def test_live_expired_executor_protects_cancel_and_slot(tmp_path):
    store = _store(tmp_path)
    job_id = _job(store)
    _job(store, "bob")
    attempt = store.claim_task(["cpu-test-0"])
    assert attempt["job_id"] == job_id
    _expire(store, attempt, live=True)
    store.cancel_job("alice", job_id)
    store.reconcile()
    assert store.job("alice", job_id)["state"] == "CANCEL_REQUESTED"
    assert store.claim_task(["cpu-test-0"]) is None
    # Observe a confirmed absent executor; this test never signals its own PID.
    with store.transaction() as connection:
        connection.execute(
            "UPDATE task_attempts SET pid=NULL,process_start_time=NULL WHERE id=?", (attempt["attempt_id"],)
        )
    store.reconcile()
    assert store.job("alice", job_id)["state"] == "CANCELLED"


def test_expiry_retries_are_bounded_and_tokens_monotonic(tmp_path):
    store = _store(tmp_path, max_attempts=3)
    job_id = _job(store)
    tokens = []
    for _ in range(3):
        attempt = store.claim_task(["cpu-test-0"])
        assert attempt is not None
        tokens.append(attempt["token"])
        _expire(store, attempt)
        store.reconcile()
        _make_retry_ready(store)
    assert tokens == sorted(set(tokens))
    assert store.claim_task(["cpu-test-0"]) is None
    assert store.job("alice", job_id)["state"] == "FAILED"


def test_gpu_budget_exhaustion_blocks_new_stage(tmp_path):
    store = _store(tmp_path, max_job_gpu_seconds=5)
    job_id = _job(store)
    with store.transaction() as connection:
        connection.execute("UPDATE jobs SET gpu_seconds_charged=5 WHERE id=?", (job_id,))
    assert store.claim_task(["cpu-test-0"]) is None
    job = store.job("alice", job_id)
    assert job["state"] == "FAILED" and job["error_code"] == "GPU_BUDGET_EXCEEDED"


def test_job_captures_nontraining_settings_before_worker_changes(tmp_path):
    store = _store(tmp_path)
    original_data = asdict(store.settings.data)
    original_eval = asdict(store.settings.evaluation)
    job_id = _job(store)
    snapshot = store.job("alice", job_id)["profile"]
    store.settings.caption_mode = "blip"
    store.settings.data.seed = 123
    store.settings.evaluation.prompts = ["different"]
    assert snapshot["data"] == original_data
    assert snapshot["evaluation"] == original_eval
    assert snapshot["caption"]["mode"] == "template"


def test_failed_mutation_rolls_back_resource_and_idempotency(tmp_path):
    store = _store(tmp_path)

    def fail_after_create():
        _job(store)
        raise PipelineError("INJECTED", "Failure before idempotency commit")

    with pytest.raises(PipelineError):
        store.idempotent("alice", "POST /test", "key", {}, fail_after_create)
    with store.reader() as connection:
        assert connection.execute("SELECT COUNT(*) FROM jobs").fetchone()[0] == 0
        assert connection.execute("SELECT COUNT(*) FROM idempotency").fetchone()[0] == 0


def test_invalid_retained_dataset_still_uses_storage_quota(tmp_path):
    store = _store(tmp_path, storage_quota_bytes=1)
    entry = {
        "name": "a.png",
        "size_bytes": 1,
        "mime_type": "image/png",
        "sha256": hashlib.sha256(b"a").hexdigest(),
    }
    dataset = store.create_dataset("alice", [entry])
    store.finish_verification(
        dataset["id"], valid=False, error=PipelineError("INVALID_IMAGE", "Injected validation failure")
    )
    with pytest.raises(PipelineError) as error:
        store.create_dataset("alice", [entry])
    assert "QUOTA" in error.value.code or "LIMIT" in error.value.code


def test_quality_failure_never_schedules_publication(tmp_path):
    store = _store(tmp_path)
    job_id = _job(store, stage="EVALUATE")
    attempt = store.claim_task(["cpu-test-0"])
    report = {"technical_pass": True, "quality_status": "FAIL", "quality_failures": ["diversity"]}
    assert store.complete_task(attempt["attempt_id"], attempt["token"], report)
    assert store.job("alice", job_id)["state"] == "QUALITY_REJECTED"
    assert store.evaluation("alice", job_id)["quality_status"] == "FAIL"
    assert store.claim_task(["cpu-test-0"]) is None
    with store.reader() as connection:
        assert connection.execute("SELECT COUNT(*) FROM models WHERE job_id=?", (job_id,)).fetchone()[0] == 0


def test_progress_heartbeat_preserves_checkpoint_and_loss(tmp_path):
    store = _store(tmp_path)
    _job(store)
    attempt = store.claim_task(["cpu-test-0"])
    assert store.attempt_heartbeat(attempt["attempt_id"], attempt["token"],
                                   {"loss": 0.5, "global_step": 1})
    assert store.attempt_heartbeat(attempt["attempt_id"], attempt["token"], {"executor": "alive"})
    import json
    with store.reader() as connection:
        progress = json.loads(connection.execute("SELECT progress_json FROM task_attempts WHERE id=?",
                                                  (attempt["attempt_id"],)).fetchone()[0])
    assert progress["loss"] == 0.5 and progress["global_step"] == 1
