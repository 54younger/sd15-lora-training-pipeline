"""Manual four-step orchestration contracts without model execution."""

from __future__ import annotations

import hashlib
import json
import concurrent.futures
import sqlite3

import pytest

from lora_pipeline.common import PipelineError, write_manifest
from lora_pipeline.config import DataConfig, Settings, TrainConfig
from lora_pipeline.store import Store


def _store(tmp_path):
    return Store(
        Settings(
            data_dir=tmp_path,
            api_keys={"secret": "alice"},
            enable_test_backend=True,
            fake_slots=1,
            data=DataConfig(
                min_images=1,
                max_images=5,
                min_train_images=1,
                min_validation_images=1,
                min_groups_per_split=1,
                min_side=1,
            ),
            train=TrainConfig(backend="tiny", resolution=16, device="cpu", precision="fp32", max_steps=2),
        )
    )


def _manual_job(store: Store, owner: str = "alice") -> str:
    dataset = store.create_dataset(
        owner,
        [{"name": "a.png", "size_bytes": 1, "sha256": hashlib.sha256(b"a").hexdigest(), "mime_type": "image/png"}],
    )
    store.finish_verification(dataset["id"], valid=True, manifest={"valid_count": 1, "invalid_count": 0})
    return store.create_job(
        owner,
        {"dataset_id": dataset["id"], "profile_revision_id": "local-tiny-v1", "trigger_token": "style", "execution_mode": "manual"},
    )["job_id"]


def _set_job_json(store: Store, job_id: str, column: str, value: dict) -> None:
    with store.transaction() as connection:
        connection.execute(f"UPDATE jobs SET {column}=? WHERE id=?", (json.dumps(value), job_id))


def _pause_after_prepare(store: Store, job_id: str) -> None:
    prepare = store.claim_task([])
    assert prepare and prepare["stage"] == "PREPARE"
    assert store.complete_task(
        prepare["attempt_id"],
        prepare["token"],
        {"prepared": {"manifest_path": "prepared"}, "_input": {"manifest_path": "input"}},
    )


def test_manual_inline_caption_pauses_then_freezes_each_user_stage(tmp_path):
    store = _store(tmp_path)
    job_id = _manual_job(store)
    prepare = store.claim_task([])
    assert prepare and prepare["stage"] == "PREPARE"
    assert store.complete_task(prepare["attempt_id"], prepare["token"], {"prepared": {"manifest_path": "prepared"}, "_input": {"manifest_path": "input"}})
    job = store.job("alice", job_id)
    assert (job["state"], job["waiting_for_stage"]) == ("WAITING_FOR_USER", "TRAIN")
    assert store.claim_task(["cpu-test-0"]) is None

    store.advance_job("alice", job_id, {"stage": "TRAIN", "training_overrides": {"max_steps": 3}})
    train = store.claim_task(["cpu-test-0"])
    assert train and train["stage"] == "TRAIN"
    assert store.complete_task(train["attempt_id"], train["token"], {"state": "COMPLETED"})
    job = store.job("alice", job_id)
    assert job["waiting_for_stage"] == "EVALUATE"
    assert job["training_overrides"] == {"max_steps": 3}

    store.advance_job("alice", job_id, {"stage": "EVALUATE", "evaluation_overrides": {"prompts": ["test"], "seeds": [9]}})
    evaluation = store.claim_task(["cpu-test-0"])
    assert evaluation and evaluation["stage"] == "EVALUATE"
    assert store.complete_task(evaluation["attempt_id"], evaluation["token"], {"quality_status": "UNCALIBRATED"})
    assert store.job("alice", job_id)["waiting_for_stage"] == "PUBLISH"


def test_advance_conflict_and_waiting_cancel_are_atomic(tmp_path):
    store = _store(tmp_path)
    job_id = _manual_job(store)
    with pytest.raises(PipelineError, match="not waiting"):
        store.advance_job("alice", job_id, {"stage": "TRAIN"})
    prepare = store.claim_task([])
    assert prepare
    store.complete_task(prepare["attempt_id"], prepare["token"], {"prepared": {"manifest_path": "prepared"}, "_input": {"manifest_path": "input"}})
    assert store.cancel_job("alice", job_id)["state"] == "CANCELLED"
    assert store.job("alice", job_id)["state"] == "CANCELLED"


def test_manual_caption_task_remains_automatic_before_training_pause(tmp_path):
    store = _store(tmp_path)
    job_id = _manual_job(store)
    prepare = store.claim_task([])
    assert prepare
    assert store.complete_task(prepare["attempt_id"], prepare["token"], {"manifest_path": "prepared"})
    caption = store.claim_task([])
    assert caption and caption["stage"] == "CAPTION"
    assert store.complete_task(caption["attempt_id"], caption["token"], {"manifest_path": "input"})
    job = store.job("alice", job_id)
    assert (job["state"], job["waiting_for_stage"]) == ("WAITING_FOR_USER", "TRAIN")


def test_advance_uses_frozen_profile_not_later_operator_settings(tmp_path):
    store = _store(tmp_path)
    job_id = _manual_job(store)
    prepare = store.claim_task([])
    assert prepare
    store.complete_task(prepare["attempt_id"], prepare["token"], {"prepared": {"manifest_path": "prepared"}, "_input": {"manifest_path": "input"}})
    # A mutable process-level configuration must not change validation or the
    # eventual task configuration of a job already admitted to the queue.
    store.settings.train.learning_rate = 0
    store.advance_job("alice", job_id, {"stage": "TRAIN", "training_overrides": {"max_steps": 3}})
    assert store.job("alice", job_id)["training_overrides"] == {"max_steps": 3}


def test_additive_migration_exposes_manual_workflow_columns(tmp_path):
    store = _store(tmp_path)
    with store.reader() as connection:
        job_columns = {row["name"] for row in connection.execute("PRAGMA table_info(jobs)")}
        attempt_columns = {row["name"] for row in connection.execute("PRAGMA table_info(task_attempts)")}
    assert {"execution_mode", "waiting_for_stage", "evaluation_overrides_json", "error_details_json"} <= job_columns
    assert {"error_details_json"} <= attempt_columns


def test_additive_migration_retains_legacy_jobs_under_concurrent_initialization(tmp_path):
    settings = Settings(data_dir=tmp_path / "legacy", api_keys={"secret": "alice"})
    settings.data_dir.mkdir(parents=True)
    with sqlite3.connect(settings.db_path) as connection:
        connection.execute(
            """CREATE TABLE jobs (
            id TEXT PRIMARY KEY, owner_id TEXT NOT NULL, dataset_id TEXT NOT NULL, state TEXT NOT NULL,
            current_stage TEXT, candidate_index INTEGER NOT NULL DEFAULT 0, trigger_token TEXT NOT NULL,
            profile_json TEXT NOT NULL, training_overrides_json TEXT NOT NULL, prepared_json TEXT,
            input_json TEXT, training_json TEXT, evaluation_json TEXT, error_code TEXT,
            error_message TEXT, cancel_requested_at REAL, gpu_seconds_charged REAL NOT NULL DEFAULT 0,
            created_at REAL NOT NULL, updated_at REAL NOT NULL, model_id TEXT)"""
        )
        connection.execute(
            "INSERT INTO jobs(id,owner_id,dataset_id,state,trigger_token,profile_json,training_overrides_json,created_at,updated_at) VALUES(?,?,?,?,?,?,?,?,?)",
            ("legacy-job", "alice", "legacy-dataset", "READY", "style", "{}", "{}", 1, 1),
        )
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        stores = list(executor.map(lambda _: Store(settings), range(2)))
    with stores[0].reader() as connection:
        legacy = connection.execute("SELECT execution_mode,waiting_for_stage,evaluation_overrides_json,error_details_json FROM jobs WHERE id='legacy-job'").fetchone()
    assert tuple(legacy) == ("auto", None, "{}", None)


def test_artifact_listing_is_owner_scoped_and_never_accepts_a_path(tmp_path):
    store = _store(tmp_path)
    job_id = _manual_job(store)
    report = write_manifest(store.artifacts / job_id / "attempt-one" / "prepared.json", {"kind": "prepared"})
    with store.transaction() as connection:
        connection.execute("UPDATE jobs SET prepared_json=? WHERE id=?", (json.dumps(report), job_id))
    artifacts = store.artifacts_for_job("alice", job_id)
    assert [item["kind"] for item in artifacts] == ["prepared_manifest"]
    assert store.artifact_for_job("alice", job_id, artifacts[0]["id"])["path"] == report["manifest_path"]
    with pytest.raises(PipelineError) as error:
        store.artifact_for_job("bob", job_id, artifacts[0]["id"])
    assert error.value.code == "NOT_FOUND"


@pytest.mark.parametrize("path_kind", ["outside", "lexical_cross_job", "symlink_within", "symlink_outside"])
def test_artifact_download_rejects_paths_outside_job_or_symlinks(tmp_path, path_kind):
    store = _store(tmp_path)
    job_id = _manual_job(store)
    job_root = store.artifacts / job_id
    job_root.mkdir(parents=True, exist_ok=True)

    if path_kind == "outside":
        declared = tmp_path / "outside.json"
        declared.write_text("outside")
    elif path_kind == "lexical_cross_job":
        other_job_id = _manual_job(store)
        other_file = store.artifacts / other_job_id / "other.json"
        other_file.parent.mkdir(parents=True, exist_ok=True)
        other_file.write_text("other")
        declared = job_root / ".." / other_job_id / other_file.name
    elif path_kind == "symlink_within":
        target = job_root / "target.json"
        target.write_text("target")
        declared = job_root / "link.json"
        declared.symlink_to(target)
    else:
        target = tmp_path / "outside.json"
        target.write_text("outside")
        declared = job_root / "link.json"
        declared.symlink_to(target)

    _set_job_json(store, job_id, "prepared_json", {"manifest_path": str(declared)})
    artifact = next(item for item in store.artifacts_for_job("alice", job_id) if item["kind"] == "prepared_manifest")
    with pytest.raises(PipelineError) as error:
        store.artifact_for_job("alice", job_id, artifact["id"])
    assert error.value.code == "NOT_FOUND"


def test_evaluation_relative_symlink_artifacts_are_not_downloadable(tmp_path):
    store = _store(tmp_path)
    job_id = _manual_job(store)
    report_dir = store.artifacts / job_id / "evaluation"
    report = write_manifest(report_dir / "report.json", {"kind": "evaluation"})
    for name in ("base.png", "adapter.png"):
        target = report_dir / f"{name}.target"
        target.write_bytes(b"image")
        (report_dir / name).symlink_to(target)
    _set_job_json(
        store,
        job_id,
        "evaluation_json",
        {
            "manifest_path": report["manifest_path"],
            "paired_outputs": [{"prompt": "test", "seed": 1, "base_image": "base.png", "adapter_image": "adapter.png"}],
        },
    )

    images = [item for item in store.artifacts_for_job("alice", job_id) if item["kind"] == "evaluation_image"]
    assert {item["variant"] for item in images} == {"base", "adapter"}
    for image in images:
        with pytest.raises(PipelineError) as error:
            store.artifact_for_job("alice", job_id, image["id"])
        assert error.value.code == "NOT_FOUND"


def test_job_listing_is_owner_scoped_and_paginated(tmp_path):
    store = _store(tmp_path)
    alice_jobs = {_manual_job(store) for _ in range(3)}
    bob_job = _manual_job(store, "bob")

    page = store.jobs("alice", limit=2, offset=1)
    assert page["total"] == 3
    assert (page["limit"], page["offset"]) == (2, 1)
    assert len(page["items"]) == 2
    assert {item["id"] for item in page["items"]} <= alice_jobs

    bob_page = store.jobs("bob", limit=50)
    assert bob_page["total"] == 1
    assert [item["id"] for item in bob_page["items"]] == [bob_job]


def test_waiting_manual_job_survives_store_reconstruction_without_claim(tmp_path):
    store = _store(tmp_path)
    job_id = _manual_job(store)
    _pause_after_prepare(store, job_id)

    reconstructed = Store(store.settings)
    job = reconstructed.job("alice", job_id)
    assert (job["state"], job["waiting_for_stage"]) == ("WAITING_FOR_USER", "TRAIN")
    assert reconstructed.claim_task(["gpu-0"]) is None


def test_waiting_manual_job_does_not_charge_gpu_time(tmp_path):
    store = _store(tmp_path)
    job_id = _manual_job(store)
    _pause_after_prepare(store, job_id)
    assert store.job("alice", job_id)["gpu_seconds_charged"] == 0
    assert store.claim_task(["gpu-0"]) is None
    store.reconcile()
    assert store.job("alice", job_id)["gpu_seconds_charged"] == 0

    store.advance_job("alice", job_id, {"stage": "TRAIN", "training_overrides": {"max_steps": 3}})
    train = store.claim_task(["gpu-0"])
    assert train and train["stage"] == "TRAIN" and train["gpu_slot"] == "gpu-0"
    assert store.complete_task(train["attempt_id"], train["token"], {"state": "COMPLETED"})
    charged = store.job("alice", job_id)["gpu_seconds_charged"]
    assert store.job("alice", job_id)["waiting_for_stage"] == "EVALUATE"
    store.reconcile()
    assert store.job("alice", job_id)["gpu_seconds_charged"] == charged
    assert store.claim_task(["gpu-0"]) is None


def test_manual_advance_rejects_overrides_for_other_stage(tmp_path):
    store = _store(tmp_path)
    job_id = _manual_job(store)
    _pause_after_prepare(store, job_id)

    with pytest.raises(PipelineError) as error:
        store.advance_job("alice", job_id, {"stage": "TRAIN", "evaluation_overrides": {"prompts": ["wrong"]}})
    assert error.value.code == "INVALID_OVERRIDES"
    assert store.job("alice", job_id)["waiting_for_stage"] == "TRAIN"

    store.advance_job("alice", job_id, {"stage": "TRAIN"})
    train = store.claim_task(["gpu-0"])
    assert train
    assert store.complete_task(train["attempt_id"], train["token"], {"state": "COMPLETED"})
    with pytest.raises(PipelineError) as error:
        store.advance_job("alice", job_id, {"stage": "EVALUATE", "training_overrides": {"max_steps": 3}})
    assert error.value.code == "INVALID_OVERRIDES"
    assert store.job("alice", job_id)["waiting_for_stage"] == "EVALUATE"

    store.advance_job("alice", job_id, {"stage": "EVALUATE"})
    evaluation = store.claim_task(["gpu-0"])
    assert evaluation
    assert store.complete_task(evaluation["attempt_id"], evaluation["token"], {"quality_status": "UNCALIBRATED"})
    with pytest.raises(PipelineError) as error:
        store.advance_job("alice", job_id, {"stage": "PUBLISH", "evaluation_overrides": {"prompts": ["wrong"]}})
    assert error.value.code == "INVALID_OVERRIDES"
    assert store.job("alice", job_id)["waiting_for_stage"] == "PUBLISH"


def test_evaluation_quality_failure_rejects_without_waiting_to_publish(tmp_path):
    store = _store(tmp_path)
    job_id = _manual_job(store)
    _pause_after_prepare(store, job_id)
    store.advance_job("alice", job_id, {"stage": "TRAIN"})
    train = store.claim_task(["gpu-0"])
    assert train
    assert store.complete_task(train["attempt_id"], train["token"], {"state": "COMPLETED"})
    store.advance_job("alice", job_id, {"stage": "EVALUATE"})
    evaluation = store.claim_task(["gpu-0"])
    assert evaluation
    assert store.complete_task(evaluation["attempt_id"], evaluation["token"], {"quality_status": "FAIL"})

    job = store.job("alice", job_id)
    assert (job["state"], job["waiting_for_stage"]) == ("QUALITY_REJECTED", None)
    assert store.claim_task(["gpu-0"]) is None
    with pytest.raises(PipelineError) as error:
        store.advance_job("alice", job_id, {"stage": "PUBLISH"})
    assert error.value.code == "JOB_TERMINAL"


@pytest.mark.parametrize("evaluation_overrides", [{"prompts": "not-a-list"}, {"seeds": 7}, {"prompts": None}, {"seeds": None}])
def test_evaluation_overrides_reject_non_collection_values(tmp_path, evaluation_overrides):
    store = _store(tmp_path)
    job_id = _manual_job(store)
    _pause_after_prepare(store, job_id)
    store.advance_job("alice", job_id, {"stage": "TRAIN"})
    train = store.claim_task(["gpu-0"])
    assert train
    assert store.complete_task(train["attempt_id"], train["token"], {"state": "COMPLETED"})
    with pytest.raises(PipelineError) as error:
        store.advance_job("alice", job_id, {"stage": "EVALUATE", "evaluation_overrides": evaluation_overrides})
    assert error.value.code == "INVALID_OVERRIDES"
