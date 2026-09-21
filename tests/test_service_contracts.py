"""Black-box contracts for the local REST service and durable store boundary."""

from __future__ import annotations

import concurrent.futures
import hashlib
import uuid
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from lora_pipeline.api import create_app
from lora_pipeline.common import write_manifest
from lora_pipeline.config import DataConfig, Settings


ALICE = {"Authorization": "Bearer alice-key"}
BOB = {"Authorization": "Bearer bob-key"}
ADMIN = {"Authorization": "Bearer admin-key"}


def _settings(tmp_path: Path) -> Settings:
    return Settings(
        data_dir=tmp_path / "service-data",
        api_keys={"alice-key": "alice", "bob-key": "bob"},
        admin_keys=["admin-key"],
        enable_test_backend=True,
        data=DataConfig(
            min_images=1,
            max_images=20,
            min_train_images=1,
            min_validation_images=1,
            min_groups_per_split=1,
            min_side=1,
        ),
    )


@pytest.fixture
def app(tmp_path: Path):
    return create_app(_settings(tmp_path))


@pytest.fixture
def client(app):
    with TestClient(app) as value:
        yield value


def _file(name: str, content: bytes = b"image-bytes", **extra) -> dict:
    return {
        "name": name,
        "size_bytes": len(content),
        "sha256": hashlib.sha256(content).hexdigest(),
        "mime_type": "image/png",
        **extra,
    }


def _create_dataset(client: TestClient, item: dict | None = None, *, headers: dict = ALICE) -> dict:
    response = client.post(
        "/v1/datasets",
        headers={**headers, "Idempotency-Key": f"dataset-{uuid.uuid4()}"},
        json={"files": [item or _file("sample.png")]},
    )
    assert response.status_code == 201, response.text
    return response.json()


def _verified_dataset(client: TestClient, app, *, content: bytes = b"image-bytes") -> str:
    dataset = _create_dataset(client, _file("sample.png", content))
    file_id = dataset["files"][0]["id"]
    response = client.put(f"/v1/datasets/{dataset['id']}/files/{file_id}", headers=ALICE, content=content)
    assert response.status_code == 204, response.text
    # Training admission only needs a durable, worker-produced frozen manifest;
    # no ML model or image decoder is involved in these API contract tests.
    manifest = write_manifest(
        Path(app.state.store.artifacts) / f"{dataset['id']}-prepared.json",
        {"schema_version": 1, "dataset_id": dataset["id"], "train": [], "validation": []},
    )
    app.state.store.finish_verification(
        dataset["id"],
        valid=True,
        manifest={
            **manifest,
            "valid_count": 1,
            "invalid_count": 0,
        },
    )
    return dataset["id"]


def test_upload_returns_empty_204_on_initial_and_repeated_upload(client: TestClient):
    content = b"image-bytes"
    dataset = _create_dataset(client, _file("sample.png", content))
    path = f"/v1/datasets/{dataset['id']}/files/{dataset['files'][0]['id']}"
    for _ in range(2):
        response = client.put(path, headers=ALICE, content=content)
        assert response.status_code == 204
        assert response.content == b""
        assert "content-length" not in response.headers
        assert "content-type" not in response.headers


@pytest.mark.parametrize("caption", ["x" * 513, "caption\x00with-control"])
def test_dataset_declaration_rejects_unsafe_caption_text(client: TestClient, caption: str):
    response = client.post(
        "/v1/datasets",
        headers={**ALICE, "Idempotency-Key": f"caption-{uuid.uuid4()}"},
        json={"files": [_file("sample.png", caption=caption)]},
    )
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "INVALID_CAPTION"


@pytest.mark.parametrize("seed", ["42", True])
def test_training_override_seed_must_be_json_integer(client: TestClient, app, seed):
    dataset_id = _verified_dataset(client, app)
    response = client.post(
        "/v1/training-jobs",
        headers={**ALICE, "Idempotency-Key": f"job-{uuid.uuid4()}"},
        json={
            "dataset_id": dataset_id,
            "profile_revision_id": "local-tiny-v1",
            "trigger_token": "teststyle",
            "training_overrides": {"seed": seed},
        },
    )
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "INVALID_OVERRIDES"


def test_bad_upload_can_retry_and_freeze_makes_object_immutable(client: TestClient):
    good = b"correct-bytes"
    dataset = _create_dataset(client, _file("sample.png", good))
    dataset_id, file_id = dataset["id"], dataset["files"][0]["id"]
    bad = client.put(f"/v1/datasets/{dataset_id}/files/{file_id}", headers=ALICE, content=b"wrong-bytes!!")
    assert bad.status_code == 422
    app_store = client.app.state.store
    assert app_store is not None
    assert app_store.dataset("alice", dataset_id, include_files=True)["files"][0]["uploaded"] == 0

    assert (
        client.put(f"/v1/datasets/{dataset_id}/files/{file_id}", headers=ALICE, content=good).status_code
        == 204
    )
    frozen = client.post(
        f"/v1/datasets/{dataset_id}/complete", headers={**ALICE, "Idempotency-Key": "freeze-1"}, json={}
    )
    assert frozen.status_code == 202
    retry = client.put(f"/v1/datasets/{dataset_id}/files/{file_id}", headers=ALICE, content=good)
    assert retry.status_code == 409
    assert retry.json()["error"]["code"] == "UPLOAD_NOT_ALLOWED"


def test_tenant_lookups_are_masked_as_not_found(client: TestClient):
    dataset = _create_dataset(client)
    response = client.get(f"/v1/datasets/{dataset['id']}", headers=BOB)
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "NOT_FOUND"


def test_concurrent_same_idempotency_key_creates_exactly_one_dataset(app):
    payload = {"files": [_file("same.png")]}

    def create_once(_):
        with TestClient(app) as concurrent_client:
            return concurrent_client.post(
                "/v1/datasets", headers={**ALICE, "Idempotency-Key": "same-key"}, json=payload
            )

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        responses = list(executor.map(create_once, range(4)))
    assert {response.status_code for response in responses} == {201}
    ids = {response.json()["id"] for response in responses}
    assert len(ids) == 1

    changed = {"files": [_file("changed.png")]}
    with TestClient(app) as concurrent_client:
        conflict = concurrent_client.post(
            "/v1/datasets", headers={**ALICE, "Idempotency-Key": "same-key"}, json=changed
        )
    assert conflict.status_code == 409
    assert conflict.json()["error"]["code"] == "IDEMPOTENCY_KEY_REUSED"


def test_health_metrics_and_unverified_model_download_opt_in(client: TestClient, app, tmp_path: Path):
    ready = client.get("/health/ready")
    assert ready.status_code == 503
    assert ready.json()["worker"] is False
    assert client.get("/metrics", headers=ALICE).status_code == 401
    metrics = client.get("/metrics", headers=ADMIN)
    assert metrics.status_code == 200
    assert metrics.headers["content-type"].startswith("text/plain")
    assert "# TYPE lora_jobs gauge" in metrics.text

    # Insert only the final local artifact record needed to exercise download
    # authorization.  Publication itself is tested with worker/store tests.
    adapter = tmp_path / "adapter.safetensors"
    model_manifest = tmp_path / "model-manifest.json"
    adapter.write_bytes(b"tiny-adapter")
    manifest = write_manifest(model_manifest, {"schema_version": 1})
    report_manifest = write_manifest(
        tmp_path / "evaluation-report.json", {"schema_version": 1, "kind": "evaluation"}
    )
    dataset_id = _verified_dataset(client, app, content=b"another-image")
    job = app.state.store.create_job(
        "alice",
        {
            "dataset_id": dataset_id,
            "profile_revision_id": "local-tiny-v1",
            "trigger_token": "teststyle",
            "training_overrides": {},
        },
    )
    task = app.state.store.claim_task([])
    assert task and task["job_id"] == job["job_id"]
    assert app.state.store.complete_task(task["attempt_id"], task["token"], {"progress": {}})
    for _stage in ("CAPTION", "TRAIN", "EVALUATE"):
        task = app.state.store.claim_task(["fake-gpu"])
        assert task
        assert app.state.store.complete_task(task["attempt_id"], task["token"], {"progress": {}})
    publish = app.state.store.claim_task([])
    assert publish and publish["stage"] == "PUBLISH"
    adapter_sha256 = hashlib.sha256(adapter.read_bytes()).hexdigest()
    created = app.state.store.publish(
        publish["attempt_id"],
        publish["token"],
        {
            "adapter_path": str(adapter),
            "adapter_sha256": adapter_sha256,
            "manifest_path": str(model_manifest),
            "manifest_sha256": manifest["manifest_sha256"],
            "input_manifest_sha256": None,
            "test_only": True,
        },
        {
            "quality_status": "UNCALIBRATED",
            "technical_pass": True,
            "adapter_sha256": adapter_sha256,
            "input_manifest_sha256": None,
            "manifest_path": report_manifest["manifest_path"],
            "manifest_sha256": report_manifest["manifest_sha256"],
        },
    )
    assert created and created["state"] == "UNVERIFIED"
    denied = client.get(f"/v1/models/{created['model_id']}/download", headers=ALICE)
    assert denied.status_code == 409
    permitted = client.get(f"/v1/models/{created['model_id']}/download?allow_unverified=true", headers=ALICE)
    assert permitted.status_code == 200
    assert permitted.content == b"tiny-adapter"
