"""Full offline API -> independently spawned supervisor -> LoRA artifact flow."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import concurrent.futures
from pathlib import Path

from fastapi.testclient import TestClient
import pytest

from lora_pipeline.api import create_app
from lora_pipeline.common import sha256_file
from lora_pipeline.config import EvalConfig, Settings, TrainConfig
from lora_pipeline.data import generate_synthetic_dataset


@pytest.mark.parametrize("manual", [False, True])
def test_http_dataset_to_independent_worker_and_unverified_download(tmp_path, manual):
    raw = {
        "data_dir": str(tmp_path / "service"),
        "api_keys": {"alice-key": "alice", "bob-key": "bob"},
        "admin_keys": ["alice-key"],
        "enable_test_backend": True,
        "fake_slots": 1,
        "heartbeat_seconds": 1,
        "lease_seconds": 30,
        "train": {
            "backend": "tiny",
            "device": "cpu",
            "precision": "fp32",
            "resolution": 16,
            "rank": 2,
            "lora_alpha": 2,
            "max_steps": 2,
            "checkpoint_every": 1,
            "gradient_accumulation_steps": 1,
        },
        "evaluation": {"device": "cpu", "prompts": ["a red circle"], "seeds": [42], "inference_steps": 1},
    }
    settings = Settings(
        **{**raw, "train": TrainConfig(**raw["train"]), "evaluation": EvalConfig(**raw["evaluation"])}
    )
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(raw))
    files = generate_synthetic_dataset(tmp_path / "images", count=100)
    headers = {"Authorization": "Bearer alice-key"}
    client = TestClient(create_app(settings))
    request = {
        "files": [
            {
                "name": item["name"],
                "size_bytes": Path(item["path"]).stat().st_size,
                "sha256": sha256_file(item["path"]),
                "mime_type": "image/png",
                "caption": item["caption"],
            }
            for item in files
        ]
    }
    response = client.post("/v1/datasets", json=request, headers={**headers, "Idempotency-Key": "dataset"})
    assert response.status_code == 201, response.text
    dataset = response.json()
    dataset_id = dataset.get("dataset_id", dataset.get("id"))
    source = {f["name"]: f for f in files}
    for item in dataset["files"]:
        file_id = item.get("file_id", item.get("id"))
        response = client.put(
            f"/v1/datasets/{dataset_id}/files/{file_id}",
            headers=headers,
            content=Path(source[item["name"]]["path"]).read_bytes(),
        )
        assert response.status_code == 204, response.text
    response = client.post(
        f"/v1/datasets/{dataset_id}/complete", json={}, headers={**headers, "Idempotency-Key": "complete"}
    )
    assert response.status_code == 202, response.text
    env = {key: value for key, value in os.environ.items() if not key.startswith("LORA_")}
    env.update(
        LORA_CONFIG=str(config_path),
        HF_HUB_OFFLINE="1",
        OMP_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
    )
    log_path = tmp_path / "worker.log"
    with log_path.open("w") as log:
        process = subprocess.Popen(
            [sys.executable, "-m", "lora_pipeline", "worker"],
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            deadline = time.monotonic() + 120
            while time.monotonic() < deadline:
                state = client.get(f"/v1/datasets/{dataset_id}", headers=headers).json()
                if state["state"] == "COMPLETED":
                    break
                assert state["state"] != "INVALID", state
                assert process.poll() is None, log_path.read_text()
                time.sleep(0.1)
            else:
                raise AssertionError(log_path.read_text())
            response = client.post(
                "/v1/training-jobs",
                json={
                    "dataset_id": dataset_id,
                    "profile_revision_id": "local-tiny-v1",
                    "trigger_token": "geometry",
                    "execution_mode": "manual" if manual else "auto",
                },
                headers={**headers, "Idempotency-Key": "train"},
            )
            assert response.status_code == 202, response.text
            job_id = response.json().get("job_id", response.json().get("id"))
            if manual:
                def wait_for_waiting(stage: str):
                    latest = None
                    while time.monotonic() < deadline:
                        latest = client.get(f"/v1/training-jobs/{job_id}", headers=headers).json()
                        if latest["state"] == "WAITING_FOR_USER" and latest["waiting_for_stage"] == stage:
                            assert stage not in latest["stages"]
                            # Reloading after a pause must not enqueue work by itself.
                            time.sleep(0.25)
                            reloaded = client.get(f"/v1/training-jobs/{job_id}", headers=headers).json()
                            assert (reloaded["state"], reloaded["waiting_for_stage"], reloaded["stages"]) == (
                                latest["state"], latest["waiting_for_stage"], latest["stages"]
                            )
                            return reloaded
                        assert process.poll() is None, log_path.read_text()
                        time.sleep(0.1)
                    raise AssertionError(f"Timed out waiting for {stage}: {latest}\n{log_path.read_text()}")

                wait_for_waiting("TRAIN")

                def advance_once(key: str):
                    with TestClient(create_app(settings)) as another_client:
                        return another_client.post(
                            f"/v1/training-jobs/{job_id}/advance",
                            json={"stage": "TRAIN", "training_overrides": {"max_steps": 2, "seed": 7}},
                            headers={**headers, "Idempotency-Key": key},
                        )

                # Separate idempotency keys still race through the same atomic
                # expected-stage check: exactly one task enters the queue.
                with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
                    advances = list(executor.map(advance_once, ["advance-a", "advance-b"]))
                assert sorted(value.status_code for value in advances) == [202, 409]
                successful_key = next(
                    key for key, response in zip(["advance-a", "advance-b"], advances, strict=True)
                    if response.status_code == 202
                )
                replay = client.post(
                    f"/v1/training-jobs/{job_id}/advance",
                    json={"stage": "TRAIN", "training_overrides": {"max_steps": 2, "seed": 7}},
                    headers={**headers, "Idempotency-Key": successful_key},
                )
                assert replay.status_code == 202
                job_after_advance = client.get(f"/v1/training-jobs/{job_id}", headers=headers).json()
                assert job_after_advance["training_overrides"] == {"max_steps": 2, "seed": 7}
                wait_for_waiting("EVALUATE")
                response = client.post(
                    f"/v1/training-jobs/{job_id}/advance",
                    json={"stage": "EVALUATE", "evaluation_overrides": {"prompts": ["a red circle"], "seeds": [7], "inference_steps": 1, "guidance_scale": 2}},
                    headers={**headers, "Idempotency-Key": "advance-evaluate"},
                )
                assert response.status_code == 202, response.text
                wait_for_waiting("PUBLISH")
                response = client.post(
                    f"/v1/training-jobs/{job_id}/advance",
                    json={"stage": "PUBLISH"}, headers={**headers, "Idempotency-Key": "advance-publish"},
                )
                assert response.status_code == 202, response.text
            while time.monotonic() < deadline:
                job = client.get(f"/v1/training-jobs/{job_id}", headers=headers).json()
                if job["state"] in {"READY", "COMPLETED_UNVERIFIED", "FAILED", "QUALITY_REJECTED"}:
                    break
                assert process.poll() is None, log_path.read_text()
                time.sleep(0.2)
            else:
                raise AssertionError(f"Timed out: {job}\n{log_path.read_text()}")
            assert job["state"] == "COMPLETED_UNVERIFIED", f"{job}\n{log_path.read_text()}"
            model_id = job["model_id"]
            report = client.get(f"/v1/training-jobs/{job_id}/evaluation", headers=headers)
            assert report.status_code == 200, report.text
            assert client.get(f"/v1/models/{model_id}/download", headers=headers).status_code == 409
            result = client.get(f"/v1/models/{model_id}/download?allow_unverified=true", headers=headers)
            assert result.status_code == 200 and len(result.content) > 0
            assert (
                client.get(f"/v1/models/{model_id}", headers={"Authorization": "Bearer bob-key"}).status_code
                == 404
            )
            assert client.get("/health/ready").status_code == 200
        finally:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
