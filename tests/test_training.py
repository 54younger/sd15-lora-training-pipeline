from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image
from safetensors.torch import load_file

from lora_pipeline.common import sha256_file, write_manifest
from lora_pipeline.common import PipelineError
from lora_pipeline.config import TrainConfig
from lora_pipeline.training import (
    _base_identity,
    _cuda_peak_memory_stats,
    _load_sd15,
    _resolve_sd15_base,
    train,
    validate_input_integrity,
)


def _manifest(tmp_path: Path) -> dict:
    entries = []
    for index, colour in enumerate(((32, 90, 160), (160, 90, 32))):
        path = tmp_path / f"sample-{index}.png"
        Image.new("RGB", (16, 16), colour).save(path)
        entries.append(
            {
                "id": str(index),
                "path": str(path),
                "sha256": sha256_file(path),
                "caption": f"a coloured sample {index}",
            }
        )
    return write_manifest(tmp_path / "training-input.json", {"train": entries, "validation": entries})


def _config(**overrides) -> TrainConfig:
    defaults = {
        "backend": "tiny",
        "resolution": 16,
        "device": "cpu",
        "precision": "fp32",
        "rank": 2,
        "lora_alpha": 2,
        "max_steps": 2,
        "checkpoint_every": 1,
        "batch_size": 1,
        "gradient_accumulation_steps": 1,
        "seed": 7,
    }
    return TrainConfig(**(defaults | overrides))


def test_tiny_diffusers_training_updates_only_lora_and_exports_adapter(tmp_path: Path):
    result = train(_manifest(tmp_path), tmp_path / "run", _config(max_steps=1))

    assert result["state"] == "COMPLETED"
    assert result["global_step"] == 1
    assert result["trainable_parameter_count"] > 0
    assert result["frozen_parameter_count"] > result["trainable_parameter_count"]
    state = load_file(result["adapter_path"])
    assert state and all("lora_" in key for key in state)
    assert Path(result["checkpoint_path"]).is_file()


def test_checkpoint_resume_matches_continuous_deterministic_training(tmp_path: Path):
    manifest = _manifest(tmp_path)
    continuous = train(manifest, tmp_path / "continuous", _config())
    stopped = train(manifest, tmp_path / "stopped", _config(), stop_after_step=1)
    resumed_updates: list[dict] = []
    resumed = train(
        manifest,
        tmp_path / "resumed",
        _config(),
        resume_from=Path(stopped["checkpoint_path"]),
        progress=resumed_updates.append,
    )

    left, right = load_file(continuous["adapter_path"]), load_file(resumed["adapter_path"])
    assert continuous["global_step"] == resumed["global_step"] == 2
    assert left.keys() == right.keys()
    for key in left:
        assert (left[key] == right[key]).all(), key
    assert resumed_updates[-1]["samples_processed"] == 2
    assert resumed_updates[-1]["samples_processed_this_run"] == 1


def test_training_progress_has_lifecycle_and_step_schema(tmp_path: Path):
    updates: list[dict] = []
    train(_manifest(tmp_path), tmp_path / "run", _config(), progress=updates.append)

    phases = [update["phase"] for update in updates]
    assert phases[0] == "training_started"
    assert "model_loading" in phases
    assert "tiny_components_loading" in phases
    assert "optimizer_initializing" in phases
    assert "optimizer_initialized" in phases
    assert "checkpoint_saving" in phases
    assert phases[-1] == "training_completed"
    step_updates = [update for update in updates if update["phase"] == "training"]
    assert [update["global_step"] for update in step_updates] == [1, 2]
    assert all(update["current"] == update["global_step"] and update["total"] == 2 for update in step_updates)


def test_training_rejects_manifest_or_image_byte_drift(tmp_path: Path):
    manifest = _manifest(tmp_path)
    entry = manifest["train"][0]
    Image.new("RGB", (16, 16), (255, 0, 0)).save(entry["path"])
    try:
        train(manifest, tmp_path / "run", _config(max_steps=1))
    except PipelineError as error:
        assert error.code == "CHECKSUM_MISMATCH"
        assert error.details == {
            "id": entry["id"],
            "split": "train",
            "path": entry["path"],
            "expected_sha256": entry["sha256"],
            "actual_sha256": sha256_file(entry["path"]),
        }
    else:  # pragma: no cover - makes the expected data immutability explicit
        raise AssertionError("mutated image was accepted")


def test_integrity_error_includes_missing_image_checksum_details(tmp_path: Path):
    manifest = _manifest(tmp_path)
    missing = tmp_path / "missing.png"
    manifest["train"][0]["path"] = str(missing)
    manifest = write_manifest(
        manifest["manifest_path"],
        {"train": manifest["train"], "validation": manifest["validation"]},
    )
    entry = manifest["train"][0]

    try:
        validate_input_integrity(manifest)
    except PipelineError as error:
        assert error.code == "CHECKSUM_MISMATCH"
        assert {
            key: value for key, value in error.details.items() if key != "read_error"
        } == {
            "id": entry["id"],
            "split": "train",
            "path": str(missing),
            "expected_sha256": entry["sha256"],
            "actual_sha256": None,
        }
        assert error.details["read_error"].startswith("FileNotFoundError:")
    else:  # pragma: no cover
        raise AssertionError("missing image was accepted")


def test_integrity_rejects_image_removed_during_checksum(tmp_path: Path, monkeypatch):
    manifest = _manifest(tmp_path)
    entry = manifest["train"][0]
    original_sha256_file = sha256_file

    def disappearing_checksum(path):
        if Path(path) == Path(entry["path"]):
            raise FileNotFoundError("removed while validating")
        return original_sha256_file(path)

    monkeypatch.setattr("lora_pipeline.training.sha256_file", disappearing_checksum)
    with pytest.raises(PipelineError) as error:
        validate_input_integrity(manifest)

    assert error.value.code == "CHECKSUM_MISMATCH"
    assert error.value.details == {
        "id": entry["id"],
        "split": "train",
        "path": entry["path"],
        "expected_sha256": entry["sha256"],
        "actual_sha256": None,
        "read_error": "FileNotFoundError: removed while validating",
    }


def test_integrity_rejects_unreadable_frozen_manifest(tmp_path: Path):
    manifest = _manifest(tmp_path)
    missing_manifest = tmp_path / "missing-training-input.json"
    manifest["manifest_path"] = str(missing_manifest)

    with pytest.raises(PipelineError) as error:
        validate_input_integrity(manifest)

    assert error.value.code == "CHECKSUM_MISMATCH"
    assert error.value.details["path"] == str(missing_manifest)
    assert error.value.details["expected_sha256"] == manifest["manifest_sha256"]
    assert error.value.details["actual_sha256"] is None
    assert error.value.details["read_error"].startswith("FileNotFoundError:")


def test_training_rejects_in_memory_caption_drift_from_frozen_manifest(tmp_path: Path):
    manifest = _manifest(tmp_path)
    manifest["train"][0]["caption"] = "a changed caption"
    try:
        train(manifest, tmp_path / "run", _config(max_steps=1))
    except PipelineError as error:
        assert error.code == "CHECKSUM_MISMATCH"
    else:  # pragma: no cover
        raise AssertionError("mutated caption was accepted")


def test_local_base_identity_tracks_relevant_weight_content(tmp_path: Path):
    root = tmp_path / "base"
    (root / "unet").mkdir(parents=True)
    (root / "model_index.json").write_text("{}")
    weights = root / "unet" / "diffusion_pytorch_model.safetensors"
    weights.write_bytes(b"first")
    before = _base_identity(root)
    weights.write_bytes(b"changed")
    assert _base_identity(root) != before


def _sd15_config(**overrides) -> TrainConfig:
    defaults = {
        "backend": "sd15",
        "model_name": "org/example-sd15",
        "revision": None,
        "resolution": 512,
        "device": "cpu",
        "precision": "fp32",
        "rank": 2,
        "lora_alpha": 2,
        "max_steps": 1,
        "checkpoint_every": 1,
        "local_files_only": True,
    }
    return TrainConfig(**(defaults | overrides))


def test_snapshot_resolution_error_includes_sanitized_actionable_details(monkeypatch):
    import huggingface_hub

    secret_url = "https://alice:secret@example.invalid/model?token=hf_topsecret"

    def unavailable(**_kwargs):
        raise RuntimeError(f"download failed: {secret_url}")

    monkeypatch.setenv("HF_HOME", "/tmp/hf-cache")
    monkeypatch.setattr(huggingface_hub, "snapshot_download", unavailable)
    config = _sd15_config()

    with pytest.raises(PipelineError) as error:
        _resolve_sd15_base(config)

    assert error.value.code == "BASE_MODEL_UNAVAILABLE"
    assert error.value.retryable is True
    assert error.value.message == "Could not resolve the SD 1.5 model snapshot"
    assert error.value.details == {
        "model_name": config.model_name,
        "revision": None,
        "local_files_only": True,
        "cache": "/tmp/hf-cache",
        "reason": "RuntimeError: unknown",
    }
    assert "secret" not in str(error.value.details)
    assert "hf_topsecret" not in str(error.value.details)


def test_component_load_error_includes_resolved_snapshot_details(tmp_path: Path, monkeypatch):
    import transformers

    root = tmp_path / "resolved-snapshot"
    root.mkdir()
    config = _sd15_config(revision="a1b2c3d4")

    monkeypatch.setattr(
        "lora_pipeline.training._resolve_sd15_base", lambda _config: (root, "a1b2c3d4", "fingerprint")
    )

    def unavailable(*_args, **_kwargs):
        raise OSError("token=hf_componentsecret model files are missing")

    monkeypatch.setattr(transformers.CLIPTokenizer, "from_pretrained", unavailable)
    updates: list[dict] = []
    with pytest.raises(PipelineError) as error:
        _load_sd15(config, device=object(), progress=updates.append)

    assert error.value.code == "BASE_MODEL_UNAVAILABLE"
    assert error.value.retryable is True
    assert error.value.details == {
        "model_name": config.model_name,
        "revision": "a1b2c3d4",
        "local_files_only": True,
        "cache": None,
        "resolved_root": str(root),
        "resolved_revision": "a1b2c3d4",
        "reason": "OSError: io",
    }
    assert [update["phase"] for update in updates] == [
        "sd15_snapshot_resolving",
        "sd15_snapshot_resolved",
        "sd15_tokenizer_loading",
    ]


def test_cuda_cleanup_failure_never_masks_model_loading_pipeline_error(tmp_path: Path, monkeypatch, caplog):
    class FakeCuda:
        @staticmethod
        def is_available():
            return True

        @staticmethod
        def empty_cache():
            raise OSError("driver teardown failed")

        @staticmethod
        def manual_seed_all(_seed):
            return None

    class FakeTorch:
        cuda = FakeCuda()

        @staticmethod
        def device(_name):
            return type("Device", (), {"type": "cpu"})()

        @staticmethod
        def manual_seed(_seed):
            return None

    original_error = PipelineError("BASE_MODEL_UNAVAILABLE", "Pinned model is unavailable")
    monkeypatch.setattr("lora_pipeline.training._torch", lambda: (FakeTorch(), None, None))
    monkeypatch.setattr(
        "lora_pipeline.training._tiny_model", lambda *_args: (_ for _ in ()).throw(original_error)
    )

    with pytest.raises(PipelineError) as error:
        train(_manifest(tmp_path), tmp_path / "run", _config(max_steps=1))

    assert error.value is original_error
    assert "CUDA cache cleanup failed (exception_type=OSError)" in caplog.text


def test_cuda_memory_telemetry_initialization_error_does_not_stop_training_setup(tmp_path: Path, monkeypatch, caplog):
    class FakeCuda:
        @staticmethod
        def is_available():
            return True

        @staticmethod
        def manual_seed_all(_seed):
            return None

        @staticmethod
        def reset_peak_memory_stats(_device):
            raise RuntimeError("Invalid device argument")

    class FakeTorch:
        cuda = FakeCuda()

        @staticmethod
        def device(_name):
            return type("Device", (), {"type": "cuda"})()

        @staticmethod
        def manual_seed(_seed):
            return None

    updates: list[dict] = []
    expected_error = PipelineError("MODEL_LOADING_SENTINEL", "model loading was reached")
    monkeypatch.setattr("lora_pipeline.training._torch", lambda: (FakeTorch(), None, None))
    monkeypatch.setattr(
        "lora_pipeline.training._tiny_model", lambda *_args: (_ for _ in ()).throw(expected_error)
    )

    with pytest.raises(PipelineError) as error:
        train(_manifest(tmp_path), tmp_path / "run", _config(device="cuda:0", max_steps=1), progress=updates.append)

    assert error.value is expected_error
    phases = [update["phase"] for update in updates]
    assert phases.index("cuda_memory_initializing") < phases.index("cuda_memory_telemetry_unavailable")
    assert phases.index("cuda_memory_telemetry_unavailable") < phases.index("model_loading")
    telemetry_update = updates[phases.index("cuda_memory_telemetry_unavailable")]
    assert telemetry_update["gpu_memory_telemetry_available"] is False
    assert "CUDA memory telemetry initialization failed (exception_type=RuntimeError)" in caplog.text


def test_cuda_peak_memory_stats_failure_is_best_effort(caplog):
    class FakeCuda:
        @staticmethod
        def max_memory_allocated(_device):
            raise RuntimeError("Invalid device argument")

        @staticmethod
        def max_memory_reserved(_device):
            raise AssertionError("second telemetry call must not run")

    stats = _cuda_peak_memory_stats(type("FakeTorch", (), {"cuda": FakeCuda()})(), object())

    assert stats is None
    assert "CUDA memory telemetry collection failed (exception_type=RuntimeError)" in caplog.text


@pytest.mark.parametrize(
    "message",
    [
        "Authorization: Bearer sk_live_ABC123",
        "Authorization: Basic dXNlcjpwYXNz",
        "HF_TOKEN=supersecret-value",
        "Bearer bare-secret-value",
        "Authorization: Token token-secret-value",
        '{"access_token":"json-secret-value"}',
    ],
)
def test_base_model_diagnostics_never_include_exception_text(monkeypatch, message):
    import huggingface_hub

    def unavailable(**_kwargs):
        raise RuntimeError(message)

    monkeypatch.setattr(huggingface_hub, "snapshot_download", unavailable)
    with pytest.raises(PipelineError) as error:
        _resolve_sd15_base(_sd15_config())

    assert error.value.details["reason"] == "RuntimeError: unknown"
    assert message not in str(error.value.details)


def test_base_model_diagnostics_redact_model_revision_and_cache_credentials(monkeypatch):
    import huggingface_hub

    model_name = "https://model-user:model-password@example.invalid/repository?token=model-token"
    revision = "HF_TOKEN=revision-secret"
    cache = "https://cache-user:cache-password@cache.invalid/path?access_token=cache-token"

    def unavailable(**_kwargs):
        raise RuntimeError("TOKEN=exception-secret")

    monkeypatch.setenv("HF_HOME", cache)
    monkeypatch.setattr(huggingface_hub, "snapshot_download", unavailable)
    with pytest.raises(PipelineError) as error:
        _resolve_sd15_base(_sd15_config(model_name=model_name, revision=revision))

    assert error.value.details["model_name"] == "https://***@example.invalid/repository?token=***"
    assert error.value.details["revision"] == "HF_TOKEN=***"
    assert error.value.details["cache"] == "https://***@cache.invalid/path?access_token=***"
    assert error.value.details["reason"] == "RuntimeError: unknown"
    for secret in (
        "model-password",
        "model-token",
        "revision-secret",
        "cache-password",
        "cache-token",
        "exception-secret",
    ):
        assert secret not in str(error.value.details)
