from __future__ import annotations

from pathlib import Path

from PIL import Image
from safetensors.torch import load_file

from lora_pipeline.common import sha256_file, write_manifest
from lora_pipeline.common import PipelineError
from lora_pipeline.config import TrainConfig
from lora_pipeline.training import _base_identity, train


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


def test_training_rejects_manifest_or_image_byte_drift(tmp_path: Path):
    manifest = _manifest(tmp_path)
    Image.new("RGB", (16, 16), (255, 0, 0)).save(manifest["train"][0]["path"])
    try:
        train(manifest, tmp_path / "run", _config(max_steps=1))
    except PipelineError as error:
        assert error.code == "CHECKSUM_MISMATCH"
    else:  # pragma: no cover - makes the expected data immutability explicit
        raise AssertionError("mutated image was accepted")


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
