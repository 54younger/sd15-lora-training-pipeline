from pathlib import Path

import pytest

from lora_pipeline.captions import caption_dataset
from lora_pipeline.common import PipelineError
from lora_pipeline.config import DataConfig
from lora_pipeline.data import generate_synthetic_dataset, prepare_dataset


def _prepared(tmp_path: Path) -> dict:
    files = generate_synthetic_dataset(tmp_path / "source", count=12, size=96)
    config = DataConfig(min_images=10, max_images=20, min_train_images=6, min_validation_images=2, min_groups_per_split=2, min_side=64)
    prepared = prepare_dataset(files, tmp_path / "prepared", config)
    for entry in prepared["train"]:
        entry.pop("caption", None)
    prepared["train"][0]["caption"] = "a supplied red object"
    return prepared


def test_template_captions_preserve_user_text_and_parent_digest(tmp_path: Path):
    prepared = _prepared(tmp_path)
    training_input = caption_dataset(prepared, tmp_path / "captions", trigger_token="watercolor")
    assert Path(training_input["manifest_path"]).name == "training-input.json"
    assert training_input["prepared_manifest_sha256"] == prepared["manifest_sha256"]
    supplied = next(item for item in training_input["train"] if item["caption_provenance"] == "user")
    assert supplied["caption"] == "in watercolor style, a supplied red object"
    assert all(item["caption"].startswith("in watercolor style, ") for item in training_input["train"])
    assert training_input["captioning"]["counts"]["template"] > 0
    generated = next(item for item in training_input["train"] if item["caption_provenance"] == "template")
    assert generated["caption"] == "in watercolor style, an image"


def test_caption_validation_and_blip_failure_are_actionable(tmp_path: Path):
    prepared = _prepared(tmp_path)
    prepared["train"][0]["caption"] = "bad\ncaption"
    with pytest.raises(PipelineError) as invalid:
        caption_dataset(prepared, tmp_path / "captions")
    assert invalid.value.code == "INVALID_CAPTION"

    prepared = _prepared(tmp_path / "other")
    for entry in prepared["train"]:
        entry.pop("caption", None)
    with pytest.raises(PipelineError) as failure:
        caption_dataset(prepared, tmp_path / "blip", mode="blip", model_name="not/a-real-local-model", local_files_only=True)
    assert failure.value.code in {"BLIP_UNAVAILABLE", "BLIP_CAPTION_FAILED"}
