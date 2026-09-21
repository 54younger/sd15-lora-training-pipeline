from pathlib import Path

import pytest
from PIL import Image

from lora_pipeline.common import PipelineError
from lora_pipeline.config import DataConfig
from lora_pipeline.data import generate_synthetic_dataset, prepare_dataset


def _config() -> DataConfig:
    return DataConfig(min_images=10, max_images=30, min_train_images=6, min_validation_images=2, min_groups_per_split=2, min_side=64)


def test_prepare_dataset_normalises_deduplicates_and_keeps_groups_together(tmp_path: Path):
    files = generate_synthetic_dataset(tmp_path / "source", count=12, size=96)
    duplicate = tmp_path / "source" / "same-pixels.png"
    Image.open(files[0]["path"]).save(duplicate)
    files.append({"id": "copy", "name": "client-name.png", "path": str(duplicate)})
    config = _config()
    manifest = prepare_dataset(files, tmp_path / "prepared", config, dataset_id="alpha")

    assert Path(manifest["manifest_path"]).is_file()
    assert manifest["statistics"]["exact_duplicates"] == 1
    assert manifest["statistics"]["train_images"] >= config.min_train_images
    assert manifest["statistics"]["validation_images"] >= config.min_validation_images
    train_groups = {item["group_id"] for item in manifest["train"]}
    validation_groups = {item["group_id"] for item in manifest["validation"]}
    assert train_groups.isdisjoint(validation_groups)
    assert all(Path(item["path"]).is_absolute() and Path(item["path"]).suffix == ".png" for item in manifest["train"])


def test_prepare_dataset_is_deterministic_and_rejects_invalid_images(tmp_path: Path):
    files = generate_synthetic_dataset(tmp_path / "source", count=12, size=96)
    broken = tmp_path / "source" / "broken.jpg"
    broken.write_bytes(b"not an image")
    files.append({"id": "broken", "name": "broken.jpg", "path": str(broken)})
    first = prepare_dataset(files, tmp_path / "one", _config(), dataset_id="repeat")
    second = prepare_dataset(files, tmp_path / "two", _config(), dataset_id="repeat")
    assert [item["id"] for item in first["validation"]] == [item["id"] for item in second["validation"]]
    assert first["statistics"]["rejected"] == 1


def test_prepare_dataset_fails_when_cleaning_leaves_too_few_images(tmp_path: Path):
    files = generate_synthetic_dataset(tmp_path / "source", count=10, size=96)
    with pytest.raises(PipelineError, match="Uploaded image count") as error:
        prepare_dataset(files[:9], tmp_path / "prepared", _config())
    assert error.value.code == "DATASET_IMAGE_COUNT"


def test_total_byte_limit_and_infeasible_group_split_are_dataset_errors(tmp_path: Path):
    files = generate_synthetic_dataset(tmp_path / "source", count=12, size=96)
    tight_limit = DataConfig(**{**_config().__dict__, "max_total_bytes": 1})
    with pytest.raises(PipelineError) as total_error:
        prepare_dataset(files, tmp_path / "too-large", tight_limit)
    assert total_error.value.code == "DATASET_TOO_LARGE"

    one_group = DataConfig(**{**_config().__dict__, "phash_distance": 64})
    with pytest.raises(PipelineError) as group_error:
        prepare_dataset(files, tmp_path / "one-group", one_group)
    assert group_error.value.code == "DATASET_UNSUITABLE"


def test_pixel_limit_is_checked_before_full_decode(tmp_path: Path, monkeypatch):
    files = generate_synthetic_dataset(tmp_path / "source", count=10, size=96)
    original_load = Image.Image.load

    def fail_if_decoded(self, *args, **kwargs):
        if self.size == (96, 96):
            raise AssertionError("pixel-limited image was loaded")
        return original_load(self, *args, **kwargs)

    monkeypatch.setattr(Image.Image, "load", fail_if_decoded)
    restricted = DataConfig(**{**_config().__dict__, "max_pixels": 10})
    with pytest.raises(PipelineError) as error:
        prepare_dataset(files, tmp_path / "pixel-limit", restricted)
    assert error.value.code == "DATASET_TOO_SMALL"
