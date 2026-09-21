import errno
import os
from pathlib import Path
import stat

import pytest
from PIL import Image, PngImagePlugin

import lora_pipeline.data as data
from lora_pipeline.common import PipelineError, sha256_file
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


def test_repeated_prepare_does_not_overwrite_prior_normalized_artifacts(tmp_path: Path, monkeypatch):
    files = generate_synthetic_dataset(tmp_path / "source", count=12, size=96)
    output = tmp_path / "prepared"
    first = prepare_dataset(files, output, _config(), dataset_id="repeat")
    first_entries = first["train"] + first["validation"]
    first_artifacts = {entry["path"]: entry["sha256"] for entry in first_entries}

    original_save = Image.Image.save

    def save_with_different_png_bytes(self, fp, format=None, **kwargs):
        if format == "PNG":
            pnginfo = PngImagePlugin.PngInfo()
            pnginfo.add_text("encoder-variation", "second-prepare")
            kwargs["pnginfo"] = pnginfo
        return original_save(self, fp, format, **kwargs)

    monkeypatch.setattr(Image.Image, "save", save_with_different_png_bytes)
    second = prepare_dataset(files, output, _config(), dataset_id="repeat")
    second_paths = {entry["path"] for entry in second["train"] + second["validation"]}

    assert first_artifacts.keys().isdisjoint(second_paths)
    assert all(sha256_file(path) == expected for path, expected in first_artifacts.items())
    assert not list((output / "images").glob(".normalizing-*.png"))


def test_normalized_image_syncs_file_before_its_directory(tmp_path: Path, monkeypatch):
    images_dir = tmp_path / "images"
    images_dir.mkdir()
    calls: list[str] = []
    original_fsync = data.os.fsync

    def track_fsync(descriptor: int):
        mode = os.fstat(descriptor).st_mode
        calls.append("directory" if stat.S_ISDIR(mode) else "file")
        return original_fsync(descriptor)

    monkeypatch.setattr(data.os, "fsync", track_fsync)
    image = Image.new("RGB", (96, 96), "red")
    data._write_normalized_image(image, images_dir, "a" * 64)

    assert "file" in calls and "directory" in calls
    assert calls.index("file") < calls.index("directory")


def test_prepare_preserves_legacy_ordinal_artifacts(tmp_path: Path):
    files = generate_synthetic_dataset(tmp_path / "source", count=12, size=96)
    image, pixel_hash, _ = data._normalise_image(Path(files[0]["path"]), _config())
    image.close()
    output = tmp_path / "prepared"
    legacy_path = output / "images" / f"0000-{pixel_hash[:16]}.png"
    legacy_path.parent.mkdir(parents=True)
    legacy_path.write_bytes(b"legacy artifact bytes")

    manifest = prepare_dataset(files, output, _config(), dataset_id="legacy")

    assert legacy_path.read_bytes() == b"legacy artifact bytes"
    assert all(
        Path(entry["path"]) != legacy_path.resolve()
        for entry in manifest["train"] + manifest["validation"]
    )


def test_prepare_fails_if_atomic_artifact_publishing_is_unsupported(tmp_path: Path, monkeypatch):
    files = generate_synthetic_dataset(tmp_path / "source", count=12, size=96)

    def unsupported_link(*_args):
        raise OSError(errno.EXDEV, "cross-device link")

    monkeypatch.setattr(data.os, "link", unsupported_link)
    with pytest.raises(PipelineError) as error:
        prepare_dataset(files, tmp_path / "prepared", _config())

    assert error.value.code == "ARTIFACT_PUBLISH_UNSUPPORTED"
    assert not list((tmp_path / "prepared" / "images").glob(".normalizing-*.png"))


def test_prepare_propagates_normalized_artifact_collision(tmp_path: Path):
    files = generate_synthetic_dataset(tmp_path / "source", count=12, size=96)
    output = tmp_path / "prepared"
    first = prepare_dataset(files, output, _config(), dataset_id="collision")
    Path((first["train"] + first["validation"])[0]["path"]).write_bytes(b"corrupted artifact")

    with pytest.raises(PipelineError) as error:
        prepare_dataset(files, output, _config(), dataset_id="collision")

    assert error.value.code == "OUTPUT_IMAGE_COLLISION"


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
