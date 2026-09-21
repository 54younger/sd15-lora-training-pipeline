"""Format and normalization regressions independent of the generated dataset."""

from dataclasses import replace

import pytest
from PIL import Image, ImageOps

from lora_pipeline.common import PipelineError
from lora_pipeline.config import DataConfig
from lora_pipeline.data import _normalise_image, _normalized_pixel_hash


@pytest.mark.parametrize("fmt,extension", [("JPEG", "jpg"), ("PNG", "png"), ("WEBP", "webp")])
def test_supported_static_formats(tmp_path, fmt, extension):
    path = tmp_path / f"image.{extension}"
    Image.new("RGB", (256, 256), (120, 60, 30)).save(path, fmt)
    image, digest, warnings = _normalise_image(path, DataConfig())
    assert image.mode == "RGB" and image.size == (256, 256)
    assert len(digest) == 64
    assert "low_contrast" in warnings  # Uniform art is warned about, not rejected.


def test_animated_webp_is_not_a_static_training_image(tmp_path):
    path = tmp_path / "animation.webp"
    Image.new("RGB", (256, 256), "red").save(
        path,
        "WEBP",
        save_all=True,
        append_images=[Image.new("RGB", (256, 256), "blue")],
        duration=100,
        loop=0,
    )
    with pytest.raises(PipelineError) as error:
        _normalise_image(path, DataConfig())
    assert error.value.code == "ANIMATED_IMAGE"


def test_alpha_normalization_matches_explicit_white_composite(tmp_path):
    source = Image.new("RGBA", (256, 256), (255, 0, 0, 128))
    path = tmp_path / "transparent.png"
    source.save(path)
    image, digest, _ = _normalise_image(path, DataConfig())
    expected = Image.alpha_composite(Image.new("RGBA", source.size, "white"), source).convert("RGB")
    assert image.tobytes() == expected.tobytes()
    assert digest == _normalized_pixel_hash(expected)


def test_exif_orientation_applied_before_pixel_hash(tmp_path):
    path = tmp_path / "rotated.png"
    original = Image.new("RGB", (300, 256), "blue")
    original.paste("red", (0, 0, 150, 256))
    exif = Image.Exif()
    exif[274] = 6
    original.save(path, exif=exif)
    image, digest, _ = _normalise_image(path, DataConfig())
    with Image.open(path) as loaded:
        expected = ImageOps.exif_transpose(loaded).convert("RGB")
    assert image.size == (256, 300)
    assert digest == _normalized_pixel_hash(expected)


def test_dimension_limit_precedes_payload_decode(tmp_path, monkeypatch):
    path = tmp_path / "large.png"
    Image.new("RGB", (256, 256)).save(path)
    from PIL import PngImagePlugin

    def forbidden_load(*args, **kwargs):
        raise AssertionError("Payload must not be decoded after exceeding the header limit")

    monkeypatch.setattr(PngImagePlugin.PngImageFile, "load", forbidden_load)
    with pytest.raises(PipelineError) as error:
        _normalise_image(path, replace(DataConfig(), max_pixels=100))
    assert error.value.code == "IMAGE_TOO_LARGE"


def test_blip_runtime_failure_is_not_template_fallback(tmp_path, monkeypatch):
    import transformers
    from lora_pipeline.captions import caption_dataset

    class Processor:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            return cls()

    class BrokenModel:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            raise RuntimeError("Injected weight loading failure")

    monkeypatch.setattr(transformers, "BlipProcessor", Processor)
    monkeypatch.setattr(transformers, "BlipForConditionalGeneration", BrokenModel)
    prepared = {"train": [{"id": "x", "path": str(tmp_path / "unused.png")}], "validation": []}
    with pytest.raises(PipelineError) as error:
        caption_dataset(prepared, tmp_path / "output", mode="blip", local_files_only=True)
    assert error.value.code == "BLIP_CAPTION_FAILED"
    assert not (tmp_path / "output" / "training-input.json").exists()
