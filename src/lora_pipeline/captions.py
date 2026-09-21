"""Caption construction with an offline template mode and explicit BLIP mode."""
from __future__ import annotations

from copy import deepcopy
import re
from pathlib import Path
from typing import Any

from .common import PipelineError, write_manifest


_CONTROL_CHARACTERS = re.compile(r"[\x00-\x1f\x7f]")


def _validate_text(value: object, field: str, *, maximum: int = 512) -> str:
    if not isinstance(value, str):
        raise PipelineError("INVALID_CAPTION", f"{field} must be text")
    text = value.strip()
    if not text or len(text) > maximum or _CONTROL_CHARACTERS.search(text):
        raise PipelineError("INVALID_CAPTION", f"{field} is empty, too long, or contains a control character")
    return text


def _template_content(entry: dict[str, Any]) -> str:
    """Return a safe, non-inferential fallback for offline captioning."""
    return "an image"


def _blip_contents(
    entries: list[dict[str, Any]], device: str, model_name: str, revision: str | None, local_files_only: bool
) -> tuple[dict[str, str], dict[str, Any]]:
    try:
        import torch
        from PIL import Image
        from transformers import BlipForConditionalGeneration, BlipProcessor
    except ImportError as exc:
        raise PipelineError("BLIP_UNAVAILABLE", "BLIP captioning dependencies are not installed") from exc
    try:
        processor = BlipProcessor.from_pretrained(model_name, revision=revision, local_files_only=local_files_only)
        model = BlipForConditionalGeneration.from_pretrained(model_name, revision=revision, local_files_only=local_files_only)
        model.to(device)
        model.eval()
        contents: dict[str, str] = {}
        for entry in entries:
            with Image.open(entry["path"]) as image:
                inputs = processor(images=image.convert("RGB"), return_tensors="pt")
            inputs = {key: value.to(device) for key, value in inputs.items()}
            with torch.no_grad():
                generated = model.generate(**inputs, max_new_tokens=64)
            contents[entry["id"]] = _validate_text(processor.decode(generated[0], skip_special_tokens=True), "BLIP caption")
        effective_revision = getattr(model.config, "_commit_hash", None) or revision
        return contents, {"source": "blip", "model": model_name, "revision": effective_revision, "device": device}
    except PipelineError:
        raise
    except Exception as exc:
        raise PipelineError("BLIP_CAPTION_FAILED", "BLIP caption generation failed", details={"reason": str(exc), "model": model_name}) from exc
    finally:
        if "model" in locals():
            del model
        try:
            import torch
            if str(device).startswith("cuda") and torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass


def caption_dataset(
    prepared: dict,
    output_dir: Path,
    mode: str = "template",
    trigger_token: str = "mystyle",
    device: str = "cpu",
    model_name: str = "Salesforce/blip-image-captioning-base",
    revision: str | None = None,
    local_files_only: bool = False,
) -> dict:
    """Write a frozen training-input manifest with final captions for train images."""
    if mode not in {"template", "blip"}:
        raise PipelineError("INVALID_CAPTION_MODE", "Caption mode must be template or blip")
    trigger_token = _validate_text(trigger_token, "trigger token", maximum=128)
    train = deepcopy(prepared.get("train", []))
    validation = deepcopy(prepared.get("validation", []))
    if not train or not isinstance(train, list) or not isinstance(validation, list):
        raise PipelineError("INVALID_PREPARED_MANIFEST", "Prepared manifest requires train and validation arrays")
    missing = [entry for entry in train if entry.get("caption") is None or entry.get("caption") == ""]
    generated: dict[str, str] = {}
    provenance: dict[str, Any] = {"mode": mode, "trigger_token": trigger_token}
    if missing and mode == "blip":
        generated, blip_provenance = _blip_contents(missing, device, model_name, revision, local_files_only)
        provenance.update(blip_provenance)
    elif missing:
        provenance.update({"source": "template"})
    else:
        provenance.update({"source": "user"})

    source_counts = {"user": 0, "template": 0, "blip": 0}
    for entry in train:
        if entry.get("caption") is not None and entry.get("caption") != "":
            content, source = _validate_text(entry["caption"], "user caption"), "user"
        elif mode == "blip":
            content, source = generated[entry["id"]], "blip"
        else:
            content, source = _template_content(entry), "template"
        final_caption = f"in {trigger_token} style, {content}"
        entry["caption"] = final_caption
        entry["caption_provenance"] = source
        source_counts[source] += 1

    payload = {
        "schema_version": 1,
        "dataset_id": prepared.get("dataset_id", "local"),
        "prepared_manifest_path": prepared.get("manifest_path"),
        "prepared_manifest_sha256": prepared.get("manifest_sha256"),
        "train": train,
        "validation": validation,
        "captioning": {**provenance, "counts": source_counts},
    }
    return write_manifest(Path(output_dir).resolve() / "training-input.json", payload)
