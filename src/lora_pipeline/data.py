"""Deterministic, local image preparation for LoRA jobs.

The prepared manifest references images copied into a job-owned directory.  This
keeps later training independent from client supplied paths and gives every
normalisation step a reproducible input artifact.
"""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import random
import re
from pathlib import Path
from typing import Any

import imagehash
from PIL import Image, ImageDraw, ImageFilter, ImageOps, ImageStat, UnidentifiedImageError

from .common import PipelineError, sha256_file, write_manifest
from .config import DataConfig


_ALLOWED_FORMATS = {"JPEG", "PNG", "WEBP"}
_CONTROL_CHARACTERS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


class _Groups:
    def __init__(self, size: int):
        self.parent = list(range(size))

    def find(self, item: int) -> int:
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, left: int, right: int) -> None:
        left, right = self.find(left), self.find(right)
        if left != right:
            self.parent[max(left, right)] = min(left, right)


def _error(code: str, message: str, **details: Any) -> PipelineError:
    return PipelineError(code, message, details=details or None)


def _normalized_pixel_hash(image: Image.Image) -> str:
    """Hash pixels plus dimensions, after EXIF/RGB/alpha normalisation."""
    hasher = hashlib.sha256()
    hasher.update(f"{image.width}x{image.height}:RGB".encode())
    hasher.update(image.tobytes())
    return hasher.hexdigest()


def _normalise_image(path: Path, config: DataConfig) -> tuple[Image.Image, str, list[str]]:
    try:
        with Image.open(path) as source:
            if source.format not in _ALLOWED_FORMATS:
                raise _error("UNSUPPORTED_IMAGE_FORMAT", "Only JPEG, PNG, and static WebP images are supported", path=str(path))
            if getattr(source, "is_animated", False) or getattr(source, "n_frames", 1) != 1:
                raise _error("ANIMATED_IMAGE", "Animated images are not supported", path=str(path))
            if source.width * source.height > config.max_pixels:
                raise _error("IMAGE_TOO_LARGE", "Image exceeds the configured pixel limit", path=str(path))
            # Header validation comes before decoding to avoid allocating a
            # decompression-bomb sized pixel buffer.
            source.load()
            image = ImageOps.exif_transpose(source)
            if min(image.width, image.height) < config.min_side:
                raise _error("IMAGE_TOO_SMALL", "Image is smaller than the configured minimum side", path=str(path))
            if image.mode in {"RGBA", "LA"} or (image.mode == "P" and "transparency" in image.info):
                alpha = image.convert("RGBA")
                background = Image.new("RGBA", alpha.size, "white")
                image = Image.alpha_composite(background, alpha).convert("RGB")
            else:
                image = image.convert("RGB")
    except PipelineError:
        raise
    except (UnidentifiedImageError, Image.DecompressionBombError, OSError, ValueError, SyntaxError) as exc:
        raise _error("INVALID_IMAGE", "Image cannot be decoded", path=str(path), reason=str(exc)) from exc

    warnings: list[str] = []
    luminance = image.convert("L")
    mean, deviation = ImageStat.Stat(luminance).mean[0], ImageStat.Stat(luminance).stddev[0]
    edge_deviation = ImageStat.Stat(luminance.filter(ImageFilter.FIND_EDGES)).stddev[0]
    if mean < 15 or mean > 240:
        warnings.append("extreme_brightness")
    if deviation < 12:
        warnings.append("low_contrast")
    if edge_deviation < 8:
        warnings.append("possibly_blurry")
    return image, _normalized_pixel_hash(image), warnings


def _validate_file_record(record: dict[str, Any]) -> tuple[str, str, Path]:
    if not isinstance(record, dict):
        raise _error("INVALID_FILE_RECORD", "Each dataset file must be an object")
    try:
        file_id, name, raw_path = str(record["id"]), str(record["name"]), record["path"]
    except KeyError as exc:
        raise _error("INVALID_FILE_RECORD", "Dataset file requires id, name, and path") from exc
    if not file_id or not name or not isinstance(raw_path, (str, Path)):
        raise _error("INVALID_FILE_RECORD", "Dataset file id, name, and path must be non-empty")
    if _CONTROL_CHARACTERS.search(file_id) or _CONTROL_CHARACTERS.search(name):
        raise _error("INVALID_FILE_RECORD", "Dataset file id or name contains a control character")
    path = Path(raw_path).expanduser().resolve()
    if not path.is_file():
        raise _error("FILE_NOT_FOUND", "Dataset file path does not exist", file_id=file_id)
    return file_id, name, path


def _make_groups(records: list[dict[str, Any]], distance: int) -> list[list[int]]:
    union_find = _Groups(len(records))
    for left in range(len(records)):
        for right in range(left + 1, len(records)):
            if records[left]["phash"] - records[right]["phash"] <= distance:
                union_find.union(left, right)
    grouped: dict[int, list[int]] = {}
    for index in range(len(records)):
        grouped.setdefault(union_find.find(index), []).append(index)
    return sorted(grouped.values(), key=lambda group: min(records[index]["id"] for index in group))


def _split_groups(groups: list[list[int]], records: list[dict[str, Any]], config: DataConfig, dataset_id: str) -> set[int]:
    total = len(records)
    min_groups = config.min_groups_per_split
    if len(groups) < min_groups * 2:
        raise _error("DATASET_UNSUITABLE", "Not enough independent image groups for train and validation", groups=len(groups))
    min_validation = config.min_validation_images
    max_validation = total - config.min_train_images
    if min_validation > max_validation:
        raise _error("DATASET_UNSUITABLE", "Image count cannot satisfy train and validation minima", valid_images=total)

    # A reproducible set of greedy selections gives good, stable balance while
    # still coping with groups of unequal size.  We retain the best feasible
    # candidate rather than relying on one arbitrary shuffle.
    ordering = list(range(len(groups)))
    seed_material = f"{config.seed}:{dataset_id}".encode()
    rng = random.Random(int.from_bytes(hashlib.sha256(seed_material).digest()[:8], "big"))
    rng.shuffle(ordering)
    candidates: list[set[int]] = []
    for offset in range(len(ordering)):
        selected: set[int] = set()
        count = 0
        for group_index in ordering[offset:] + ordering[:offset]:
            group_size = len(groups[group_index])
            if count + group_size > max_validation:
                continue
            if count < min_validation or (
                count < round(total * config.validation_fraction)
                or abs((count + group_size) - total * config.validation_fraction) < abs(count - total * config.validation_fraction)
            ):
                selected.add(group_index)
                count += group_size
        candidates.append(selected)

    feasible: list[tuple[float, tuple[int, ...], set[int]]] = []
    target = total * config.validation_fraction
    for candidate in candidates:
        image_count = sum(len(groups[index]) for index in candidate)
        if (min_validation <= image_count <= max_validation and min_groups <= len(candidate) <= len(groups) - min_groups):
            feasible.append((abs(image_count - target), tuple(sorted(candidate)), candidate))
    if not feasible:
        raise _error(
            "DATASET_UNSUITABLE",
            "Near-duplicate groups cannot satisfy split minima without leakage",
            groups=len(groups),
            min_validation=min_validation,
            min_train=config.min_train_images,
        )
    return min(feasible, key=lambda item: (item[0], item[1]))[2]


def prepare_dataset(
    files: list[dict], output_dir: Path, config: DataConfig | None = None, dataset_id: str = "local"
) -> dict:
    """Validate, normalise, de-duplicate, group and deterministically split images.

    Invalid individual files are recorded in the manifest.  A data set is only
    admitted when the remaining unique images satisfy the configured limits.
    """
    config = config or DataConfig()
    if not config.min_images <= len(files) <= config.max_images:
        raise _error("DATASET_IMAGE_COUNT", "Uploaded image count is outside the configured range", count=len(files))
    output_dir = Path(output_dir).resolve()
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    total_bytes = 0
    prepared: list[dict[str, Any]] = []
    rejected: list[dict[str, str]] = []
    duplicates: list[dict[str, str]] = []
    exact_hashes: dict[str, str] = {}
    seen_file_ids: set[str] = set()

    for ordinal, raw_record in enumerate(files):
        try:
            file_id, name, source_path = _validate_file_record(raw_record)
            if file_id in seen_file_ids:
                raise _error("DUPLICATE_FILE_ID", "Dataset file IDs must be unique", file_id=file_id)
            seen_file_ids.add(file_id)
            size = source_path.stat().st_size
            total_bytes += size
            if size > config.max_file_bytes:
                raise _error("FILE_TOO_LARGE", "Image exceeds the configured file size limit", file_id=file_id)
            if total_bytes > config.max_total_bytes:
                raise _error("DATASET_TOO_LARGE", "Dataset exceeds the configured total size limit")
            supplied_sha = raw_record.get("sha256")
            source_sha = sha256_file(source_path)
            if supplied_sha is not None and supplied_sha != source_sha:
                raise _error("CHECKSUM_MISMATCH", "Uploaded file checksum does not match", file_id=file_id)
            image, pixel_hash, warnings = _normalise_image(source_path, config)
            if pixel_hash in exact_hashes:
                duplicates.append({"id": file_id, "duplicate_of": exact_hashes[pixel_hash], "reason": "exact_normalized_pixels"})
                continue
            exact_hashes[pixel_hash] = file_id
            normalized_path = images_dir / f"{ordinal:04d}-{pixel_hash[:16]}.png"
            image.save(normalized_path, "PNG", optimize=True)
            entry: dict[str, Any] = {
                "id": file_id,
                "name": name,
                "path": str(normalized_path.resolve()),
                "sha256": sha256_file(normalized_path),
                "pixel_sha256": pixel_hash,
                "phash": imagehash.phash(image),
                "warnings": warnings,
            }
            if "caption" in raw_record and raw_record["caption"] is not None:
                entry["caption"] = raw_record["caption"]
            prepared.append(entry)
        except PipelineError as exc:
            if exc.code == "DATASET_TOO_LARGE":
                # Aggregate capacity is a dataset-level admission failure,
                # not a recoverable per-file data quality warning.
                raise
            rejected.append({"id": str(raw_record.get("id", ordinal)) if isinstance(raw_record, dict) else str(ordinal), "code": exc.code, "message": exc.message})

    if not config.min_images <= len(prepared) <= config.max_images:
        raise _error("DATASET_TOO_SMALL", "Valid unique images are below the configured minimum", valid_images=len(prepared), rejected=len(rejected), duplicates=len(duplicates))
    groups = _make_groups(prepared, config.phash_distance)
    validation_groups = _split_groups(groups, prepared, config, dataset_id)
    group_by_record: dict[int, str] = {}
    for group_number, members in enumerate(groups, start=1):
        for record_index in members:
            group_by_record[record_index] = f"group-{group_number:04d}"

    train: list[dict[str, Any]] = []
    validation: list[dict[str, Any]] = []
    group_indexes = {index: group_index for group_index, group in enumerate(groups) for index in group}
    for index, record in enumerate(prepared):
        public_entry = {key: value for key, value in record.items() if key not in {"pixel_sha256", "phash"}}
        public_entry["group_id"] = group_by_record[index]
        (validation if group_indexes[index] in validation_groups else train).append(public_entry)
    payload = {
        "schema_version": 1,
        "preprocessing_version": "image-normalize-rgb-v1",
        "grouping_version": f"phash64-hamming-{config.phash_distance}-v1",
        "dataset_id": dataset_id,
        "config": asdict(config),
        "train": train,
        "validation": validation,
        "statistics": {
            "submitted": len(files), "accepted_unique": len(prepared), "train_images": len(train),
            "validation_images": len(validation), "groups": len(groups), "rejected": len(rejected),
            "exact_duplicates": len(duplicates), "total_source_bytes": total_bytes,
        },
        "rejected": rejected,
        "duplicates": duplicates,
    }
    return write_manifest(output_dir / "prepared.json", payload)


def generate_synthetic_dataset(output_dir: Path, count: int = 120, seed: int = 42, size: int = 256) -> list[dict]:
    """Create deterministic, varied, captioned local images for smoke tests."""
    if count < 1 or size < 64:
        raise ValueError("count must be positive and size must be at least 64")
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    palettes = [(202, 64, 76), (47, 113, 181), (58, 148, 96), (227, 162, 53), (128, 82, 165), (47, 160, 172)]
    shapes = ["circle", "square", "triangle", "star", "stripe", "ring"]
    records: list[dict] = []
    for index in range(count):
        shape = shapes[index % len(shapes)]
        color = palettes[(index // len(shapes)) % len(palettes)]
        background = tuple(238 - ((index * factor) % 35) for factor in (3, 5, 7))
        image = Image.new("RGB", (size, size), background)
        draw = ImageDraw.Draw(image)
        margin = min((size - 8) // 2, 24 + (index * 17) % max(1, size // 5))
        box = (margin, margin, size - margin, size - margin)
        if shape == "circle":
            draw.ellipse(box, fill=color)
        elif shape == "square":
            draw.rectangle(box, fill=color)
        elif shape == "triangle":
            draw.polygon([(size // 2, margin), (size - margin, size - margin), (margin, size - margin)], fill=color)
        elif shape == "star":
            points = [(size // 2, margin), (size * 3 // 5, size * 2 // 5), (size - margin, size * 2 // 5), (size * 13 // 20, size * 3 // 5), (size * 3 // 4, size - margin), (size // 2, size * 7 // 10), (size // 4, size - margin), (size * 7 // 20, size * 3 // 5), (margin, size * 2 // 5), (size * 2 // 5, size * 2 // 5)]
            draw.polygon(points, fill=color)
        elif shape == "stripe":
            for position in range(-size, size * 2, max(10, size // 9)):
                draw.line((position, 0, position - size, size), fill=color, width=max(4, size // 18))
        else:
            draw.ellipse(box, fill=color)
            inset = max(2, (size - margin * 2) // 4)
            inner = (margin + inset, margin + inset, size - margin - inset, size - margin - inset)
            draw.ellipse(inner, fill=background)
        # A small deterministic marker keeps otherwise similar compositions in
        # separate perceptual groups without relying on filenames.
        x, y = rng.randrange(size // 4, size * 3 // 4), rng.randrange(size // 4, size * 3 // 4)
        draw.rectangle((x, y, x + max(3, size // 32), y + max(3, size // 32)), fill=(index * 31 % 255, index * 73 % 255, index * 127 % 255))
        path = output_dir / f"sample-{index:04d}.png"
        image.save(path, "PNG")
        records.append({"id": f"sample-{index:04d}", "name": path.name, "path": str(path), "caption": f"a {shape} in a colorful geometric composition"})
    return records
