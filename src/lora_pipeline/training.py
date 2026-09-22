"""LoRA training backends with durable, compatible checkpoints.

The ``tiny`` backend is deliberately small enough for offline CPU tests.  It is
still a diffusion noise-prediction loop: a frozen image encoder/denoiser is
conditioned by caption features and only the low-rank update is optimised.
The ``sd15`` backend uses the same control flow with Diffusers and PEFT.
"""

from __future__ import annotations

import gc
import hashlib
import importlib.metadata
import json
import logging
import os
import random
import re
import shutil
import tempfile
import time
from pathlib import Path
from typing import Callable

from PIL import Image, ImageOps

from .common import PipelineError, digest_json, load_manifest, sha256_file, write_manifest
from .config import TrainConfig

Progress = Callable[[dict], None]
_CODE_VERSION = "training-v1"


def _loading_progress(progress: Progress | None, phase: str, config: TrainConfig) -> None:
    """Report a bounded setup phase without inventing numerical progress."""
    if progress:
        progress({"phase": phase, "current": 0, "total": config.max_steps})


def _safe_cuda_empty_cache(torch) -> None:
    """Best-effort CUDA cleanup must not replace the error that caused it.

    CUDA teardown can itself fail after a driver/device failure. A cleanup error
    is useful operationally, but must never turn a classified training failure
    into a generic worker exception.
    """
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception as exc:  # cleanup is never allowed to mask the primary error
        logging.getLogger("lora_pipeline").warning(
            "CUDA cache cleanup failed (exception_type=%s)", type(exc).__name__
        )


def _safe_training_cleanup(torch, model) -> None:
    """Release transient training state without allowing teardown to raise."""
    if model is not None:
        try:
            del model
        except Exception as exc:  # pragma: no cover - defensive for third-party model objects
            logging.getLogger("lora_pipeline").warning(
                "Model cleanup failed (exception_type=%s)", type(exc).__name__
            )
    try:
        gc.collect()
    except Exception as exc:  # pragma: no cover - gc failures are platform-specific
        logging.getLogger("lora_pipeline").warning(
            "Garbage collection cleanup failed (exception_type=%s)", type(exc).__name__
        )
    _safe_cuda_empty_cache(torch)


def _reset_cuda_memory_stats(torch, device) -> bool:
    """Initialize optional CUDA accounting without making telemetry a training dependency."""
    try:
        torch.cuda.reset_peak_memory_stats(device)
    except Exception as exc:
        logging.getLogger("lora_pipeline").warning(
            "CUDA memory telemetry initialization failed (exception_type=%s)", type(exc).__name__
        )
        return False
    return True


def _cuda_peak_memory_stats(torch, device) -> dict[str, int] | None:
    """Return optional CUDA peak-memory statistics without interrupting training.

    These counters are observability only.  Some drivers reject the accounting
    calls despite successfully running ordinary CUDA operations on the same
    device, so a failure here must not be treated as a device failure.
    """
    try:
        return {
            "gpu_memory_allocated": int(torch.cuda.max_memory_allocated(device)),
            "gpu_memory_reserved": int(torch.cuda.max_memory_reserved(device)),
        }
    except Exception as exc:
        logging.getLogger("lora_pipeline").warning(
            "CUDA memory telemetry collection failed (exception_type=%s)", type(exc).__name__
        )
        return None


def _memory_telemetry_unavailable_progress(
    progress: Progress | None, config: TrainConfig, exception_type: str | None = None
) -> None:
    """Tell clients that training continues without optional GPU-memory metrics."""
    if not progress:
        return
    payload: dict[str, int | str | bool] = {
        "phase": "cuda_memory_telemetry_unavailable",
        "current": 0,
        "total": config.max_steps,
        "gpu_memory_telemetry_available": False,
    }
    if exception_type:
        payload["telemetry_exception_type"] = exception_type
    progress(payload)


def _torch():
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F

        return torch, nn, F
    except ImportError as exc:  # pragma: no cover - environment error
        raise PipelineError("DEPENDENCY_MISSING", "PyTorch is required for training") from exc


def _safe_device(config: TrainConfig):
    torch, _, _ = _torch()
    if config.device.startswith("cuda"):
        if not torch.cuda.is_available():
            raise PipelineError("GPU_UNAVAILABLE", "CUDA was requested but is unavailable", retryable=True)
        return torch.device(config.device)
    return torch.device("cpu")


def _seed_everything(seed: int) -> None:
    torch, _, _ = _torch()
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _rss_bytes() -> int | None:
    """Portable-enough process memory diagnostic; absence is represented as null."""
    try:
        import psutil

        return int(psutil.Process().memory_info().rss)
    except Exception:  # pragma: no cover - optional monitoring dependency/platform
        return None


def _caption_features(caption: str, width: int = 16):
    """Stable local caption embedding for the test backend, without a network model."""
    torch, _, _ = _torch()
    digest = hashlib.sha256(caption.encode("utf-8")).digest()
    values = [(digest[index % len(digest)] / 127.5) - 1.0 for index in range(width)]
    return torch.tensor(values, dtype=torch.float32)


def _load_image(path: str | Path, config: TrainConfig, *, sample_id: str, epoch: int):
    torch, _, _ = _torch()
    try:
        with Image.open(path) as source:
            image = ImageOps.exif_transpose(source).convert("RGB")
            width, height = image.size
            target = config.resolution
            scale = max(target / width, target / height)
            resized = image.resize(
                (max(target, round(width * scale)), max(target, round(height * scale))),
                Image.Resampling.LANCZOS,
            )
            extra_x, extra_y = resized.width - target, resized.height - target
            if config.random_crop and (extra_x or extra_y):
                seed = int(hashlib.sha256(f"{config.seed}:{sample_id}:{epoch}".encode()).hexdigest()[:16], 16)
                rng = random.Random(seed)
                left, top = rng.randint(0, extra_x), rng.randint(0, extra_y)
            else:
                left, top = extra_x // 2, extra_y // 2
            image = resized.crop((left, top, left + target, top + target))
            if config.horizontal_flip:
                flip = (
                    int(
                        hashlib.sha256(f"flip:{config.seed}:{sample_id}:{epoch}".encode()).hexdigest()[:2], 16
                    )
                    % 2
                )
                if flip:
                    image = ImageOps.mirror(image)
            raw = bytearray(image.tobytes())
    except (OSError, ValueError) as exc:
        raise PipelineError(
            "IMAGE_DECODE_FAILED", f"Cannot decode training image {path}", details={"path": str(path)}
        ) from exc
    tensor = torch.frombuffer(raw, dtype=torch.uint8).clone().reshape(config.resolution, config.resolution, 3)
    return tensor.permute(2, 0, 1).float().div_(127.5).sub_(1.0)


def _entries(manifest: dict, split: str, *, require_caption: bool | None = None) -> list[dict]:
    # The data module writes either top-level split arrays or a nested splits map.
    entries = manifest.get(split) or manifest.get("splits", {}).get(split) or []
    if not entries:
        raise PipelineError("DATASET_EMPTY", f"The input manifest has no {split} entries")
    need_caption = split == "train" if require_caption is None else require_caption
    required = ("id", "path", "caption") if need_caption else ("id", "path")
    for entry in entries:
        if not all(key in entry for key in required):
            raise PipelineError("INVALID_MANIFEST", f"Each {split} item needs {', '.join(required)}")
    return entries


def _manifest_digest(manifest: dict) -> str:
    return str(
        manifest.get("manifest_sha256") or manifest.get("training_manifest_sha256") or digest_json(manifest)
    )


def _checksum_error(
    message: str,
    *,
    path: Path,
    expected_sha256: str | None,
    actual_sha256: str | None,
    read_error: OSError | None = None,
    **details,
) -> PipelineError:
    error_details = {
        **details,
        "path": str(path),
        "expected_sha256": expected_sha256,
        "actual_sha256": actual_sha256,
    }
    if read_error is not None:
        error_details["read_error"] = f"{type(read_error).__name__}: {read_error}"
    return PipelineError("CHECKSUM_MISMATCH", message, details=error_details)


def validate_input_integrity(input_manifest: dict) -> None:
    """Bind work to a frozen manifest and to the bytes of every source image."""
    expected_manifest = input_manifest.get("manifest_sha256") or input_manifest.get(
        "training_manifest_sha256"
    )
    manifest_path = input_manifest.get("manifest_path")
    if manifest_path:
        path = Path(manifest_path)
        try:
            actual_manifest = sha256_file(path)
        except OSError as exc:
            raise _checksum_error(
                "Frozen input manifest checksum does not match",
                path=path,
                expected_sha256=expected_manifest,
                actual_sha256=None,
                read_error=exc,
            ) from exc
        if expected_manifest and actual_manifest != expected_manifest:
            raise _checksum_error(
                "Frozen input manifest checksum does not match",
                path=path,
                expected_sha256=expected_manifest,
                actual_sha256=actual_manifest,
            )
        try:
            on_disk = json.loads(path.read_text())
            in_memory = {
                key: value
                for key, value in input_manifest.items()
                if key not in {"manifest_path", "manifest_sha256"}
            }
            if digest_json(on_disk) != digest_json(in_memory):
                raise PipelineError("CHECKSUM_MISMATCH", "In-memory manifest differs from its frozen file")
        except PipelineError:
            raise
        except OSError as exc:
            raise _checksum_error(
                "Frozen input manifest cannot be read",
                path=path,
                expected_sha256=expected_manifest,
                actual_sha256=actual_manifest,
                read_error=exc,
            ) from exc
        except Exception as exc:
            raise PipelineError(
                "CHECKSUM_MISMATCH",
                "Frozen input manifest cannot be parsed",
                details={"path": str(path), "read_error": f"{type(exc).__name__}: {exc}"},
            ) from exc
    elif expected_manifest:
        body = {
            key: value
            for key, value in input_manifest.items()
            if key not in {"manifest_path", "manifest_sha256"}
        }
        if digest_json(body) != expected_manifest:
            raise PipelineError("CHECKSUM_MISMATCH", "In-memory input manifest checksum does not match")
    for split in ("train", "validation"):
        for entry in _entries(input_manifest, split):
            expected, path = entry.get("sha256"), Path(entry["path"])
            try:
                actual = sha256_file(path)
            except OSError as exc:
                raise _checksum_error(
                    f"Frozen {split} image checksum does not match",
                    path=path,
                    expected_sha256=expected,
                    actual_sha256=None,
                    read_error=exc,
                    id=entry.get("id"),
                    split=split,
                ) from exc
            if not expected or actual != expected:
                raise _checksum_error(
                    f"Frozen {split} image checksum does not match",
                    path=path,
                    expected_sha256=expected,
                    actual_sha256=actual,
                    id=entry.get("id"),
                    split=split,
                )


def _compatibility_key(input_manifest: dict, config: TrainConfig) -> str:
    """Only inputs that affect numerical resume behaviour participate in this key."""
    frozen_config = config.snapshot()
    # Checkpoint cadence and download policy do not affect model numerics; every
    # remaining value is part of the numerical resume contract, including the
    # original max_steps and gradient-checkpointing choice.
    for operational in ("checkpoint_every", "local_files_only"):
        frozen_config.pop(operational, None)
    versions = {}
    for package in ("torch", "diffusers", "peft", "transformers", "accelerate"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:  # pragma: no cover - dependency error later
            versions[package] = "missing"
    return digest_json(
        {
            "code_version": _CODE_VERSION,
            "input_manifest_sha256": _manifest_digest(input_manifest),
            "config": frozen_config,
            "library_versions": versions,
        }
    )


def _tiny_model(config: TrainConfig, device) -> dict:
    """Create a random, offline Diffusers graph with the SD training topology.

    It is intentionally not a toy MLP: images pass through an ``AutoencoderKL``
    and a PEFT-instrumented ``UNet2DConditionModel`` receives noisy latents plus
    frozen text-conditioning vectors.  This gives CPU tests the same LoRA state,
    scheduler and denoising-loss semantics as SD 1.5 without downloading weights.
    """
    try:
        from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
        from peft import LoraConfig
    except ImportError as exc:  # pragma: no cover - setup failure
        raise PipelineError("DEPENDENCY_MISSING", "Tiny training needs diffusers and peft") from exc
    vae = AutoencoderKL(
        sample_size=config.resolution,
        in_channels=3,
        out_channels=3,
        down_block_types=("DownEncoderBlock2D",),
        up_block_types=("UpDecoderBlock2D",),
        block_out_channels=(16,),
        layers_per_block=1,
        latent_channels=3,
        norm_num_groups=4,
    ).to(device)
    unet = UNet2DConditionModel(
        sample_size=config.resolution,
        in_channels=3,
        out_channels=3,
        layers_per_block=1,
        block_out_channels=(16,),
        down_block_types=("CrossAttnDownBlock2D",),
        up_block_types=("CrossAttnUpBlock2D",),
        cross_attention_dim=16,
        attention_head_dim=4,
        norm_num_groups=4,
    ).to(device)
    vae.requires_grad_(False).eval()
    unet.requires_grad_(False)
    unet.add_adapter(
        LoraConfig(
            r=config.rank,
            lora_alpha=config.lora_alpha,
            target_modules=["to_q", "to_k", "to_v", "to_out.0"],
            bias="none",
        )
    )
    trainable = [parameter for parameter in unet.parameters() if parameter.requires_grad]
    if not trainable:
        raise PipelineError("LORA_SETUP_FAILED", "No tiny UNet LoRA parameters were enabled")
    return {
        "unet": unet,
        "vae": vae,
        "text_encoder": None,
        "tokenizer": None,
        "scheduler": DDPMScheduler(num_train_timesteps=100, beta_schedule="squaredcos_cap_v2"),
        "trainable": trainable,
        "base_fingerprint": "tiny-diffusers-random-v1",
        "base_revision": "tiny-diffusers-random-v1",
    }


def _base_identity(root: Path) -> str:
    """Fingerprint only files defining a Stable Diffusion pipeline snapshot."""
    root = Path(root).resolve()
    files: list[Path] = []
    if (root / "model_index.json").is_file():
        files.append(root / "model_index.json")
    for component in ("unet", "vae", "text_encoder", "tokenizer", "scheduler"):
        directory = root / component
        if directory.is_dir():
            files.extend(path for path in directory.rglob("*") if path.is_file())
    if not files:
        raise PipelineError("BASE_MODEL_UNAVAILABLE", "Pinned base snapshot has no SD pipeline files")
    return digest_json(
        {
            "kind": "sd15-pipeline",
            "files": [(str(path.relative_to(root)), sha256_file(path)) for path in sorted(files)],
        }
    )


def _sanitize_diagnostic_text(value: object) -> str:
    """Preserve diagnostics while removing common credential representations."""
    reason = str(value)
    reason = re.sub(r"(?i)(https?://)[^@/\s]+@", r"\1***@", reason)
    reason = re.sub(
        r"(?i)([?&](?:hf[_-]?token|token|access[_-]?token|authorization|api[_-]?key)=)[^&\s]+",
        r"\1***",
        reason,
    )
    reason = re.sub(
        r"(?i)(\bauthorization\s*[:=]\s*(?:bearer|basic)\s+)[^\s,;]+", r"\1***", reason
    )
    reason = re.sub(
        r"(?i)(\b(?:hf[_-]?token|token|access[_-]?token|api[_-]?key)\s*[:=]\s*)[^\s,;]+",
        r"\1***",
        reason,
    )
    return re.sub(r"\bhf_[A-Za-z0-9_-]+\b", "***", reason)


def _safe_failure_reason(exc: Exception) -> str:
    """Classify failures without ever copying untrusted exception text into details."""
    name = type(exc).__name__.lower()
    module = type(exc).__module__.lower()
    if isinstance(exc, PermissionError) or any(value in name for value in ("auth", "forbidden", "unauthorized")):
        category = "authorization"
    elif isinstance(exc, FileNotFoundError) or any(value in name for value in ("cache", "entrynotfound", "offline")):
        category = "cache_miss"
    elif isinstance(exc, (ConnectionError, TimeoutError)) or any(
        value in name or value in module for value in ("connection", "timeout", "network", "proxy")
    ):
        category = "network"
    elif isinstance(exc, OSError):
        category = "io"
    else:
        category = "unknown"
    return f"{type(exc).__name__}: {category}"


def _safe_diagnostic_value(value: object) -> object:
    return _sanitize_diagnostic_text(value) if isinstance(value, (str, Path)) else value


def _base_model_details(config: TrainConfig, **details: object) -> dict[str, object]:
    return {
        "model_name": _safe_diagnostic_value(config.model_name),
        "revision": _safe_diagnostic_value(config.revision),
        "local_files_only": config.local_files_only,
        "cache": _safe_diagnostic_value(os.getenv("HF_HOME")),
        **{key: _safe_diagnostic_value(value) for key, value in details.items()},
    }


def _resolve_sd15_base(config: TrainConfig) -> tuple[Path, str, str]:
    """Resolve one immutable repository snapshot before loading any component."""
    candidate = Path(config.model_name)
    if candidate.is_dir():
        root = candidate.resolve()
        fingerprint = _base_identity(root)
        return root, f"local:{fingerprint}", fingerprint
    try:
        from huggingface_hub import snapshot_download

        root = Path(
            snapshot_download(
                repo_id=config.model_name, revision=config.revision, local_files_only=config.local_files_only
            )
        ).resolve()
    except Exception as exc:
        raise PipelineError(
            "BASE_MODEL_UNAVAILABLE",
            "Could not resolve the SD 1.5 model snapshot",
            retryable=True,
            details=_base_model_details(config, reason=_safe_failure_reason(exc)),
        ) from exc
    revision = root.name
    if len(revision) < 7:
        raise PipelineError(
            "BASE_MODEL_UNAVAILABLE",
            "Model snapshot did not resolve to an immutable revision",
            details=_base_model_details(config, resolved_root=str(root), resolved_revision=revision),
        )
    return root, revision, _base_identity(root)


def _load_sd15(config: TrainConfig, device, progress: Progress | None = None):
    try:
        from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
        from peft import LoraConfig
        from transformers import CLIPTextModel, CLIPTokenizer
    except ImportError as exc:  # pragma: no cover - dependency setup is tested by integration
        raise PipelineError(
            "DEPENDENCY_MISSING", "SD 1.5 training needs diffusers, peft and transformers"
        ) from exc
    _loading_progress(progress, "sd15_snapshot_resolving", config)
    root, resolved_revision, fingerprint = _resolve_sd15_base(config)
    _loading_progress(progress, "sd15_snapshot_resolved", config)
    kwargs = {"local_files_only": True}
    try:
        _loading_progress(progress, "sd15_tokenizer_loading", config)
        tokenizer = CLIPTokenizer.from_pretrained(root, subfolder="tokenizer", **kwargs)
        _loading_progress(progress, "sd15_text_encoder_loading", config)
        text_encoder = CLIPTextModel.from_pretrained(root, subfolder="text_encoder", **kwargs).to(device)
        _loading_progress(progress, "sd15_vae_loading", config)
        vae = AutoencoderKL.from_pretrained(root, subfolder="vae", **kwargs).to(device)
        _loading_progress(progress, "sd15_unet_loading", config)
        unet = UNet2DConditionModel.from_pretrained(root, subfolder="unet", **kwargs).to(device)
        _loading_progress(progress, "sd15_scheduler_loading", config)
        scheduler = DDPMScheduler.from_pretrained(root, subfolder="scheduler", **kwargs)
    except PipelineError:
        raise
    except Exception as exc:
        raise PipelineError(
            "BASE_MODEL_UNAVAILABLE",
            "Could not load SD 1.5 model components from the resolved snapshot",
            retryable=True,
            details=_base_model_details(
                config,
                resolved_root=str(root),
                resolved_revision=resolved_revision,
                reason=_safe_failure_reason(exc),
            ),
        ) from exc
    vae.requires_grad_(False).eval()
    text_encoder.requires_grad_(False).eval()
    unet.requires_grad_(False)
    if config.gradient_checkpointing:
        unet.enable_gradient_checkpointing()
    _loading_progress(progress, "sd15_adapter_setup", config)
    try:
        unet.add_adapter(
            LoraConfig(
                r=config.rank,
                lora_alpha=config.lora_alpha,
                target_modules=["to_q", "to_k", "to_v", "to_out.0"],
                bias="none",
            )
        )
    except PipelineError:
        raise
    except Exception as exc:
        raise PipelineError("LORA_SETUP_FAILED", "Could not configure SD 1.5 LoRA adapters") from exc
    trainable = [p for p in unet.parameters() if p.requires_grad]
    if not trainable:
        raise PipelineError("LORA_SETUP_FAILED", "No UNet LoRA parameters were enabled")
    _loading_progress(progress, "sd15_adapter_ready", config)
    return {
        "unet": unet,
        "vae": vae,
        "text_encoder": text_encoder,
        "tokenizer": tokenizer,
        "scheduler": scheduler,
        "trainable": trainable,
        "base_fingerprint": fingerprint,
        "base_revision": resolved_revision,
    }


def _save_adapter(model, backend: str, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    if backend == "tiny":
        try:
            from peft import get_peft_model_state_dict
            from safetensors.torch import save_file

            path = output_dir / "adapter.safetensors"
            save_file(
                {
                    key: value.detach().contiguous().cpu()
                    for key, value in get_peft_model_state_dict(model).items()
                },
                str(path),
            )
            return path
        except Exception as exc:  # pragma: no cover
            raise PipelineError("ADAPTER_EXPORT_FAILED", "Could not write tiny LoRA adapter") from exc
    try:
        from diffusers import StableDiffusionPipeline
        from diffusers.utils import convert_state_dict_to_diffusers
        from peft import get_peft_model_state_dict

        layers = convert_state_dict_to_diffusers(get_peft_model_state_dict(model))
        StableDiffusionPipeline.save_lora_weights(
            str(output_dir),
            unet_lora_layers=layers,
            weight_name="adapter.safetensors",
            safe_serialization=True,
        )
    except Exception as exc:  # pragma: no cover - exercised with GPU integration
        raise PipelineError(
            "ADAPTER_EXPORT_FAILED", "Could not write a loadable Diffusers LoRA adapter"
        ) from exc
    return output_dir / "adapter.safetensors"


def _load_adapter(model, backend: str, path: Path) -> None:
    torch, _, _ = _torch()
    if backend == "tiny":
        try:
            from safetensors.torch import load_file
            from peft import set_peft_model_state_dict

            set_peft_model_state_dict(model, load_file(str(path)))
            return
        except Exception as exc:
            raise PipelineError("CHECKPOINT_CORRUPT", "Tiny adapter is unreadable") from exc
    # PEFT accepts its own checkpoint state keys; checkpoints use this only internally.
    try:
        from diffusers import StableDiffusionPipeline
        from diffusers.utils import convert_unet_state_dict_to_peft
        from peft import set_peft_model_state_dict

        state, _ = StableDiffusionPipeline.lora_state_dict(str(path.parent), weight_name=path.name)
        set_peft_model_state_dict(model, convert_unet_state_dict_to_peft(state))
    except Exception as exc:  # pragma: no cover
        raise PipelineError("CHECKPOINT_CORRUPT", "SD 1.5 adapter is unreadable") from exc


def _atomic_checkpoint(
    root: Path,
    *,
    step: int,
    sample_cursor: int,
    model,
    optimizer,
    scheduler,
    scaler,
    config: TrainConfig,
    compatibility_key: str,
    base_fingerprint: str,
) -> Path:
    torch, _, _ = _torch()
    root.mkdir(parents=True, exist_ok=True)
    final = root / f"step-{step:08d}"
    temporary = Path(tempfile.mkdtemp(prefix=f".checkpoint-{step}-", dir=root))
    try:
        adapter = _save_adapter(model, config.backend, temporary)
        state_path = temporary / "state.pt"
        rng_path = temporary / "rng.pt"
        torch.save(
            {
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict(),
                "global_step": step,
                "sample_cursor": sample_cursor,
            },
            state_path,
        )
        torch.save(
            {
                "python": random.getstate(),
                "torch": torch.get_rng_state(),
                "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            },
            rng_path,
        )
        payload = {
            "kind": "lora-checkpoint",
            "global_step": step,
            "compatibility_key": compatibility_key,
            "adapter": adapter.name,
            "adapter_sha256": sha256_file(adapter),
            "state": state_path.name,
            "state_sha256": sha256_file(state_path),
            "rng": rng_path.name,
            "rng_sha256": sha256_file(rng_path),
            "config": config.snapshot(),
            "base_fingerprint": base_fingerprint,
            "transform_version": "resize-crop-flip-v1",
        }
        write_manifest(temporary / "checkpoint.json", payload)
        if final.exists():
            shutil.rmtree(final)
        os.replace(temporary, final)
        return final / "checkpoint.json"
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def _restore_checkpoint(
    resume_from: Path,
    model,
    optimizer,
    scheduler,
    scaler,
    config: TrainConfig,
    compatibility_key: str,
    base_fingerprint: str,
) -> tuple[int, int]:
    torch, _, _ = _torch()
    manifest_path = resume_from / "checkpoint.json" if resume_from.is_dir() else resume_from
    try:
        checkpoint = load_manifest(manifest_path)
        root = manifest_path.parent
        if (
            checkpoint.get("kind") != "lora-checkpoint"
            or checkpoint.get("compatibility_key") != compatibility_key
            or checkpoint.get("base_fingerprint") != base_fingerprint
        ):
            raise PipelineError(
                "CHECKPOINT_INCOMPATIBLE", "Checkpoint inputs or training configuration differ"
            )
        for label in ("adapter", "state", "rng"):
            file_path = root / checkpoint[label]
            if not file_path.is_file() or sha256_file(file_path) != checkpoint[f"{label}_sha256"]:
                raise PipelineError("CHECKPOINT_CORRUPT", f"Checkpoint {label} checksum is invalid")
        _load_adapter(model, config.backend, root / checkpoint["adapter"])
        state = torch.load(root / checkpoint["state"], map_location="cpu", weights_only=False)
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        scaler.load_state_dict(state.get("scaler", {}))
        rng = torch.load(root / checkpoint["rng"], map_location="cpu", weights_only=False)
        random.setstate(rng["python"])
        torch.set_rng_state(rng["torch"])
        if torch.cuda.is_available() and rng.get("cuda") is not None:
            torch.cuda.set_rng_state_all(rng["cuda"])
        return int(state["global_step"]), int(
            state.get(
                "sample_cursor", state["global_step"] * config.gradient_accumulation_steps * config.batch_size
            )
        )
    except PipelineError:
        raise
    except Exception as exc:
        raise PipelineError("CHECKPOINT_CORRUPT", "Checkpoint cannot be restored") from exc


def _batch(entries: list[dict], start: int, size: int, config: TrainConfig):
    epoch = start // len(entries)
    result = [entries[(start + offset) % len(entries)] for offset in range(size)]
    images = [_load_image(value["path"], config, sample_id=str(value["id"]), epoch=epoch) for value in result]
    return result, images


def train(
    input_manifest: dict,
    output_dir: Path,
    config: TrainConfig,
    progress: Progress | None = None,
    resume_from: Path | None = None,
    stop_after_step: int | None = None,
) -> dict:
    """Train one adapter and persist a result manifest.

    ``stop_after_step`` is intentionally a successful, checkpointed interruption
    point used by the smoke-test CLI; resuming keeps ``max_steps`` unchanged.
    """
    if progress:
        progress(
            {
                "phase": "training_started",
                "current": 0,
                "total": config.max_steps,
                "global_step": 0,
            }
        )
    torch, _, F = _torch()
    output_dir = Path(output_dir)
    entries = _entries(input_manifest, "train")
    if progress:
        progress({"phase": "input_validation", "current": 0, "total": config.max_steps})
    validate_input_integrity(input_manifest)
    _loading_progress(progress, "device_resolving", config)
    device = _safe_device(config)
    _loading_progress(progress, "device_resolved", config)
    if stop_after_step is not None and not 1 <= stop_after_step <= config.max_steps:
        raise PipelineError("INVALID_STOP_STEP", "stop_after_step must be within max_steps")
    _loading_progress(progress, "random_seed_initializing", config)
    _seed_everything(config.seed)
    _loading_progress(progress, "random_seed_initialized", config)
    compatibility_key = _compatibility_key(input_manifest, config)
    start_time = time.monotonic()
    cuda_memory_telemetry_available = False
    if device.type == "cuda":
        _loading_progress(progress, "cuda_memory_initializing", config)
        cuda_memory_telemetry_available = _reset_cuda_memory_stats(torch, device)
        if cuda_memory_telemetry_available:
            _loading_progress(progress, "cuda_memory_initialized", config)
        else:
            _memory_telemetry_unavailable_progress(progress, config)
    model = None
    try:
        _loading_progress(progress, "model_loading", config)
        if config.backend == "tiny":
            _loading_progress(progress, "tiny_components_loading", config)
            components = _tiny_model(config, device)
        else:
            components = _load_sd15(config, device, progress=progress)
        model, trainable = components["unet"], components["trainable"]
        _loading_progress(progress, "model_loaded", config)
        _loading_progress(progress, "optimizer_initializing", config)
        try:
            optimizer = torch.optim.AdamW(trainable, lr=config.learning_rate)
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
            scaler = torch.amp.GradScaler(
                device.type, enabled=device.type == "cuda" and config.precision == "fp16"
            )
        except Exception as exc:
            raise PipelineError("OPTIMIZER_SETUP_FAILED", "Could not initialize the training optimizer") from exc
        _loading_progress(progress, "optimizer_initialized", config)
        if resume_from:
            if progress:
                progress({"phase": "checkpoint_restoring", "current": 0, "total": config.max_steps})
            global_step, sample_cursor = _restore_checkpoint(
                Path(resume_from),
                model,
                optimizer,
                scheduler,
                scaler,
                config,
                compatibility_key,
                components["base_fingerprint"],
            )
            checkpoint_path: Path | None = (
                Path(resume_from) / "checkpoint.json" if Path(resume_from).is_dir() else Path(resume_from)
            )
            if progress:
                progress(
                    {
                        "phase": "checkpoint_restored",
                        "current": global_step,
                        "total": config.max_steps,
                        "global_step": global_step,
                    }
                )
        else:
            global_step, sample_cursor = 0, 0
            checkpoint_path = None
        invocation_start_cursor = sample_cursor
        optimizer.zero_grad(set_to_none=True)
        while global_step < config.max_steps:
            loss_sum = 0.0
            for micro in range(config.gradient_accumulation_steps):
                batch_entries, images = _batch(entries, sample_cursor, config.batch_size, config)
                sample_cursor += config.batch_size
                pixels = torch.stack(images).to(device)
                vae, encoder, tokenizer, noise_scheduler = (
                    components["vae"],
                    components["text_encoder"],
                    components["tokenizer"],
                    components["scheduler"],
                )
                with torch.no_grad():
                    if config.backend == "tiny":
                        encoded = (
                            torch.stack([_caption_features(item["caption"]) for item in batch_entries])
                            .to(device)
                            .unsqueeze(1)
                        )
                    else:
                        latents = vae.encode(pixels).latent_dist.sample() * vae.config.scaling_factor
                        tokens = tokenizer(
                            [item["caption"] for item in batch_entries],
                            padding="max_length",
                            truncation=True,
                            max_length=tokenizer.model_max_length,
                            return_tensors="pt",
                        ).input_ids.to(device)
                        encoded = encoder(tokens)[0]
                    if config.backend == "tiny":
                        latents = vae.encode(pixels).latent_dist.sample() * vae.config.scaling_factor
                noise = torch.randn_like(latents)
                timesteps = torch.randint(
                    0, noise_scheduler.config.num_train_timesteps, (len(batch_entries),), device=device
                ).long()
                noisy = noise_scheduler.add_noise(latents, noise, timesteps)
                autocast_dtype = torch.float16 if config.precision == "fp16" else torch.bfloat16
                with torch.autocast(
                    device_type=device.type,
                    dtype=autocast_dtype,
                    enabled=device.type == "cuda" and config.precision != "fp32",
                ):
                    prediction = model(noisy, timesteps, encoded).sample
                    target = (
                        noise
                        if noise_scheduler.config.prediction_type == "epsilon"
                        else noise_scheduler.get_velocity(latents, noise, timesteps)
                    )
                    loss = F.mse_loss(prediction.float(), target.float())
                if not torch.isfinite(loss):
                    raise PipelineError("NONFINITE_LOSS", "Training loss is not finite")
                scaler.scale(loss / config.gradient_accumulation_steps).backward()
                loss_sum += float(loss.detach().cpu())
            scaler.unscale_(optimizer)
            if any(
                parameter.grad is not None and not torch.isfinite(parameter.grad).all()
                for parameter in trainable
            ):
                raise PipelineError("NONFINITE_GRADIENT", "Training gradients are not finite")
            torch.nn.utils.clip_grad_norm_(trainable, config.max_grad_norm)
            scale_before = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            if scaler.is_enabled() and scaler.get_scale() < scale_before:
                optimizer.zero_grad(set_to_none=True)
                raise PipelineError(
                    "AMP_OVERFLOW",
                    "Automatic mixed precision overflow skipped an optimizer step",
                    retryable=True,
                )
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1
            checkpoint_seconds = None
            if (
                global_step % config.checkpoint_every == 0
                or global_step == config.max_steps
                or global_step == stop_after_step
            ):
                if progress:
                    progress(
                        {
                            "phase": "checkpoint_saving",
                            "current": global_step,
                            "total": config.max_steps,
                            "global_step": global_step,
                        }
                    )
                checkpoint_started = time.monotonic()
                checkpoint_path = _atomic_checkpoint(
                    output_dir / "checkpoints",
                    step=global_step,
                    sample_cursor=sample_cursor,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=scaler,
                    config=config,
                    compatibility_key=compatibility_key,
                    base_fingerprint=components["base_fingerprint"],
                )
                checkpoint_seconds = time.monotonic() - checkpoint_started
                if progress:
                    progress(
                        {
                            "phase": "checkpoint_saved",
                            "current": global_step,
                            "total": config.max_steps,
                            "global_step": global_step,
                        }
                    )
            payload = {
                "phase": "training",
                "current": global_step,
                "total": config.max_steps,
                "global_step": global_step,
                "loss": loss_sum / config.gradient_accumulation_steps,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "elapsed_seconds": time.monotonic() - start_time,
                "samples_processed": sample_cursor,
                "samples_processed_this_run": sample_cursor - invocation_start_cursor,
                "samples_per_second": (sample_cursor - invocation_start_cursor)
                / max(time.monotonic() - start_time, 1e-9),
            }
            if checkpoint_path is not None:
                payload["checkpoint_path"] = str(checkpoint_path.resolve())
            if checkpoint_seconds is not None:
                payload["checkpoint_seconds"] = checkpoint_seconds
            rss = _rss_bytes()
            if rss is not None:
                payload["cpu_rss_bytes"] = rss
            if device.type == "cuda" and cuda_memory_telemetry_available:
                memory_stats = _cuda_peak_memory_stats(torch, device)
                if memory_stats is None:
                    cuda_memory_telemetry_available = False
                    _memory_telemetry_unavailable_progress(progress, config)
                    payload["gpu_memory_telemetry_available"] = False
                else:
                    payload.update(memory_stats)
                    payload["gpu_memory_telemetry_available"] = True
            if progress:
                progress(payload)
            if global_step == stop_after_step:
                break
        if progress:
            progress(
                {
                    "phase": "adapter_saving",
                    "current": global_step,
                    "total": config.max_steps,
                    "global_step": global_step,
                }
            )
        adapter = _save_adapter(model, config.backend, output_dir / "adapter")
        if progress:
            progress(
                {
                    "phase": "adapter_saved",
                    "current": global_step,
                    "total": config.max_steps,
                    "global_step": global_step,
                }
            )
        state = "STOPPED" if global_step < config.max_steps else "COMPLETED"
        trainable_parameter_count = sum(parameter.numel() for parameter in trainable)
        total_parameter_count = sum(parameter.numel() for parameter in model.parameters())
        result = write_manifest(
            output_dir / "training-result.json",
            {
                "kind": "training-result",
                "state": state,
                "adapter_path": str(adapter.resolve()),
                "adapter_sha256": sha256_file(adapter),
                "checkpoint_path": str(checkpoint_path.resolve()) if checkpoint_path else None,
                "config": config.snapshot(),
                "compatibility_key": compatibility_key,
                "global_step": global_step,
                "test_only": config.backend == "tiny",
                "base_model": config.model_name,
                "base_revision": components["base_revision"],
                "input_manifest_sha256": _manifest_digest(input_manifest),
                "base_fingerprint": components["base_fingerprint"],
                "transform_version": "resize-crop-flip-v1",
                "elapsed_seconds": time.monotonic() - start_time,
                "trainable_parameter_count": trainable_parameter_count,
                "frozen_parameter_count": total_parameter_count - trainable_parameter_count,
                "total_parameter_count": total_parameter_count,
            },
        )
        if progress:
            progress(
                {
                    "phase": "training_completed",
                    "current": global_step,
                    "total": config.max_steps,
                    "global_step": global_step,
                    "state": state,
                    "samples_processed": sample_cursor,
                    "samples_processed_this_run": sample_cursor - invocation_start_cursor,
                    "samples_per_second": (sample_cursor - invocation_start_cursor)
                    / max(time.monotonic() - start_time, 1e-9),
                }
            )
        return result
    except RuntimeError as exc:
        if "out of memory" in str(exc).lower():
            _safe_cuda_empty_cache(torch)
            raise PipelineError(
                "GPU_OUT_OF_MEMORY", "Training ran out of GPU memory", retryable=False
            ) from exc
        raise
    finally:
        _safe_training_cleanup(torch, model)
