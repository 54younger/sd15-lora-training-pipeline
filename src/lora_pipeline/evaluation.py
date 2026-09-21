"""Adapter evaluation, calibrated gates, and paired A/B comparison."""

from __future__ import annotations

import gc
import html
import math
import time
from dataclasses import asdict
from pathlib import Path
from typing import Callable, Iterable

from .common import PipelineError, atomic_write, digest_json, sha256_file, write_manifest
from .config import EvalConfig, TrainConfig
from .training import (
    _caption_features,
    _resolve_sd15_base,
    _entries,
    _load_adapter,
    _safe_device,
    _seed_everything,
    _tiny_model,
    _torch,
    validate_input_integrity,
)

Progress = Callable[[dict], None]


def cosine_similarity(left: Iterable[float], right: Iterable[float]) -> float:
    """Numerically safe cosine similarity used for CLIP diagnostics and unit tests."""
    left_values, right_values = list(left), list(right)
    if len(left_values) != len(right_values) or not left_values:
        raise ValueError("Embeddings must be non-empty and have the same length")
    dot = sum(float(a) * float(b) for a, b in zip(left_values, right_values))
    left_norm = math.sqrt(sum(float(a) ** 2 for a in left_values))
    right_norm = math.sqrt(sum(float(b) ** 2 for b in right_values))
    return 0.0 if left_norm == 0 or right_norm == 0 else dot / (left_norm * right_norm)


def diversity_score(embeddings: list[list[float]]) -> float | None:
    if len(embeddings) < 2:
        return None
    pairs = [
        cosine_similarity(embeddings[first], embeddings[second])
        for first in range(len(embeddings))
        for second in range(first + 1, len(embeddings))
    ]
    return 1.0 - sum(pairs) / len(pairs)


def _report_html(title: str, report: dict) -> str:
    rows = "".join(
        f"<tr><th>{html.escape(str(key))}</th><td><pre>{html.escape(str(value))}</pre></td></tr>"
        for key, value in report.items()
        if key not in {"manifest_path", "manifest_sha256"}
    )
    pairs = "".join(
        "<figure><figcaption>{prompt} · seed {seed} · Δ {delta}</figcaption>"
        "<img src='{left}' alt='left image'><img src='{right}' alt='right image'></figure>".format(
            prompt=html.escape(str(item.get("prompt", ""))),
            seed=html.escape(str(item.get("seed", ""))),
            delta=html.escape(str(item.get("clip_score_delta", "n/a"))),
            left=html.escape(str(item.get("left_image", item.get("base_image", "")))),
            right=html.escape(str(item.get("right_image", item.get("adapter_image", "")))),
        )
        for item in report.get("paired_outputs", [])
    )
    return f"<!doctype html><html><head><meta charset='utf-8'><title>{html.escape(title)}</title><style>body{{font-family:system-ui;margin:2rem}}table{{border-collapse:collapse;width:100%}}th,td{{border:1px solid #ccc;padding:.5rem;text-align:left;vertical-align:top}}pre{{white-space:pre-wrap;margin:0}}figure{{display:inline-block;max-width:48%;margin:1rem}}img{{max-width:46%;margin:.25rem}}</style></head><body><h1>{html.escape(title)}</h1><table>{rows}</table>{pairs}</body></html>"


def _policy_verdict(metrics: dict, technical_pass: bool, policy: dict | None) -> tuple[str, list[str]]:
    """A policy cannot become passing merely because it happens to contain numbers."""
    if not technical_pass:
        return "FAIL", ["Technical adapter smoke test failed"]
    if not policy or not policy.get("version") or not policy.get("calibration_reference"):
        return "UNCALIBRATED", ["No versioned calibration reference is configured"]
    bounds = policy.get("bounds")
    if not isinstance(bounds, dict) or not bounds:
        return "UNCALIBRATED", ["The calibrated policy has no metric bounds"]
    required = {"clip_prompt_score", "heldout_similarity", "diversity", "max_train_similarity"}
    missing = sorted(required - set(bounds))
    if missing:
        return "UNCALIBRATED", [f"The calibrated policy is missing required bounds: {', '.join(missing)}"]
    failures: list[str] = []
    for name, bound in bounds.items():
        if not isinstance(bound, dict):
            raise PipelineError("QUALITY_POLICY_INVALID", f"Invalid bound object for {name}")
        if not {"min", "max"} & set(bound):
            raise PipelineError("QUALITY_POLICY_INVALID", f"Bound for {name} needs min and/or max")
        for limit in ("min", "max"):
            if limit in bound and (
                isinstance(bound[limit], bool)
                or not isinstance(bound[limit], (int, float))
                or not math.isfinite(float(bound[limit]))
            ):
                raise PipelineError("QUALITY_POLICY_INVALID", f"Bound {name}.{limit} must be a finite number")
        if "min" in bound and "max" in bound and float(bound["min"]) > float(bound["max"]):
            raise PipelineError("QUALITY_POLICY_INVALID", f"Bound {name} has min greater than max")
        value = metrics.get(name)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            failures.append(f"Required metric {name} is unavailable")
        elif "min" in bound and value < float(bound["min"]):
            failures.append(f"{name} is below its minimum")
        elif "max" in bound and value > float(bound["max"]):
            failures.append(f"{name} is above its maximum")
    return ("PASS" if not failures else "FAIL"), failures


def _train_config(training_result: dict) -> TrainConfig:
    try:
        raw = dict(training_result["config"])
        # A saved config is expected to reflect the original device. Evaluation's
        # device is not relevant to the tiny technical check.
        return TrainConfig(**raw)
    except Exception as exc:
        raise PipelineError(
            "INVALID_TRAINING_RESULT", "Training result does not contain a valid frozen configuration"
        ) from exc


def _tiny_technical_smoke(training_result: dict) -> bool:
    config = _train_config(training_result)
    if config.backend != "tiny":
        raise PipelineError("BACKEND_MISMATCH", "Tiny smoke received a non-tiny adapter")
    config.device, config.precision = "cpu", "fp32"
    _seed_everything(config.seed)
    torch, _, _ = _torch()
    components = _tiny_model(config, torch.device("cpu"))
    model = components["unet"]
    try:
        adapter = Path(training_result["adapter_path"])
        if not adapter.is_file() or sha256_file(adapter) != training_result["adapter_sha256"]:
            raise PipelineError("CHECKSUM_MISMATCH", "Adapter checksum does not match its training result")
        _load_adapter(model, "tiny", adapter)
        with torch.no_grad():
            pixels = torch.zeros(1, 3, config.resolution, config.resolution)
            latents = (
                components["vae"].encode(pixels).latent_dist.sample()
                * components["vae"].config.scaling_factor
            )
            output = model(
                latents,
                torch.ones(1, dtype=torch.long),
                _caption_features("technical smoke").unsqueeze(0).unsqueeze(1),
            ).sample
        return bool(torch.isfinite(output).all())
    finally:
        del model
        gc.collect()


def _sd15_images(
    training_result: dict,
    config: EvalConfig,
    output_dir: Path,
    *,
    with_adapter: bool,
    progress: Progress | None = None,
    phase: str = "generation",
) -> list[tuple[str, int, Path]]:
    """Generate a deterministic image suite. Called only by the actual backend."""
    try:
        import torch
        from diffusers import StableDiffusionPipeline
    except ImportError as exc:  # pragma: no cover
        raise PipelineError("DEPENDENCY_MISSING", "Diffusers is required for SD 1.5 evaluation") from exc
    train_config = _train_config(training_result)
    frozen_revision = training_result.get("base_revision")
    if not frozen_revision:
        raise PipelineError("BASE_REVISION_UNPINNED", "Training result lacks an immutable base revision")
    load_config = TrainConfig(
        **{
            **asdict(train_config),
            "device": config.device,
            "precision": "fp32" if config.device == "cpu" else train_config.precision,
            "revision": None if str(frozen_revision).startswith("local:") else frozen_revision,
        }
    )
    device = _safe_device(load_config)
    dtype = torch.float16 if device.type == "cuda" and train_config.precision == "fp16" else torch.float32
    base_root, resolved_revision, base_fingerprint = _resolve_sd15_base(load_config)
    if base_fingerprint != training_result.get("base_fingerprint") or resolved_revision != frozen_revision:
        raise PipelineError("BASE_SNAPSHOT_MISMATCH", "Current base model content differs from training")
    kwargs = {"local_files_only": True, "torch_dtype": dtype}
    try:
        pipe = StableDiffusionPipeline.from_pretrained(base_root, **kwargs).to(device)
        pipe.set_progress_bar_config(disable=True)
        if with_adapter:
            adapter = Path(training_result["adapter_path"])
            if sha256_file(adapter) != training_result["adapter_sha256"]:
                raise PipelineError(
                    "CHECKSUM_MISMATCH", "Adapter checksum does not match its training result"
                )
            pipe.load_lora_weights(str(adapter.parent), weight_name=adapter.name)
    except PipelineError:
        raise
    except Exception as exc:  # pragma: no cover - GPU integration
        raise PipelineError(
            "EVALUATION_MODEL_UNAVAILABLE", "Could not load SD 1.5 evaluation pipeline", retryable=True
        ) from exc
    saved: list[tuple[str, int, Path]] = []
    try:
        total = len(config.prompts) * len(config.seeds)
        for prompt_index, prompt in enumerate(config.prompts):
            for seed in config.seeds:
                generator = torch.Generator(device=device).manual_seed(seed)
                image = pipe(
                    prompt,
                    num_inference_steps=config.inference_steps,
                    guidance_scale=config.guidance_scale,
                    generator=generator,
                    height=train_config.resolution,
                    width=train_config.resolution,
                ).images[0]
                path = (
                    output_dir / ("adapter" if with_adapter else "base") / f"p{prompt_index:02d}-s{seed}.png"
                )
                path.parent.mkdir(parents=True, exist_ok=True)
                image.save(path)
                saved.append((prompt, seed, path))
                if progress:
                    progress({"phase": phase, "generated_images": len(saved), "generation_total": total})
    finally:
        del pipe
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return saved


def _clip_metrics(
    generated: list[tuple[str, int, Path]], heldout: list[dict], training: list[dict], config: EvalConfig
) -> dict:
    """CLIP is used as a diagnostic proxy, never as a claim of human quality."""
    try:
        import torch
        from PIL import Image
        from transformers import CLIPModel, CLIPProcessor
    except ImportError as exc:  # pragma: no cover
        raise PipelineError("DEPENDENCY_MISSING", "transformers is required for CLIP evaluation") from exc
    device = "cuda" if config.device.startswith("cuda") and torch.cuda.is_available() else "cpu"
    kwargs = {"revision": config.clip_revision, "local_files_only": config.local_files_only}
    try:
        processor = CLIPProcessor.from_pretrained(config.clip_model, **kwargs)
        model = CLIPModel.from_pretrained(config.clip_model, **kwargs).to(device).eval()
    except Exception as exc:  # pragma: no cover - integration only
        raise PipelineError(
            "CLIP_MODEL_UNAVAILABLE", "Could not load the configured CLIP scorer", retryable=True
        ) from exc
    try:
        image_embeddings: list[list[float]] = []
        text_scores: list[float] = []
        pair_scores: list[dict] = []
        with torch.no_grad():
            for prompt, seed, path in generated:
                with Image.open(path) as image:
                    inputs = processor(
                        text=[prompt], images=[image.convert("RGB")], return_tensors="pt", padding=True
                    ).to(device)
                    outputs = model(**inputs)
                    embed = outputs.image_embeds[0].detach().float().cpu().tolist()
                    image_embeddings.append(embed)
                    score = max(
                        100.0
                        * cosine_similarity(
                            outputs.image_embeds[0].detach().float().cpu().tolist(),
                            outputs.text_embeds[0].detach().float().cpu().tolist(),
                        ),
                        0.0,
                    )
                    text_scores.append(score)
                    pair_scores.append(
                        {
                            "prompt": prompt,
                            "seed": seed,
                            "image_path": str(path.resolve()),
                            "clip_score": score,
                        }
                    )

            def embed_references(entries: list[dict], clip_model) -> list[list[float]]:
                reference_embeddings: list[list[float]] = []
                for item in entries:
                    with Image.open(item["path"]) as image:
                        inputs = processor(images=[image.convert("RGB")], return_tensors="pt").to(device)
                        reference_embeddings.append(
                            clip_model.get_image_features(**inputs)[0].detach().float().cpu().tolist()
                        )
                return reference_embeddings

            heldout_embeddings = embed_references(heldout, model)
            training_embeddings = embed_references(training, model)
        heldout_max = [
            max((cosine_similarity(item, reference) for reference in heldout_embeddings), default=0.0)
            for item in image_embeddings
        ]
        training_max = [
            max((cosine_similarity(item, reference) for reference in training_embeddings), default=0.0)
            for item in image_embeddings
        ]
        return {
            "clip_prompt_score": sum(text_scores) / len(text_scores) if text_scores else None,
            "heldout_similarity": sum(heldout_max) / len(heldout_max) if heldout_max else None,
            "diversity": diversity_score(image_embeddings),
            "max_train_similarity": max(training_max) if training_max else None,
            "generated_count": len(generated),
            "prompt_pairs": pair_scores,
            "clip_model": config.clip_model,
            "clip_revision": getattr(model.config, "_commit_hash", None) or config.clip_revision,
        }
    finally:
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _sd15_technical_smoke(training_result: dict, config: EvalConfig, output_dir: Path) -> bool:
    # The first adapter image is both an adapter reload test and an inference smoke test.
    return bool(
        _sd15_images(
            training_result,
            EvalConfig(
                prompts=["a technical adapter smoke test"],
                seeds=[17],
                inference_steps=1,
                guidance_scale=config.guidance_scale,
                clip_model=config.clip_model,
                clip_revision=config.clip_revision,
                device=config.device,
                local_files_only=config.local_files_only,
            ),
            output_dir / "technical-smoke",
            with_adapter=True,
        )
    )


def evaluate(
    training_result: dict,
    input_manifest: dict,
    output_dir: Path,
    config: EvalConfig,
    progress: Progress | None = None,
) -> dict:
    """Evaluate one completed adapter and write both JSON and static HTML reports."""
    output_dir = Path(output_dir)
    validate_input_integrity(input_manifest)
    adapter_path = Path(training_result.get("adapter_path", ""))
    if training_result.get("state") != "COMPLETED":
        raise PipelineError("TRAINING_NOT_COMPLETE", "Only completed training results can be evaluated")
    if not adapter_path.is_file():
        raise PipelineError("ARTIFACT_MISSING", "Training adapter does not exist")
    if sha256_file(adapter_path) != training_result.get("adapter_sha256"):
        raise PipelineError("CHECKSUM_MISMATCH", "Training adapter checksum does not match")
    expected_input = training_result.get("input_manifest_sha256")
    actual_input = str(
        input_manifest.get("manifest_sha256")
        or input_manifest.get("training_manifest_sha256")
        or digest_json(input_manifest)
    )
    if expected_input and expected_input != actual_input:
        raise PipelineError("INPUT_INCOMPATIBLE", "Evaluation manifest differs from the training input")
    start = time.monotonic()
    if progress:
        progress({"phase": "evaluation_started"})
    backend = _train_config(training_result).backend
    metrics: dict
    artifact_paths: dict[str, str] = {}
    if backend == "tiny":
        technical_pass = _tiny_technical_smoke(training_result)
        metrics = {
            "clip_prompt_score": None,
            "heldout_similarity": None,
            "diversity": None,
            "max_train_similarity": None,
            "generated_count": 0,
            "semantic_metrics_status": "unavailable_for_test_backend",
        }
    elif backend == "sd15":
        technical_pass = _sd15_technical_smoke(training_result, config, output_dir)
        heldout, training = _entries(input_manifest, "validation"), _entries(input_manifest, "train")
        # Every scored LoRA suite has an identical base-model control: prompt,
        # seed, inference steps and guidance stay fixed, leaving the adapter as
        # the only changed variable.
        baseline = _sd15_images(
            training_result,
            config,
            output_dir / "generation",
            with_adapter=False,
            progress=progress,
            phase="baseline_generation",
        )
        generated = _sd15_images(
            training_result,
            config,
            output_dir / "generation",
            with_adapter=True,
            progress=progress,
            phase="adapter_generation",
        )
        metrics = _clip_metrics(generated, heldout, training, config)
        metrics["baseline"] = _clip_metrics(baseline, heldout, training, config)
        metrics["paired_count"] = len(generated)
        pair_scores = {(item["prompt"], item["seed"]): item["clip_score"] for item in metrics["prompt_pairs"]}
        baseline_scores = {
            (item["prompt"], item["seed"]): item["clip_score"] for item in metrics["baseline"]["prompt_pairs"]
        }
        paired_outputs = [
            {
                "prompt": prompt,
                "seed": seed,
                "base_image": str(base_path.relative_to(output_dir)),
                "adapter_image": str(adapter_path.relative_to(output_dir)),
                "clip_score_delta": pair_scores[(prompt, seed)] - baseline_scores[(prompt, seed)],
            }
            for (prompt, seed, base_path), (_, _, adapter_path) in zip(baseline, generated, strict=True)
        ]
        artifact_paths["generated_dir"] = str((output_dir / "generation" / "adapter").resolve())
        artifact_paths["baseline_dir"] = str((output_dir / "generation" / "base").resolve())
    else:  # configuration protects this, but result manifests can be untrusted disk input
        raise PipelineError("UNSUPPORTED_BACKEND", f"Unsupported evaluation backend {backend}")
    quality_status, failures = _policy_verdict(metrics, technical_pass, config.quality_policy)
    if training_result.get("test_only") and quality_status == "PASS":
        quality_status = "UNCALIBRATED"
        failures.append("A test-only backend cannot be published as quality-qualified")
    result = {
        "kind": "evaluation-report",
        "technical_pass": technical_pass,
        "quality_status": quality_status,
        "quality_failures": failures,
        "metrics": metrics,
        "test_only": bool(training_result.get("test_only")),
        "adapter_path": str(adapter_path.resolve()),
        "adapter_sha256": training_result["adapter_sha256"],
        "input_manifest_sha256": actual_input,
        "base_model": training_result.get("base_model"),
        "base_revision": training_result.get("base_revision"),
        "config": asdict(config),
        "artifact_paths": artifact_paths,
        "paired_outputs": paired_outputs if backend == "sd15" else [],
        "elapsed_seconds": time.monotonic() - start,
    }
    report = write_manifest(output_dir / "evaluation.json", result)
    html_path = output_dir / "evaluation.html"
    atomic_write(html_path, _report_html("LoRA evaluation report", report).encode("utf-8"))
    # Re-publish the JSON manifest including the separately checksummed HTML artifact.
    final = write_manifest(
        output_dir / "evaluation.json",
        {**result, "report_path": str(html_path.resolve()), "report_sha256": sha256_file(html_path)},
    )
    if progress:
        progress({"phase": "evaluation_completed", "quality_status": final["quality_status"]})
    return final


def compare_adapters(
    left_training_result: dict,
    right_training_result: dict,
    input_manifest: dict,
    output_dir: Path,
    config: EvalConfig,
    progress: Progress | None = None,
) -> dict:
    """Run an apples-to-apples paired comparison for two compatible adapters."""
    validate_input_integrity(input_manifest)
    if left_training_result.get("base_model") != right_training_result.get(
        "base_model"
    ) or left_training_result.get("base_revision") != right_training_result.get("base_revision"):
        raise PipelineError("AB_INCOMPATIBLE", "Adapters use different base model revisions")
    if left_training_result.get("base_fingerprint") != right_training_result.get("base_fingerprint"):
        raise PipelineError("AB_INCOMPATIBLE", "Adapters use different frozen base model snapshots")
    if left_training_result.get("input_manifest_sha256") != right_training_result.get(
        "input_manifest_sha256"
    ):
        raise PipelineError("AB_INCOMPATIBLE", "Adapters use different frozen training inputs")
    left_config, right_config = (
        left_training_result.get("config", {}),
        right_training_result.get("config", {}),
    )
    for field in ("resolution", "precision"):
        if left_config.get(field) != right_config.get(field):
            raise PipelineError("AB_INCOMPATIBLE", f"Adapters use different training {field}")
    output_dir = Path(output_dir)
    left_backend, right_backend = (
        _train_config(left_training_result).backend,
        _train_config(right_training_result).backend,
    )
    if left_backend != right_backend:
        raise PipelineError("AB_INCOMPATIBLE", "Adapters use different backends")
    if left_backend == "tiny":
        left_ok, right_ok = (
            _tiny_technical_smoke(left_training_result),
            _tiny_technical_smoke(right_training_result),
        )
        payload = {
            "kind": "ab-comparison",
            "test_only": True,
            "left_technical_pass": left_ok,
            "right_technical_pass": right_ok,
            "metrics": {"status": "semantic_metrics_unavailable_for_test_backend"},
            "paired_conditions": {"prompts": config.prompts, "seeds": config.seeds},
        }
    else:
        left_images = _sd15_images(left_training_result, config, output_dir / "left", with_adapter=True)
        right_images = _sd15_images(right_training_result, config, output_dir / "right", with_adapter=True)
        # Paths are deterministic per prompt/seed, and therefore are explicitly paired.
        heldout, training = _entries(input_manifest, "validation"), _entries(input_manifest, "train")
        left_metrics = _clip_metrics(left_images, heldout, training, config)
        right_metrics = _clip_metrics(right_images, heldout, training, config)
        delta = {
            name: right_metrics[name] - left_metrics[name]
            for name in left_metrics
            if isinstance(left_metrics.get(name), (float, int))
            and isinstance(right_metrics.get(name), (float, int))
        }
        left_pair_scores = {
            (item["prompt"], item["seed"]): item["clip_score"] for item in left_metrics["prompt_pairs"]
        }
        right_pair_scores = {
            (item["prompt"], item["seed"]): item["clip_score"] for item in right_metrics["prompt_pairs"]
        }
        pairs = [
            {
                "prompt": prompt,
                "seed": seed,
                "left_image": str(left_path.relative_to(output_dir)),
                "right_image": str(right_path.relative_to(output_dir)),
                "clip_score_delta": right_pair_scores[(prompt, seed)] - left_pair_scores[(prompt, seed)],
            }
            for (prompt, seed, left_path), (_, _, right_path) in zip(left_images, right_images, strict=True)
        ]
        payload = {
            "kind": "ab-comparison",
            "test_only": False,
            "paired_conditions": {
                "prompts": config.prompts,
                "seeds": config.seeds,
                "inference_steps": config.inference_steps,
                "guidance_scale": config.guidance_scale,
            },
            "left_images": [str(value[2].resolve()) for value in left_images],
            "right_images": [str(value[2].resolve()) for value in right_images],
            "left_metrics": left_metrics,
            "right_metrics": right_metrics,
            "right_minus_left": delta,
            "paired_outputs": pairs,
        }
    report = write_manifest(output_dir / "comparison.json", payload)
    html_path = output_dir / "comparison.html"
    atomic_write(html_path, _report_html("LoRA A/B comparison", report).encode("utf-8"))
    return write_manifest(
        output_dir / "comparison.json",
        {**payload, "report_path": str(html_path.resolve()), "report_sha256": sha256_file(html_path)},
    )
