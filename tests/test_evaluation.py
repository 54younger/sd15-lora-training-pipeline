from __future__ import annotations

from copy import deepcopy
from pathlib import Path

from PIL import Image

from lora_pipeline.common import PipelineError, sha256_file, write_manifest
from lora_pipeline.config import EvalConfig, TrainConfig
from lora_pipeline.evaluation import (
    _policy_verdict,
    compare_adapters,
    cosine_similarity,
    diversity_score,
    evaluate,
)
from lora_pipeline.training import train


def _completed_result(tmp_path: Path):
    tmp_path.mkdir(parents=True, exist_ok=True)
    paths = []
    for index in range(2):
        path = tmp_path / f"image-{index}.png"
        Image.new("RGB", (16, 16), (index * 80, 50, 120)).save(path)
        paths.append(
            {
                "id": str(index),
                "path": str(path),
                "sha256": sha256_file(path),
                "caption": f"test image {index}",
            }
        )
    manifest = write_manifest(tmp_path / "training-input.json", {"train": paths, "validation": paths})
    config = TrainConfig(
        backend="tiny",
        resolution=16,
        device="cpu",
        precision="fp32",
        rank=2,
        lora_alpha=2,
        max_steps=1,
        checkpoint_every=1,
        gradient_accumulation_steps=1,
    )
    return train(manifest, tmp_path / "training", config), manifest


def test_metric_math_and_calibrated_gate_rules():
    assert cosine_similarity([1, 0], [1, 0]) == 1.0
    assert round(diversity_score([[1, 0], [0, 1]]) or 0, 6) == 1.0
    assert _policy_verdict({"score": 0.9}, True, None)[0] == "UNCALIBRATED"
    policy = {
        "version": "v1",
        "calibration_reference": "fixture",
        "bounds": {
            "clip_prompt_score": {"min": 0.8},
            "heldout_similarity": {"min": 0.8},
            "diversity": {"min": 0.2},
            "max_train_similarity": {"max": 0.98},
        },
    }
    passing = {
        "clip_prompt_score": 0.9,
        "heldout_similarity": 0.9,
        "diversity": 0.4,
        "max_train_similarity": 0.5,
    }
    assert _policy_verdict(passing, True, policy)[0] == "PASS"
    verdict, failures = _policy_verdict({**passing, "clip_prompt_score": None}, True, policy)
    assert verdict == "FAIL" and failures
    try:
        _policy_verdict(passing, True, {**policy, "bounds": {**policy["bounds"], "diversity": {}}})
    except PipelineError as error:
        assert error.code == "QUALITY_POLICY_INVALID"
    else:  # pragma: no cover
        raise AssertionError("policy without a threshold was accepted")


def test_tiny_evaluation_is_technical_only_and_never_quality_ready(tmp_path: Path):
    training_result, manifest = _completed_result(tmp_path)
    updates: list[dict] = []
    report = evaluate(
        training_result,
        manifest,
        tmp_path / "evaluation",
        EvalConfig(
            device="cpu", quality_policy={"version": "v1", "calibration_reference": "fixture", "bounds": {}}
        ),
        progress=updates.append,
    )

    assert report["technical_pass"] is True
    assert report["quality_status"] == "UNCALIBRATED"
    assert report["test_only"] is True
    assert report["metrics"]["semantic_metrics_status"] == "unavailable_for_test_backend"
    assert Path(report["report_path"]).is_file()
    assert updates[0]["phase"] == "evaluation_started"
    assert updates[-1]["phase"] == "evaluation_completed"


def test_tiny_ab_comparison_is_paired_but_never_claims_semantic_quality(tmp_path: Path):
    left, manifest = _completed_result(tmp_path / "left-input")
    right = train(manifest, tmp_path / "right-training", TrainConfig(**left["config"]))
    comparison = compare_adapters(left, right, manifest, tmp_path / "comparison", EvalConfig(device="cpu"))

    assert comparison["test_only"] is True
    assert comparison["metrics"]["status"] == "semantic_metrics_unavailable_for_test_backend"


def test_ab_comparison_rejects_mismatched_resolution_before_loading_weights(tmp_path: Path):
    left, manifest = _completed_result(tmp_path / "input")
    right = deepcopy(left)
    right["config"]["resolution"] = 32
    try:
        compare_adapters(left, right, manifest, tmp_path / "comparison", EvalConfig(device="cpu"))
    except PipelineError as error:
        assert error.code == "AB_INCOMPATIBLE"
    else:  # pragma: no cover
        raise AssertionError("incompatible A/B pair was accepted")
