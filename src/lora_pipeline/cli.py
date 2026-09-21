"""Operator CLI. Heavy ML dependencies are imported only after device selection."""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import platform
import sys
from dataclasses import asdict
from pathlib import Path

from .common import PipelineError, atomic_write, canonical_json, load_manifest, write_manifest
from .config import EvalConfig, TrainConfig, settings_from_env


def _print(value):
    print(json.dumps(value, indent=2, default=str, allow_nan=False), flush=True)


@contextlib.contextmanager
def _device(device: str, selected_uuid: str | None = None):
    if not device.startswith("cuda"):
        yield
        return
    from .store import device_lock, discover_gpu_uuids

    settings = settings_from_env()
    available = discover_gpu_uuids()
    selected_uuid = selected_uuid or (settings.gpu_uuids or available or [None])[0]
    if not selected_uuid or selected_uuid not in available:
        raise PipelineError("GPU_UNAVAILABLE", "No permitted physical GPU detected by nvidia-smi")
    os.environ["CUDA_VISIBLE_DEVICES"] = selected_uuid
    with device_lock(settings, selected_uuid):
        yield


def preflight() -> dict:
    from importlib.metadata import version
    from .store import discover_gpu_uuids
    import torch

    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "dependencies": {p: version(p) for p in ("torch", "diffusers", "transformers", "peft", "accelerate")},
        "cuda_available": torch.cuda.is_available(),
        "cuda_runtime": torch.version.cuda,
        "physical_gpu_uuids": discover_gpu_uuids(),
        "devices": [
            {
                "index": i,
                "name": torch.cuda.get_device_name(i),
                "total_memory_bytes": torch.cuda.get_device_properties(i).total_memory,
            }
            for i in range(torch.cuda.device_count())
        ],
        "hf_cache": os.getenv("HF_HOME", "default Hugging Face cache"),
    }


def _smoke(args):
    from .data import generate_synthetic_dataset, prepare_dataset
    from .captions import caption_dataset

    out = args.output.resolve()
    if out.exists() and any(out.iterdir()):
        raise PipelineError("OUTPUT_EXISTS", "Smoke output must be new or empty")
    files = generate_synthetic_dataset(out / "images", count=120)
    prepared = prepare_dataset(files, out / "prepared")
    device = "cuda" if args.command == "gpu-smoke" else "cpu"
    with _device(device, getattr(args, "gpu_uuid", None)):
        from .training import train
        from .evaluation import evaluate

        inputs = caption_dataset(prepared, out / "input", trigger_token="geometricstyle")
        gpu = device == "cuda"
        config = TrainConfig(
            backend="sd15" if gpu else "tiny",
            device=device,
            precision="fp16" if gpu else "fp32",
            resolution=512 if gpu else 32,
            model_name=getattr(args, "model", TrainConfig().model_name),
            revision=getattr(args, "revision", None),
            max_steps=10 if gpu else 4,
            checkpoint_every=5 if gpu else 2,
            gradient_accumulation_steps=4 if gpu else 1,
            local_files_only=getattr(args, "local_files_only", False),
        )
        half = config.max_steps // 2
        first = train(inputs, out / "training", config, progress=_print, stop_after_step=half)
        resumed = train(
            inputs, out / "training", config, progress=_print, resume_from=Path(first["checkpoint_path"])
        )
        if resumed["global_step"] != config.max_steps:
            raise PipelineError("SMOKE_FAILED", "Resumed training did not reach expected optimizer step")
        if first["adapter_sha256"] == resumed["adapter_sha256"]:
            raise PipelineError("SMOKE_FAILED", "Adapter did not change after resumed optimization")
        evaluation = evaluate(
            resumed,
            inputs,
            out / "evaluation",
            EvalConfig(
                prompts=["a red circle on a white background", "a blue square on a yellow background"],
                seeds=[42],
                inference_steps=10 if gpu else 2,
                device=device,
                local_files_only=config.local_files_only,
            ),
            progress=_print,
        )
        if not evaluation["technical_pass"]:
            raise PipelineError("SMOKE_FAILED", "Saved adapter failed inference smoke test")
        result = write_manifest(
            out / "smoke-result.json",
            {
                "technical_pass": True,
                "quality_status": "UNCALIBRATED",
                "test_only": not gpu,
                "environment": preflight(),
                "config": asdict(config),
                "interrupted_step": half,
                "resumed_step": resumed["global_step"],
                "training_manifest": resumed["manifest_path"],
                "evaluation_manifest": evaluation["manifest_path"],
                "gpu_benchmarks": "not_measured"
                if not gpu
                else "see training/evaluation metrics; smoke workload only",
            },
        )
        _print(result)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    api = sub.add_parser("api")
    api.add_argument("--host", default="127.0.0.1")
    api.add_argument("--port", type=int, default=8000)
    worker = sub.add_parser("worker")
    worker.add_argument("--once", action="store_true")
    sub.add_parser("worker-health")
    sub.add_parser("preflight")
    generate = sub.add_parser("generate-data")
    generate.add_argument("--output", type=Path, required=True)
    generate.add_argument("--count", type=int, default=120)
    prepare = sub.add_parser("prepare")
    prepare.add_argument("--files", type=Path, required=True, help="Trusted local file-entry JSON list")
    prepare.add_argument("--output", type=Path, required=True)
    caption = sub.add_parser("caption")
    caption.add_argument("--prepared", type=Path, required=True)
    caption.add_argument("--output", type=Path, required=True)
    caption.add_argument("--mode", choices=["template", "blip"], default="template")
    caption.add_argument("--trigger-token", default="mystyle")
    caption.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    caption.add_argument("--local-files-only", action="store_true")
    train_cmd = sub.add_parser("train")
    train_cmd.add_argument("--input", type=Path, required=True)
    train_cmd.add_argument("--config", type=Path, required=True)
    train_cmd.add_argument("--output", type=Path, required=True)
    train_cmd.add_argument("--resume", type=Path)
    train_cmd.add_argument("--stop-after-step", type=int)
    evaluate_cmd = sub.add_parser("evaluate")
    evaluate_cmd.add_argument("--training-result", type=Path, required=True)
    evaluate_cmd.add_argument("--input", type=Path, required=True)
    evaluate_cmd.add_argument("--output", type=Path, required=True)
    evaluate_cmd.add_argument("--config", type=Path)
    compare_cmd = sub.add_parser("compare")
    compare_cmd.add_argument("--left", type=Path, required=True)
    compare_cmd.add_argument("--right", type=Path, required=True)
    compare_cmd.add_argument("--input", type=Path, required=True)
    compare_cmd.add_argument("--output", type=Path, required=True)
    compare_cmd.add_argument("--config", type=Path)
    for command in ("cpu-smoke", "gpu-smoke"):
        smoke = sub.add_parser(command)
        smoke.add_argument("--output", type=Path, required=True)
        if command == "gpu-smoke":
            smoke.add_argument("--model", default=TrainConfig().model_name)
            smoke.add_argument("--revision")
            smoke.add_argument("--gpu-uuid")
            smoke.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args(argv)
    from .observability import configure_logging

    configure_logging()
    try:
        if args.command == "api":
            import uvicorn
            from .api import create_app

            settings = settings_from_env()
            if not settings.api_keys:
                raise PipelineError("AUTH_NOT_CONFIGURED", "Set LORA_API_KEYS before starting the API")
            uvicorn.run(create_app(settings), host=args.host, port=args.port)
        elif args.command == "worker":
            from .worker import Worker

            Worker(settings_from_env()).run(once=args.once)
        elif args.command == "worker-health":
            from .store import Store

            return 0 if Store(settings_from_env()).worker_ready() else 1
        elif args.command == "preflight":
            _print(preflight())
        elif args.command == "generate-data":
            from .data import generate_synthetic_dataset

            files = generate_synthetic_dataset(args.output, count=args.count)
            atomic_write(args.output / "files.json", canonical_json(files))
            _print({"files": str(args.output / "files.json"), "count": len(files)})
        elif args.command == "prepare":
            from .data import prepare_dataset

            _print(prepare_dataset(json.loads(args.files.read_text()), args.output))
        elif args.command == "caption":
            from .captions import caption_dataset

            with _device(args.device):
                _print(
                    caption_dataset(
                        load_manifest(args.prepared),
                        args.output,
                        mode=args.mode,
                        trigger_token=args.trigger_token,
                        device=args.device,
                        local_files_only=args.local_files_only,
                    )
                )
        elif args.command == "train":
            config = TrainConfig(**json.loads(args.config.read_text()))
            with _device(config.device):
                from .training import train

                _print(
                    train(
                        load_manifest(args.input),
                        args.output,
                        config,
                        progress=_print,
                        resume_from=args.resume,
                        stop_after_step=args.stop_after_step,
                    )
                )
        elif args.command in {"evaluate", "compare"}:
            config = EvalConfig(**json.loads(args.config.read_text())) if args.config else EvalConfig()
            with _device(config.device):
                from .evaluation import compare_adapters, evaluate

                inputs = load_manifest(args.input)
                if args.command == "evaluate":
                    result = evaluate(
                        load_manifest(args.training_result), inputs, args.output, config, progress=_print
                    )
                else:
                    result = compare_adapters(
                        load_manifest(args.left),
                        load_manifest(args.right),
                        inputs,
                        args.output,
                        config,
                        progress=_print,
                    )
                _print(result)
        else:
            _smoke(args)
        return 0
    except (PipelineError, ValueError, OSError) as exc:
        error = {
            "code": getattr(exc, "code", "CONFIGURATION_ERROR"),
            "message": str(exc),
            "details": getattr(exc, "details", {}),
        }
        print(json.dumps({"error": error}), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
