#!/usr/bin/env python3
"""Launch a real, offline CPU fixture for the browser workflow."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import signal
import stat
import subprocess
import sys
import tempfile
import time

from lora_pipeline.data import generate_synthetic_dataset


ROOT = Path(__file__).resolve().parents[1]
TOKENS = {"studio-test-token": "studio-test", "other-test-token": "other"}
_ISOLATED_ENV = {
    "HF_HUB_OFFLINE",
    "TRANSFORMERS_OFFLINE",
    "HF_TOKEN",
    "HUGGINGFACEHUB_API_TOKEN",
    "HF_HOME",
    "HF_HUB_CACHE",
    "TRANSFORMERS_CACHE",
    "TOKENIZERS_PARALLELISM",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "TORCH_NUM_THREADS",
}


def _root(value: Path | None) -> Path:
    if value is None:
        return Path(tempfile.mkdtemp(prefix="lora-web-fixture-")).resolve()
    root = value.expanduser().resolve()
    if root.exists():
        if not root.is_dir() or any(root.iterdir()):
            raise ValueError(f"--data-dir must be a new or empty directory: {root}")
    else:
        root.mkdir(parents=True)
    return root


def _settings(root: Path) -> Path:
    config = {
        "data_dir": str(root / "pipeline"),
        "api_keys": TOKENS,
        "admin_keys": [],
        "enable_test_backend": True,
        "fake_slots": 1,
        "cpu_slots": 1,
        "train": {
            "backend": "tiny",
            "model_name": "tiny-test-model",
            "resolution": 16,
            "rank": 2,
            "lora_alpha": 2,
            "batch_size": 1,
            "gradient_accumulation_steps": 1,
            "learning_rate": 0.0001,
            "max_steps": 4,
            "checkpoint_every": 2,
            "seed": 42,
            "precision": "fp32",
            "device": "cpu",
            "local_files_only": True,
        },
        "evaluation": {
            "prompts": ["a colorful geometric shape"],
            "seeds": [42],
            "inference_steps": 1,
            "guidance_scale": 7.5,
            "device": "cpu",
            "local_files_only": True,
        },
        "caption_mode": "template",
        "caption_device": "cpu",
    }
    path = root / "settings.json"
    path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    os.chmod(path, stat.S_IRUSR | stat.S_IWUSR)
    return path


def _fixture(root: Path, port: int) -> dict[str, str | int]:
    images = root / "images"
    records = generate_synthetic_dataset(images, count=100)
    captions_path = root / "captions.json"
    captions = {Path(item["path"]).name: item["caption"] for item in records}
    captions_path.write_text(json.dumps(captions, indent=2) + "\n", encoding="utf-8")
    return {
        "api_url": f"http://127.0.0.1:{port}",
        "token": "studio-test-token",
        "owner": "studio-test",
        "other_token": "other-test-token",
        "other_owner": "other",
        "images_dir": str(images),
        "captions_path": str(captions_path),
        "data_dir": str(root),
        "config_path": str(root / "settings.json"),
    }


def _environment(config: Path) -> dict[str, str]:
    env = os.environ.copy()
    for name in tuple(env):
        if name.startswith("LORA_") or name.startswith("HF_") or name.startswith("HUGGINGFACE"):
            env.pop(name, None)
        elif name in _ISOLATED_ENV:
            env.pop(name, None)
    env.update(
        {
            "LORA_CONFIG": str(config),
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "TOKENIZERS_PARALLELISM": "false",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "TORCH_NUM_THREADS": "1",
            "PYTHONPATH": str(ROOT / "src")
            + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""),
        }
    )
    return env


def _process_groups(process: subprocess.Popen[bytes]) -> set[int]:
    pids = {process.pid}
    try:
        import psutil

        pids.update(child.pid for child in psutil.Process(process.pid).children(recursive=True))
    except (ImportError, OSError):
        pass
    groups: set[int] = set()
    own_group = os.getpgrp()
    for pid in pids:
        try:
            group = os.getpgid(pid)
            if group != own_group:
                groups.add(group)
        except ProcessLookupError:
            continue
    return groups


def _stop(process: subprocess.Popen[bytes], timeout: float = 3.0) -> None:
    if process.poll() is not None:
        return
    for group in _process_groups(process):
        try:
            os.killpg(group, signal.SIGTERM)
        except ProcessLookupError:
            pass
    deadline = time.monotonic() + timeout
    while process.poll() is None and time.monotonic() < deadline:
        time.sleep(0.05)
    if process.poll() is None:
        for group in _process_groups(process):
            try:
                os.killpg(group, signal.SIGKILL)
            except ProcessLookupError:
                pass
        process.wait(timeout=timeout)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, help="new or empty directory retained after shutdown")
    parser.add_argument("--metadata-path", type=Path, help="also write fixture metadata JSON here")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--prepare-only", action="store_true", help="generate the fixture without starting services")
    args = parser.parse_args(argv)
    try:
        root = _root(args.data_dir)
        config = _settings(root)
        metadata = _fixture(root, args.port)
        if args.metadata_path:
            args.metadata_path.parent.mkdir(parents=True, exist_ok=True)
            args.metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
            os.chmod(args.metadata_path, stat.S_IRUSR | stat.S_IWUSR)
        if args.prepare_only:
            print(json.dumps(metadata, sort_keys=True), flush=True)
            return 0
        env = _environment(config)
        command = [sys.executable, "-m", "lora_pipeline"]
        children = []
        for name, arguments in (
            ("api", ["api", "--host", "127.0.0.1", "--port", str(args.port)]),
            ("worker", ["worker"]),
        ):
            with (root / f"{name}.log").open("ab") as log:
                children.append(subprocess.Popen(
                    command + arguments,
                    cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
                ))
        print(json.dumps(metadata, sort_keys=True), flush=True)
        stopping = False

        def request_stop(_signum: int, _frame: object) -> None:
            nonlocal stopping
            stopping = True

        signal.signal(signal.SIGINT, request_stop)
        signal.signal(signal.SIGTERM, request_stop)
        while not stopping:
            dead = [child for child in children if child.poll() is not None]
            if dead:
                return_code = next((child.returncode for child in dead if child.returncode), 1)
                break
            time.sleep(0.2)
        else:
            return_code = 0
    finally:
        for child in locals().get("children", []):
            _stop(child)
    return return_code


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError) as exc:
        print(json.dumps({"error": str(exc)}), file=sys.stderr)
        raise SystemExit(2)
