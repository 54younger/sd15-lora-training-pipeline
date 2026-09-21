"""Shared artifact contracts. Published manifests never include their own digest."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any


class PipelineError(Exception):
    def __init__(self, code: str, message: str, *, retryable: bool = False, details: dict | None = None):
        super().__init__(message)
        self.code, self.message, self.retryable = code, message, retryable
        self.details = details or {}


def canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode()


def digest_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value)).hexdigest()


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic_write(path: str | Path, content: bytes) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".writing-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(content)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def write_manifest(path: str | Path, payload: dict) -> dict:
    body = {k: v for k, v in payload.items() if k not in ("manifest_path", "manifest_sha256")}
    data = canonical_json(body)
    path = Path(path).resolve()
    atomic_write(path, data)
    return {**body, "manifest_path": str(path), "manifest_sha256": hashlib.sha256(data).hexdigest()}


def load_manifest(path: str | Path, expected_sha256: str | None = None) -> dict:
    path = Path(path).resolve()
    digest = sha256_file(path)
    if expected_sha256 is not None and digest != expected_sha256:
        raise PipelineError("CHECKSUM_MISMATCH", "Artifact checksum does not match")
    return {**json.loads(path.read_text()), "manifest_path": str(path), "manifest_sha256": digest}
