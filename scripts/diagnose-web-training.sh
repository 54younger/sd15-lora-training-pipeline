#!/usr/bin/env bash
set -eu

cd "$(dirname "$0")/.."

compose() {
  docker compose -f compose.yaml -f compose.gpu.yaml "$@"
}

echo "## Compose services"
compose ps

echo
echo "## Recent worker logs"
# Keep the useful structured log fields while bounding/redacting a line that
# may contain an older third-party exception or a credential-bearing URL.
compose logs --since=30m --tail=500 worker \
  | sed -E \
      -e 's#([Aa]uthorization[=:][[:space:]]*[Bb]earer[[:space:]]+)[^[:space:],;]+#\1***#g' \
      -e 's#([Bb]earer[[:space:]]+)[^[:space:],;]+#\1***#g' \
      -e 's#([?&](hf[_-]?token|access[_-]?token|api[_-]?key|token|authorization)=)[^&[:space:]]+#\1***#g' \
      -e 's#(https?://)[^@/[:space:]]+@#\1***@#g' \
  | awk '{ if (length($0) > 4000) print substr($0, 1, 4000) "... [truncated]"; else print }'

echo
echo "## Worker-side training initialization checks"
compose exec -T -e DIAG_JOB_ID="${1:-}" worker python - <<'PY'
import json
import os
import re
import sqlite3
import subprocess
import traceback

failures = []
MAX_DIAGNOSTIC_CHARS = 800


def sanitize_text(value, limit=MAX_DIAGNOSTIC_CHARS):
    """Keep actionable text without copying credentials or unbounded payloads."""
    raw = str(value or "")
    if "traceback (most recent call last):" in raw.lower():
        # Persisted tracebacks can contain locals, request headers, and long
        # filesystem paths. Keep their final exception/type line; call sites
        # for checks are reported separately by traceback_locations().
        lines = [line.strip() for line in raw.splitlines() if line.strip()]
        raw = "traceback omitted; final_exception=" + (lines[-1] if lines else "unknown")
    text = " ".join(raw.split())
    text = re.sub(r"(?i)(https?://)[^@/\s]+@", r"\1***@", text)
    text = re.sub(
        r"(?i)([?&](?:hf[_-]?token|access[_-]?token|api[_-]?key|token|authorization)=)[^&\s]+",
        r"\1***",
        text,
    )
    text = re.sub(
        r"(?i)(\bauthorization\s*[:=]\s*(?:bearer|basic)\s+)[^\s,;]+",
        r"\1***",
        text,
    )
    text = re.sub(r"(?i)(\bbearer\s+)[^\s,;]+", r"\1***", text)
    text = re.sub(
        r"(?i)([\"']?\b(?:hf[_-]?token|access[_-]?token|api[_-]?key|token|password|secret)\b[\"']?\s*[:=]\s*[\"']?)[^\"'\s,;}]+",
        r"\1***",
        text,
    )
    text = re.sub(r"\b(?:hf_|sk-|gh[pousr]_|glpat-)[A-Za-z0-9_-]+\b", "***", text)
    if len(text) > limit:
        return text[:limit] + "... [truncated]"
    return text


def traceback_locations(exc):
    """Expose call sites only; traceback text can include exception arguments."""
    return [
        {"file": os.path.basename(frame.filename), "line": frame.lineno, "function": frame.name}
        for frame in traceback.extract_tb(exc.__traceback__)[-12:]
    ]


def safe_exception(exc):
    summary = f"{type(exc).__name__}: {sanitize_text(exc)}"
    locations = traceback_locations(exc)
    if locations:
        return f"{summary}; traceback_locations={json.dumps(locations, sort_keys=True)}"
    return summary


def section(title):
    print(f"\n### {title}", flush=True)


def check(label, fn):
    print(f"[{label}] starting", flush=True)
    try:
        value = fn()
        print(f"[{label}] OK: {sanitize_text(repr(value))}", flush=True)
        return value
    except Exception as exc:
        print(f"[{label}] FAILED: {safe_exception(exc)}", flush=True)
        failures.append(label)
        return None


section("Runtime")
print("pid:", os.getpid())
print("initial CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))
print("LORA_GPU_UUIDS:", os.environ.get("LORA_GPU_UUIDS"))
check(
    "nvidia-smi",
    lambda: subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,uuid,name", "--format=csv,noheader"],
        text=True,
    ).strip(),
)

section("Latest failed web job")
database = sqlite3.connect("/data/pipeline.sqlite3")
database.row_factory = sqlite3.Row
requested_job_id = os.environ.get("DIAG_JOB_ID", "").strip()
parameters = ()
job_filter = ""
if requested_job_id:
    job_filter = "AND j.id = ?"
    parameters = (requested_job_id,)
job = database.execute(
    f"""
    SELECT id, state, input_json, profile_json, training_overrides_json,
           error_code, error_message, updated_at
      FROM jobs j
     WHERE j.state = 'FAILED'
       AND EXISTS (
           SELECT 1
             FROM stage_tasks st
            WHERE st.job_id = j.id AND st.stage = 'TRAIN'
       )
       {job_filter}
     ORDER BY j.updated_at DESC
     LIMIT 1
    """,
    parameters,
).fetchone()

if job is None:
    print("No matching failed TRAIN job was found.")
    raise SystemExit(1)

print(
    "job:",
    {
        key: sanitize_text(job[key]) if key in {"error_code", "error_message"} else job[key]
        for key in ("id", "state", "error_code", "error_message", "updated_at")
    },
)

attempts = database.execute(
    """
    SELECT st.stage, ta.attempt_no, ta.state, ta.gpu_slot, ta.error_code, ta.error_message,
           ta.started_at, ta.ended_at
      FROM task_attempts ta
      JOIN stage_tasks st ON st.id = ta.task_id
     WHERE st.job_id = ?
     ORDER BY ta.started_at
    """,
    (job["id"],),
).fetchall()
print("attempts:")
for attempt in attempts:
    item = dict(attempt)
    for key in ("error_code", "error_message"):
        item[key] = sanitize_text(item[key])
    print(" ", item)

failed_train_attempt = database.execute(
    """
    SELECT ta.gpu_slot
      FROM task_attempts ta
      JOIN stage_tasks st ON st.id = ta.task_id
     WHERE st.job_id = ? AND st.stage = 'TRAIN' AND ta.state = 'FAILED'
     ORDER BY ta.attempt_no DESC
     LIMIT 1
    """,
    (job["id"],),
).fetchone()
gpu_slot = failed_train_attempt["gpu_slot"] if failed_train_attempt else None
print("failed TRAIN gpu_slot:", gpu_slot)
if gpu_slot and not str(gpu_slot).startswith("cpu-test-"):
    # Match worker.py: the physical UUID must be selected before torch is imported,
    # while the training config addresses it locally as cuda:0.
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_slot
print("diagnostic CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))

import torch

print("torch:", torch.__version__)
print("torch CUDA build:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
print("CUDA device count:", torch.cuda.device_count())

input_manifest = json.loads(job["input_json"] or "{}")
profile = json.loads(job["profile_json"] or "{}")
overrides = json.loads(job["training_overrides_json"] or "{}")
raw_config = dict(profile.get("config") or {})
raw_config.update(overrides)
if gpu_slot and str(gpu_slot).startswith("cpu-test-"):
    # Match Worker._stage exactly: test slots are an explicit CPU/tiny domain,
    # even when this container happens to expose a CUDA runtime.
    raw_config.update({"backend": "tiny", "device": "cpu", "precision": "fp32", "resolution": 16})
elif gpu_slot:
    raw_config["device"] = "cuda:0"

from lora_pipeline.config import TrainConfig
from lora_pipeline.training import (
    _compatibility_key,
    _safe_device,
    _seed_everything,
    validate_input_integrity,
)

config = check("build_train_config", lambda: TrainConfig(**raw_config))
if config is None:
    raise SystemExit(1)
visible_config = {
    key: getattr(config, key, None)
    for key in (
        "backend",
        "device",
        "precision",
        "resolution",
        "max_steps",
        "model_name",
        "revision",
        "local_files_only",
    )
}
visible_config = {
    key: sanitize_text(value) if isinstance(value, (str, bytes)) else value
    for key, value in visible_config.items()
}
print("training config:", visible_config)
print("input keys:", sorted(input_manifest.keys()))

section("Exact pre-model-load sequence")
check("validate_input_integrity", lambda: validate_input_integrity(input_manifest))
device = check("safe_device", lambda: _safe_device(config))
check("seed_everything", lambda: _seed_everything(config.seed))
check("compatibility_key", lambda: _compatibility_key(input_manifest, config))
if device is not None:
    tensor_label = "cuda_tensor" if device.type == "cuda" else "cpu_tensor"
    check(tensor_label, lambda: (torch.ones(1, device=device) * 2).cpu().tolist())
    if device.type == "cuda":
        check("reset_peak_memory_stats", lambda: torch.cuda.reset_peak_memory_stats(device))
        check("memory_allocated", lambda: torch.cuda.memory_allocated(device))

print("\nDiagnostic complete; no model was loaded and no training was started.")
if failures:
    print("Failed checks:", ", ".join(failures))
    raise SystemExit(1)
PY
