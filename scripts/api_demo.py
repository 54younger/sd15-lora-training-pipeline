"""Exercise real HTTP upload/queue/worker/download using generated image files.

Start the API and worker in separate terminals first. No model downloads are
needed when the service runs with LORA_TEST_BACKEND=1.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
import uuid
from pathlib import Path

import httpx


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--files", type=Path, required=True)
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--token", default=os.getenv("LORA_DEMO_TOKEN"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=600)
    args = parser.parse_args()
    if not args.token:
        parser.error("Set LORA_DEMO_TOKEN or --token")
    entries = json.loads(args.files.read_text())
    metadata = []
    for item in entries:
        content = Path(item["path"]).read_bytes()
        metadata.append(
            {
                "name": item["name"],
                "size_bytes": len(content),
                "sha256": hashlib.sha256(content).hexdigest(),
                "mime_type": "image/png",
                "caption": item.get("caption"),
            }
        )
    with httpx.Client(
        base_url=args.url, headers={"Authorization": f"Bearer {args.token}"}, timeout=60
    ) as client:

        def mutation(path, body):
            response = client.post(path, json=body, headers={"Idempotency-Key": str(uuid.uuid4())})
            response.raise_for_status()
            return response.json()

        created = mutation("/v1/datasets", {"files": metadata})
        dataset_id = created.get("dataset_id", created.get("id"))
        uploaded = created.get("files", [])
        by_name = {entry["name"]: entry for entry in entries}
        for item in uploaded:
            file_id = item.get("file_id", item.get("id"))
            source = by_name[item["name"]]
            response = client.put(
                f"/v1/datasets/{dataset_id}/files/{file_id}",
                content=Path(source["path"]).read_bytes(),
                headers={"Content-Type": "image/png"},
            )
            response.raise_for_status()
        if len(uploaded) != len(entries):
            raise RuntimeError("Dataset response did not contain one upload entry per image")
        mutation(f"/v1/datasets/{dataset_id}/complete", {})
        deadline = time.monotonic() + args.timeout
        while time.monotonic() < deadline:
            response = client.get(f"/v1/datasets/{dataset_id}")
            response.raise_for_status()
            dataset = response.json()
            if dataset["state"] == "COMPLETED":
                break
            if dataset["state"] == "INVALID":
                raise RuntimeError(json.dumps(dataset))
            time.sleep(0.25)
        else:
            raise TimeoutError("Dataset verification timed out; is the worker running?")
        profiles = client.get("/v1/training-profiles").json()["profiles"]
        profile = next((p for p in profiles if p.get("config", {}).get("backend") == "tiny"), profiles[0])
        job = mutation(
            "/v1/training-jobs",
            {
                "dataset_id": dataset_id,
                "profile_revision_id": profile["profile_revision_id"],
                "trigger_token": "geometricstyle",
            },
        )
        job_id = job.get("job_id", job.get("id"))
        while time.monotonic() < deadline:
            response = client.get(f"/v1/training-jobs/{job_id}")
            response.raise_for_status()
            job = response.json()
            if job["state"] in {"READY", "COMPLETED_UNVERIFIED"}:
                break
            if job["state"] in {"FAILED", "QUALITY_REJECTED", "CANCELLED"}:
                raise RuntimeError(json.dumps(job))
            time.sleep(0.5)
        else:
            raise TimeoutError("Job timed out")
        args.output.mkdir(parents=True, exist_ok=True)
        evaluation = client.get(f"/v1/training-jobs/{job_id}/evaluation")
        evaluation.raise_for_status()
        model_id = job["model_id"]
        download = client.get(f"/v1/models/{model_id}/download", params={"allow_unverified": "true"})
        download.raise_for_status()
        from lora_pipeline.common import atomic_write, canonical_json

        atomic_write(args.output / "adapter.safetensors", download.content)
        atomic_write(args.output / "evaluation.json", canonical_json(evaluation.json()))
        atomic_write(args.output / "job.json", canonical_json(job))
        print(json.dumps({"job_id": job_id, "state": job["state"], "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
