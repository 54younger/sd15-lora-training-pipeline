"""Validated immutable-by-convention configuration snapshots."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import math
import os
from pathlib import Path


@dataclass
class DataConfig:
    min_images: int = 100
    max_images: int = 1000
    max_file_bytes: int = 20 * 1024 * 1024
    max_total_bytes: int = 2 * 1024**3
    max_pixels: int = 40_000_000
    min_side: int = 256
    validation_fraction: float = 0.1
    min_train_images: int = 80
    min_validation_images: int = 10
    min_groups_per_split: int = 2
    phash_distance: int = 4
    seed: int = 42

    def __post_init__(self):
        counts = (
            self.min_images,
            self.max_images,
            self.max_file_bytes,
            self.max_total_bytes,
            self.max_pixels,
            self.min_side,
            self.min_train_images,
            self.min_validation_images,
            self.min_groups_per_split,
        )
        if any(type(v) is not int or v < 1 for v in counts):
            raise ValueError("Data limits must be positive integers")
        if self.min_images > self.max_images or not 0 < self.validation_fraction < 1:
            raise ValueError("Invalid dataset count or validation fraction")
        if not 0 <= self.phash_distance <= 64:
            raise ValueError("Invalid perceptual hash distance")
        if type(self.seed) is not int or not 0 <= self.seed < 2**63:
            raise ValueError("Split seed must be an integer in [0, 2**63)")
        if type(self.phash_distance) is not int:
            raise ValueError("Perceptual hash distance must be an integer")


@dataclass
class TrainConfig:
    backend: str = "sd15"
    model_name: str = "stable-diffusion-v1-5/stable-diffusion-v1-5"
    revision: str | None = None
    resolution: int = 512
    rank: int = 4
    lora_alpha: int = 4
    batch_size: int = 1
    gradient_accumulation_steps: int = 4
    learning_rate: float = 1e-4
    max_steps: int = 500
    checkpoint_every: int = 50
    seed: int = 42
    precision: str = "fp16"
    device: str = "cuda"
    gradient_checkpointing: bool = True
    random_crop: bool = False
    horizontal_flip: bool = False
    max_grad_norm: float = 1.0
    local_files_only: bool = False

    def __post_init__(self):
        integer_fields = (
            self.resolution,
            self.rank,
            self.lora_alpha,
            self.batch_size,
            self.gradient_accumulation_steps,
            self.max_steps,
            self.checkpoint_every,
            self.seed,
        )
        if any(type(value) is not int for value in integer_fields):
            raise ValueError("Training counts and seed must be integers")
        if not 0 <= self.seed < 2**63:
            raise ValueError("Seed must be in [0, 2**63)")
        if self.device not in {"cpu", "cuda", "cuda:0"}:
            raise ValueError("Device must be cpu or the single assigned CUDA device")
        if self.backend not in {"sd15", "tiny"} or self.precision not in {"fp16", "fp32", "bf16"}:
            raise ValueError("Unsupported training backend or precision")
        if not 1 <= self.rank <= 64 or not 1 <= self.lora_alpha <= 128:
            raise ValueError("Invalid LoRA rank/alpha")
        if not 1 <= self.batch_size <= 16 or not 1 <= self.gradient_accumulation_steps <= 128:
            raise ValueError("Invalid batch/accumulation")
        if not 1 <= self.max_steps <= 100_000 or self.checkpoint_every < 1:
            raise ValueError("Invalid step/checkpoint count")
        if (
            not 0 < self.learning_rate <= 0.1
            or not math.isfinite(self.max_grad_norm)
            or self.max_grad_norm <= 0
        ):
            raise ValueError("Invalid optimizer parameters")
        if self.backend == "sd15" and self.resolution not in {256, 512}:
            raise ValueError("SD 1.5 resolution must be 256 or 512")
        if self.backend == "tiny" and self.resolution not in {16, 32}:
            raise ValueError("Tiny test resolution must be 16 or 32")
        if self.device == "cpu" and self.precision != "fp32":
            raise ValueError("CPU training requires fp32")

    def snapshot(self) -> dict:
        return asdict(self)


@dataclass
class EvalConfig:
    prompts: list[str] = field(
        default_factory=lambda: [
            "a red bicycle",
            "a blue ceramic cup",
            "a mountain village",
            "a small sailboat",
            "a reading room",
            "a yellow bird",
            "a stone bridge",
            "a flower garden",
            "a vintage camera",
            "a forest path",
            "a green teapot",
            "a lighthouse",
            "a wooden chair",
            "a snowy cabin",
            "a violin",
            "a city skyline",
            "a basket of apples",
            "a train station",
            "a butterfly",
            "a desert landscape",
        ]
    )
    seeds: list[int] = field(default_factory=lambda: [42, 123])
    inference_steps: int = 30
    guidance_scale: float = 7.5
    clip_model: str = "openai/clip-vit-base-patch32"
    clip_revision: str | None = None
    device: str = "cuda"
    local_files_only: bool = False
    quality_policy: dict | None = None

    def __post_init__(self):
        if self.device not in {"cpu", "cuda", "cuda:0"}:
            raise ValueError("Evaluation device must be cpu or the assigned CUDA device")
        if type(self.inference_steps) is not int:
            raise ValueError("Inference steps must be an integer")
        if not self.prompts or not self.seeds or len(self.prompts) * len(self.seeds) > 100:
            raise ValueError("Evaluation requires 1..100 prompt/seed pairs per variant")
        if any(not isinstance(p, str) or not p.strip() or len(p) > 512 for p in self.prompts):
            raise ValueError("Invalid evaluation prompt")
        if any(type(s) is not int or not 0 <= s < 2**63 for s in self.seeds):
            raise ValueError("Invalid evaluation seed")
        if not 1 <= self.inference_steps <= 100 or not 0 <= self.guidance_scale <= 30:
            raise ValueError("Invalid inference settings")


@dataclass
class Settings:
    data_dir: Path = Path("./var")
    api_keys: dict[str, str] = field(default_factory=dict)  # token -> owner ID
    admin_keys: list[str] = field(default_factory=list)
    enable_test_backend: bool = False
    gpu_uuids: list[str] = field(default_factory=list)
    fake_slots: int = 0
    cpu_slots: int = 2
    owner_job_limit: int = 5
    global_job_limit: int = 100
    storage_quota_bytes: int = 10 * 1024**3
    heartbeat_seconds: float = 10
    lease_seconds: float = 60
    max_stage_seconds: int = 3600
    max_job_gpu_seconds: int = 7200
    max_attempts: int = 3
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    evaluation: EvalConfig = field(default_factory=EvalConfig)
    caption_mode: str = "template"
    caption_device: str = "cpu"
    caption_model: str = "Salesforce/blip-image-captioning-base"
    caption_revision: str | None = None

    def __post_init__(self):
        self.data_dir = Path(self.data_dir).resolve()
        if not 1 <= self.cpu_slots <= 16 or not 0 <= self.fake_slots <= 4:
            raise ValueError("Invalid CPU or fake-device slot count")
        if self.fake_slots and not self.enable_test_backend:
            raise ValueError("Fake device slots require the explicit test backend")
        if len(self.gpu_uuids) != len(set(self.gpu_uuids)):
            raise ValueError("Physical GPU UUIDs must be unique")
        if not 0 < self.heartbeat_seconds < self.lease_seconds:
            raise ValueError("Heartbeat must be shorter than the lease")
        if self.caption_mode not in {"template", "blip"}:
            raise ValueError("Unsupported caption mode")
        if self.caption_device not in {"cpu", "cuda", "cuda:0"}:
            raise ValueError("Unsupported caption device")
        if any(
            not isinstance(k, str) or not k or not isinstance(v, str) or not v
            for k, v in self.api_keys.items()
        ):
            raise ValueError("API keys must map nonempty tokens to nonempty owner IDs")
        if any(
            v < 1
            for v in (
                self.owner_job_limit,
                self.global_job_limit,
                self.storage_quota_bytes,
                self.max_stage_seconds,
                self.max_job_gpu_seconds,
                self.max_attempts,
            )
        ):
            raise ValueError("Resource limits must be positive")

    @property
    def db_path(self) -> Path:
        return Path(self.data_dir) / "pipeline.sqlite3"


def settings_from_env() -> Settings:
    """Operator-only JSON config; env takes precedence for deployment values."""
    raw = json.loads(Path(os.environ["LORA_CONFIG"]).read_text()) if os.getenv("LORA_CONFIG") else {}
    environment = {
        "LORA_DATA_DIR": "data_dir",
        "LORA_CAPTION_MODE": "caption_mode",
        "LORA_CAPTION_DEVICE": "caption_device",
    }
    for env, key in environment.items():
        if env in os.environ:
            raw[key] = os.environ[env]
    for env, key in (("LORA_API_KEYS", "api_keys"), ("LORA_ADMIN_KEYS", "admin_keys")):
        if env in os.environ:
            raw[key] = json.loads(os.environ[env])
    if "LORA_TEST_BACKEND" in os.environ:
        raw["enable_test_backend"] = os.environ["LORA_TEST_BACKEND"].lower() in {"1", "true"}
    if "LORA_FAKE_SLOTS" in os.environ:
        raw["fake_slots"] = int(os.environ["LORA_FAKE_SLOTS"])
    if "LORA_GPU_UUIDS" in os.environ:
        raw["gpu_uuids"] = [v.strip() for v in os.environ["LORA_GPU_UUIDS"].split(",") if v.strip()]
    if raw.get("enable_test_backend"):
        raw.setdefault("fake_slots", 1)
        raw.setdefault(
            "train",
            {
                "backend": "tiny",
                "device": "cpu",
                "precision": "fp32",
                "resolution": 32,
                "rank": 2,
                "lora_alpha": 2,
                "max_steps": 4,
                "checkpoint_every": 2,
                "gradient_accumulation_steps": 1,
            },
        )
        raw.setdefault(
            "evaluation",
            {
                "device": "cpu",
                "prompts": ["a red circle", "a blue square"],
                "seeds": [42],
                "inference_steps": 2,
            },
        )
    raw["data"] = DataConfig(**raw.get("data", {}))
    raw["train"] = TrainConfig(**raw.get("train", {}))
    raw["evaluation"] = EvalConfig(**raw.get("evaluation", {}))
    return Settings(**raw)
