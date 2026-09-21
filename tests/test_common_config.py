import json

import pytest

from lora_pipeline.common import PipelineError, load_manifest, write_manifest
from lora_pipeline.config import DataConfig, EvalConfig, Settings, TrainConfig, settings_from_env


def test_manifest_integrity_and_no_self_hash(tmp_path):
    path = tmp_path / "manifest.json"
    result = write_manifest(path, {"schema_version": 1, "values": [1, 2]})
    assert "manifest_sha256" not in json.loads(path.read_text())
    assert load_manifest(path, result["manifest_sha256"]) == result
    path.write_text("{}")
    with pytest.raises(PipelineError, match="checksum"):
        load_manifest(path, result["manifest_sha256"])


def test_config_boundaries():
    for kwargs in (
        {"rank": 0},
        {"resolution": 1024},
        {"learning_rate": float("nan")},
        {"max_grad_norm": float("nan")},
        {"device": "cpu"},
    ):
        with pytest.raises(ValueError):
            TrainConfig(**kwargs)
    with pytest.raises(ValueError):
        DataConfig(max_images=99)
    with pytest.raises(ValueError):
        EvalConfig(prompts=[])
    with pytest.raises(ValueError):
        Settings(fake_slots=1)


@pytest.mark.parametrize("seed", ["not-an-int", 1.5, True, -1, 2**63])
def test_seed_must_be_valid_integer(seed):
    with pytest.raises(ValueError):
        TrainConfig(seed=seed)
    with pytest.raises(ValueError):
        DataConfig(seed=seed)


def test_env_test_backend_is_explicit(monkeypatch, tmp_path):
    monkeypatch.delenv("LORA_CONFIG", raising=False)
    monkeypatch.setenv("LORA_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("LORA_TEST_BACKEND", "1")
    monkeypatch.setenv("LORA_API_KEYS", '{"secret":"alice"}')
    settings = settings_from_env()
    assert settings.train.backend == "tiny"
    assert settings.fake_slots == 1
    assert settings.train.device == "cpu"
    assert settings.api_keys == {"secret": "alice"}
