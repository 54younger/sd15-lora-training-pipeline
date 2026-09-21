import json
from pathlib import Path

from lora_pipeline import cli
from lora_pipeline.common import write_manifest
from lora_pipeline.config import TrainConfig


def test_train_cli_keeps_final_json_on_stdout_and_progress_on_stderr(tmp_path: Path, monkeypatch, capsys):
    # ``configure_logging`` installs a process-global handler.  Avoid binding
    # it to pytest's temporary captured stderr, which is closed after this test.
    monkeypatch.setattr("lora_pipeline.observability.configure_logging", lambda: None)
    input_path = tmp_path / "training-input.json"
    write_manifest(input_path, {"train": [], "validation": []})
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(TrainConfig(backend="tiny", device="cpu", resolution=16, precision="fp32").snapshot())
    )

    def fake_train(_manifest, _output, _config, *, progress, **_kwargs):
        progress({"phase": "training", "current": 1, "total": 2, "global_step": 1})
        progress({"phase": "training_completed", "current": 2, "total": 2})
        return {"kind": "training-result", "state": "COMPLETED"}

    monkeypatch.setattr("lora_pipeline.training.train", fake_train)
    assert cli.main(
        [
            "train",
            "--input",
            str(input_path),
            "--config",
            str(config_path),
            "--output",
            str(tmp_path / "output"),
        ]
    ) == 0

    captured = capsys.readouterr()
    assert json.loads(captured.out) == {"kind": "training-result", "state": "COMPLETED"}
    assert "[training]" in captured.err
    assert "1/2" in captured.err
    assert "50.0%" in captured.err
    assert "[training completed]" in captured.err


def test_cli_progress_renderer_handles_unknown_or_zero_total_events(capsys):
    cli._progress_to_stderr({"phase": "legacy_event", "global_step": 3})
    cli._progress_to_stderr(object())

    captured = capsys.readouterr()
    assert "[legacy event]" in captured.err
    assert "3/?" in captured.err
    assert "[working]" in captured.err
