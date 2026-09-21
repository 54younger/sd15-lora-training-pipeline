import json
import logging

from lora_pipeline.observability import HttpMetrics, log_event


def test_metric_registries_are_independent_and_units_are_seconds():
    first, second = HttpMetrics(), HttpMetrics()
    first.observe("GET", "/v1/training-jobs/{job_id}", 200, 0.25)
    output = first.render()
    assert 'route="/v1/training-jobs/{job_id}"' in output
    assert (
        'lora_http_request_duration_seconds_sum{method="GET",route="/v1/training-jobs/{job_id}"} 0.25'
        in output
    )
    assert "lora_http_requests_total{" not in second.render()


def test_structured_event_has_parseable_context(caplog):
    with caplog.at_level(logging.INFO, logger="lora_pipeline"):
        log_event("stage_started", job_id="j", attempt_id="a", stage="TRAIN")
    record = json.loads(caplog.records[-1].message)
    assert record["event"] == "stage_started"
    assert record["job_id"] == "j" and record["attempt_id"] == "a"
    assert "timestamp" in record
