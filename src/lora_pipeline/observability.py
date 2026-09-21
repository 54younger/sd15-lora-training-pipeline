"""Bounded structured logs and in-process HTTP metrics, without payload logging."""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone

from prometheus_client import CollectorRegistry, Counter, Histogram, generate_latest


def log_event(event: str, **fields) -> None:
    """Callers pass identifiers and summaries, never tokens, captions or images."""
    payload = {"timestamp": datetime.now(timezone.utc).isoformat(), "event": event, **fields}
    logging.getLogger("lora_pipeline").info(json.dumps(payload, default=str, allow_nan=False))


def configure_logging() -> None:
    logger = logging.getLogger("lora_pipeline")
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False


class HttpMetrics:
    """A registry per app prevents duplicate collector registration in tests."""

    def __init__(self):
        self.registry = CollectorRegistry()
        self.requests = Counter(
            "lora_http_requests_total",
            "Completed API requests",
            ["method", "route", "status"],
            registry=self.registry,
        )
        self.latency = Histogram(
            "lora_http_request_duration_seconds",
            "API request duration",
            ["method", "route"],
            buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10),
            registry=self.registry,
        )

    def observe(self, method: str, route: str, status: int, seconds: float) -> None:
        self.requests.labels(method, route, str(status)).inc()
        self.latency.labels(method, route).observe(seconds)

    def render(self) -> str:
        return generate_latest(self.registry).decode("utf-8")
