"""
Performance expectations for the HTTP layer (no LLM execution).

These are smoke benchmarks: they catch accidental regressions such as
synchronous blocking in middleware, not absolute SLA guarantees.
"""

from __future__ import annotations

import statistics
import time

import pytest


@pytest.mark.performance
def test_health_p50_under_budget(client):
    durations_ms = []
    for _ in range(30):
        t0 = time.perf_counter()
        r = client.get("/health")
        durations_ms.append((time.perf_counter() - t0) * 1000)
        assert r.status_code == 200

    p50 = statistics.median(durations_ms)
    assert p50 < 250, f"health median {p50:.1f}ms exceeds 250ms budget"


@pytest.mark.performance
def test_root_throughput_smoke(client):
    t0 = time.perf_counter()
    n = 50
    for _ in range(n):
        assert client.get("/").status_code == 200
    elapsed = time.perf_counter() - t0
    rps = n / elapsed
    assert rps > 20, f"root RPS {rps:.1f} below smoke threshold"
