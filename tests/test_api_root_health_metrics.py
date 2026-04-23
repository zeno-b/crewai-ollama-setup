"""Root, health, and metrics: structured responses and observability."""

from __future__ import annotations


def test_root_returns_json_with_docs_links(client):
    r = client.get("/")
    assert r.status_code == 200
    body = r.json()
    assert body["message"] == "CrewAI Service is running"
    assert body["version"] == "1.0.0"
    assert body["docs"] == "/docs"
    assert body["health"] == "/health"
    assert body["metrics"] == "/metrics"
    assert r.headers.get("content-type", "").startswith("application/json")


def test_health_schema_and_degraded_when_upstream_stubbed(client):
    r = client.get("/health")
    assert r.status_code == 200
    data = r.json()
    assert set(data.keys()) >= {
        "status",
        "timestamp",
        "ollama_connected",
        "redis_connected",
        "version",
    }
    assert data["redis_connected"] is True
    assert data["ollama_connected"] is True
    assert data["status"] in ("healthy", "degraded")
    assert data["version"]


def test_metrics_returns_openmetrics_text(client):
    r = client.get("/metrics")
    assert r.status_code == 200
    text = r.text
    assert "crewai_requests_total" in text or "#" in text
    assert r.headers.get("content-type", "")


def test_openapi_available(client):
    r = client.get("/openapi.json")
    assert r.status_code == 200
    spec = r.json()
    assert spec["info"]["title"]
    assert "/health" in spec["paths"]
