"""Security-oriented checks: input validation and safe error surfaces."""

from __future__ import annotations

import pytest


@pytest.mark.security
def test_error_bodies_are_json_strings_not_tracebacks(client):
    r = client.get("/datasets/%00broken")
    assert r.status_code in (400, 404)
    body = r.json()
    assert isinstance(body.get("detail"), str)
    assert "Traceback" not in body.get("detail", "")


@pytest.mark.security
def test_retraining_job_id_rejects_non_hex_suffix(client):
    # Single path segment (Starlette collapses "/../" across segments, so use one segment).
    r = client.get("/retraining/jobs/job_not_a_valid_hex_id/logs")
    assert r.status_code == 400


@pytest.mark.security
def test_openapi_does_not_echo_env_secrets(client):
    r = client.get("/openapi.json")
    spec = r.json()
    dumped = str(spec)
    assert "your-secret-key-change-this" not in dumped.lower()
