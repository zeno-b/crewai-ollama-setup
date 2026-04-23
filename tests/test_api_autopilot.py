"""Autopilot HTTP surface."""

from __future__ import annotations


def test_autopilot_status_ok(client):
    r = client.get("/retraining/autopilot/status")
    assert r.status_code == 200
    body = r.json()
    assert "enabled" in body
    assert "configured" in body
    assert "state" in body
    assert "settings_summary" in body


def test_autopilot_run_requires_auth(app_module):
    from fastapi.testclient import TestClient

    with TestClient(app_module.app) as anon:
        r = anon.post("/retraining/autopilot/run")
    assert r.status_code == 401


def test_autopilot_run_with_auth(client):
    r = client.post("/retraining/autopilot/run")
    assert r.status_code == 200
    data = r.json()
    assert "result" in data
