"""Agent and crew endpoints: validation and error responses."""

from __future__ import annotations


def test_models_returns_placeholder_when_llm_configured(client):
    r = client.get("/models")
    assert r.status_code == 200
    data = r.json()
    assert "models" in data
    assert len(data["models"]) >= 1


def test_create_agent_success(client):
    r = client.post(
        "/create_agent",
        json={
            "name": "agent-one",
            "role": "Tester",
            "goal": "Verify API",
            "backstory": "Runs pytest",
            "tools": [],
            "verbose": False,
        },
    )
    assert r.status_code == 200
    assert "created successfully" in r.json()["message"].lower()


def test_run_crew_missing_agent_reference_400(client):
    r = client.post(
        "/run_crew",
        json={
            "agents": [
                {
                    "name": "a1",
                    "role": "R",
                    "goal": "G",
                    "backstory": "B",
                }
            ],
            "tasks": [
                {
                    "description": "Do work",
                    "expected_output": "Done",
                    "agent": "missing",
                }
            ],
            "verbose": False,
        },
    )
    assert r.status_code == 400
    assert "not found" in r.json()["detail"].lower()


def test_crew_result_invalid_id_400(client):
    assert client.get("/crew_results/crew_<invalid>").status_code == 400


def test_crew_result_not_found_404(client):
    r = client.get("/crew_results/crew_deadbeef1234567890")
    assert r.status_code == 404
