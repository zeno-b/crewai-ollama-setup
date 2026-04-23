"""Retraining jobs: scheduling, listing, bounded query params."""

from __future__ import annotations


def _sample_dataset(client):
    client.post(
        "/datasets",
        json={
            "name": "rt-sample",
            "content": "line1\nline2\n",
            "format": "text",
            "overwrite": True,
        },
    )


def test_create_retraining_job_202_and_get(client):
    _sample_dataset(client)
    job = {
        "model_name": "test-model",
        "base_model": "llama2:7b",
        "dataset_name": "rt-sample",
        "timeout": 120,
    }
    r = client.post("/retraining/jobs", json=job)
    assert r.status_code == 202
    body = r.json()
    assert body["status"] == "queued"
    job_id = body["job_id"]
    assert job_id.startswith("job_")

    g = client.get(f"/retraining/jobs/{job_id}")
    assert g.status_code == 200
    assert g.json()["job_id"] == job_id


def test_retraining_missing_dataset_404(client):
    r = client.post(
        "/retraining/jobs",
        json={
            "model_name": "m",
            "base_model": "llama2:7b",
            "dataset_name": "no-such-dataset",
            "timeout": 120,
        },
    )
    assert r.status_code == 404


def test_retraining_distill_requires_teacher_422(client):
    _sample_dataset(client)
    r = client.post(
        "/retraining/jobs",
        json={
            "model_name": "m",
            "base_model": "llama2:7b",
            "dataset_name": "rt-sample",
            "job_type": "distill",
            "modelfile_template": "FROM {{BASE_MODEL}}\nMESSAGE user hi\n",
            "timeout": 120,
        },
    )
    assert r.status_code == 422
    assert "teacher_model" in r.json()["detail"].lower()


def test_retraining_distill_requires_template_400(client):
    _sample_dataset(client)
    r = client.post(
        "/retraining/jobs",
        json={
            "model_name": "m",
            "base_model": "llama2:7b",
            "dataset_name": "rt-sample",
            "job_type": "distill",
            "teacher_model": "teacher:latest",
            "timeout": 120,
        },
    )
    assert r.status_code == 400
    detail = r.json()["detail"].lower()
    assert "modelfile_template" in detail or "template_name" in detail


def test_retraining_timeout_validation_422(client):
    _sample_dataset(client)
    r = client.post(
        "/retraining/jobs",
        json={
            "model_name": "m",
            "base_model": "llama2:7b",
            "dataset_name": "rt-sample",
            "timeout": 30,
        },
    )
    assert r.status_code == 422


def test_list_jobs_limit_bounds_422(client):
    assert client.get("/retraining/jobs?limit=0").status_code == 422
    assert client.get("/retraining/jobs?limit=501").status_code == 422


def test_job_logs_tail_bounds_422(client):
    jid = "job_" + "a" * 32
    assert client.get(f"/retraining/jobs/{jid}/logs?tail=0").status_code == 422


def test_invalid_job_id_400(client):
    assert client.get("/retraining/jobs/not-a-job-id").status_code == 400
    assert client.get("/retraining/jobs/job_/logs").status_code == 400


def test_invalid_job_id_logs_malformed_job_segment_400(client):
    assert client.get("/retraining/jobs/job_not_hex/logs").status_code == 400
