"""Dataset API: validation, persistence, and error mapping."""

from __future__ import annotations


def test_create_dataset_201_and_list(client):
    payload = {
        "name": "unit-test-ds",
        "description": "pytest",
        "tags": ["t1"],
        "format": "text",
        "content": "hello world\n",
        "overwrite": False,
    }
    r = client.post("/datasets", json=payload)
    assert r.status_code == 201
    created = r.json()
    assert created["name"] == "unit-test-ds"
    assert created["format"] == "text"
    assert created["size_bytes"] > 0

    listed = client.get("/datasets").json()
    names = {d["name"] for d in listed["datasets"]}
    assert "unit-test-ds" in names


def test_create_dataset_duplicate_400(client):
    body = {
        "name": "dup-ds",
        "content": "a",
        "format": "text",
        "overwrite": False,
    }
    assert client.post("/datasets", json=body).status_code == 201
    r2 = client.post("/datasets", json=body)
    assert r2.status_code == 400
    assert "already exists" in r2.json()["detail"].lower()


def test_get_dataset_404(client):
    r = client.get("/datasets/does-not-exist-xyz")
    assert r.status_code == 404


def test_get_dataset_invalid_name_400(client):
    r = client.get("/datasets/@@@")
    assert r.status_code == 400


def test_delete_dataset_404(client):
    r = client.delete("/datasets/missing-dataset-12345")
    assert r.status_code == 404
