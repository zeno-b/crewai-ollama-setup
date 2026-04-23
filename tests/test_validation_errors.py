"""Structured validation errors (clear client-facing output)."""

from __future__ import annotations


def test_validation_error_shape_on_dataset_create(client):
    r = client.post(
        "/datasets",
        json={
            "name": "bad-format",
            "content": "x",
            "format": "not-a-format",
            "overwrite": True,
        },
    )
    assert r.status_code == 422
    body = r.json()
    assert "detail" in body
    assert isinstance(body["detail"], list)
    assert len(body["detail"]) >= 1
    err0 = body["detail"][0]
    assert "type" in err0 or "msg" in err0 or "loc" in err0
