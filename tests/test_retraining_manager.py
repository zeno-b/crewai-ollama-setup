"""Unit tests for retraining.manager (filesystem + modelfile safety)."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from retraining.manager import DatasetManager, RetrainingJobManager, MAX_DATASET_CHARS


@pytest.fixture
def dataset_dir(tmp_path: Path) -> Path:
    d = tmp_path / "datasets"
    d.mkdir()
    return d


def test_dataset_name_sanitization_and_collision(dataset_dir: Path):
    dm = DatasetManager(dataset_dir)
    dm.save_dataset("My DS!", "content", None, [], "text", overwrite=False)
    with pytest.raises(ValueError, match="already exists"):
        dm.save_dataset("My DS!", "x", None, [], "text", overwrite=False)
    dm.save_dataset("My DS!", "new", None, [], "text", overwrite=True)


def test_modelfile_rejects_oversized_default_template(dataset_dir: Path):
    dm = DatasetManager(dataset_dir)
    huge = "x" * (MAX_DATASET_CHARS + 1)
    dm.save_dataset("huge", huge, None, [], "text", overwrite=True)

    jobs_dir = dataset_dir.parent / "jobs_root"
    mgr = RetrainingJobManager(
        base_dir=jobs_dir,
        dataset_manager=dm,
        redis_client=None,
        result_ttl=60,
        ollama_base_url="http://127.0.0.1:9",
        metrics=None,
    )
    payload = {
        "model_name": "m",
        "base_model": "base",
        "dataset_name": "huge",
        "parameters": {},
    }
    with pytest.raises(ValueError, match="too large"):
        mgr._render_modelfile(payload, {"content": huge, "format": "text"})


@pytest.mark.asyncio
async def test_run_job_marks_failed_on_http_error(dataset_dir: Path):
    dm = DatasetManager(dataset_dir)
    dm.save_dataset("small", "hello", None, [], "text", overwrite=True)

    jobs_dir = dataset_dir.parent / "jobs_root2"
    mgr = RetrainingJobManager(
        base_dir=jobs_dir,
        dataset_manager=dm,
        redis_client=None,
        result_ttl=60,
        ollama_base_url="http://127.0.0.1:9",
        metrics=None,
    )
    rec = await mgr.create_job(
        {
            "model_name": "m",
            "base_model": "b",
            "dataset_name": "small",
            "stream": False,
            "parameters": {},
        }
    )
    job_id = rec["job_id"]

    class Resp:
        def raise_for_status(self) -> None:
            raise RuntimeError("connection refused")

        def json(self) -> dict:
            return {}

    with patch("retraining.manager.httpx.AsyncClient") as mock_client_cls:
        instance = MagicMock()
        instance.__aenter__.return_value = instance
        instance.__aexit__.return_value = None
        instance.post = AsyncMock(return_value=Resp())
        mock_client_cls.return_value = instance

        await mgr.run_job(
            job_id,
            {
                "model_name": "m",
                "base_model": "b",
                "dataset_name": "small",
                "stream": False,
                "parameters": {},
            },
            timeout=5,
        )

    final = json.loads(mgr._job_status_path(job_id).read_text(encoding="utf-8"))
    assert final["status"] == "failed"
