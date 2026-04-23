"""
Test bootstrap: isolate filesystem, stub Redis/Ollama, pin httpx, mock Ollama HTTP API.
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional
from unittest.mock import AsyncMock

import httpx
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

_TEST_ROOT = tempfile.mkdtemp(prefix="crewai-pytest-")

os.environ.setdefault("ENVIRONMENT", "testing")
os.environ["DATASET_DIR"] = os.path.join(_TEST_ROOT, "datasets")
os.environ["RETRAINING_DIR"] = os.path.join(_TEST_ROOT, "retraining")
os.environ["OLLAMA_BASE_URL"] = os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:9")
os.environ["REDIS_URL"] = os.environ.get("REDIS_URL", "redis://127.0.0.1:9/0")
os.environ["SECRET_KEY"] = "test-secret-key-for-pytest-only-32chars!!"
os.environ["API_BEARER_TOKEN"] = "test-api-token"


class _FakeOllamaTagsResponse:
    def __init__(self) -> None:
        self.status_code = 200

    def raise_for_status(self) -> None:
        return None

    def json(self) -> Dict[str, Any]:
        return {
            "models": [
                {
                    "name": "llama2:7b",
                    "size": 3825819519,
                    "digest": "fe938a131f40e6f6d40083c9f0f430a515233eb2edaa6d72eb85c50d64f2300e",
                    "modified_at": "2024-01-01T00:00:00Z",
                }
            ]
        }


class FakeHttpxAsyncClient:
    """Minimal AsyncClient: GET /api/tags for list_models; POST unused in default tests."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self._timeout = kwargs.get("timeout")

    async def __aenter__(self) -> "FakeHttpxAsyncClient":
        return self

    async def __aexit__(self, *args: Any) -> None:
        return None

    async def get(self, url: str, **kwargs: Any) -> _FakeOllamaTagsResponse:
        if url.endswith("/api/tags"):
            return _FakeOllamaTagsResponse()
        raise httpx.HTTPError(f"Unexpected URL in test fake: {url}")

    async def post(self, *args: Any, **kwargs: Any) -> Any:
        raise httpx.HTTPError("POST not configured in FakeHttpxAsyncClient")


httpx.AsyncClient = FakeHttpxAsyncClient  # type: ignore[misc, assignment]

import redis.asyncio as redis_asyncio  # noqa: E402


class FakeRedis:
    """Minimal async Redis stub matching decode_responses=True string I/O."""

    def __init__(self) -> None:
        self._store: Dict[str, str] = {}

    async def ping(self) -> bool:
        return True

    async def close(self) -> None:
        return None

    async def set(self, key: str, value: str, ex: Optional[int] = None) -> None:
        self._store[key] = value

    async def get(self, key: str) -> Optional[str]:
        return self._store.get(key)


def _fake_redis_from_url(*_args: Any, **_kwargs: Any) -> FakeRedis:
    return FakeRedis()


redis_asyncio.from_url = _fake_redis_from_url  # type: ignore[method-assign]

import langchain_community.llms as lc_llms  # noqa: E402
from langchain_core.language_models.fake import FakeListLLM  # noqa: E402


def _fake_ollama_factory(*_args: Any, **_kwargs: Any) -> FakeListLLM:
    return FakeListLLM(responses=["test-llm-response"])


lc_llms.Ollama = _fake_ollama_factory  # type: ignore[misc, assignment]

import main as main_module  # noqa: E402


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    shutil.rmtree(_TEST_ROOT, ignore_errors=True)


@pytest.fixture
def client():
    from fastapi.testclient import TestClient

    headers = {"Authorization": f"Bearer {os.environ['API_BEARER_TOKEN']}"}
    with TestClient(main_module.app, headers=headers) as test_client:
        rc = main_module.redis_client
        if rc is not None and hasattr(rc, "_store"):
            rc._store.clear()
        mgr = main_module.retraining_manager
        if mgr is not None:
            mgr.run_job = AsyncMock(return_value=None)  # type: ignore[method-assign]
        yield test_client


@pytest.fixture
def app_module():
    return main_module
