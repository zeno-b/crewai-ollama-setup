"""
Test bootstrap: isolate filesystem, stub Redis/Ollama before importing the app.
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
from unittest.mock import AsyncMock

_TEST_ROOT = tempfile.mkdtemp(prefix="crewai-pytest-")

os.environ.setdefault("ENVIRONMENT", "testing")
os.environ["DATASET_DIR"] = os.path.join(_TEST_ROOT, "datasets")
os.environ["RETRAINING_DIR"] = os.path.join(_TEST_ROOT, "retraining")
os.environ["OLLAMA_BASE_URL"] = os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:9")
os.environ["REDIS_URL"] = os.environ.get("REDIS_URL", "redis://127.0.0.1:9/0")


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


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    shutil.rmtree(_TEST_ROOT, ignore_errors=True)


# Patch external clients before application import
import redis.asyncio as redis_asyncio  # noqa: E402

redis_asyncio.from_url = _fake_redis_from_url  # type: ignore[method-assign]

import langchain_community.llms as lc_llms  # noqa: E402
from langchain_core.language_models.fake import FakeListLLM  # noqa: E402


def _fake_ollama_factory(*_args: Any, **kwargs: Any) -> FakeListLLM:
    return FakeListLLM(responses=["test-llm-response"])


lc_llms.Ollama = _fake_ollama_factory  # type: ignore[misc, assignment]

import main as main_module  # noqa: E402


@pytest.fixture
def client():
    from fastapi.testclient import TestClient

    with TestClient(main_module.app) as test_client:
        rc = main_module.redis_client
        if rc is not None and hasattr(rc, "_store"):
            rc._store.clear()
        mgr = main_module.retraining_manager
        if mgr is not None:
            # Prevent background retraining from opening real HTTP connections during tests.
            mgr.run_job = AsyncMock(return_value=None)  # type: ignore[method-assign]
        yield test_client


@pytest.fixture
def app_module():
    return main_module
