"""Utility helpers for the crewctl CLI."""

from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


def slugify(value: str) -> str:
    """Normalize a human-friendly identifier for filesystem usage."""
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", value).strip("-").lower()
    return slug or "agent"


def utcnow() -> str:
    """Return an ISO-8601 timestamp in UTC with second precision."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def ensure_parents(path: Path) -> None:
    """Ensure parent directories exist."""
    path.parent.mkdir(parents=True, exist_ok=True)


def pluralize(noun: str, count: int) -> str:
    """Simple pluralization helper for messages."""
    return noun if count == 1 else f"{noun}s"


def bullet_list(items: Iterable[str]) -> str:
    """Render items as bullet list string."""
    return "\n".join(f"- {item}" for item in items)
