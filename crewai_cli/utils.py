from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path


def snake_case(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9]+", "_", value)
    return cleaned.strip("_").lower()


def pascal_case(value: str) -> str:
    parts = re.split(r"[^A-Za-z0-9]+", value)
    return "".join(part.capitalize() for part in parts if part)


def timestamp_slug() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
