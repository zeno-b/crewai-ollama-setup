"""Utility helpers for the crewctl CLI."""

from __future__ import annotations

import json
import os
import re
import subprocess
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from rich.console import Console

console = Console()


class CrewCtlError(RuntimeError):
    """CLI level exception."""


def resolve_project_root(start: Optional[Path] = None) -> Path:
    """Resolve the CrewAI project root directory."""
    env_root = os.getenv("CREWCTL_ROOT")
    if env_root:
        root = Path(env_root).expanduser().resolve()
        if (root / "config" / "settings.py").exists():
            return root
        raise CrewCtlError(
            f"CREWCTL_ROOT={env_root} does not look like a CrewAI project (missing config/settings.py)."
        )

    start_path = (start or Path.cwd()).resolve()
    for candidate in [start_path] + list(start_path.parents):
        if (candidate / "config" / "settings.py").exists() and (candidate / "pyproject.toml").exists():
            return candidate
    raise CrewCtlError(
        "Unable to locate the CrewAI project root. Run from within the project directory or set CREWCTL_ROOT."
    )


def run_command(
    cmd: Iterable[str],
    *,
    cwd: Optional[Path] = None,
    check: bool = True,
    env: Optional[Dict[str, str]] = None,
) -> subprocess.CompletedProcess:
    """Run a subprocess command and capture output."""
    cmd_list = list(cmd)
    console.log(f"[bold green]$ {' '.join(cmd_list)}")
    try:
        result = subprocess.run(
            cmd_list,
            cwd=str(cwd) if cwd else None,
            env=env,
            text=True,
            capture_output=True,
            check=check,
        )
    except FileNotFoundError as exc:
        raise CrewCtlError(f"Required command not found: {cmd_list[0]}") from exc
    if check and result.returncode != 0:
        raise CrewCtlError(
            f"Command failed ({result.returncode}): {' '.join(cmd_list)}\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )
    return result


def ensure_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_json(path: Path, default: Optional[Dict] = None) -> Dict:
    if not path.exists():
        return default or {}
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise CrewCtlError(f"Failed to parse JSON file {path}: {exc}") from exc


def save_json(path: Path, data: Dict) -> None:
    ensure_directory(path.parent)
    path.write_text(json.dumps(data, indent=2))


def snake_case(value: str) -> str:
    value = value.strip()
    value = re.sub(r"[\s\-]+", "_", value)
    value = re.sub(r"[^0-9a-zA-Z_]", "", value)
    value = re.sub(r"_{2,}", "_", value)
    return value.lower()


def title_case(value: str) -> str:
    return " ".join(part.capitalize() for part in re.split(r"[\s_\-]+", value) if part)


def camel_case(value: str) -> str:
    parts = [part.capitalize() for part in re.split(r"[\s_\-]+", value) if part]
    return "".join(parts) or "Agent"


def update_env_file(path: Path, updates: Dict[str, str]) -> None:
    ensure_directory(path.parent)
    existing_lines: List[str] = []
    if path.exists():
        existing_lines = path.read_text().splitlines()

    env_map: Dict[str, str] = {}
    for line in existing_lines:
        if line.strip().startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        env_map[key.strip()] = value.strip()

    env_map.update({k: v for k, v in updates.items() if v is not None})

    lines = [f"{key}={value}" for key, value in env_map.items()]
    path.write_text("\n".join(lines) + "\n")


def read_env_value(path: Path, key: str) -> Optional[str]:
    if not path.exists():
        return None
    for line in path.read_text().splitlines():
        if line.strip().startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        if k.strip() == key:
            return v.strip()
    return None


def format_bytes(size: int) -> str:
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if size < 1024:
            return f"{size:.2f}{unit}"
        size /= 1024
    return f"{size:.2f}PB"
