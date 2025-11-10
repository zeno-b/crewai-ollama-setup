"""Configuration helpers for crewctl."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional

from dotenv import dotenv_values
import yaml

from .utils import ensure_parents, utcnow

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENV_PATH = PROJECT_ROOT / ".env"
REGISTRY_PATH = PROJECT_ROOT / "config" / "model_registry.yaml"


class ConfigError(RuntimeError):
    """Raised when a configuration operation cannot be completed."""


def load_env() -> Dict[str, str]:
    """Load configuration values, preferring .env but respecting process env overrides."""
    env_map: Dict[str, str] = {k: v for k, v in dotenv_values(ENV_PATH).items() if v is not None}
    for key, value in os.environ.items():
        if key.startswith("OLLAMA_") or key.startswith("CREWAI_"):
            env_map[key] = value
    return env_map


def update_env_var(key: str, value: str) -> None:
    """Update or append a key in the .env file."""
    ensure_parents(ENV_PATH)
    if not ENV_PATH.exists():
        ENV_PATH.write_text("", encoding="utf-8")
    lines = ENV_PATH.read_text(encoding="utf-8").splitlines()
    updated = False
    for idx, line in enumerate(lines):
        if line.strip().startswith(f"{key}="):
            lines[idx] = f"{key}={value}"
            updated = True
            break
    if not updated:
        lines.append(f"{key}={value}")
    ENV_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def load_model_registry() -> Dict[str, Any]:
    """Load the model registry YAML."""
    if not REGISTRY_PATH.exists():
        return {
            "version": 1,
            "active_model": None,
            "history": [],
        }
    with REGISTRY_PATH.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    data.setdefault("version", 1)
    data.setdefault("history", [])
    if "active_model" not in data:
        data["active_model"] = None
    return data


def save_model_registry(registry: Dict[str, Any]) -> None:
    """Persist the model registry to disk."""
    ensure_parents(REGISTRY_PATH)
    with REGISTRY_PATH.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(registry, handle, sort_keys=False)


def record_model_activation(
    model_name: str,
    digest: Optional[str],
    *,
    reason: str,
) -> Dict[str, Any]:
    """Update registry with new active model entry."""
    registry = load_model_registry()
    entry = {
        "name": model_name,
        "digest": digest,
        "activated_at": utcnow(),
        "reason": reason,
    }
    registry["active_model"] = entry
    registry.setdefault("history", []).append(entry)
    save_model_registry(registry)
    return entry


def get_previous_model() -> Optional[Dict[str, Any]]:
    """Return the model entry prior to the currently active one."""
    registry = load_model_registry()
    history = registry.get("history") or []
    if len(history) < 2:
        return None
    return history[-2]


def find_history_entry(model_name: str) -> Optional[Dict[str, Any]]:
    """Find the most recent history entry for a model."""
    registry = load_model_registry()
    history = registry.get("history") or []
    for entry in reversed(history):
        if entry.get("name") == model_name:
            return entry
    return None


def ensure_registry() -> None:
    """Ensure the registry file exists on disk."""
    if REGISTRY_PATH.exists():
        return
    save_model_registry(
        {
            "version": 1,
            "active_model": None,
            "history": [],
        }
    )
