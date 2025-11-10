from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = PROJECT_ROOT / "config"
MODEL_REGISTRY_PATH = CONFIG_DIR / "model_registry.yaml"
ENV_PATH = PROJECT_ROOT / ".env"
STATE_DIR = PROJECT_ROOT / "deployment" / "state"

DEFAULT_MODEL_REGISTRY = {
    "default_model": "llama2:latest",
    "history": ["llama2:latest"],
    "models": {
        "llama2": {
            "versions": [],
        }
    },
}


def ensure_runtime() -> None:
    """Make sure configuration directories and files exist."""
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    if os.name != "nt":
        for path in (CONFIG_DIR, STATE_DIR):
            try:
                os.chmod(path, 0o750)
            except OSError:
                pass
    if not MODEL_REGISTRY_PATH.exists():
        save_model_registry(DEFAULT_MODEL_REGISTRY)


def load_model_registry() -> Dict[str, Any]:
    """Load the model registry YAML file."""
    ensure_runtime()
    if MODEL_REGISTRY_PATH.exists():
        with MODEL_REGISTRY_PATH.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
    else:
        data = {}
    return _normalise_registry(data)


def save_model_registry(data: Dict[str, Any]) -> None:
    """Persist the model registry to disk."""
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    with MODEL_REGISTRY_PATH.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle, sort_keys=False)


def record_model_version(
    tag: str,
    digest: Optional[str],
    source: str,
    metadata: Optional[Dict[str, Any]] = None,
    *,
    set_default: bool = False,
) -> None:
    """
    Record (or update) a model version entry in the registry.
    """
    registry = load_model_registry()
    base_name = tag.split(":", 1)[0]
    models_map = registry.setdefault("models", {})
    model_entry = models_map.setdefault(base_name, {}).setdefault("versions", [])

    payload = {
        "tag": tag,
        "digest": digest,
        "source": source,
        "metadata": metadata or {},
        "created_at": datetime.now(timezone.utc).isoformat(),
    }

    for existing in model_entry:
        if existing.get("tag") == tag:
            existing.update(payload)
            break
    else:
        model_entry.append(payload)

    history = registry.setdefault("history", [])
    if not history or history[-1] != tag:
        history.append(tag)

    if set_default:
        registry["default_model"] = tag
        save_model_registry(registry)
        update_env_var("OLLAMA_MODEL", tag)
    else:
        save_model_registry(registry)


def set_default_model(tag: str, *, add_to_history: bool = True) -> None:
    """Set the default model and optionally append to history."""
    registry = load_model_registry()
    registry["default_model"] = tag
    if add_to_history:
        history = registry.setdefault("history", [])
        if not history or history[-1] != tag:
            history.append(tag)
    save_model_registry(registry)
    update_env_var("OLLAMA_MODEL", tag)


def get_default_model() -> str:
    """Return the currently configured default model."""
    registry = load_model_registry()
    return registry.get("default_model", DEFAULT_MODEL_REGISTRY["default_model"])


def get_history() -> list[str]:
    registry = load_model_registry()
    return registry.get("history", [])


def resolve_ollama_base_url(override: Optional[str] = None) -> str:
    """Determine the Ollama base URL from CLI options, environment, or defaults."""
    if override:
        return override.rstrip("/")
    env_value = get_env_value("OLLAMA_BASE_URL")
    if env_value:
        return env_value.rstrip("/")
    return "http://localhost:11434"


def get_env_value(key: str, default: Optional[str] = None) -> Optional[str]:
    """Read a value from the operating environment or the .env file."""
    if key in os.environ:
        return os.environ[key]
    if ENV_PATH.exists():
        for raw_line in ENV_PATH.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            current_key, value = line.split("=", 1)
            if current_key.strip() == key:
                return value.strip()
    return default


def update_env_var(key: str, value: str) -> None:
    """Update (or append) a key/value pair in the .env file."""
    ENV_PATH.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    if ENV_PATH.exists():
        lines = ENV_PATH.read_text(encoding="utf-8").splitlines()
    replaced = False
    for idx, raw_line in enumerate(lines):
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        current_key, _ = line.split("=", 1)
        if current_key.strip() == key:
            lines[idx] = f"{key}={value}"
            replaced = True
            break
    if not replaced:
        lines.append(f"{key}={value}")
    ENV_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    os.environ[key] = value


def _normalise_registry(data: Dict[str, Any]) -> Dict[str, Any]:
    """Ensure required keys exist in the model registry structure."""
    if not data:
        data = {}
    if "default_model" not in data:
        data["default_model"] = DEFAULT_MODEL_REGISTRY["default_model"]
    if "history" not in data or not isinstance(data["history"], list):
        data["history"] = list(DEFAULT_MODEL_REGISTRY["history"])
    if "models" not in data or not isinstance(data["models"], dict):
        data["models"] = {}
    # Merge defaults without overwriting user data
    for name, entry in DEFAULT_MODEL_REGISTRY["models"].items():
        data.setdefault("models", {}).setdefault(name, {}).setdefault("versions", entry["versions"][:])
    return data
