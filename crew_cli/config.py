from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from .utils import (
    DeploymentError,
    dump_json,
    dump_yaml,
    ensure_directory,
    load_json,
    load_yaml,
    platform_key,
    project_root,
)


def _config_path() -> Path:
    return project_root() / "deployment" / "config.yml"


def _checksum_path() -> Path:
    return project_root() / "deployment" / "ollama_checksums.json"


def _state_path() -> Path:
    return project_root() / "deployment" / "state.json"


def _cli_config() -> Dict[str, Any]:
    config = get_config()
    if "cli" not in config:
        raise DeploymentError("Missing 'cli' section in deployment/config.yml")
    return config["cli"]


def get_config() -> Dict[str, Any]:
    """Load the deployment configuration."""
    config = load_yaml(_config_path(), default={})
    if not config:
        raise DeploymentError("Deployment configuration is empty or missing.")
    return config


def get_checksum_data() -> Dict[str, Any]:
    """Load checksum data for Ollama releases."""
    data = load_json(_checksum_path(), default={})
    return data


def get_state() -> Dict[str, Any]:
    """Load deployment state."""
    return load_json(_state_path(), default={})


def write_state(state: Dict[str, Any]) -> None:
    """Persist deployment state."""
    dump_json(_state_path(), state)


def model_registry_path() -> Path:
    """Return the path to the model registry file."""
    cli = _cli_config()
    return project_root() / cli["model_registry"]


def agent_registry_path() -> Path:
    """Return path to the agent registry file."""
    cli = _cli_config()
    return project_root() / cli["agent_registry"]


def lrm_directory() -> Path:
    """Return path to the LRM archive directory."""
    cli = _cli_config()
    path = project_root() / cli["lrm_directory"]
    ensure_directory(path)
    return path


def load_model_registry() -> Dict[str, Any]:
    """Load the model registry file."""
    data = load_yaml(model_registry_path(), default={})
    if not data:
        data = {
            "active_model": None,
            "models": {},
            "history": [],
        }
    return data


def save_model_registry(data: Dict[str, Any]) -> None:
    """Persist the model registry."""
    dump_yaml(model_registry_path(), data)


def load_agent_registry() -> Dict[str, Any]:
    """Load the agent registry."""
    data = load_yaml(agent_registry_path(), default={})
    if not data:
        data = {"agents": []}
    return data


def save_agent_registry(data: Dict[str, Any]) -> None:
    """Persist agent registry data."""
    dump_yaml(agent_registry_path(), data)


def python_venv_path() -> Path:
    """Return configured virtual environment path."""
    config = get_config()
    python_cfg = config.get("python", {})
    venv_path = python_cfg.get("venv_path", ".venv")
    return project_root() / venv_path


def ollama_install_strategy() -> str:
    """Return configured installation strategy."""
    config = get_config()
    ollama_cfg = config.get("ollama", {})
    return ollama_cfg.get("install_strategy", "system")


def ollama_release_entry() -> Optional[Dict[str, Any]]:
    """Return release entry for current platform."""
    config = get_config()
    version = config.get("ollama", {}).get("version")
    if not version:
        return None
    checksum_data = get_checksum_data()
    releases = checksum_data.get(version, {})
    return releases.get(platform_key())


def agent_template_path() -> Path:
    cli = _cli_config()
    return project_root() / cli["agent_template"]


def modelfile_template_path() -> Path:
    cli = _cli_config()
    return project_root() / cli["modelfile_template"]

