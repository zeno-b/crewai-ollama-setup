#!/usr/bin/env python3
"""
Unified deployment script for CrewAI + Ollama.

This script bootstraps a virtual environment, installs dependencies,
configures a secure CLI (crewctl), and optionally manages Ollama binaries
for multiple platforms.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import platform
import shutil
import stat
import subprocess
import sys
import tarfile
import urllib.request
import venv
import zipfile
from pathlib import Path
from typing import Any, Dict, Optional

LOG = logging.getLogger("deploy")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "deployment" / "config.yml"
CHECKSUM_PATH = PROJECT_ROOT / "deployment" / "ollama_checksums.json"
STATE_PATH = PROJECT_ROOT / "deployment" / "state.json"

DEFAULT_CONFIG: Dict[str, Any] = {
    "python": {
        "min_version": "3.10",
        "venv_path": ".venv",
        "requirements_file": "requirements.txt",
        "pip_flags": ["--upgrade", "--no-cache-dir", "--disable-pip-version-check"],
        "ensure_pip": True,
    },
    "ollama": {
        "install_strategy": "system",
        "version": "0.3.14",
        "listen_host": "127.0.0.1",
        "listen_port": 11434,
        "binary_directory": "runtime/ollama",
        "artifacts_directory": "deployment/artifacts",
        "managed_binary_name": "ollama",
        "verify_checksums": True,
        "default_models": ["llama3", "mistral"],
    },
    "cli": {
        "entrypoint_name": "crewctl",
        "agent_template": "deployment/templates/agent.py.j2",
        "modelfile_template": "deployment/templates/modelfile.j2",
        "model_registry": "config/model_registry.yaml",
        "agent_registry": "config/agent_registry.yaml",
        "lrm_directory": "models/lrms",
        "history_limit": 20,
    },
    "security": {
        "restrict_file_permissions": True,
        "allowed_users": ["current"],
        "validate_downloads": True,
    },
    "logging": {"level": "INFO"},
}


class DeploymentError(RuntimeError):
    """Raised when deployment operations fail."""


def project_root() -> Path:
    return PROJECT_ROOT


def ensure_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def compute_sha256(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8192), b""):
            sha.update(chunk)
    return sha.hexdigest()


def make_executable(path: Path) -> None:
    if platform.system() != "Windows":
        current_mode = path.stat().st_mode
        path.chmod(current_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def sanitize_permissions(path: Path) -> None:
    if platform.system() == "Windows":
        return
    if path.exists():
        path.chmod(stat.S_IRUSR | stat.S_IWUSR)


def _parse_scalar(value: str) -> Any:
    token = value.strip()
    if token in {"null", "Null", "NULL", "~"}:
        return None
    if token in {"true", "True", "TRUE"}:
        return True
    if token in {"false", "False", "FALSE"}:
        return False
    if token.startswith('"') and token.endswith('"'):
        return token[1:-1]
    if token.startswith("'") and token.endswith("'"):
        return token[1:-1]
    try:
        if "." in token:
            return float(token)
        return int(token)
    except ValueError:
        return token


def _load_yaml_minimal(path: Path) -> Any:
    lines: list[tuple[int, str]] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        stripped = raw_line.split("#", 1)[0].rstrip()
        if not stripped:
            continue
        indent = len(raw_line) - len(raw_line.lstrip(" "))
        lines.append((indent, stripped))

    root: Any = {}
    stack: list[tuple[int, Any]] = [(-1, root)]

    idx = 0
    while idx < len(lines):
        indent, content = lines[idx]
        while stack and indent <= stack[-1][0]:
            stack.pop()
        parent = stack[-1][1]

        if content.startswith("- "):
            value = content[2:].strip()
            if not isinstance(parent, list):
                raise DeploymentError("Invalid YAML structure: list item without list context.")
            if value == "":
                # Nested structure
                next_is_list = (
                    idx + 1 < len(lines) and lines[idx + 1][0] > indent and lines[idx + 1][1].startswith("- ")
                )
                container: Any = [] if next_is_list else {}
                parent.append(container)
                stack.append((indent, container))
            else:
                parent.append(_parse_scalar(value))
        else:
            key, _, value = content.partition(":")
            key = key.strip()
            value = value.strip()
            if value == "":
                next_is_list = (
                    idx + 1 < len(lines) and lines[idx + 1][0] > indent and lines[idx + 1][1].startswith("- ")
                )
                container = [] if next_is_list else {}
                if isinstance(parent, list):
                    parent.append({key: container})
                    stack.append((indent, container))
                else:
                    parent[key] = container
                    stack.append((indent, container))
            else:
                if isinstance(parent, list):
                    parent.append({key: _parse_scalar(value)})
                else:
                    parent[key] = _parse_scalar(value)
        idx += 1

    return root


def _format_scalar(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    text = str(value)
    if text == "" or any(ch in text for ch in [":", "#", "-", "{", "}", "[", "]", " "]):
        return f'"{text}"'
    return text


def _dump_yaml_minimal(data: Any, indent: int = 0) -> str:
    spacing = " " * indent
    if isinstance(data, dict):
        lines = []
        for key, value in data.items():
            if isinstance(value, (dict, list)):
                lines.append(f"{spacing}{key}:")
                lines.append(_dump_yaml_minimal(value, indent + 2))
            else:
                lines.append(f"{spacing}{key}: {_format_scalar(value)}")
        return "\n".join(lines)
    if isinstance(data, list):
        lines = []
        for item in data:
            if isinstance(item, (dict, list)):
                lines.append(f"{spacing}-")
                lines.append(_dump_yaml_minimal(item, indent + 2))
            else:
                lines.append(f"{spacing}- {_format_scalar(item)}")
        return "\n".join(lines)
    return f"{spacing}{_format_scalar(data)}"


def load_yaml_file(path: Path, default: Optional[Any] = None) -> Any:
    if not path.exists():
        return default
    try:
        import yaml  # type: ignore
    except ImportError:
        try:
            data = _load_yaml_minimal(path)
        except DeploymentError as exc:
            raise DeploymentError(f"Failed to parse YAML without PyYAML ({path}): {exc}") from exc
    else:
        with path.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle)
    return default if data is None else data


def dump_yaml_file(path: Path, data: Any) -> None:
    ensure_directory(path.parent)
    try:
        import yaml  # type: ignore
    except ImportError:
        rendered = _dump_yaml_minimal(data)
    else:
        rendered = yaml.safe_dump(data, sort_keys=False)
    path.write_text(rendered.rstrip() + "\n", encoding="utf-8")
    sanitize_permissions(path)


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    result = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def load_config() -> Dict[str, Any]:
    file_config = load_yaml_file(CONFIG_PATH, default={})
    if not file_config:
        return dict(DEFAULT_CONFIG)
    if not isinstance(file_config, dict):
        raise DeploymentError(f"Configuration file {CONFIG_PATH} must contain a mapping.")
    return _deep_merge(DEFAULT_CONFIG, file_config)


def load_checksums() -> Dict[str, Any]:
    if not CHECKSUM_PATH.exists():
        return {}
    with CHECKSUM_PATH.open("r", encoding="utf-8") as handle:
        try:
            return json.load(handle)
        except json.JSONDecodeError as exc:
            raise DeploymentError(f"Invalid JSON in {CHECKSUM_PATH}: {exc}") from exc


def write_state(state: Dict[str, Any]) -> None:
    ensure_directory(STATE_PATH.parent)
    STATE_PATH.write_text(json.dumps(state, indent=2), encoding="utf-8")
    sanitize_permissions(STATE_PATH)


def python_venv_path(config: Dict[str, Any]) -> Path:
    venv_relative = config.get("python", {}).get("venv_path", ".venv")
    return project_root() / venv_relative


def model_registry_path(config: Dict[str, Any]) -> Path:
    path = config.get("cli", {}).get("model_registry", "config/model_registry.yaml")
    return project_root() / path


def agent_registry_path(config: Dict[str, Any]) -> Path:
    path = config.get("cli", {}).get("agent_registry", "config/agent_registry.yaml")
    return project_root() / path


def ollama_install_strategy(config: Dict[str, Any]) -> str:
    return config.get("ollama", {}).get("install_strategy", "system")


def _platform_key() -> str:
    system = platform.system().lower()
    machine = platform.machine().lower()
    if system.startswith("linux"):
        if "aarch64" in machine or "arm64" in machine:
            return "linux-arm64"
        return "linux-amd64"
    if system.startswith("darwin"):
        if "arm64" in machine or "apple" in machine:
            return "macos-arm64"
        return "macos-amd64"
    if system.startswith("windows"):
        return "windows-amd64"
    return f"{system}-{machine}"


def ollama_release_entry(config: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    version = config.get("ollama", {}).get("version")
    if not version:
        return None
    checksums = load_checksums()
    releases = checksums.get(version, {})
    return releases.get(_platform_key())


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)-8s | %(message)s",
    )


def check_python_version(min_version: str) -> None:
    current = sys.version_info
    major, minor = map(int, min_version.split(".", maxsplit=1))
    if current < (major, minor):
        raise DeploymentError(
            f"Python {min_version}+ required, found {current.major}.{current.minor}."
        )


def create_virtualenv(path: Path, recreate: bool = False) -> None:
    if recreate and path.exists():
        LOG.info("Removing existing virtual environment at %s", path)
        shutil.rmtree(path)
    if path.exists():
        LOG.info("Virtual environment already present at %s", path)
        return
    LOG.info("Creating virtual environment at %s", path)
    builder = venv.EnvBuilder(with_pip=True, clear=False, upgrade=False)
    builder.create(str(path))


def venv_python(venv_path: Path) -> Path:
    if platform.system() == "Windows":
        return venv_path / "Scripts" / "python.exe"
    return venv_path / "bin" / "python"


def install_requirements(
    python_path: Path,
    requirements_file: Path,
    flags: Optional[list[str]] = None,
) -> None:
    if not requirements_file.exists():
        raise DeploymentError(f"Requirements file not found: {requirements_file}")
    command = [str(python_path), "-m", "pip", "install"]
    if flags:
        command.extend(flags)
    command.extend(["-r", str(requirements_file)])
    LOG.info("Installing dependencies: %s", " ".join(command))
    result = subprocess.run(command, check=False)
    if result.returncode != 0:
        raise DeploymentError("Failed to install Python dependencies.")


def ensure_directories(config: Dict[str, Any]) -> None:
    cli_cfg = config.get("cli", {})
    ollama_cfg = config.get("ollama", {})
    directories = [
        project_root() / "agents",
        project_root() / "models" / "training",
        project_root() / cli_cfg.get("lrm_directory", "models/lrms"),
        project_root() / "logs",
        project_root() / ollama_cfg.get("artifacts_directory", "deployment/artifacts"),
    ]
    for path in directories:
        ensure_directory(path)
        LOG.debug("Ensured directory %s", path)


def initialize_registries(config: Dict[str, Any]) -> None:
    default_models = config.get("ollama", {}).get("default_models", [])

    model_registry_file = model_registry_path(config)
    model_registry = load_yaml_file(model_registry_file, default={})
    if not model_registry:
        LOG.info("Creating model registry at %s", model_registry_file)
        model_registry = {
            "active_model": default_models[0] if default_models else None,
            "models": {},
            "history": [],
        }
        dump_yaml_file(model_registry_file, model_registry)

    agent_registry_file = agent_registry_path(config)
    agent_registry = load_yaml_file(agent_registry_file, default={})
    if not agent_registry:
        LOG.info("Creating agent registry at %s", agent_registry_file)
        agent_registry = {"agents": []}
        dump_yaml_file(agent_registry_file, agent_registry)


def write_cli_entrypoints(venv_path: Path, entrypoint_name: str) -> Dict[str, str]:
    bin_dir = venv_path / ("Scripts" if platform.system() == "Windows" else "bin")
    ensure_directory(bin_dir)

    entrypoints: Dict[str, str] = {}

    if platform.system() == "Windows":
        script_path = bin_dir / f"{entrypoint_name}.cmd"
        content = (
            "@echo off\n"
            "setlocal enabledelayedexpansion\n"
            "set SCRIPT_DIR=%~dp0\n"
            "set PROJECT_ROOT=%SCRIPT_DIR%..\\..\\\n"
            "set PYTHONPATH=%PROJECT_ROOT%;%PYTHONPATH%\n"
            f"\"%SCRIPT_DIR%python.exe\" -m crew_cli %*\n"
        )
        script_path.write_text(content, encoding="utf-8")
        entrypoints["cmd"] = str(script_path)
    else:
        script_path = bin_dir / entrypoint_name
        content = (
            "#!/usr/bin/env bash\n"
            "set -euo pipefail\n"
            'SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"\n'
            'PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"\n'
            'export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"\n'
            '"${SCRIPT_DIR}/python" -m crew_cli "$@"\n'
        )
        script_path.write_text(content, encoding="utf-8")
        make_executable(script_path)
        entrypoints["posix"] = str(script_path)

    return entrypoints


def download_with_validation(url: str, destination: Path, expected_sha256: Optional[str]) -> None:
    LOG.info("Downloading %s -> %s", url, destination)
    with urllib.request.urlopen(url) as response, destination.open("wb") as handle:
        shutil.copyfileobj(response, handle)
    if expected_sha256:
        actual = compute_sha256(destination)
        if actual.lower() != expected_sha256.lower():
            destination.unlink(missing_ok=True)
            raise DeploymentError(
                f"Checksum mismatch for {url}. Expected {expected_sha256}, got {actual}."
            )


def extract_archive(archive_path: Path, destination: Path) -> None:
    ensure_directory(destination)
    if zipfile.is_zipfile(archive_path):
        with zipfile.ZipFile(archive_path, "r") as archive:
            archive.extractall(destination)
    elif tarfile.is_tarfile(archive_path):
        with tarfile.open(archive_path, "r:*") as archive:
            archive.extractall(destination)
    else:
        raise DeploymentError(f"Unsupported archive format: {archive_path.name}")


def install_ollama_managed(config: Dict[str, Any]) -> Path:
    release = ollama_release_entry()
    if not release:
        raise DeploymentError(
            "No release entry found for current platform. Update deployment/ollama_checksums.json."
        )

    verify_checksums = config.get("ollama", {}).get("verify_checksums", True)
    expected_sha = release.get("sha256")
    if verify_checksums and expected_sha in (None, "", "null"):
        raise DeploymentError(
            "Checksum missing for the configured Ollama release. "
            "Update deployment/ollama_checksums.json with a verified sha256."
        )

    url = release["url"]
    artifacts_dir = project_root() / config.get("ollama", {}).get(
        "artifacts_directory", "deployment/artifacts"
    )
    ensure_directory(artifacts_dir)
    archive_path = artifacts_dir / Path(url).name
    download_with_validation(url, archive_path, expected_sha if verify_checksums else None)

    target_dir = project_root() / config.get("ollama", {}).get(
        "binary_directory", "runtime/ollama"
    )
    ensure_directory(target_dir)
    extract_archive(archive_path, target_dir)

    candidate = None
    for name in ("ollama", "ollama.exe"):
        for item in target_dir.rglob(name):
            if item.is_file() and (platform.system() == "Windows" or os.access(item, os.X_OK)):
                candidate = item
                break
        if candidate:
            break
    if not candidate:
        raise DeploymentError("Unable to locate extracted Ollama binary.")

    if platform.system() != "Windows":
        candidate.chmod(candidate.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)

    LOG.info("Managed Ollama binary prepared at %s", candidate)
    return candidate


def resolve_system_ollama() -> Path:
    path = shutil.which("ollama")
    if not path:
        raise DeploymentError(
            "Ollama executable not found on PATH. Install Ollama or switch install_strategy to 'managed'."
        )
    LOG.info("Using system Ollama at %s", path)
    return Path(path)


def handle_ollama_installation(config: Dict[str, Any]) -> Path:
    strategy = ollama_install_strategy(config)
    if strategy == "skip":
        LOG.info("Skipping Ollama installation per configuration.")
        return Path(shutil.which("ollama") or "")
    if strategy == "system":
        return resolve_system_ollama()
    if strategy == "managed":
        return install_ollama_managed(config)
    raise DeploymentError(f"Unknown install strategy: {strategy}")


def update_state(
    config: Dict[str, Any], venv_path: Path, ollama_path: Path, entrypoints: Dict[str, str]
) -> None:
    state = {
        "python": {
            "version": sys.version,
            "venv_path": str(venv_path.relative_to(project_root())),
        },
        "ollama": {
            "binary_path": str(ollama_path) if ollama_path else None,
            "install_strategy": ollama_install_strategy(config),
            "version": config.get("ollama", {}).get("version"),
        },
        "cli": {
            "entrypoints": entrypoints,
        },
    }
    write_state(state)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Deploy CrewAI + Ollama stack.")
    parser.add_argument(
        "--recreate-venv",
        action="store_true",
        help="Recreate the virtual environment from scratch.",
    )
    parser.add_argument(
        "--skip-install",
        action="store_true",
        help="Skip installing Python dependencies.",
    )
    parser.add_argument(
        "--skip-ollama",
        action="store_true",
        help="Skip Ollama installation even if configured.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Logging level (DEBUG, INFO, WARNING, ERROR).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config()
    configure_logging(config.get("logging", {}).get("level", args.log_level))

    try:
        python_cfg = config.get("python", {})
        min_version = python_cfg.get("min_version", "3.10")
        check_python_version(min_version)

        ensure_directories(config)
        initialize_registries(config)

        venv_path = python_venv_path(config)
        create_virtualenv(venv_path, recreate=args.recreate_venv)
        python_exec = venv_python(venv_path)

        if not args.skip_install:
            pip_flags = python_cfg.get("pip_flags", [])
            requirements_file = project_root() / python_cfg.get("requirements_file", "requirements.txt")
            install_requirements(python_exec, requirements_file, pip_flags)

        entrypoint_name = config.get("cli", {}).get("entrypoint_name", "crewctl")
        entrypoints = write_cli_entrypoints(venv_path, entrypoint_name)

        ollama_path = Path()
        if not args.skip_ollama:
            ollama_path = handle_ollama_installation(config)

        update_state(config, venv_path, ollama_path, entrypoints)

        LOG.info("Deployment completed successfully.")
        if platform.system() == "Windows":
            LOG.info("Activate the environment with: %s", venv_path / "Scripts" / "activate")
            LOG.info("Use the CLI via: %s agents list", entrypoints.get("cmd"))
        else:
            LOG.info("Activate the environment with: source %s", venv_path / "bin" / "activate")
            LOG.info("Use the CLI via: %s agents list", entrypoints.get("posix"))

    except DeploymentError as exc:
        LOG.error("Deployment failed: %s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
