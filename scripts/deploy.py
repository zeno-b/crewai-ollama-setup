#!/usr/bin/env python3
"""
Cross-platform deployment utility for CrewAI + Ollama.

This script provisions a self-contained environment with the following goals:
  * create an isolated virtual environment for Python dependencies
  * scaffold secure configuration files (.env, model registry, deployment state)
  * optionally generate hardened Docker Compose manifests for Ollama / CrewAI
  * install the local management CLI (`crewctl`) inside the virtual environment

It is intentionally conservative about privileged operations:
  * never executes remote install scripts without explicit opt-in
  * validates prerequisites before mutation
  * applies restrictive file permissions where the platform supports them

Usage:
    python scripts/deploy.py [--config config/deploy_config.yaml] [--no-docker]
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import secrets
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable

try:
    import yaml  # type: ignore
except ImportError:
    yaml = None  # Defer requirement until after venv bootstrap


PROJECT_ROOT = Path(__file__).resolve().parent.parent


DEFAULT_CONFIG: Dict[str, object] = {
    "python_min_version": "3.10",
    "venv_path": ".venv",
    "requirements_file": "requirements.txt",
    "state_file": "config/deploy_state.json",
    "model_registry": "config/model_registry.yaml",
    "env_file": ".env",
    "env_template": ".env.template",
    "directories": [
        "agents",
        "backups",
        "config",
        "data",
        "logs",
        "models",
        "models/build",
        "models/templates",
        "scripts",
    ],
    "docker": {
        "enabled": True,
        "compose_file": "docker-compose.deploy.yml",
        "network": "crewai_stack",
        "ollama": {
            "image": "ollama/ollama:latest",
            "port": 11434,
            "volumes": {
                "./models": "/root/.ollama",
                "./data/ollama": "/data",
            },
            "env": {
                "OLLAMA_HOST": "0.0.0.0",
            },
        },
        "crewai": {
            "build_context": ".",
            "dockerfile": "Dockerfile.crewai",
            "port": 8000,
            "env": {
                "OLLAMA_BASE_URL": "http://ollama:11434",
                "CREWCTL_STATE_PATH": "/app/config/model_registry.yaml",
            },
            "volumes": {
                "./agents": "/app/agents",
                "./config": "/app/config",
                "./logs": "/app/logs",
            },
        },
    },
    "secrets": {
        "SECRET_KEY": {"length": 64},
        "JWT_SECRET": {"length": 64},
    },
}


class DeploymentError(Exception):
    """Custom exception for deployment failures."""


def parse_semver(value: str) -> tuple[int, int, int]:
    parts = value.split(".")
    normalized = (parts + ["0", "0", "0"])[:3]
    return tuple(int(x) for x in normalized)  # type: ignore[return-value]


def chmod_safe(path: Path, mode: int) -> None:
    """Best-effort permission hardening on POSIX systems."""
    if os.name == "posix":
        try:
            os.chmod(path, mode)
        except PermissionError:
            print(f"[warn] Unable to set permissions on {path}")


def load_yaml_config(path: Path) -> Dict[str, object]:
    if not path.exists():
        return {}
    if yaml is None:
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise DeploymentError(
                "PyYAML is required to parse YAML configuration. "
                "Install it (`pip install pyyaml`) or provide JSON content."
            ) from exc
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def deep_merge(base: Dict[str, object], override: Dict[str, object]) -> Dict[str, object]:
    result: Dict[str, object] = dict(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)  # type: ignore[arg-type]
        else:
            result[key] = value
    return result


@dataclass
class DeployConfig:
    raw: Dict[str, object]

    @property
    def python_min_version(self) -> tuple[int, int, int]:
        version_str = str(self.raw.get("python_min_version", "3.10"))
        return parse_semver(version_str)

    @property
    def venv_path(self) -> Path:
        return PROJECT_ROOT / str(self.raw.get("venv_path", ".venv"))

    @property
    def requirements_file(self) -> Path:
        return PROJECT_ROOT / str(self.raw.get("requirements_file", "requirements.txt"))

    @property
    def state_file(self) -> Path:
        return PROJECT_ROOT / str(self.raw.get("state_file", "config/deploy_state.json"))

    @property
    def model_registry(self) -> Path:
        return PROJECT_ROOT / str(self.raw.get("model_registry", "config/model_registry.yaml"))

    @property
    def env_file(self) -> Path:
        return PROJECT_ROOT / str(self.raw.get("env_file", ".env"))

    @property
    def env_template(self) -> Path:
        return PROJECT_ROOT / str(self.raw.get("env_template", ".env.template"))

    @property
    def directories(self) -> Iterable[Path]:
        dirs = self.raw.get("directories", [])
        return [PROJECT_ROOT / str(d) for d in dirs] if isinstance(dirs, list) else []

    @property
    def docker_config(self) -> Dict[str, object]:
        return self.raw.get("docker", {}) if isinstance(self.raw.get("docker"), dict) else {}

    @property
    def docker_enabled(self) -> bool:
        docker = self.docker_config
        return bool(docker.get("enabled", True))

    @property
    def docker_compose_file(self) -> Path:
        docker = self.docker_config
        filename = docker.get("compose_file", "docker-compose.deploy.yml")
        return PROJECT_ROOT / str(filename)

    @property
    def secrets(self) -> Dict[str, Dict[str, object]]:
        secrets_cfg = self.raw.get("secrets", {})
        if isinstance(secrets_cfg, dict):
            return {str(k): dict(v) for k, v in secrets_cfg.items() if isinstance(v, dict)}
        return {}


class DeployManager:
    def __init__(self, config: DeployConfig, use_docker: bool):
        self.config = config
        self.use_docker = use_docker and config.docker_enabled
        self.system = platform.system().lower()

    def run(self) -> None:
        print("[info] Starting CrewAI + Ollama deployment")
        self._validate_python()
        self._ensure_directories()
        self._ensure_env_template()
        self._ensure_env_file()
        self._ensure_model_registry()
        self._create_venv()
        self._install_requirements()
        self._install_cli()
        if self.use_docker:
            self._ensure_docker_prereqs()
            self._write_compose_file()
        self._record_state()
        print("[success] Deployment finished. Activate the virtualenv and run `crewctl --help`.")

    def _validate_python(self) -> None:
        min_version = self.config.python_min_version
        if sys.version_info < min_version:
            raise DeploymentError(
                f"Python {min_version[0]}.{min_version[1]}.{min_version[2]}+ required. "
                f"Detected {platform.python_version()}."
            )
        print(f"[info] Python {platform.python_version()} satisfies minimum requirement.")

    def _ensure_directories(self) -> None:
        for directory in self.config.directories:
            directory.mkdir(parents=True, exist_ok=True)
        print("[info] Directory scaffold ensured.")

    def _ensure_env_template(self) -> None:
        template_path = self.config.env_template
        if template_path.exists():
            return
        template_content = (
            "# CrewAI / Ollama environment template\n"
            "OLLAMA_BASE_URL=http://localhost:11434\n"
            "OLLAMA_MODEL=llama2:7b\n"
            "CREWAI_PORT=8000\n"
            "REDIS_URL=redis://localhost:6379/0\n"
            "SECRET_KEY=\n"
            "JWT_SECRET=\n"
            "LOG_LEVEL=INFO\n"
            "ENVIRONMENT=development\n"
        )
        template_path.write_text(template_content, encoding="utf-8")
        chmod_safe(template_path, 0o640)
        print(f"[info] Wrote env template to {template_path.relative_to(PROJECT_ROOT)}.")

    def _ensure_env_file(self) -> None:
        env_path = self.config.env_file
        if not env_path.exists() and self.config.env_template.exists():
            shutil.copy2(self.config.env_template, env_path)
            print("[info] Created .env from template.")
        if not env_path.exists():
            env_path.write_text("", encoding="utf-8")
        self._inject_secrets(env_path)
        chmod_safe(env_path, 0o600)

    def _inject_secrets(self, env_path: Path) -> None:
        lines = env_path.read_text(encoding="utf-8").splitlines()
        secrets_cfg = self.config.secrets
        existing = {line.split("=", 1)[0]: idx for idx, line in enumerate(lines) if "=" in line}
        mutated = False

        for key, metadata in secrets_cfg.items():
            length = int(metadata.get("length", 64))
            if key in existing and lines[existing[key]].split("=", 1)[1].strip():
                continue
            token = secrets.token_urlsafe(length)[:length]
            assignment = f"{key}={token}"
            if key in existing:
                lines[existing[key]] = assignment
            else:
                lines.append(assignment)
            mutated = True
            print(f"[info] Generated secret for {key}.")

        if mutated:
            env_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    def _ensure_model_registry(self) -> None:
        registry_path = self.config.model_registry
        if registry_path.exists():
            return
        initial_registry = (
            "version: 1\n"
            "active_model:\n"
            "  name: llama2:7b\n"
            "  activated_at: null\n"
            "  digest: null\n"
            "  source: default\n"
            "history: []\n"
        )
        registry_path.parent.mkdir(parents=True, exist_ok=True)
        registry_path.write_text(initial_registry, encoding="utf-8")
        chmod_safe(registry_path, 0o640)
        print(f"[info] Initialized model registry at {registry_path.relative_to(PROJECT_ROOT)}.")

    def _create_venv(self) -> None:
        venv_path = self.config.venv_path
        if venv_path.exists():
            print(f"[info] Virtual environment already present at {venv_path}.")
            return
        print(f"[info] Creating virtual environment at {venv_path} ...")
        subprocess.run([sys.executable, "-m", "venv", str(venv_path)], check=True)

    def _venv_python(self) -> Path:
        if platform.system() == "Windows":
            return self.config.venv_path / "Scripts" / "python.exe"
        return self.config.venv_path / "bin" / "python"

    def _install_requirements(self) -> None:
        python_bin = self._venv_python()
        if not python_bin.exists():
            raise DeploymentError("Virtual environment python binary not found.")
        req_file = self.config.requirements_file
        if not req_file.exists():
            print(f"[warn] Requirements file {req_file} missing; skipping dependency install.")
            return
        print(f"[info] Installing requirements from {req_file} ...")
        subprocess.run([str(python_bin), "-m", "pip", "install", "--upgrade", "pip"], check=True)
        subprocess.run([str(python_bin), "-m", "pip", "install", "-r", str(req_file)], check=True)

    def _install_cli(self) -> None:
        python_bin = self._venv_python()
        cli_package = PROJECT_ROOT / "crewctl"
        if not cli_package.exists():
            print("[warn] crewctl package not found; skipping CLI bootstrap.")
            return
        self._write_cli_shim()

    def _write_cli_shim(self) -> None:
        """Create convenience launchers for crewctl."""
        venv = self.config.venv_path
        python_bin = self._venv_python()
        project_path = PROJECT_ROOT.as_posix()
        project_path_win = str(PROJECT_ROOT)
        if platform.system() == "Windows":
            shim = venv / "Scripts" / "crewctl.cmd"
            content = (
                "@echo off\n"
                "setlocal\n"
                f"set \"_PROJECT_ROOT={project_path_win}\"\n"
                "if defined PYTHONPATH (\n"
                "  set \"PYTHONPATH=%_PROJECT_ROOT%;%PYTHONPATH%\"\n"
                ") else (\n"
                "  set \"PYTHONPATH=%_PROJECT_ROOT%\"\n"
                ")\n"
                f"\"{python_bin}\" -m crewctl %*\n"
                "endlocal\n"
            )
        else:
            shim = venv / "bin" / "crewctl"
            content = (
                "#!/usr/bin/env bash\n"
                f'export PYTHONPATH="{project_path}:${{PYTHONPATH:-}}"\n'
                f'"{python_bin}" -m crewctl "$@"\n'
            )
        shim.write_text(content, encoding="utf-8")
        chmod_safe(shim, 0o750)
        print(f"[info] Wrote CLI launcher at {shim.relative_to(PROJECT_ROOT)}.")

    def _ensure_docker_prereqs(self) -> None:
        print("[info] Docker support enabled; verifying prerequisites...")
        for binary in ("docker", "docker compose"):
            cmd = binary.split()
            if shutil.which(cmd[0]) is None:
                raise DeploymentError(
                    f"{cmd[0]!r} not found in PATH. Install Docker Desktop or CLI to continue."
                )
            try:
                subprocess.run(cmd + ["version"], capture_output=True, check=True)
            except subprocess.CalledProcessError as exc:
                raise DeploymentError(f"Unable to execute {' '.join(cmd)}: {exc}") from exc

    def _write_compose_file(self) -> None:
        docker_cfg = self.config.docker_config
        compose_path = self.config.docker_compose_file
        network = docker_cfg.get("network", "crewai_stack")
        ollama = docker_cfg.get("ollama", {}) if isinstance(docker_cfg.get("ollama"), dict) else {}
        crewai = docker_cfg.get("crewai", {}) if isinstance(docker_cfg.get("crewai"), dict) else {}

        def render_env(env_map: Dict[str, str]) -> str:
            if not env_map:
                return "      # none"
            return "\n".join([f"      - {key}={value}" for key, value in env_map.items()])

        def render_volumes(vol_map: Dict[str, str]) -> str:
            if not vol_map:
                return "      # none"
            return "\n".join([f"      - {src}:{dest}" for src, dest in vol_map.items()])

        compose_content = f"""version: '3.9'

networks:
  {network}:
    driver: bridge

services:
  ollama:
    image: {ollama.get('image', 'ollama/ollama:latest')}
    restart: unless-stopped
    networks:
      - {network}
    ports:
      - "{ollama.get('port', 11434)}:11434"
    environment:
{render_env(ollama.get('env', {}))}
    volumes:
{render_volumes(ollama.get('volumes', {}))}
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:11434/api/tags"]
      interval: 30s
      timeout: 10s
      retries: 5

  crewai:
    build:
      context: {crewai.get('build_context', '.')}
      dockerfile: {crewai.get('dockerfile', 'Dockerfile.crewai')}
    restart: unless-stopped
    depends_on:
      ollama:
        condition: service_healthy
    networks:
      - {network}
    ports:
      - "{crewai.get('port', 8000)}:8000"
    environment:
{render_env(crewai.get('env', {}))}
    volumes:
{render_volumes(crewai.get('volumes', {}))}
"""
        compose_path.write_text(compose_content, encoding="utf-8")
        print(f"[info] Docker Compose manifest written to {compose_path.relative_to(PROJECT_ROOT)}.")

    def _record_state(self) -> None:
        state_path = self.config.state_file
        record = {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "venv": str(self.config.venv_path.relative_to(PROJECT_ROOT)),
            "requirements": str(self.config.requirements_file.relative_to(PROJECT_ROOT)),
            "docker_enabled": self.use_docker,
        }
        state_path.parent.mkdir(parents=True, exist_ok=True)
        state_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
        chmod_safe(state_path, 0o640)
        print(f"[info] Deployment state recorded at {state_path.relative_to(PROJECT_ROOT)}.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Deploy CrewAI + Ollama stack.")
    parser.add_argument(
        "--config",
        type=str,
        default="config/deploy_config.yaml",
        help="Path to deployment configuration YAML file.",
    )
    parser.add_argument(
        "--no-docker",
        action="store_true",
        help="Skip Docker manifest generation and validation.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = PROJECT_ROOT / args.config
    overrides = load_yaml_config(config_path) if config_path.exists() else {}
    merged = deep_merge(DEFAULT_CONFIG, overrides if isinstance(overrides, dict) else {})
    deploy_config = DeployConfig(merged)
    manager = DeployManager(deploy_config, use_docker=not args.no_docker)
    try:
        manager.run()
    except DeploymentError as exc:
        print(f"[error] {exc}")
        sys.exit(1)
    except subprocess.CalledProcessError as exc:
        print(f"[error] Command failed with exit code {exc.returncode}: {' '.join(exc.cmd)}")
        sys.exit(exc.returncode)


if __name__ == "__main__":
    main()
