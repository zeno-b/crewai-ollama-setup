#!/usr/bin/env python3
"""
Self-contained CrewAI + Ollama deployment utility.

This script prepares a secure, multi-platform environment by:
  * verifying local prerequisites (Python, Docker, Docker Compose),
  * scaffolding runtime directories and configuration,
  * creating/refreshing a Python virtual environment,
  * installing project dependencies,
  * wiring in a management CLI (`crewaictl`) inside the venv,
  * optionally bootstrapping container services via Docker Compose.
"""

from __future__ import annotations

import argparse
import logging
import os
import platform
import secrets
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Iterable, List, Sequence

LOGGER = logging.getLogger("deploy")


class DeployError(RuntimeError):
    """Raised when deployment steps fail."""


def configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


class DeployManager:
    """Coordinates the deployment lifecycle."""

    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.project_root = Path(__file__).resolve().parent.parent
        self.deployment_dir = self.project_root / "deployment"
        self.templates_dir = self.deployment_dir / "templates"
        self.state_dir = self.deployment_dir / "state"
        self.config_dir = self.project_root / "config"
        self.models_dir = self.project_root / "models"
        self.logs_dir = self.project_root / "logs"
        self.data_dir = self.project_root / "data"
        self.venv_path = (
            Path(args.venv_path).expanduser().resolve()
            if args.venv_path
            else self.project_root / ".venv"
        )
        self.python_executable = (
            Path(args.python).expanduser().resolve() if args.python else Path(sys.executable)
        )
        self.compose_file = (
            Path(args.compose_file).expanduser().resolve()
            if args.compose_file
            else self.project_root / "docker-compose.yml"
        )
        self.compose_command: Sequence[str] | None = None

    # --------------------------------------------------------------------- #
    # Public API
    # --------------------------------------------------------------------- #

    def run(self) -> None:
        LOGGER.info("Starting CrewAI & Ollama deployment routine")
        self._check_prerequisites()
        self._prepare_directories()
        self._ensure_env_file()
        self._ensure_model_registry()
        self._create_virtualenv()
        self._install_dependencies()
        self._create_cli_entrypoints()
        if self.args.start_services:
            self._start_services()
        self._print_summary()

    # ------------------------------------------------------------------ #
    # Individual steps
    # ------------------------------------------------------------------ #

    def _check_prerequisites(self) -> None:
        LOGGER.debug("Checking prerequisites")

        if not self.python_executable.exists():
            raise DeployError(f"Python executable not found at {self.python_executable}")

        py_version = sys.version_info
        if py_version < (3, 10):
            raise DeployError("Python 3.10 or newer is required to run this deployment script.")
        LOGGER.debug("Detected Python %s.%s.%s", py_version.major, py_version.minor, py_version.micro)

        docker_binary = shutil.which("docker")
        if not docker_binary:
            raise DeployError(
                "Docker CLI not detected in PATH. Install Docker Desktop (macOS/Windows) "
                "or Docker Engine (Linux) before continuing."
            )
        LOGGER.debug("Docker binary located at %s", docker_binary)

        self.compose_command = self._detect_compose_command()
        LOGGER.debug("Using Docker Compose command: %s", " ".join(self.compose_command))

        if not self.compose_file.exists():
            raise DeployError(f"Docker Compose file not found at {self.compose_file}")

    def _prepare_directories(self) -> None:
        LOGGER.debug("Preparing project directories")
        directories = [
            self.data_dir,
            self.logs_dir,
            self.models_dir,
            self.models_dir / "lrms",
            self.state_dir,
            self.config_dir / "grafana" / "datasources",
            self.config_dir / "grafana" / "dashboards",
        ]
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            LOGGER.debug("Ensured directory exists: %s", directory)
            if os.name != "nt":
                try:
                    os.chmod(directory, 0o750)
                except OSError as exc:
                    LOGGER.debug("Skipping chmod on %s: %s", directory, exc)

    def _ensure_env_file(self) -> None:
        env_path = self.project_root / ".env"
        template_path = self.templates_dir / ".env.example"

        if env_path.exists():
            LOGGER.info("Environment file already exists at %s", env_path)
            self._patch_env_secrets(env_path)
            return

        LOGGER.info("Creating environment file from template at %s", env_path)
        template_content = template_path.read_text(encoding="utf-8")
        replacements = {
            "SECRET_KEY": secrets.token_urlsafe(32),
            "JWT_SECRET": secrets.token_urlsafe(32),
            "GRAFANA_PASSWORD": secrets.token_urlsafe(16),
        }
        for key, value in replacements.items():
            template_content = template_content.replace(f"{{{{{key}}}}}", value)
        env_path.write_text(template_content, encoding="utf-8")
        if os.name != "nt":
            env_path.chmod(0o600)

    def _patch_env_secrets(self, env_path: Path) -> None:
        LOGGER.debug("Verifying required secrets in %s", env_path)
        required_keys = {
            "SECRET_KEY": secrets.token_urlsafe(32),
            "JWT_SECRET": secrets.token_urlsafe(32),
            "GRAFANA_PASSWORD": secrets.token_urlsafe(16),
        }
        lines = env_path.read_text(encoding="utf-8").splitlines()
        updated = False
        for index, line in enumerate(lines):
            if not line or line.strip().startswith("#"):
                continue
            if "=" not in line:
                continue
            key, current = line.split("=", 1)
            key = key.strip()
            if key in required_keys and (not current or current.startswith("{{")):
                lines[index] = f"{key}={required_keys[key]}"
                updated = True
        for key, value in required_keys.items():
            if not any(line.startswith(f"{key}=") for line in lines):
                lines.append(f"{key}={value}")
                updated = True
        if updated:
            env_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
            LOGGER.info("Updated missing secret values in %s", env_path)

    def _ensure_model_registry(self) -> None:
        registry_path = self.config_dir / "model_registry.yaml"
        if registry_path.exists():
            LOGGER.debug("Model registry already present at %s", registry_path)
            return
        LOGGER.info("Initializing model registry at %s", registry_path)
        default_registry = (
            "default_model: llama2:latest\n"
            "history:\n"
            "  - llama2:latest\n"
            "models:\n"
            "  llama2:\n"
            "    versions: []\n"
        )
        registry_path.write_text(default_registry, encoding="utf-8")

    def _create_virtualenv(self) -> None:
        if self.venv_path.exists():
            if self.args.recreate_venv:
                LOGGER.warning("Recreating existing virtual environment at %s", self.venv_path)
                shutil.rmtree(self.venv_path)
            else:
                LOGGER.info("Virtual environment already present at %s", self.venv_path)
                return

        LOGGER.info("Creating virtual environment at %s", self.venv_path)
        cmd = [str(self.python_executable), "-m", "venv", str(self.venv_path)]
        if sys.version_info >= (3, 11):
            cmd.insert(3, "--upgrade-deps")
        self._run_subprocess(cmd, "Failed to create virtual environment")

    def _install_dependencies(self) -> None:
        if self.args.skip_install:
            LOGGER.info("Skipping dependency installation (per --skip-install)")
            return

        pip_executable = self._resolve_venv_executable("pip")
        LOGGER.info("Upgrading pip/setuptools in virtual environment")
        self._run_subprocess(
            [pip_executable, "install", "--upgrade", "pip", "setuptools", "wheel"],
            "Failed to upgrade pip/setuptools",
        )
        requirements_path = self.project_root / "requirements.txt"
        if requirements_path.exists():
            LOGGER.info("Installing project dependencies from %s", requirements_path)
            self._run_subprocess(
                [pip_executable, "install", "-r", str(requirements_path)],
                "Failed to install requirements",
            )
        else:
            LOGGER.warning("requirements.txt not found at %s; skipping dependency install", requirements_path)

    def _create_cli_entrypoints(self) -> None:
        scripts_dir = self._venv_scripts_dir()
        scripts_dir.mkdir(parents=True, exist_ok=True)

        unix_entry = scripts_dir / "crewaictl"
        windows_entry = scripts_dir / "crewaictl.cmd"
        python_in_venv = scripts_dir / ("python.exe" if os.name == "nt" else "python")

        if platform.system() != "Windows":
            LOGGER.debug("Creating UNIX CLI wrapper at %s", unix_entry)
            unix_entry.write_text(
                "\n".join(
                    [
                        "#!/usr/bin/env sh",
                        'SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"',
                        'PROJECT_ROOT="$(cd "$(dirname "$SCRIPT_DIR")/.." && pwd)"',
                        'if [ -n "$PYTHONPATH" ]; then',
                        '  export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"',
                        "else",
                        '  export PYTHONPATH="$PROJECT_ROOT"',
                        "fi",
                        f'"$SCRIPT_DIR/python" -m crewai_cli "$@"',
                        "",
                    ]
                ),
                encoding="utf-8",
            )
            unix_entry.chmod(0o750)

        LOGGER.debug("Creating Windows CLI wrapper at %s", windows_entry)
        windows_entry.write_text(
            "\r\n".join(
                [
                    "@echo off",
                    "setlocal",
                    "set SCRIPT_DIR=%~dp0",
                    "set PROJECT_ROOT=%SCRIPT_DIR%..\\..",
                    'if defined PYTHONPATH (',
                    '  set "PYTHONPATH=%PROJECT_ROOT%;%PYTHONPATH%"',
                    ") else (",
                    '  set "PYTHONPATH=%PROJECT_ROOT%"',
                    ")",
                    f'"{python_in_venv}" -m crewai_cli %*',
                    "endlocal",
                    "",
                ]
            ),
            encoding="utf-8",
        )

    def _start_services(self) -> None:
        if not self.compose_command:
            raise DeployError("Docker Compose command not detected; cannot start services.")

        LOGGER.info("Starting services using Docker Compose file %s", self.compose_file)
        cmd = list(self.compose_command) + ["-f", str(self.compose_file), "up", "-d"]
        self._run_subprocess(cmd, "Failed to start Docker Compose services")

    def _print_summary(self) -> None:
        bin_dir = self._venv_scripts_dir()
        cli_hint = f"{bin_dir / 'crewaictl'}" if platform.system() != "Windows" else f"{bin_dir / 'crewaictl.cmd'}"

        LOGGER.info("Deployment routine completed successfully.")
        LOGGER.info("Next steps:")
        LOGGER.info("  1. Activate the virtualenv: %s", self._activation_hint())
        LOGGER.info("  2. Use the management CLI: %s --help", cli_hint)
        LOGGER.info("  3. To launch services: %s -f %s up -d", " ".join(self.compose_command or ['docker', 'compose']), self.compose_file)

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    def _detect_compose_command(self) -> Sequence[str]:
        candidates: List[Sequence[str]] = [
            ("docker", "compose"),
            ("docker-compose",),
        ]
        for candidate in candidates:
            try:
                result = subprocess.run(
                    list(candidate) + ["version"],
                    capture_output=True,
                    text=True,
                    check=True,
                )
                if result.returncode == 0:
                    return candidate
            except (subprocess.CalledProcessError, FileNotFoundError):
                continue
        raise DeployError(
            "Docker Compose was not detected. Install Docker Compose v2 (docker compose) "
            "or v1 (docker-compose) before running this script."
        )

    def _venv_scripts_dir(self) -> Path:
        return self.venv_path / ("Scripts" if platform.system() == "Windows" else "bin")

    def _resolve_venv_executable(self, name: str) -> str:
        suffix = ".exe" if platform.system() == "Windows" else ""
        executable = self._venv_scripts_dir() / f"{name}{suffix}"
        if not executable.exists():
            raise DeployError(f"Expected executable not found in virtualenv: {executable}")
        return str(executable)

    @staticmethod
    def _run_subprocess(command: Iterable[str], error_message: str) -> None:
        LOGGER.debug("Executing: %s", " ".join(str(part) for part in command))
        try:
            subprocess.run(command, check=True)
        except subprocess.CalledProcessError as exc:
            raise DeployError(f"{error_message}: {exc}") from exc

    def _activation_hint(self) -> str:
        if platform.system() == "Windows":
            return f"{self.venv_path}\\Scripts\\activate"
        return f"source {self.venv_path}/bin/activate"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Deploy CrewAI + Ollama infrastructure.")
    parser.add_argument(
        "--venv-path",
        help="Path to create the Python virtual environment (defaults to .venv in project root).",
    )
    parser.add_argument(
        "--python",
        help="Python interpreter to use for virtual environment creation.",
    )
    parser.add_argument(
        "--compose-file",
        help="Path to the Docker Compose file to manage services.",
    )
    parser.add_argument(
        "--recreate-venv",
        action="store_true",
        help="If set, recreate the virtual environment even if it already exists.",
    )
    parser.add_argument(
        "--skip-install",
        action="store_true",
        help="Skip pip installation inside the virtual environment.",
    )
    parser.add_argument(
        "--start-services",
        action="store_true",
        help="Run docker compose up -d after provisioning.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging output.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    configure_logging(args.verbose)
    manager = DeployManager(args)
    try:
        manager.run()
    except DeployError as exc:
        LOGGER.error("Deployment failed: %s", exc)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
