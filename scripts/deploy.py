#!/usr/bin/env python3
"""
Self-contained deployment and rollback manager for the CrewAI + Ollama stack.

This script orchestrates dependency checks, directory preparation, Docker
Compose deployments, and full environment rollbacks (including containers,
volumes, networks, local runtime data, and optional Python virtual
environments).
"""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Iterable, List, Sequence

# Project metadata -----------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOG_PATH = PROJECT_ROOT / "deploy.log"
DEFAULT_VENV_DIR = PROJECT_ROOT / ".venv"

DEFAULT_DIRECTORIES: List[Path] = [
    PROJECT_ROOT / "data",
    PROJECT_ROOT / "data" / "ollama",
    PROJECT_ROOT / "data" / "crewai",
    PROJECT_ROOT / "logs",
    PROJECT_ROOT / "models",
    PROJECT_ROOT / "backups",
]

COMPOSE_FILES = {
    "dev": ["docker-compose.yml", "docker-compose.override.yml"],
    "prod": ["docker-compose.yml", "docker-compose.prod.yml"],
}

CONTAINER_IDENTIFIERS = [
    "crewai-service",
    "ollama-service",
    "redis-service",
    "prometheus-service",
    "grafana-service",
]

VOLUME_NAMES = [
    "redis_data",
    "prometheus_data",
    "grafana_data",
    "crewai_dev_data",
    "crewai_dev_logs",
    "ollama_dev_data",
]

NETWORK_NAMES = [
    "crewai-network",
]


# Utility helpers ------------------------------------------------------------------

def configure_logging(verbose: bool) -> None:
    """Configure root logger for console and file output."""
    log_level = logging.DEBUG if verbose else logging.INFO
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

    root_logger = logging.getLogger()
    if root_logger.handlers:
        for handler in list(root_logger.handlers):
            root_logger.removeHandler(handler)

    handlers: List[logging.Handler] = [
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_PATH, mode="a", encoding="utf-8"),
    ]

    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=handlers,
    )


def run_subprocess(
    command: Sequence[str],
    *,
    cwd: Path | None = None,
    check: bool = True,
) -> subprocess.CompletedProcess:
    """Run a subprocess command and raise a RuntimeError on failure."""
    logging.getLogger("deploy").debug("Executing command: %s", " ".join(command))
    try:
        return subprocess.run(
            command,
            cwd=str(cwd) if cwd else None,
            check=check,
        )
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"Command failed: {' '.join(command)}") from exc


def which(executable: str) -> str | None:
    """Resolve an executable path using shutil.which with logging."""
    path = shutil.which(executable)
    logging.getLogger("deploy").debug("which(%s) -> %s", executable, path)
    return path


# Deployment manager ---------------------------------------------------------------

class DeploymentManager:
    """Deployment and rollback operations for the CrewAI stack."""

    def __init__(self, project_root: Path, venv_dir: Path = DEFAULT_VENV_DIR):
        self.project_root = project_root
        self.venv_dir = venv_dir
        self.logger = logging.getLogger(self.__class__.__name__)
        self._compose_cmd: List[str] | None = None

    # ------------------------------------------------------------------ dependencies
    def check_dependencies(self) -> None:
        """Ensure required executables are available."""
        if not which("docker"):
            raise RuntimeError(
                "Docker CLI not found. Please install Docker before continuing."
            )

        self._compose_cmd = self._resolve_compose_command()
        self.logger.info(
            "Using Docker Compose command: %s", " ".join(self._compose_cmd)
        )

    def _resolve_compose_command(self) -> List[str]:
        """Determine whether to use `docker compose` or `docker-compose`."""
        docker_path = which("docker")
        if docker_path:
            result = subprocess.run(
                [docker_path, "compose", "version"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            if result.returncode == 0:
                return [docker_path, "compose"]

        docker_compose_path = which("docker-compose")
        if docker_compose_path:
            result = subprocess.run(
                [docker_compose_path, "version"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            if result.returncode == 0:
                return [docker_compose_path]

        raise RuntimeError(
            "Docker Compose v2 or v1 is required. Install Docker Desktop or the "
            "`docker-compose` plugin and retry."
        )

    def _ensure_compose_cmd(self) -> List[str]:
        if self._compose_cmd is None:
            self._compose_cmd = self._resolve_compose_command()
        return self._compose_cmd

    # ---------------------------------------------------------------- directory setup
    def prepare_directories(self) -> None:
        """Create required directory structure for bind mounts."""
        for directory in DEFAULT_DIRECTORIES:
            directory.mkdir(parents=True, exist_ok=True)
            self.logger.debug("Ensured directory exists: %s", directory)

    # ------------------------------------------------------------------- python deps
    def install_python_dependencies(
        self,
        *,
        use_venv: bool = True,
        requirements_filename: str = "requirements.txt",
    ) -> None:
        """Install Python dependencies either in a venv or the active interpreter."""
        requirements_path = self.project_root / requirements_filename
        if not requirements_path.exists():
            self.logger.warning("requirements.txt not found; skipping Python dependency installation.")
            return

        if use_venv:
            if not self.venv_dir.exists():
                self.logger.info("Creating Python virtual environment at %s", self.venv_dir)
                run_subprocess([sys.executable, "-m", "venv", str(self.venv_dir)])

            pip_path = self.venv_dir / ("Scripts" if os.name == "nt" else "bin") / ("pip.exe" if os.name == "nt" else "pip")
            if not pip_path.exists():
                raise RuntimeError(f"pip executable not found inside virtualenv at {pip_path}")
            pip_command = [str(pip_path)]
        else:
            pip_command = [sys.executable, "-m", "pip"]

        self.logger.info("Installing Python dependencies from %s", requirements_path)
        run_subprocess(pip_command + ["install", "-r", str(requirements_path)])

    # ---------------------------------------------------------------------- compose
    def _compose_files_for_env(self, environment: str) -> List[Path]:
        try:
            filenames = COMPOSE_FILES[environment]
        except KeyError as exc:
            raise ValueError(f"Unknown environment: {environment}") from exc

        files: List[Path] = []
        for name in filenames:
            path = self.project_root / name
            if path.exists():
                files.append(path)
            else:
                self.logger.debug("Compose file missing for %s: %s", environment, path)
        return files

    def _compose_base_command(self, environment: str) -> List[str]:
        files = self._compose_files_for_env(environment)
        if not files:
            raise RuntimeError(f"No compose files available for environment '{environment}'.")

        command = list(self._ensure_compose_cmd())
        for compose_file in files:
            command.extend(["-f", str(compose_file)])
        return command

    def _run_compose(self, environment: str, *args: str) -> None:
        command = self._compose_base_command(environment) + list(args)
        run_subprocess(command, cwd=self.project_root)

    # ------------------------------------------------------------------------ deploy
    def deploy(
        self,
        environment: str,
        *,
        build: bool = True,
        force_recreate: bool = False,
        install_python_deps: bool = True,
        use_venv: bool = True,
        prune: bool = False,
    ) -> None:
        """Deploy the requested environment."""
        self.check_dependencies()
        self.prepare_directories()

        if install_python_deps:
            self.install_python_dependencies(use_venv=use_venv)

        if prune:
            self.prune_docker_system()

        compose_files = self._compose_files_for_env(environment)
        self.logger.info(
            "Deploying '%s' using compose files: %s",
            environment,
            ", ".join(file.name for file in compose_files),
        )

        up_args: List[str] = ["up", "-d", "--remove-orphans"]
        if build:
            up_args.append("--build")
        if force_recreate:
            up_args.append("--force-recreate")

        self._run_compose(environment, *up_args)
        self.logger.info("Deployment for environment '%s' completed successfully.", environment)

    # --------------------------------------------------------------------- rollback
    def rollback(
        self,
        environment: str,
        *,
        remove_data: bool = True,
        remove_venv: bool = True,
        force: bool = True,
    ) -> None:
        """Perform a full rollback by stopping services and deleting artifacts."""
        self.check_dependencies()

        environments: Iterable[str]
        if environment == "all":
            environments = ("dev", "prod")
        else:
            environments = (environment,)

        for env_name in environments:
            compose_files = self._compose_files_for_env(env_name)
            if not compose_files:
                self.logger.debug("Skipping '%s': no compose files found.", env_name)
                continue
            try:
                self.logger.info("Stopping services for environment '%s'.", env_name)
                self._run_compose(env_name, "down", "--volumes", "--remove-orphans")
            except RuntimeError as exc:
                self.logger.warning("Compose down failed for '%s': %s", env_name, exc)

        if force:
            self.force_remove_containers()

        if remove_data:
            self.remove_volumes()
            self.remove_data_paths()

        if remove_venv:
            self.remove_virtualenv()

        self.remove_networks()
        self.logger.info("Rollback complete. All services stopped and artifacts removed.")

    # ----------------------------------------------------------------------- status
    def status(self, environment: str) -> None:
        """Show docker-compose ps output for the selected environment."""
        self.check_dependencies()
        compose_files = self._compose_files_for_env(environment)
        if not compose_files:
            self.logger.warning("No compose configuration found for '%s'.", environment)
            return
        self._run_compose(environment, "ps")

    # ---------------------------------------------------------------- ancillary ops
    def prune_docker_system(self) -> None:
        """Run docker system prune."""
        self.logger.info("Pruning unused Docker resources...")
        run_subprocess(["docker", "system", "prune", "-f"])

    def force_remove_containers(self) -> None:
        """Force-remove containers matching known identifiers."""
        try:
            result = subprocess.run(
                ["docker", "ps", "-a", "--format", "{{.Names}}"],
                capture_output=True,
                text=True,
                check=True,
            )
            existing_containers = result.stdout.strip().splitlines()
        except subprocess.CalledProcessError:
            self.logger.warning("Failed to list Docker containers for force removal.")
            return

        for name in existing_containers:
            if any(
                name == identifier or name.startswith(f"{identifier}-")
                for identifier in CONTAINER_IDENTIFIERS
            ):
                self.logger.debug("Force-removing container: %s", name)
                run_subprocess(["docker", "rm", "-f", name], check=False)

    def remove_volumes(self) -> None:
        """Remove known Docker volumes."""
        for volume in VOLUME_NAMES:
            self.logger.debug("Removing Docker volume if present: %s", volume)
            run_subprocess(["docker", "volume", "rm", "-f", volume], check=False)

    def remove_networks(self) -> None:
        """Remove Docker networks created for the stack."""
        for network in NETWORK_NAMES:
            self.logger.debug("Removing Docker network if present: %s", network)
            run_subprocess(["docker", "network", "rm", network], check=False)

    def remove_data_paths(self) -> None:
        """Delete local data directories and runtime artifacts."""
        targets = {
            self.project_root / "data",
            self.project_root / "logs",
            self.project_root / "models",
            self.project_root / "backups",
            LOG_PATH,
        }

        for path in targets:
            if not path.exists():
                continue
            try:
                if path.is_dir():
                    shutil.rmtree(path, ignore_errors=True)
                    self.logger.debug("Removed directory: %s", path)
                else:
                    path.unlink(missing_ok=True)  # type: ignore[arg-type]
                    self.logger.debug("Removed file: %s", path)
            except Exception as exc:  # pragma: no cover - defensive logging
                self.logger.warning("Failed to remove %s: %s", path, exc)

    def remove_virtualenv(self) -> None:
        """Delete the managed Python virtual environment."""
        if self.venv_dir.exists():
            shutil.rmtree(self.venv_dir, ignore_errors=True)
            self.logger.debug("Removed virtual environment at %s", self.venv_dir)


# CLI ------------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Deployment manager for the CrewAI + Ollama stack.",
    )
    parser.add_argument(
        "--project-root",
        default=str(PROJECT_ROOT),
        help="Path to the project root directory (default: repository root).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging.",
    )

    subparsers = parser.add_subparsers(dest="command", title="commands")

    # deploy -----------------------------------------------------------------
    deploy_parser = subparsers.add_parser(
        "deploy",
        help="Deploy the stack using Docker Compose.",
    )
    deploy_parser.add_argument(
        "--env",
        choices=("dev", "prod"),
        default="dev",
        help="Deployment environment (default: dev).",
    )
    deploy_parser.add_argument(
        "--skip-build",
        action="store_true",
        help="Skip rebuilding images during deployment.",
    )
    deploy_parser.add_argument(
        "--force-recreate",
        action="store_true",
        help="Force recreate Docker containers.",
    )
    deploy_parser.add_argument(
        "--skip-python-deps",
        action="store_true",
        help="Skip installing Python dependencies.",
    )
    deploy_parser.add_argument(
        "--system-python",
        action="store_true",
        help="Install Python dependencies into the current interpreter instead of a dedicated virtualenv.",
    )
    deploy_parser.add_argument(
        "--prune",
        action="store_true",
        help="Run 'docker system prune -f' before deployment.",
    )

    # rollback ---------------------------------------------------------------
    rollback_parser = subparsers.add_parser(
        "rollback",
        help="Stop all services and delete containers, data, and environments.",
    )
    rollback_parser.add_argument(
        "--env",
        choices=("dev", "prod", "all"),
        default="all",
        help="Environment to rollback (default: all).",
    )
    rollback_parser.add_argument(
        "--keep-data",
        action="store_true",
        help="Preserve local data directories and Docker volumes.",
    )
    rollback_parser.add_argument(
        "--keep-venv",
        action="store_true",
        help="Preserve the managed Python virtual environment.",
    )
    rollback_parser.add_argument(
        "--no-force",
        action="store_true",
        help="Do not force removal of lingering containers.",
    )

    # status -----------------------------------------------------------------
    status_parser = subparsers.add_parser(
        "status",
        help="Show docker-compose status for an environment.",
    )
    status_parser.add_argument(
        "--env",
        choices=("dev", "prod"),
        default="dev",
        help="Environment to inspect (default: dev).",
    )

    # check ------------------------------------------------------------------
    subparsers.add_parser(
        "check",
        help="Verify required dependencies are installed.",
    )

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if not args.command:
        parser.print_help()
        return 0

    configure_logging(verbose=args.verbose)
    project_root = Path(args.project_root).resolve()
    manager = DeploymentManager(project_root=project_root)

    try:
        if args.command == "deploy":
            manager.deploy(
                args.env,
                build=not args.skip_build,
                force_recreate=args.force_recreate,
                install_python_deps=not args.skip_python_deps,
                use_venv=not args.system_python,
                prune=args.prune,
            )
        elif args.command == "rollback":
            manager.rollback(
                args.env,
                remove_data=not args.keep_data,
                remove_venv=not args.keep_venv,
                force=not args.no_force,
            )
        elif args.command == "status":
            manager.status(args.env)
        elif args.command == "check":
            manager.check_dependencies()
            logging.getLogger("deploy").info("All required dependencies are available.")
        else:  # pragma: no cover - defensive branch
            parser.print_help()
            return 2
    except (RuntimeError, ValueError) as exc:
        logging.getLogger("deploy").error(str(exc))
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
