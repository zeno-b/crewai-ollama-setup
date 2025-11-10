#!/usr/bin/env python3
"""
Deployment management CLI for the CrewAI + Ollama stack.

This script is designed to be self-contained: it verifies system dependencies,
sets up the local Python environment, prepares configuration files, starts the
Docker services, and can fully roll everything back (containers, volumes, and
local artifacts).
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

import typer


DEFAULT_PROMETHEUS_CONFIG = """global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: 'crewai'
    metrics_path: /metrics
    static_configs:
      - targets: ['crewai:9090']
"""

DEFAULT_GRAFANA_DATASOURCE = """apiVersion: 1

datasources:
  - name: Prometheus
    type: prometheus
    access: proxy
    orgId: 1
    url: http://prometheus:9090
    isDefault: true
    editable: true
"""

DEFAULT_GRAFANA_DASHBOARD_CONFIG = """apiVersion: 1

providers:
  - name: default
    orgId: 1
    folder: ''
    type: file
    disableDeletion: false
    editable: true
    options:
      path: /etc/grafana/provisioning/dashboards/dashboards
"""


class DeploymentError(RuntimeError):
    """Raised when deployment operations fail."""


class DeploymentManager:
    """Manage deployment, status, and rollback for the CrewAI + Ollama stack."""

    def __init__(self, project_root: Path, compose_files: Optional[List[Path]] = None) -> None:
        self.project_root = project_root.resolve()
        self.venv_path = self.project_root / ".venv"
        self.requirements_file = self.project_root / "requirements.txt"
        self.log_file = self.project_root / "deploy.log"
        base_compose = self.project_root / "docker-compose.yml"
        if not base_compose.exists():
            raise DeploymentError("docker-compose.yml not found in project root.")
        self.compose_files: List[Path] = [base_compose]
        if compose_files:
            for extra in compose_files:
                extra = extra.resolve()
                if extra.exists():
                    if extra not in self.compose_files:
                        self.compose_files.append(extra)
                else:
                    raise DeploymentError(f"Compose file not found: {extra}")
        self._compose_cmd: Optional[List[str]] = None
        self.logger = logging.getLogger("deployment")
        if not self.logger.handlers:
            self.logger.setLevel(logging.INFO)
            file_handler = logging.FileHandler(self.log_file)
            formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
        self.logger.propagate = False
        self.logger.info("Initialized deployment manager with compose files: %s", self.compose_files_arg())

    def docker_executable(self) -> str:
        docker_path = shutil.which("docker")
        if not docker_path:
            raise DeploymentError(
                "Docker CLI not found. Install Docker Desktop / Engine and ensure `docker` is on PATH."
            )
        return docker_path

    def get_compose_command(self) -> List[str]:
        if self._compose_cmd:
            return self._compose_cmd

        docker_compose_path = shutil.which("docker-compose")
        if docker_compose_path:
            self._compose_cmd = [docker_compose_path]
            self.logger.info("Using docker-compose V1 at %s", docker_compose_path)
            return self._compose_cmd

        docker_path = shutil.which("docker")
        if docker_path:
            try:
                result = subprocess.run(
                    [docker_path, "compose", "version"],
                    capture_output=True,
                    check=False,
                )
                if result.returncode == 0:
                    self._compose_cmd = [docker_path, "compose"]
                    self.logger.info("Using Docker Compose V2 via `docker compose`.")
                    return self._compose_cmd
                self.logger.error("`docker compose version` returned %s", result.returncode)
            except FileNotFoundError as exc:
                raise DeploymentError("Docker CLI not available when checking compose.") from exc

        raise DeploymentError(
            "Docker Compose not found. Install either docker-compose (V1) or Docker Compose V2."
        )

    def compose_files_arg(self) -> List[str]:
        args: List[str] = []
        for compose_file in self.compose_files:
            args.extend(["-f", str(compose_file)])
        return args

    def ensure_prerequisites(self) -> None:
        docker_cmd = self.docker_executable()
        docker_version = subprocess.run([docker_cmd, "--version"], capture_output=True, check=False)
        if docker_version.returncode != 0:
            raise DeploymentError("Docker CLI is installed but failed to execute `docker --version`.")
        self.logger.info("Docker detected: %s", docker_version.stdout.decode().strip() or docker_cmd)

        compose_cmd = self.get_compose_command()
        compose_version = subprocess.run(compose_cmd + ["version"], capture_output=True, check=False)
        if compose_version.returncode != 0:
            raise DeploymentError("Docker Compose is installed but `compose version` failed.")
        self.logger.info(
            "Docker Compose detected: %s",
            compose_version.stdout.decode().strip() or "compose command available",
        )

    def prepare_environment(self) -> None:
        directories = [
            self.project_root / "data",
            self.project_root / "data" / "ollama",
            self.project_root / "data" / "crewai",
            self.project_root / "logs",
            self.project_root / "models",
            self.project_root / "backups",
            self.project_root / "config",
            self.project_root / "config" / "grafana",
            self.project_root / "config" / "grafana" / "provisioning",
            self.project_root / "config" / "grafana" / "provisioning" / "datasources",
            self.project_root / "config" / "grafana" / "provisioning" / "dashboards",
            self.project_root / "config" / "grafana" / "provisioning" / "dashboards" / "dashboards",
        ]
        for directory in directories:
            if not directory.exists():
                directory.mkdir(parents=True, exist_ok=True)
                self.logger.info("Created directory %s", self._rel(directory))

        prometheus_config = self.project_root / "config" / "prometheus.yml"
        if not prometheus_config.exists():
            prometheus_config.write_text(DEFAULT_PROMETHEUS_CONFIG)
            self.logger.info(
                "Created default Prometheus config at %s",
                self._rel(prometheus_config),
            )

        grafana_datasource = (
            self.project_root / "config" / "grafana" / "provisioning" / "datasources" / "datasource.yml"
        )
        if not grafana_datasource.exists():
            grafana_datasource.write_text(DEFAULT_GRAFANA_DATASOURCE)
            self.logger.info("Created default Grafana datasource config at %s", self._rel(grafana_datasource))

        grafana_dashboard_cfg = (
            self.project_root / "config" / "grafana" / "provisioning" / "dashboards" / "dashboard.yml"
        )
        if not grafana_dashboard_cfg.exists():
            grafana_dashboard_cfg.write_text(DEFAULT_GRAFANA_DASHBOARD_CONFIG)
            self.logger.info("Created default Grafana dashboard config at %s", self._rel(grafana_dashboard_cfg))

        dashboards_dir = (
            self.project_root / "config" / "grafana" / "provisioning" / "dashboards" / "dashboards"
        )
        if dashboards_dir.exists():
            gitkeep = dashboards_dir / ".gitkeep"
            if not gitkeep.exists():
                gitkeep.write_text("")
                self.logger.info("Created placeholder dashboard directory at %s", self._rel(gitkeep))

    def ensure_virtualenv(self) -> None:
        if self.venv_path.exists():
            self.logger.info("Virtual environment already present at %s", self._rel(self.venv_path))
            return
        self.logger.info("Creating virtual environment at %s", self._rel(self.venv_path))
        cmd = [sys.executable, "-m", "venv", str(self.venv_path)]
        self._run(cmd)

    def install_requirements(self) -> None:
        if not self.requirements_file.exists():
            self.logger.warning("requirements.txt not found; skipping Python dependency installation.")
            return
        pip_executable = self._pip_path()
        self.logger.info("Upgrading pip/setuptools/wheel inside the virtual environment.")
        self._run([pip_executable, "install", "--upgrade", "pip", "setuptools", "wheel"])
        self.logger.info("Installing Python dependencies from %s", self._rel(self.requirements_file))
        self._run([pip_executable, "install", "-r", str(self.requirements_file)])

    def pull_images(self) -> None:
        compose_cmd = self.get_compose_command()
        cmd = compose_cmd + self.compose_files_arg() + ["pull"]
        self.logger.info("Pulling container images with command: %s", " ".join(cmd))
        self._run(cmd, check=False)

    def start_services(self, build: bool = True) -> None:
        compose_cmd = self.get_compose_command()
        cmd = compose_cmd + self.compose_files_arg() + ["up"]
        if build:
            cmd.append("--build")
        cmd.extend(["-d", "--remove-orphans"])
        self.logger.info("Starting services with command: %s", " ".join(cmd))
        self._run(cmd)
        self.show_status()

    def show_status(self) -> None:
        compose_cmd = self.get_compose_command()
        cmd = compose_cmd + self.compose_files_arg() + ["ps", "--all"]
        self.logger.info("Fetching service status with command: %s", " ".join(cmd))
        self._run(cmd)

    def stop_services(self) -> None:
        compose_cmd = self.get_compose_command()
        cmd = compose_cmd + self.compose_files_arg() + ["down", "--volumes", "--remove-orphans", "--rmi", "local"]
        self.logger.info("Stopping services with command: %s", " ".join(cmd))
        result = self._run(cmd, check=False)
        if result.returncode != 0:
            self.logger.warning("`docker compose down` exited with %s; attempting manual cleanup.", result.returncode)

        docker_cmd = self.docker_executable()
        containers = [
            "crewai-service",
            "ollama-service",
            "redis-service",
            "prometheus-service",
            "grafana-service",
        ]
        for container in containers:
            self.logger.info("Force removing container %s", container)
            self._run([docker_cmd, "rm", "-f", container], check=False)

        self.logger.info("Removing network crewai-network if present.")
        self._run([docker_cmd, "network", "rm", "crewai-network"], check=False)

        if result.returncode != 0:
            raise DeploymentError("docker compose down did not complete successfully; manual cleanup attempted.")

    def clean_artifacts(self, remove_venv: bool = True) -> None:
        targets = [
            self.project_root / "data",
            self.project_root / "logs",
            self.project_root / "models",
            self.project_root / "backups",
        ]
        for target in targets:
            self._remove_path(target)

        if remove_venv:
            self._remove_path(self.venv_path)

        for cache_dir in list(self.project_root.rglob("__pycache__")):
            self._remove_path(cache_dir)

        setup_log = self.project_root / "setup.log"
        self._remove_path(setup_log)

        if self.log_file.exists():
            self.logger.info("Removing deployment log %s", self._rel(self.log_file))
        handlers = list(self.logger.handlers)
        for handler in handlers:
            handler.flush()
            handler.close()
            self.logger.removeHandler(handler)
        if self.log_file.exists():
            try:
                self.log_file.unlink()
            except OSError as exc:
                raise DeploymentError(f"Failed to remove {self._rel(self.log_file)}: {exc}") from exc

    def _pip_path(self) -> str:
        if sys.platform.startswith("win"):
            pip_path = self.venv_path / "Scripts" / "pip.exe"
        else:
            pip_path = self.venv_path / "bin" / "pip"
        if not pip_path.exists():
            raise DeploymentError("pip executable not found inside the virtual environment.")
        return str(pip_path)

    def _run(self, command: List[str], check: bool = True) -> subprocess.CompletedProcess:
        self.logger.info("Executing command: %s", " ".join(command))
        try:
            result = subprocess.run(command, cwd=self.project_root)
        except FileNotFoundError as exc:
            raise DeploymentError(f"Command not found: {command[0]}") from exc
        if check and result.returncode != 0:
            raise DeploymentError(f"Command failed ({result.returncode}): {' '.join(command)}")
        return result

    def _remove_path(self, path: Path) -> None:
        if not path.exists():
            return
        try:
            if path.is_dir():
                shutil.rmtree(path)
                self.logger.info("Removed directory %s", self._rel(path))
            else:
                path.unlink()
                self.logger.info("Removed file %s", self._rel(path))
        except OSError as exc:
            raise DeploymentError(f"Failed to remove {self._rel(path)}: {exc}") from exc

    def _rel(self, path: Path) -> str:
        try:
            return str(path.relative_to(self.project_root))
        except ValueError:
            return str(path)


app = typer.Typer(help="Manage deployment for the CrewAI + Ollama stack.")


def build_manager(use_prod: bool) -> DeploymentManager:
    project_root = Path(__file__).resolve().parent.parent
    compose_overrides: List[Path] = []
    default_override = project_root / "docker-compose.override.yml"
    if default_override.exists():
        compose_overrides.append(default_override)
    if use_prod:
        prod_file = project_root / "docker-compose.prod.yml"
        if not prod_file.exists():
            raise DeploymentError("docker-compose.prod.yml requested but not found.")
        compose_overrides.append(prod_file)
    return DeploymentManager(project_root, compose_files=compose_overrides)


@app.command()
def deploy(
    build: bool = typer.Option(True, "--build/--no-build", help="Rebuild images before starting services."),
    pull_images: bool = typer.Option(True, "--pull/--no-pull", help="Pull latest container images before deploy."),
    skip_deps: bool = typer.Option(False, "--skip-deps", help="Skip Python dependency installation."),
    prod: bool = typer.Option(False, "--prod", help="Include docker-compose.prod.yml overrides."),
) -> None:
    """
    Deploy the CrewAI + Ollama stack. Ensures dependencies, prepares configuration,
    installs Python requirements, and starts Docker services.
    """
    try:
        manager = build_manager(prod)
        typer.secho("Checking system prerequisites...", fg="cyan")
        manager.ensure_prerequisites()
        if not skip_deps:
            typer.secho("Ensuring Python virtual environment...", fg="cyan")
            manager.ensure_virtualenv()
            typer.secho("Installing Python dependencies...", fg="cyan")
            manager.install_requirements()
        typer.secho("Preparing directories and configuration...", fg="cyan")
        manager.prepare_environment()
        if pull_images:
            typer.secho("Pulling latest container images...", fg="cyan")
            manager.pull_images()
        typer.secho("Starting Docker services...", fg="cyan")
        manager.start_services(build=build)
        typer.secho("Deployment completed successfully ✅", fg=typer.colors.GREEN)
    except DeploymentError as exc:
        typer.secho(f"Deployment failed: {exc}", fg=typer.colors.RED)
        raise typer.Exit(code=1)


@app.command()
def status(
    prod: bool = typer.Option(False, "--prod", help="Include docker-compose.prod.yml overrides."),
) -> None:
    """Show the current status of all services."""
    try:
        manager = build_manager(prod)
        manager.ensure_prerequisites()
        typer.secho("Current service status:", fg="cyan")
        manager.show_status()
    except DeploymentError as exc:
        typer.secho(f"Unable to get status: {exc}", fg=typer.colors.RED)
        raise typer.Exit(code=1)


@app.command()
def rollback(
    force: bool = typer.Option(False, "--force", "-f", help="Skip confirmation and continue on errors."),
    keep_venv: bool = typer.Option(False, "--keep-venv", help="Preserve the existing Python virtual environment."),
    prod: bool = typer.Option(False, "--prod", help="Include docker-compose.prod.yml overrides."),
) -> None:
    """
    Fully roll back the deployment: stop containers, remove Docker resources, and
    delete generated local artifacts (data, logs, models, backups, virtualenv).
    """
    if not force:
        confirm = typer.confirm(
            "This will stop all running services, remove containers, volumes, and delete generated files. Continue?"
        )
        if not confirm:
            typer.secho("Rollback aborted.", fg=typer.colors.YELLOW)
            raise typer.Exit(code=0)

    compose_available = True
    try:
        manager = build_manager(prod)
    except DeploymentError as exc:
        typer.secho(f"Unable to initialise rollback: {exc}", fg=typer.colors.RED)
        raise typer.Exit(code=1)

    try:
        manager.ensure_prerequisites()
    except DeploymentError as exc:
        compose_available = False
        typer.secho(f"Warning: {exc}", fg=typer.colors.YELLOW)
        if not force:
            typer.secho("Use --force to clean local files without Docker cleanup.", fg=typer.colors.YELLOW)
            raise typer.Exit(code=1)

    if compose_available:
        try:
            typer.secho("Stopping Docker services and removing resources...", fg="cyan")
            manager.stop_services()
        except DeploymentError as exc:
            typer.secho(f"Error stopping services: {exc}", fg=typer.colors.RED)
            if not force:
                raise typer.Exit(code=1)
            typer.secho("Continuing with local cleanup because --force was provided.", fg=typer.colors.YELLOW)

    try:
        typer.secho("Cleaning local artifacts...", fg="cyan")
        manager.clean_artifacts(remove_venv=not keep_venv)
        typer.secho("Rollback completed successfully ✅", fg=typer.colors.GREEN)
    except DeploymentError as exc:
        typer.secho(f"Rollback cleanup failed: {exc}", fg=typer.colors.RED)
        raise typer.Exit(code=1)


def main() -> None:
    app()


if __name__ == "__main__":
    main()
