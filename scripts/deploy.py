#!/usr/bin/env python3
"""
Self-contained deployment utility for the CrewAI + Ollama stack.

Features:
  - Dependency verification and installation.
  - Environment preparation (directories, config templates).
  - Docker Compose orchestration for deploy and status.
  - Full rollback that stops services, removes containers/volumes/networks,
    and deletes generated runtime artifacts.
"""

from __future__ import annotations

import argparse
import logging
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import List, Sequence


LOG_FORMAT = "%(asctime)s %(levelname)s %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger("deploy")


PROMETHEUS_CONFIG = textwrap.dedent(
    """
    global:
      scrape_interval: 15s
      evaluation_interval: 15s

    scrape_configs:
      - job_name: 'crewai-api'
        metrics_path: /metrics
        static_configs:
          - targets:
              - crewai:9090
      - job_name: 'ollama'
        metrics_path: /
        static_configs:
          - targets:
              - ollama:11434
    """
).strip() + "\n"


GRAFANA_DATASOURCE = textwrap.dedent(
    """
    apiVersion: 1

    datasources:
      - name: Prometheus
        type: prometheus
        access: proxy
        isDefault: true
        url: http://prometheus:9090
        editable: true
    """
).strip() + "\n"


GRAFANA_DASHBOARD_PROVIDER = textwrap.dedent(
    """
    apiVersion: 1

    providers:
      - name: 'default'
        orgId: 1
        folder: ''
        type: file
        disableDeletion: false
        allowUiUpdates: true
        options:
          path: /etc/grafana/provisioning/dashboards
    """
).strip() + "\n"


GRAFANA_SAMPLE_DASHBOARD = textwrap.dedent(
    """
    {
      "annotations": {
        "list": [
          {
            "builtIn": 1,
            "datasource": { "type": "datasource", "uid": "grafana" },
            "enable": true,
            "hide": true,
            "iconColor": "rgba(0, 211, 255, 1)",
            "name": "Annotations & Alerts",
            "type": "dashboard"
          }
        ]
      },
      "editable": true,
      "fiscalYearStartMonth": 0,
      "graphTooltip": 0,
      "panels": [
        {
          "datasource": { "type": "prometheus", "uid": "prometheus" },
          "fieldConfig": {
            "defaults": {
              "color": { "mode": "palette-classic" },
              "mappings": [],
              "thresholds": {
                "mode": "absolute",
                "steps": [
                  { "color": "green", "value": null },
                  { "color": "red", "value": 80 }
                ]
              }
            },
            "overrides": []
          },
          "gridPos": { "h": 8, "w": 12, "x": 0, "y": 0 },
          "id": 1,
          "options": {
            "legend": {
              "calcs": [ "mean" ],
              "displayMode": "table",
              "placement": "bottom",
              "showLegend": true
            },
            "tooltip": { "mode": "single", "sort": "none" }
          },
          "targets": [
            {
              "expr": "sum(rate(http_requests_total[1m]))",
              "legendFormat": "{{method}} {{handler}}",
              "refId": "A"
            }
          ],
          "title": "CrewAI Request Rate",
          "type": "timeseries"
        }
      ],
      "refresh": "30s",
      "schemaVersion": 38,
      "style": "dark",
      "tags": [ "crewai" ],
      "templating": { "list": [] },
      "time": { "from": "now-6h", "to": "now" },
      "timepicker": {},
      "timezone": "",
      "title": "CrewAI Overview",
      "uid": "crewai-overview",
      "version": 1,
      "weekStart": ""
    }
    """
).strip() + "\n"


class DeployManager:
    """Encapsulates deployment lifecycle tasks."""

    CONTAINER_NAMES = [
        "crewai-service",
        "ollama-service",
        "redis-service",
        "prometheus-service",
        "grafana-service",
    ]

    VOLUME_NAMES = [
        "workspace_redis_data",
        "workspace_prometheus_data",
        "workspace_grafana_data",
        "redis_data",
        "prometheus_data",
        "grafana_data",
    ]

    NETWORK_NAMES = [
        "workspace_crewai-network",
        "crewai-network",
    ]

    RUNTIME_DIRECTORIES = [
        Path("data/ollama"),
        Path("data/crewai"),
        Path("logs"),
        Path("models"),
        Path("config/grafana/datasources"),
        Path("config/grafana/dashboards"),
    ]

    def __init__(self, project_root: Path | None = None) -> None:
        self.project_root = project_root or Path(__file__).resolve().parent.parent
        self.requirements_file = self.project_root / "requirements.txt"
        self.compose_file = self.project_root / "docker-compose.yml"
        self.compose_override = self.project_root / "docker-compose.override.yml"
        self.compose_prod = self.project_root / "docker-compose.prod.yml"
        self.compose_command = self._detect_compose_command()

    def deploy(self, prod: bool = False) -> None:
        """Perform a full deployment."""
        logger.info("Starting deployment (prod=%s)", prod)
        self._verify_prerequisites()
        self._install_python_dependencies()
        self._prepare_runtime_environment()
        self._compose_run(["pull"], allow_missing=True, prod=prod)
        self._compose_run(["up", "-d", "--build"], prod=prod)
        logger.info("Deployment completed successfully.")

    def status(self, prod: bool = False) -> None:
        """Show status of the stack."""
        self._verify_prerequisites()
        self._compose_run(["ps"], prod=prod)

    def rollback(self, prod: bool = False, assume_yes: bool = False) -> None:
        """Stop all services, remove artifacts, and clean up Docker resources."""
        if not assume_yes and not self._confirm_destruction():
            logger.info("Rollback aborted by user.")
            return

        logger.info("Initiating rollback (prod=%s)...", prod)
        self._verify_prerequisites()
        self._stop_named_containers()
        self._compose_run(["down", "--volumes", "--remove-orphans"], allow_missing=True, prod=prod)
        self._remove_named_volumes()
        self._remove_named_networks()
        self._delete_runtime_artifacts()
        logger.info("Rollback completed. All managed resources have been removed.")

    # Internal helpers -----------------------------------------------------

    def _verify_prerequisites(self) -> None:
        if not shutil.which("docker"):
            raise RuntimeError("Docker is required but not found on PATH.")
        if not self.compose_command:
            raise RuntimeError("Docker Compose command could not be determined.")
        if not self.compose_file.exists():
            raise RuntimeError(f"Missing docker-compose file: {self.compose_file}")
        logger.debug("Prerequisites verified.")

    def _install_python_dependencies(self) -> None:
        if not self.requirements_file.exists():
            logger.warning("requirements.txt not found; skipping Python dependency installation.")
            return

        logger.info("Installing Python dependencies from %s", self.requirements_file)
        cmd = [sys.executable, "-m", "pip", "install", "-r", str(self.requirements_file)]
        self._run_command(cmd)

    def _prepare_runtime_environment(self) -> None:
        logger.info("Preparing runtime directories and configuration templates...")
        for relative_path in self.RUNTIME_DIRECTORIES:
            path = self.project_root / relative_path
            path.mkdir(parents=True, exist_ok=True)
            logger.debug("Ensured directory exists: %s", path)

        self._ensure_file(self.project_root / "config" / "prometheus.yml", PROMETHEUS_CONFIG)
        self._ensure_file(
            self.project_root / "config" / "grafana" / "datasources" / "datasource.yml",
            GRAFANA_DATASOURCE,
        )
        self._ensure_file(
            self.project_root / "config" / "grafana" / "dashboards" / "dashboard.yml",
            GRAFANA_DASHBOARD_PROVIDER,
        )
        self._ensure_file(
            self.project_root / "config" / "grafana" / "dashboards" / "crewai-overview.json",
            GRAFANA_SAMPLE_DASHBOARD,
        )

    def _stop_named_containers(self) -> None:
        docker_path = shutil.which("docker")
        assert docker_path is not None  # Guarded by _verify_prerequisites

        for name in self.CONTAINER_NAMES:
            logger.debug("Stopping container %s (if present)...", name)
            self._run_command([docker_path, "rm", "-f", name], check=False)

    def _remove_named_volumes(self) -> None:
        docker_path = shutil.which("docker")
        assert docker_path is not None

        for volume in self.VOLUME_NAMES:
            logger.debug("Removing volume %s (if present)...", volume)
            self._run_command([docker_path, "volume", "rm", volume], check=False)

    def _remove_named_networks(self) -> None:
        docker_path = shutil.which("docker")
        assert docker_path is not None

        for network in self.NETWORK_NAMES:
            logger.debug("Removing network %s (if present)...", network)
            self._run_command([docker_path, "network", "rm", network], check=False)

    def _delete_runtime_artifacts(self) -> None:
        targets = [
            self.project_root / "data",
            self.project_root / "logs",
            self.project_root / "models",
            self.project_root / "scripts" / "__pycache__",
        ]

        for target in targets:
            if target.exists():
                if target.is_file() or target.is_symlink():
                    logger.debug("Removing file %s", target)
                    target.unlink(missing_ok=True)
                else:
                    logger.debug("Removing directory tree %s", target)
                    shutil.rmtree(target, ignore_errors=True)

    def _compose_run(
        self,
        args: Sequence[str],
        *,
        allow_missing: bool = False,
        prod: bool = False,
    ) -> None:
        command = list(self.compose_command)
        command.extend(["-f", str(self.compose_file)])
        if prod and self.compose_prod.exists():
            command.extend(["-f", str(self.compose_prod)])
        elif self.compose_override.exists():
            command.extend(["-f", str(self.compose_override)])
        command.extend(args)

        logger.debug("Running compose command: %s", " ".join(command))
        self._run_command(command, check=not allow_missing)

    def _run_command(self, command: Sequence[str], *, check: bool = True) -> subprocess.CompletedProcess:
        try:
            result = subprocess.run(command, check=check)
            return result
        except subprocess.CalledProcessError as exc:
            if check:
                raise RuntimeError(
                    f"Command failed ({exc.returncode}): {' '.join(command)}"
                ) from exc
            return exc

    def _ensure_file(self, path: Path, contents: str) -> None:
        if path.exists():
            logger.debug("File already exists, skipping: %s", path)
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(contents, encoding="utf-8")
        logger.info("Created default configuration file: %s", path)

    def _confirm_destruction(self) -> bool:
        prompt = (
            "This will stop all containers, delete Docker volumes/networks, and remove "
            "runtime data directories. Continue? [y/N]: "
        )
        response = input(prompt).strip().lower()
        return response in {"y", "yes"}

    @staticmethod
    def _detect_compose_command() -> List[str] | None:
        docker_path = shutil.which("docker")
        if docker_path:
            try:
                subprocess.run([docker_path, "compose", "version"], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                return [docker_path, "compose"]
            except subprocess.CalledProcessError:
                pass

        legacy_path = shutil.which("docker-compose")
        if legacy_path:
            return [legacy_path]

        return None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Deploy and manage the CrewAI + Ollama stack.")
    parser.add_argument(
        "--prod",
        action="store_true",
        help="Use production docker-compose overrides during deploy/status/rollback.",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Automatically confirm destructive rollback actions.",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("deploy", help="Install dependencies, prepare environment, and start services.")
    subparsers.add_parser("status", help="Display docker-compose service status.")
    subparsers.add_parser("rollback", help="Stop services and delete all generated resources.")

    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    manager = DeployManager()

    if args.command == "deploy":
        manager.deploy(prod=args.prod)
    elif args.command == "status":
        manager.status(prod=args.prod)
    elif args.command == "rollback":
        manager.rollback(prod=args.prod, assume_yes=args.yes)
    else:
        parser.error(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
