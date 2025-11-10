#!/usr/bin/env python3
"""
Deployment and rollback orchestration for the CrewAI + Ollama stack.

This script is designed to be self-contained: it validates host dependencies,
manages a local Python virtual environment, installs application requirements,
prepares runtime directories, deploys Docker resources, and provides a
destructive rollback routine that stops containers and removes generated
artifacts.
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


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
COMPOSE_FILE = PROJECT_ROOT / "docker-compose.yml"
COMPOSE_OVERRIDE = PROJECT_ROOT / "docker-compose.override.yml"
DEFAULT_PROJECT_NAME = os.getenv("COMPOSE_PROJECT_NAME", PROJECT_ROOT.name.replace("-", "_"))
VENV_DIR = PROJECT_ROOT / ".venv"
REQUIREMENTS_FILE = PROJECT_ROOT / "requirements.txt"
SETUP_LOG = PROJECT_ROOT / "setup.log"
DATA_DIRECTORIES = [
    PROJECT_ROOT / "data",
    PROJECT_ROOT / "logs",
    PROJECT_ROOT / "models",
    PROJECT_ROOT / "backups",
]
ENV_FILE = PROJECT_ROOT / ".env"
ENV_TEMPLATE_FILE = PROJECT_ROOT / ".env.template"

LOG_LEVEL = os.getenv("DEPLOY_LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("deploy")


# -----------------------------------------------------------------------------
# Utility helpers
# -----------------------------------------------------------------------------

class DeploymentError(RuntimeError):
    """Raised when a deployment step fails."""


def run_command(
    command: Sequence[str],
    *,
    description: str,
    cwd: Path | None = PROJECT_ROOT,
    check: bool = True,
    capture_output: bool = False,
) -> subprocess.CompletedProcess:
    """Execute a shell command with optional error handling."""
    log_cmd = " ".join(command)
    logger.debug("Executing command (%s): %s", description, log_cmd)

    result = subprocess.run(
        command,
        cwd=cwd,
        check=False,
        text=True,
        capture_output=capture_output,
    )

    if result.returncode != 0:
        msg = f"{description} failed with exit code {result.returncode}"
        if capture_output:
            msg += f"\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        logger.error(msg)
        if check:
            raise DeploymentError(msg)
    else:
        logger.debug("%s succeeded", description)

    return result


def which(command: str) -> str | None:
    """Wrapper around shutil.which with logging."""
    path = shutil.which(command)
    logger.debug("Command '%s' resolved to: %s", command, path)
    return path


def confirm(prompt: str) -> bool:
    """Prompt the user for a yes/no confirmation."""
    try:
        response = input(f"{prompt} [y/N]: ").strip().lower()
    except EOFError:
        return False
    return response in {"y", "yes"}


# -----------------------------------------------------------------------------
# Dependency management
# -----------------------------------------------------------------------------

def detect_compose_command() -> List[str]:
    """
    Determine which Docker Compose invocation is available.

    Returns:
        List[str]: The base command to run docker compose actions.
    Raises:
        DeploymentError: If neither docker compose nor docker-compose is available.
    """
    compose_project_flag = ["-p", DEFAULT_PROJECT_NAME]
    compose_file_flags: List[str] = ["-f", str(COMPOSE_FILE)]

    if COMPOSE_OVERRIDE.exists():
        compose_file_flags.extend(["-f", str(COMPOSE_OVERRIDE)])

    compose_candidates = [
        (["docker", "compose"], "docker compose"),
        (["docker-compose"], "docker-compose"),
    ]

    for command, label in compose_candidates:
        binary = command[0]
        if which(binary) is None:
            continue

        try:
            run_command(
                command + ["version"],
                description=f"Checking {label} availability",
                capture_output=True,
            )
            logger.info("Using %s for orchestration", label)
            return command + compose_project_flag + compose_file_flags
        except DeploymentError:
            continue

    raise DeploymentError(
        "Docker Compose is required but neither 'docker compose' nor 'docker-compose' "
        "is available. Install Docker Desktop or the Docker Compose plugin."
    )


def ensure_host_dependencies():
    """Validate host-level dependencies required for deployment."""
    logger.info("Validating host dependencies...")

    if which("docker") is None:
        raise DeploymentError(
            "Docker CLI not found. Install Docker Engine or Docker Desktop before continuing."
        )

    try:
        run_command(
            ["docker", "info"],
            description="Checking Docker daemon",
            capture_output=True,
        )
    except DeploymentError as exc:
        raise DeploymentError(
            "Docker daemon is not reachable. Ensure Docker is running and your user "
            "has sufficient privileges."
        ) from exc

    detect_compose_command()
    logger.info("Host dependencies look good.")


def ensure_virtualenv():
    """Create (if needed) and return the path to the project virtual environment."""
    if VENV_DIR.exists():
        logger.info("Using existing virtual environment at %s", VENV_DIR)
    else:
        logger.info("Creating project virtual environment at %s", VENV_DIR)
        run_command(
            [sys.executable, "-m", "venv", str(VENV_DIR)],
            description="Creating virtual environment",
        )

    if os.name == "nt":
        pip_executable = VENV_DIR / "Scripts" / "pip.exe"
        python_executable = VENV_DIR / "Scripts" / "python.exe"
    else:
        pip_executable = VENV_DIR / "bin" / "pip"
        python_executable = VENV_DIR / "bin" / "python"

    if not pip_executable.exists():
        raise DeploymentError(f"pip not found in virtual environment at {pip_executable}")

    return python_executable, pip_executable


def install_python_dependencies():
    """Install Python dependencies inside the managed virtual environment."""
    if not REQUIREMENTS_FILE.exists():
        logger.warning("requirements.txt not found; skipping Python dependency installation.")
        return

    python_executable, pip_executable = ensure_virtualenv()

    logger.info("Upgrading pip inside the virtual environment...")
    run_command(
        [str(pip_executable), "install", "--upgrade", "pip", "setuptools", "wheel"],
        description="Upgrading pip",
    )

    logger.info("Installing project dependencies from requirements.txt...")
    run_command(
        [str(pip_executable), "install", "-r", str(REQUIREMENTS_FILE)],
        description="Installing Python dependencies",
    )

    setup_file = PROJECT_ROOT / "setup.py"
    if setup_file.exists():
        logger.info("Installing project package in editable mode...")
        run_command(
            [str(pip_executable), "install", "-e", str(PROJECT_ROOT)],
            description="Installing package editable",
        )


def ensure_environment_files():
    """Ensure that an .env exists, deriving it from the template if needed."""
    if ENV_FILE.exists():
        logger.info(".env file already present.")
        return

    if not ENV_TEMPLATE_FILE.exists():
        logger.warning(".env template not found; skipping environment bootstrap.")
        return

    logger.info("Creating .env from template...")
    shutil.copyfile(ENV_TEMPLATE_FILE, ENV_FILE)


def ensure_directories():
    """Create runtime directories required by the stack."""
    for directory in DATA_DIRECTORIES:
        if not directory.exists():
            logger.info("Creating directory %s", directory.relative_to(PROJECT_ROOT))
            directory.mkdir(parents=True, exist_ok=True)


# -----------------------------------------------------------------------------
# Deployment operations
# -----------------------------------------------------------------------------

def docker_compose(*args: str):
    """Execute a docker compose command with the detected CLI."""
    base_command = detect_compose_command()
    run_command(
        list(base_command) + list(args),
        description=f"docker compose {' '.join(args)}",
    )


def deploy(skip_build: bool = False, no_pull: bool = False):
    """Perform a full deployment of the stack."""
    logger.info("Starting deployment...")
    ensure_host_dependencies()
    ensure_environment_files()
    ensure_directories()
    install_python_dependencies()

    if not no_pull:
        logger.info("Pulling latest container images...")
        docker_compose("pull")

    compose_args = ["up", "-d"]
    if not skip_build:
        compose_args.append("--build")

    logger.info("Bringing up services with docker compose...")
    docker_compose(*compose_args)
    logger.info("Deployment completed successfully.")


def stop_processes(containers: Iterable[str]):
    """Stop and remove specific Docker containers if they are still running."""
    for container in containers:
        if not container:
            continue
        logger.info("Stopping container %s", container)
        run_command(
            ["docker", "rm", "-f", container],
            description=f"Stopping container {container}",
            check=False,
        )


def cleanup_directories(include_venv: bool):
    """Remove data directories and optional virtual environment."""
    for directory in DATA_DIRECTORIES:
        if directory.exists():
            logger.info("Removing directory %s", directory.relative_to(PROJECT_ROOT))
            shutil.rmtree(directory, ignore_errors=True)

    if SETUP_LOG.exists():
        logger.info("Removing setup log file %s", SETUP_LOG.name)
        SETUP_LOG.unlink(missing_ok=True)

    if include_venv and VENV_DIR.exists():
        logger.info("Removing virtual environment at %s", VENV_DIR)
        shutil.rmtree(VENV_DIR, ignore_errors=True)


def rollback(force: bool = False, include_images: bool = False, remove_venv: bool = True):
    """Rollback deployment by stopping services and removing generated artifacts."""
    if not force:
        logger.warning("Rollback is destructive: containers, volumes, data, and optionally the virtualenv will be removed.")
        if not confirm("Proceed with rollback?"):
            logger.info("Rollback cancelled by user.")
            return

    logger.info("Stopping services and removing containers/volumes...")
    docker_compose("down", "--volumes", "--remove-orphans")

    # Catch any lingering containers by name
    result = run_command(
        ["docker", "ps", "-a", "--filter", f"name={DEFAULT_PROJECT_NAME}", "--format", "{{.ID}}"],
        description="Listing project containers",
        capture_output=True,
        check=False,
    )
    if result.stdout:
        containers = [line.strip() for line in result.stdout.splitlines() if line.strip()]
        stop_processes(containers)

    logger.info("Removing project networks if any remain...")
    run_command(
        ["docker", "network", "ls", "--filter", f"name={DEFAULT_PROJECT_NAME}", "--format", "{{.ID}}"],
        description="Listing project networks",
        capture_output=True,
        check=False,
    )
    run_command(
        ["docker", "network", "prune", "-f"],
        description="Pruning unused networks",
        check=False,
    )

    cleanup_directories(include_venv=remove_venv)

    if include_images:
        logger.info("Removing project Docker images...")
        services = ["crewai-service", "ollama-service"]
        for service in services:
            run_command(
                ["docker", "rmi", service],
                description=f"Removing image {service}",
                check=False,
            )
        run_command(
            ["docker", "image", "prune", "-f"],
            description="Pruning dangling images",
            check=False,
        )

    logger.info("Rollback completed.")


def show_status():
    """Display current status of the Docker Compose stack."""
    ensure_host_dependencies()
    docker_compose("ps")


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Deploy and rollback the CrewAI + Ollama stack.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    deploy_parser = subparsers.add_parser("deploy", help="Deploy the stack.")
    deploy_parser.add_argument(
        "--skip-build",
        action="store_true",
        help="Skip rebuilding Docker images (useful when images are already built).",
    )
    deploy_parser.add_argument(
        "--no-pull",
        action="store_true",
        help="Skip pulling updated container images.",
    )

    rollback_parser = subparsers.add_parser(
        "rollback",
        help="Stop services and remove generated artifacts.",
    )
    rollback_parser.add_argument(
        "--force",
        action="store_true",
        help="Do not prompt for confirmation.",
    )
    rollback_parser.add_argument(
        "--keep-venv",
        action="store_true",
        help="Preserve the managed virtual environment during rollback.",
    )
    rollback_parser.add_argument(
        "--remove-images",
        action="store_true",
        help="Also remove Docker images built for the stack.",
    )

    subparsers.add_parser("status", help="Show docker compose status.")
    subparsers.add_parser("check", help="Validate host and Python dependencies without deploying.")

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        if args.command == "deploy":
            deploy(skip_build=args.skip_build, no_pull=args.no_pull)
        elif args.command == "rollback":
            rollback(force=args.force, include_images=args.remove_images, remove_venv=not args.keep_venv)
        elif args.command == "status":
            show_status()
        elif args.command == "check":
            ensure_host_dependencies()
            ensure_environment_files()
            ensure_directories()
            install_python_dependencies()
            logger.info("All dependency checks completed successfully.")
        else:
            parser.print_help()
            return 1
    except DeploymentError as exc:
        logger.error(str(exc))
        return 1
    except KeyboardInterrupt:
        logger.warning("Operation cancelled by user.")
        return 130

    return 0


if __name__ == "__main__":
    sys.exit(main())
