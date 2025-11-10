#!/usr/bin/env python3  # Provide shebang comment compatible with Unix environments
import logging  # Provide logging utilities for situational awareness
import subprocess  # Provide subprocess utilities for dependency checks
import sys  # Provide access to interpreter details and exit codes
from pathlib import Path  # Provide path utilities for filesystem operations
from textwrap import dedent  # Provide dedent helper for formatted instructional text

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")  # Configure logging with timestamped format
logger = logging.getLogger("setup")  # Acquire module-specific logger for contextualized diagnostics


def check_command(command: str) -> bool:  # Define helper to verify command availability securely
    try:  # Provide error handling for subprocess invocation
        subprocess.run([command, "--version"], check=False, capture_output=True)  # Execute command with --version to confirm presence
        return True  # Return True when command invocation succeeds without raising exceptions
    except FileNotFoundError:  # Handle case where command is missing from PATH
        logger.error("Required command '%s' is not available on PATH", command)  # Emit error log identifying missing command
        return False  # Return False to signal failure


def ensure_prerequisites() -> bool:  # Define helper to validate environment prerequisites
    logger.info("Validating local environment prerequisites")  # Emit informational log describing action
    python_ok = sys.version_info >= (3, 10)  # Determine whether running interpreter meets version requirement
    if not python_ok:  # Handle interpreter version failure
        logger.error("Python 3.10 or newer is required")  # Emit error log describing required version
    docker_ok = check_command("docker")  # Verify Docker availability using helper
    compose_ok = check_command("docker-compose")  # Verify Docker Compose availability using helper
    return python_ok and docker_ok and compose_ok  # Return aggregated boolean summarizing prerequisite status


def create_directories(paths: list[str]) -> None:  # Define helper to create project directories securely
    for relative in paths:  # Iterate through requested directory names
        directory = Path(relative)  # Construct Path instance for directory
        directory.mkdir(parents=True, exist_ok=True)  # Create directory tree idempotently
        logger.info("Ensured directory exists: %s", directory)  # Emit informational log confirming directory presence


def print_instructions() -> None:  # Define helper to present post-setup instructions
    message = dedent(
        """
        CrewAI secure setup complete.

        Next steps:
        1. Populate environment variables based on config/settings.py.
        2. Provision Redis and Ollama endpoints reachable by the FastAPI service.
        3. Run `docker compose up --build` to start the stack.
        4. Rotate SECRET_KEY and other secrets before deploying to production.
        """
    ).strip()  # Compose multi-line instructional message and remove surrounding whitespace
    for line in message.splitlines():  # Iterate through each instruction line
        logger.info(line)  # Emit instruction line via logger to maintain formatting


def main() -> int:  # Define entrypoint returning process exit code
    if not ensure_prerequisites():  # Validate environment prerequisites early
        return 1  # Signal failure via non-zero exit code when prerequisites missing
    create_directories(["logs", "data", "backups"])  # Create minimal set of persistent directories
    print_instructions()  # Display next-step instructions for operators
    return 0  # Return zero exit code to signal success


if __name__ == "__main__":  # Execute script logic only when invoked directly
    sys.exit(main())  # Invoke main function and exit interpreter with returned status code
