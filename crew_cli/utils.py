import hashlib
import platform
import stat
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import yaml
from jinja2 import Environment, FileSystemLoader, TemplateNotFound


class DeploymentError(RuntimeError):
    """Raised when deployment-related operations fail."""


def project_root() -> Path:
    """Return the repository root."""
    return Path(__file__).resolve().parent.parent


def ensure_directory(path: Path) -> None:
    """Create a directory if it does not exist."""
    path.mkdir(parents=True, exist_ok=True)


def load_yaml(path: Path, default: Optional[Any] = None) -> Any:
    """Load YAML content from a path."""
    if not path.exists():
        return default
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
        return data if data is not None else default


def dump_yaml(path: Path, data: Any) -> None:
    """Write YAML data to the path."""
    ensure_directory(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle, sort_keys=False)


def load_json(path: Path, default: Optional[Any] = None) -> Any:
    """Load JSON data from a path."""
    if not path.exists():
        return default
    import json

    with path.open("r", encoding="utf-8") as handle:
        try:
            return json.load(handle)
        except json.JSONDecodeError as exc:
            raise DeploymentError(f"Invalid JSON in {path}: {exc}") from exc


def dump_json(path: Path, data: Any) -> None:
    """Write JSON data to the path."""
    import json

    ensure_directory(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2)


def timestamp() -> str:
    """Return an ISO 8601 timestamp."""
    return datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def slugify(value: str) -> str:
    """Generate a safe slug for filesystem usage."""
    import re

    slug = re.sub(r"[^a-zA-Z0-9]+", "_", value.strip())
    slug = re.sub(r"_+", "_", slug)
    return slug.strip("_").lower()


def render_template(template_path: Path, **context: Any) -> str:
    """Render a Jinja template located relative to project root."""
    template_dir = template_path.parent
    env = Environment(loader=FileSystemLoader(str(template_dir)))
    try:
        template = env.get_template(template_path.name)
    except TemplateNotFound as exc:
        raise DeploymentError(f"Template not found: {template_path}") from exc
    return template.render(**context)


def make_executable(path: Path) -> None:
    """Ensure a script is executable on POSIX systems."""
    if platform.system() != "Windows":
        current_mode = path.stat().st_mode
        path.chmod(current_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def compute_sha256(path: Path) -> str:
    """Compute SHA256 digest for a file."""
    sha = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8192), b""):
            sha.update(chunk)
    return sha.hexdigest()


def run_command(
    command: Iterable[str],
    *,
    env: Optional[Dict[str, str]] = None,
    check: bool = True,
    capture_output: bool = False,
) -> subprocess.CompletedProcess:
    """Execute a command, raising DeploymentError on failure."""
    result = subprocess.run(
        list(command),
        env=env,
        check=False,
        capture_output=capture_output,
        text=True,
    )
    if check and result.returncode != 0:
        stdout = result.stdout.strip() if result.stdout else ""
        stderr = result.stderr.strip() if result.stderr else ""
        message = f"Command {' '.join(command)} failed ({result.returncode})"
        if stdout:
            message += f"\nstdout: {stdout}"
        if stderr:
            message += f"\nstderr: {stderr}"
        raise DeploymentError(message)
    return result


def platform_key() -> str:
    """Return a normalized platform key for release metadata."""
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


def sanitize_permissions(path: Path) -> None:
    """Restrict permissions for sensitive files on POSIX systems."""
    if platform.system() == "Windows":
        return
    secure_mode = stat.S_IRUSR | stat.S_IWUSR
    if path.exists():
        path.chmod(secure_mode)

