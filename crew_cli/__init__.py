"""
Crew control utilities for managing agents and Ollama models.
"""

from importlib import metadata


def get_version() -> str:
    """Return the package version if available."""
    try:
        return metadata.version("crew-cli")
    except metadata.PackageNotFoundError:
        return "0.1.0"
