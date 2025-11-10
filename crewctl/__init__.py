"""crewctl package exposing deployment-aware CLI utilities."""

from __future__ import annotations

from pathlib import Path

__all__ = ["__version__", "resolve_project_root"]

__version__ = "0.1.0"


def resolve_project_root() -> Path:
    """Resolve the CrewAI project root from the CREWCTL_ROOT env var."""
    from .utils import resolve_project_root  # Local import to avoid circular dependency

    return resolve_project_root()
