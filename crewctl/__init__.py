"""
crewctl
=======

Management CLI for the CrewAI + Ollama stack.

The package exposes a Typer application (`app`) that is invoked by
``python -m crewctl`` or the `crewctl` shim created during deployment.
"""

from .cli import app

__all__ = ["app"]
