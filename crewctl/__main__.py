"""Entrypoint for crewctl CLI."""

from __future__ import annotations

from .cli import run_cli


def main() -> None:
    """Execute the Typer app."""
    run_cli()


if __name__ == "__main__":
    main()
