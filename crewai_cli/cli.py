from __future__ import annotations

import typer
from rich.console import Console

from . import config
from .agents import app as agents_app
from .models import app as models_app

console = Console()

app = typer.Typer(
    help="Command-line interface for managing CrewAI agents and Ollama models.",
    no_args_is_help=True,
    add_completion=False,
    rich_markup_mode="markdown",
)
app.add_typer(agents_app, name="agents", help="Manage CrewAI agents")
app.add_typer(models_app, name="models", help="Manage Ollama models and versions")


@app.callback(invoke_without_command=True)
def main(ctx: typer.Context) -> None:
    """
    Ensure runtime prerequisites are in place before handling sub-commands.
    """
    config.ensure_runtime()
    if ctx.invoked_subcommand is None:
        console.print(":rocket: **CrewAI CLI** ready. See `crewaictl --help` for usage.")
