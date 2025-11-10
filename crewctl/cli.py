"""Typer-based CLI for managing CrewAI agents and Ollama models."""

from __future__ import annotations

from typing import Optional

import typer
from rich import box
from rich.table import Table

from .agent import AgentManager
from .config import CrewCtlConfig
from .model import ModelManager
from .training import TrainingJob, train_lrm
from .utils import CrewCtlError, console, format_bytes, read_env_value

app = typer.Typer(help="CrewAI + Ollama management CLI", add_completion=False)
agent_app = typer.Typer(help="Manage CrewAI agents", add_completion=False)
model_app = typer.Typer(help="Manage Ollama models", add_completion=False)
train_app = typer.Typer(help="Train adapters / LRMs", add_completion=False)

app.add_typer(agent_app, name="agent")
app.add_typer(model_app, name="model")
app.add_typer(train_app, name="train")


def _get_config(ctx: typer.Context) -> CrewCtlConfig:
    if "config" not in ctx.obj:
        ctx.obj["config"] = CrewCtlConfig()
    return ctx.obj["config"]


def _get_agent_manager(ctx: typer.Context) -> AgentManager:
    if "agent_manager" not in ctx.obj:
        ctx.obj["agent_manager"] = AgentManager(_get_config(ctx))
    return ctx.obj["agent_manager"]


def _get_model_manager(ctx: typer.Context) -> ModelManager:
    if "model_manager" not in ctx.obj:
        ctx.obj["model_manager"] = ModelManager(_get_config(ctx))
    return ctx.obj["model_manager"]


@app.callback(invoke_without_command=True)
def main(ctx: typer.Context) -> None:
    """Initialize CLI context."""
    if ctx.invoked_subcommand:
        return
    config = _get_config(ctx)
    console.print(f"[bold green]crewctl[/] ready. Project root: {config.paths.root}")
    console.print("Use `crewctl --help` to list commands.")


@app.command()
def doctor(ctx: typer.Context) -> None:
    """Run environment diagnostics."""
    config = _get_config(ctx)
    model_manager = _get_model_manager(ctx)
    active_env_model = read_env_value(config.paths.env_file, "OLLAMA_MODEL") or "<not set>"
    registry_entry = model_manager.registry.active

    table = Table(box=box.SIMPLE_HEAVY)
    table.add_column("Check")
    table.add_column("Status")

    table.add_row("Project root", str(config.paths.root))
    table.add_row("Virtualenv", str(config.paths.root / ".crewctl" / "venv"))
    table.add_row("Ollama binary", str(config.ollama_binary))
    table.add_row("Active model (.env)", active_env_model)
    table.add_row("Registry active", registry_entry.model if registry_entry else "—")

    console.print(table)


@agent_app.command("list")
def list_agents(ctx: typer.Context) -> None:
    """List registered agent definitions."""
    manager = _get_agent_manager(ctx)
    agents = manager.list_agents()
    if not agents:
        console.print("[yellow]No agents found. Use `crewctl agent create` to add one.")
        raise typer.Exit(code=0)

    table = Table(title="Agents", box=box.MINIMAL_DOUBLE_HEAD)
    table.add_column("Name")
    table.add_column("Path")

    for agent in agents:
        table.add_row(agent.stem, str(agent))

    console.print(table)


@agent_app.command("create")
def create_agent(
    ctx: typer.Context,
    name: str = typer.Argument(..., help="Human friendly agent name"),
    role: Optional[str] = typer.Option(None, help="Agent role description"),
    goal: Optional[str] = typer.Option(None, help="Primary goal"),
    backstory: Optional[str] = typer.Option(None, help="Backstory context"),
    overwrite: bool = typer.Option(False, "--overwrite", help="Overwrite existing agent file"),
) -> None:
    """Create a new agent template."""
    manager = _get_agent_manager(ctx)
    manager.create_agent(name=name, role=role, goal=goal, backstory=backstory, overwrite=overwrite)


@model_app.command("list")
def list_models(ctx: typer.Context) -> None:
    """List models available via Ollama."""
    manager = _get_model_manager(ctx)
    models = manager.list_models()
    if not models:
        console.print("[yellow]No models found in Ollama. Use `crewctl model pull <name>`.")
        return

    table = Table(title="Installed Ollama Models", box=box.MINIMAL)
    table.add_column("Name")
    table.add_column("Digest", overflow="fold")
    table.add_column("Size")
    table.add_column("Modified")

    for model in models:
        table.add_row(
            model.get("name", "unknown"),
            (model.get("digest") or "")[:12] + "…" if model.get("digest") else "—",
            format_bytes(model.get("size", 0)) if model.get("size") else "—",
            model.get("modified_at", "—"),
        )

    console.print(table)


@model_app.command("pull")
def pull_model(ctx: typer.Context, name: str = typer.Argument(..., help="Model name to pull")) -> None:
    """Download a model into Ollama."""
    manager = _get_model_manager(ctx)
    manager.pull_model(name)


@model_app.command("use")
def use_model(ctx: typer.Context, name: str = typer.Argument(..., help="Model name to activate")) -> None:
    """Mark a model as the default for CrewAI."""
    manager = _get_model_manager(ctx)
    manager.use_model(name)


@model_app.command("show")
def show_model(ctx: typer.Context, name: str = typer.Argument(..., help="Model name to inspect")) -> None:
    """Show detailed model metadata."""
    manager = _get_model_manager(ctx)
    details = manager.show_model(name)
    console.print_json(data=details if isinstance(details, dict) else {"raw": str(details)})


@model_app.command("history")
def model_history(ctx: typer.Context) -> None:
    """Display model activation history."""
    manager = _get_model_manager(ctx)
    table = manager.history_table()
    console.print(table)


@model_app.command("activate")
def activate_model(
    ctx: typer.Context,
    identifier: str = typer.Argument(..., help="History index, ID prefix, or model name"),
) -> None:
    """Activate a previously used model."""
    manager = _get_model_manager(ctx)
    try:
        idx = int(identifier)
        identifier_value: Optional[object] = idx
    except ValueError:
        identifier_value = identifier
    manager.activate(identifier_value)


@train_app.command("lrm")
def train_lrm_cmd(
    ctx: typer.Context,
    dataset: str = typer.Argument(..., help="Dataset path or HuggingFace dataset id"),
    base_model: str = typer.Option("llama3", help="Base Ollama model identifier"),
    output_name: str = typer.Option("custom-lrm", help="Name for the trained adapter"),
    epochs: int = typer.Option(3, min=1, help="Training epochs"),
    batch_size: int = typer.Option(2, min=1, help="Per-device batch size"),
    learning_rate: float = typer.Option(2e-4, help="Learning rate"),
    lora_r: int = typer.Option(16, help="LoRA rank"),
    lora_alpha: int = typer.Option(32, help="LoRA alpha"),
    lora_dropout: float = typer.Option(0.05, help="LoRA dropout"),
    register: bool = typer.Option(False, "--register", help="Register resulting adapter as an Ollama model"),
) -> None:
    """Fine-tune a base model into an LRM (LoRA adapter)."""
    job = TrainingJob(
        base_model=base_model,
        dataset=dataset,
        output_name=output_name,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        register_with_ollama=register,
    )
    config = _get_config(ctx)
    train_lrm(job, config)


def run_cli() -> None:
    """Wrapper to execute the Typer app with error handling."""
    try:
        app(standalone_mode=False, obj={})
    except CrewCtlError as exc:
        console.print(f"[bold red]Error:[/] {exc}")
        raise typer.Exit(code=1) from exc
    except typer.Exit:
        raise
    except Exception as exc:  # pragma: no cover - fallback
        console.print(f"[bold red]Unhandled error:[/] {exc}")
        raise typer.Exit(code=1) from exc


if __name__ == "__main__":
    run_cli()
