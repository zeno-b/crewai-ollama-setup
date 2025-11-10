"""Typer-based CLI for managing the CrewAI + Ollama stack."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, List, Optional

import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box
from jinja2 import Template
import yaml

from .configuration import (
    ensure_registry,
    find_history_entry,
    get_previous_model,
    load_env,
    load_model_registry,
    record_model_activation,
    update_env_var,
)
from .ollama import OllamaClient, OllamaClientError
from .utils import ensure_parents, slugify, utcnow

console = Console()
app = typer.Typer(help="Manage CrewAI agents and Ollama models.")
agents_app = typer.Typer(help="Scaffold CrewAI agents.")
models_app = typer.Typer(help="Control Ollama models and versions.")
app.add_typer(agents_app, name="agents")
app.add_typer(models_app, name="models")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
AGENTS_DIR = PROJECT_ROOT / "agents"
MODELS_TEMPLATE_DIR = PROJECT_ROOT / "models" / "templates"
MODELS_BUILD_DIR = PROJECT_ROOT / "models" / "build"


def _client() -> OllamaClient:
    env = load_env()
    base_url = env.get("OLLAMA_BASE_URL", "http://localhost:11434")
    timeout = int(env.get("OLLAMA_TIMEOUT", 60))
    verify = env.get("OLLAMA_VERIFY_TLS", "true").lower() not in {"0", "false", "no"}
    return OllamaClient(base_url, timeout=timeout, verify=verify)


def _print_events(events: Iterable[dict]) -> None:
    for event in events:
        if "error" in event:
            console.print(f"[red]error:[/red] {event['error']}")
            raise typer.Exit(code=1)
        status = event.get("status")
        progress_value = event.get("progress")
        if status:
            if progress_value is not None:
                console.log(f"{status} ({progress_value:.0%})")
            else:
                console.log(status)


def _activate_model(name: str, *, reason: str, allow_missing: bool = False) -> None:
    client = _client()
    info = client.show_model(name)
    if not info and not allow_missing:
        console.print(f"[red]Model '{name}' not found on Ollama server.[/red]")
        raise typer.Exit(code=1)
    update_env_var("OLLAMA_MODEL", name)
    digest = info.get("digest") if info else None
    entry = record_model_activation(name, digest, reason=reason)
    console.print(
        f"[green]Activated model[/green] [bold]{entry['name']}[/bold] "
        f"(digest: {entry.get('digest') or 'unknown'})"
    )


@agents_app.command("create")
def create_agent(
    name: str = typer.Argument(..., help="Human-readable agent name."),
    role: str = typer.Option(..., "--role", prompt=True, help="Role description."),
    goal: str = typer.Option(..., "--goal", prompt=True, help="Primary objective."),
    backstory: str = typer.Option(..., "--backstory", prompt=True, help="Narrative context."),
    tool: List[str] = typer.Option(None, "--tool", "-t", help="Tool identifiers to attach."),
    force: bool = typer.Option(False, "--force", "-f", help="Overwrite if agent already exists."),
) -> None:
    """Create a YAML agent definition under the agents directory."""
    slug = slugify(name)
    agent_path = AGENTS_DIR / f"{slug}.yaml"
    if agent_path.exists() and not force:
        console.print(
            f"[red]Agent file {agent_path.relative_to(PROJECT_ROOT)} already exists. "
            "Use --force to overwrite.[/red]"
        )
        raise typer.Exit(code=1)

    agent_definition = {
        "name": name,
        "role": role,
        "goal": goal,
        "backstory": backstory,
        "tools": tool or [],
        "created_at": utcnow(),
    }

    ensure_parents(agent_path)
    with agent_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(agent_definition, handle, sort_keys=False)

    console.print(
        f"[green]Agent scaffolded at[/green] [bold]{agent_path.relative_to(PROJECT_ROOT)}[/bold]"
    )


@agents_app.command("list")
def list_agents() -> None:
    """List available agent definitions."""
    if not AGENTS_DIR.exists():
        console.print("[yellow]No agents directory found yet.[/yellow]")
        raise typer.Exit()

    agent_files = sorted(AGENTS_DIR.glob("*.yaml"))
    if not agent_files:
        console.print("[yellow]No agent definitions discovered.[/yellow]")
        raise typer.Exit()

    table = Table(title="Agents", box=box.SIMPLE)
    table.add_column("Name")
    table.add_column("File")
    table.add_column("Updated")

    for path in agent_files:
        with path.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
        table.add_row(
            data.get("name", path.stem),
            str(path.relative_to(PROJECT_ROOT)),
            data.get("created_at", "?"),
        )

    console.print(table)


@agents_app.command("show")
def show_agent(
    name: str = typer.Argument(..., help="Slug or filename of the agent (without extension).")
) -> None:
    """Display the contents of an agent definition."""
    path = AGENTS_DIR / f"{slugify(name)}.yaml"
    if not path.exists():
        console.print(f"[red]Agent definition {path.relative_to(PROJECT_ROOT)} not found.[/red]")
        raise typer.Exit(code=1)

    with path.open("r", encoding="utf-8") as handle:
        content = handle.read()

    console.print(Panel.fit(content, title=path.name))


@models_app.command("list")
def list_models() -> None:
    """List models available via Ollama."""
    try:
        client = _client()
        models = client.list_models()
    except OllamaClientError as exc:
        console.print(f"[red]Failed to query Ollama: {exc}[/red]")
        raise typer.Exit(code=1)

    if not models:
        console.print("[yellow]No models available on Ollama server.[/yellow]")
        return

    table = Table(title="Ollama Models", box=box.SIMPLE)
    table.add_column("Name", style="bold")
    table.add_column("Digest", overflow="fold")
    table.add_column("Size (GB)")
    table.add_column("Updated")

    for model in models:
        size = model.get("size", 0)
        size_gb = f"{size / (1024**3):.2f}" if size else "?"
        table.add_row(
            model.get("name", "<unknown>"),
            (model.get("digest") or "")[:16] + "…" if model.get("digest") else "–",
            size_gb,
            model.get("modified_at", "unknown"),
        )

    console.print(table)


@models_app.command("pull")
def pull_model(
    name: str = typer.Argument(..., help="Model identifier to pull via Ollama."),
) -> None:
    """Download a model onto the Ollama host."""
    console.print(f"[info] Pulling model [bold]{name}[/bold]...")
    try:
        _print_events(_client().pull_model(name))
    except OllamaClientError as exc:
        console.print(f"[red]Model pull failed: {exc}[/red]")
        raise typer.Exit(code=1)
    console.print(f"[green]Model '{name}' pull completed.[/green]")


@models_app.command("switch")
def switch_model(
    name: str = typer.Argument(..., help="Model to activate."),
    pull_if_missing: bool = typer.Option(
        False,
        "--pull",
        help="Attempt to pull the model if not present locally.",
    ),
) -> None:
    """Activate a model and update the registry."""
    client = _client()
    info = client.show_model(name)
    if not info and pull_if_missing:
        console.print(f"[yellow]Model '{name}' not present locally. Pulling...[/yellow]")
        try:
            _print_events(client.pull_model(name))
            info = client.show_model(name)
        except OllamaClientError as exc:
            console.print(f"[red]Pull failed: {exc}[/red]")
            raise typer.Exit(code=1)
    if not info:
        console.print(
            f"[red]Model '{name}' not available. Use `crewctl models pull {name}` first.[/red]"
        )
        raise typer.Exit(code=1)

    update_env_var("OLLAMA_MODEL", name)
    entry = record_model_activation(name, info.get("digest"), reason="switch")
    console.print(
        f"[green]Switched active model to[/green] [bold]{entry['name']}[/bold] "
        f"(digest: {entry.get('digest') or 'unknown'})"
    )


@models_app.command("history")
def model_history() -> None:
    """Display activation history of models."""
    ensure_registry()
    registry = load_model_registry()
    history = registry.get("history") or []
    if not history:
        console.print("[yellow]No model activations recorded yet.[/yellow]")
        return

    table = Table(title="Model History", box=box.SIMPLE)
    table.add_column("Activated At")
    table.add_column("Model")
    table.add_column("Digest")
    table.add_column("Reason")

    for entry in history:
        table.add_row(
            entry.get("activated_at", "?"),
            entry.get("name", "?"),
            (entry.get("digest") or "")[:16] + "…" if entry.get("digest") else "–",
            entry.get("reason", ""),
        )

    console.print(table)


@models_app.command("rollback")
def rollback_model() -> None:
    """Revert to the previously active model."""
    previous = get_previous_model()
    if not previous:
        console.print("[yellow]No previous model recorded.[/yellow]")
        raise typer.Exit()
    _activate_model(previous["name"], reason="rollback")


@models_app.command("activate")
def activate_model(
    name: str = typer.Argument(..., help="Model from history to reactivate."),
) -> None:
    """Activate a model that already exists on the server."""
    entry = find_history_entry(name)
    if not entry:
        console.print(f"[red]Model '{name}' not present in history.[/red]")
        raise typer.Exit(code=1)
    _activate_model(entry["name"], reason="history-activate")


@models_app.command("pretrain")
def pretrain_model(
    name: str = typer.Argument(..., help="Name for the new fine-tuned model."),
    base: str = typer.Option(
        ...,
        "--base",
        help="Base model to start from.",
        prompt=True,
    ),
    template: Path = typer.Option(
        MODELS_TEMPLATE_DIR / "lrm_modelfile.j2",
        "--template",
        help="Jinja2 template to use for the modelfile.",
    ),
    system_prompt: Optional[str] = typer.Option(
        None,
        "--system-prompt",
        help="Optional system prompt to bake into the model.",
    ),
    system_prompt_file: Optional[Path] = typer.Option(
        None,
        "--system-prompt-file",
        help="Read system prompt text from file.",
    ),
    temperature: float = typer.Option(0.7, "--temperature", min=0.0, max=2.0),
    top_p: float = typer.Option(0.9, "--top-p", min=0.0, max=1.0),
    adapter: Optional[Path] = typer.Option(
        None,
        "--adapter",
        help="Optional adapter (LoRA) directory to include.",
    ),
    quantize: Optional[str] = typer.Option(
        None,
        "--quantize",
        help="Quantization target (e.g. Q4_0, Q4_KM).",
    ),
    activate_after: bool = typer.Option(
        False,
        "--activate",
        help="Activate the newly created model after build completes.",
    ),
) -> None:
    """Create a custom LRM modelfile and register it with Ollama."""
    if not template.exists():
        console.print(f"[red]Template {template} not found.[/red]")
        raise typer.Exit(code=1)

    prompt_text = system_prompt
    if system_prompt_file:
        prompt_text = system_prompt_file.read_text(encoding="utf-8")

    template_text = template.read_text(encoding="utf-8")
    modelfile = Template(template_text).render(
        base_model=base,
        system_prompt=prompt_text or "",
        temperature=temperature,
        top_p=top_p,
        adapter_path=str(adapter) if adapter else "",
        created_at=utcnow(),
        name=name,
    )

    build_path = MODELS_BUILD_DIR / f"{slugify(name)}.modelfile"
    ensure_parents(build_path)
    build_path.write_text(modelfile, encoding="utf-8")

    console.print(
        f"[info] Rendered modelfile written to {build_path.relative_to(PROJECT_ROOT)}. "
        "Submitting to Ollama…"
    )

    try:
        _print_events(_client().create_model(name, modelfile, quantize=quantize))
    except OllamaClientError as exc:
        console.print(f"[red]Model build failed: {exc}[/red]")
        raise typer.Exit(code=1)

    console.print(f"[green]Model '{name}' created successfully.[/green]")
    if activate_after:
        _activate_model(name, reason="pretrain-activate", allow_missing=False)


@models_app.command("inspect")
def inspect_model(
    name: str = typer.Argument(..., help="Model identifier to inspect."),
) -> None:
    """Show raw metadata for a model."""
    info = _client().show_model(name)
    if not info:
        console.print(f"[red]Model '{name}' not found.[/red]")
        raise typer.Exit(code=1)
    console.print_json(json.dumps(info, indent=2))


@models_app.command("info")
def server_info() -> None:
    """Display version information from the Ollama server."""
    try:
        info = _client().show_info()
    except OllamaClientError as exc:
        console.print(f"[red]Unable to reach Ollama: {exc}[/red]")
        raise typer.Exit(code=1)
    console.print_json(json.dumps(info, indent=2))
