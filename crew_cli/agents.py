from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer

from . import get_version
from .config import agent_registry_path, agent_template_path, load_agent_registry, save_agent_registry
from .utils import DeploymentError, ensure_directory, project_root, render_template, sanitize_permissions, slugify, timestamp

agents_app = typer.Typer(help="Manage CrewAI agents.")


def _agents_directory() -> Path:
    path = project_root() / "agents"
    ensure_directory(path)
    return path


def _register_agent(
    slug: str,
    name: str,
    role: str,
    goal: str,
    backstory: str,
    verbose: bool,
    model: Optional[str],
    file_path: Path,
) -> None:
    registry = load_agent_registry()
    agents = registry.setdefault("agents", [])
    existing = next((item for item in agents if item["slug"] == slug), None)
    record = {
        "slug": slug,
        "name": name,
        "role": role,
        "goal": goal,
        "backstory": backstory,
        "verbose": verbose,
        "model": model,
        "file": str(file_path.relative_to(project_root())),
        "updated_at": timestamp(),
        "generated_by": f"crew-cli/{get_version()}",
    }
    if existing:
        agents[agents.index(existing)] = record
    else:
        record["created_at"] = record["updated_at"]
        agents.append(record)
    save_agent_registry(registry)
    sanitize_permissions(agent_registry_path())


@agents_app.command("list")
def list_agents() -> None:
    """List registered agents."""
    registry = load_agent_registry()
    agents = registry.get("agents", [])
    if not agents:
        typer.echo("No agents registered yet. Use `crewctl agents create` to add one.")
        return
    for item in agents:
        typer.echo(
            f"- {item['name']} ({item['slug']}) -> file={item['file']} model={item.get('model') or 'default'}"
        )


@agents_app.command("create")
def create_agent(
    name: str = typer.Argument(..., help="Display name for the agent."),
    role: str = typer.Option(..., "--role", help="Primary role for the agent."),
    goal: str = typer.Option(..., "--goal", help="Core objective for the agent."),
    backstory: str = typer.Option("", "--backstory", help="Background story and context."),
    model: Optional[str] = typer.Option(None, "--model", help="Preferred model for this agent."),
    verbose: bool = typer.Option(True, "--verbose/--quiet", help="Enable verbose logging."),
    overwrite: bool = typer.Option(False, "--overwrite", help="Overwrite existing agent file."),
) -> None:
    """Create a new agent module from the configured template."""
    slug = slugify(name)
    if not slug:
        raise DeploymentError("Name must contain at least one alphanumeric character.")

    template_path = agent_template_path()
    agents_dir = _agents_directory()
    file_path = agents_dir / f"{slug}.py"

    if file_path.exists() and not overwrite:
        raise DeploymentError(
            f"Agent file {file_path} already exists. Use --overwrite to replace it."
        )

    content = render_template(
        template_path,
        name=name,
        role=role,
        goal=goal,
        backstory=backstory,
        verbose=verbose,
    )

    file_path.write_text(content, encoding="utf-8")
    sanitize_permissions(file_path)
    _register_agent(slug, name, role, goal, backstory, verbose, model, file_path)
    typer.echo(f"Agent '{name}' created at {file_path.relative_to(project_root())}.")


@agents_app.command("show")
def show_agent(
    name: str = typer.Argument(..., help="Slug or name of the agent to show."),
) -> None:
    """Display metadata for a registered agent."""
    registry = load_agent_registry()
    agents = registry.get("agents", [])
    match = next(
        (agent for agent in agents if agent["slug"] == slugify(name) or agent["name"] == name),
        None,
    )
    if not match:
        raise DeploymentError(f"No agent found with name or slug '{name}'.")

    typer.echo(f"Name       : {match['name']}")
    typer.echo(f"Slug       : {match['slug']}")
    typer.echo(f"Role       : {match['role']}")
    typer.echo(f"Goal       : {match['goal']}")
    typer.echo(f"Verbose    : {match['verbose']}")
    typer.echo(f"Model      : {match.get('model') or 'default'}")
    typer.echo(f"File       : {match['file']}")
    typer.echo(f"Created At : {match.get('created_at', 'n/a')}")
    typer.echo(f"Updated At : {match.get('updated_at', 'n/a')}")
