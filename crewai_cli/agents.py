from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import typer
from jinja2 import Environment, FileSystemLoader, select_autoescape
from rich.console import Console
from rich.table import Table

from . import config, utils

console = Console()

templates_dir = Path(__file__).resolve().parent / "templates"
jinja_env = Environment(
    loader=FileSystemLoader(str(templates_dir)),
    autoescape=select_autoescape(default=True, disabled_extensions=("j2",)),
    trim_blocks=True,
    lstrip_blocks=True,
)
agent_template = jinja_env.get_template("agent.py.j2")

agents_root = config.PROJECT_ROOT / "agents"

app = typer.Typer(help="Create and inspect CrewAI agents.")


@app.command("list")
def list_agents() -> None:
    """List available agent modules."""
    if not agents_root.exists():
        console.print(":warning: Agents directory does not exist yet.")
        raise typer.Exit(0)

    agents = sorted(agents_root.glob("*.py"))
    if not agents:
        console.print(":information_source: No agent modules found under `agents/`.")
        raise typer.Exit(0)

    table = Table(title="Registered Agents")
    table.add_column("Name", style="cyan")
    table.add_column("File", style="green")
    for path in agents:
        table.add_row(path.stem, str(path.relative_to(config.PROJECT_ROOT)))
    console.print(table)


@app.command("create")
def create_agent(
    name: str = typer.Argument(..., help="Human-readable agent name"),
    role: Optional[str] = typer.Option(None, "--role", "-r", help="Role/Persona for the agent", prompt=True),
    goal: Optional[str] = typer.Option(None, "--goal", "-g", help="Primary goal for the agent", prompt=True),
    backstory: Optional[str] = typer.Option(None, "--backstory", help="Short backstory/description"),
    backstory_file: Optional[Path] = typer.Option(
        None,
        "--backstory-file",
        help="Path to a file containing the agent backstory.",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
    ),
    tool: Optional[List[str]] = typer.Option(
        None,
        "--tool",
        "-t",
        help="Tool identifier to register with the agent (repeat for multiple).",
    ),
    verbose: bool = typer.Option(False, "--verbose/--no-verbose", help="Set agent verbosity"),
    allow_delegation: bool = typer.Option(
        False,
        "--delegate/--no-delegate",
        help="Whether the agent can delegate tasks to sub-agents.",
    ),
    force: bool = typer.Option(False, "--force", help="Overwrite existing agent module if present."),
) -> None:
    """Generate a new agent module from a template."""
    if not name.strip():
        raise typer.BadParameter("Agent name must not be empty.")

    class_name = utils.pascal_case(name) + "Agent"
    filename = utils.snake_case(name) or "agent"
    target_path = agents_root / f"{filename}.py"

    if target_path.exists() and not force:
        console.print(f":warning: Agent file already exists at `{target_path}`.")
        overwrite = typer.confirm("Overwrite existing file?", default=False)
        if not overwrite:
            raise typer.Exit(1)

    if backstory_file:
        backstory = backstory_file.read_text(encoding="utf-8").strip()
    elif not backstory:
        backstory = typer.prompt("Provide a brief backstory", default="Describe the agent's expertise and history.")

    rendered = agent_template.render(
        class_name=class_name,
        agent_name=name,
        role=role or "",
        goal=goal or "",
        backstory=backstory or "",
        tools=tool or [],
        verbose=verbose,
        allow_delegation=allow_delegation,
    )

    utils.ensure_parent(target_path)
    target_path.write_text(rendered.rstrip() + "\n", encoding="utf-8")
    console.print(f":white_check_mark: Created agent module at `{target_path}`.")
