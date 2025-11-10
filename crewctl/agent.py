"""Agent management helpers."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from .config import CrewCtlConfig
from .utils import CrewCtlError, camel_case, console, ensure_directory, snake_case, title_case


class AgentManager:
    """Create and enumerate CrewAI agents."""

    TEMPLATE_NAME = "agent.py.j2"

    def __init__(self, config: Optional[CrewCtlConfig] = None) -> None:
        self.config = config or CrewCtlConfig()

    @property
    def template_path(self) -> Path:
        path = self.config.paths.templates_dir / self.TEMPLATE_NAME
        if not path.exists():
            raise CrewCtlError(f"Agent template not found: {path}")
        return path

    def list_agents(self) -> List[Path]:
        agents_dir = self.config.agents_dir
        if not agents_dir.exists():
            return []
        return sorted(p for p in agents_dir.glob("*.py") if p.is_file())

    def create_agent(
        self,
        *,
        name: str,
        role: Optional[str] = None,
        goal: Optional[str] = None,
        backstory: Optional[str] = None,
        overwrite: bool = False,
    ) -> Path:
        agents_dir = self.config.agents_dir
        ensure_directory(agents_dir)

        slug = snake_case(name)
        class_name = camel_case(name) + "Agent"
        file_path = agents_dir / f"{slug}.py"

        if file_path.exists() and not overwrite:
            raise CrewCtlError(f"Agent file already exists: {file_path}. Use --overwrite to replace.")

        role = role or f"{title_case(name)} Specialist"
        goal = goal or f"Deliver outstanding results as {title_case(name)}."
        backstory = backstory or f"{title_case(name)} is an expert CrewAI agent."

        template_content = self.template_path.read_text()
        file_content = template_content.format(
            class_name=class_name,
            agent_name=title_case(name),
            role=role,
            goal=goal,
            backstory=backstory.replace('"', '\\"'),
        )

        file_path.write_text(file_content)
        console.print(f"[bold green]Created agent definition at {file_path}")
        return file_path
