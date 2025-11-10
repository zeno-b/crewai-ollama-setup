"""Model lifecycle management for crewctl."""

from __future__ import annotations

import json
import os
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Union

from rich.table import Table

from .config import CrewCtlConfig
from .utils import (
    CrewCtlError,
    console,
    ensure_directory,
    load_json,
    run_command,
    save_json,
    update_env_file,
)


@dataclass
class RegistryEntry:
    id: str
    model: str
    digest: Optional[str]
    size: Optional[int]
    created_at: datetime
    metadata: Dict[str, str]

    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "model": self.model,
            "digest": self.digest,
            "size": self.size,
            "created_at": self.created_at.isoformat(),
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "RegistryEntry":
        return cls(
            id=data["id"],
            model=data["model"],
            digest=data.get("digest"),
            size=data.get("size"),
            created_at=datetime.fromisoformat(data["created_at"]),
            metadata=data.get("metadata", {}),
        )


class ModelRegistry:
    """Persistent registry of model versions."""

    def __init__(self, path: Path) -> None:
        self.path = path
        ensure_directory(path.parent)
        self.data = load_json(path, default={"active_id": None, "history": []})
        self._index_history()

    def _index_history(self) -> None:
        history = [RegistryEntry.from_dict(entry) for entry in self.data.get("history", [])]
        self.history: List[RegistryEntry] = history
        self.id_map = {entry.id: entry for entry in history}

    def save(self) -> None:
        payload = {
            "active_id": self.data.get("active_id"),
            "history": [entry.to_dict() for entry in self.history],
        }
        save_json(self.path, payload)

    def record(self, entry: RegistryEntry, *, make_active: bool = True) -> RegistryEntry:
        self.history = [entry] + [e for e in self.history if e.id != entry.id]
        self.history = self.history[:50]
        self.id_map = {item.id: item for item in self.history}
        if make_active:
            self.data["active_id"] = entry.id
        self.save()
        return entry

    def set_active(self, identifier: Union[str, int]) -> RegistryEntry:
        entry = self.get(identifier)
        if not entry:
            raise CrewCtlError(f"Unable to locate model with identifier {identifier}")
        self.data["active_id"] = entry.id
        self.save()
        return entry

    def get(self, identifier: Union[str, int, None]) -> Optional[RegistryEntry]:
        if identifier is None:
            return None
        if isinstance(identifier, int):
            if 1 <= identifier <= len(self.history):
                return self.history[identifier - 1]
            return None
        identifier = str(identifier)
        if identifier in self.id_map:
            return self.id_map[identifier]
        for entry in self.history:
            if entry.id.startswith(identifier) or entry.model == identifier:
                return entry
        return None

    @property
    def active(self) -> Optional[RegistryEntry]:
        return self.get(self.data.get("active_id"))


class ModelManager:
    """High level model operations using the Ollama binary."""

    def __init__(self, config: Optional[CrewCtlConfig] = None) -> None:
        self.config = config or CrewCtlConfig()
        ensure_directory(self.config.paths.model_registry.parent)
        self.registry = ModelRegistry(self.config.paths.model_registry)

    # Ollama integration ---------------------------------------------------
    def _ollama(self, *args: str) -> Dict:
        binary = self.config.ensure_ollama_binary()
        env = os.environ.copy()
        env.setdefault("OLLAMA_HOST", self.config.data.get("ollama", {}).get("host", "http://127.0.0.1:11434"))
        result = run_command([str(binary), *args], env=env, check=True)
        output = result.stdout.strip()
        if not output:
            return {}
        try:
            # Some Ollama commands stream JSON lines
            lines = [json.loads(line) for line in output.splitlines() if line.strip()]
            if len(lines) == 1:
                return lines[0]
            return {"items": lines}
        except json.JSONDecodeError:
            return {"raw": output}

    def list_models(self) -> List[Dict]:
        payload = self._ollama("list", "--json")
        items = payload.get("models") or payload.get("items") or []
        return items

    def pull_model(self, model: str) -> None:
        console.print(f"[bold blue]Pulling model {model}...")
        payload = self._ollama("pull", model, "--json")
        if payload.get("status") == "success":
            console.print(f"[bold green]Model {model} pulled successfully.")
        else:
            console.print(f"[yellow]Pull completed: {payload}")

    def show_model(self, model: str) -> Dict:
        payload = self._ollama("show", model, "--json")
        return payload

    def _create_entry(self, model: str, details: Optional[Dict]) -> RegistryEntry:
        digest = None
        size = None
        metadata: Dict[str, str] = {}
        if details:
            digest = details.get("digest") or details.get("hash")
            size = details.get("size")
            metadata = {k: str(v) for k, v in details.items() if isinstance(v, (str, int, float))}
        return RegistryEntry(
            id=str(uuid.uuid4())[:8],
            model=model,
            digest=digest,
            size=size,
            created_at=datetime.utcnow(),
            metadata=metadata,
        )

    def use_model(self, model: str) -> RegistryEntry:
        details = self.show_model(model)
        entry = self._create_entry(model, details if isinstance(details, dict) else None)
        self.registry.record(entry, make_active=True)
        update_env_file(self.config.paths.env_file, {"OLLAMA_MODEL": model})
        self.config.set_default_model(model)
        console.print(f"[bold green]Activated model {model}")
        return entry

    def history_table(self) -> Table:
        table = Table(title="Model History")
        table.add_column("#", justify="right")
        table.add_column("ID")
        table.add_column("Model")
        table.add_column("Digest")
        table.add_column("Size")
        table.add_column("Created")

        for idx, entry in enumerate(self.registry.history, start=1):
            is_active = self.registry.active.id == entry.id if self.registry.active else False
            model_label = f"[bold]{entry.model}[/]" if is_active else entry.model
            table.add_row(
                f"{idx}",
                entry.id,
                model_label,
                entry.digest[:12] + "…" if entry.digest else "—",
                f"{entry.size:,}" if entry.size else "—",
                entry.created_at.strftime("%Y-%m-%d %H:%M"),
            )
        return table

    def activate(self, identifier: Union[str, int]) -> RegistryEntry:
        entry = self.registry.set_active(identifier)
        update_env_file(self.config.paths.env_file, {"OLLAMA_MODEL": entry.model})
        self.config.set_default_model(entry.model)
        console.print(f"[bold green]Activated {entry.model} (id={entry.id})")
        return entry
