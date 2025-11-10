"""Configuration management for crewctl."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

from .utils import CrewCtlError, ensure_directory, resolve_project_root

DEFAULT_CONFIG: Dict[str, Any] = {
    "version": 1,
    "agents_dir": "agents",
    "models_dir": "models",
    "training": {
        "output_dir": "models/lrms",
        "default_epochs": 3,
        "default_batch_size": 2,
        "default_learning_rate": 2e-4,
    },
    "ollama": {
        "binary": ".crewctl/bin/ollama",
        "host": "http://127.0.0.1:11434",
        "default_model": "llama3",
    },
}


@dataclass
class CrewCtlPaths:
    """Convenience wrapper for shared project paths."""

    root: Path
    state_dir: Path
    config_file: Path
    env_file: Path
    templates_dir: Path
    agents_dir: Path
    models_dir: Path
    training_dir: Path
    model_registry: Path
    ollama_binary: Path

    @classmethod
    def create(cls, root: Optional[Path] = None) -> "CrewCtlPaths":
        root = root or resolve_project_root()
        state_dir = root / ".crewctl"
        config_file = state_dir / "config.yaml"
        env_file = root / ".env"
        templates_dir = Path(__file__).resolve().parent / "templates"
        agents_dir = root / DEFAULT_CONFIG["agents_dir"]
        models_dir = root / DEFAULT_CONFIG["models_dir"]
        training_dir = root / DEFAULT_CONFIG["training"]["output_dir"]
        model_registry = root / "config" / "model_registry.json"
        ollama_binary = root / DEFAULT_CONFIG["ollama"]["binary"]
        return cls(
            root=root,
            state_dir=state_dir,
            config_file=config_file,
            env_file=env_file,
            templates_dir=templates_dir,
            agents_dir=agents_dir,
            models_dir=models_dir,
            training_dir=training_dir,
            model_registry=model_registry,
            ollama_binary=ollama_binary,
        )


class CrewCtlConfig:
    """Read/write access to crewctl configuration values."""

    def __init__(self, paths: Optional[CrewCtlPaths] = None) -> None:
        self.paths = paths or CrewCtlPaths.create()
        ensure_directory(self.paths.state_dir)
        self.data = self._load_or_init()

    def _load_or_init(self) -> Dict[str, Any]:
        if self.paths.config_file.exists():
            content = yaml.safe_load(self.paths.config_file.read_text()) or {}
            # Ensure keys exist even if older versions omitted them
            merged = DEFAULT_CONFIG | content
            merged["training"] = DEFAULT_CONFIG["training"] | merged.get("training", {})
            merged["ollama"] = DEFAULT_CONFIG["ollama"] | merged.get("ollama", {})
            return merged

        ensure_directory(self.paths.config_file.parent)
        self.paths.config_file.write_text(yaml.safe_dump(DEFAULT_CONFIG, sort_keys=False))
        return DEFAULT_CONFIG.copy()

    def save(self) -> None:
        ensure_directory(self.paths.config_file.parent)
        self.paths.config_file.write_text(yaml.safe_dump(self.data, sort_keys=False))

    # Convenience accessors -------------------------------------------------
    @property
    def agents_dir(self) -> Path:
        return self.paths.root / self.data.get("agents_dir", DEFAULT_CONFIG["agents_dir"])

    @property
    def models_dir(self) -> Path:
        return self.paths.root / self.data.get("models_dir", DEFAULT_CONFIG["models_dir"])

    @property
    def training_output_dir(self) -> Path:
        return self.paths.root / self.data.get("training", {}).get(
            "output_dir", DEFAULT_CONFIG["training"]["output_dir"]
        )

    @property
    def ollama_binary(self) -> Path:
        override = self.data.get("ollama", {}).get("binary")
        return (self.paths.root / override).resolve()

    @property
    def default_model(self) -> str:
        return self.data.get("ollama", {}).get("default_model", DEFAULT_CONFIG["ollama"]["default_model"])

    def set_default_model(self, model: str) -> None:
        self.data.setdefault("ollama", {})["default_model"] = model
        self.save()

    def update_training_defaults(
        self,
        *,
        output_dir: Optional[str] = None,
        epochs: Optional[int] = None,
        batch_size: Optional[int] = None,
        learning_rate: Optional[float] = None,
    ) -> None:
        training = self.data.setdefault("training", {})
        if output_dir:
            training["output_dir"] = output_dir
        if epochs is not None:
            training["default_epochs"] = epochs
        if batch_size is not None:
            training["default_batch_size"] = batch_size
        if learning_rate is not None:
            training["default_learning_rate"] = learning_rate
        self.save()

    def ensure_ollama_binary(self) -> Path:
        binary = self.ollama_binary
        if not binary.exists():
            raise CrewCtlError(
                f"Ollama binary not found at {binary}. Run `python scripts/deploy.py` to install the stack."
            )
        return binary
