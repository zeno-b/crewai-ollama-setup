from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import httpx
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
modelfile_template = jinja_env.get_template("modelfile.j2")

DEFAULT_TIMEOUT = httpx.Timeout(60.0, connect=15.0)

app = typer.Typer(help="Manage models, versions, and training workflows.")


@app.command("list")
def list_models(
    base_url: Optional[str] = typer.Option(None, "--base-url", "-b", help="Override the Ollama base URL."),
) -> None:
    """List available Ollama models from the running instance."""
    resolved = config.resolve_ollama_base_url(base_url)
    try:
        response = httpx.get(f"{resolved}/api/tags", timeout=DEFAULT_TIMEOUT)
        response.raise_for_status()
    except httpx.HTTPError as exc:
        console.print(f":x: Failed to list models: {exc}")
        raise typer.Exit(1)

    payload = response.json()
    models = payload.get("models", [])
    if not models:
        console.print(":information_source: No models reported by Ollama.")
        return

    table = Table(title=f"Ollama models @ {resolved}", show_lines=False)
    table.add_column("Name", style="cyan")
    table.add_column("Size", style="magenta")
    table.add_column("Digest", style="green")
    table.add_column("Updated", style="yellow")

    for model in models:
        table.add_row(
            model.get("name", "unknown"),
            _format_bytes(model.get("size", 0)),
            (model.get("digest") or "")[:18] + "...",
            model.get("modified_at", ""),
        )
    console.print(table)


@app.command("pull")
def pull_model(
    name: str = typer.Argument(..., help="Model name (e.g. llama2:latest) to pull from registry."),
    base_url: Optional[str] = typer.Option(None, "--base-url", "-b", help="Override the Ollama base URL."),
    set_default: bool = typer.Option(False, "--set-default", help="Promote pulled model as default."),
) -> None:
    """Download a model into the Ollama instance."""
    resolved = config.resolve_ollama_base_url(base_url)
    console.print(f":inbox_tray: Pulling `{name}` from {resolved}")

    digest: Optional[str] = None
    try:
        with httpx.stream(
            "POST",
            f"{resolved}/api/pull",
            json={"name": name},
            timeout=httpx.Timeout(600.0, connect=30.0),
        ) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if not line:
                    continue
                data = json.loads(line)
                if "error" in data:
                    console.print(f":x: {data['error']}")
                    raise typer.Exit(1)
                status = data.get("status")
                if status:
                    console.print(f"  - {status}")
                if data.get("digest"):
                    digest = data["digest"]
    except httpx.HTTPError as exc:
        console.print(f":x: Failed to pull model: {exc}")
        raise typer.Exit(1)

    config.record_model_version(name, digest, source="pull", set_default=set_default)
    console.print(":white_check_mark: Model pull complete.")
    if set_default:
        console.print(f":sparkles: Default model updated to `{name}`.")


@app.command("train")
def train_model(
    name: str = typer.Argument(..., help="Logical name for the new model (no version suffix)."),
    version: Optional[str] = typer.Option(
        None,
        "--version",
        "-v",
        help="Version tag suffix (default: timestamp).",
    ),
    base_model: str = typer.Option(
        "llama2:latest",
        "--base-model",
        help="Base model to fine-tune / specialise.",
    ),
    system_prompt: str = typer.Option(
        "You are a specialised assistant fine-tuned for organisation-specific knowledge.",
        "--system-prompt",
        help="System prompt to bake into the model.",
    ),
    dataset: Optional[Path] = typer.Option(
        None,
        "--dataset",
        "-d",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        help="Dataset text file to embed into the system prompt context.",
    ),
    modelfile: Optional[Path] = typer.Option(
        None,
        "--modelfile",
        "-m",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        help="Path to a pre-defined Modelfile to submit instead of generating one.",
    ),
    temperature: float = typer.Option(0.7, "--temperature", min=0.0, max=1.5, help="Generation temperature."),
    num_ctx: int = typer.Option(4096, "--num-ctx", min=512, help="Context window size."),
    base_url: Optional[str] = typer.Option(None, "--base-url", "-b", help="Override the Ollama base URL."),
    set_default: bool = typer.Option(
        True,
        "--set-default/--no-set-default",
        help="Promote the newly trained model as default (default: true).",
    ),
) -> None:
    """Create a tuned LRM model via Ollama's /api/create endpoint."""
    resolved = config.resolve_ollama_base_url(base_url)
    tag_version = version or utils.timestamp_slug()
    full_tag = f"{name}:{tag_version}"

    if modelfile:
        modelfile_content = modelfile.read_text(encoding="utf-8")
    else:
        dataset_preview = ""
        if dataset:
            dataset_preview = _summarise_dataset(dataset)
        rendered = modelfile_template.render(
            base_model=base_model,
            temperature=temperature,
            num_ctx=num_ctx,
            system_prompt=system_prompt,
            dataset=dataset_preview,
            generated_at=utils.timestamp_slug(),
        )
        lrms_dir = config.PROJECT_ROOT / "models" / "lrms" / utils.snake_case(name)
        modelfile_path = lrms_dir / f"Modelfile.{tag_version}"
        utils.ensure_parent(modelfile_path)
        modelfile_path.write_text(rendered.rstrip() + "\n", encoding="utf-8")
        console.print(f":memo: Generated Modelfile scaffold at `{modelfile_path}`.")
        modelfile_content = rendered

    console.print(f":hammer_and_wrench: Creating model `{full_tag}` from base `{base_model}`")
    digest: Optional[str] = None
    metadata = {"base_model": base_model, "dataset": str(dataset) if dataset else None}
    try:
        with httpx.stream(
            "POST",
            f"{resolved}/api/create",
            json={"name": full_tag, "modelfile": modelfile_content},
            timeout=httpx.Timeout(1200.0, connect=30.0),
        ) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if not line:
                    continue
                data = json.loads(line)
                if "error" in data:
                    console.print(f":x: {data['error']}")
                    raise typer.Exit(1)
                status = data.get("status")
                if status:
                    console.print(f"  - {status}")
                if data.get("digest"):
                    digest = data["digest"]
    except httpx.HTTPError as exc:
        console.print(f":x: Model creation failed: {exc}")
        raise typer.Exit(1)

    config.record_model_version(full_tag, digest, source="train", metadata=metadata, set_default=set_default)
    console.print(f":white_check_mark: Model `{full_tag}` registered successfully.")
    if set_default:
        console.print(f":sparkles: Default model updated to `{full_tag}`.")


@app.command("switch")
def switch_default(
    tag: str = typer.Argument(..., help="Model tag to promote as default (e.g. llama2:7b)."),
    verify: bool = typer.Option(
        True,
        "--verify/--no-verify",
        help="Check with Ollama that the model tag exists before switching.",
    ),
    base_url: Optional[str] = typer.Option(None, "--base-url", "-b", help="Override the Ollama base URL."),
) -> None:
    """Set a new default model for the application."""
    if verify:
        resolved = config.resolve_ollama_base_url(base_url)
        try:
            response = httpx.get(f"{resolved}/api/tags", timeout=DEFAULT_TIMEOUT)
            response.raise_for_status()
            models = {model.get("name") for model in response.json().get("models", [])}
            if tag not in models:
                console.print(f":warning: Model `{tag}` not found on Ollama instance {resolved}.")
                proceed = typer.confirm("Proceed anyway?", default=False)
                if not proceed:
                    raise typer.Exit(1)
        except httpx.HTTPError as exc:
            console.print(f":warning: Unable to verify model availability: {exc}")
            proceed = typer.confirm("Proceed with switch despite verification failure?", default=False)
            if not proceed:
                raise typer.Exit(1)

    config.set_default_model(tag)
    console.print(f":white_check_mark: Default model is now `{tag}`.")


@app.command("versions")
def show_versions() -> None:
    """Display tracked model versions and metadata."""
    registry = config.load_model_registry()
    history = registry.get("history", [])
    models = registry.get("models", {})
    default = registry.get("default_model")

    if not models:
        console.print(":information_source: No model metadata recorded yet.")
        return

    table = Table(title="Model Registry", show_lines=False)
    table.add_column("Model", style="cyan")
    table.add_column("Tag", style="magenta")
    table.add_column("Source", style="green")
    table.add_column("Created", style="yellow")
    table.add_column("Digest", style="white")

    for model_name, info in models.items():
        for version in info.get("versions", []):
            tag = version.get("tag")
            marker = " (default)" if tag == default else ""
            table.add_row(
                model_name,
                f"{tag}{marker}",
                version.get("source", ""),
                version.get("created_at", ""),
                (version.get("digest") or "")[:18] + "...",
            )
    console.print(table)

    console.print(f"\nHistory: {' → '.join(history) if history else '—'}")


@app.command("rollback")
def rollback_model(
    steps: int = typer.Option(1, "--steps", "-s", min=1, help="How many versions to step back."),
) -> None:
    """Revert the default model to a previous entry in the history."""
    registry = config.load_model_registry()
    history = registry.get("history", [])
    if len(history) <= 1:
        console.print(":warning: Not enough history to roll back.")
        return

    if steps >= len(history):
        console.print(":warning: Steps exceed history length; rolling back to earliest entry.")
        steps = len(history) - 1

    for _ in range(steps):
        history.pop()

    target = history[-1]
    registry["history"] = history
    registry["default_model"] = target
    config.save_model_registry(registry)
    config.update_env_var("OLLAMA_MODEL", target)
    console.print(f":rewind: Rolled back default model to `{target}`.")


def _format_bytes(size: Optional[int]) -> str:
    if not size:
        return "—"
    units = ["B", "KB", "MB", "GB", "TB"]
    value = float(size)
    unit = 0
    while value >= 1024 and unit < len(units) - 1:
        value /= 1024
        unit += 1
    return f"{value:.2f} {units[unit]}"


def _summarise_dataset(path: Path, max_chars: int = 1200) -> str:
    text = path.read_text(encoding="utf-8")
    if len(text) <= max_chars:
        return text.strip()
    return text[: max_chars - 3].strip() + "..."
