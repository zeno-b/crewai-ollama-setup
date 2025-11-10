from __future__ import annotations

import json
import os
import shutil
import tempfile
import zipfile
from pathlib import Path
from typing import Any, Dict, Optional

import httpx
import typer

from . import get_version
from .config import (
    get_config,
    get_state,
    load_model_registry,
    lrm_directory,
    model_registry_path,
    modelfile_template_path,
    save_model_registry,
)
from .utils import (
    DeploymentError,
    compute_sha256,
    ensure_directory,
    project_root,
    render_template,
    run_command,
    sanitize_permissions,
    slugify,
    timestamp,
)

models_app = typer.Typer(help="Manage Ollama models and LRM archives.")


def _resolve_ollama_binary() -> str:
    env_override = os.environ.get("OLLAMA_BIN")
    if env_override:
        return env_override

    state = get_state()
    cand = state.get("ollama", {}).get("binary_path")
    if cand and Path(cand).exists():
        return cand

    found = shutil.which("ollama")
    if found:
        return found

    raise DeploymentError(
        "Unable to locate Ollama binary. Set OLLAMA_BIN or run the deployment script."
    )


def _ollama_base_url() -> str:
    cfg = get_config().get("ollama", {})
    host = os.environ.get("OLLAMA_HOST", cfg.get("listen_host", "127.0.0.1"))
    port = int(os.environ.get("OLLAMA_PORT", cfg.get("listen_port", 11434)))
    base_url = os.environ.get("OLLAMA_BASE_URL")
    if base_url:
        return base_url
    return f"http://{host}:{port}"


def _store_lrm_archive(
    model_name: str,
    version_id: str,
    modelfile_path: Path,
    metadata: Dict[str, Any],
) -> Path:
    directory = lrm_directory()
    archive_name = f"{slugify(model_name)}-{version_id}.lrm"
    archive_path = directory / archive_name
    ensure_directory(directory)
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.write(modelfile_path, arcname="Modelfile")
        archive.writestr("metadata.json", json.dumps(metadata, indent=2))
    sanitize_permissions(archive_path)
    return archive_path


def _append_model_version(
    model_name: str,
    version_id: str,
    lrm_path: Path,
    base_model: str,
    dataset: Optional[str],
    notes: Optional[str],
) -> None:
    registry = load_model_registry()
    models = registry.setdefault("models", {})
    entry = models.setdefault(model_name, {"versions": []})
    entry["base_model"] = base_model
    entry["latest_version"] = version_id
    entry["versions"].append(
        {
            "version": version_id,
            "lrm": str(lrm_path.relative_to(project_root())),
            "dataset": dataset,
            "notes": notes,
            "created_at": timestamp(),
        }
    )
    registry["active_model"] = model_name
    history_entry = {
        "timestamp": timestamp(),
        "model": model_name,
        "action": "pretrain",
        "version": version_id,
        "lrm": str(lrm_path.relative_to(project_root())),
        "base_model": base_model,
        "dataset": dataset,
        "notes": notes,
    }
    registry.setdefault("history", []).append(history_entry)
    save_model_registry(registry)
    sanitize_permissions(model_registry_path())


@models_app.command("list")
def list_models() -> None:
    """List models available in the Ollama instance."""
    url = f"{_ollama_base_url().rstrip('/')}/api/tags"
    try:
        response = httpx.get(url, timeout=10.0)
        response.raise_for_status()
        payload = response.json()
        models = payload.get("models", [])
        if not models:
            typer.echo("No models reported by Ollama.")
            return
        for model in models:
            typer.echo(f"- {model.get('name')} ({model.get('modified')})")
    except Exception as exc:  # noqa: BLE001
        typer.echo(f"Failed to query Ollama API: {exc}")
        typer.echo("Trying local `ollama list` command...")
        binary = _resolve_ollama_binary()
        result = run_command([binary, "list"], check=False, capture_output=True)
        typer.echo(result.stdout)


@models_app.command("switch")
def switch_model(
    name: str = typer.Argument(..., help="Target model name to switch to."),
) -> None:
    """Mark a model as active and verify its presence."""
    binary = _resolve_ollama_binary()
    result = run_command([binary, "show", name], capture_output=True, check=False)
    if result.returncode != 0:
        raise DeploymentError(
            f"Model '{name}' is not available locally. Pull or create it before switching."
        )

    registry = load_model_registry()
    registry["active_model"] = name
    registry.setdefault("history", []).append(
        {"timestamp": timestamp(), "model": name, "action": "switch"}
    )
    save_model_registry(registry)
    typer.echo(f"Active model set to '{name}'.")


@models_app.command("pull")
def pull_model(
    name: str = typer.Argument(..., help="Model name to pull from Ollama registry."),
) -> None:
    """Download a model from the upstream repository."""
    binary = _resolve_ollama_binary()
    typer.echo(f"Pulling model '{name}'...")
    run_command([binary, "pull", name])
    typer.echo("Model pull completed.")


@models_app.command("pretrain")
def pretrain_model(
    name: str = typer.Argument(..., help="Name for the resulting model."),
    base_model: str = typer.Option(..., "--base-model", help="Base model to fine-tune from."),
    dataset: Optional[Path] = typer.Option(
        None,
        "--dataset",
        help="Optional dataset file used during training (referenced in metadata).",
    ),
    system_prompt: str = typer.Option(
        "You are a helpful multimodal reasoning agent.",
        "--system-prompt",
        help="System prompt embedded into the Modelfile.",
    ),
    temperature: float = typer.Option(
        0.7, "--temperature", help="Temperature parameter for the generated model."
    ),
    modelfile: Optional[Path] = typer.Option(
        None,
        "--modelfile",
        help="Existing Modelfile to consume instead of generating from template.",
    ),
    notes: Optional[str] = typer.Option(None, "--notes", help="Notes stored alongside the LRM."),
    adapter: Optional[str] = typer.Option(
        None,
        "--adapter",
        help="Optional adapter path for low-rank matrices (LRM) integration.",
    ),
) -> None:
    """Pre-train a model and persist it as a local LRM archive."""
    binary = _resolve_ollama_binary()
    slug = slugify(name)
    version_id = timestamp()

    if modelfile:
        modelfile_path = modelfile.resolve()
        if not modelfile_path.exists():
            raise DeploymentError(f"Provided Modelfile does not exist: {modelfile_path}")
        modelfile_content = modelfile_path.read_text(encoding="utf-8")
    else:
        template_path = modelfile_template_path()
        modelfile_content = render_template(
            template_path,
            base_model=base_model,
            system_prompt=system_prompt,
            temperature=temperature,
            adapter=adapter,
            dataset=str(dataset) if dataset else None,
            custom_parameters=None,
        )
        training_dir = project_root() / "models" / "training"
        ensure_directory(training_dir)
        modelfile_path = training_dir / f"{slug}-{version_id}.Modelfile"
        modelfile_path.write_text(modelfile_content, encoding="utf-8")
        typer.echo(f"Generated Modelfile at {modelfile_path.relative_to(project_root())}.")

    with tempfile.NamedTemporaryFile("w", delete=False, encoding="utf-8", suffix=".Modelfile") as tmp:
        tmp.write(modelfile_content)
        tmp_path = Path(tmp.name)

    try:
        run_command([binary, "create", name, "-f", str(tmp_path)])
    finally:
        tmp_path.unlink(missing_ok=True)

    metadata = {
        "model": name,
        "base_model": base_model,
        "version": version_id,
        "dataset": str(dataset) if dataset else None,
        "notes": notes,
        "generated_at": timestamp(),
        "generated_by": f"crew-cli/{get_version()}",
        "temperature": temperature,
        "adapter": adapter,
        "checksum": compute_sha256(modelfile_path),
    }
    archive_path = _store_lrm_archive(name, version_id, modelfile_path, metadata)
    _append_model_version(name, version_id, archive_path, base_model, metadata["dataset"], notes)
    typer.echo(f"Model '{name}' trained and archived at {archive_path.relative_to(project_root())}.")


@models_app.command("history")
def show_history() -> None:
    """Display recorded model history."""
    registry = load_model_registry()
    history = registry.get("history", [])
    if not history:
        typer.echo("No model history recorded yet.")
        return
    for entry in history:
        typer.echo(
            f"- {entry.get('timestamp')} {entry.get('model')} {entry.get('action', '')} "
            f"{entry.get('version', '')} {entry.get('lrm', '')}".strip()
        )


@models_app.command("rollback")
def rollback(
    version: Optional[str] = typer.Option(
        None,
        "--version",
        help="Version identifier to restore. If omitted, the latest available snapshot is used.",
    ),
    model: Optional[str] = typer.Option(
        None,
        "--model",
        help="Model to rollback. Required when multiple models exist.",
    ),
) -> None:
    """Restore a model version from its LRM archive."""
    registry = load_model_registry()
    models = registry.get("models", {})
    if not models:
        raise DeploymentError("No model snapshots found.")

    target_entry = None
    if model:
        target_entry = models.get(model)
        if not target_entry:
            raise DeploymentError(f"No registry entry for model '{model}'.")
    elif len(models) == 1:
        (single_model, target_entry) = next(iter(models.items()))
        model = single_model
    else:
        raise DeploymentError("Multiple models tracked; specify --model.")

    versions = target_entry.get("versions", [])
    if not versions:
        raise DeploymentError(f"No snapshots stored for model '{model}'.")

    if version:
        snapshot = next((item for item in versions if item["version"] == version), None)
        if not snapshot:
            raise DeploymentError(f"Version '{version}' not found for model '{model}'.")
    else:
        snapshot = versions[-1]

    archive_path = project_root() / snapshot["lrm"]
    if not archive_path.exists():
        raise DeploymentError(f"LRM archive missing: {archive_path}")

    with zipfile.ZipFile(archive_path, "r") as archive:
        if "Modelfile" not in archive.namelist():
            raise DeploymentError(f"Archive {archive_path} is missing Modelfile.")
        modelfile_content = archive.read("Modelfile").decode("utf-8")

    with tempfile.NamedTemporaryFile("w", delete=False, encoding="utf-8", suffix=".Modelfile") as tmp:
        tmp.write(modelfile_content)
        tmp_path = Path(tmp.name)

    binary = _resolve_ollama_binary()
    try:
        run_command([binary, "create", model, "-f", str(tmp_path)])
    finally:
        tmp_path.unlink(missing_ok=True)

    registry["active_model"] = model
    registry.setdefault("history", []).append(
        {
            "timestamp": timestamp(),
            "model": model,
            "action": "rollback",
            "version": snapshot["version"],
            "lrm": snapshot["lrm"],
        }
    )
    save_model_registry(registry)

    typer.echo(
        f"Restored model '{model}' to version {snapshot['version']} from {snapshot['lrm']}."
    )

