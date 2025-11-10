# CrewAI + Ollama Automation Toolkit

Self-contained automation scripts for building, securing, and operating a CrewAI + Ollama environment across Linux, macOS, and Windows.

Key capabilities:
- Python-based deploy script that hardens configuration, provisions a virtual environment, and produces Docker manifests.
- `crewctl` CLI inside the virtual environment for day-to-day operations (agents, models, model versioning, pre-training).
- Secure defaults: random secret generation, deterministic config scaffolding, isolated directories, and optional Docker usage.

---

## Deployment Workflow

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd crewai-ollama-setup
   ```

2. **Run the multi-platform deployer**
   ```bash
   python scripts/deploy.py
   ```
   - Creates `.venv`, installs dependencies, and generates `.env` with random secrets.
   - Writes `docker-compose.deploy.yml` (override with `--no-docker`).
   - Initializes `config/model_registry.yaml` for version tracking.
   - Records deployment metadata in `config/deploy_state.json`.

   Options:
   - `--config config/deploy_config.yaml` – supply custom settings (JSON-compatible YAML).
   - `--no-docker` – skip Docker validation/manifest generation if running Ollama outside containers.

3. **Activate the environment and launch the CLI**
   ```bash
   source .venv/bin/activate          # PowerShell: .\.venv\Scripts\Activate.ps1
   crewctl --help
   ```

4. **Bring services online (optional)**
   ```bash
   docker compose -f docker-compose.deploy.yml up -d
   ```

---

## crewctl CLI Overview

`crewctl` ships inside the virtual environment created by `scripts/deploy.py`. All commands assume the venv is active.

### Agent Management
- `crewctl agents create "<Name>" --role ... --goal ... --backstory ...`
- `crewctl agents list`
- `crewctl agents show <slug>`

Agent definitions are stored under `agents/` as YAML with metadata (name, role, goal, backstory, tools).

### Model Lifecycle
- `crewctl models list` – view models available on the Ollama server.
- `crewctl models pull <model>` – download a model with streaming progress.
- `crewctl models switch <model> [--pull]` – activate a model and record the change.
- `crewctl models history` – inspect activation history from `config/model_registry.yaml`.
- `crewctl models rollback` – revert to the previously active model.
- `crewctl models activate <model>` – reactivate any model from history.
- `crewctl models pretrain <name> --base <model> [options]` – render a modelfile from `models/templates/lrm_modelfile.j2`, build an LRM via Ollama, optionally activate it.
- `crewctl models inspect <model>` – raw metadata, `crewctl models info` – server version data.

Version changes automatically update `.env` (`OLLAMA_MODEL`), append to the registry, and enforce reproducibility.

### Pre-Training Parameters
- `--system-prompt` or `--system-prompt-file` embed custom instructions.
- `--adapter` path adds LoRA adapters.
- `--quantize` sets Ollama quantization target.
- `--activate` flips to the new model after creation.

The rendered modelfile is saved to `models/build/<name>.modelfile` for auditability.

---

## Security Posture

- **Secrets**: `SECRET_KEY` and `JWT_SECRET` generated with `secrets.token_urlsafe`; `.env` permissions hardened on POSIX.
- **Deterministic Infrastructure**: Docker manifest uses explicit environment and volume configuration; no remote install scripts run implicitly.
- **Registry**: `config/model_registry.yaml` keeps append-only activation history for forensic traceability.
- **Isolation**: `.venv` ensures dependencies never leak to the system interpreter; CLI shim injects `PYTHONPATH` instead of installing editable packages globally.
- **Config Auditability**: Deployment config (`config/deploy_config.yaml`) is JSON-compatible to avoid parser ambiguity before PyYAML is installed.

---

## File Layout (Key Additions)

```
scripts/
  deploy.py                # Cross-platform deployment orchestrator
crewctl/
  __init__.py
  __main__.py
  cli.py                   # Typer CLI for agents/models
  configuration.py         # .env + registry helpers
  ollama.py                # REST client for Ollama
  utils.py                 # Shared helpers
models/
  templates/lrm_modelfile.j2
  build/                   # Generated modelfiles (.gitkeep)
config/
  deploy_config.yaml       # JSON-compatible defaults
  model_registry.yaml      # Auto-managed model history
.env.template              # Secure defaults (secrets populated on deployment)
```

Legacy service code (FastAPI app, custom agents/tasks/tools) remains under `main.py`, `agents/`, `tasks/`, `crews/`, and `tools/`.

---

## Operating the Stack

1. **Start dependencies (optional)**
   ```bash
   docker compose -f docker-compose.deploy.yml up -d
   ```
2. **Launch FastAPI service**
   ```bash
   source .venv/bin/activate
   uvicorn main:app --reload
   ```
3. **Iterate on agents/models using crewctl**
   - Scaffold new agents, adjust modelfiles, and switch models without editing `.env` manually.
   - Use `crewctl models rollback` to recover quickly if a deployment regresses.

---

## Troubleshooting Tips

- **Ollama not reachable**: verify `docker compose ps`, check `OLLAMA_BASE_URL`, and inspect `crewctl models info`.
- **Model pull fails**: ensure the server has network access; re-run with `--pull` or manually `crewctl models pull`.
- **Permission issues**: on Linux/macOS, ensure workspace owner matches Docker user ID; rerun `scripts/deploy.py` to reset perms.
- **CLI not found**: confirm venv is activated or call `.venv/bin/crewctl` directly.

---

## Next Steps

- Extend `models/templates/lrm_modelfile.j2` with organization-specific parameters.
- Add CI/CD hooks that run `scripts/deploy.py --no-docker` for infrastructure validation.
- Leverage `config/settings.py` (once patched) or environment variables to integrate with secured redis/databases.

For deeper CrewAI customization, continue using the existing `agents/`, `tasks/`, and `crews/` modules alongside the new automation tooling.
