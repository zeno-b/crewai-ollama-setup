# CrewAI + Ollama Self-Contained Deployment

This repository bundles everything required to spin up a secure CrewAI + Ollama stack with a single deployment script. The deployer provisions a Python virtual environment, installs all dependencies, prepares configuration, and wires in a management CLI (`crewaictl`) for working with agents and models across Linux, macOS, and Windows.

---

## Key Capabilities

- **One-command bootstrap** via `deployment/deploy.py` (creates `.venv`, secrets, config, directories).
- **Docker-native runtime** for CrewAI, Ollama, Redis, Prometheus, and Grafana with hardening defaults.
- **Management CLI** (`crewaictl`) to:
  - scaffold and list CrewAI agents,
  - pull/switch/promote Ollama models,
  - pre-train custom LRMs (via `ollama create`) with version tracking,
  - roll back to previous model versions.
- **Security-conscious defaults**: least-privilege containers, randomly generated secrets, segregated state.
- **Monitoring ready**: Prometheus scrape config and Grafana dashboards included.

---

## Prerequisites

- Python **3.10+**
- Docker Engine (Linux) or Docker Desktop (macOS/Windows) with Docker Compose v2+
- Internet access for pulling Docker images / Ollama models (unless pre-seeded)

---

## Quick Deployment

```bash
git clone <repository-url>
cd crewai-ollama-setup

# Provision virtualenv, install deps, scaffold config & CLI
python deployment/deploy.py

# (Optional) start the full stack once provisioning completes
python deployment/deploy.py --start-services
```

What the script does:
1. Checks Python + Docker prerequisites.
2. Creates runtime directories (`data/`, `logs/`, `models/lrms/`, `deployment/state/`).
3. Generates `.env` with random `SECRET_KEY`, `JWT_SECRET`, `GRAFANA_PASSWORD`.
4. Creates/updates `config/model_registry.yaml`.
5. Builds a virtual environment (`.venv`) and installs `requirements.txt`.
6. Drops CLI entry points at `.venv/bin/crewaictl` (or `.venv\Scripts\crewaictl.cmd` on Windows).
7. Optionally launches the Docker Compose stack.

---

## Using the Management CLI (`crewaictl`)

Activate the virtual environment first:

```bash
source .venv/bin/activate        # Linux / macOS
.venv\Scripts\activate           # Windows PowerShell/CMD
```

Then explore the CLI:

```bash
crewaictl --help
```

### Agent commands

```bash
crewaictl agents list
crewaictl agents create "Research Analyst" \
  --role "Research Specialist" \
  --goal "Synthesize market intelligence" \
  --backstory-file docs/agent_backstories/analyst.txt \
  --tool serpapi --tool webpilot
```

Generates `agents/research_analyst.py` from a secure template that auto-connects to the configured Ollama model.

### Model + LRM commands

```bash
crewaictl models list
crewaictl models pull llama3:8b --set-default

crewaictl models train finance-lrm \
  --dataset datasets/finance_corpus.txt \
  --base-model llama3:8b \
  --system-prompt "You are a financial domain expert." \
  --temperature 0.6

crewaictl models switch finance-lrm:20251110104512
crewaictl models versions
crewaictl models rollback --steps 1
```

Every pull/train updates `config/model_registry.yaml`, records digests, and keeps a history for fast rollback. `.env` is automatically patched whenever the default model changes.

---

## Docker Stack Overview

`docker-compose.yml` (used by deployment) runs:

- `ollama`: LLM runtime with no-new-privileges, read-only filesystem, health checks.
- `crewai`: FastAPI service exposing crews with Redis caching and metrics.
- `redis`: persistence for agent/task state.
- `prometheus` & `grafana`: monitoring with pre-baked configs/dashboards.

All services share an isolated Docker network (`crewai-network`); volumes are scoped to `data/`, `logs/`, and `models/`.

Launch manually if desired:

```bash
docker compose up -d          # or: docker-compose up -d
docker compose logs -f crewai
```

---

## Configuration Notes

- `.env` drives runtime configuration; regenerate secrets by deleting it and rerunning the deployer.
- `config/settings.py` consumes environment variables (ensure typos are fixed before production).
- `config/model_registry.yaml` persists model history; edit carefully or use `crewaictl`.
- Generated Modelfiles live under `models/lrms/<name>/`.

---

## Project Layout

```
deployment/
  deploy.py                # main deployment orchestrator
  templates/.env.example   # environment template consumed by deployer
crewai_cli/                # CLI package (invoked via crewaictl)
config/
  settings.py              # application settings (uses .env)
  model_registry.yaml      # model/version history
  prometheus.yml           # scrape targets
  grafana/…                # datasource + dashboards
agents/, tasks/, crews/    # domain logic (extend via CLI templates)
docker-compose.yml         # secure multi-service stack
requirements.txt           # runtime + CLI dependencies
```

---

## Security Defaults & Recommendations

- Containers run as non-root users with `no-new-privileges`, read-only filesystems, and tmpfs scratch space.
- Secrets are randomised on first deploy; rotate by removing `.env` and rerunning.
- `.venv`, `deployment/state/`, and `models/lrms/` are ignored by Git via `.gitignore`.
- Consider enabling TLS termination (e.g., Traefik/Caddy) and network firewalls for production deployments.

---

## Troubleshooting

- **Docker not detected**: ensure `docker --version` works and current user can run Docker commands.
- **Ollama API errors**: check `docker compose logs ollama`; confirm GPU/CPU resources meet model requirements.
- **CLI cannot reach Ollama**: verify `OLLAMA_BASE_URL` in `.env`, container health (`docker compose ps`), or network ACLs.
- **Model training stalled**: inspect the generated Modelfile under `models/lrms/` and rerun `crewaictl models train --modelfile <path>`.

---

## Next Steps

- Add custom agents/tasks using `crewaictl agents create`.
- Expand monitoring dashboards (`config/grafana/dashboards/`).
- Automate deployments via CI/CD calling `python deployment/deploy.py --start-services`.

Enjoy building autonomous crews! 🚀
