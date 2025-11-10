# CrewAI + Ollama Setup

A comprehensive setup for running CrewAI with Ollama as the LLM backend, featuring custom agents, tasks, crews, and tools. The environment can now be provisioned in a few minutes with a single command and comes with a powerful CLI for day-to-day operations.

## Features

- **Self-contained deployment**: multi-platform installer (`scripts/deploy.py`) creates an isolated virtual environment, installs Ollama, and bootstraps configuration automatically.
- **Ollama integration**: sandboxed binary lives under `.crewctl/bin`, keeping system packages untouched.
- **crewctl CLI**: manage agents, switch models, fine-tune adapters (LRMs), and inspect system health from a unified interface.
- **Custom agents / tasks / tools**: templates and scaffolding to build bespoke automations quickly.
- **Security-conscious defaults**: non-root operation, checksum verification, and hardened file permissions.
- **Optional training extras**: enable LoRA/LRM fine-tuning with a single flag.

## Quick Start

### 1. Clone the repository

```bash
git clone <repository-url>
cd crewai-ollama-setup
```

### 2. Run the deployment script

```bash
python scripts/deploy.py
```

What it does:

- Detects the host OS (Linux, macOS, Windows) and installs the correct Ollama build into `.crewctl/bin`
- Creates a Python virtual environment at `.crewctl/venv`
- Installs application dependencies together with the `crewctl` CLI
- Generates supportive configuration (`.env`, `.crewctl/config.yaml`, `config/model_registry.json`)

Helpful flags:

- `--with-training-extras` – install PyTorch, Transformers, PEFT, etc. for LoRA/LRM training
- `--force` – recreate the virtual environment and reinstall Ollama from scratch
- `--verbose` – enable detailed logs (also written to `.crewctl/logs/deploy.log`)

The script is idempotent and safe to rerun; it reconciles components as needed.

### 3. Use the crewctl CLI

```bash
# macOS/Linux
.crewctl/venv/bin/crewctl --help

# Windows
.crewctl\venv\Scripts\crewctl.cmd --help
```

Key commands:

- `crewctl doctor` – verify environment health and active configuration
- `crewctl agent create "Name"` – scaffold a new agent from a secure template
- `crewctl model list|pull|use|history|activate` – manage Ollama models and roll back/forward between versions
- `crewctl train lrm --dataset <spec>` – fine-tune base models into LRMs (LoRA adapters) and optionally register them with Ollama

Example – create a new analyst agent:

```bash
.crewctl/venv/bin/crewctl agent create "Senior Analyst" \
  --role "Senior Data Analyst" \
  --goal "Deliver exceptional insights" \
  --backstory "A seasoned analyst with a security-first mindset."
```

### 4. (Optional) Activate the virtual environment

```bash
source .crewctl/venv/bin/activate       # macOS/Linux
.crewctl\venv\Scripts\activate          # Windows PowerShell/CMD
```

## Configuration

- `.env` – runtime overrides (the deploy script creates one from `.env.template` if needed). `crewctl model use` keeps `OLLAMA_MODEL` up to date here.
- `.crewctl/config.yaml` – CLI preferences (paths, default training hyperparameters).
- `config/settings.py` – FastAPI/ application settings (endpoints, logging, cache sizing, etc.).
- `config/model_registry.json` – tracked model history for easy rollbacks.

## Usage

### Basic Example

```python
from crews.custom_crew import CustomCrew
from config.settings import settings

# Initialize crew
crew = CustomCrew(
    name="research_crew",
    agents=["researcher", "analyst"],
    tasks=["web_search", "data_analysis"]
)

# Run crew
result = crew.run()
```

### Custom Agents

Define agents manually in `agents/custom_agent.py` **or** generate new agents via the CLI:

```bash
.crewctl/venv/bin/crewctl agent create "Security Researcher" \
  --role "Security Researcher" \
  --goal "Identify emerging threats" \
  --backstory "A veteran analyst focused on defensive research."
```

Every generated agent inherits from `crewai.Agent` and can be customized further.

### Custom Tasks

Define tasks in `tasks/custom_task.py`:

```python
from tasks.custom_task import CustomTask

task = CustomTask(
    description="Research topic: {topic}",
    expected_output="Detailed research report"
)
```

### Custom Tools

Extend functionality via `tools/custom_tools.py`:

```python
from tools.custom_tools import ToolFactory

# Get all tools
tools = ToolFactory.get_all_tools()

# Use specific tool
search_tool = ToolFactory.get_tool_by_name("Web Search")
result = search_tool._run(query="AI latest developments")
```

## Project Structure

```
crewai-ollama-setup/
├── scripts/
│   └── deploy.py                 # Multi-platform deployment script
├── crewctl/                      # CLI package
│   ├── cli.py
│   ├── agent.py
│   ├── model.py
│   ├── training.py
│   └── templates/
├── .crewctl/                     # Created at runtime (venv, logs, downloads, etc.)
├── config/
│   ├── settings.py
│   └── model_registry.json
├── agents/
├── tasks/
├── crews/
├── tools/
├── requirements.txt
├── requirements-deploy.txt
├── requirements-train.txt
├── pyproject.toml
└── README.md
```

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `OLLAMA_BASE_URL` | Ollama endpoint used by CrewAI | `http://localhost:11434` |
| `OLLAMA_MODEL` | Default model name (managed by `crewctl model use`) | `llama3` |
| `CREWAI_PORT` | HTTP port for FastAPI service | `8000` |
| `CREWAI_LOG_LEVEL` | Application log level | `INFO` |
| `REDIS_URL` | Redis connection string (if used) | `redis://localhost:6379` |

## Troubleshooting

1. **Deploy script cannot find Ollama binary**
   - Re-run `python scripts/deploy.py --force`
   - Ensure `.crewctl/bin/ollama` exists and is executable (`chmod +x` on Unix)

2. **Model not available**
   - Pull it through the CLI: `.crewctl/venv/bin/crewctl model pull llama3`

3. **Training dependencies missing**
   - Re-run deployment with extras: `python scripts/deploy.py --with-training-extras`

4. **Permission errors**
   - Avoid running the repository as `root`
   - On Unix systems `.crewctl` directories are hardened to mode `750`

### Debug Mode

Enable verbose logs for the FastAPI service:

```bash
export CREWAI_LOG_LEVEL=DEBUG
python -m uvicorn main:app --reload
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Implement changes and add tests when possible
4. Submit a pull request

## License

MIT License – see `LICENSE` for details.

## Support

- Run `crewctl doctor` for environment diagnostics
- Review this README and the troubleshooting section
- Consult official CrewAI and Ollama documentation for advanced topics
