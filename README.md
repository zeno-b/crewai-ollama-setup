# CrewAI + Ollama Setup

A comprehensive setup for running CrewAI with Ollama as the LLM backend, featuring custom agents, tasks, crews, and tools.

## Features

- **Ollama Integration**: Seamless integration with Ollama for local LLM inference
- **Custom Agents**: Pre-configured agents for various tasks
- **Task Templates**: Reusable task definitions
- **Crew Management**: Organized crew structures
- **Custom Tools**: Extensible tool collection for web search, file operations, code execution, and more
- **Docker Support**: Containerized setup for easy deployment
- **Configuration Management**: Centralized settings management

## Quick Start

### Prerequisites

- Docker and Docker Compose
- Ollama installed and running locally

You can let the project install and configure these prerequisites for you (Ubuntu/Debian):

```bash
chmod +x scripts/bootstrap.sh
./scripts/bootstrap.sh --with-ollama-cli
```

Alternatively, run the bootstrap via Python:

```bash
python setup.py --prereqs --with-ollama-cli
```

Both commands verify Docker, Docker Compose, and Ollama, start the local Ollama container, and pre-pull the default model (`llama2:7b`). Use `--non-interactive` to run without prompts.

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd crewai-ollama-setup
```

2. (Optional) Confirm `requirements.txt` resolves in the same image stack as the CrewAI container (no host Python needed):

```bash
./scripts/verify-requirements.sh
```

3. Start the services:
```bash
docker-compose up -d
```

4. Verify Ollama is running:
```bash
curl http://localhost:11434/api/tags
```

### Testing

Run the suite on the host (Python 3.11+ recommended to match the container):

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pytest tests/ -v
```

Run the same tests **inside** the `python:3.11-slim` image used by the service (fully containerized):

```bash
./scripts/run-tests-docker.sh
```

Performance smoke checks are marked `@pytest.mark.performance`; security checks use `@pytest.mark.security`. Skip performance tests with `pytest tests/ -m "not performance"`.

### Configuration

Edit `config/settings.py` to customize:
- Ollama endpoint
- Model selection
- Agent configurations
- Tool settings

### Usage

#### Basic Example

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

#### Custom Agents

Create custom agents in `agents/custom_agent.py`:

```python
from agents.custom_agent import CustomAgent

agent = CustomAgent(
    role="Senior Researcher",
    goal="Find comprehensive information",
    backstory="Expert researcher with 10 years experience"
)
```

#### Custom Tasks

Define tasks in `tasks/custom_task.py`:

```python
from tasks.custom_task import CustomTask

task = CustomTask(
    description="Research topic: {topic}",
    expected_output="Detailed research report"
)
```

#### Custom Tools

Use built-in tools or create new ones:

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
├── docker-compose.yml          # Docker services configuration
├── Dockerfile.crewai           # CrewAI container setup
├── requirements.txt            # Python dependencies
├── main.py                     # Entry point
├── config/
│   └── settings.py             # Configuration management
├── agents/
│   └── custom_agent.py         # Custom agent definitions
├── tasks/
│   └── custom_task.py          # Custom task definitions
├── crews/
│   └── custom_crew.py          # Crew management
├── tools/
│   └── custom_tools.py         # Custom tools and utilities
├── setup.py                    # Package setup
└── README.md                   # This file
```

## Available Tools

### Web Search
- **WebSearchTool**: Search the web for information
- Supports multiple search engines
- Configurable result limits

### File Operations
- **FileReadTool**: Read file contents
- **FileWriteTool**: Write content to files
- Supports various encodings

### Code Execution
- **CodeExecuteTool**: Execute Python code safely
- Timeout protection
- Error handling

### API Integration
- **APIRequestTool**: Make HTTP requests
- Support for all HTTP methods
- Custom headers and data

### Data Analysis
- **DataAnalysisTool**: Analyze structured data
- Summary statistics
- Correlation analysis

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `OLLAMA_HOST` | Ollama server host | `localhost` |
| `OLLAMA_PORT` | Ollama server port | `11434` |
| `OLLAMA_MODEL` | Default model name | `llama2` |
| `CREWAI_LOG_LEVEL` | Logging level | `INFO` |
| `CREWAI_WORKERS` | Override auto-calculated Gunicorn worker count | Auto (`2 * CPU + 1`) |
| `CREWAI_TIMEOUT` | Gunicorn worker timeout (seconds) | `90` |
| `CREWAI_GRACEFUL_TIMEOUT` | Gunicorn graceful shutdown timeout (seconds) | `30` |
| `GUNICORN_EXTRA` | Additional flags appended to the Gunicorn command | *(empty)* |
| `DATASET_DIR` | Filesystem path for retraining datasets | `data/datasets` |
| `RETRAINING_DIR` | Filesystem path for retraining job artifacts | `data/retraining` |
| `SECRET_KEY` | Signing / session secret (min 32 chars; random in dev if unset) | *(random per process in dev)* |
| `API_BEARER_TOKEN` | If set, required on `POST`/`DELETE` mutating routes (datasets, retraining, agents, crews) | *(unset)* |
| `METRICS_BEARER_TOKEN` | If set, required for `GET /metrics` | *(unset)* |
| `CORS_ALLOW_ORIGINS` | Comma-separated browser origins, or `*` (blocked when `ENVIRONMENT=production`) | `http://localhost:3000,...` |
| `DATASET_MAX_CONTENT_BYTES` | Max raw bytes per dataset upload | `5242880` |
| `MODELFILE_TEMPLATE_DIR` | Directory of `<name>.template` files for `template_name` on jobs | `config/modelfiles` |
| `REDIS_PASSWORD` | Optional; injected into `REDIS_URL` when the URL has no `@` credentials | *(unset)* |
| `NEWS_AUTOPILOT_*` | Background RSS ingestion and auto-retrain policy | see `.env.example` |

### Performance Tuning

- Gunicorn worker count auto-scales to `2 * CPU + 1`; set `CREWAI_WORKERS` to pin an exact value.
- Adjust long-running requests with `CREWAI_TIMEOUT` and graceful shutdown with `CREWAI_GRACEFUL_TIMEOUT`.
- Provide extra Gunicorn flags (for example, `--access-logfile -`) via `GUNICORN_EXTRA`.

## Docker Services

- **ollama**: Ollama server (port 11434)
- **crewai**: CrewAI application container
- **redis**: In-memory cache and task result store
- **prometheus** *(optional, profile `monitoring`)*: Metrics scraping and storage
- **grafana** *(optional, profile `monitoring`)*: Metrics visualization

To include the monitoring stack:

```bash
docker compose up -d
docker compose --profile monitoring up -d
```

> Start the core services first (`docker compose up -d`) and only then enable the monitoring profile. You can later stop the dashboards without touching the app via `docker compose --profile monitoring down`.

Grafana provides a pre-built **CrewAI Overview** dashboard and connects automatically to Prometheus. Default credentials are `admin / ${GRAFANA_PASSWORD:-admin}`.

### Compose scenario overlays

The `compose/` directory holds **optional merge files** for common setups (see `compose/README.md`):

- **Redis password — option A**: `REDIS_URL` without credentials + `REDIS_PASSWORD` (app merges password into the DSN).
- **Redis password — option B**: `REDIS_URL` with embedded `redis://:pass@redis:6379/0` + same `REDIS_PASSWORD` for the Redis server only (`REDIS_PASSWORD` cleared for `crewai` so the URL is not double-encoded).
- **News autopilot**: toggles `NEWS_AUTOPILOT_*` variables on the `crewai` service.

Example:

```bash
docker compose -f docker-compose.yml \
  -f compose/docker-compose.redis-password-option-a.yml \
  -f compose/docker-compose.scenario.news-autopilot.yml \
  up -d
```

### Autonomous news → training data → retrain

When `NEWS_AUTOPILOT_ENABLED=true`, the API process periodically downloads an RSS/Atom feed, normalizes headlines into **`NEWS_AUTOPILOT_DATASET_NAME`** (`text` or `jsonl`), optionally merges with the existing corpus, and **schedules an Ollama retraining job** when configurable deltas (`NEWS_AUTOPILOT_MIN_NEW_LINES_TO_RETRAIN`, `NEWS_AUTOPILOT_MIN_NEW_BYTES_TO_RETRAIN`, or `NEWS_AUTOPILOT_RETRAIN_ON_ANY_CHANGE`) and `NEWS_AUTOPILOT_MIN_HOURS_BETWEEN_RETRAINS` allow it. State (last digest, sizes, last retrain time) is stored in Redis when available.

| Variable | Role |
|----------|------|
| `NEWS_AUTOPILOT_RSS_URL` | Feed to poll |
| `NEWS_AUTOPILOT_POLL_INTERVAL_SECONDS` | Sleep between attempts |
| `NEWS_AUTOPILOT_OUTPUT_FORMAT` | `text` (SYSTEM-friendly) or `jsonl` (pairs for `distill`) |
| `NEWS_AUTOPILOT_MIN_*` | Heuristics for “enough new data” before retrain |
| `NEWS_AUTOPILOT_JOB_TYPE` | `system_prompt` or `distill` (distill requires `NEWS_AUTOPILOT_TEACHER_MODEL`) |

## Retraining Workflow

### Manage Datasets

When `API_BEARER_TOKEN` is set (recommended outside local dev), send `Authorization: Bearer <token>` on mutating requests.

- Upload or replace a dataset:
  ```bash
  curl -X POST http://localhost:8000/datasets \
       -H "Authorization: Bearer ${API_BEARER_TOKEN}" \
       -H "Content-Type: application/json" \
       -d '{
             "name": "domain_faq",
             "description": "Frequently asked questions for the support domain",
             "tags": ["support", "faq"],
             "content": "Q: ...\nA: ...",
             "overwrite": true
           }'
  ```
- List datasets: `curl http://localhost:8000/datasets`
- Inspect a dataset (with content): `curl "http://localhost:8000/datasets/domain_faq?include_content=true"`
- Remove a dataset: `curl -X DELETE -H "Authorization: Bearer ${API_BEARER_TOKEN}" http://localhost:8000/datasets/domain_faq`

### Launch Retraining

- **System prompt style** (`job_type` defaults to `system_prompt`): embeds dataset text in a `SYSTEM` block. Optional `modelfile_template` with placeholders: `{{DATASET}}`, `{{BASE_MODEL}}`, `{{MODEL_NAME}}`, `{{TEACHER_MODEL}}`, `{{INSTRUCTIONS}}`, `{{ADAPTER}}`.
- **Distillation-oriented** (`job_type`: `distill`): set `teacher_model` to the Ollama teacher name and supply `modelfile_template` or `template_name` (files under `MODELFILE_TEMPLATE_DIR`, e.g. `example_distill`). For `format: jsonl`, each line can be `{"role":"user","content":"..."} / {"role":"assistant","content":"..."}`; the service expands them to `MESSAGE user` / `MESSAGE assistant` lines for the Modelfile. You still define how the teacher is invoked inside your template (Ollama `TEMPLATE` / `MESSAGE` patterns).

- Kick off a job (accepts an optional `modelfile_template` with `{{DATASET}}` placeholder):
  ```bash
  curl -X POST http://localhost:8000/retraining/jobs \
       -H "Authorization: Bearer ${API_BEARER_TOKEN}" \
       -H "Content-Type: application/json" \
       -d '{
             "model_name": "domain-faq:latest",
             "base_model": "llama2:7b",
             "dataset_name": "domain_faq",
             "job_type": "system_prompt",
             "instructions": "Prioritize answers sourced from the inlined dataset.",
             "parameters": {"temperature": 0.2},
             "stream": true
           }'
  ```
- Check status: `curl http://localhost:8000/retraining/jobs`
- Inspect a specific job: `curl http://localhost:8000/retraining/jobs/<job_id>`
- Tail logs: `curl http://localhost:8000/retraining/jobs/<job_id>/logs?tail=50`

> The default retraining template embeds dataset content into the Modelfile `SYSTEM` prompt (limited to ~120k characters). For larger corpora, provide a custom `modelfile_template` that references external artifacts and include `{{DATASET}}` where you want the dataset injected. Docker Compose requires `SECRET_KEY` in `.env` (see `.env.example`).

### Models API

`GET /models` calls Ollama’s `GET /api/tags` on `OLLAMA_BASE_URL` and returns the live catalog (no hard-coded placeholder list).

## Development

### Adding New Tools

1. Create tool class in `tools/custom_tools.py`
2. Inherit from `BaseTool`
3. Define input schema using Pydantic
4. Register in `ToolFactory`

### Adding New Agents

1. Define agent in `agents/custom_agent.py`
2. Set role, goal, and backstory
3. Configure tools and model

### Adding New Tasks

1. Define task in `tasks/custom_task.py`
2. Set description and expected output
3. Configure agent assignment

## Troubleshooting

### Common Issues

1. **Ollama Connection Failed**
   - Check if Ollama is running: `docker ps`
   - Verify endpoint: `curl http://localhost:11434/api/tags`

2. **Model Not Found**
   - Pull required model: `docker exec ollama ollama pull llama2`

3. **Permission Errors**
   - Check file permissions in mounted volumes
   - Ensure Docker has necessary access

### Debug Mode

Enable debug logging:
```bash
export CREWAI_LOG_LEVEL=DEBUG
docker-compose up
```

## Contributing

1. Fork the repository
2. Create feature branch
3. Add tests for new functionality
4. Submit pull request

## License

MIT License - see LICENSE file for details

## Support

For issues and questions:
- Check the troubleshooting section
- Open an issue on GitHub
- Review CrewAI documentation
