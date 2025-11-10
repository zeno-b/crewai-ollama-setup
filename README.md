# CrewAI Service

Modernized FastAPI service that orchestrates CrewAI workflows on top of an Ollama LLM backend. The codebase has been streamlined for maintainability, audited for obsolete components, and hardened in line with SANS-oriented security practices.

## Key Features

- **FastAPI + Ollama Integration**: Asynchronous API surface for Agent and Crew orchestration backed by the Ollama runtime.
- **Secure by Default**: Mandatory bearer-token authentication, defensive Redis usage, and explicit dependency pinning to reduce supply-chain risk.
- **Observability**: Prometheus metrics, structured logging, and health checks for production readiness.
- **Background Execution**: Crew runs execute asynchronously while results persist to Redis with expiring TTLs.
- **Config Management**: Single `config/settings.py` source centralizes environment-driven configuration.

## Getting Started

### 1. Install dependencies

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Configure environment

Create a `.env` file or export variables before running:

```
API_BEARER_TOKEN=your-strong-token
REDIS_URL=redis://localhost:6379/0
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama2:latest
```

### 3. Launch the API

```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

## API Overview

| Endpoint | Method | Auth | Description |
| --- | --- | --- | --- |
| `/health` | GET | Optional | Returns service, Redis, and Ollama status. |
| `/metrics` | GET | Optional | Exposes Prometheus metrics. |
| `/models` | GET | Required | Lists Ollama models via upstream `/api/tags`. |
| `/create_agent` | POST | Required | Registers agent metadata and persists it in Redis. |
| `/run_crew` | POST | Required | Launches a crew in the background and returns a crew ID. |
| `/crew_results/{crew_id}` | GET | Required | Retrieves stored crew execution results. |
| `/` | GET | Optional | Provides quick navigation links. |

All protected routes expect an `Authorization: Bearer <API_BEARER_TOKEN>` header.

## Configuration

`config/settings.py` centralizes all tunables, including:

- Service host/port/debug behavior.
- Ollama endpoint, model, and timeout.
- Redis URL, max connections, and TTLs for stored results.
- Logging format and level.
- Allowed CORS origins and API bearer token.

Each field is annotated and comment-documented for quick understanding.

## Security Notes

- Rotate `API_BEARER_TOKEN` regularly and avoid using the default value (`change-me`).
- Use TLS termination in front of the service when deploying externally.
- Restrict Redis access to trusted network segments.
- Keep dependencies patched by periodically bumping versions in `requirements.txt`.

## Observability

- `/metrics` exposes counters for request volume, request latency, agent creation, and crew executions.
- Logs include context and stack traces where appropriate for easier incident response.

## Project Layout

```
.
├── config/
│   └── settings.py   # Environment-driven configuration (each line documented)
├── docker-compose.yml
├── Dockerfile.crewai
├── main.py           # FastAPI application (each line documented)
├── README.md
└── requirements.txt
```

Legacy agent/task/tool scaffolding has been removed to eliminate obsolete, unused code paths.

## Development Tips

- Run `uvicorn main:app --reload` during development to leverage hot reload (enabled automatically when `ENVIRONMENT=development`).
- Linting and tests can be layered on top of this base; no prior suite is included after the cleanup.
- When adding code, follow the established convention of documenting every executable line for traceability.

## License

MIT License. See `LICENSE` for terms.
