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

2. Start the services:
```bash
docker-compose up -d
```

3. Verify Ollama is running:
```bash
curl http://localhost:11434/api/tags
```

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

Grafana provides a pre-built **CrewAI Overview** dashboard and connects automatically to Prometheus. Default credentials are `admin / ${GRAFANA_PASSWORD:-admin}`.

## Retraining Workflow

### Manage Datasets

- Upload or replace a dataset:
  ```bash
  curl -X POST http://localhost:8000/datasets \
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
- Remove a dataset: `curl -X DELETE http://localhost:8000/datasets/domain_faq`

### Launch Retraining

- Kick off a job (accepts an optional `modelfile_template` with `{{DATASET}}` placeholder):
  ```bash
  curl -X POST http://localhost:8000/retraining/jobs \
       -H "Content-Type: application/json" \
       -d '{
             "model_name": "domain-faq:latest",
             "base_model": "llama2:7b",
             "dataset_name": "domain_faq",
             "instructions": "Prioritize answers sourced from the inlined dataset.",
             "parameters": {"temperature": 0.2},
             "stream": true
           }'
  ```
- Check status: `curl http://localhost:8000/retraining/jobs`
- Inspect a specific job: `curl http://localhost:8000/retraining/jobs/<job_id>`
- Tail logs: `curl http://localhost:8000/retraining/jobs/<job_id>/logs?tail=50`

> The default retraining template embeds dataset content into the Modelfile `SYSTEM` prompt (limited to ~120k characters). For larger corpora, provide a custom `modelfile_template` that references external artifacts and include `{{DATASET}}` where you want the dataset injected.

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
