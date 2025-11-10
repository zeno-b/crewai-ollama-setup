# CrewAI + Ollama Setup

A comprehensive setup for running CrewAI with Ollama as the LLM backend, featuring custom agents, tasks, crews, and tools.

## Features

- **Self-Contained Deployment**: Cross-platform `scripts/deploy.py` bootstraps a secure virtual environment, dependencies, and optional managed Ollama binary.
- **Crew Control CLI**: `crewctl` automates agent scaffolding, model pulls/switches, pre-training into LRM snapshots, and rollbacks.
- **Ollama Integration**: Seamless integration with Ollama for local LLM inference.
- **Custom Agents & Tasks**: Templates and registry tracking for repeatable automation workflows.
- **Model Versioning**: LRM archive generation enables travel between model versions with a single command.
- **Docker Support**: Containerized setup for easy deployment (optional).
- **Configuration Management**: Centralized settings management.

## Self-Contained Deployment (Recommended)

### 1. Prerequisites

- Python ≥ 3.10
- Internet access for dependency/model downloads
- Verified checksums for the Ollama release you plan to manage (see `deployment/ollama_checksums.json`)

### 2. Configure

1. Review `deployment/config.yml` to adjust:
   - Virtual environment path
   - Ollama install strategy (`system`, `managed`, or `skip`)
   - CLI templates and registries
2. If using `managed` Ollama installs, populate SHA-256 digests in `deployment/ollama_checksums.json`. The deploy script refuses to proceed with placeholder values, ensuring tamper detection.

### 3. Deploy

```bash
python scripts/deploy.py
```

Key options:

- `--recreate-venv` – rebuild the virtualenv from scratch
- `--skip-install` – skip Python dependency installation
- `--skip-ollama` – bypass Ollama setup (useful when relying on an external installation)

### 4. Activate the Toolkit

```bash
# macOS/Linux
source .venv/bin/activate

# Windows (PowerShell)
.venv\Scripts\activate
```

The deploy script also creates a convenience launcher inside the virtualenv:

- macOS/Linux: `.venv/bin/crewctl`
- Windows: `.venv\Scripts\crewctl.cmd`

### 5. Use the CLI

```bash
# list registered agents
crewctl agents list

# scaffold a new agent file from the template
crewctl agents create "Research Analyst" --role "Research Analyst" --goal "Synthesize reports"

# inspect available models
crewctl models list

# pull or switch models
crewctl models pull llama3
crewctl models switch llama3

# pre-train a custom model (creates an LRM archive you can roll back to)
crewctl models pretrain custom-lrm --base-model llama3 --notes "Finance tuning"
```

Snapshots generated via `crewctl models pretrain` are stored under `models/lrms/` and logged in `config/model_registry.yaml`. Roll back at any point:

```bash
crewctl models rollback --model custom-lrm
```

### 6. Environment Files

The script seeds `config/agent_registry.yaml` and `config/model_registry.yaml` on first run and tightens permissions for sensitive metadata.

---

## Docker-Based Setup (Optional)

The legacy Docker workflow remains available if you prefer containers.

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

## Docker Services

- **ollama**: Ollama server (port 11434)
- **crewai**: CrewAI application container

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
