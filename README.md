# CrewAI + Ollama Setup

A comprehensive setup for running CrewAI with Ollama as the LLM backend, featuring custom agents, tasks, crews, and tools.

## Features

- **Ollama Integration**: Seamless integration with Ollama for local LLM inference
- **Custom Agents**: Pre-configured agents for various tasks
- **Task Templates**: Reusable task definitions
- **Crew Management**: Organized crew structures
- **Custom Tools**: Extensible tool collection for web search, file operations, code execution, and more
- **Self-Contained Deployment**: Automated deploy/rollback script with dependency management
- **Docker Support**: Containerized setup for easy deployment
- **Configuration Management**: Centralized settings management

## Quick Start

### Prerequisites

- Docker and Docker Compose
- Ollama installed and running locally

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd crewai-ollama-setup
```

2. Install dependencies and start the stack with the deployment script (runs in development mode by default):
```bash
python3 scripts/deploy.py deploy --env dev
```

   For production overrides:
```bash
python3 scripts/deploy.py deploy --env prod
```

3. Verify Ollama is running:
```bash
curl http://localhost:11434/api/tags
```

4. Check service status at any time:
```bash
python3 scripts/deploy.py status --env dev
```

5. Roll back all services and clean up (stops containers, removes volumes/networks/data, and deletes the managed virtualenv):
```bash
python3 scripts/deploy.py rollback
```

   Use `--help` on any command to see additional options (e.g., `python3 scripts/deploy.py deploy --help`).

### Deployment Script Highlights

- Creates required directories for bind mounts (`data/`, `logs/`, `models/`, etc.)
- Builds and starts Docker Compose services with optional `--prune` and `--force-recreate`
- Manages Python dependencies automatically (installs into `.venv` unless `--system-python` is specified)
- Provides `check` and `status` utilities for quick diagnostics
- Writes activity logs to `deploy.log` and removes runtime artifacts on rollback

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
