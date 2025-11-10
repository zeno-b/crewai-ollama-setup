#!/usr/bin/env python3  # Specify the interpreter used to execute this script

import argparse  # Import argparse to parse command line arguments
import json  # Import json to serialize and persist configuration details
import logging  # Import logging to emit structured diagnostic information
import os  # Import os to interact with environment variables and the filesystem
from pathlib import Path  # Import Path to handle filesystem paths elegantly
from typing import Dict, List  # Import typing helpers to annotate configuration structures

logging.basicConfig(  # Configure the global logging system used by the script
    level=logging.INFO,  # Emit informational messages by default
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"  # Use a timestamped log format
)  # Close the logging configuration call
logger = logging.getLogger(__name__)  # Acquire a module-level logger for this script


DEFAULT_DIRECTORIES: List[str] = [  # Declare the directories that should exist for a healthy project layout
    "agents",  # Store application-specific agent definitions
    "config",  # Store configuration modules and artifacts
    "data",  # Store persistent data files
    "logs",  # Store application log files
    "models",  # Store model artifacts such as Ollama downloads
    "tasks",  # Store task definitions
    "tools"  # Store custom tool implementations
]  # Close the list of default directories

REQUIREMENTS: List[str] = [  # Declare the Python dependencies required by the project
    "crewai>=0.30.0",  # Provide the core CrewAI framework
    "crewai-tools>=0.4.0",  # Provide additional CrewAI tool integrations
    "fastapi>=0.109.0",  # Provide the FastAPI web framework
    "uvicorn[standard]>=0.27.0",  # Provide the Uvicorn ASGI server with standard extras
    "pydantic>=2.5.0",  # Provide Pydantic for data validation
    "redis>=5.0.0",  # Provide the Redis client for caching and persistence
    "requests>=2.31.0",  # Provide an HTTP client for integrations
    "prometheus-client>=0.19.0"  # Provide Prometheus instrumentation
]  # Close the requirements list


class CrewAISetup:  # Define a helper class that encapsulates setup operations
    def __init__(self, project_root: Path, config_path: Path) -> None:  # Initialize the setup helper with paths
        self.project_root = project_root  # Store the project root directory for later operations
        self.config_path = config_path  # Store the configuration file path for persistence

    def ensure_directories(self) -> None:  # Create required directories if they do not exist
        for directory in DEFAULT_DIRECTORIES:  # Iterate over each required directory name
            target = self.project_root / directory  # Construct the absolute path for the directory
            target.mkdir(parents=True, exist_ok=True)  # Create the directory hierarchy when missing
            logger.info("Ensured directory exists: %s", target)  # Log the directory creation or verification

    def write_requirements(self) -> None:  # Persist the Python dependency list to requirements.txt
        requirements_path = self.project_root / "requirements.txt"  # Determine the path for the requirements file
        requirements_path.write_text("\n".join(REQUIREMENTS), encoding="utf-8")  # Write the dependencies as newline-separated text
        logger.info("Wrote dependency list to %s", requirements_path)  # Log that the requirements file was refreshed

    def write_env_template(self) -> None:  # Persist an environment variable template for operators
        env_template_path = self.project_root / ".env.template"  # Determine the path for the template file
        template_contents = (  # Build the template contents as a single string
            "# CrewAI Environment Template\n"
            "API_BEARER_TOKEN=replace-with-secure-token\n"
            "REDIS_URL=redis://localhost:6379/0\n"
            "OLLAMA_BASE_URL=http://localhost:11434\n"
            "OLLAMA_MODEL=llama2:7b\n"
            "RATE_LIMIT_PER_MINUTE=60\n"
        )  # Close the template string
        env_template_path.write_text(template_contents, encoding="utf-8")  # Write the template to disk
        logger.info("Wrote environment template to %s", env_template_path)  # Log that the template was updated

    def write_docker_compose(self) -> None:  # Persist a minimal Docker Compose configuration for local deployment
        docker_compose_path = self.project_root / "docker-compose.yml"  # Determine the path for the compose file
        compose_contents = (  # Build the docker-compose configuration as a string
            "version: '3.9'\n"
            "services:\n"
            "  redis:\n"
            "    image: redis:7-alpine\n"
            "    restart: unless-stopped\n"
            "    ports:\n"
            "      - \"6379:6379\"\n"
            "  crewai:\n"
            "    build:\n"
            "      context: .\n"
            "      dockerfile: Dockerfile.crewai\n"
            "    environment:\n"
            "      - REDIS_URL=redis://redis:6379/0\n"
            "      - OLLAMA_BASE_URL=http://ollama:11434\n"
            "    ports:\n"
            "      - \"8000:8000\"\n"
            "    depends_on:\n"
            "      - redis\n"
            "  ollama:\n"
            "    image: ollama/ollama:latest\n"
            "    ports:\n"
            "      - \"11434:11434\"\n"
        )  # Close the docker-compose configuration
        docker_compose_path.write_text(compose_contents, encoding="utf-8")  # Write the compose file to disk
        logger.info("Wrote docker-compose definition to %s", docker_compose_path)  # Log that the compose file was updated

    def write_dockerfile(self) -> None:  # Persist a Dockerfile tailored for the CrewAI service
        dockerfile_path = self.project_root / "Dockerfile.crewai"  # Determine the path for the Dockerfile
        dockerfile_contents = (  # Build the Dockerfile as a string
            "FROM python:3.11-slim\n"
            "WORKDIR /app\n"
            "COPY requirements.txt .\n"
            "RUN pip install --no-cache-dir -r requirements.txt\n"
            "COPY . .\n"
            "EXPOSE 8000\n"
            "CMD [\"python\", \"main.py\"]\n"
        )  # Close the Dockerfile string
        dockerfile_path.write_text(dockerfile_contents, encoding="utf-8")  # Write the Dockerfile to disk
        logger.info("Wrote Dockerfile to %s", dockerfile_path)  # Log that the Dockerfile was refreshed

    def persist_config(self, config: Dict[str, str]) -> None:  # Persist setup metadata for future reference
        self.config_path.parent.mkdir(parents=True, exist_ok=True)  # Ensure the configuration directory exists
        self.config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")  # Serialize the configuration to JSON
        logger.info("Persisted setup metadata to %s", self.config_path)  # Log that the configuration file was written

    def run(self) -> None:  # Execute the full setup routine using helper methods
        logger.info("Starting CrewAI project initialization")  # Log the beginning of the setup process
        self.ensure_directories()  # Create required project directories
        self.write_requirements()  # Generate the requirements file
        self.write_env_template()  # Generate the environment template
        self.write_docker_compose()  # Generate the docker-compose file
        self.write_dockerfile()  # Generate the Dockerfile
        setup_metadata = {  # Prepare metadata describing the generated artifacts
            "requirements": "requirements.txt",  # Record the location of the requirements file
            "env_template": ".env.template",  # Record the location of the environment template
            "docker_compose": "docker-compose.yml",  # Record the location of the docker-compose file
            "dockerfile": "Dockerfile.crewai"  # Record the location of the Dockerfile
        }  # Close the metadata dictionary
        self.persist_config(setup_metadata)  # Persist the metadata for future reference
        logger.info("CrewAI project initialization completed successfully")  # Log the successful completion of setup


def parse_arguments() -> argparse.Namespace:  # Parse command line arguments provided to the script
    parser = argparse.ArgumentParser(description="Initialize CrewAI project scaffolding")  # Create an argument parser with descriptive help text
    parser.add_argument("--config", default="config/setup.json", help="Path to persist setup metadata")  # Allow overriding the config output path
    parser.add_argument("--root", default=".", help="Project root directory")  # Allow overriding the project root directory
    return parser.parse_args()  # Parse and return the supplied arguments


def main() -> None:  # Entrypoint that orchestrates argument parsing and setup execution
    arguments = parse_arguments()  # Parse command line arguments
    project_root = Path(arguments.root).resolve()  # Resolve the project root into an absolute path
    config_path = project_root / arguments.config  # Determine the absolute path for the configuration file
    setup = CrewAISetup(project_root=project_root, config_path=config_path)  # Instantiate the setup helper with resolved paths
    setup.run()  # Execute the setup routine


if __name__ == "__main__":  # Execute the script only when run directly
    main()  # Invoke the main entrypoint to perform setup
