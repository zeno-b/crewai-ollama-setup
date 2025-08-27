#!/usr/bin/env python3
"""
CrewAI and Ollama Setup Script
Comprehensive setup with containerization, security, and performance optimization
"""

import os
import sys
import subprocess
import json
import logging
import platform
import shutil
from pathlib import Path
from typing import Dict, List, Optional
import argparse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('setup.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class CrewAISetup:
    """Main setup class for CrewAI and Ollama infrastructure"""
    
    def __init__(self, config_path: str = "config/setup_config.json"):
        self.config_path = Path(config_path)
        self.project_root = Path(__file__).parent
        self.config = self.load_config()
        
    def load_config(self) -> Dict:
        """Load configuration from JSON file"""
        try:
            if self.config_path.exists():
                with open(self.config_path, 'r') as f:
                    return json.load(f)
            else:
                logger.warning("Config file not found, using defaults")
                return self.get_default_config()
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            return self.get_default_config()
    
    def get_default_config(self) -> Dict:
        """Default configuration for setup"""
        return {
            "docker": {
                "compose_version": "3.8",
                "network_name": "crewai-network",
                "subnet": "172.20.0.0/16"
            },
            "ollama": {
                "image": "ollama/ollama:latest",
                "port": 11434,
                "memory_limit": "4g",
                "cpu_limit": "2.0"
            },
            "crewai": {
                "image": "crewai:latest",
                "port": 8000,
                "memory_limit": "2g",
                "cpu_limit": "1.0"
            },
            "security": {
                "enable_firewall": True,
                "enable_ssl": False,
                "user_id": 1000,
                "group_id": 1000
            },
            "performance": {
                "cache_size": "1g",
                "workers": 4,
                "timeout": 300
            }
        }
    
    def check_system_requirements(self) -> bool:
        """Check if system meets requirements"""
        logger.info("Checking system requirements...")
        
        # Check Python version
        if sys.version_info < (3, 8):
            logger.error("Python 3.8+ required")
            return False
            
        # Check Docker
        try:
            result = subprocess.run(['docker', '--version'], 
                                  capture_output=True, text=True)
            if result.returncode != 0:
                logger.error("Docker not found")
                return False
        except FileNotFoundError:
            logger.error("Docker not installed")
            return False
            
        # Check Docker Compose
        try:
            result = subprocess.run(['docker-compose', '--version'], 
                                  capture_output=True, text=True)
            if result.returncode != 0:
                logger.error("Docker Compose not found")
                return False
        except FileNotFoundError:
            logger.error("Docker Compose not installed")
            return False
            
        # Check available memory
        if platform.system() == "Windows":
            try:
                result = subprocess.run(['wmic', 'OS', 'get', 'TotalVisibleMemorySize'], 
                                      capture_output=True, text=True)
                memory_kb = int(result.stdout.split()[1])
                memory_gb = memory_kb / (1024 * 1024)
                if memory_gb < 8:
                    logger.warning(f"Low memory detected: {memory_gb:.1f}GB")
            except:
                pass
        else:
            try:
                result = subprocess.run(['free', '-g'], 
                                      capture_output=True, text=True)
                memory_gb = int(result.stdout.split()[7])
                if memory_gb < 8:
                    logger.warning(f"Low memory detected: {memory_gb}GB")
            except:
                pass
                
        logger.info("System requirements check completed")
        return True
    
    def create_directory_structure(self):
        """Create necessary directory structure"""
        directories = [
            "config",
            "data",
            "logs",
            "models",
            "agents",
            "scripts",
            "backups"
        ]
        
        for directory in directories:
            dir_path = self.project_root / directory
            dir_path.mkdir(exist_ok=True)
            logger.info(f"Created directory: {directory}")
    
    def create_docker_compose(self):
        """Create Docker Compose configuration"""
        compose_config = f"""
version: '{self.config["docker"]["compose_version"]}'

networks:
  {self.config["docker"]["network_name"]}:
    driver: bridge
    ipam:
      config:
        - subnet: {self.config["docker"]["subnet"]}

services:
  ollama:
    image: {self.config["ollama"]["image"]}
    container_name: ollama-service
    ports:
      - "{self.config['ollama']['port']}:11434"
    volumes:
      - ./data/ollama:/root/.ollama
      - ./models:/models
    environment:
      - OLLAMA_HOST=0.0.0.0
    networks:
      - {self.config["docker"]["network_name"]}
    deploy:
      resources:
        limits:
          memory: {self.config["ollama"]["memory_limit"]}
          cpus: {self.config["ollama"]["cpu_limit"]}
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:11434/api/tags"]
      interval: 30s
      timeout: 10s
      retries: 3

  crewai:
    build:
      context: .
      dockerfile: Dockerfile.crewai
    container_name: crewai-service
    ports:
      - "{self.config['crewai']['port']}:8000"
    volumes:
      - ./agents:/app/agents
      - ./config:/app/config
      - ./logs:/app/logs
    environment:
      - OLLAMA_BASE_URL=http://ollama:11434
    networks:
      - {self.config["docker"]["network_name"]}
    depends_on:
      ollama:
        condition: service_healthy
    deploy:
      resources:
        limits:
          memory: {self.config["crewai"]["memory_limit"]}
          cpus: {self.config["crewai"]["cpu_limit"]}
    restart: unless-stopped
    user: "{self.config['security']['user_id']}:{self.config['security']['group_id']}"
"""
        
        with open(self.project_root / "docker-compose.yml", "w") as f:
            f.write(compose_config)
        logger.info("Docker Compose configuration created")
    
    def create_dockerfile_crewai(self):
        """Create Dockerfile for CrewAI service"""
        dockerfile_content = f"""
FROM python:3.11-slim

# Security: Create non-root user
RUN groupadd -g {self.config['security']['group_id']} crewai && \\
    useradd -u {self.config['security']['user_id']} -g crewai -m crewai

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    curl \\
    git \\
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p /app/logs /app/agents && \\
    chown -R crewai:crewai /app

# Switch to non-root user
USER crewai

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \\
    CMD curl -f http://localhost:8000/health || exit 1

# Start command
CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
"""
        
        with open(self.project_root / "Dockerfile.crewai", "w") as f:
            f.write(dockerfile_content)
        logger.info("CrewAI Dockerfile created")
    
    def create_requirements(self):
        """Create requirements.txt for CrewAI"""
        requirements = [
            "crewai>=0.30.0",
            "crewai-tools>=0.4.0",
            "langchain>=0.1.0",
            "langchain-community>=0.0.10",
            "fastapi>=0.104.0",
            "uvicorn[standard]>=0.24.0",
            "pydantic>=2.5.0",
            "python-dotenv>=1.0.0",
            "requests>=2.31.0",
            "aiohttp>=3.9.0",
            "asyncio>=3.4.3",
            "psutil>=5.9.0",
            "cryptography>=41.0.0",
            "redis>=5.0.0",
            "sqlalchemy>=2.0.0",
            "alembic>=1.12.0"
        ]
        
        with open(self.project_root / "requirements.txt", "w") as f:
            f.write("\n".join(requirements))
        logger.info("Requirements file created")
    
    def create_environment_template(self):
        """Create environment template file"""
        env_template = """# Ollama Configuration
OLLAMA_BASE_URL=http://ollama:11434
OLLAMA_MODEL=llama2:7b

# CrewAI Configuration
CREWAI_PORT=8000
CREWAI_LOG_LEVEL=INFO

# Security
SECRET_KEY=your-secret-key-here
JWT_SECRET=your-jwt-secret-here

# Database
DATABASE_URL=sqlite:///./data/crewai.db
REDIS_URL=redis://localhost:6379

# Monitoring
ENABLE_METRICS=true
METRICS_PORT=9090
"""
        
        with open(self.project_root / ".env.template", "w") as f:
            f.write(env_template)
        logger.info("Environment template created")
    
    def create_main_app(self):
        """Create main FastAPI application"""
        main_app = '''from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import logging
import os
from dotenv import load_dotenv
from crewai import Agent, Task, Crew
from pydantic import BaseModel
from typing import List, Dict, Any
import uvicorn

load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/crewai.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="CrewAI Service",
    description="CrewAI and Ollama Integration Service",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

security = HTTPBearer()

class AgentConfig(BaseModel):
    name: str
    role: str
    goal: str
    backstory: str
    tools: List[str] = []

class TaskConfig(BaseModel):
    description: str
    expected_output: str
    agent: str

class CrewConfig(BaseModel):
    agents: List[AgentConfig]
    tasks: List[TaskConfig]

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "crewai"}

@app.post("/create_agent")
async def create_agent(agent_config: AgentConfig):
    """Create a new agent"""
    try:
        agent = Agent(
            name=agent_config.name,
            role=agent_config.role,
            goal=agent_config.goal,
            backstory=agent_config.backstory,
            verbose=True
        )
        return {"status": "success", "agent": agent_config.name}
    except Exception as e:
        logger.error(f"Error creating agent: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/run_crew")
async def run_crew(crew_config: CrewConfig):
    """Run a crew with given agents and tasks"""
    try:
        agents = {}
        for agent_config in crew_config.agents:
            agents[agent_config.name] = Agent(
                name=agent_config.name,
                role=agent_config.role,
                goal=agent_config.goal,
                backstory=agent_config.backstory,
                verbose=True
            )
        
        tasks = []
        for task_config in crew_config.tasks:
            agent = agents.get(task_config.agent)
            if not agent:
                raise HTTPException(status_code=400, detail=f"Agent {task_config.agent} not found")
            
            task = Task(
                description=task_config.description,
                expected_output=task_config.expected_output,
                agent=agent
            )
            tasks.append(task)
        
        crew = Crew(
            agents=list(agents.values()),
            tasks=tasks,
            verbose=True
        )
        
        result = crew.kickoff()
        return {"status": "success", "result": str(result)}
        
    except Exception as e:
        logger.error(f"Error running crew: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/models")
async def list_models():
    """List available Ollama models"""
    import requests
    try:
        response = requests.get(f"{os.getenv('OLLAMA_BASE_URL', 'http://ollama:11434')}/api/tags")
        return response.json()
    except Exception as e:
        logger.error(f"Error listing models: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
'''
        
        with open(self.project_root / "main.py", "w") as f:
            f.write(main_app)
        logger.info("Main application created")
    
    def run_setup(self):
        """Execute the complete setup process"""
        logger.info("Starting CrewAI and Ollama setup...")
        
        if not self.check_system_requirements():
            logger.error("System requirements not met")
            return False
            
        try:
            self.create_directory_structure()
            self.create_docker_compose()
            self.create_dockerfile_crewai()
            self.create_requirements()
            self.create_environment_template()
            self.create_main_app()
            
            # Save configuration
            with open(self.config_path, 'w') as f:
                json.dump(self.config, f, indent=2)
            
            logger.info("Setup completed successfully!")
            logger.info("Next steps:")
            logger.info("1. Copy .env.template to .env and configure")
            logger.info("2. Run: docker-compose up -d")
            logger.info("3. Access services:")
            logger.info("   - Ollama: http://localhost:11434")
            logger.info("   - CrewAI: http://localhost:8000")
            
            return True
            
        except Exception as e:
            logger.error(f"Setup failed: {e}")
            return False

def main():
    parser = argparse.ArgumentParser(description="CrewAI and Ollama Setup")
    parser.add_argument("--config", help="Configuration file path")
    args = parser.parse_args()
    
    setup = CrewAISetup(args.config) if args.config else CrewAISetup()
    success = setup.run_setup()
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
