import os
import logging
from typing import List, Dict, Any, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
import uvicorn
from prometheus_client import Counter, Histogram, generate_latest
from prometheus_client.openmetrics.exposition import CONTENT_TYPE_LATEST
from fastapi.responses import PlainTextResponse

from crewai import Agent, Task, Crew
from langchain_ollama import OllamaLLM
import redis.asyncio as redis
import asyncio
from datetime import datetime
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Metrics
REQUEST_COUNT = Counter('crewai_requests_total', 'Total requests', ['method', 'endpoint'])
REQUEST_DURATION = Histogram('crewai_request_duration_seconds', 'Request duration')
AGENT_CREATIONS = Counter('crewai_agent_creations_total', 'Total agent creations')
CREW_EXECUTIONS = Counter('crewai_crew_executions_total', 'Total crew executions')

# Global variables
redis_client = None
ollama_llm = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan"""
    global redis_client, ollama_llm
    
    # Startup
    logger.info("Starting CrewAI service...")
    
    # Initialize Redis
    redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
    redis_client = redis.from_url(redis_url)
    
    # Initialize Ollama
    ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    ollama_model = os.getenv("OLLAMA_MODEL", "llama2:7b")
    
    try:
        ollama_llm = OllamaLLM(
            base_url=ollama_base_url,
            model=ollama_model
        )
        logger.info(f"Connected to Ollama at {ollama_base_url} with model {ollama_model}")
    except Exception as e:
        logger.error(f"Failed to connect to Ollama: {e}")
        ollama_llm = None
    
    yield
    
    # Shutdown
    logger.info("Shutting down CrewAI service...")
    if redis_client:
        await redis_client.close()

# Initialize FastAPI app
app = FastAPI(
    title="CrewAI Service",
    description="Production-ready CrewAI and Ollama integration",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security
security = HTTPBearer()

# Pydantic models
class AgentConfig(BaseModel):
    name: str = Field(..., description="Agent name")
    role: str = Field(..., description="Agent role")
    goal: str = Field(..., description="Agent goal")
    backstory: str = Field(..., description="Agent backstory")
    tools: Optional[List[str]] = Field(default=[], description="Agent tools")

class TaskConfig(BaseModel):
    description: str = Field(..., description="Task description")
    expected_output: str = Field(..., description="Expected output")
    agent: str = Field(..., description="Assigned agent name")

class CrewConfig(BaseModel):
    agents: List[AgentConfig]
    tasks: List[TaskConfig]
    verbose: bool = Field(default=False, description="Enable verbose logging")

class HealthResponse(BaseModel):
    status: str
    timestamp: datetime
    ollama_connected: bool
    redis_connected: bool
    version: str

class ModelInfo(BaseModel):
    name: str
    size: int
    digest: str
    modified_at: str

class ModelsResponse(BaseModel):
    models: List[ModelInfo]

# Dependency to get current user (placeholder)
async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    # Implement your authentication logic here
    return {"user_id": "default_user"}

# Health check endpoint
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    ollama_status = ollama_llm is not None
    
    redis_status = False
    if redis_client:
        try:
            await redis_client.ping()
            redis_status = True
        except:
            redis_status = False
    
    return HealthResponse(
        status="healthy" if ollama_status and redis_status else "degraded",
        timestamp=datetime.utcnow(),
        ollama_connected=ollama_status,
        redis_connected=redis_status,
        version="1.0.0"
    )

# Metrics endpoint
@app.get("/metrics", response_class=PlainTextResponse)
async def metrics():
    """Prometheus metrics endpoint"""
    return PlainTextResponse(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST
    )

# List available models
@app.get("/models", response_model=ModelsResponse)
async def list_models():
    """List available Ollama models"""
    if not ollama_llm:
        raise HTTPException(status_code=503, detail="Ollama not connected")
    
    try:
        # This is a placeholder - implement actual Ollama model listing
        models = [
            ModelInfo(
                name="llama2:7b",
                size=3825819519,
                digest="fe938a131f40e6f6d40083c9f0f430a515233eb2edaa6d72eb85c50d64f2300e",
                modified_at="2024-01-01T00:00:00Z"
            )
        ]
        return ModelsResponse(models=models)
    except Exception as e:
        logger.error(f"Error listing models: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Create agent endpoint
@app.post("/create_agent")
async def create_agent(agent_config: AgentConfig):
    """Create a new agent"""
    REQUEST_COUNT.labels(method="POST", endpoint="/create_agent").inc()
    
    try:
        agent = Agent(
            role=agent_config.role,
            goal=agent_config.goal,
            backstory=agent_config.backstory,
            llm=ollama_llm,
            verbose=agent_config.verbose
        )
        
        # Store agent in Redis for persistence
        agent_data = {
            "name": agent_config.name,
            "role": agent_config.role,
            "goal": agent_config.goal,
            "backstory": agent_config.backstory,
            "created_at": datetime.utcnow().isoformat()
        }
        
        if redis_client:
            await redis_client.set(
                f"agent:{agent_config.name}",
                json.dumps(agent_data),
                ex=3600  # 1 hour TTL
            )
        
        AGENT_CREATIONS.inc()
        logger.info(f"Created agent: {agent_config.name}")
        
        return {"message": f"Agent {agent_config.name} created successfully"}
        
    except Exception as e:
        logger.error(f"Error creating agent: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Run crew endpoint
@app.post("/run_crew")
async def run_crew(crew_config: CrewConfig, background_tasks: BackgroundTasks):
    """Run a crew with specified agents and tasks"""
    REQUEST_COUNT.labels(method="POST", endpoint="/run_crew").inc()
    
    if not ollama_llm:
        raise HTTPException(status_code=503, detail="Ollama not connected")
    
    try:
        # Create agents
        agents = {}
        for agent_config in crew_config.agents:
            agent = Agent(
                role=agent_config.role,
                goal=agent_config.goal,
                backstory=agent_config.backstory,
                llm=ollama_llm,
                verbose=crew_config.verbose
            )
            agents[agent_config.name] = agent
        
        # Create tasks
        tasks = []
        for task_config in crew_config.tasks:
            if task_config.agent not in agents:
                raise HTTPException(
                    status_code=400,
                    detail=f"Agent {task_config.agent} not found"
                )
            
            task = Task(
                description=task_config.description,
                expected_output=task_config.expected_output,
                agent=agents[task_config.agent]
            )
            tasks.append(task)
        
        # Create and run crew
        crew = Crew(
            agents=list(agents.values()),
            tasks=tasks,
            verbose=crew_config.verbose
        )
        
        # Run crew in background
        background_tasks.add_task(run_crew_async, crew, crew_config)
        
        CREW_EXECUTIONS.inc()
        logger.info(f"Started crew execution with {len(agents)} agents and {len(tasks)} tasks")
        
        return {"message": "Crew execution started", "crew_id": f"crew_{datetime.utcnow().timestamp()}"}
        
    except Exception as e:
        logger.error(f"Error running crew: {e}")
        raise HTTPException(status_code=500, detail=str(e))

async def run_crew_async(crew: Crew, crew_config: CrewConfig):
    """Run crew execution asynchronously"""
    try:
        result = crew.kickoff()
        logger.info(f"Crew execution completed: {result}")
        
        # Store result in Redis
        if redis_client:
            result_data = {
                "config": crew_config.dict(),
                "result": str(result),
                "completed_at": datetime.utcnow().isoformat()
            }
            await redis_client.set(
                f"crew_result:{datetime.utcnow().timestamp()}",
                json.dumps(result_data),
                ex=86400  # 24 hours TTL
            )
            
    except Exception as e:
        logger.error(f"Error in crew execution: {e}")

# Get crew results
@app.get("/crew_results/{crew_id}")
async def get_crew_result(crew_id: str):
    """Get crew execution results"""
    if not redis_client:
        raise HTTPException(status_code=503, detail="Redis not available")
    
    try:
        result = await redis_client.get(f"crew_result:{crew_id}")
        if not result:
            raise HTTPException(status_code=404, detail="Crew result not found")
        
        return json.loads(result)
        
    except Exception as e:
        logger.error(f"Error retrieving crew result: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Root endpoint
@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "CrewAI Service is running",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
        "metrics": "/metrics"
    }

if __name__ == "__main__":
    port = int(os.getenv("CREWAI_PORT", 8000))
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        reload=os.getenv("RELOAD", "false").lower() == "true"
    )
