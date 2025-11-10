import os
import logging
from typing import List, Dict, Any, Optional, Literal
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
import uvicorn
from prometheus_client import Counter, Histogram, generate_latest
from prometheus_client.openmetrics.exposition import CONTENT_TYPE_LATEST
from fastapi.responses import PlainTextResponse

from crewai import Agent, Task, Crew
from langchain_community.llms import Ollama
import redis.asyncio as redis
import asyncio
from datetime import datetime
import json
from uuid import uuid4
import time

from config.settings import settings
from retraining.manager import DatasetManager, RetrainingJobManager

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
RETRAINING_JOBS_COUNTER = Counter('crewai_retraining_jobs_total', 'Total retraining jobs processed', ['status'])
RETRAINING_JOB_DURATION = Histogram('crewai_retraining_job_duration_seconds', 'Retraining job duration in seconds')

# Global variables
redis_client = None
ollama_llm: Optional[Ollama] = None
dataset_manager: Optional[DatasetManager] = None
retraining_manager: Optional[RetrainingJobManager] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan"""
    global redis_client, ollama_llm, dataset_manager, retraining_manager
    
    # Startup
    logger.info("Starting CrewAI service...")
    
    # Initialize Redis
    redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
    redis_client = redis.from_url(
        redis_url,
        max_connections=settings.redis_max_connections,
        encoding="utf-8",
        decode_responses=True
    )
    
    # Initialize Ollama
    ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    ollama_model = os.getenv("OLLAMA_MODEL", "llama2:7b")
    
    try:
        ollama_llm = Ollama(
            base_url=ollama_base_url,
            model=ollama_model
        )
        logger.info(f"Connected to Ollama at {ollama_base_url} with model {ollama_model}")
    except Exception as e:
        logger.error(f"Failed to connect to Ollama: {e}")
        ollama_llm = None

    # Initialize dataset and retraining managers
    dataset_root = Path(os.getenv("DATASET_DIR", "data/datasets"))
    retraining_root = Path(os.getenv("RETRAINING_DIR", "data/retraining"))
    dataset_manager = DatasetManager(dataset_root)
    retraining_manager = RetrainingJobManager(
        base_dir=retraining_root,
        dataset_manager=dataset_manager,
        redis_client=redis_client,
        result_ttl=settings.result_ttl,
        ollama_base_url=ollama_base_url,
        metrics={
            "counter": RETRAINING_JOBS_COUNTER,
            "duration": RETRAINING_JOB_DURATION
        },
    )
    
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

@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    """Track basic request metrics for Prometheus."""
    method = request.method
    endpoint = request.url.path
    start_time = time.perf_counter()
    
    try:
        response = await call_next(request)
        return response
    finally:
        duration = time.perf_counter() - start_time
        REQUEST_COUNT.labels(method=method, endpoint=endpoint).inc()
        REQUEST_DURATION.observe(duration)

# Security
security = HTTPBearer()

# Pydantic models
class AgentConfig(BaseModel):
    name: str = Field(..., description="Agent name")
    role: str = Field(..., description="Agent role")
    goal: str = Field(..., description="Agent goal")
    backstory: str = Field(..., description="Agent backstory")
    tools: Optional[List[str]] = Field(default=[], description="Agent tools")
    verbose: bool = Field(default=False, description="Enable verbose logging for the agent")

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

class DatasetCreateRequest(BaseModel):
    name: str = Field(..., description="Dataset name")
    description: Optional[str] = Field(default=None, description="Dataset description")
    tags: List[str] = Field(default_factory=list, description="Dataset tags")
    format: Literal["text", "jsonl"] = Field(default="text", description="Dataset storage format")
    content: str = Field(..., description="Dataset content")
    overwrite: bool = Field(default=False, description="Overwrite existing dataset if present")

class DatasetInfo(BaseModel):
    name: str
    display_name: str
    description: Optional[str]
    tags: List[str]
    format: str
    size_bytes: int
    created_at: datetime
    updated_at: datetime

class DatasetDetail(DatasetInfo):
    content: Optional[str] = None

class DatasetListResponse(BaseModel):
    datasets: List[DatasetInfo]

class RetrainingJobRequest(BaseModel):
    model_name: str = Field(..., description="Name for the new or updated model")
    base_model: str = Field(..., description="Base Ollama model to fine-tune")
    dataset_name: str = Field(..., description="Dataset identifier to use during retraining")
    instructions: Optional[str] = Field(default=None, description="Additional system instructions")
    modelfile_template: Optional[str] = Field(
        default=None,
        description="Custom Modelfile template; include {{DATASET}} to inline dataset content",
    )
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Extra PARAMETER entries for the Modelfile")
    stream: bool = Field(default=False, description="Stream Ollama create output")
    keep_alive: Optional[str] = Field(default=None, description="Ollama keep-alive duration for the new model")
    timeout: int = Field(default=1800, ge=60, le=7200, description="Maximum time allowed for retraining (seconds)")

class RetrainingJobStatus(BaseModel):
    job_id: str
    status: str
    model_name: str
    base_model: str
    dataset_name: str
    created_at: datetime
    updated_at: datetime
    message: Optional[str] = None
    parameters: Optional[Dict[str, Any]] = None
    stream: Optional[bool] = None

class RetrainingJobListResponse(BaseModel):
    jobs: List[RetrainingJobStatus]

class RetrainingJobLogsResponse(BaseModel):
    job_id: str
    logs: List[Dict[str, Any]]

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

@app.post("/datasets", response_model=DatasetInfo, status_code=201)
async def create_dataset(dataset: DatasetCreateRequest):
    """Create or replace a training dataset."""
    if not dataset_manager:
        raise HTTPException(status_code=503, detail="Dataset manager not initialized")
    
    try:
        record = await asyncio.to_thread(
            dataset_manager.save_dataset,
            dataset.name,
            dataset.content,
            dataset.description,
            dataset.tags,
            dataset.format,
            dataset.overwrite,
        )
        return DatasetInfo(**record)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        logger.error("Failed to save dataset: %s", exc)
        raise HTTPException(status_code=500, detail="Failed to save dataset")

@app.get("/datasets", response_model=DatasetListResponse)
async def list_datasets():
    """List available training datasets."""
    if not dataset_manager:
        raise HTTPException(status_code=503, detail="Dataset manager not initialized")
    
    records = await asyncio.to_thread(dataset_manager.list_datasets)
    datasets = [DatasetInfo(**record) for record in records]
    return DatasetListResponse(datasets=datasets)

@app.get("/datasets/{dataset_name}", response_model=DatasetDetail)
async def get_dataset(dataset_name: str, include_content: bool = False):
    """Retrieve dataset metadata (and optionally content)."""
    if not dataset_manager:
        raise HTTPException(status_code=503, detail="Dataset manager not initialized")
    
    try:
        record = await asyncio.to_thread(dataset_manager.get_dataset, dataset_name, include_content)
        return DatasetDetail(**record)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))

@app.delete("/datasets/{dataset_name}")
async def delete_dataset(dataset_name: str):
    """Delete a dataset."""
    if not dataset_manager:
        raise HTTPException(status_code=503, detail="Dataset manager not initialized")
    
    try:
        await asyncio.to_thread(dataset_manager.delete_dataset, dataset_name)
        return {"message": f"Dataset '{dataset_name}' deleted"}
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))

@app.post("/retraining/jobs", response_model=RetrainingJobStatus, status_code=202)
async def create_retraining_job(job_request: RetrainingJobRequest):
    """Schedule a new retraining job."""
    if not retraining_manager or not dataset_manager:
        raise HTTPException(status_code=503, detail="Retraining components not initialized")
    
    # Ensure dataset exists before scheduling the job
    try:
        await asyncio.to_thread(dataset_manager.get_dataset, job_request.dataset_name, False)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    
    payload = job_request.model_dump()
    try:
        job_record = await retraining_manager.create_job(payload)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        logger.error("Failed to create retraining job: %s", exc)
        raise HTTPException(status_code=500, detail="Failed to create retraining job")
    
    job_id = job_record["job_id"]
    asyncio.create_task(retraining_manager.run_job(job_id, payload, job_request.timeout))
    return RetrainingJobStatus(**job_record)

@app.get("/retraining/jobs", response_model=RetrainingJobListResponse)
async def list_retraining_jobs(limit: int = 50):
    """List retraining jobs."""
    if not retraining_manager:
        raise HTTPException(status_code=503, detail="Retraining components not initialized")
    
    records = await retraining_manager.list_jobs(limit)
    jobs = [RetrainingJobStatus(**record) for record in records]
    return RetrainingJobListResponse(jobs=jobs)

@app.get("/retraining/jobs/{job_id}", response_model=RetrainingJobStatus)
async def get_retraining_job(job_id: str):
    """Get retraining job status."""
    if not retraining_manager:
        raise HTTPException(status_code=503, detail="Retraining components not initialized")
    
    try:
        record = await retraining_manager.get_job(job_id)
        return RetrainingJobStatus(**record)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))

@app.get("/retraining/jobs/{job_id}/logs", response_model=RetrainingJobLogsResponse)
async def get_retraining_logs(job_id: str, tail: int = 100):
    """Retrieve retraining job logs."""
    if not retraining_manager:
        raise HTTPException(status_code=503, detail="Retraining components not initialized")
    
    logs = await retraining_manager.get_logs(job_id, tail)
    return RetrainingJobLogsResponse(job_id=job_id, logs=logs)

# Create agent endpoint
@app.post("/create_agent")
async def create_agent(agent_config: AgentConfig):
    """Create a new agent"""
    if not ollama_llm:
        raise HTTPException(status_code=503, detail="Ollama not connected")
    
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
                ex=settings.cache_ttl
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
        crew_id = f"crew_{uuid4().hex}"
        background_tasks.add_task(run_crew_async, crew, crew_config, crew_id)
    
        CREW_EXECUTIONS.inc()
        logger.info(f"Started crew execution with {len(agents)} agents and {len(tasks)} tasks")
        
        return {"message": "Crew execution started", "crew_id": crew_id}
    
    except Exception as e:
        logger.error(f"Error running crew: {e}")
        raise HTTPException(status_code=500, detail=str(e))

async def run_crew_async(crew: Crew, crew_config: CrewConfig, crew_id: str):
    """Run crew execution asynchronously"""
    try:
        result = crew.kickoff()
        logger.info(f"Crew execution completed: {result}")
        
        # Store result in Redis
        if redis_client:
            result_data = {
                "config": crew_config.model_dump(),
                "result": str(result),
                "completed_at": datetime.utcnow().isoformat()
            }
            await redis_client.set(
                f"crew_result:{crew_id}",
                json.dumps(result_data),
                ex=settings.result_ttl
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
        app,
        host="0.0.0.0",
        port=port,
        reload=os.getenv("RELOAD", "false").lower() == "true"
    )
