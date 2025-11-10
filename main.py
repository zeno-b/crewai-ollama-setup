import asyncio  # Import asyncio to run blocking Crew executions without blocking the event loop
import json  # Import json to serialize and deserialize data stored in Redis
import logging  # Import logging to provide structured application logs
from contextlib import asynccontextmanager  # Import asynccontextmanager to manage FastAPI lifespan events
from datetime import datetime  # Import datetime to timestamp stored records and responses
from typing import Any, Dict, List, Optional  # Import typing helpers for type annotations

import httpx  # Import httpx for robust asynchronous HTTP requests to Ollama
import redis.asyncio as redis  # Import redis asyncio client for non-blocking Redis operations
import uvicorn  # Import uvicorn to run the FastAPI application when executed directly
from crewai import Agent, Crew, Task  # Import CrewAI classes to build AI workflows dynamically
from fastapi import Depends, FastAPI, HTTPException, status  # Import FastAPI primitives for API definition
from fastapi.middleware.cors import CORSMiddleware  # Import CORS middleware to control cross-origin requests
from fastapi.responses import PlainTextResponse  # Import response type for Prometheus metrics exposure
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer  # Import HTTP bearer auth utilities
from langchain_community.llms import Ollama  # Import Ollama wrapper to interact with the Ollama LLM runtime
from pydantic import BaseModel, Field  # Import BaseModel and Field to validate request payloads
from prometheus_client import Counter, Histogram, generate_latest  # Import Prometheus instrumentation helpers
from prometheus_client.openmetrics.exposition import CONTENT_TYPE_LATEST  # Import Prometheus content type constant

from config.settings import settings  # Import application settings to centralize configuration management


logging.basicConfig(level=getattr(logging, settings.log_level.upper(), logging.INFO), format=settings.log_format)  # Configure logging with severity and format derived from settings
logger = logging.getLogger("crewai_service")  # Acquire a module-specific logger for contextual messages


REQUEST_COUNT = Counter("crewai_requests_total", "Total HTTP requests received by method and endpoint", ["method", "endpoint"])  # Track the total request volume per route for observability
REQUEST_DURATION = Histogram("crewai_request_duration_seconds", "Duration of HTTP requests in seconds", ["endpoint"])  # Track request latencies to monitor performance over time
AGENT_CREATIONS = Counter("crewai_agent_creations_total", "Total number of agents successfully registered")  # Track agent creation events for capacity monitoring
CREW_EXECUTIONS = Counter("crewai_crew_executions_total", "Total number of crew executions kicked off")  # Track crew executions to monitor usage patterns


redis_client: Optional[redis.Redis] = None  # Maintain a global Redis client reference for reuse across requests
ollama_llm: Optional[Ollama] = None  # Maintain a global Ollama client reference for shared use


security = HTTPBearer(auto_error=False)  # Configure bearer token extraction without automatic exception handling


class AgentConfig(BaseModel):  # Define the payload schema for registering agents
    name: str = Field(..., min_length=1, max_length=64, description="Unique agent identifier")  # Require an agent name to uniquely track the agent configuration
    role: str = Field(..., min_length=1, description="Agent role description")  # Require a role definition to guide the agent behavior
    goal: str = Field(..., min_length=1, description="Agent goal narrative")  # Require a goal to instruct the agent about desired outcome
    backstory: str = Field(..., min_length=1, description="Agent backstory context")  # Require backstory to provide additional context to the agent
    tools: List[str] = Field(default_factory=list, description="Optional list of tool identifiers")  # Allow clients to request tool attachments by identifier
    verbose: bool = Field(default=False, description="Enable verbose logging for the agent")  # Allow clients to opt-in to verbose agent logging
    allow_delegation: bool = Field(default=False, description="Permit the agent to delegate tasks")  # Allow clients to control delegation capabilities
    max_iter: int = Field(default=25, ge=1, le=200, description="Maximum iterations per agent run")  # Limit iterations to avoid runaway loops
    max_rpm: Optional[int] = Field(default=None, ge=1, description="Maximum requests per minute throttle")  # Optionally constrain agent request throughput
    cache: bool = Field(default=True, description="Enable or disable internal caching")  # Allow clients to toggle agent caching behavior


class TaskConfig(BaseModel):  # Define the payload schema for registering tasks
    name: str = Field(default="task", min_length=1, description="Human-readable task name")  # Allow clients to tag tasks with meaningful identifiers
    description: str = Field(..., min_length=1, description="Task description provided to the agent")  # Require a task prompt to drive agent execution
    expected_output: str = Field(..., min_length=1, description="Expected task output description")  # Require an expected output to guide agent evaluation
    agent: str = Field(..., min_length=1, description="Agent name responsible for the task")  # Require the associated agent name to resolve assignment


class CrewConfig(BaseModel):  # Define the payload schema for launching crew executions
    agents: List[AgentConfig] = Field(..., min_items=1, description="List of agent configurations")  # Require at least one agent for a crew to operate
    tasks: List[TaskConfig] = Field(..., min_items=1, description="List of task configurations")  # Require at least one task for the crew to execute
    verbose: bool = Field(default=False, description="Enable verbose logging during crew execution")  # Allow clients to toggle verbose crew logging


class HealthResponse(BaseModel):  # Define the response schema for health checks
    status: str  # Report overall service health classification
    timestamp: datetime  # Report the time the health status was generated
    ollama_connected: bool  # Indicate whether the Ollama client is available
    redis_connected: bool  # Indicate whether the Redis client is available
    version: str  # Report the running API version


class ModelInfo(BaseModel):  # Define the schema for individual Ollama model metadata
    name: str  # Store the Ollama model identifier
    size: int  # Store the model size in bytes
    digest: str  # Store the model digest hash
    modified_at: str  # Store the last modification timestamp


class ModelsResponse(BaseModel):  # Define the schema for the models listing response
    models: List[ModelInfo]  # Provide a list of Ollama model metadata instances


async def initialize_redis() -> Optional[redis.Redis]:  # Establish a Redis connection pool using configured settings
    try:
        client = redis.from_url(settings.redis_url, password=settings.redis_password, max_connections=settings.redis_max_connections, decode_responses=True)  # Build a Redis client from the configured URL and options
        await client.ping()  # Validate the connection eagerly to fail fast if Redis is unreachable
        logger.info("Connected to Redis successfully")  # Log a positive connection status for observability
        return client  # Return the connected Redis client for reuse
    except Exception as exc:
        logger.error("Failed to connect to Redis: %s", exc, exc_info=True)  # Log the connection failure with stack context for debugging
        return None  # Return None to indicate Redis is unavailable


async def initialize_ollama() -> Optional[Ollama]:  # Initialize the Ollama client and verify connectivity
    try:
        client = Ollama(base_url=settings.ollama_base_url, model=settings.ollama_model)  # Instantiate the Ollama client with configured endpoint and model
        async with httpx.AsyncClient(timeout=settings.request_timeout_seconds) as http_client:  # Open an HTTP client with a bounded timeout for verification
            response = await http_client.get(f"{settings.ollama_base_url}/api/tags")  # Request available models from Ollama to confirm connectivity
            response.raise_for_status()  # Raise an exception if the Ollama service returns an error status
        logger.info("Connected to Ollama at %s", settings.ollama_base_url)  # Log successful Ollama connectivity
        return client  # Return the configured Ollama client ready for use
    except Exception as exc:
        logger.error("Failed to connect to Ollama: %s", exc, exc_info=True)  # Log the Ollama connectivity failure with stack context
        return None  # Return None to indicate Ollama is currently unavailable


async def close_redis(client: Optional[redis.Redis]) -> None:  # Gracefully close the Redis connection pool during shutdown
    if client is None:  # Skip cleanup if Redis was never initialized
        return  # Exit early to avoid attribute errors
    await client.close()  # Close the Redis connection pool to release resources
    await client.connection_pool.disconnect()  # Ensure the underlying pool disconnects all connections cleanly
    logger.info("Redis connection closed")  # Log completion of Redis cleanup


@asynccontextmanager  # Decorate the lifespan function to run during FastAPI startup and shutdown
async def lifespan(app: FastAPI):  # Define the application lifespan hook
    global redis_client  # Reference the module-level Redis client inside the function
    global ollama_llm  # Reference the module-level Ollama client inside the function
    redis_client = await initialize_redis()  # Initialize Redis at startup for reuse during the application lifetime
    ollama_llm = await initialize_ollama()  # Initialize Ollama at startup to avoid per-request setup cost
    if ollama_llm is None:  # Check if Ollama failed to initialize
        logger.warning("Ollama is unavailable; dependent endpoints will return 503")  # Warn operators about degraded functionality
    try:
        yield  # Yield control back to FastAPI while keeping resources alive
    finally:
        await close_redis(redis_client)  # Ensure Redis connections are cleaned up on shutdown
        redis_client = None  # Reset the global Redis reference after cleanup
        ollama_llm = None  # Reset the global Ollama reference after cleanup


app = FastAPI(title=settings.api_title, description="CrewAI and Ollama orchestration service", version=settings.api_version, lifespan=lifespan, debug=settings.api_debug)  # Instantiate the FastAPI application with metadata and lifespan management


allowed_origins = [origin.strip() for origin in settings.allowed_origins.split(",") if origin.strip()]  # Parse the configured CORS origins into a clean list
app.add_middleware(CORSMiddleware, allow_origins=allowed_origins or ["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])  # Register CORS middleware with configured origins


async def get_current_user(credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)) -> Dict[str, Any]:  # Validate incoming requests against the configured bearer token
    if settings.security_token == "change-me":  # Detect insecure default token usage
        logger.warning("API_BEARER_TOKEN is set to the insecure default value")  # Warn operators about insecure configuration
    if credentials is None:  # Reject requests missing bearer credentials
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing authorization token")  # Inform the client about the missing token
    if credentials.scheme.lower() != "bearer":  # Enforce the expected scheme
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid authorization scheme")  # Inform the client about incorrect scheme usage
    if credentials.credentials != settings.security_token:  # Validate the provided token against configuration
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Invalid authorization token")  # Reject invalid tokens to protect the API
    return {"token": credentials.credentials}  # Return a minimal authenticated context for downstream handlers


def ensure_redis_available() -> redis.Redis:  # Helper to ensure Redis is ready before performing operations
    if redis_client is None:  # Detect missing Redis availability
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Redis is not available")  # Provide a clear service unavailable response
    return redis_client  # Return the initialized Redis client


def ensure_ollama_available() -> Ollama:  # Helper to ensure Ollama is ready before performing operations
    if ollama_llm is None:  # Detect missing Ollama availability
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Ollama is not available")  # Provide a clear service unavailable response
    return ollama_llm  # Return the initialized Ollama client


@app.get("/health", response_model=HealthResponse)  # Expose a health check endpoint for monitoring systems
async def health_check() -> HealthResponse:  # Define the health check handler
    redis_ok = False  # Initialize Redis status flag
    if redis_client is not None:  # Only attempt a ping when Redis is initialized
        try:
            await redis_client.ping()  # Perform a ping to verify connectivity
            redis_ok = True  # Mark Redis as healthy when the ping succeeds
        except Exception as exc:  # Catch any errors to avoid raising from health check
            logger.warning("Redis ping failed during health check: %s", exc)  # Log the degraded status for diagnostics
    ollama_ok = ollama_llm is not None  # Check whether Ollama initialized successfully
    overall_status = "healthy" if redis_ok and ollama_ok else "degraded"  # Determine the aggregate health classification
    return HealthResponse(status=overall_status, timestamp=datetime.utcnow(), ollama_connected=ollama_ok, redis_connected=redis_ok, version=settings.api_version)  # Return the health response payload


@app.get("/metrics", response_class=PlainTextResponse)  # Expose Prometheus metrics under /metrics
async def metrics() -> PlainTextResponse:  # Define the metrics handler
    metric_bytes = generate_latest()  # Generate the latest Prometheus metrics snapshot
    return PlainTextResponse(content=metric_bytes.decode("utf-8"), media_type=CONTENT_TYPE_LATEST)  # Return metrics as text/plain in the OpenMetrics format


@app.get("/models", response_model=ModelsResponse, dependencies=[Depends(get_current_user)])  # Expose the Ollama models list with authentication enforced
async def list_models() -> ModelsResponse:  # Define the models listing handler
    ensure_ollama_available()  # Ensure Ollama is ready before proceeding
    async with httpx.AsyncClient(timeout=settings.request_timeout_seconds) as client:  # Create an HTTP client for the request
        response = await client.get(f"{settings.ollama_base_url}/api/tags")  # Request the model listing from Ollama
        response.raise_for_status()  # Raise an exception if the Ollama service returned an error
    payload = response.json()  # Parse the JSON response body into a Python dict
    models_data = payload.get("models", [])  # Extract the models list from the response payload
    models = [ModelInfo(name=item.get("name", ""), size=item.get("size", 0), digest=item.get("digest", ""), modified_at=item.get("modified_at", "")) for item in models_data]  # Transform the raw payload into ModelInfo instances
    return ModelsResponse(models=models)  # Return the structured models response


@app.post("/create_agent", dependencies=[Depends(get_current_user)])  # Expose an endpoint to register agents securely
async def create_agent(agent_config: AgentConfig) -> Dict[str, str]:  # Define the agent creation handler
    ensure_ollama_available()  # Ensure Ollama is available before attempting agent creation
    REQUEST_COUNT.labels(method="POST", endpoint="/create_agent").inc()  # Increment request counter for observability
    with REQUEST_DURATION.labels(endpoint="/create_agent").time():  # Measure handler latency for performance tracking
        agent = Agent(role=agent_config.role, goal=agent_config.goal, backstory=agent_config.backstory, llm=ollama_llm, verbose=agent_config.verbose, allow_delegation=agent_config.allow_delegation, max_iter=agent_config.max_iter, max_rpm=agent_config.max_rpm, cache=agent_config.cache, tools=None)  # Instantiate the CrewAI agent to validate configuration
        agent_record = {  # Build a serializable agent record for persistence
            "name": agent_config.name,  # Persist the agent name
            "role": agent_config.role,  # Persist the agent role
            "goal": agent_config.goal,  # Persist the agent goal
            "backstory": agent_config.backstory,  # Persist the agent backstory
            "tools": agent_config.tools,  # Persist the requested tool identifiers
            "verbose": agent_config.verbose,  # Persist verbose preference
            "allow_delegation": agent_config.allow_delegation,  # Persist delegation preference
            "max_iter": agent_config.max_iter,  # Persist iteration limit
            "max_rpm": agent_config.max_rpm,  # Persist throughput limit
            "cache": agent_config.cache,  # Persist cache preference
            "created_at": datetime.utcnow().isoformat(),  # Timestamp the record creation time
        }
        client = ensure_redis_available()  # Ensure Redis is available before persisting data
        await client.set(f"agent:{agent_config.name}", json.dumps(agent_record), ex=settings.result_ttl_seconds)  # Persist the agent record with a configurable TTL
        AGENT_CREATIONS.inc()  # Increment the agent creation counter
        logger.info("Registered agent '%s'", agent_config.name)  # Log the successful agent registration
        return {"message": f"Agent {agent_config.name} registered"}  # Return a confirmation response to the caller


def build_task(task_config: TaskConfig, agents: Dict[str, Agent]) -> Task:  # Create a CrewAI Task from the provided configuration
    agent = agents.get(task_config.agent)  # Look up the assigned agent by name
    if agent is None:  # Validate that the referenced agent exists
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Unknown agent '{task_config.agent}' referenced in task '{task_config.name}'")  # Inform the client about the invalid reference
    return Task(description=task_config.description, expected_output=task_config.expected_output, agent=agent, async_execution=False)  # Construct the CrewAI task with the associated agent


async def execute_crew_job(crew_id: str, crew: Crew, crew_config: CrewConfig) -> None:  # Execute a crew asynchronously and persist the outcome
    start_time = datetime.utcnow()  # Record the start time for auditing
    try:
        result = await asyncio.to_thread(crew.kickoff)  # Run the blocking crew kickoff in a worker thread
        end_time = datetime.utcnow()  # Record the completion time
        record = {  # Build the execution record payload
            "crew_id": crew_id,  # Persist the crew identifier
            "config": crew_config.dict(),  # Persist the original crew configuration
            "result": str(result),  # Persist the textual representation of the result
            "started_at": start_time.isoformat(),  # Persist the start timestamp
            "completed_at": end_time.isoformat(),  # Persist the completion timestamp
        }
        client = ensure_redis_available()  # Ensure Redis is available before storing results
        await client.set(f"crew_result:{crew_id}", json.dumps(record), ex=settings.result_ttl_seconds)  # Persist the execution record with a TTL
        logger.info("Crew '%s' completed successfully", crew_id)  # Log the successful completion
    except Exception as exc:  # Handle execution failures gracefully
        logger.error("Crew '%s' execution failed: %s", crew_id, exc, exc_info=True)  # Log the failure with stack trace
        failure_record = {  # Build a failure record for persistence
            "crew_id": crew_id,  # Persist the crew identifier
            "config": crew_config.dict(),  # Persist the original configuration
            "error": str(exc),  # Persist the error message
            "failed_at": datetime.utcnow().isoformat(),  # Persist the failure timestamp
        }
        client = ensure_redis_available()  # Ensure Redis is available before storing failure records
        await client.set(f"crew_result:{crew_id}", json.dumps(failure_record), ex=settings.result_ttl_seconds)  # Persist the failure record with a TTL


@app.post("/run_crew", dependencies=[Depends(get_current_user)])  # Expose an endpoint to launch crew executions securely
async def run_crew(crew_config: CrewConfig) -> Dict[str, str]:  # Define the crew execution handler
    ensure_ollama_available()  # Ensure Ollama is ready before proceeding
    REQUEST_COUNT.labels(method="POST", endpoint="/run_crew").inc()  # Increment request counter for observability
    with REQUEST_DURATION.labels(endpoint="/run_crew").time():  # Measure handler latency for performance tracking
        agents: Dict[str, Agent] = {}  # Initialize the agent lookup dictionary
        for agent_config in crew_config.agents:  # Iterate over requested agent configurations
            agents[agent_config.name] = Agent(role=agent_config.role, goal=agent_config.goal, backstory=agent_config.backstory, llm=ollama_llm, verbose=crew_config.verbose or agent_config.verbose, allow_delegation=agent_config.allow_delegation, max_iter=agent_config.max_iter, max_rpm=agent_config.max_rpm, cache=agent_config.cache, tools=None)  # Instantiate each agent and store by name
        tasks = [build_task(task_config, agents) for task_config in crew_config.tasks]  # Build each CrewAI task while validating agent references
        crew = Crew(agents=list(agents.values()), tasks=tasks, verbose=crew_config.verbose)  # Instantiate the Crew with the prepared agents and tasks
        crew_id = f"crew_{int(datetime.utcnow().timestamp() * 1000)}"  # Generate a unique crew identifier using milliseconds precision
        asyncio.create_task(execute_crew_job(crew_id, crew, crew_config))  # Launch the crew execution asynchronously without blocking the request
        CREW_EXECUTIONS.inc()  # Increment the crew execution counter
        logger.info("Crew '%s' launched with %d agents and %d tasks", crew_id, len(agents), len(tasks))  # Log the launch event for observability
        return {"message": "Crew execution started", "crew_id": crew_id}  # Return the acknowledgement payload to the caller


@app.get("/crew_results/{crew_id}", dependencies=[Depends(get_current_user)])  # Expose an endpoint to fetch stored crew execution results
async def get_crew_result(crew_id: str) -> Dict[str, Any]:  # Define the crew result retrieval handler
    client = ensure_redis_available()  # Ensure Redis availability before fetching results
    payload = await client.get(f"crew_result:{crew_id}")  # Retrieve the stored execution record from Redis
    if payload is None:  # Handle missing results gracefully
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Crew result not found")  # Inform the client that the result is unavailable
    return json.loads(payload)  # Deserialize and return the execution record


@app.get("/")  # Expose a simple root endpoint for convenience
async def root() -> Dict[str, str]:  # Define the root handler
    return {"message": "CrewAI Service is running", "version": settings.api_version, "docs": "/docs", "health": "/health", "metrics": "/metrics"}  # Return service metadata for quick inspection


if __name__ == "__main__":  # Execute this block only when running the module as a script
    uvicorn.run(app, host=settings.host, port=settings.port, reload=settings.reload, log_level=settings.log_level.lower())  # Launch the Uvicorn server with configuration derived from settings
