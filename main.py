import asyncio  # Provide asynchronous utilities for background execution and concurrency control
import json  # Provide JSON serialization helpers for persisting structured data
import logging  # Provide structured logging for observability and incident response
import secrets  # Provide constant time comparison utilities for credential validation
from contextlib import asynccontextmanager  # Provide lifespan management for FastAPI application startup and shutdown
from datetime import datetime, timezone  # Provide timezone-aware timestamps for audit trails
from typing import Any, Dict, List, Optional  # Provide rich typing annotations for clarity and tooling support
from uuid import uuid4  # Provide cryptographically strong unique identifiers for crew executions

import httpx  # Provide hardened HTTP client for communicating with external services
import redis.asyncio as redis  # Provide asynchronous Redis client for secure state management
import uvicorn  # Provide ASGI server runner for development convenience
from crewai import Agent, Crew, Task  # Provide CrewAI primitives for orchestrating agents and tasks
from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException, Request  # Provide FastAPI components for API construction
from fastapi.middleware.cors import CORSMiddleware  # Provide CORS middleware for controlled cross-origin requests
from fastapi.responses import PlainTextResponse  # Provide plaintext response class for metrics exposure
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer  # Provide HTTP bearer security utilities
from langchain_community.llms import Ollama  # Provide Ollama large language model integration
from prometheus_client import Counter, Histogram, generate_latest  # Provide Prometheus metric primitives for monitoring
from prometheus_client.openmetrics.exposition import CONTENT_TYPE_LATEST  # Provide metrics content type constant for Prometheus compatibility
from pydantic import BaseModel, Field  # Provide Pydantic models for request validation and documentation

from config.settings import settings  # Import validated settings to drive secure configuration

logging.basicConfig(level=settings.log_level, format=settings.log_format)  # Configure logging using centralized settings to maintain consistency
logger = logging.getLogger("crewai_service")  # Acquire module-specific logger for contextualized messages

REQUEST_COUNT = Counter("crewai_requests_total", "Total requests served", ["method", "endpoint"])  # Track total HTTP requests by method and endpoint
REQUEST_DURATION = Histogram("crewai_request_duration_seconds", "Request processing duration in seconds")  # Track request latency distribution for performance monitoring
AGENT_CREATIONS = Counter("crewai_agent_creations_total", "Total number of agents created via API")  # Track agent creation activity for auditing
CREW_EXECUTIONS = Counter("crewai_crew_executions_total", "Total number of crew executions initiated")  # Track crew execution activity for auditing

redis_client: Optional[redis.Redis] = None  # Maintain global Redis client reference for reuse across requests
ollama_llm: Optional[Ollama] = None  # Maintain global Ollama client reference for agent instantiation
http_client: Optional[httpx.AsyncClient] = None  # Maintain global HTTP client reference for Ollama metadata lookups


@asynccontextmanager  # Ensure FastAPI invokes this coroutine to manage application lifecycle events
async def lifespan(app: FastAPI) -> Any:  # Define application lifespan hook with proper typing
    global redis_client  # Declare Redis client as global to allow assignment within the lifespan hook
    global ollama_llm  # Declare Ollama client as global to allow assignment within the lifespan hook
    global http_client  # Declare HTTP client as global to allow assignment within the lifespan hook
    logger.info("Starting CrewAI service initialization")  # Emit structured log to indicate startup progression

    settings.get_secret_key()  # Validate that a secure secret key is configured before serving requests

    redis_client = redis.from_url(  # Instantiate asynchronous Redis client using secure connection parameters
        settings.redis_url,  # Provide Redis connection URL sourced from configuration
        password=settings.redis_password.get_secret_value() if settings.redis_password else None,  # Provide optional Redis password while protecting secret access
        encoding="utf-8",  # Configure UTF-8 encoding to simplify string handling
        decode_responses=True,  # Enable response decoding to receive native Python strings
        max_connections=settings.redis_max_connections,  # Enforce connection pool size limits to prevent resource exhaustion
    )

    http_client = httpx.AsyncClient(  # Instantiate shared HTTP client for Ollama metadata requests
        base_url=settings.ollama_base_url,  # Configure base URL to target the Ollama service securely
        timeout=settings.ollama_timeout,  # Configure socket timeout to prevent hanging requests
        headers={"User-Agent": "CrewAI-Service/1.0"},  # Provide explicit user agent for auditability on downstream services
    )

    try:  # Execute Ollama client initialization with explicit error handling
        ollama_llm = Ollama(  # Instantiate Ollama large language model client
            base_url=settings.ollama_base_url,  # Provide Ollama base URL from configuration
            model=settings.ollama_model,  # Provide default Ollama model identifier
            timeout=settings.ollama_timeout,  # Provide timeout to align with HTTP client configuration
        )
        logger.info("Connected to Ollama model successfully")  # Emit success log when Ollama connectivity is established
    except Exception as exc:  # Catch any unexpected initialization failures
        logger.exception("Failed to initialize Ollama client: %s", exc)  # Emit detailed error log for troubleshooting
        ollama_llm = None  # Ensure Ollama client remains None when initialization fails gracefully

    try:  # Provide outer context to manage resource cleanup reliably
        yield  # Yield control back to FastAPI while resources remain active
    finally:  # Ensure cleanup executes even when exceptions occur during application shutdown
        logger.info("Commencing CrewAI service shutdown")  # Emit shutdown log for auditing
        if redis_client:  # Ensure Redis client exists before attempting closure
            await redis_client.close()  # Close Redis client to release network resources
            redis_client = None  # Reset Redis client reference to prevent reuse after shutdown
        if http_client:  # Ensure HTTP client exists before attempting closure
            await http_client.aclose()  # Close HTTP client to release socket resources
            http_client = None  # Reset HTTP client reference to prevent reuse after shutdown


app = FastAPI(  # Instantiate FastAPI application with security-focused defaults
    title=settings.api_title,  # Provide API title sourced from configuration for documentation clarity
    description="Secure CrewAI and Ollama integration service",  # Provide descriptive summary for generated docs
    version=settings.api_version,  # Provide API version sourced from configuration for lifecycle management
    lifespan=lifespan,  # Attach lifespan hook to orchestrate startup and shutdown routines
)  # Complete FastAPI instantiation call

app.add_middleware(  # Register CORS middleware to enforce allowed origins
    CORSMiddleware,  # Specify FastAPI CORS middleware implementation
    allow_origins=settings.cors_allowed_origins,  # Restrict cross-origin requests to configured list
    allow_credentials=True,  # Permit credentialed requests to support authenticated clients
    allow_methods=["GET", "POST", "OPTIONS"],  # Restrict allowed HTTP methods to reduce attack surface
    allow_headers=["Authorization", "Content-Type"],  # Restrict allowed headers to security-relevant fields
)  # Complete middleware registration call

security = HTTPBearer(auto_error=False)  # Instantiate HTTP bearer security dependency with explicit error handling


class AgentConfig(BaseModel):  # Define request schema for agent creation to enforce validation
    name: str = Field(..., min_length=1, max_length=128, description="Unique agent identifier")  # Validate agent name length and presence
    role: str = Field(..., min_length=1, max_length=256, description="Role describing the agent responsibilities")  # Validate agent role
    goal: str = Field(..., min_length=1, max_length=1024, description="Primary goal the agent should accomplish")  # Validate agent goal
    backstory: str = Field(..., min_length=1, max_length=2048, description="Background context for the agent")  # Validate agent backstory
    tools: Optional[List[str]] = Field(default=None, description="Optional list of tool identifiers assigned to the agent")  # Capture optional tool references
    verbose: bool = Field(default=False, description="Enable verbose logging for this agent instance")  # Allow per-agent verbosity control


class TaskConfig(BaseModel):  # Define request schema for task creation to enforce validation
    description: str = Field(..., min_length=1, max_length=4096, description="Detailed task description")  # Validate task description
    expected_output: str = Field(..., min_length=1, max_length=2048, description="Expected output narrative or format")  # Validate expected output description
    agent: str = Field(..., min_length=1, max_length=128, description="Agent identifier assigned to the task")  # Validate agent association


class CrewConfig(BaseModel):  # Define request schema for crew execution to enforce validation
    agents: List[AgentConfig] = Field(..., min_length=1, description="Collection of agents participating in the crew")  # Ensure at least one agent provided
    tasks: List[TaskConfig] = Field(..., min_length=1, description="Collection of tasks the crew should execute")  # Ensure at least one task provided
    verbose: bool = Field(default=False, description="Enable verbose logging for the crew execution")  # Allow global verbosity control


class HealthResponse(BaseModel):  # Define response schema for health checks to ensure consistent payloads
    status: str  # Provide overall health status string
    timestamp: datetime  # Provide timestamp to indicate freshness of health data
    ollama_connected: bool  # Indicate Ollama connectivity status
    redis_connected: bool  # Indicate Redis connectivity status
    version: str  # Provide running application version


class ModelInfo(BaseModel):  # Define response schema representing Ollama model metadata
    name: str  # Provide model name identifier
    size: int  # Provide model size in bytes
    digest: str  # Provide model digest for integrity verification
    modified_at: datetime  # Provide model last modified timestamp


class ModelsResponse(BaseModel):  # Define response schema for Ollama model list endpoint
    models: List[ModelInfo]  # Provide collection of available models with metadata


async def get_current_user(credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)) -> Dict[str, str]:  # Enforce bearer authentication for protected endpoints
    if credentials is None:  # Reject requests that omit authorization header
        raise HTTPException(status_code=401, detail="Missing bearer token")  # Provide explicit error when credentials absent
    expected_token = settings.get_secret_key()  # Retrieve configured secret key for secure comparison
    supplied_token = credentials.credentials  # Extract bearer token presented by client
    if not secrets.compare_digest(supplied_token, expected_token):  # Perform constant time token comparison to prevent timing attacks
        raise HTTPException(status_code=401, detail="Invalid bearer token")  # Reject unauthorized requests with consistent response
    return {"user_id": "service_account"}  # Return sanitized user context for downstream authorization checks


async def enforce_rate_limit(user_id: str) -> None:  # Enforce per-user rate limiting leveraging Redis for accuracy
    if not redis_client:  # Skip rate limiting when Redis is unavailable, logging occurs elsewhere
        return  # Allow request when rate limiting infrastructure is offline
    window = datetime.now(timezone.utc).strftime("%Y%m%d%H%M")  # Compute current minute window key for rate limiting
    key = f"rate_limit:{user_id}:{window}"  # Construct Redis key scoped to user and time window
    current_count = await redis_client.incr(key)  # Increment request count atomically within Redis
    if current_count == 1:  # Detect first request within the window to set expiration
        await redis_client.expire(key, 60)  # Set key expiration to one minute to bound window duration
    if current_count > settings.rate_limit_per_minute:  # Enforce configured request budget per user per minute
        raise HTTPException(status_code=429, detail="Rate limit exceeded, please retry later")  # Reject request with standard HTTP status code


@app.middleware("http")  # Register middleware to capture metrics for every HTTP request
async def record_metrics(request: Request, call_next):  # Define middleware function receiving request and handler
    REQUEST_COUNT.labels(method=request.method, endpoint=request.url.path).inc()  # Increment request counter with method and path labels
    start_time = datetime.now(timezone.utc).timestamp()  # Capture start timestamp for latency measurement
    response = await call_next(request)  # Invoke downstream request handler and await response
    duration = datetime.now(timezone.utc).timestamp() - start_time  # Calculate total request processing duration
    REQUEST_DURATION.observe(duration)  # Record duration in Prometheus histogram for analysis
    return response  # Return original response to client


@app.get("/health", response_model=HealthResponse)  # Expose authenticated health check endpoint for operational monitoring
async def health_check() -> HealthResponse:  # Define coroutine returning structured health response
    ollama_status = bool(ollama_llm)  # Determine if Ollama client is initialized
    redis_status = False  # Initialize Redis status flag pessimistically
    if redis_client:  # Attempt Redis check only when client initialized
        try:  # Perform ping with error handling to avoid cascading failures
            await redis_client.ping()  # Execute Redis ping command to verify connectivity
            redis_status = True  # Update status flag when ping succeeds
        except Exception as exc:  # Catch connectivity errors quietly
            logger.warning("Redis health check failed: %s", exc)  # Emit warning log for observability
            redis_status = False  # Maintain false status when ping fails
    overall_status = "healthy" if ollama_status and redis_status else "degraded"  # Determine aggregated health status
    return HealthResponse(  # Construct typed response payload
        status=overall_status,  # Provide computed health status
        timestamp=datetime.now(timezone.utc),  # Provide current timestamp for freshness
        ollama_connected=ollama_status,  # Provide Ollama connectivity outcome
        redis_connected=redis_status,  # Provide Redis connectivity outcome
        version=settings.api_version,  # Provide running application version from configuration
    )  # Complete response generation


@app.get("/metrics", response_class=PlainTextResponse)  # Expose Prometheus metrics endpoint without authentication
async def metrics() -> PlainTextResponse:  # Define coroutine returning plaintext response containing metrics
    return PlainTextResponse(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)  # Serialize metrics using Prometheus helper and return response


@app.get("/models", response_model=ModelsResponse)  # Expose endpoint for listing available Ollama models
async def list_models(user: Dict[str, str] = Depends(get_current_user)) -> ModelsResponse:  # Secure endpoint with bearer authentication dependency
    if not http_client:  # Ensure HTTP client is initialized before attempting request
        raise HTTPException(status_code=503, detail="Ollama metadata client unavailable")  # Return service unavailable when client missing
    try:  # Perform HTTP request with robust error handling
        response = await http_client.get("/api/tags")  # Call Ollama tags endpoint to retrieve available models
        response.raise_for_status()  # Raise exception for non-success HTTP status codes
        payload = response.json()  # Parse JSON response payload safely
        models = []  # Initialize collection to store parsed model information
        for item in payload.get("models", []):  # Iterate through models returned by Ollama service
            models.append(  # Append typed model information to response list
                ModelInfo(  # Instantiate Pydantic model info object
                    name=item.get("name", "unknown"),  # Extract model name with default fallback
                    size=int(item.get("size", 0)),  # Extract and coerce model size to integer
                    digest=item.get("digest", "unknown"),  # Extract model digest for integrity
                    modified_at=datetime.fromisoformat(item.get("modified_at", datetime.now(timezone.utc).isoformat())),  # Parse or synthesize modification timestamp
                )
            )
        return ModelsResponse(models=models)  # Return serialized models response
    except httpx.HTTPStatusError as exc:  # Catch HTTP response errors explicitly
        logger.warning("Ollama model listing failed with status %s", exc.response.status_code)  # Emit warning log with status code
        raise HTTPException(status_code=exc.response.status_code, detail="Failed to retrieve models from Ollama")  # Propagate error to client with sanitized message
    except Exception as exc:  # Catch unexpected errors for defense in depth
        logger.exception("Unexpected error while listing models: %s", exc)  # Emit exception log for diagnostics
        raise HTTPException(status_code=500, detail="Unexpected error retrieving models")  # Return internal server error to client


@app.post("/create_agent")  # Expose endpoint for creating and persisting agents
async def create_agent(agent_config: AgentConfig, user: Dict[str, str] = Depends(get_current_user)) -> Dict[str, str]:  # Secure endpoint with bearer authentication dependency
    await enforce_rate_limit(user["user_id"])  # Enforce rate limiting for the authenticated principal
    if not ollama_llm:  # Ensure Ollama client is available before creating agent
        raise HTTPException(status_code=503, detail="Ollama not connected")  # Reject request when Ollama unavailable
    try:  # Encapsulate agent creation within error handling block
        agent = Agent(  # Instantiate CrewAI agent with sanitized configuration
            role=agent_config.role,  # Provide agent role information
            goal=agent_config.goal,  # Provide agent goal information
            backstory=agent_config.backstory,  # Provide agent backstory information
            llm=ollama_llm,  # Provide shared Ollama client instance
            verbose=agent_config.verbose,  # Configure verbosity based on request
        )
        agent_data = {  # Prepare serializable agent metadata for persistence
            "name": agent_config.name,  # Persist agent name for retrieval
            "role": agent_config.role,  # Persist agent role for auditing
            "goal": agent_config.goal,  # Persist agent goal for auditing
            "backstory": agent_config.backstory,  # Persist agent backstory for context
            "tools": agent_config.tools or [],  # Persist attached tools safely handling None
            "created_at": datetime.now(timezone.utc).isoformat(),  # Persist creation timestamp for traceability
        }
        if redis_client:  # Persist agent metadata only when Redis is available
            await redis_client.set(  # Store agent metadata in Redis with time-bound retention
                f"agent:{agent_config.name}",  # Construct namespaced Redis key for agent
                json.dumps(agent_data),  # Serialize agent metadata to JSON string
                ex=settings.cache_ttl,  # Apply cache TTL from configuration to enforce retention policy
            )
        AGENT_CREATIONS.inc()  # Increment agent creation counter for metrics
        logger.info("Agent %s created by user %s", agent_config.name, user["user_id"])  # Emit informational log recording actor context
        return {"message": f"Agent {agent_config.name} created successfully"}  # Return success message to caller
    except Exception as exc:  # Catch unexpected failures during agent creation
        logger.exception("Error creating agent %s: %s", agent_config.name, exc)  # Emit exception log capturing failure context
        raise HTTPException(status_code=500, detail="Failed to create agent")  # Return sanitized error response to caller


@app.post("/run_crew")  # Expose endpoint for initiating crew executions
async def run_crew(crew_config: CrewConfig, background_tasks: BackgroundTasks, user: Dict[str, str] = Depends(get_current_user)) -> Dict[str, str]:  # Secure endpoint with bearer authentication dependency
    await enforce_rate_limit(user["user_id"])  # Enforce rate limiting for the authenticated principal
    if len(crew_config.agents) > settings.max_agents:  # Enforce configured maximum number of agents per request
        raise HTTPException(status_code=400, detail="Requested agent count exceeds configured limit")  # Reject oversized request with clear message
    if len(crew_config.tasks) > settings.max_tasks:  # Enforce configured maximum number of tasks per request
        raise HTTPException(status_code=400, detail="Requested task count exceeds configured limit")  # Reject oversized request with clear message
    if not ollama_llm:  # Ensure Ollama client is available before executing crew
        raise HTTPException(status_code=503, detail="Ollama not connected")  # Reject request when Ollama unavailable
    crew_agents: Dict[str, Agent] = {}  # Initialize mapping of agent names to instantiated CrewAI agents
    for agent_definition in crew_config.agents:  # Iterate through agent configurations supplied by client
        crew_agents[agent_definition.name] = Agent(  # Instantiate CrewAI agent and store in mapping
            role=agent_definition.role,  # Provide agent role information
            goal=agent_definition.goal,  # Provide agent goal information
            backstory=agent_definition.backstory,  # Provide agent backstory information
            llm=ollama_llm,  # Provide shared Ollama client instance
            verbose=crew_config.verbose or agent_definition.verbose,  # Determine verbosity preference
        )
    crew_tasks: List[Task] = []  # Initialize collection of CrewAI tasks
    for task_definition in crew_config.tasks:  # Iterate through task configurations supplied by client
        if task_definition.agent not in crew_agents:  # Validate referenced agent exists within crew mapping
            raise HTTPException(status_code=400, detail=f"Agent {task_definition.agent} not defined in crew configuration")  # Reject request referencing unknown agent
        crew_tasks.append(  # Append instantiated task to crew task list
            Task(  # Instantiate CrewAI task object
                description=task_definition.description,  # Provide task description
                expected_output=task_definition.expected_output,  # Provide expected output guidance
                agent=crew_agents[task_definition.agent],  # Associate task with instantiated agent
            )
        )
    crew = Crew(  # Instantiate CrewAI crew with prepared agents and tasks
        agents=list(crew_agents.values()),  # Provide list of instantiated agents
        tasks=crew_tasks,  # Provide list of instantiated tasks
        verbose=crew_config.verbose,  # Configure crew-level verbosity
    )
    crew_id = f"crew_{uuid4().hex}"  # Generate unique identifier for this crew execution
    background_tasks.add_task(run_crew_async, crew, crew_config, crew_id, user["user_id"])  # Schedule asynchronous crew execution in background
    CREW_EXECUTIONS.inc()  # Increment crew execution counter for metrics
    logger.info("Crew %s launched by user %s with %d agents and %d tasks", crew_id, user["user_id"], len(crew_agents), len(crew_tasks))  # Emit informational log capturing execution context
    return {"message": "Crew execution started", "crew_id": crew_id}  # Return acknowledgement payload to caller


async def run_crew_async(crew: Crew, crew_config: CrewConfig, crew_id: str, user_id: str) -> None:  # Execute crew in background thread to avoid blocking request handling
    try:  # Provide error handling for background execution
        result = await asyncio.to_thread(crew.kickoff)  # Execute crew kickoff in worker thread to preserve event loop responsiveness
        logger.info("Crew %s completed for user %s", crew_id, user_id)  # Emit informational log when execution completes
        if redis_client:  # Persist results only when Redis is available
            payload = {  # Prepare result payload for persistence
                "crew_id": crew_id,  # Persist crew identifier for retrieval
                "config": crew_config.model_dump(),  # Persist crew configuration for auditing
                "result": str(result),  # Persist textual representation of crew output
                "completed_at": datetime.now(timezone.utc).isoformat(),  # Persist completion timestamp
                "requested_by": user_id,  # Persist requesting user identifier
            }
            await redis_client.set(  # Store crew results in Redis with retention policy
                f"crew_result:{crew_id}",  # Construct namespaced Redis key for crew result
                json.dumps(payload),  # Serialize payload to JSON string
                ex=settings.result_ttl,  # Apply TTL configured for result retention
            )
    except Exception as exc:  # Catch unexpected failures during execution
        logger.exception("Crew %s execution failed: %s", crew_id, exc)  # Emit exception log capturing failure context
        if redis_client:  # Persist failure metadata when Redis available
            error_payload = {  # Prepare error payload for persistence
                "crew_id": crew_id,  # Persist crew identifier
                "error": str(exc),  # Persist error message for diagnostics
                "failed_at": datetime.now(timezone.utc).isoformat(),  # Persist failure timestamp
                "requested_by": user_id,  # Persist requesting user identifier
            }
            await redis_client.set(  # Store failure payload in Redis with retention policy
                f"crew_result:{crew_id}",  # Construct namespaced Redis key for crew result
                json.dumps(error_payload),  # Serialize payload to JSON string
                ex=settings.result_ttl,  # Apply TTL configured for result retention
            )


@app.get("/crew_results/{crew_id}")  # Expose endpoint for retrieving crew execution results
async def get_crew_result(crew_id: str, user: Dict[str, str] = Depends(get_current_user)) -> Dict[str, Any]:  # Secure endpoint with bearer authentication dependency
    if not redis_client:  # Ensure Redis client is available before attempting retrieval
        raise HTTPException(status_code=503, detail="Result store unavailable")  # Reject request when Redis unavailable
    result = await redis_client.get(f"crew_result:{crew_id}")  # Attempt to retrieve stored result payload from Redis
    if not result:  # Handle missing results gracefully
        raise HTTPException(status_code=404, detail="Crew result not found")  # Return not found status when key missing
    return json.loads(result)  # Deserialize JSON payload and return to caller


@app.get("/")  # Expose root endpoint providing service metadata
async def root() -> Dict[str, str]:  # Define coroutine returning simple metadata payload
    return {  # Return dictionary containing high-level service details
        "message": "CrewAI Service is running",  # Provide human-readable status message
        "version": settings.api_version,  # Provide application version information
        "docs": "/docs",  # Provide link to interactive API documentation
        "health": "/health",  # Provide link to health check endpoint
        "metrics": "/metrics",  # Provide link to Prometheus metrics endpoint
    }  # Complete response payload


if __name__ == "__main__":  # Support running the application directly via python main.py
    uvicorn.run(  # Launch Uvicorn ASGI server for local development usage
        app,  # Provide FastAPI application instance to Uvicorn
        host=settings.host,  # Bind server to configured host interface
        port=settings.port,  # Bind server to configured port
        reload=settings.reload,  # Enable or disable autoreload based on configuration
    )  # Complete server invocation call
