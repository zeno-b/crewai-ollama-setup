import asyncio  # Import asyncio to coordinate asynchronous operations and background scheduling
import json  # Import json to serialize data structures for storage in Redis
import logging  # Import logging to configure structured application logs
import os  # Import os to access environment variables that influence runtime configuration
import secrets  # Import secrets to perform constant-time comparisons for security-sensitive tokens
import time  # Import time to measure request durations and enforce rate limiting windows
from collections import defaultdict, deque  # Import defaultdict and deque to implement efficient sliding window rate limiting
from contextlib import asynccontextmanager  # Import asynccontextmanager to manage startup and shutdown lifecycle events
from datetime import datetime  # Import datetime to timestamp records stored in Redis
from typing import Any, Dict, List, Optional  # Import typing helpers to document data structures and function signatures
from uuid import uuid4  # Import uuid4 to generate unique identifiers for crew executions

import redis.asyncio as redis  # Import the asynchronous Redis client to persist state
import uvicorn  # Import uvicorn to provide a production-ready ASGI server entrypoint
from crewai import Agent, Crew, Task  # Import CrewAI primitives to build agents, tasks, and crews
from fastapi import Depends, FastAPI, HTTPException, Request  # Import FastAPI core types to build the API
from fastapi.middleware.cors import CORSMiddleware  # Import CORSMiddleware to configure cross-origin access
from fastapi.responses import PlainTextResponse  # Import PlainTextResponse to expose Prometheus metrics correctly
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer  # Import HTTPBearer components to protect endpoints with bearer tokens
from langchain_community.llms import Ollama  # Import Ollama integration to communicate with the local language model runtime
from prometheus_client import Counter, Histogram, generate_latest  # Import Prometheus helpers to instrument the service
from prometheus_client.openmetrics.exposition import CONTENT_TYPE_LATEST  # Import content type constant for metrics endpoint
from pydantic import BaseModel, Field  # Import BaseModel and Field to define validated request bodies and response models

from config.settings import settings  # Import project settings to respect configuration sourced from the environment
from tools.custom_tools import ToolRegistry  # Import the tool registry to resolve tool names into CrewAI tool instances

# Configure logging before instantiating the application to ensure consistent formatting and levels
logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),  # Map the configured log level string to a logging constant
    format=settings.log_format  # Apply the configured log formatting template
)  # Close the logging.basicConfig call after supplying configuration arguments
logger = logging.getLogger(__name__)  # Acquire a module-scoped logger for structured logging

# Initialize Prometheus metrics that capture request volume, latency, and business events
REQUEST_COUNT = Counter("crewai_requests_total", "Total API requests handled", ["method", "endpoint"])  # Counter for total request volume segmented by method and endpoint
REQUEST_DURATION = Histogram("crewai_request_duration_seconds", "API request latency in seconds", ["endpoint"])  # Histogram to capture latency distributions per endpoint
AGENT_CREATIONS = Counter("crewai_agent_creations_total", "Total agents created via the API")  # Counter to track agent creation events
CREW_EXECUTIONS = Counter("crewai_crew_executions_total", "Total crew executions initiated")  # Counter to track crew execution events

# Prepare global variables that hold service-wide state
redis_client: Optional[redis.Redis] = None  # Store the asynchronous Redis client for reuse across requests
ollama_llm: Optional[Ollama] = None  # Store the Ollama language model client for reuse across requests
tool_registry = ToolRegistry()  # Instantiate the tool registry to resolve tool names on demand
security = HTTPBearer(auto_error=True)  # Configure bearer authentication that rejects missing or malformed credentials immediately
rate_limit_records: defaultdict[str, deque] = defaultdict(deque)  # Track request timestamps per client identifier to enforce rate limits
rate_limit_lock = asyncio.Lock()  # Use an asyncio lock to synchronize access to the rate limit data structure
RATE_LIMIT_WINDOW_SECONDS = 60  # Define the time window for the sliding rate limit in seconds
API_BEARER_TOKEN = os.getenv("API_BEARER_TOKEN")  # Read the expected bearer token from the environment for request authentication

# Warn operators if a secure bearer token has not been configured, as this weakens security posture
if not API_BEARER_TOKEN:  # Check whether the token was omitted
    logger.warning("API_BEARER_TOKEN is not set; secure endpoints will reject requests until it is configured")  # Emit a warning to prompt configuration hardening


class AgentConfig(BaseModel):  # Define a validated request model for agent creation requests
    name: str = Field(..., description="Agent name")  # Require the agent name and describe its purpose in API docs
    role: str = Field(..., description="Agent role")  # Require the agent role to inform the CrewAI agent
    goal: str = Field(..., description="Agent goal")  # Require the agent goal to guide the agent's objective
    backstory: str = Field(..., description="Agent backstory")  # Require the agent backstory for contextualization
    tools: Optional[List[str]] = Field(default=None, description="Names of tools the agent should use")  # Allow optional tool names to attach to the agent
    verbose: bool = Field(default=False, description="Enable verbose logging for this agent")  # Allow toggling verbose mode per agent
    allow_delegation: bool = Field(default=False, description="Allow the agent to delegate tasks to other agents")  # Control delegation capabilities
    max_iter: int = Field(default=25, ge=1, le=100, description="Maximum number of agent reasoning iterations")  # Bound the reasoning iterations for efficiency
    max_rpm: Optional[int] = Field(default=None, ge=1, description="Maximum requests per minute for the agent")  # Allow optional speed limits for agent interactions
    cache: bool = Field(default=True, description="Enable CrewAI caching for this agent")  # Allow caching to optimize repeated requests


class TaskConfig(BaseModel):  # Define a validated request model for task creation within a crew
    description: str = Field(..., description="Task description presented to the agent")  # Require a detailed task description
    expected_output: str = Field(..., description="Expected output the agent should produce")  # Require the expected output specification
    agent: str = Field(..., description="Name of the agent responsible for the task")  # Require the agent assignment for the task


class CrewConfig(BaseModel):  # Define the request model for launching a crew execution
    agents: List[AgentConfig] = Field(..., description="Definitions of agents participating in the crew")  # Require at least one agent for the crew
    tasks: List[TaskConfig] = Field(..., description="Definitions of tasks the crew should execute")  # Require at least one task for the crew
    verbose: bool = Field(default=False, description="Toggle verbose crew execution logs")  # Allow toggling verbose logging for the crew


class HealthResponse(BaseModel):  # Define the schema returned by the health check endpoint
    status: str  # Include a status string that summarizes overall health
    timestamp: datetime  # Include a timestamp for the health check response
    ollama_connected: bool  # Include whether the Ollama client is currently available
    redis_connected: bool  # Include whether Redis connectivity is currently healthy
    version: str  # Include the service version reported to clients


class ModelInfo(BaseModel):  # Define the schema for an individual model entry in the models response
    name: str  # Report the model name
    size: int  # Report the model size in bytes
    digest: str  # Report the unique digest identifying the model artifact
    modified_at: str  # Report the timestamp indicating when the model was last updated


class ModelsResponse(BaseModel):  # Define the schema returned by the models listing endpoint
    models: List[ModelInfo]  # Provide a list of available models with metadata


def _resolve_tools(tool_names: Optional[List[str]]) -> List[Any]:  # Resolve tool names into instantiated tool objects
    if not tool_names:  # Handle cases where no tool names were provided
        return []  # Return an empty list when no tools are requested
    resolved_tools = []  # Initialize the list that will hold resolved tool instances
    for name in tool_names:  # Iterate through each requested tool name
        tool = tool_registry.get_tool(name)  # Attempt to retrieve the tool from the registry
        if tool:  # Check whether the tool was found
            resolved_tools.append(tool)  # Append the resolved tool to the list
        else:  # Handle the case where a tool name could not be resolved
            logger.warning("Requested tool '%s' is not registered and will be ignored", name)  # Warn operators about the missing tool
    return resolved_tools  # Return the list of resolved tool instances


async def _require_bearer_token(credentials: HTTPAuthorizationCredentials = Depends(security)) -> Dict[str, Any]:  # Enforce bearer token authentication for protected endpoints
    if not API_BEARER_TOKEN:  # Reject access when the token has not been configured securely
        raise HTTPException(status_code=500, detail="API_BEARER_TOKEN is not configured")  # Signal misconfiguration to the caller
    if not credentials:  # Reject requests that fail to provide credentials
        raise HTTPException(status_code=401, detail="Missing authorization credentials")  # Indicate that authorization was missing
    if not secrets.compare_digest(credentials.credentials, API_BEARER_TOKEN):  # Compare the provided token securely against the expected token
        raise HTTPException(status_code=401, detail="Invalid authorization token")  # Reject invalid tokens without leaking timing information
    return {"token": credentials.credentials}  # Return minimal user context indicating authentication success


@asynccontextmanager  # Declare an async context manager to orchestrate startup and shutdown hooks
async def lifespan(app: FastAPI):  # Define the lifespan function required by FastAPI
    global redis_client  # Reference the module-level Redis client so it can be assigned
    global ollama_llm  # Reference the module-level Ollama client so it can be assigned
    logger.info("Starting CrewAI service initialization")  # Log the start of service initialization
    try:  # Attempt to connect to Redis using the configured settings
        redis_client = redis.from_url(
            settings.redis_url,  # Use the configured Redis URL
            encoding="utf-8",  # Decode Redis responses using UTF-8 to simplify handling
            decode_responses=True,  # Decode bytes into strings automatically
            max_connections=settings.redis_max_connections  # Respect the configured connection pool size
        )  # Complete the Redis client instantiation
        await redis_client.ping()  # Verify the Redis connection eagerly to catch configuration issues early
        logger.info("Connected to Redis at %s", settings.redis_url)  # Log successful Redis connectivity
    except Exception as redis_error:  # Handle connection failures gracefully
        logger.error("Redis connection failed: %s", redis_error)  # Log the connection failure for troubleshooting
        redis_client = None  # Clear the Redis client so later code can detect the failure
    try:  # Attempt to connect to the Ollama language model service
        ollama_llm = Ollama(
            base_url=settings.ollama_base_url,  # Use the configured Ollama base URL
            model=settings.ollama_model,  # Select the configured default model
            timeout=settings.ollama_timeout  # Respect the configured timeout for Ollama interactions
        )  # Complete the Ollama client instantiation
        logger.info("Connected to Ollama at %s using model %s", settings.ollama_base_url, settings.ollama_model)  # Log successful Ollama connectivity
    except Exception as ollama_error:  # Handle connection failures gracefully
        logger.error("Ollama connection failed: %s", ollama_error)  # Log the connection failure for troubleshooting
        ollama_llm = None  # Clear the Ollama client so later code can detect the failure
    yield  # Yield control to allow FastAPI to begin serving requests with the initialized resources
    logger.info("Shutting down CrewAI service")  # Log the start of shutdown procedures
    if redis_client:  # Check whether a Redis client was successfully created
        await redis_client.close()  # Close the Redis client to release connections cleanly
        logger.info("Redis connection closed")  # Log closure


app = FastAPI(  # Instantiate the FastAPI application with metadata and lifecycle management
    title=settings.api_title,  # Apply the API title from settings
    description="Production-ready CrewAI and Ollama integration",  # Provide a descriptive summary for documentation
    version=settings.api_version,  # Apply the API version from settings
    lifespan=lifespan,  # Supply the lifespan handler to manage startup and shutdown
    debug=settings.api_debug  # Enable or disable debug mode based on configuration
)  # Close the FastAPI constructor call


app.add_middleware(  # Register the CORS middleware to control cross-origin resource sharing
    CORSMiddleware,  # Specify the middleware class to instantiate
    allow_origins=["*"],  # Permit requests from any origin; adjust in production for tighter security
    allow_credentials=True,  # Permit credentials to be included in cross-origin requests
    allow_methods=["*"],  # Permit all HTTP methods
    allow_headers=["*"]  # Permit all headers
)  # Close the middleware registration call


@app.middleware("http")  # Register an HTTP middleware to enforce rate limiting and capture metrics
async def metrics_and_rate_limit_middleware(request: Request, call_next):  # Define the middleware function executed for every request
    start_time = time.perf_counter()  # Capture the start time to compute request duration
    identifier = request.client.host if request.client else "anonymous"  # Derive a client identifier for rate limiting
    async with rate_limit_lock:  # Serialize access to the shared rate limit state
        timestamps = rate_limit_records[identifier]  # Retrieve or create the deque of timestamps for this client
        now = time.monotonic()  # Capture a monotonic timestamp for accurate rate limiting calculations
        while timestamps and now - timestamps[0] > RATE_LIMIT_WINDOW_SECONDS:  # Remove timestamps that fall outside the rate limit window
            timestamps.popleft()  # Discard the oldest timestamp beyond the window
        if len(timestamps) >= settings.rate_limit_per_minute:  # Check whether the client exceeded the configured rate limit
            raise HTTPException(status_code=429, detail="Rate limit exceeded")  # Reject the request with a 429 response
        timestamps.append(now)  # Record the current request timestamp for future evaluations
    response = await call_next(request)  # Invoke the downstream handler to process the request
    duration = time.perf_counter() - start_time  # Compute the total request duration using the captured timestamps
    endpoint_label = request.url.path  # Extract the endpoint path for metrics labeling
    REQUEST_COUNT.labels(method=request.method, endpoint=endpoint_label).inc()  # Increment the request counter for the observed method and endpoint
    REQUEST_DURATION.labels(endpoint=endpoint_label).observe(duration)  # Record the observed request duration in the histogram
    return response  # Return the downstream response to the client


@app.get("/health", response_model=HealthResponse)  # Expose a structured health check endpoint
async def health_check() -> HealthResponse:  # Define the health check handler
    ollama_status = ollama_llm is not None  # Determine whether the Ollama client is available
    redis_status = False  # Initialize the Redis status flag to false
    if redis_client:  # Check whether a Redis client has been initialized
        try:  # Attempt to ping Redis to verify connectivity
            await redis_client.ping()  # Perform a ping to test Redis health
            redis_status = True  # Mark Redis as healthy when the ping succeeds
        except Exception as redis_error:  # Catch errors encountered while pinging Redis
            logger.error("Redis health check failed: %s", redis_error)  # Log the failure details
            redis_status = False  # Ensure the status remains false on failure
    overall_status = "healthy" if ollama_status and redis_status else "degraded"  # Determine the aggregate status based on dependencies
    return HealthResponse(  # Return a typed health response payload
        status=overall_status,  # Provide the aggregate health status
        timestamp=datetime.utcnow(),  # Provide the current UTC timestamp
        ollama_connected=ollama_status,  # Report the Ollama connectivity status
        redis_connected=redis_status,  # Report the Redis connectivity status
        version=settings.api_version  # Provide the configured API version
    )  # Close the response constructor


@app.get("/metrics", response_class=PlainTextResponse)  # Expose a Prometheus metrics endpoint
async def metrics() -> PlainTextResponse:  # Define the metrics endpoint handler
    return PlainTextResponse(  # Return the generated metrics as plain text
        content=generate_latest(),  # Generate the latest metrics snapshot
        media_type=CONTENT_TYPE_LATEST  # Supply the correct content type for Prometheus scrapers
    )  # Close the response construction


@app.get("/models", response_model=ModelsResponse, dependencies=[Depends(_require_bearer_token)])  # Secure endpoint that lists available Ollama models
async def list_models() -> ModelsResponse:  # Define the models list handler
    if not ollama_llm:  # Ensure the Ollama client is available before proceeding
        raise HTTPException(status_code=503, detail="Ollama not connected")  # Return a service unavailable error when Ollama is absent
    try:  # Attempt to retrieve model metadata from Ollama
        models = [  # Build a list of model metadata entries
            ModelInfo(  # Instantiate a model metadata record
                name=settings.ollama_model,  # Report the configured model name
                size=0,  # Size retrieval is not yet implemented, so default to zero
                digest="unknown",  # Digest retrieval is not yet implemented, so default to a placeholder
                modified_at="unknown"  # Modification timestamp retrieval is not yet implemented, so default to a placeholder
            )  # Close the ModelInfo instantiation
        ]  # Close the list literal containing model metadata
        return ModelsResponse(models=models)  # Return the structured models response
    except Exception as ollama_error:  # Handle unexpected failures communicating with Ollama
        logger.error("Error listing models: %s", ollama_error)  # Log the failure details
        raise HTTPException(status_code=500, detail=str(ollama_error))  # Propagate the error to the client


@app.post("/create_agent", dependencies=[Depends(_require_bearer_token)])  # Secure endpoint that provisions a CrewAI agent
async def create_agent(agent_config: AgentConfig) -> Dict[str, str]:  # Define the agent creation handler
    if not ollama_llm:  # Ensure the Ollama client is available before creating agents
        raise HTTPException(status_code=503, detail="Ollama not connected")  # Return a service unavailable error when Ollama is absent
    resolved_tools = _resolve_tools(agent_config.tools)  # Resolve the requested tool names into tool instances
    try:  # Attempt to instantiate the CrewAI agent using the provided configuration
        agent = Agent(  # Create the CrewAI agent instance
            role=agent_config.role,  # Supply the agent role
            goal=agent_config.goal,  # Supply the agent goal
            backstory=agent_config.backstory,  # Supply the agent backstory
            llm=ollama_llm,  # Supply the shared Ollama language model client
            tools=resolved_tools,  # Attach the resolved tool instances
            verbose=agent_config.verbose,  # Apply the verbose flag
            allow_delegation=agent_config.allow_delegation,  # Apply the delegation flag
            max_iter=agent_config.max_iter,  # Apply the maximum reasoning iterations
            max_rpm=agent_config.max_rpm,  # Apply the optional rate limit for the agent
            cache=agent_config.cache  # Apply the caching preference
        )  # Close the agent instantiation
        agent_data = {  # Construct a serializable representation of the agent for persistence
            "name": agent_config.name,  # Store the agent name
            "role": agent_config.role,  # Store the agent role
            "goal": agent_config.goal,  # Store the agent goal
            "backstory": agent_config.backstory,  # Store the agent backstory
            "verbose": agent_config.verbose,  # Store the verbose flag
            "allow_delegation": agent_config.allow_delegation,  # Store the delegation flag
            "max_iter": agent_config.max_iter,  # Store the iteration limit
            "max_rpm": agent_config.max_rpm,  # Store the optional per-minute limit
            "cache": agent_config.cache,  # Store the caching preference
            "tools": [tool.name for tool in resolved_tools],  # Store the resolved tool names for auditing
            "created_at": datetime.utcnow().isoformat()  # Timestamp the agent creation event
        }  # Close the agent_data dictionary
        if redis_client:  # Check whether Redis is available for persistence
            await redis_client.set(  # Store the agent data in Redis with a TTL
                f"agent:{agent_config.name}",  # Use a namespaced key per agent name
                json.dumps(agent_data),  # Serialize the agent data to JSON
                ex=settings.cache_ttl  # Apply the configured cache TTL
            )  # Close the Redis set call
        AGENT_CREATIONS.inc()  # Increment the agent creation counter for observability
        logger.info("Created agent %s with role %s", agent_config.name, agent_config.role)  # Log the agent creation event
        return {"message": f"Agent {agent_config.name} created successfully"}  # Return a success message to the caller
    except Exception as agent_error:  # Handle failures during agent creation
        logger.error("Error creating agent: %s", agent_error)  # Log the failure details
        raise HTTPException(status_code=500, detail=str(agent_error))  # Propagate the error to the client


@app.post("/run_crew", dependencies=[Depends(_require_bearer_token)])  # Secure endpoint that launches crew executions
async def run_crew(crew_config: CrewConfig) -> Dict[str, str]:  # Define the crew execution handler
    if not ollama_llm:  # Ensure the Ollama client is available before launching crews
        raise HTTPException(status_code=503, detail="Ollama not connected")  # Return a service unavailable error when Ollama is absent
    if len(crew_config.agents) > settings.max_agents:  # Enforce the configured maximum number of agents
        raise HTTPException(status_code=400, detail="Agent count exceeds configured maximum")  # Reject requests that exceed limits
    if len(crew_config.tasks) > settings.max_tasks:  # Enforce the configured maximum number of tasks
        raise HTTPException(status_code=400, detail="Task count exceeds configured maximum")  # Reject requests that exceed limits
    agents: Dict[str, Agent] = {}  # Prepare a mapping of agent names to instantiated CrewAI agents
    for agent_entry in crew_config.agents:  # Iterate through the agent definitions
        resolved_tools = _resolve_tools(agent_entry.tools)  # Resolve requested tools for the agent
        agents[agent_entry.name] = Agent(  # Instantiate the CrewAI agent and store it in the mapping
            role=agent_entry.role,  # Supply the agent role
            goal=agent_entry.goal,  # Supply the agent goal
            backstory=agent_entry.backstory,  # Supply the agent backstory
            llm=ollama_llm,  # Reuse the shared Ollama client
            tools=resolved_tools,  # Attach resolved tools
            verbose=crew_config.verbose or agent_entry.verbose,  # Enable verbose mode if either the crew or agent requests it
            allow_delegation=agent_entry.allow_delegation,  # Apply the delegation flag
            max_iter=agent_entry.max_iter,  # Apply the iteration limit
            max_rpm=agent_entry.max_rpm,  # Apply the optional per-minute limit
            cache=agent_entry.cache  # Apply the caching preference
        )  # Close the agent instantiation
    crew_tasks: List[Task] = []  # Prepare the list of CrewAI tasks to execute
    for task_entry in crew_config.tasks:  # Iterate through the task definitions
        if task_entry.agent not in agents:  # Ensure the referenced agent exists
            raise HTTPException(status_code=400, detail=f"Agent {task_entry.agent} not found")  # Reject tasks that reference missing agents
        crew_tasks.append(  # Append the new task to the list
            Task(  # Instantiate the CrewAI task
                description=task_entry.description,  # Supply the task description
                expected_output=task_entry.expected_output,  # Supply the expected output
                agent=agents[task_entry.agent]  # Assign the task to the resolved agent
            )  # Close the Task instantiation
        )  # Close the append call
    crew = Crew(  # Instantiate the CrewAI crew with the configured agents and tasks
        agents=list(agents.values()),  # Supply the list of agent instances
        tasks=crew_tasks,  # Supply the list of task instances
        verbose=crew_config.verbose  # Apply the verbose flag at the crew level
    )  # Close the Crew instantiation
    crew_id = f"crew_{uuid4().hex}"  # Generate a unique identifier for the crew execution
    asyncio.create_task(run_crew_async(crew, crew_config, crew_id))  # Schedule the crew execution asynchronously without blocking the request
    CREW_EXECUTIONS.inc()  # Increment the crew execution counter for observability
    logger.info("Started crew %s with %d agents and %d tasks", crew_id, len(agents), len(crew_tasks))  # Log the crew launch details
    return {"message": "Crew execution started", "crew_id": crew_id}  # Return confirmation to the caller along with the crew identifier


async def run_crew_async(crew: Crew, crew_config: CrewConfig, crew_id: str) -> None:  # Execute a crew asynchronously and persist results
    try:  # Attempt to execute the crew workflow
        result = await asyncio.to_thread(crew.kickoff)  # Execute the blocking CrewAI kickoff in a worker thread
        logger.info("Crew %s completed successfully", crew_id)  # Log successful completion
        if redis_client:  # Check whether Redis is available for result persistence
            result_payload = {  # Build a payload describing the execution outcome
                "crew_id": crew_id,  # Store the crew identifier
                "config": crew_config.model_dump(),  # Store the original configuration for auditing
                "result": str(result),  # Store the textual representation of the result
                "completed_at": datetime.utcnow().isoformat()  # Timestamp the completion time
            }  # Close the result payload dictionary
            await redis_client.set(  # Persist the result payload to Redis with an expiration
                f"crew_result:{crew_id}",  # Use a namespaced key tied to the crew identifier
                json.dumps(result_payload),  # Serialize the payload as JSON
                ex=settings.result_ttl  # Apply the configured TTL for result retention
            )  # Close the Redis set call
    except Exception as crew_error:  # Handle failures encountered during crew execution
        logger.error("Crew %s execution failed: %s", crew_id, crew_error)  # Log the failure details


@app.get("/crew_results/{crew_id}", dependencies=[Depends(_require_bearer_token)])  # Secure endpoint that retrieves crew execution results
async def get_crew_result(crew_id: str) -> Dict[str, Any]:  # Define the result retrieval handler
    if not redis_client:  # Ensure Redis is available before attempting retrieval
        raise HTTPException(status_code=503, detail="Redis not available")  # Return a service unavailable error when Redis is absent
    try:  # Attempt to fetch the result payload from Redis
        payload = await redis_client.get(f"crew_result:{crew_id}")  # Retrieve the stored JSON payload for the crew
        if not payload:  # Handle cases where no result was found
            raise HTTPException(status_code=404, detail="Crew result not found")  # Return a not found error to the caller
        return json.loads(payload)  # Deserialize and return the stored result payload
    except HTTPException:  # Allow HTTP exceptions to propagate without alteration
        raise  # Re-raise the HTTP exception to preserve status codes
    except Exception as redis_error:  # Handle unexpected Redis errors
        logger.error("Error retrieving crew result %s: %s", crew_id, redis_error)  # Log the failure details
        raise HTTPException(status_code=500, detail=str(redis_error))  # Propagate the error to the caller


@app.get("/")  # Expose a simple root endpoint for quick service verification
async def root() -> Dict[str, Any]:  # Define the root endpoint handler
    return {  # Return a JSON payload summarizing service metadata
        "message": "CrewAI Service is running",  # Provide a friendly service status message
        "version": settings.api_version,  # Report the configured API version
        "docs": "/docs",  # Point to the autogenerated documentation
        "health": "/health",  # Point to the health check endpoint
        "metrics": "/metrics"  # Point to the Prometheus metrics endpoint
    }  # Close the response dictionary


if __name__ == "__main__":  # Allow the module to be executed directly for local development
    uvicorn.run(  # Launch the Uvicorn ASGI server
        app,  # Provide the FastAPI application instance
        host=settings.host,  # Bind the server to the configured host
        port=settings.port,  # Bind the server to the configured port
        reload=settings.reload  # Enable auto-reload when requested for development workflows
    )  # Close the uvicorn.run invocation
