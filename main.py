from __future__ import annotations  # Enable postponed evaluation of annotations for forward references

import asyncio  # Import asyncio to support asynchronous task execution
import hashlib  # Import hashlib to hash bearer tokens securely
import hmac  # Import hmac to compare hashes in constant time
import json  # Import json to serialize responses when storing in Redis
import logging  # Import logging to provide audit-friendly diagnostics
from contextlib import asynccontextmanager  # Import asynccontextmanager to manage FastAPI lifespan events
from datetime import datetime  # Import datetime for timestamp generation
from typing import Any, Dict, List, Optional  # Import typing helpers for clarity and linting

import redis.asyncio as redis  # Import asyncio-compatible Redis client
import uvicorn  # Import uvicorn to run the ASGI application when invoked directly
from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException  # Import FastAPI primitives for API creation
from fastapi.middleware.cors import CORSMiddleware  # Import CORS middleware to control cross-origin access
from fastapi.responses import PlainTextResponse  # Import PlainTextResponse to return Prometheus metrics
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer  # Import HTTP bearer authentication helpers
from prometheus_client import Counter, Histogram, generate_latest  # Import Prometheus metrics helpers
from prometheus_client.openmetrics.exposition import CONTENT_TYPE_LATEST  # Import Prometheus content type constant
from pydantic import BaseModel, Field  # Import Pydantic BaseModel to define request and response schemas

from crewai import Agent, Crew, Task  # Import CrewAI primitives to manage agents, tasks, and crews
from langchain_community.llms import Ollama  # Import Ollama LLM client for agent configuration

from config.settings import settings  # Import application settings for configuration consistency

logging.basicConfig(level=getattr(logging, settings.log_level), format=settings.log_format)  # Configure root logging using settings
logger = logging.getLogger(__name__)  # Initialize a module-level logger for structured logging

REQUEST_COUNT = Counter("crewai_requests_total", "Total requests", ["method", "endpoint"])  # Track request counts per method and endpoint
REQUEST_DURATION = Histogram("crewai_request_duration_seconds", "Request duration")  # Track request durations for latency insights
AGENT_CREATIONS = Counter("crewai_agent_creations_total", "Total agent creations")  # Track agent creation events
CREW_EXECUTIONS = Counter("crewai_crew_executions_total", "Total crew executions")  # Track crew execution events

redis_client: Optional[redis.Redis] = None  # Maintain a module-level Redis client reference
ollama_llm: Optional[Ollama] = None  # Maintain a module-level Ollama client reference

security = HTTPBearer(auto_error=False)  # Configure HTTP bearer authentication without automatic exceptions


def hash_token(token: str) -> str:  # Derive a deterministic hash for bearer tokens
    return hashlib.sha256(token.encode("utf-8")).hexdigest()  # Compute and return the SHA-256 hash of the token


EXPECTED_TOKEN_HASH = settings.api_token_hash or hash_token(settings.secret_key)  # Determine the expected token hash for authentication


async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> Dict[str, str]:  # Validate bearer tokens and return user context
    if credentials is None:  # Reject requests lacking credentials
        raise HTTPException(status_code=401, detail="Missing authentication credentials.")  # Signal unauthorized access
    provided_hash = hash_token(credentials.credentials)  # Hash the provided bearer token for comparison
    if not hmac.compare_digest(provided_hash, EXPECTED_TOKEN_HASH):  # Compare hashes in constant time to avoid timing attacks
        raise HTTPException(status_code=403, detail="Invalid authentication token.")  # Signal forbidden access on mismatch
    return {"token_hash": provided_hash}  # Return a minimal user context for downstream use


@asynccontextmanager
async def lifespan(app: FastAPI):  # Manage application startup and shutdown routines
    global redis_client, ollama_llm  # Reference module-level clients for modification
    logger.info("Starting CrewAI service...")  # Log application startup
    redis_client = redis.from_url(  # Initialize the Redis client using application settings
        settings.redis_url,
        encoding="utf-8",
        decode_responses=True,
        max_connections=settings.redis_max_connections,
        password=settings.redis_password,
    )
    try:  # Attempt to verify Redis connectivity
        await redis_client.ping()  # Ping Redis to ensure the connection is healthy
        logger.info("Connected to Redis at %s", settings.redis_url)  # Log successful Redis connection
    except Exception as error:  # Handle Redis connection failures gracefully
        logger.warning("Redis connection check failed: %s", error)  # Log the warning for observability
    try:  # Attempt to initialize the Ollama client
        ollama_llm = Ollama(base_url=settings.ollama_base_url, model=settings.ollama_model, timeout=settings.ollama_timeout)  # Instantiate the Ollama client
        logger.info("Connected to Ollama at %s using model %s", settings.ollama_base_url, settings.ollama_model)  # Log successful Ollama connection
    except Exception as error:  # Handle Ollama connection failures gracefully
        logger.error("Failed to connect to Ollama: %s", error)  # Log the error for diagnostics
        ollama_llm = None  # Reset the Ollama client reference on failure
    try:  # Yield control back to FastAPI while the app runs
        yield  # Allow the application to serve requests
    finally:  # Execute shutdown routines after FastAPI completes
        logger.info("Shutting down CrewAI service...")  # Log application shutdown
        if redis_client:  # Close the Redis connection gracefully when available
            await redis_client.close()  # Close the Redis client to release resources


app = FastAPI(  # Instantiate the FastAPI application with metadata and lifespan management
    title=settings.api_title,  # Supply the API title from configuration
    description="Production-ready CrewAI and Ollama integration.",  # Provide a descriptive summary
    version=settings.api_version,  # Supply the API version from configuration
    debug=settings.api_debug,  # Toggle debug mode based on configuration
    lifespan=lifespan,  # Attach the lifespan context manager for startup and shutdown
)  # Finish instantiating the FastAPI application

app.add_middleware(  # Register CORS middleware for the API
    CORSMiddleware,  # Specify the middleware class
    allow_origins=settings.allowed_origins,  # Supply the list of allowed origins
    allow_credentials=True,  # Permit credentialed requests for authenticated access
    allow_methods=settings.allowed_methods,  # Supply the list of allowed HTTP methods
    allow_headers=settings.allowed_headers,  # Supply the list of allowed HTTP headers
)  # Finish configuring the CORS middleware


class AgentConfig(BaseModel):  # Define the request schema for creating agents
    name: str = Field(..., description="Agent name")  # Capture the agent identifier
    role: str = Field(..., description="Agent role")  # Capture the agent role description
    goal: str = Field(..., description="Agent goal")  # Capture the agent goal statement
    backstory: str = Field(..., description="Agent backstory")  # Capture the agent backstory context
    tools: Optional[List[str]] = Field(default_factory=list, description="Agent tool identifiers")  # Capture optional tool identifiers
    verbose: bool = Field(default=False, description="Enable verbose agent logging")  # Capture the verbosity preference


class TaskConfig(BaseModel):  # Define the request schema for creating tasks
    description: str = Field(..., description="Task description")  # Capture the task instructions
    expected_output: str = Field(..., description="Expected output description")  # Capture the expected task output
    agent: str = Field(..., description="Name of the responsible agent")  # Capture the agent assigned to the task


class CrewConfig(BaseModel):  # Define the request schema for creating crews
    agents: List[AgentConfig]  # Capture the list of agent definitions
    tasks: List[TaskConfig]  # Capture the list of task definitions
    verbose: bool = Field(default=False, description="Enable verbose crew logging")  # Capture the crew verbosity preference


class HealthResponse(BaseModel):  # Define the response schema for the health check endpoint
    status: str  # Capture the overall service status indicator
    timestamp: datetime  # Capture the timestamp when the health check was generated
    ollama_connected: bool  # Indicate whether Ollama connectivity is healthy
    redis_connected: bool  # Indicate whether Redis connectivity is healthy
    version: str  # Report the API version


class ModelInfo(BaseModel):  # Define the response schema for individual Ollama models
    name: str  # Capture the model name
    size: int  # Capture the model size in bytes
    digest: str  # Capture the model digest for integrity verification
    modified_at: str  # Capture the model modification timestamp


class ModelsResponse(BaseModel):  # Define the response schema for listing Ollama models
    models: List[ModelInfo]  # Capture the list of available models


@app.get("/health", response_model=HealthResponse)  # Expose the health check endpoint
async def health_check() -> HealthResponse:  # Implement the health check logic
    ollama_status = ollama_llm is not None  # Determine Ollama connectivity status
    redis_status = False  # Default Redis connectivity status to False
    if redis_client is not None:  # Check Redis connectivity when the client is initialized
        try:  # Attempt to ping Redis
            await redis_client.ping()  # Ping Redis to confirm connectivity
            redis_status = True  # Update status when the ping succeeds
        except Exception as error:  # Handle Redis ping failures gracefully
            logger.warning("Redis ping failed: %s", error)  # Log the failure for diagnostics
    status = "healthy" if ollama_status and redis_status else "degraded"  # Determine the overall status
    return HealthResponse(  # Return the health check response payload
        status=status,  # Include the computed status
        timestamp=datetime.utcnow(),  # Include the current UTC timestamp
        ollama_connected=ollama_status,  # Include the Ollama connectivity flag
        redis_connected=redis_status,  # Include the Redis connectivity flag
        version=settings.api_version,  # Include the API version from settings
    )  # Finish constructing the health response


@app.get("/metrics", response_class=PlainTextResponse)  # Expose the Prometheus metrics endpoint
async def metrics() -> PlainTextResponse:  # Implement the metrics endpoint
    return PlainTextResponse(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)  # Return the latest metrics with the correct content type


@app.get("/models", response_model=ModelsResponse, dependencies=[Depends(get_current_user)])  # Expose the model listing endpoint with authentication
async def list_models() -> ModelsResponse:  # Implement the model listing logic
    if not ollama_llm:  # Ensure the Ollama client is available before responding
        raise HTTPException(status_code=503, detail="Ollama not connected")  # Signal service unavailability when Ollama is offline
    try:  # Build a placeholder model list
        models = [  # Construct a static model list until dynamic retrieval is implemented
            ModelInfo(  # Instantiate model metadata
                name=settings.ollama_model,  # Use the configured model name
                size=0,  # Placeholder for model size when not available
                digest="unknown",  # Placeholder for model digest when not available
                modified_at="1970-01-01T00:00:00Z",  # Placeholder modification timestamp
            )
        ]  # Close the models list
        return ModelsResponse(models=models)  # Return the model list response
    except Exception as error:  # Capture and log unexpected issues
        logger.exception("Error listing Ollama models.")  # Record the failure for diagnostics
        raise HTTPException(status_code=500, detail=str(error))  # Signal an internal server error


@app.post("/create_agent", dependencies=[Depends(get_current_user)])  # Expose the agent creation endpoint with authentication
async def create_agent(agent_config: AgentConfig) -> Dict[str, Any]:  # Implement the agent creation logic
    REQUEST_COUNT.labels(method="POST", endpoint="/create_agent").inc()  # Increment the request counter for observability
    if ollama_llm is None:  # Ensure the Ollama client is available before proceeding
        raise HTTPException(status_code=503, detail="Ollama not connected")  # Signal service unavailability when Ollama is offline
    try:  # Attempt to create the agent
        agent = Agent(  # Instantiate the CrewAI agent
            role=agent_config.role,  # Supply the agent role
            goal=agent_config.goal,  # Supply the agent goal
            backstory=agent_config.backstory,  # Supply the agent backstory
            llm=ollama_llm,  # Supply the Ollama client
            verbose=agent_config.verbose,  # Supply the verbosity preference
            tools=agent_config.tools or None,  # Supply the optional tool identifiers
        )  # Finish constructing the agent
        agent_data = {  # Build the agent persistence payload
            "name": agent_config.name,  # Include the agent name
            "role": agent_config.role,  # Include the agent role
            "goal": agent_config.goal,  # Include the agent goal
            "backstory": agent_config.backstory,  # Include the agent backstory
            "tools": agent_config.tools,  # Include the agent tool identifiers
            "verbose": agent_config.verbose,  # Include the verbosity preference
            "created_at": datetime.utcnow().isoformat(),  # Include the creation timestamp
        }  # Close the agent data payload
        if redis_client:  # Persist the agent definition when Redis is available
            await redis_client.set(f"agent:{agent_config.name}", json.dumps(agent_data), ex=settings.cache_ttl)  # Store the agent definition with expiration
        AGENT_CREATIONS.inc()  # Increment the agent creation counter
        logger.info("Created agent %s with role %s", agent_config.name, agent_config.role)  # Log the successful creation
        return {"message": f"Agent {agent_config.name} created successfully", "agent": agent_data}  # Return a success response including agent metadata
    except Exception as error:  # Capture and log unexpected issues
        logger.exception("Error creating agent %s", agent_config.name)  # Record the failure for diagnostics
        raise HTTPException(status_code=500, detail=str(error))  # Signal an internal server error


@app.post("/run_crew", dependencies=[Depends(get_current_user)])  # Expose the crew execution endpoint with authentication
async def run_crew(crew_config: CrewConfig, background_tasks: BackgroundTasks) -> Dict[str, Any]:  # Implement the crew execution logic
    REQUEST_COUNT.labels(method="POST", endpoint="/run_crew").inc()  # Increment the request counter for observability
    if ollama_llm is None:  # Ensure the Ollama client is available before proceeding
        raise HTTPException(status_code=503, detail="Ollama not connected")  # Signal service unavailability when Ollama is offline
    try:  # Attempt to construct and execute the crew
        agents: Dict[str, Agent] = {}  # Initialize the agent mapping by name
        for agent_definition in crew_config.agents:  # Iterate over agent definitions supplied in the request
            agents[agent_definition.name] = Agent(  # Instantiate each agent and store by name
                role=agent_definition.role,  # Supply the role for the agent
                goal=agent_definition.goal,  # Supply the goal for the agent
                backstory=agent_definition.backstory,  # Supply the backstory for the agent
                llm=ollama_llm,  # Supply the shared Ollama client
                verbose=crew_config.verbose or agent_definition.verbose,  # Determine verbosity preference
                tools=agent_definition.tools or None,  # Supply optional tool identifiers
            )  # Finish constructing the agent
        tasks: List[Task] = []  # Initialize the task list
        for task_definition in crew_config.tasks:  # Iterate over task definitions supplied in the request
            if task_definition.agent not in agents:  # Validate that the referenced agent exists
                raise HTTPException(status_code=400, detail=f"Agent {task_definition.agent} not found")  # Signal bad request for missing agents
            tasks.append(  # Append the constructed task to the task list
                Task(  # Instantiate the CrewAI task
                    description=task_definition.description,  # Supply the task description
                    expected_output=task_definition.expected_output,  # Supply the expected output
                    agent=agents[task_definition.agent],  # Supply the responsible agent
                )
            )  # Finish constructing and appending the task
        crew = Crew(agents=list(agents.values()), tasks=tasks, verbose=crew_config.verbose)  # Instantiate the CrewAI crew with the configured agents and tasks
        crew_id = f"crew_{int(datetime.utcnow().timestamp())}"  # Generate a deterministic crew identifier
        background_tasks.add_task(run_crew_async, crew, crew_config, crew_id)  # Schedule the crew execution in the background
        CREW_EXECUTIONS.inc()  # Increment the crew execution counter
        logger.info("Scheduled crew %s with %d agents and %d tasks", crew_id, len(agents), len(tasks))  # Log the scheduled execution
        return {"message": "Crew execution started", "crew_id": crew_id}  # Return the crew identifier to the caller
    except HTTPException:  # Allow HTTP exceptions to propagate unchanged
        raise  # Re-raise the HTTPException for FastAPI to handle
    except Exception as error:  # Capture and log unexpected issues
        logger.exception("Error scheduling crew execution")  # Record the failure for diagnostics
        raise HTTPException(status_code=500, detail=str(error))  # Signal an internal server error


async def run_crew_async(crew: Crew, crew_config: CrewConfig, crew_id: str) -> None:  # Execute the crew workflow asynchronously
    try:  # Attempt to run the crew workflow
        result = await asyncio.to_thread(crew.kickoff)  # Run the synchronous crew kickoff in a thread pool
        logger.info("Crew %s completed with result: %s", crew_id, result)  # Log the completion event and result
        if redis_client:  # Persist the result when Redis is available
            payload = {  # Build the result payload for storage
                "crew_id": crew_id,  # Include the crew identifier
                "config": crew_config.model_dump(),  # Include the original crew configuration
                "result": str(result),  # Include the serialized result
                "completed_at": datetime.utcnow().isoformat(),  # Include the completion timestamp
            }  # Close the payload dictionary
            await redis_client.set(f"crew_result:{crew_id}", json.dumps(payload), ex=settings.result_ttl)  # Store the result with expiration
    except Exception as error:  # Capture and log unexpected issues
        logger.exception("Error executing crew %s", crew_id)  # Record the failure for diagnostics


@app.get("/crew_results/{crew_id}", dependencies=[Depends(get_current_user)])  # Expose the crew result retrieval endpoint with authentication
async def get_crew_result(crew_id: str) -> Dict[str, Any]:  # Implement the crew result retrieval logic
    if redis_client is None:  # Ensure Redis is available before proceeding
        raise HTTPException(status_code=503, detail="Redis not available")  # Signal service unavailability when Redis is offline
    result = await redis_client.get(f"crew_result:{crew_id}")  # Retrieve the stored crew result
    if result is None:  # Handle missing results gracefully
        raise HTTPException(status_code=404, detail="Crew result not found")  # Signal that the result could not be found
    return json.loads(result)  # Deserialize and return the stored result payload


@app.get("/")  # Expose the root endpoint to provide a quick status summary
async def root() -> Dict[str, Any]:  # Implement the root endpoint logic
    return {  # Return a simple status payload
        "message": "CrewAI Service is running",  # Provide a friendly status message
        "version": settings.api_version,  # Include the API version
        "docs": "/docs",  # Provide a link to the interactive documentation
        "health": "/health",  # Provide a link to the health endpoint
        "metrics": "/metrics",  # Provide a link to the metrics endpoint
    }  # Close the status payload


if __name__ == "__main__":  # Support running the application directly
    uvicorn.run(  # Launch the FastAPI application using uvicorn
        app,  # Supply the FastAPI application instance
        host=settings.host,  # Bind to the configured host
        port=settings.port,  # Bind to the configured port
        reload=settings.reload,  # Toggle autoreload based on configuration
    )  # Finish launching the application
