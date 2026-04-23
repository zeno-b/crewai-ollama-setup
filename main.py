import asyncio
import json
import logging
import os
import re
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional
from uuid import uuid4

import httpx
import redis.asyncio as redis
import uvicorn
from crewai import Agent, Crew, Task
from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from langchain_community.llms import Ollama
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from pydantic import BaseModel, Field

from config.settings import settings
from retraining.autopilot import AutopilotStatusStore, FingerprintStore, RetrainingAutopilot
from retraining.manager import DatasetManager, RetrainingClientError, RetrainingJobManager

_JOB_ID_PATTERN = re.compile(r"^job_[0-9a-f]{8,64}$", re.IGNORECASE)
_CREW_ID_PATTERN = re.compile(r"^[a-zA-Z0-9_-]{1,128}$")

logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format=settings.log_format,
)
logger = logging.getLogger(__name__)

REQUEST_COUNT = Counter("crewai_requests_total", "Total requests", ["method", "endpoint"])
REQUEST_DURATION = Histogram("crewai_request_duration_seconds", "Request duration")
AGENT_CREATIONS = Counter("crewai_agent_creations_total", "Total agent creations")
CREW_EXECUTIONS = Counter("crewai_crew_executions_total", "Total crew executions")
RETRAINING_JOBS_COUNTER = Counter(
    "crewai_retraining_jobs_total",
    "Total retraining jobs processed",
    ["status"],
)
RETRAINING_JOB_DURATION = Histogram(
    "crewai_retraining_job_duration_seconds",
    "Retraining job duration in seconds",
)

redis_client: Optional[redis.Redis] = None
ollama_llm: Optional[Ollama] = None
dataset_manager: Optional[DatasetManager] = None
retraining_manager: Optional[RetrainingJobManager] = None
autopilot: Optional[RetrainingAutopilot] = None

optional_bearer = HTTPBearer(auto_error=False)


async def require_api_token(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(optional_bearer),
) -> None:
    expected = settings.api_bearer_token
    if not expected:
        return
    if not credentials:
        raise HTTPException(
            status_code=401,
            detail="Missing Authorization header. Send: Authorization: Bearer <API_BEARER_TOKEN>.",
        )
    if credentials.credentials != expected:
        raise HTTPException(
            status_code=401,
            detail="Invalid bearer token. The credential did not match API_BEARER_TOKEN.",
        )


def _metrics_authorized(request: Request) -> None:
    expected = settings.metrics_bearer_token
    if not expected:
        return
    auth = request.headers.get("Authorization", "")
    if not auth:
        raise HTTPException(
            status_code=401,
            detail="Missing Authorization header. Send: Authorization: Bearer <METRICS_BEARER_TOKEN> to access /metrics.",
        )
    if auth != f"Bearer {expected}":
        raise HTTPException(
            status_code=401,
            detail="Invalid metrics bearer token. The credential did not match METRICS_BEARER_TOKEN.",
        )


@asynccontextmanager
async def lifespan(app: FastAPI):
    global redis_client, ollama_llm, dataset_manager, retraining_manager, autopilot

    logger.info("Starting CrewAI service...")

    redis_client = redis.from_url(
        settings.redis_dsn(),
        max_connections=settings.redis_max_connections,
        encoding="utf-8",
        decode_responses=True,
    )

    ollama_base_url = settings.ollama_base_url
    ollama_model = settings.ollama_model
    try:
        ollama_llm = Ollama(base_url=ollama_base_url, model=ollama_model)
        logger.info("Connected to Ollama at %s with model %s", ollama_base_url, ollama_model)
    except OSError as exc:
        logger.error("Failed to connect to Ollama: %s", exc)
        ollama_llm = None

    dataset_root = Path(os.getenv("DATASET_DIR", "data/datasets"))
    retraining_root = Path(os.getenv("RETRAINING_DIR", "data/retraining"))
    template_dir = Path(settings.modelfile_template_dir)
    if not template_dir.is_absolute():
        template_dir = Path(__file__).resolve().parent / template_dir

    dataset_manager = DatasetManager(
        dataset_root,
        max_content_bytes=settings.dataset_max_content_bytes,
    )
    retraining_manager = RetrainingJobManager(
        base_dir=retraining_root,
        dataset_manager=dataset_manager,
        redis_client=redis_client,
        result_ttl=settings.result_ttl,
        ollama_base_url=ollama_base_url,
        metrics={"counter": RETRAINING_JOBS_COUNTER, "duration": RETRAINING_JOB_DURATION},
        modelfile_template_dir=template_dir if template_dir.is_dir() else None,
    )

    state_dir = Path(settings.autopilot_state_dir)
    if not state_dir.is_absolute():
        state_dir = Path(__file__).resolve().parent / state_dir
    state_dir.mkdir(parents=True, exist_ok=True)
    fp_store = FingerprintStore(state_dir / "fingerprints.json", redis=redis_client)
    ap_status = AutopilotStatusStore(state_dir / "status.json")
    autopilot = RetrainingAutopilot(
        settings=settings,
        dataset_manager=dataset_manager,
        job_manager=retraining_manager,
        fingerprint_store=fp_store,
        status_store=ap_status,
    )
    if settings.autopilot_enabled:
        autopilot.start_background()
        logger.info(
            "Retraining autopilot enabled (poll every %ss, feeds configured: %s)",
            settings.autopilot_poll_interval_seconds,
            bool(settings.autopilot_news_feeds.strip()),
        )

    yield

    logger.info("Shutting down CrewAI service...")
    if autopilot:
        await autopilot.stop()
        autopilot = None
    if redis_client:
        await redis_client.close()


app = FastAPI(
    title=settings.api_title,
    description="Production-ready CrewAI and Ollama integration",
    version=settings.api_version,
    lifespan=lifespan,
)

_cors = settings.cors_origins_list()
_allow_creds = "*" not in _cors
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors,
    allow_credentials=_allow_creds,
    allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)


@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    method = request.method
    endpoint = request.url.path
    start_time = time.perf_counter()
    try:
        return await call_next(request)
    finally:
        REQUEST_COUNT.labels(method=method, endpoint=endpoint).inc()
        REQUEST_DURATION.observe(time.perf_counter() - start_time)


class AgentConfig(BaseModel):
    name: str = Field(..., description="Agent name")
    role: str = Field(..., description="Agent role")
    goal: str = Field(..., description="Agent goal")
    backstory: str = Field(..., description="Agent backstory")
    tools: Optional[List[str]] = Field(default_factory=list, description="Agent tools")
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
    size: int = 0
    digest: str = ""
    modified_at: str = ""


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
    base_model: str = Field(..., description="Base Ollama model (student / target weights)")
    dataset_name: str = Field(..., description="Dataset identifier to use during retraining")
    job_type: Literal["system_prompt", "distill"] = Field(
        default="system_prompt",
        description="system_prompt: SYSTEM + dataset in Modelfile; distill: MESSAGE-based template "
        "with optional JSONL auto-conversion (see README).",
    )
    teacher_model: Optional[str] = Field(
        default=None,
        description="Ollama model name for teacher (required when job_type is distill).",
    )
    template_name: Optional[str] = Field(
        default=None,
        description="Load Modelfile from MODELFILE_TEMPLATE_DIR/<name>.template",
    )
    instructions: Optional[str] = Field(default=None, description="Additional system instructions")
    modelfile_template: Optional[str] = Field(
        default=None,
        description="Custom Modelfile body; placeholders: {{DATASET}}, {{BASE_MODEL}}, "
        "{{MODEL_NAME}}, {{TEACHER_MODEL}}, {{INSTRUCTIONS}}, {{ADAPTER}}",
    )
    parameters: Dict[str, Any] = Field(
        default_factory=dict,
        description="Extra PARAMETER lines (e.g. temperature, num_ctx, stop)",
    )
    stream: bool = Field(default=False, description="Stream Ollama create output")
    keep_alive: Optional[str] = Field(
        default=None,
        description="Ollama keep-alive duration for the new model",
    )
    timeout: int = Field(
        default=1800,
        ge=60,
        le=7200,
        description="Maximum time allowed for retraining (seconds)",
    )


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
    job_type: Optional[str] = None
    teacher_model: Optional[str] = None
    template_name: Optional[str] = None


class RetrainingJobListResponse(BaseModel):
    jobs: List[RetrainingJobStatus]


class RetrainingJobLogsResponse(BaseModel):
    job_id: str
    logs: List[Dict[str, Any]]


class AutopilotStatusResponse(BaseModel):
    enabled: bool
    configured: bool
    state: Dict[str, Any]
    settings_summary: Dict[str, Any]


class AutopilotRunResponse(BaseModel):
    result: Dict[str, Any]


@app.get("/health", response_model=HealthResponse)
async def health_check():
    ollama_status = ollama_llm is not None
    redis_status = False
    if redis_client:
        try:
            await redis_client.ping()
            redis_status = True
        except Exception as exc:  # noqa: BLE001 - health must not fail on unexpected Redis errors
            logger.warning("Redis ping failed during health check: %s", exc)
            redis_status = False

    return HealthResponse(
        status="healthy" if ollama_status and redis_status else "degraded",
        timestamp=datetime.now(timezone.utc),
        ollama_connected=ollama_status,
        redis_connected=redis_status,
        version=settings.api_version,
    )


@app.get("/metrics", response_class=PlainTextResponse)
async def metrics_endpoint(request: Request):
    _metrics_authorized(request)
    return PlainTextResponse(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/models", response_model=ModelsResponse)
async def list_models():
    if not settings.ollama_base_url:
        raise HTTPException(
            status_code=503,
            detail="Ollama is not configured: OLLAMA_BASE_URL is empty. Set OLLAMA_BASE_URL to your Ollama HTTP base (for example http://ollama:11434).",
        )
    url = f"{settings.ollama_base_url.rstrip('/')}/api/tags"
    try:
        async with httpx.AsyncClient(timeout=min(settings.ollama_timeout, 60)) as client:
            resp = await client.get(url)
            resp.raise_for_status()
            data = resp.json()
    except httpx.TimeoutException as exc:
        logger.error("Timeout listing Ollama models at %s: %s", url, exc)
        raise HTTPException(
            status_code=504,
            detail=f"Ollama model list timed out after {min(settings.ollama_timeout, 60)}s while calling {url}.",
        ) from exc
    except httpx.HTTPStatusError as exc:
        status = exc.response.status_code if exc.response is not None else "unknown"
        body_preview = ""
        if exc.response is not None:
            try:
                body_preview = exc.response.text[:500]
            except OSError:
                body_preview = ""
        logger.error("Ollama returned HTTP %s for %s: %s", status, url, exc)
        raise HTTPException(
            status_code=502,
            detail=(
                f"Ollama model API returned HTTP {status} for GET {url}. "
                f"Response preview: {body_preview or '(empty body)'}"
            ),
        ) from exc
    except httpx.HTTPError as exc:
        logger.error("Error listing Ollama models at %s: %s", url, exc)
        raise HTTPException(
            status_code=502,
            detail=f"Failed to reach Ollama model API at {url}: {exc!s}",
        ) from exc

    raw_models = data.get("models") if isinstance(data, dict) else None
    if not isinstance(raw_models, list):
        raise HTTPException(
            status_code=502,
            detail="Unexpected response from Ollama: top-level JSON must be an object with a 'models' array.",
        )

    models: List[ModelInfo] = []
    for item in raw_models:
        if not isinstance(item, dict):
            continue
        name = item.get("name") or item.get("model")
        if not name:
            continue
        models.append(
            ModelInfo(
                name=str(name),
                size=int(item.get("size") or 0),
                digest=str(item.get("digest") or ""),
                modified_at=str(item.get("modified_at") or item.get("expires_at") or ""),
            )
        )
    return ModelsResponse(models=models)


@app.post("/datasets", response_model=DatasetInfo, status_code=201)
async def create_dataset(dataset: DatasetCreateRequest, _: None = Depends(require_api_token)):
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
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        logger.error("Dataset save incomplete: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except OSError as exc:
        logger.error("Failed to save dataset: %s", exc)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to save dataset to disk: {exc!s}",
        ) from exc


@app.get("/datasets", response_model=DatasetListResponse)
async def list_datasets():
    if not dataset_manager:
        raise HTTPException(status_code=503, detail="Dataset manager not initialized")
    records = await asyncio.to_thread(dataset_manager.list_datasets)
    return DatasetListResponse(datasets=[DatasetInfo(**record) for record in records])


@app.get("/datasets/{dataset_name}", response_model=DatasetDetail)
async def get_dataset(dataset_name: str, include_content: bool = False):
    if not dataset_manager:
        raise HTTPException(status_code=503, detail="Dataset manager not initialized")
    try:
        record = await asyncio.to_thread(dataset_manager.get_dataset, dataset_name, include_content)
        return DatasetDetail(**record)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.delete("/datasets/{dataset_name}")
async def delete_dataset(dataset_name: str, _: None = Depends(require_api_token)):
    if not dataset_manager:
        raise HTTPException(status_code=503, detail="Dataset manager not initialized")
    try:
        await asyncio.to_thread(dataset_manager.delete_dataset, dataset_name)
        return {"message": f"Dataset '{dataset_name}' deleted"}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post("/retraining/jobs", response_model=RetrainingJobStatus, status_code=202)
async def create_retraining_job(
    job_request: RetrainingJobRequest,
    _: None = Depends(require_api_token),
):
    if not retraining_manager or not dataset_manager:
        raise HTTPException(status_code=503, detail="Retraining components not initialized")

    try:
        await asyncio.to_thread(dataset_manager.get_dataset, job_request.dataset_name, False)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    payload = job_request.model_dump()
    try:
        await asyncio.to_thread(retraining_manager.validate_new_job_payload, payload)
    except RetrainingClientError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.message) from exc
    except (ValueError, FileNotFoundError) as exc:
        code = 404 if isinstance(exc, FileNotFoundError) else 400
        raise HTTPException(status_code=code, detail=str(exc)) from exc

    try:
        job_record = await retraining_manager.create_job(payload)
    except RetrainingClientError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.message) from exc
    except (ValueError, FileNotFoundError) as exc:
        code = 404 if isinstance(exc, FileNotFoundError) else 400
        raise HTTPException(status_code=code, detail=str(exc)) from exc
    except OSError as exc:
        logger.error("Failed to create retraining job: %s", exc)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to persist retraining job metadata: {exc!s}",
        ) from exc

    job_id = job_record["job_id"]
    asyncio.create_task(retraining_manager.run_job(job_id, payload, job_request.timeout))
    return RetrainingJobStatus(**job_record)


@app.get("/retraining/jobs", response_model=RetrainingJobListResponse)
async def list_retraining_jobs(limit: int = Query(50, ge=1, le=500)):
    if not retraining_manager:
        raise HTTPException(status_code=503, detail="Retraining components not initialized")
    records = await retraining_manager.list_jobs(limit)
    return RetrainingJobListResponse(jobs=[RetrainingJobStatus(**record) for record in records])


@app.get("/retraining/jobs/{job_id}", response_model=RetrainingJobStatus)
async def get_retraining_job(job_id: str):
    if not retraining_manager:
        raise HTTPException(status_code=503, detail="Retraining components not initialized")
    if not _JOB_ID_PATTERN.match(job_id):
        raise HTTPException(
            status_code=400,
            detail=(
                "Invalid job id: expected 'job_' followed by 8-64 hexadecimal characters "
                f"(received {job_id!r})."
            ),
        )
    try:
        record = await retraining_manager.get_job(job_id)
        return RetrainingJobStatus(**record)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/retraining/jobs/{job_id}/logs", response_model=RetrainingJobLogsResponse)
async def get_retraining_logs(job_id: str, tail: int = Query(100, ge=1, le=10_000)):
    if not retraining_manager:
        raise HTTPException(status_code=503, detail="Retraining components not initialized")
    if not _JOB_ID_PATTERN.match(job_id):
        raise HTTPException(
            status_code=400,
            detail=(
                "Invalid job id: expected 'job_' followed by 8-64 hexadecimal characters "
                f"(received {job_id!r})."
            ),
        )
    logs = await retraining_manager.get_logs(job_id, tail)
    return RetrainingJobLogsResponse(job_id=job_id, logs=logs)


@app.get("/retraining/autopilot/status", response_model=AutopilotStatusResponse)
async def autopilot_status():
    if not autopilot:
        raise HTTPException(status_code=503, detail="Autopilot not initialized")
    feeds = [f.strip() for f in settings.autopilot_news_feeds.split(",") if f.strip()]
    return AutopilotStatusResponse(
        enabled=settings.autopilot_enabled,
        configured=bool(feeds),
        state=autopilot.status.read_dict(),
        settings_summary={
            "poll_interval_seconds": settings.autopilot_poll_interval_seconds,
            "max_items_per_feed": settings.autopilot_max_items_per_feed,
            "dataset_name": settings.autopilot_dataset_name,
            "auto_finetune": settings.autopilot_auto_finetune,
            "min_new_items_to_finetune": settings.autopilot_min_new_items_to_finetune,
            "finetune_cooldown_seconds": settings.autopilot_finetune_cooldown_seconds,
            "base_model": settings.autopilot_base_model,
            "output_model_template": settings.autopilot_output_model_template,
        },
    )


@app.post("/retraining/autopilot/run", response_model=AutopilotRunResponse)
async def autopilot_run_now(_: None = Depends(require_api_token)):
    """Trigger one ingest + optional finetune cycle (for operators or cron sidecars)."""
    if not autopilot:
        raise HTTPException(status_code=503, detail="Autopilot not initialized")
    try:
        result = await autopilot.run_cycle()
    except Exception as exc:  # noqa: BLE001
        logger.exception("Manual autopilot run failed: %s", exc)
        raise HTTPException(
            status_code=500,
            detail=f"Autopilot run failed: {type(exc).__name__}: {exc}",
        ) from exc
    return AutopilotRunResponse(result=result)


@app.post("/create_agent")
async def create_agent(agent_config: AgentConfig, _: None = Depends(require_api_token)):
    if not ollama_llm:
        raise HTTPException(
            status_code=503,
            detail=(
                "Ollama LLM client is not available. Check that OLLAMA_BASE_URL is reachable "
                f"({settings.ollama_base_url}) and the service logs for connection errors."
            ),
        )
    try:
        _agent = Agent(
            role=agent_config.role,
            goal=agent_config.goal,
            backstory=agent_config.backstory,
            llm=ollama_llm,
            verbose=agent_config.verbose,
        )
        del _agent

        agent_data = {
            "name": agent_config.name,
            "role": agent_config.role,
            "goal": agent_config.goal,
            "backstory": agent_config.backstory,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        if redis_client:
            await redis_client.set(
                f"agent:{agent_config.name}",
                json.dumps(agent_data),
                ex=settings.cache_ttl,
            )
        AGENT_CREATIONS.inc()
        logger.info("Created agent: %s", agent_config.name)
        return {"message": f"Agent {agent_config.name} created successfully"}
    except Exception as exc:  # noqa: BLE001 - surface agent construction failures clearly
        logger.error("Error creating agent: %s", exc)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to create agent '{agent_config.name}': {type(exc).__name__}: {exc}",
        ) from exc


@app.post("/run_crew")
async def run_crew(
    crew_config: CrewConfig,
    background_tasks: BackgroundTasks,
    _: None = Depends(require_api_token),
):
    if not ollama_llm:
        raise HTTPException(
            status_code=503,
            detail=(
                "Ollama LLM client is not available; cannot run a crew. "
                f"Verify Ollama at {settings.ollama_base_url} and model {settings.ollama_model!r}."
            ),
        )
    try:
        agents: Dict[str, Agent] = {}
        for agent_config in crew_config.agents:
            agents[agent_config.name] = Agent(
                role=agent_config.role,
                goal=agent_config.goal,
                backstory=agent_config.backstory,
                llm=ollama_llm,
                verbose=crew_config.verbose,
            )
        tasks: List[Task] = []
        for task_config in crew_config.tasks:
            if task_config.agent not in agents:
                known = ", ".join(sorted(agents)) or "(none)"
                raise HTTPException(
                    status_code=400,
                    detail=(
                        f"Task references unknown agent {task_config.agent!r}. "
                        f"Defined agent names: {known}."
                    ),
                )
            tasks.append(
                Task(
                    description=task_config.description,
                    expected_output=task_config.expected_output,
                    agent=agents[task_config.agent],
                )
            )
        crew = Crew(agents=list(agents.values()), tasks=tasks, verbose=crew_config.verbose)
        crew_id = f"crew_{uuid4().hex}"
        background_tasks.add_task(run_crew_async, crew, crew_config, crew_id)
        CREW_EXECUTIONS.inc()
        logger.info("Started crew execution with %s agents and %s tasks", len(agents), len(tasks))
        return {"message": "Crew execution started", "crew_id": crew_id}
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001 - crew wiring can raise varied library errors
        logger.error("Error running crew: %s", exc)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to start crew execution: {type(exc).__name__}: {exc}",
        ) from exc


async def run_crew_async(crew: Crew, crew_config: CrewConfig, crew_id: str):
    try:
        result = crew.kickoff()
        logger.info("Crew execution completed: %s", result)
        if redis_client:
            result_data = {
                "status": "completed",
                "config": crew_config.model_dump(),
                "result": str(result),
                "completed_at": datetime.now(timezone.utc).isoformat(),
            }
            await redis_client.set(
                f"crew_result:{crew_id}",
                json.dumps(result_data),
                ex=settings.result_ttl,
            )
    except Exception as exc:  # noqa: BLE001 - background task must record any kickoff failure
        logger.exception("Crew execution failed for %s", crew_id)
        if redis_client:
            error_payload = {
                "status": "failed",
                "config": crew_config.model_dump(),
                "error": f"{type(exc).__name__}: {exc}",
                "completed_at": datetime.now(timezone.utc).isoformat(),
            }
            try:
                await redis_client.set(
                    f"crew_result:{crew_id}",
                    json.dumps(error_payload),
                    ex=settings.result_ttl,
                )
            except OSError as redis_exc:
                logger.error("Failed to store crew failure in Redis: %s", redis_exc)


@app.get("/crew_results/{crew_id}")
async def get_crew_result(crew_id: str):
    if not redis_client:
        raise HTTPException(status_code=503, detail="Redis not available")
    if not _CREW_ID_PATTERN.match(crew_id):
        raise HTTPException(
            status_code=400,
            detail=(
                "Invalid crew id: use only letters, digits, underscore, or hyphen, "
                f"1-128 characters (received length {len(crew_id)})."
            ),
        )
    try:
        result = await redis_client.get(f"crew_result:{crew_id}")
        if not result:
            raise HTTPException(
                status_code=404,
                detail=(
                    f"No crew result found for id {crew_id!r}. "
                    "The job may still be running, may have failed before writing a result, "
                    "or the id may be wrong / expired (see RESULT_TTL)."
                ),
            )
        return json.loads(result)
    except HTTPException:
        raise
    except json.JSONDecodeError as exc:
        logger.error("Invalid JSON in crew result for %s: %s", crew_id, exc)
        raise HTTPException(status_code=500, detail="Invalid stored crew result") from exc
    except OSError as exc:
        logger.error("Error retrieving crew result: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/")
async def root():
    return {
        "message": "CrewAI Service is running",
        "version": settings.api_version,
        "docs": "/docs",
        "health": "/health",
        "metrics": "/metrics",
    }


if __name__ == "__main__":
    port = int(os.getenv("CREWAI_PORT", settings.port))
    uvicorn.run(
        app,
        host=settings.host,
        port=port,
        reload=os.getenv("RELOAD", "false").lower() == "true",
    )
