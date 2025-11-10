import asyncio
import json
import logging
import re
import textwrap
from datetime import datetime, timezone
from pathlib import Path
from threading import RLock
from typing import Any, Dict, List, Optional
from uuid import uuid4

import httpx

logger = logging.getLogger(__name__)

_NAME_SANITIZER = re.compile(r"[^a-zA-Z0-9_-]+")
MAX_DATASET_CHARS = 120_000


def _utcnow() -> str:
    return datetime.utcnow().replace(tzinfo=timezone.utc).isoformat()


def _sanitize_name(name: str) -> str:
    candidate = _NAME_SANITIZER.sub("-", name.strip().lower()).strip("-")
    if not candidate:
        raise ValueError("Dataset name must contain alphanumeric characters after sanitization.")
    return candidate


class DatasetManager:
    """Manage retraining datasets stored on disk."""

    def __init__(self, dataset_dir: Path):
        self.dataset_dir = dataset_dir
        self.dataset_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_path = self.dataset_dir / "metadata.json"
        self._lock = RLock()
        self._metadata = self._load_metadata()

    def _load_metadata(self) -> Dict[str, Dict[str, Any]]:
        if self.metadata_path.exists():
            try:
                with self.metadata_path.open("r", encoding="utf-8") as f:
                    data = json.load(f)
                    if isinstance(data, dict):
                        return data
            except Exception as exc:
                logger.warning("Failed to load dataset metadata: %s", exc)
        return {}

    def _persist_metadata(self) -> None:
        tmp_path = self.metadata_path.with_suffix(".tmp")
        with tmp_path.open("w", encoding="utf-8") as f:
            json.dump(self._metadata, f, indent=2)
        tmp_path.replace(self.metadata_path)

    def save_dataset(
        self,
        name: str,
        content: str,
        description: Optional[str],
        tags: List[str],
        fmt: str,
        overwrite: bool = False,
    ) -> Dict[str, Any]:
        with self._lock:
            dataset_id = _sanitize_name(name)
            extension = ".jsonl" if fmt == "jsonl" else ".txt"
            dataset_path = self.dataset_dir / f"{dataset_id}{extension}"

            if dataset_id in self._metadata and not overwrite:
                raise ValueError(f"Dataset '{dataset_id}' already exists. Use overwrite flag to replace it.")

            with dataset_path.open("w", encoding="utf-8") as f:
                f.write(content)

            size_bytes = dataset_path.stat().st_size
            timestamp = _utcnow()
            record = {
                "name": dataset_id,
                "display_name": name,
                "description": description,
                "tags": tags,
                "format": fmt,
                "size_bytes": size_bytes,
                "path": str(dataset_path),
                "created_at": self._metadata.get(dataset_id, {}).get("created_at", timestamp),
                "updated_at": timestamp,
            }

            self._metadata[dataset_id] = record
            self._persist_metadata()

            return record

    def list_datasets(self) -> List[Dict[str, Any]]:
        with self._lock:
            return sorted(
                self._metadata.values(),
                key=lambda item: item.get("updated_at", ""),
                reverse=True,
            )

    def get_dataset(self, name: str, include_content: bool = False) -> Dict[str, Any]:
        dataset_id = _sanitize_name(name)
        with self._lock:
            data = self._metadata.get(dataset_id)
            if not data:
                raise FileNotFoundError(f"Dataset '{dataset_id}' not found.")

        if not include_content:
            return data

        dataset_path = Path(data["path"])
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset file for '{dataset_id}' is missing.")

        with dataset_path.open("r", encoding="utf-8") as f:
            content = f.read()

        enriched = dict(data)
        enriched["content"] = content
        return enriched

    def delete_dataset(self, name: str) -> None:
        dataset_id = _sanitize_name(name)
        with self._lock:
            data = self._metadata.pop(dataset_id, None)
            if not data:
                raise FileNotFoundError(f"Dataset '{dataset_id}' not found.")
            self._persist_metadata()

        dataset_path = Path(data["path"])
        if dataset_path.exists():
            dataset_path.unlink(missing_ok=True)


class RetrainingJobManager:
    """Manage Ollama retraining jobs."""

    def __init__(
        self,
        base_dir: Path,
        dataset_manager: DatasetManager,
        redis_client: Optional[Any],
        result_ttl: int,
        ollama_base_url: str,
        metrics: Optional[Dict[str, Any]] = None,
    ):
        self.base_dir = base_dir
        self.dataset_manager = dataset_manager
        self.redis = redis_client
        self.result_ttl = result_ttl
        self.ollama_base_url = ollama_base_url.rstrip("/")
        self.jobs_dir = self.base_dir / "jobs"
        self.jobs_dir.mkdir(parents=True, exist_ok=True)
        self.metrics = metrics or {}
        self._lock = asyncio.Lock()

    def _job_dir(self, job_id: str) -> Path:
        path = self.jobs_dir / job_id
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _job_status_path(self, job_id: str) -> Path:
        return self._job_dir(job_id) / "status.json"

    def _job_log_path(self, job_id: str) -> Path:
        return self._job_dir(job_id) / "logs.ndjson"

    def _job_key(self, job_id: str) -> str:
        return f"retrain:job:{job_id}"

    async def create_job(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        job_id = f"job_{uuid4().hex}"
        job_record = {
            "job_id": job_id,
            "status": "queued",
            "model_name": payload["model_name"],
            "base_model": payload["base_model"],
            "dataset_name": payload["dataset_name"],
            "created_at": _utcnow(),
            "updated_at": _utcnow(),
            "parameters": payload.get("parameters", {}),
            "stream": payload.get("stream", False),
        }
        await self._store_job(job_id, job_record)
        await self._append_log(job_id, {"event": "created", "payload": payload})
        self._increment_metric("counter", status="queued")
        return job_record

    async def _store_job(self, job_id: str, data: Dict[str, Any]) -> None:
        job_path = self._job_status_path(job_id)
        await asyncio.to_thread(self._write_json, job_path, data)
        if self.redis:
            await self.redis.set(self._job_key(job_id), json.dumps(data), ex=self.result_ttl)

    @staticmethod
    def _write_json(path: Path, data: Dict[str, Any]) -> None:
        tmp_path = path.with_suffix(".tmp")
        with tmp_path.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        tmp_path.replace(path)

    async def get_job(self, job_id: str) -> Dict[str, Any]:
        if self.redis:
            cached = await self.redis.get(self._job_key(job_id))
            if cached:
                return json.loads(cached)

        job_path = self._job_status_path(job_id)
        if not job_path.exists():
            raise FileNotFoundError(f"Job '{job_id}' not found.")

        return await asyncio.to_thread(self._read_json, job_path)

    @staticmethod
    def _read_json(path: Path) -> Dict[str, Any]:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)

    async def list_jobs(self, limit: int = 50) -> List[Dict[str, Any]]:
        jobs = []
        for job_path in sorted(self.jobs_dir.glob("*/status.json"), key=lambda p: p.stat().st_mtime, reverse=True):
            jobs.append(await asyncio.to_thread(self._read_json, job_path))
            if len(jobs) >= limit:
                break
        return jobs

    async def _append_log(self, job_id: str, entry: Dict[str, Any]) -> None:
        log_path = self._job_log_path(job_id)
        entry_with_ts = dict(entry)
        entry_with_ts.setdefault("timestamp", _utcnow())
        line = json.dumps(entry_with_ts)
        await asyncio.to_thread(self._write_log_line, log_path, line)

    @staticmethod
    def _write_log_line(path: Path, line: str) -> None:
        with path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")

    async def get_logs(self, job_id: str, tail: int = 100) -> List[Dict[str, Any]]:
        log_path = self._job_log_path(job_id)
        if not log_path.exists():
            return []
        lines = await asyncio.to_thread(log_path.read_text, "utf-8")
        entries = [json.loads(line) for line in lines.splitlines() if line.strip()]
        if tail > 0:
            entries = entries[-tail:]
        return entries

    async def run_job(self, job_id: str, payload: Dict[str, Any], timeout: int) -> None:
        start = datetime.utcnow()
        try:
            await self._update_job_status(job_id, "running", message="Retraining started.")
            self._increment_metric("counter", status="running")
            dataset = await asyncio.to_thread(
                self.dataset_manager.get_dataset,
                payload["dataset_name"],
                True,
            )
            dataset_content = dataset.get("content", "")
            modelfile = self._render_modelfile(payload, dataset_content)

            job_dir = self._job_dir(job_id)
            modelfile_path = job_dir / "Modelfile"
            await asyncio.to_thread(modelfile_path.write_text, modelfile, "utf-8")
            await self._append_log(job_id, {"event": "modelfile.generated", "path": str(modelfile_path)})

            await self._call_ollama_create(job_id, payload, modelfile, timeout)
            await self._update_job_status(job_id, "completed", message="Retraining complete.")
            self._increment_metric("counter", status="completed")
        except Exception as exc:
            logger.exception("Retraining job %s failed: %s", job_id, exc)
            await self._update_job_status(job_id, "failed", message=str(exc))
            self._increment_metric("counter", status="failed")
        finally:
            duration = (datetime.utcnow() - start).total_seconds()
            self._observe_duration(duration)

    def _render_modelfile(self, payload: Dict[str, Any], dataset_content: str) -> str:
        template = payload.get("modelfile_template")
        if template:
            return template.replace("{{DATASET}}", dataset_content)

        instructions = payload.get("instructions") or "Incorporate the domain knowledge below when answering."
        sanitized_dataset = dataset_content.replace('"""', r'\"\"\"')
        if len(sanitized_dataset) > MAX_DATASET_CHARS:
            raise ValueError(
                "Dataset content is too large for the default template. "
                "Provide a custom modelfile_template with {{DATASET}} placeholder."
            )

        parameters = payload.get("parameters") or {}
        lines = [
            f"FROM {payload['base_model']}",
            "",
            'SYSTEM """',
            textwrap.dedent(instructions).strip(),
            "",
            sanitized_dataset.strip(),
            '"""',
        ]
        for key, value in parameters.items():
            lines.append(f"PARAMETER {key} {value}")

        return "\n".join(lines)

    async def _call_ollama_create(
        self,
        job_id: str,
        payload: Dict[str, Any],
        modelfile: str,
        timeout: int,
    ) -> None:
        url = f"{self.ollama_base_url}/api/create"
        request_payload: Dict[str, Any] = {
            "model": payload["model_name"],
            "modelfile": modelfile,
            "stream": payload.get("stream", False),
        }
        if payload.get("keep_alive"):
            request_payload["keep_alive"] = payload["keep_alive"]

        async with httpx.AsyncClient(timeout=timeout) as client:
            if payload.get("stream", False):
                async with client.stream("POST", url, json=request_payload) as response:
                    response.raise_for_status()
                    async for line in response.aiter_lines():
                        if not line:
                            continue
                        try:
                            parsed = json.loads(line)
                        except json.JSONDecodeError:
                            parsed = {"message": line}
                        await self._append_log(job_id, {"event": "ollama", "data": parsed})
                        if isinstance(parsed, dict) and parsed.get("error"):
                            raise RuntimeError(parsed["error"])
            else:
                resp = await client.post(url, json=request_payload)
                resp.raise_for_status()
                data = resp.json()
                await self._append_log(job_id, {"event": "ollama", "data": data})
                if isinstance(data, dict) and data.get("error"):
                    raise RuntimeError(data["error"])

    async def _update_job_status(self, job_id: str, status: str, message: Optional[str] = None) -> None:
        job = await self.get_job(job_id)
        job["status"] = status
        job["updated_at"] = _utcnow()
        if message:
            job["message"] = message
        await self._store_job(job_id, job)
        await self._append_log(job_id, {"event": "status", "status": status, "message": message})

    def _increment_metric(self, key: str, **labels: Any) -> None:
        metric = self.metrics.get(key)
        if metric:
            metric.labels(**labels).inc()

    def _observe_duration(self, duration: float) -> None:
        metric = self.metrics.get("duration")
        if metric:
            metric.observe(duration)
