import asyncio
import json
import logging
import re
import textwrap
from datetime import datetime, timezone
from pathlib import Path
from threading import RLock
from typing import Any, Dict, List, Optional, Set
from uuid import uuid4

import httpx

logger = logging.getLogger(__name__)


class RetrainingClientError(Exception):
    """Invalid retraining job request with a preferred HTTP status code."""

    __slots__ = ("message", "status_code")

    def __init__(self, message: str, status_code: int = 400) -> None:
        self.message = message
        self.status_code = status_code
        super().__init__(message)


_NAME_SANITIZER = re.compile(r"[^a-zA-Z0-9_-]+")
MAX_DATASET_CHARS = 120_000

# Optional placeholders in modelfile_template / named templates: {{NAME}}
_PLACEHOLDER = re.compile(r"\{\{([A-Za-z0-9_]+)\}\}")


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sanitize_name(name: str) -> str:
    candidate = _NAME_SANITIZER.sub("-", name.strip()).strip("-").lower()
    if not candidate:
        raise ValueError(
            "Dataset name must contain alphanumeric characters after sanitization."
        )
    return candidate


def _placeholder_names(template: str) -> List[str]:
    return list(dict.fromkeys(_PLACEHOLDER.findall(template)))


def _template_suggests_distillation(template: str) -> bool:
    lowered = template.lower()
    return "message" in lowered and "from" in lowered


def _build_message_dataset_from_jsonl(content: str) -> str:
    """JSONL rows -> Modelfile MESSAGE blocks (distillation-style templates)."""
    blocks: List[str] = []
    for line in content.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        role = (row.get("role") or row.get("from") or "user").strip().lower()
        msg = row.get("content") or row.get("text") or row.get("message") or ""
        if not str(msg).strip():
            continue
        r = "assistant" if role in ("assistant", "model", "bot") else "user"
        text = str(msg).replace('"""', r"\"\"\"")
        blocks.append(f"MESSAGE {r} {text}")
    if not blocks:
        raise ValueError("JSONL dataset contained no usable message rows for MESSAGE blocks.")
    return "\n".join(blocks)


class DatasetManager:
    """Manage retraining datasets stored on disk."""

    def __init__(self, dataset_dir: Path, max_content_bytes: int = 5_242_880):
        self.dataset_dir = dataset_dir
        self.dataset_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_path = self.dataset_dir / "metadata.json"
        self.max_content_bytes = max_content_bytes
        self._lock = RLock()
        self._metadata = self._load_metadata()

    def _load_metadata(self) -> Dict[str, Dict[str, Any]]:
        if self.metadata_path.exists():
            try:
                with self.metadata_path.open("r", encoding="utf-8") as f:
                    data = json.load(f)
                    if isinstance(data, dict):
                        return data
            except OSError as exc:
                logger.warning("Failed to load dataset metadata: %s", exc)
        return {}

    def _persist_metadata(self) -> None:
        tmp_path = self.metadata_path.with_suffix(".tmp")
        try:
            with tmp_path.open("w", encoding="utf-8") as f:
                json.dump(self._metadata, f, indent=2)
            tmp_path.replace(self.metadata_path)
        except OSError:
            if tmp_path.exists():
                try:
                    tmp_path.unlink(missing_ok=True)
                except OSError:
                    pass
            raise

    def save_dataset(
        self,
        name: str,
        content: str,
        description: Optional[str],
        tags: List[str],
        fmt: str,
        overwrite: bool = False,
    ) -> Dict[str, Any]:
        encoded = content.encode("utf-8")
        if len(encoded) > self.max_content_bytes:
            raise ValueError(
                f"Dataset content exceeds maximum size ({self.max_content_bytes} bytes)."
            )

        with self._lock:
            dataset_id = _sanitize_name(name)
            extension = ".jsonl" if fmt == "jsonl" else ".txt"
            dataset_path = self.dataset_dir / f"{dataset_id}{extension}"

            if dataset_id in self._metadata and not overwrite:
                raise ValueError(
                    f"Dataset '{dataset_id}' already exists. Use overwrite flag to replace it."
                )

            try:
                with dataset_path.open("w", encoding="utf-8") as f:
                    f.write(content)
            except OSError as exc:
                logger.error("Failed to write dataset file %s: %s", dataset_path, exc)
                raise

            try:
                size_bytes = dataset_path.stat().st_size
            except OSError as exc:
                logger.error("Failed to stat dataset file %s: %s", dataset_path, exc)
                raise
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
            try:
                self._persist_metadata()
            except OSError as exc:
                logger.error(
                    "Dataset %s was written but metadata persist failed: %s",
                    dataset_id,
                    exc,
                )
                raise RuntimeError(
                    f"Dataset '{dataset_id}' file was saved but metadata could not be updated."
                ) from exc

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
    """Ollama retraining jobs: system-prompt style and distillation-oriented Modelfiles."""

    def __init__(
        self,
        base_dir: Path,
        dataset_manager: DatasetManager,
        redis_client: Optional[Any],
        result_ttl: int,
        ollama_base_url: str,
        metrics: Optional[Dict[str, Any]] = None,
        modelfile_template_dir: Optional[Path] = None,
    ):
        self.base_dir = base_dir
        self.dataset_manager = dataset_manager
        self.redis = redis_client
        self.result_ttl = result_ttl
        self.ollama_base_url = ollama_base_url.rstrip("/")
        self.jobs_dir = self.base_dir / "jobs"
        self.jobs_dir.mkdir(parents=True, exist_ok=True)
        self.metrics = metrics or {}
        self.modelfile_template_dir = (
            Path(modelfile_template_dir) if modelfile_template_dir else None
        )
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
            "job_type": payload.get("job_type", "system_prompt"),
            "model_name": payload["model_name"],
            "base_model": payload["base_model"],
            "dataset_name": payload["dataset_name"],
            "teacher_model": payload.get("teacher_model"),
            "template_name": payload.get("template_name"),
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
        paths = sorted(
            self.jobs_dir.glob("*/status.json"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        for job_path in paths:
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
        entries: List[Dict[str, Any]] = []
        for line in lines.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                parsed = json.loads(line)
                if isinstance(parsed, dict):
                    entries.append(parsed)
                else:
                    entries.append(
                        {
                            "event": "log_parse_skipped",
                            "reason": "JSON root was not an object",
                            "preview": line[:500],
                        }
                    )
            except json.JSONDecodeError as exc:
                logger.warning("Skipping malformed retraining log line for %s: %s", job_id, exc)
                entries.append(
                    {
                        "event": "log_parse_error",
                        "reason": f"Invalid JSON on retraining log line: {exc}",
                        "preview": line[:500],
                    }
                )
        if tail > 0:
            entries = entries[-tail:]
        return entries

    def validate_new_job_payload(self, payload: Dict[str, Any]) -> None:
        """Raise before a job is queued so clients get immediate, specific errors."""
        template_name = (payload.get("template_name") or "").strip()
        if template_name:
            if not self.modelfile_template_dir:
                raise ValueError(
                    "template_name was set but MODELFILE_TEMPLATE_DIR is not configured on the server."
                )
            self._load_named_template(template_name)

        job_type = (payload.get("job_type") or "system_prompt").lower()
        if job_type != "distill":
            return

        teacher = (payload.get("teacher_model") or "").strip()
        if not teacher:
            raise RetrainingClientError(
                "teacher_model is required when job_type is distill.",
                status_code=422,
            )
        has_inline = bool((payload.get("modelfile_template") or "").strip())
        if not has_inline and not template_name:
            raise ValueError(
                "Distillation jobs require modelfile_template or template_name "
                "(MESSAGE-style Modelfile; see README)."
            )

    def _load_named_template(self, name: str) -> str:
        if not self.modelfile_template_dir:
            raise ValueError(
                "template_name was set but MODELFILE_TEMPLATE_DIR is not configured on the server."
            )
        safe = _sanitize_name(name)
        path = self.modelfile_template_dir / f"{safe}.template"
        if not path.is_file():
            raise FileNotFoundError(f"Named Modelfile template not found: {path.name}")
        resolved = path.resolve()
        root = self.modelfile_template_dir.resolve()
        try:
            resolved.relative_to(root)
        except ValueError as exc:
            raise ValueError("Invalid template path.") from exc
        return path.read_text(encoding="utf-8")

    def _dataset_block_for_template(
        self,
        job_type: str,
        dataset_format: str,
        dataset_content: str,
    ) -> str:
        jt = (job_type or "system_prompt").lower()
        if jt == "distill" and dataset_format == "jsonl":
            return _build_message_dataset_from_jsonl(dataset_content)
        return dataset_content.replace('"""', r"\"\"\"")

    def _fill_placeholders(
        self,
        template: str,
        mapping: Dict[str, str],
        *,
        allowed_extra: Optional[Set[str]] = None,
    ) -> str:
        names = set(_placeholder_names(template))
        missing = [n for n in sorted(names) if n not in mapping]
        if missing:
            raise ValueError(f"Missing values for Modelfile placeholders: {', '.join(missing)}")
        if allowed_extra is not None:
            unknown = set(mapping) - allowed_extra - names
            if unknown:
                keys = ", ".join(sorted(unknown))
                raise ValueError(f"Unexpected placeholder keys supplied: {keys}")
        out = template
        for key, val in mapping.items():
            out = out.replace("{{" + key + "}}", val)
        return out

    def _render_modelfile(self, payload: Dict[str, Any], dataset: Dict[str, Any]) -> str:
        job_type = (payload.get("job_type") or "system_prompt").lower()
        teacher = (payload.get("teacher_model") or "").strip()
        template_name = (payload.get("template_name") or "").strip()
        template = payload.get("modelfile_template")
        if template_name:
            template = self._load_named_template(template_name)

        dataset_content = dataset.get("content", "")
        dataset_format = str(dataset.get("format", "text"))
        base_model = str(payload["base_model"])
        model_name = str(payload["model_name"])
        instructions = textwrap.dedent(payload.get("instructions") or "").strip() or (
            "Incorporate the domain knowledge below when answering."
        )

        if job_type == "distill":
            if not teacher:
                raise ValueError("teacher_model is required when job_type is 'distill'.")
            if not template:
                raise ValueError(
                    "Distillation jobs require modelfile_template or template_name "
                    "(e.g. MESSAGE user / MESSAGE assistant blocks referencing a teacher)."
                )
            if not _template_suggests_distillation(template):
                logger.warning(
                    "Distillation job_type set but template does not look like MESSAGE-based; "
                    "ensure it uses Ollama MESSAGE blocks for student/teacher prompts."
                )

        if template:
            dataset_block = self._dataset_block_for_template(
                job_type,
                dataset_format,
                dataset_content,
            )
            if len(dataset_block) > MAX_DATASET_CHARS and "{{DATASET}}" in template:
                raise ValueError(
                    "Dataset content is too large for this template. "
                    "Trim the dataset or use a smaller excerpt."
                )
            mapping = {
                "DATASET": dataset_block,
                "BASE_MODEL": base_model,
                "MODEL_NAME": model_name,
                "TEACHER_MODEL": teacher,
                "INSTRUCTIONS": instructions,
                "ADAPTER": str(payload.get("adapter_path") or "").strip(),
            }
            allowed = {
                "DATASET",
                "BASE_MODEL",
                "MODEL_NAME",
                "TEACHER_MODEL",
                "INSTRUCTIONS",
                "ADAPTER",
            }
            filled = self._fill_placeholders(template, mapping, allowed_extra=allowed)
            nonempty = [ln for ln in filled.splitlines() if ln.strip()]
            has_from = bool(nonempty) and nonempty[0].strip().upper().startswith("FROM")
            lines = [filled] if has_from else [f"FROM {base_model}", "", filled]
            parameters = payload.get("parameters") or {}
            for key, value in parameters.items():
                lines.append(f"PARAMETER {key} {value}")
            return "\n".join(lines)

        sanitized_dataset = dataset_content.replace('"""', r"\"\"\"")
        if len(sanitized_dataset) > MAX_DATASET_CHARS:
            raise ValueError(
                "Dataset content is too large for the default template. "
                "Provide a custom modelfile_template with {{DATASET}} placeholder "
                "or raise DATASET_MAX_CONTENT_BYTES / split the dataset."
            )

        parameters = payload.get("parameters") or {}
        lines = [
            f"FROM {base_model}",
            "",
            'SYSTEM """',
            instructions,
            "",
            sanitized_dataset.strip(),
            '"""',
        ]
        for key, value in parameters.items():
            lines.append(f"PARAMETER {key} {value}")

        return "\n".join(lines)

    async def run_job(self, job_id: str, payload: Dict[str, Any], timeout: int) -> None:
        start = datetime.now(timezone.utc)
        try:
            await self._update_job_status(job_id, "running", message="Retraining started.")
            self._increment_metric("counter", status="running")
            dataset = await asyncio.to_thread(
                self.dataset_manager.get_dataset,
                payload["dataset_name"],
                True,
            )
            modelfile = self._render_modelfile(payload, dataset)

            job_dir = self._job_dir(job_id)
            modelfile_path = job_dir / "Modelfile"
            await asyncio.to_thread(modelfile_path.write_text, modelfile, "utf-8")
            await self._append_log(
                job_id,
                {"event": "modelfile.generated", "path": str(modelfile_path)},
            )

            await self._call_ollama_create(job_id, payload, modelfile, timeout)
            await self._update_job_status(job_id, "completed", message="Retraining complete.")
            self._increment_metric("counter", status="completed")
        except Exception as exc:
            logger.exception("Retraining job %s failed: %s", job_id, exc)
            await self._update_job_status(job_id, "failed", message=str(exc))
            self._increment_metric("counter", status="failed")
        finally:
            duration = (datetime.now(timezone.utc) - start).total_seconds()
            self._observe_duration(duration)

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

    async def _update_job_status(
        self,
        job_id: str,
        status: str,
        message: Optional[str] = None,
    ) -> None:
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
