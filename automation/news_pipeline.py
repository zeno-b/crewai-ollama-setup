"""
Fetch public news/RSS, normalize into training text or JSONL, merge into a dataset,
and optionally enqueue Ollama retraining when configurable thresholds are met.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import httpx

logger = logging.getLogger(__name__)

_NS_STRIP = re.compile(r"\{[^}]+\}")


def _strip_ns(tag: str) -> str:
    return _NS_STRIP.sub("", tag)


def _text(el: Optional[ET.Element]) -> str:
    if el is None or el.text is None:
        return ""
    return " ".join(el.text.split())


def _child_text(parent: ET.Element, local_names: Tuple[str, ...]) -> str:
    for child in parent:
        base = _strip_ns(child.tag)
        if base in local_names:
            t = (child.text or "").strip()
            if t:
                return t
    return ""


def _parse_rss_or_atom(xml_bytes: bytes) -> List[Dict[str, str]]:
    """Best-effort RSS 2.0 / Atom entry extraction (no external feedparser dependency)."""
    items: List[Dict[str, str]] = []
    try:
        root = ET.fromstring(xml_bytes)
    except ET.ParseError as exc:
        logger.warning("RSS/Atom XML parse failed: %s", exc)
        return []

    root_tag = _strip_ns(root.tag).lower()
    if root_tag == "rss":
        channel = root.find("channel")
        if channel is None:
            for c in root:
                if _strip_ns(c.tag).lower() == "channel":
                    channel = c
                    break
        if channel is None:
            return []
        for node in channel:
            if _strip_ns(node.tag).lower() != "item":
                continue
            title = _child_text(node, ("title",))
            link = _child_text(node, ("link",))
            desc = _child_text(node, ("description", "summary"))
            pub = _child_text(node, ("pubDate", "published", "updated"))
            guid = _child_text(node, ("guid", "id"))
            key = guid or link or title
            if not key:
                continue
            items.append(
                {
                    "title": title,
                    "link": link,
                    "description": desc,
                    "published": pub,
                    "id": key,
                }
            )
        return items

    if root_tag in ("feed",):
        for entry in root:
            if _strip_ns(entry.tag).lower() != "entry":
                continue
            title = _child_text(entry, ("title",))
            link = ""
            for child in entry:
                if _strip_ns(child.tag).lower() == "link" and child.get("href"):
                    link = child.get("href") or ""
                    break
            if not link:
                link = _child_text(entry, ("id",))
            desc = _child_text(entry, ("summary", "content"))
            pub = _child_text(entry, ("updated", "published"))
            uid = _child_text(entry, ("id",))
            key = uid or link or title
            if not key:
                continue
            items.append(
                {
                    "title": title,
                    "link": link,
                    "description": desc,
                    "published": pub,
                    "id": key,
                }
            )
    return items


def _format_items_text(items: List[Dict[str, str]], max_chars: int) -> str:
    blocks: List[str] = []
    size = 0
    for it in items:
        block = (
            f"Title: {it['title']}\n"
            f"Date: {it['published']}\n"
            f"URL: {it['link']}\n"
            f"Summary: {it['description']}\n"
            f"---\n"
        )
        if size + len(block) > max_chars:
            break
        blocks.append(block)
        size += len(block)
    return "\n".join(blocks).strip()


def _format_items_jsonl(items: List[Dict[str, str]], max_chars: int) -> str:
    """One JSON object per line (role/content) for distill MESSAGE expansion."""
    lines: List[str] = []
    size = 0
    for it in items:
        user = f"News: {it['title']}\n{it['description']}".strip()
        assistant = (
            f"Headline: {it['title']}\n"
            f"Published: {it['published']}\n"
            f"Source: {it['link']}"
        ).strip()
        for role, content in (("user", user), ("assistant", assistant)):
            row = {"role": role, "content": content}
            line = json.dumps(row, ensure_ascii=False) + "\n"
            if size + len(line) > max_chars:
                return "\n".join(lines)
            lines.append(line.rstrip("\n"))
            size += len(line)
    return "\n".join(lines)


@dataclass
class NewsAutopilotConfig:
    enabled: bool
    rss_url: str
    poll_interval_seconds: int
    request_timeout_seconds: float
    max_items_per_fetch: int
    output_dataset_name: str
    output_format: str  # text | jsonl
    merge_with_existing: bool
    max_merged_bytes: int
    min_new_lines_to_retrain: int
    min_new_bytes_to_retrain: int
    min_hours_between_retrains: float
    retrain_on_any_change: bool
    base_model: str
    model_name_prefix: str
    job_type: str
    teacher_model: Optional[str]
    template_name: Optional[str]
    instructions: Optional[str]
    retrain_parameters_json: str
    retrain_timeout_seconds: int
    user_agent: str


class NewsAutopilot:
    """Periodic RSS ingestion and optional automatic retraining."""

    def __init__(
        self,
        cfg: NewsAutopilotConfig,
        dataset_manager: Any,
        retraining_manager: Any,
        redis_client: Any,
    ) -> None:
        self.cfg = cfg
        self.dataset_manager = dataset_manager
        self.retraining_manager = retraining_manager
        self.redis = redis_client
        self._task: Optional[asyncio.Task[None]] = None
        self._stop = asyncio.Event()

    def start(self) -> None:
        if self._task and not self._task.done():
            return
        self._stop.clear()
        self._task = asyncio.create_task(self._run_loop(), name="news_autopilot")

    async def stop(self) -> None:
        self._stop.set()
        if self._task:
            try:
                await asyncio.wait_for(self._task, timeout=30)
            except asyncio.TimeoutError:
                self._task.cancel()
            self._task = None

    def _redis_key(self, suffix: str) -> str:
        return f"news_autopilot:{suffix}"

    async def _state_get(self, key: str) -> Optional[str]:
        if not self.redis:
            return None
        try:
            return await self.redis.get(self._redis_key(key))
        except Exception as exc:
            logger.warning("Redis get failed (%s): %s", key, exc)
            return None

    async def _state_set(self, key: str, value: str, ex: int = 86400 * 30) -> None:
        if not self.redis:
            return
        try:
            await self.redis.set(self._redis_key(key), value, ex=ex)
        except Exception as exc:
            logger.warning("Redis set failed (%s): %s", key, exc)

    async def _run_loop(self) -> None:
        logger.info("News autopilot loop started (interval=%ss)", self.cfg.poll_interval_seconds)
        while not self._stop.is_set():
            try:
                await self._tick_once()
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("News autopilot tick failed; will retry after interval")
            try:
                await asyncio.wait_for(self._stop.wait(), timeout=self.cfg.poll_interval_seconds)
            except asyncio.TimeoutError:
                continue
        logger.info("News autopilot loop stopped")

    async def _fetch_feed(self) -> bytes:
        headers = {"User-Agent": self.cfg.user_agent}
        timeout = httpx.Timeout(self.cfg.request_timeout_seconds)
        async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
            resp = await client.get(self.cfg.rss_url, headers=headers)
            resp.raise_for_status()
            return resp.content

    async def _tick_once(self) -> None:
        raw = await self._fetch_feed()
        items = _parse_rss_or_atom(raw)[: self.cfg.max_items_per_fetch]
        if not items:
            logger.info("News autopilot: no items parsed from feed")
            return

        if self.cfg.output_format == "jsonl":
            new_block = _format_items_jsonl(items, self.cfg.max_merged_bytes)
            fmt = "jsonl"
        else:
            new_block = _format_items_text(items, self.cfg.max_merged_bytes)
            fmt = "text"

        merged = new_block
        if self.cfg.merge_with_existing:
            try:
                existing = await asyncio.to_thread(
                    self.dataset_manager.get_dataset,
                    self.cfg.output_dataset_name,
                    True,
                )
                prev = existing.get("content") or ""
                merged = (prev.strip() + "\n\n" + new_block.strip()).strip()
            except FileNotFoundError:
                merged = new_block
            except ValueError as exc:
                logger.warning("Could not load existing dataset for merge: %s", exc)
                merged = new_block

        raw_bytes = merged.encode("utf-8")
        if len(raw_bytes) > self.cfg.max_merged_bytes:
            merged = raw_bytes[: self.cfg.max_merged_bytes].decode("utf-8", errors="ignore")

        digest = hashlib.sha256(merged.encode("utf-8")).hexdigest()
        prev_digest = await self._state_get("last_saved_digest")
        prev_size_s = await self._state_get("last_saved_bytes")
        prev_lines_s = await self._state_get("last_saved_lines")
        prev_size = int(prev_size_s) if prev_size_s and prev_size_s.isdigit() else 0
        prev_lines = int(prev_lines_s) if prev_lines_s and prev_lines_s.isdigit() else 0

        if prev_digest == digest:
            logger.debug("News autopilot: merged corpus unchanged; skipping write")
            return

        encoded = merged.encode("utf-8")
        new_size = len(encoded)
        new_lines = merged.count("\n")
        delta_bytes = new_size - prev_size if prev_digest else new_size
        delta_lines = new_lines - prev_lines if prev_digest else new_lines

        try:
            await asyncio.to_thread(
                self.dataset_manager.save_dataset,
                self.cfg.output_dataset_name,
                merged,
                "Autogenerated from news/RSS",
                ["autopilot", "news"],
                fmt,
                True,
            )
        except ValueError as exc:
            logger.error("Failed to save autopilot dataset: %s", exc)
            return
        except OSError as exc:
            logger.error("Filesystem error saving autopilot dataset: %s", exc)
            return

        await self._state_set("last_saved_digest", digest)
        await self._state_set("last_saved_bytes", str(new_size))
        await self._state_set("last_saved_lines", str(new_lines))
        await self._state_set("last_ingest_at", datetime.now(timezone.utc).isoformat())

        should = self._should_retrain(
            digest_changed=bool(prev_digest) and prev_digest != digest,
            new_lines=delta_lines,
            new_bytes=delta_bytes,
        )
        if not should:
            logger.info(
                "News autopilot: thresholds not met "
                "(delta_lines=%s, delta_bytes=%s); skipping retrain",
                delta_lines,
                delta_bytes,
            )
            return

        last_job = await self._state_get("last_retrain_at")
        if last_job:
            try:
                last_ts = datetime.fromisoformat(last_job.replace("Z", "+00:00"))
                hours = (datetime.now(timezone.utc) - last_ts).total_seconds() / 3600
                if hours < self.cfg.min_hours_between_retrains:
                    logger.info(
                        "News autopilot: min_hours_between_retrains not elapsed (%.2fh < %.2fh)",
                        hours,
                        self.cfg.min_hours_between_retrains,
                    )
                    return
            except ValueError:
                pass

        ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M")
        model_name = f"{self.cfg.model_name_prefix}:{ts}"
        try:
            params = json.loads(self.cfg.retrain_parameters_json or "{}")
            if not isinstance(params, dict):
                raise ValueError("NEWS_AUTOPILOT_RETRAIN_PARAMETERS_JSON must be a JSON object")
        except json.JSONDecodeError as exc:
            logger.error("Invalid NEWS_AUTOPILOT_RETRAIN_PARAMETERS_JSON: %s", exc)
            return

        payload: Dict[str, Any] = {
            "model_name": model_name,
            "base_model": self.cfg.base_model,
            "dataset_name": self.cfg.output_dataset_name,
            "job_type": self.cfg.job_type,
            "teacher_model": self.cfg.teacher_model,
            "template_name": self.cfg.template_name,
            "instructions": self.cfg.instructions,
            "parameters": params,
            "stream": False,
        }
        try:
            job = await self.retraining_manager.create_job(payload)
            job_id = job["job_id"]
            asyncio.create_task(
                self.retraining_manager.run_job(job_id, payload, self.cfg.retrain_timeout_seconds),
                name=f"news_autopilot_retrain_{job_id}",
            )
            await self._state_set("last_retrain_at", datetime.now(timezone.utc).isoformat())
            await self._state_set("last_retrain_job_id", job_id)
            logger.info("News autopilot: scheduled retraining job %s -> %s", job_id, model_name)
        except (ValueError, FileNotFoundError) as exc:
            logger.error("Could not create retraining job: %s", exc)
        except OSError as exc:
            logger.error("Could not persist retraining job: %s", exc)

    def _should_retrain(self, *, digest_changed: bool, new_lines: int, new_bytes: int) -> bool:
        if self.cfg.retrain_on_any_change and digest_changed:
            return True
        if new_lines >= self.cfg.min_new_lines_to_retrain:
            return True
        if new_bytes >= self.cfg.min_new_bytes_to_retrain:
            return True
        return False
