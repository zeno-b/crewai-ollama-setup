"""
Background news ingestion, dataset formatting, and autonomous finetune triggers.

Uses urllib for HTTP fetches so test suites that monkeypatch httpx.AsyncClient
remain isolated. All thresholds and URLs are driven by Settings.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import re
import textwrap
import time
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from threading import RLock
from typing import Any, Callable, Dict, List, Optional, Set
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

_NS_STRIP = re.compile(r"\{[^}]+\}")


def _strip_tag(tag: str) -> str:
    return _NS_STRIP.sub("", tag)


def _elem_text(el: Optional[ET.Element]) -> str:
    if el is None:
        return ""
    return " ".join(el.itertext()).strip()


def _find_child(parent: ET.Element, *local_names: str) -> Optional[ET.Element]:
    for child in parent:
        base = _strip_tag(child.tag)
        if base in local_names:
            return child
    return None


def _find_all_children(parent: ET.Element, local_name: str) -> List[ET.Element]:
    out: List[ET.Element] = []
    for child in parent:
        if _strip_tag(child.tag) == local_name:
            out.append(child)
    return out


def _atom_link_href(entry: ET.Element) -> str:
    for child in entry:
        if _strip_tag(child.tag) != "link":
            continue
        href = child.attrib.get("href") or ""
        if href:
            return href.strip()
    return ""


@dataclass
class NewsItem:
    title: str
    link: str
    summary: str
    published: str

    def fingerprint(self) -> str:
        base = (self.link or "").strip() or f"{self.title}|{self.summary}"
        return hashlib.sha256(base.encode("utf-8", errors="replace")).hexdigest()

    def to_training_block(self) -> str:
        title = self.title.replace("\n", " ").strip() or "(untitled)"
        link = self.link.strip()
        pub = self.published.strip() or "unknown"
        body = textwrap.fill(self.summary.strip() or "(no summary)", width=100)
        lines = [
            f"### {title}",
            f"Published: {pub}",
        ]
        if link:
            lines.append(f"URL: {link}")
        lines.append(body)
        lines.append("---")
        return "\n".join(lines)


class FingerprintStore:
    """Deduplicate ingested items across restarts (disk + optional Redis cache)."""

    def __init__(self, path: Path, redis: Optional[Any] = None, key_prefix: str = "autopilot:fp:"):
        self.path = path
        self.redis = redis
        self.key_prefix = key_prefix
        self._lock = RLock()
        self._seen: Dict[str, float] = {}
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            raw = self.path.read_text(encoding="utf-8")
            data = json.loads(raw)
            if isinstance(data, dict):
                for k, v in data.items():
                    if isinstance(k, str) and isinstance(v, (int, float)):
                        self._seen[k] = float(v)
        except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
            logger.warning("Could not load autopilot fingerprint store %s: %s", self.path, exc)

    def _persist(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        with tmp.open("w", encoding="utf-8") as f:
            json.dump(self._seen, f, indent=2)
        tmp.replace(self.path)

    def prune(self, max_entries: int, ttl_seconds: float) -> None:
        if max_entries <= 0 and ttl_seconds <= 0:
            return
        now = time.time()
        with self._lock:
            if ttl_seconds > 0:
                cutoff = now - ttl_seconds
                self._seen = {k: v for k, v in self._seen.items() if v >= cutoff}
            if max_entries > 0 and len(self._seen) > max_entries:
                sorted_keys = sorted(self._seen, key=lambda k: self._seen[k], reverse=True)
                for drop in sorted_keys[max_entries:]:
                    self._seen.pop(drop, None)

    def mark_sync(self, fp: str) -> None:
        with self._lock:
            self._seen[fp] = time.time()
            try:
                self._persist()
            except OSError as exc:
                logger.error("Failed to persist fingerprint store: %s", exc)

    async def has_async(self, fp: str) -> bool:
        if self.redis:
            try:
                key = self.key_prefix + fp
                exists = await self.redis.exists(key)
                if exists:
                    return True
            except Exception as exc:  # noqa: BLE001
                logger.warning("Redis fingerprint check failed: %s", exc)
        with self._lock:
            return fp in self._seen

    async def mark_async(self, fp: str, ttl: int) -> None:
        if self.redis:
            try:
                await self.redis.set(self.key_prefix + fp, "1", ex=max(ttl, 60))
            except Exception as exc:  # noqa: BLE001
                logger.warning("Redis fingerprint set failed: %s", exc)
        self.mark_sync(fp)


@dataclass
class AutopilotState:
    last_run_started: Optional[str] = None
    last_run_finished: Optional[str] = None
    last_error: Optional[str] = None
    last_ingested_count: int = 0
    last_new_count: int = 0
    last_trigger: bool = False
    last_job_id: Optional[str] = None
    total_cycles: int = 0
    total_items_ingested: int = 0
    total_finetune_jobs: int = 0


class AutopilotStatusStore:
    def __init__(self, path: Path):
        self.path = path
        self._lock = RLock()
        self.state = AutopilotState()

    def update(self, **kwargs: Any) -> None:
        with self._lock:
            for k, v in kwargs.items():
                if hasattr(self.state, k):
                    setattr(self.state, k, v)
            self._write_unlocked()

    def _write_unlocked(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {k: v for k, v in self.state.__dict__.items()}
        tmp = self.path.with_suffix(".tmp")
        try:
            with tmp.open("w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
            tmp.replace(self.path)
        except OSError as exc:
            logger.error("Failed to write autopilot status: %s", exc)

    def read_dict(self) -> Dict[str, Any]:
        with self._lock:
            return {k: v for k, v in self.state.__dict__.items()}


def _fetch_url(url: str, timeout: float, max_bytes: int) -> str:
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise ValueError(f"Unsupported URL scheme for news feed: {parsed.scheme!r}")
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "CrewAI-Autopilot/1.0 (+https://github.com)"},
        method="GET",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 - URL validated
            raw = resp.read(max_bytes + 1)
    except urllib.error.HTTPError as exc:
        body = ""
        try:
            body = exc.read(512).decode("utf-8", errors="replace")
        except OSError:
            body = ""
        raise RuntimeError(f"HTTP {exc.code} fetching feed: {exc.reason}. Body preview: {body!r}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Network error fetching feed: {exc!s}") from exc
    if len(raw) > max_bytes:
        raise ValueError(f"Feed response exceeded max_bytes={max_bytes}")
    return raw.decode("utf-8", errors="replace")


def _parse_feed(xml_text: str, max_items: int) -> List[NewsItem]:
    items: List[NewsItem] = []
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as exc:
        raise ValueError(f"Invalid XML in feed: {exc}") from exc

    root_tag = _strip_tag(root.tag).lower()
    if root_tag == "rss":
        channel = _find_child(root, "channel")
        if channel is None:
            raise ValueError("RSS feed missing channel element.")
        for node in _find_all_children(channel, "item")[:max_items]:
            title = _elem_text(_find_child(node, "title"))
            link = _elem_text(_find_child(node, "link"))
            summary = _elem_text(_find_child(node, "description")) or _elem_text(
                _find_child(node, "summary")
            )
            pub = _elem_text(_find_child(node, "pubDate")) or _elem_text(_find_child(node, "date"))
            items.append(NewsItem(title=title, link=link, summary=summary, published=pub))
    elif root_tag in ("feed",):
        for entry in _find_all_children(root, "entry")[:max_items]:
            title = _elem_text(_find_child(entry, "title"))
            link = _atom_link_href(entry)
            summary = _elem_text(_find_child(entry, "summary")) or _elem_text(
                _find_child(entry, "content")
            )
            pub = _elem_text(_find_child(entry, "updated")) or _elem_text(_find_child(entry, "published"))
            items.append(NewsItem(title=title, link=link, summary=summary, published=pub))
    else:
        raise ValueError(f"Unsupported feed root element: {root_tag!r}")

    return items


class RetrainingAutopilot:
    """
    Polls RSS/Atom feeds, appends deduplicated formatted text to a rolling dataset,
    and when thresholds are met queues an Ollama Modelfile create (finetune) job.
    """

    def __init__(
        self,
        *,
        settings: Any,
        dataset_manager: Any,
        job_manager: Any,
        fingerprint_store: FingerprintStore,
        status_store: AutopilotStatusStore,
        utcnow: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
    ) -> None:
        self.settings = settings
        self.dataset_manager = dataset_manager
        self.job_manager = job_manager
        self.fingerprints = fingerprint_store
        self.status = status_store
        self._utcnow = utcnow
        self._task: Optional[asyncio.Task] = None

    def start_background(self) -> None:
        if self._task and not self._task.done():
            return
        self._task = asyncio.create_task(self._run_forever())

    async def stop(self) -> None:
        import asyncio

        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        self._task = None

    async def _run_forever(self) -> None:
        import asyncio

        interval = max(30, int(self.settings.autopilot_poll_interval_seconds))
        while True:
            try:
                await self.run_cycle()
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001
                logger.exception("Autopilot cycle crashed: %s", exc)
                self.status.update(last_error=f"{type(exc).__name__}: {exc}")
            await asyncio.sleep(interval)

    async def run_cycle(self) -> Dict[str, Any]:
        import asyncio

        if not self.settings.autopilot_enabled:
            return {"skipped": True, "reason": "disabled"}

        self.status.update(
            last_run_started=self._utcnow().isoformat(),
            last_error=None,
        )
        feeds = [f.strip() for f in self.settings.autopilot_news_feeds.split(",") if f.strip()]
        if not feeds:
            self.status.update(
                last_run_finished=self._utcnow().isoformat(),
                last_error="No AUTOPILOT_NEWS_FEEDS configured.",
                last_ingested_count=0,
                last_new_count=0,
            )
            return {"error": "no_feeds"}

        max_items = max(1, self.settings.autopilot_max_items_per_feed)
        timeout = float(self.settings.autopilot_http_timeout_seconds)
        max_bytes = max(4096, self.settings.autopilot_max_feed_bytes)
        all_items: List[NewsItem] = []
        errors: List[str] = []

        for url in feeds:
            try:
                xml_text = await asyncio.to_thread(_fetch_url, url, timeout, max_bytes)
                parsed = await asyncio.to_thread(_parse_feed, xml_text, max_items)
                all_items.extend(parsed)
            except (RuntimeError, ValueError, OSError) as exc:
                msg = f"{url}: {type(exc).__name__}: {exc}"
                logger.warning("Feed ingest failed: %s", msg)
                errors.append(msg)

        new_blocks: List[str] = []
        fps_pending: List[str] = []
        dedupe_cycle: Set[str] = set()
        for item in all_items:
            fp = item.fingerprint()
            if fp in dedupe_cycle:
                continue
            dedupe_cycle.add(fp)
            try:
                seen = await self.fingerprints.has_async(fp)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Fingerprint check failed, assuming new: %s", exc)
                seen = False
            if seen:
                continue
            new_blocks.append(item.to_training_block())
            fps_pending.append(fp)

        new_count = len(new_blocks)
        self.fingerprints.prune(
            self.settings.autopilot_fingerprint_max_entries,
            float(self.settings.autopilot_fingerprint_ttl_seconds),
        )

        dataset_name = self.settings.autopilot_dataset_name.strip() or "news-rolling"
        appended = 0
        if new_blocks:
            try:
                appended = await asyncio.to_thread(self._append_to_dataset, dataset_name, new_blocks)
                for fp in fps_pending:
                    try:
                        await self.fingerprints.mark_async(fp, self.settings.result_ttl)
                    except Exception as exc:  # noqa: BLE001
                        logger.warning("Fingerprint mark failed after append: %s", exc)
            except (ValueError, FileNotFoundError, OSError) as exc:
                err = f"Dataset append failed: {type(exc).__name__}: {exc}"
                logger.error(err)
                errors.append(err)
                new_count = 0

        trigger = self._should_trigger_finetune(new_count)
        job_id: Optional[str] = None
        if trigger and new_count > 0:
            try:
                job_id = await self._queue_finetune(dataset_name)
            except (ValueError, FileNotFoundError, OSError, RuntimeError) as exc:
                err = f"Finetune queue failed: {type(exc).__name__}: {exc}"
                logger.error(err)
                errors.append(err)
                trigger = False

        self.status.state.total_cycles += 1
        self.status.state.total_items_ingested += new_count
        if trigger and job_id:
            self.status.state.total_finetune_jobs += 1

        self.status.update(
            last_run_finished=self._utcnow().isoformat(),
            last_ingested_count=len(all_items),
            last_new_count=new_count,
            last_trigger=bool(trigger and job_id),
            last_job_id=job_id,
            last_error="; ".join(errors) if errors else None,
        )

        return {
            "ingested": len(all_items),
            "new_items": new_count,
            "appended_chars": appended,
            "triggered_finetune": bool(trigger and job_id),
            "job_id": job_id,
            "errors": errors,
        }

    def _append_to_dataset(self, dataset_name: str, blocks: List[str]) -> int:
        chunk = "\n\n".join(blocks).strip()
        if not chunk:
            return 0
        try:
            existing = self.dataset_manager.get_dataset(dataset_name, True)
            prior = str(existing.get("content") or "")
            fmt = str(existing.get("format") or "text")
        except FileNotFoundError:
            prior = ""
            fmt = "text"
        merged = (prior.rstrip() + "\n\n" + chunk).strip() if prior else chunk
        self.dataset_manager.save_dataset(
            dataset_name,
            merged,
            description="Rolling news corpus (autopilot)",
            tags=["autopilot", "news"],
            fmt=fmt if fmt in ("text", "jsonl") else "text",
            overwrite=True,
        )
        return len(chunk.encode("utf-8"))

    def _last_finetune_ts(self) -> float:
        p = Path(self.settings.autopilot_state_dir) / "last_finetune.json"
        if not p.exists():
            return 0.0
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                return float(data.get("ts") or 0.0)
        except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
            logger.warning("Could not read last finetune timestamp: %s", exc)
        return 0.0

    def _record_finetune_ts(self) -> None:
        p = Path(self.settings.autopilot_state_dir) / "last_finetune.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".tmp")
        try:
            with tmp.open("w", encoding="utf-8") as f:
                json.dump({"ts": time.time(), "iso": self._utcnow().isoformat()}, f, indent=2)
            tmp.replace(p)
        except OSError as exc:
            logger.error("Failed to record finetune timestamp: %s", exc)

    def _should_trigger_finetune(self, new_count: int) -> bool:
        if not self.settings.autopilot_auto_finetune:
            return False
        if new_count < max(1, self.settings.autopilot_min_new_items_to_finetune):
            return False
        cooldown = max(60, self.settings.autopilot_finetune_cooldown_seconds)
        if time.time() - self._last_finetune_ts() < cooldown:
            return False
        return True

    async def _queue_finetune(self, dataset_name: str) -> str:
        import asyncio

        await asyncio.to_thread(self.dataset_manager.get_dataset, dataset_name, False)
        model_name = self._render_model_name()
        payload = {
            "model_name": model_name,
            "base_model": self.settings.autopilot_base_model,
            "dataset_name": dataset_name,
            "job_type": "system_prompt",
            "instructions": self.settings.autopilot_system_instructions,
            "parameters": {},
            "stream": False,
            "timeout": max(60, self.settings.autopilot_job_timeout_seconds),
        }
        await asyncio.to_thread(self.job_manager.validate_new_job_payload, payload)
        job = await self.job_manager.create_job(payload)
        job_id = str(job["job_id"])
        asyncio.create_task(
            self.job_manager.run_job(job_id, payload, int(payload["timeout"])),
        )
        self._record_finetune_ts()
        return job_id

    def _render_model_name(self) -> str:
        template = self.settings.autopilot_output_model_template.strip() or "crew-news-{date}"
        now = self._utcnow()
        return template.format(
            date=now.strftime("%Y%m%d"),
            time=now.strftime("%H%M%S"),
            ts=str(int(now.timestamp())),
        )
