"""Retraining autopilot: RSS parsing, dedupe, finetune decision."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from retraining.autopilot import (
    AutopilotStatusStore,
    FingerprintStore,
    NewsItem,
    RetrainingAutopilot,
    _parse_feed,
)


def test_parse_rss_minimal():
    xml = """<?xml version="1.0"?>
    <rss><channel>
      <item><title>A</title><link>http://a</link><description>Body</description><pubDate>Mon, 1 Jan 2024</pubDate></item>
    </channel></rss>"""
    items = _parse_feed(xml, 10)
    assert len(items) == 1
    assert items[0].title == "A"
    assert items[0].link == "http://a"
    assert "Body" in items[0].summary


def test_parse_atom_minimal():
    xml = """<?xml version="1.0"?>
    <feed xmlns="http://www.w3.org/2005/Atom">
      <entry>
        <title>T</title>
        <link href="http://b"/>
        <summary>S</summary>
        <updated>2024-01-01</updated>
      </entry>
    </feed>"""
    items = _parse_feed(xml, 10)
    assert len(items) == 1
    assert items[0].link == "http://b"


def test_news_item_training_block():
    n = NewsItem(title="Hi", link="http://x", summary="S" * 200, published="p")
    block = n.to_training_block()
    assert "### Hi" in block
    assert "http://x" in block


@pytest.mark.asyncio
async def test_autopilot_run_cycle_skips_when_disabled(tmp_path: Path):
    settings = SimpleNamespace(autopilot_enabled=False)
    ap = RetrainingAutopilot(
        settings=settings,
        dataset_manager=None,
        job_manager=None,
        fingerprint_store=FingerprintStore(tmp_path / "fp.json"),
        status_store=AutopilotStatusStore(tmp_path / "st.json"),
    )
    out = await ap.run_cycle()
    assert out.get("skipped") is True


@pytest.mark.asyncio
async def test_autopilot_appends_and_marks_fingerprints(tmp_path: Path):
    from retraining.manager import DatasetManager

    ds_dir = tmp_path / "ds"
    dm = DatasetManager(ds_dir, max_content_bytes=1_000_000)

    job_mgr = SimpleNamespace()
    job_mgr.validate_new_job_payload = lambda p: None
    job_mgr.create_job = AsyncMock(
        return_value={"job_id": "job_test123456789012345678901234"}
    )
    job_mgr.run_job = AsyncMock()

    settings = SimpleNamespace(
        autopilot_enabled=True,
        autopilot_news_feeds="",
        autopilot_max_items_per_feed=5,
        autopilot_http_timeout_seconds=10,
        autopilot_max_feed_bytes=100_000,
        autopilot_dataset_name="roll",
        autopilot_auto_finetune=False,
        autopilot_min_new_items_to_finetune=99,
        autopilot_finetune_cooldown_seconds=3600,
        autopilot_base_model="base",
        autopilot_output_model_template="m-{date}",
        autopilot_system_instructions="sys",
        autopilot_job_timeout_seconds=120,
        autopilot_fingerprint_max_entries=1000,
        autopilot_fingerprint_ttl_seconds=0,
        result_ttl=3600,
        autopilot_state_dir=str(tmp_path / "state"),
    )

    xml = """<rss><channel>
      <item><title>X</title><link>http://u</link><description>d</description></item>
    </channel></rss>"""

    fp_store = FingerprintStore(tmp_path / "fp.json", redis=None)
    ap = RetrainingAutopilot(
        settings=settings,
        dataset_manager=dm,
        job_manager=job_mgr,
        fingerprint_store=fp_store,
        status_store=AutopilotStatusStore(tmp_path / "st.json"),
    )

    with patch("retraining.autopilot._fetch_url", return_value=xml):
        settings.autopilot_news_feeds = "http://example.com/feed.xml"
        r1 = await ap.run_cycle()
    assert r1["new_items"] == 1
    rec = dm.get_dataset("roll", True)
    assert "http://u" in rec["content"]

    with patch("retraining.autopilot._fetch_url", return_value=xml):
        r2 = await ap.run_cycle()
    assert r2["new_items"] == 0


@pytest.mark.asyncio
async def test_autopilot_triggers_finetune_when_threshold_met(tmp_path: Path):
    from retraining.manager import DatasetManager

    ds_dir = tmp_path / "ds2"
    dm = DatasetManager(ds_dir, max_content_bytes=2_000_000)

    job_mgr = SimpleNamespace()
    job_mgr.validate_new_job_payload = lambda p: None
    job_mgr.create_job = AsyncMock(
        return_value={"job_id": "job_abcdefabcdefabcdefabcdefabcdef"}
    )
    job_mgr.run_job = AsyncMock()

    state_sub = tmp_path / "state2"
    state_sub.mkdir()
    last_ft = state_sub / "last_finetune.json"
    last_ft.write_text(json.dumps({"ts": 0.0}), encoding="utf-8")

    settings = SimpleNamespace(
        autopilot_enabled=True,
        autopilot_news_feeds="http://f",
        autopilot_max_items_per_feed=10,
        autopilot_http_timeout_seconds=10,
        autopilot_max_feed_bytes=200_000,
        autopilot_dataset_name="roll2",
        autopilot_auto_finetune=True,
        autopilot_min_new_items_to_finetune=2,
        autopilot_finetune_cooldown_seconds=1,
        autopilot_base_model="base",
        autopilot_output_model_template="ft-{ts}",
        autopilot_system_instructions="sys",
        autopilot_job_timeout_seconds=120,
        autopilot_fingerprint_max_entries=5000,
        autopilot_fingerprint_ttl_seconds=0,
        result_ttl=3600,
        autopilot_state_dir=str(state_sub),
    )

    xml = """<rss><channel>
      <item><title>a</title><link>http://1</link><description>d1</description></item>
      <item><title>b</title><link>http://2</link><description>d2</description></item>
    </channel></rss>"""

    ap = RetrainingAutopilot(
        settings=settings,
        dataset_manager=dm,
        job_manager=job_mgr,
        fingerprint_store=FingerprintStore(tmp_path / "fp2.json"),
        status_store=AutopilotStatusStore(tmp_path / "st2.json"),
    )

    with patch("retraining.autopilot._fetch_url", return_value=xml):
        r = await ap.run_cycle()
    await asyncio.sleep(0)
    assert r["new_items"] == 2
    assert r.get("triggered_finetune") is True
    assert r.get("job_id")
    job_mgr.create_job.assert_awaited_once()
    job_mgr.run_job.assert_awaited()
