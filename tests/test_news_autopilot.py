"""Unit tests for RSS parsing and JSONL formatting (no network)."""

from __future__ import annotations

import json

from automation.news_pipeline import _format_items_jsonl, _format_items_text, _parse_rss_or_atom


def test_parse_rss_basic():
    xml = b"""<?xml version="1.0"?>
    <rss version="2.0"><channel>
      <item><title>Hello</title><link>http://x</link>
      <description>World</description><pubDate>Mon, 01 Jan 2024 00:00:00 GMT</pubDate></item>
    </channel></rss>"""
    items = _parse_rss_or_atom(xml)
    assert len(items) == 1
    assert items[0]["title"] == "Hello"
    assert "World" in items[0]["description"]


def test_format_jsonl_two_lines_per_item():
    items = [
        {"title": "T", "link": "L", "description": "D", "published": "P", "id": "1"},
    ]
    out = _format_items_jsonl(items, max_chars=10_000)
    lines = [ln for ln in out.splitlines() if ln.strip()]
    assert len(lines) == 2
    u = json.loads(lines[0])
    a = json.loads(lines[1])
    assert u["role"] == "user"
    assert a["role"] == "assistant"


def test_format_text_respects_max_chars():
    items = [
        {"title": "A" * 100, "link": "", "description": "B" * 100, "published": "", "id": "1"},
    ]
    out = _format_items_text(items, max_chars=50)
    assert len(out) <= 50
