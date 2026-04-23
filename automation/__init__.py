"""Background automation (news ingestion, auto-retrain scheduling)."""

from .news_pipeline import NewsAutopilot, NewsAutopilotConfig

__all__ = ["NewsAutopilot", "NewsAutopilotConfig"]
