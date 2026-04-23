import os
import secrets
from typing import List, Optional

from pydantic import AliasChoices, Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


def _default_secret_key() -> str:
    """Ephemeral default for local/testing only; production must set SECRET_KEY."""
    return secrets.token_urlsafe(48)


class Settings(BaseSettings):
    """Application settings loaded from environment and optional .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # API
    api_title: str = Field(default="CrewAI Service", validation_alias="API_TITLE")
    api_version: str = Field(default="1.0.0", validation_alias="API_VERSION")
    api_debug: bool = Field(default=False, validation_alias="API_DEBUG")

    # Server
    host: str = Field(default="0.0.0.0", validation_alias="HOST")
    port: int = Field(default=8000, validation_alias="PORT")
    reload: bool = Field(default=False, validation_alias="RELOAD")

    # Ollama
    ollama_base_url: str = Field(
        default="http://localhost:11434",
        validation_alias="OLLAMA_BASE_URL",
    )
    ollama_model: str = Field(default="llama2:7b", validation_alias="OLLAMA_MODEL")
    ollama_timeout: int = Field(default=300, ge=5, validation_alias="OLLAMA_TIMEOUT")

    # Redis
    redis_url: str = Field(default="redis://localhost:6379", validation_alias="REDIS_URL")
    redis_password: Optional[str] = Field(default=None, validation_alias="REDIS_PASSWORD")
    redis_db: int = Field(default=0, ge=0, validation_alias="REDIS_DB")
    redis_max_connections: int = Field(default=10, ge=1, validation_alias="REDIS_MAX_CONNECTIONS")

    # Security
    secret_key: str = Field(
        default_factory=_default_secret_key,
        min_length=32,
        validation_alias=AliasChoices("SECRET_KEY", "JWT_SECRET", "JWT_SECRET_KEY"),
    )
    algorithm: str = Field(default="HS256", validation_alias="ALGORITHM")
    access_token_expire_minutes: int = Field(
        default=30,
        ge=1,
        validation_alias="ACCESS_TOKEN_EXPIRE_MINUTES",
    )
    api_bearer_token: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices("API_BEARER_TOKEN", "CREWAI_API_TOKEN"),
        description="If set, required as Authorization: Bearer <token> on mutating routes.",
    )
    metrics_bearer_token: Optional[str] = Field(
        default=None,
        validation_alias="METRICS_BEARER_TOKEN",
        description="If set, /metrics requires Authorization: Bearer <token>.",
    )

    # CORS (comma-separated origins; use * for allow-all — not allowed in production)
    cors_allow_origins: str = Field(
        default="http://localhost:3000,http://127.0.0.1:3000",
        validation_alias="CORS_ALLOW_ORIGINS",
    )

    # Logging
    log_level: str = Field(default="INFO", validation_alias="LOG_LEVEL")
    log_format: str = Field(
        default="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        validation_alias="LOG_FORMAT",
    )

    # Monitoring
    metrics_enabled: bool = Field(default=True, validation_alias="METRICS_ENABLED")
    metrics_port: int = Field(default=9090, validation_alias="METRICS_PORT")

    # CrewAI
    max_agents: int = Field(default=10, ge=1, validation_alias="MAX_AGENTS")
    max_tasks: int = Field(default=50, ge=1, validation_alias="MAX_TASKS")
    crew_verbose: bool = Field(default=False, validation_alias="CREW_VERBOSE")

    # Cache / jobs
    cache_ttl: int = Field(default=3600, ge=60, validation_alias="CACHE_TTL")
    result_ttl: int = Field(default=86400, ge=60, validation_alias="RESULT_TTL")

    # Rate limiting (reserved for future middleware)
    rate_limit_per_minute: int = Field(default=60, ge=1, validation_alias="RATE_LIMIT_PER_MINUTE")

    # Datasets / retraining
    dataset_max_content_bytes: int = Field(
        default=5_242_880,
        ge=1024,
        validation_alias="DATASET_MAX_CONTENT_BYTES",
        description="Maximum raw dataset upload size (bytes).",
    )
    modelfile_template_dir: str = Field(
        default="config/modelfiles",
        validation_alias="MODELFILE_TEMPLATE_DIR",
        description="Directory of reusable Modelfile templates (optional).",
    )

    # Autopilot: RSS/Atom news -> rolling dataset -> optional autonomous Ollama finetune
    autopilot_enabled: bool = Field(default=False, validation_alias="AUTOPILOT_ENABLED")
    autopilot_news_feeds: str = Field(
        default="",
        validation_alias="AUTOPILOT_NEWS_FEEDS",
        description="Comma-separated RSS or Atom feed URLs.",
    )
    autopilot_poll_interval_seconds: int = Field(
        default=900,
        ge=30,
        le=86400,
        validation_alias="AUTOPILOT_POLL_INTERVAL_SECONDS",
    )
    autopilot_max_items_per_feed: int = Field(
        default=25,
        ge=1,
        le=500,
        validation_alias="AUTOPILOT_MAX_ITEMS_PER_FEED",
    )
    autopilot_http_timeout_seconds: int = Field(
        default=45,
        ge=5,
        le=600,
        validation_alias="AUTOPILOT_HTTP_TIMEOUT_SECONDS",
    )
    autopilot_max_feed_bytes: int = Field(
        default=8_388_608,
        ge=65536,
        le=50_000_000,
        validation_alias="AUTOPILOT_MAX_FEED_BYTES",
    )
    autopilot_dataset_name: str = Field(
        default="news-rolling",
        validation_alias="AUTOPILOT_DATASET_NAME",
    )
    autopilot_state_dir: str = Field(
        default="data/autopilot",
        validation_alias="AUTOPILOT_STATE_DIR",
        description="Fingerprints, last finetune time, and status JSON.",
    )
    autopilot_auto_finetune: bool = Field(
        default=False,
        validation_alias="AUTOPILOT_AUTO_FINETUNE",
        description="When true, queue Ollama create after enough new items (see thresholds).",
    )
    autopilot_min_new_items_to_finetune: int = Field(
        default=5,
        ge=1,
        le=10_000,
        validation_alias="AUTOPILOT_MIN_NEW_ITEMS_TO_FINETUNE",
    )
    autopilot_finetune_cooldown_seconds: int = Field(
        default=3600,
        ge=60,
        le=604800,
        validation_alias="AUTOPILOT_FINETUNE_COOLDOWN_SECONDS",
    )
    autopilot_base_model: str = Field(
        default="llama2:7b",
        validation_alias="AUTOPILOT_BASE_MODEL",
    )
    autopilot_output_model_template: str = Field(
        default="crew-news-{date}-{time}",
        validation_alias="AUTOPILOT_OUTPUT_MODEL_TEMPLATE",
        description="Python format keys: {date}, {time}, {ts}.",
    )
    autopilot_system_instructions: str = Field(
        default="Summarize and reason about recent news accurately; cite themes from the corpus.",
        validation_alias="AUTOPILOT_SYSTEM_INSTRUCTIONS",
    )
    autopilot_job_timeout_seconds: int = Field(
        default=1800,
        ge=60,
        le=7200,
        validation_alias="AUTOPILOT_JOB_TIMEOUT_SECONDS",
    )
    autopilot_fingerprint_max_entries: int = Field(
        default=50_000,
        ge=100,
        le=5_000_000,
        validation_alias="AUTOPILOT_FINGERPRINT_MAX_ENTRIES",
    )
    autopilot_fingerprint_ttl_seconds: int = Field(
        default=2_592_000,
        ge=0,
        le=31_536_000,
        validation_alias="AUTOPILOT_FINGERPRINT_TTL_SECONDS",
        description="0 disables TTL pruning for the on-disk fingerprint map.",
    )

    @field_validator("algorithm")
    @classmethod
    def algorithm_must_be_hs256(cls, v: str) -> str:
        if v.upper() != "HS256":
            raise ValueError("Only HS256 is supported for signing in this service.")
        return v.upper()

    @model_validator(mode="after")
    def production_hardening(self) -> "Settings":
        env = (os.getenv("ENVIRONMENT") or "").lower()
        weak = (
            "your-secret-key-change-this",
            "your-secret-key-here",
            "default-secret-key",
            "change-this",
            "secret",
        )
        if env == "production":
            if self.secret_key.lower() in {w.lower() for w in weak}:
                raise ValueError(
                    "SECRET_KEY must be set to a strong random value in production "
                    "(not a placeholder)."
                )
            if self.cors_allow_origins.strip() == "*":
                raise ValueError(
                    "CORS_ALLOW_ORIGINS must not be '*' in production; list explicit origins."
                )
        return self

    def cors_origins_list(self) -> List[str]:
        raw = self.cors_allow_origins.strip()
        if raw == "*":
            return ["*"]
        return [part.strip() for part in raw.split(",") if part.strip()]

    def redis_dsn(self) -> str:
        if not self.redis_password:
            return self.redis_url
        if "@" in self.redis_url:
            return self.redis_url
        # Insert password before host: redis://host -> redis://:pass@host
        url = self.redis_url
        scheme, rest = url.split("://", 1)
        return f"{scheme}://:{self.redis_password}@{rest}"


# Environment-specific overrides (legacy behavior)
settings = Settings()
if os.getenv("ENVIRONMENT") == "production":
    settings.api_debug = False
    settings.log_level = "WARNING"
    settings.reload = False
elif os.getenv("ENVIRONMENT") == "development":
    settings.api_debug = True
    settings.log_level = "DEBUG"
    settings.reload = True
elif os.getenv("ENVIRONMENT") == "testing":
    settings.api_debug = True
    settings.log_level = "DEBUG"
    settings.reload = False
