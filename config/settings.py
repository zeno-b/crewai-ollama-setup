import os
from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field

class Settings(BaseSettings):
    """Application settings"""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # API Settings
    api_title: str = Field(default="CrewAI Service")
    api_version: str = Field(default="1.0.0")
    api_debug: bool = Field(default=False)

    # Server Settings
    host: str = Field(default="0.0.0.0")
    port: int = Field(default=8000)
    reload: bool = Field(default=False)

    # Ollama Settings
    ollama_base_url: str = Field(default="http://localhost:11434")
    ollama_model: str = Field(default="llama2:7b")
    ollama_timeout: int = Field(default=300)

    # Redis Settings
    redis_url: str = Field(default="redis://localhost:6379")
    redis_password: Optional[str] = Field(default=None)
    redis_db: int = Field(default=0)
    redis_max_connections: int = Field(default=10)

    # Security Settings
    secret_key: str = Field(default="your-secret-key-change-this")
    algorithm: str = Field(default="HS256")
    access_token_expire_minutes: int = Field(default=30)

    # Logging Settings
    log_level: str = Field(default="INFO")
    log_format: str = Field(
        default="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Monitoring Settings
    metrics_enabled: bool = Field(default=True)
    metrics_port: int = Field(default=9090)

    # CrewAI Settings
    max_agents: int = Field(default=10)
    max_tasks: int = Field(default=50)
    crew_verbose: bool = Field(default=False)

    # Cache Settings
    cache_ttl: int = Field(default=3600)  # 1 hour
    result_ttl: int = Field(default=86400)  # 24 hours

    # Rate Limiting
    rate_limit_per_minute: int = Field(default=60)

    @property
    def effective_log_level(self) -> str:
        env = os.getenv("ENVIRONMENT")
        if env == "production":
            return "WARNING"
        if env in ("development", "testing"):
            return "DEBUG"
        return self.log_level

    @property
    def effective_debug(self) -> bool:
        env = os.getenv("ENVIRONMENT")
        return env in ("development", "testing")

    @property
    def effective_reload(self) -> bool:
        return os.getenv("ENVIRONMENT") == "development"


# Initialize settings
settings = Settings()
