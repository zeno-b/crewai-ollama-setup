import os
from typing import Optional
from pydantic import Field

try:
    from pydantic_settings import BaseSettings
except ImportError:  # pragma: no cover - compatibility for legacy environments
    from pydantic.v1 import BaseSettings

class Settings(BaseSettings):
    """Application settings"""
    
    # API Settings
    api_title: str = Field(default="CrewAI Service", env="API_TITLE")
    api_version: str = Field(default="1.0.0", env="API_VERSION")
    api_debug: bool = Field(default=False, env="API_DEBUG")
    
    # Server Settings
    host: str = Field(default="0.0.0.0", env="HOST")
    port: int = Field(default=8000, env="PORT")
    reload: bool = Field(default=False, env="RELOAD")
    
    # Ollama Settings
    ollama_base_url: str = Field(default="http://localhost:11434", env="OLLAMA_BASE_URL")
    ollama_model: str = Field(default="llama2:7b", env="OLLAMA_MODEL")
    ollama_timeout: int = Field(default=300, env="OLLAMA_TIMEOUT")
    
    # Redis Settings
    redis_url: str = Field(default="redis://localhost:6379", env="REDIS_URL")
    redis_password: Optional[str] = Field(default=None, env="REDIS_PASSWORD")
    redis_db: int = Field(default=0, env="REDIS_DB")
    redis_max_connections: int = Field(default=10, env="REDIS_MAX_CONNECTIONS")
    
    # Security Settings
    secret_key: str = Field(default="your-secret-key-change-this", env="SECRET_KEY")
    algorithm: str = Field(default="HS256", env="ALGORITHM")
    access_token_expire_minutes: int = Field(default=30, env="ACCESS_TOKEN_EXPIRE_MINUTES")
    
    # Logging Settings
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    log_format: str = Field(
        default="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        env="LOG_FORMAT"
    )
    
    # Monitoring Settings
    metrics_enabled: bool = Field(default=True, env="METRICS_ENABLED")
    metrics_port: int = Field(default=9090, env="METRICS_PORT")
    
    # CrewAI Settings
    max_agents: int = Field(default=10, env="MAX_AGENTS")
    max_tasks: int = Field(default=50, env="MAX_TASKS")
    crew_verbose: bool = Field(default=False, env="CREW_VERBOSE")
    
    # Cache Settings
    cache_ttl: int = Field(default=3600, env="CACHE_TTL")  # 1 hour
    result_ttl: int = Field(default=86400, env="RESULT_TTL")  # 24 hours
    
    # Rate Limiting
    rate_limit_per_minute: int = Field(default=60, env="RATE_LIMIT_PER_MINUTE")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False

# Initialize settings
settings = Settings()

# Environment-specific overrides
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
