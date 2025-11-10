import os  # Import the operating system interface to read raw environment variables for overrides
from typing import Optional  # Import Optional to annotate fields that may not have values set

from pydantic import BaseSettings, Field  # Import BaseSettings and Field to declare typed environment-driven settings


class Settings(BaseSettings):  # Define a Pydantic settings class that loads configuration from environment variables
    api_title: str = Field(default="CrewAI Service", env="API_TITLE")  # Configure the API title with an environment override
    api_version: str = Field(default="1.0.0", env="API_VERSION")  # Configure the API version identifier with override support
    api_debug: bool = Field(default=False, env="API_DEBUG")  # Enable or disable FastAPI debug mode via environment
    host: str = Field(default="0.0.0.0", env="HOST")  # Configure the HTTP host binding for the FastAPI application
    port: int = Field(default=8000, env="PORT")  # Configure the TCP port the FastAPI application listens on
    reload: bool = Field(default=False, env="RELOAD")  # Configure auto-reload behavior for the development server
    ollama_base_url: str = Field(default="http://localhost:11434", env="OLLAMA_BASE_URL")  # Configure the Ollama service URL
    ollama_model: str = Field(default="llama2:latest", env="OLLAMA_MODEL")  # Configure the Ollama model identifier
    ollama_timeout: int = Field(default=60, env="OLLAMA_TIMEOUT")  # Configure the Ollama HTTP timeout window in seconds
    redis_url: str = Field(default="redis://localhost:6379/0", env="REDIS_URL")  # Configure the Redis connection URL
    redis_password: Optional[str] = Field(default=None, env="REDIS_PASSWORD")  # Configure an optional Redis password for authentication
    redis_max_connections: int = Field(default=20, env="REDIS_MAX_CONNECTIONS")  # Configure the maximum Redis connection pool size
    result_ttl_seconds: int = Field(default=86400, env="RESULT_TTL_SECONDS")  # Configure the Redis TTL for stored crew results
    request_timeout_seconds: int = Field(default=30, env="REQUEST_TIMEOUT_SECONDS")  # Configure generic HTTP timeout for dependent services
    metrics_enabled: bool = Field(default=True, env="METRICS_ENABLED")  # Enable or disable Prometheus metrics exposure
    security_token: str = Field(default="change-me", env="API_BEARER_TOKEN")  # Configure the static bearer token for API authentication
    allowed_origins: str = Field(default="*", env="ALLOWED_ORIGINS")  # Configure the comma separated CORS origin whitelist
    log_level: str = Field(default="INFO", env="LOG_LEVEL")  # Configure the default logging level for the application
    log_format: str = Field(default="%(asctime)s - %(name)s - %(levelname)s - %(message)s", env="LOG_FORMAT")  # Configure the logging formatter pattern

    class Config:  # Define Pydantic configuration metadata for the Settings class
        env_file = ".env"  # Instruct Pydantic to load variables from a .env file when present
        env_file_encoding = "utf-8"  # Define the expected encoding for environment files
        case_sensitive = False  # Treat environment variable names as case insensitive for flexibility


settings = Settings()  # Instantiate the settings object so it can be imported across the application


detected_environment = os.getenv("ENVIRONMENT", "development").lower()  # Detect the runtime environment to adjust defaults securely
if detected_environment == "production":  # Apply production-safe overrides
    settings.api_debug = False  # Force debug mode off to avoid leaking internal information
    settings.reload = False  # Disable auto-reload in production for stability
    settings.log_level = "INFO"  # Ensure production logs stay at INFO or higher by default
elif detected_environment == "testing":  # Apply testing-focused overrides
    settings.api_debug = True  # Allow debug output to simplify troubleshooting in tests
    settings.reload = False  # Keep reload disabled to match test runner expectations
    settings.log_level = "DEBUG"  # Increase logging verbosity for testing scenarios
else:  # Apply development defaults for all other cases
    settings.api_debug = True  # Turn on debug mode to support iterative development
    settings.reload = True  # Enable auto-reload to refresh the server on file changes
    settings.log_level = "DEBUG"  # Increase logging verbosity locally to surface issues quickly
