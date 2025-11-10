import os  # Import os to access environment variables for runtime configuration
from typing import Optional  # Import Optional to annotate fields that may not be provided
from pydantic import BaseSettings, Field  # Import BaseSettings and Field to define environment-aware settings models


class Settings(BaseSettings):  # Define a strongly typed settings container that reads values from environment variables
    api_title: str = Field(default="CrewAI Service", env="API_TITLE")  # Configure the human-readable API title
    api_version: str = Field(default="1.0.0", env="API_VERSION")  # Set the semantic version exposed by the API
    api_debug: bool = Field(default=False, env="API_DEBUG")  # Toggle debug mode for FastAPI based on environment
    host: str = Field(default="0.0.0.0", env="HOST")  # Specify the network interface that the service should bind to
    port: int = Field(default=8000, env="PORT")  # Define the default listening port for the service
    reload: bool = Field(default=False, env="RELOAD")  # Control auto-reload behavior for development convenience
    ollama_base_url: str = Field(default="http://localhost:11434", env="OLLAMA_BASE_URL")  # Provide the base URL for the Ollama service
    ollama_model: str = Field(default="llama2:7b", env="OLLAMA_MODEL")  # Set the default LLM to request from Ollama
    ollama_timeout: int = Field(default=300, env="OLLAMA_TIMEOUT")  # Configure the maximum wait time for Ollama responses in seconds
    redis_url: str = Field(default="redis://localhost:6379", env="REDIS_URL")  # Provide the Redis connection URL with host and port
    redis_password: Optional[str] = Field(default=None, env="REDIS_PASSWORD")  # Allow optional Redis password for secured deployments
    redis_db: int = Field(default=0, env="REDIS_DB")  # Select which Redis database index to use
    redis_max_connections: int = Field(default=10, env="REDIS_MAX_CONNECTIONS")  # Limit the total Redis connection pool size
    secret_key: str = Field(default="change-me-in-production", env="SECRET_KEY")  # Store the symmetric key used for signing tokens
    algorithm: str = Field(default="HS256", env="ALGORITHM")  # Declare the JWT signing algorithm
    access_token_expire_minutes: int = Field(default=30, env="ACCESS_TOKEN_EXPIRE_MINUTES")  # Set the JWT expiration window in minutes
    log_level: str = Field(default="INFO", env="LOG_LEVEL")  # Control the verbosity level for log output
    log_format: str = Field(default="%(asctime)s - %(name)s - %(levelname)s - %(message)s", env="LOG_FORMAT")  # Determine the formatting template for log messages
    metrics_enabled: bool = Field(default=True, env="METRICS_ENABLED")  # Decide whether to expose Prometheus metrics
    metrics_port: int = Field(default=9090, env="METRICS_PORT")  # Set the port for standalone metrics exporters if needed
    max_agents: int = Field(default=10, env="MAX_AGENTS")  # Enforce an upper bound on concurrent agents per crew
    max_tasks: int = Field(default=50, env="MAX_TASKS")  # Enforce an upper bound on concurrent tasks per crew
    crew_verbose: bool = Field(default=False, env="CREW_VERBOSE")  # Decide whether crew operations should run in verbose mode by default
    cache_ttl: int = Field(default=3600, env="CACHE_TTL")  # Set the default time to live for cached data in seconds
    result_ttl: int = Field(default=86400, env="RESULT_TTL")  # Set the default time to live for result records in seconds
    rate_limit_per_minute: int = Field(default=60, env="RATE_LIMIT_PER_MINUTE")  # Configure the global rate limit for requests per minute

    class Config:  # Provide Pydantic configuration overrides for the settings model
        env_file = ".env"  # Load environment variables from the .env file when present
        env_file_encoding = "utf-8"  # Specify the encoding for the env file
        case_sensitive = False  # Make environment variable lookups case insensitive


def _with_environment_overrides(base_settings: Settings) -> Settings:  # Helper to apply environment-specific overrides without mutating input
    environment = os.getenv("ENVIRONMENT", "").lower()  # Read the current runtime environment indicator
    if environment == "production":  # Apply production-hardened defaults if running in production
        return base_settings.model_copy(update={"api_debug": False, "log_level": "WARNING", "reload": False})  # Disable debug features and reduce logging noise in production
    if environment == "development":  # Apply development-friendly defaults when applicable
        return base_settings.model_copy(update={"api_debug": True, "log_level": "DEBUG", "reload": True})  # Enable detailed logging and auto reload for development convenience
    if environment == "testing":  # Apply testing-specific overrides when running test suites
        return base_settings.model_copy(update={"api_debug": True, "log_level": "DEBUG", "reload": False})  # Enable debugging while avoiding reload loops during tests
    return base_settings  # Return the original settings if no specific environment overrides were requested


settings = _with_environment_overrides(Settings())  # Instantiate settings and apply environment-specific adjustments
