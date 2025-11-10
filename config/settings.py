from typing import List, Optional  # Import typing helpers for type hints
from pydantic import BaseSettings, Field, SecretStr, field_validator  # Import Pydantic classes for validated settings management


class Settings(BaseSettings):  # Define strongly typed application settings leveraging Pydantic for validation
    environment: str = Field(default="development", env="ENVIRONMENT")  # Track the current deployment environment label
    api_title: str = Field(default="CrewAI Service", env="API_TITLE")  # Configure the API title with environment override capability
    api_version: str = Field(default="1.0.0", env="API_VERSION")  # Configure the API version for documentation consistency
    api_debug: bool = Field(default=False, env="API_DEBUG")  # Toggle FastAPI debugging features based on environment configuration
    host: str = Field(default="0.0.0.0", env="HOST")  # Configure the host interface the application binds to
    port: int = Field(default=8000, env="PORT")  # Configure the HTTP port exposed by the application
    reload: bool = Field(default=False, env="RELOAD")  # Toggle automatic reload useful for development workflows
    ollama_base_url: str = Field(default="http://localhost:11434", env="OLLAMA_BASE_URL")  # Configure the Ollama service base URL
    ollama_model: str = Field(default="llama2:7b", env="OLLAMA_MODEL")  # Configure the default Ollama model identifier
    ollama_timeout: int = Field(default=300, env="OLLAMA_TIMEOUT")  # Configure the Ollama client timeout to mitigate hanging requests
    redis_url: str = Field(default="redis://localhost:6379/0", env="REDIS_URL")  # Configure the Redis connection URL using secure defaults
    redis_password: Optional[SecretStr] = Field(default=None, env="REDIS_PASSWORD")  # Optionally configure a Redis password stored as a secret string
    redis_db: int = Field(default=0, env="REDIS_DB")  # Configure the Redis database index for logical separation
    redis_max_connections: int = Field(default=10, env="REDIS_MAX_CONNECTIONS")  # Configure the Redis connection pool size for resource control
    secret_key: SecretStr = Field(default=SecretStr("CHANGE_ME"), env="SECRET_KEY")  # Configure the application secret key and flag insecure defaults
    algorithm: str = Field(default="HS256", env="ALGORITHM")  # Configure the JWT signing algorithm used for token validation
    access_token_expire_minutes: int = Field(default=30, env="ACCESS_TOKEN_EXPIRE_MINUTES")  # Configure JWT access token lifetime for session management
    log_level: str = Field(default="INFO", env="LOG_LEVEL")  # Configure application log level for observability
    log_format: str = Field(default="%(asctime)s - %(name)s - %(levelname)s - %(message)s", env="LOG_FORMAT")  # Configure log message formatting for consistency
    metrics_enabled: bool = Field(default=True, env="METRICS_ENABLED")  # Toggle Prometheus metrics exposure to support monitoring
    metrics_port: int = Field(default=9090, env="METRICS_PORT")  # Configure the metrics endpoint port when running standalone exporters
    max_agents: int = Field(default=10, env="MAX_AGENTS")  # Configure an upper bound on concurrently configured agents
    max_tasks: int = Field(default=50, env="MAX_TASKS")  # Configure an upper bound on concurrently configured tasks
    crew_verbose: bool = Field(default=False, env="CREW_VERBOSE")  # Toggle verbose logging within crew executions for debugging
    cache_ttl: int = Field(default=3600, env="CACHE_TTL")  # Configure cache time to live in seconds to control data freshness
    result_ttl: int = Field(default=86400, env="RESULT_TTL")  # Configure crew result retention period in seconds
    rate_limit_per_minute: int = Field(default=60, env="RATE_LIMIT_PER_MINUTE")  # Configure API rate limiting budget per minute
    cors_allowed_origins: List[str] = Field(default_factory=lambda: ["https://localhost"], env="CORS_ALLOWED_ORIGINS")  # Configure allowed CORS origins to restrict cross-site access

    @field_validator("cors_allowed_origins", mode="before")  # Normalize configured CORS origins irrespective of string or list input
    def _split_origins(cls, value):  # Define validator to convert comma separated strings into lists
        if isinstance(value, str):  # Check if value is provided as a comma separated string
            return [origin.strip() for origin in value.split(",") if origin.strip()]  # Convert string into sanitized list of origins
        return value  # Return list values unchanged when already structured correctly

    class Config:  # Configure Pydantic model behavior and environment loading
        env_file = ".env"  # Specify the .env file used for local configuration overrides
        env_file_encoding = "utf-8"  # Specify encoding for deterministic .env parsing
        case_sensitive = False  # Allow case-insensitive environment variable lookups for flexibility

    def get_secret_key(self) -> str:  # Provide accessor enforcing validation of the application secret key
        secret_value = self.secret_key.get_secret_value()  # Retrieve the underlying secret string safely
        if secret_value == "CHANGE_ME":  # Detect insecure default secret keys to prevent accidental deployment
            raise ValueError("SECRET_KEY environment variable must be set to a secure value.")  # Raise explicit error to enforce secure configuration
        return secret_value  # Return the validated secret key for downstream consumers


settings = Settings()  # Instantiate settings using environment variables with validation safeguards

if settings.environment.lower() == "production":  # Apply production hardened defaults when environment indicates production
    settings.api_debug = False  # Disable debug mode to avoid leaking internal information
    settings.log_level = "WARNING"  # Reduce log verbosity to critical information for production operations
    settings.reload = False  # Disable auto reload to prevent unintended restarts in production
elif settings.environment.lower() == "development":  # Apply development friendly defaults when environment indicates development
    settings.api_debug = True  # Enable debug mode for developer feedback loops
    settings.log_level = "DEBUG"  # Increase log verbosity to assist with troubleshooting
    settings.reload = True  # Enable auto reload to streamline iterative development
elif settings.environment.lower() == "testing":  # Apply testing specific defaults when environment indicates testing
    settings.api_debug = True  # Enable debug mode to surface test failures clearly
    settings.log_level = "DEBUG"  # Increase log verbosity to capture detailed test traces
    settings.reload = False  # Disable reload to keep test execution deterministic
