import os  # Import the operating system interface to read environment variables securely
from typing import List, Optional  # Import type hints for better readability and tooling support
from pydantic import BaseSettings, Field, field_validator  # Import Pydantic helpers to manage configuration safely


class Settings(BaseSettings):  # Define a strongly typed settings container that reads from environment variables
    """Application settings with security-conscious defaults and validation."""  # Provide a concise description of the configuration model

    environment: str = Field(default="development", env="ENVIRONMENT")  # Track the current deployment environment for conditional logic

    api_title: str = Field(default="CrewAI Service", env="API_TITLE")  # Configure the FastAPI title and allow overriding via environment
    api_version: str = Field(default="1.0.0", env="API_VERSION")  # Expose the API version string with environment override support
    api_debug: bool = Field(default=False, env="API_DEBUG")  # Control FastAPI debug mode while defaulting to secure off state

    host: str = Field(default="0.0.0.0", env="HOST")  # Bind address for the API server, defaulting to all interfaces
    port: int = Field(default=8000, env="PORT")  # TCP port for the API server with configurable override
    reload: bool = Field(default=False, env="RELOAD")  # Control auto-reload behavior for development convenience

    ollama_base_url: str = Field(default="http://localhost:11434", env="OLLAMA_BASE_URL")  # Configure the Ollama endpoint base URL
    ollama_model: str = Field(default="llama2:7b", env="OLLAMA_MODEL")  # Select the default Ollama model to load
    ollama_timeout: int = Field(default=300, env="OLLAMA_TIMEOUT")  # Set an upper bound on Ollama call duration in seconds

    redis_url: str = Field(default="redis://localhost:6379/0", env="REDIS_URL")  # Provide a full Redis URL including database index
    redis_password: Optional[str] = Field(default=None, env="REDIS_PASSWORD")  # Allow setting Redis authentication while defaulting to none
    redis_max_connections: int = Field(default=10, env="REDIS_MAX_CONNECTIONS")  # Cap concurrent Redis connections to protect resources

    secret_key: str = Field(default="change-me-immediately", env="SECRET_KEY")  # Store the API secret key with a clearly insecure placeholder default
    algorithm: str = Field(default="HS256", env="ALGORITHM")  # Define the JWT signing algorithm to maintain compatibility across services
    access_token_expire_minutes: int = Field(default=30, env="ACCESS_TOKEN_EXPIRE_MINUTES")  # Configure token expiry in minutes to limit exposure
    api_token_hash: Optional[str] = Field(default=None, env="API_TOKEN_HASH")  # Support hashed API tokens for simple bearer authentication

    log_level: str = Field(default="INFO", env="LOG_LEVEL")  # Establish the default logging level and allow environment overrides
    log_format: str = Field(default="%(asctime)s - %(name)s - %(levelname)s - %(message)s", env="LOG_FORMAT")  # Specify the logging format string

    metrics_enabled: bool = Field(default=True, env="METRICS_ENABLED")  # Toggle Prometheus metrics exposure for observability
    metrics_port: int = Field(default=9090, env="METRICS_PORT")  # Provide a port for auxiliary metrics servers when enabled

    max_agents: int = Field(default=10, env="MAX_AGENTS")  # Limit the number of concurrent agents to control resource usage
    max_tasks: int = Field(default=50, env="MAX_TASKS")  # Bound the total number of tasks to prevent runaway workloads
    crew_verbose: bool = Field(default=False, env="CREW_VERBOSE")  # Allow enabling verbose CrewAI logging when troubleshooting

    cache_ttl: int = Field(default=3600, env="CACHE_TTL")  # Set default cache time-to-live (seconds) for transient data storage
    result_ttl: int = Field(default=86400, env="RESULT_TTL")  # Set default persistence duration (seconds) for crew results

    rate_limit_per_minute: int = Field(default=60, env="RATE_LIMIT_PER_MINUTE")  # Impose a simple per-minute rate limit for API throttling

    allowed_origins: List[str] = Field(default_factory=lambda: ["http://localhost:3000"], env="ALLOWED_ORIGINS")  # Define CORS allowlist with safe defaults
    allowed_methods: List[str] = Field(default_factory=lambda: ["GET", "POST", "OPTIONS"], env="ALLOWED_METHODS")  # Restrict HTTP methods permitted by CORS policy
    allowed_headers: List[str] = Field(default_factory=lambda: ["Authorization", "Content-Type"], env="ALLOWED_HEADERS")  # Restrict headers permitted by CORS policy

    class Config:  # Configure Pydantic BaseSettings behaviour
        env_file = ".env"  # Load environment variables from a standard dotenv file
        env_file_encoding = "utf-8"  # Interpret the dotenv file using UTF-8 encoding
        case_sensitive = False  # Allow case-insensitive environment variable names for compatibility

    @field_validator("secret_key")  # Apply validation logic each time the secret key is loaded
    @classmethod
    def enforce_secret_key_change(cls, value: str) -> str:  # Ensure insecure placeholder secrets are not used in production
        placeholder = "change-me-immediately"  # Provide a single source of truth for the insecure placeholder value
        current_env = os.getenv("ENVIRONMENT", "development").lower()  # Read the active environment to contextualize validation
        if value == placeholder and current_env == "production":  # Disallow insecure placeholders only when running in production
            raise ValueError("SECRET_KEY must be overridden for security compliance.")  # Prevent startup when insecure defaults are detected
        return value  # Return the validated secret key value

    @field_validator("allowed_origins", mode="before")  # Normalize CORS origin configuration regardless of input type
    @classmethod
    def parse_allowed_origins(cls, value) -> List[str]:  # Convert comma-delimited strings into lists for FastAPI compatibility
        if isinstance(value, str):  # Detect string inputs provided via environment variables
            return [origin.strip() for origin in value.split(",") if origin.strip()]  # Split, sanitize, and filter origin entries
        return value  # Return lists unchanged to respect python-based configuration

    @field_validator("allowed_methods", mode="before")  # Normalize allowed HTTP method configuration
    @classmethod
    def parse_allowed_methods(cls, value) -> List[str]:  # Ensure FastAPI receives a list regardless of environment formatting
        if isinstance(value, str):  # Handle comma-delimited environment strings
            return [method.strip().upper() for method in value.split(",") if method.strip()]  # Split, sanitize, and normalize casing
        return value  # Return lists unchanged

    @field_validator("allowed_headers", mode="before")  # Normalize allowed HTTP headers configuration
    @classmethod
    def parse_allowed_headers(cls, value) -> List[str]:  # Ensure consistent list format for FastAPI middleware
        if isinstance(value, str):  # Handle comma-separated header strings
            return [header.strip() for header in value.split(",") if header.strip()]  # Split and sanitize header names
        return value  # Return lists unchanged


def load_settings() -> Settings:  # Provide a helper to instantiate and post-process settings consistently
    settings_instance = Settings()  # Create a new Settings instance pulling from environment variables
    if settings_instance.environment == "production":  # Apply additional hardening when running in production
        settings_instance = settings_instance.model_copy(update={"api_debug": False, "log_level": "WARNING", "reload": False})  # Disable debug features and reduce logging noise in production
    elif settings_instance.environment == "development":  # Apply development-friendly defaults
        settings_instance = settings_instance.model_copy(update={"api_debug": True, "log_level": "DEBUG", "reload": True})  # Enable debugging aids during development iterations
    elif settings_instance.environment == "testing":  # Apply predictable settings during automated testing
        settings_instance = settings_instance.model_copy(update={"api_debug": True, "log_level": "DEBUG", "reload": False})  # Keep test runs deterministic while still surfacing debug information
    return settings_instance  # Return the fully configured settings instance for application use


settings = load_settings()  # Instantiate and expose settings for the rest of the application to import
