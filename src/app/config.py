"""
Rossmann Sales Pipeline Configuration
Pydantic settings for environment-based configuration
"""

from functools import lru_cache
from typing import List, Optional
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables"""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Application
    app_name: str = "rossmann"
    app_env: str = Field(default="development", alias="APP_ENV")
    app_log_level: str = Field(default="INFO", alias="APP_LOG_LEVEL")
    api_host: str = Field(default="0.0.0.0", alias="API_HOST")
    api_port: int = Field(default=8000, alias="API_PORT")
    api_workers: int = Field(default=2, alias="API_WORKERS")

    # Model
    model_path: str = Field(default="models/model.pkl", alias="MODEL_PATH")
    model_type: str = Field(default="random_forest", alias="MODEL_TYPE")

    # MLflow
    mlflow_tracking_uri: str = Field(default="http://localhost:5000", alias="MLFLOW_TRACKING_URI")
    mlflow_model_name: str = Field(default="rossmann-sales-model", alias="MLFLOW_MODEL_NAME")
    mlflow_model_stage: str = Field(default="Production", alias="MLFLOW_MODEL_STAGE")

    # Database
    database_url: str = Field(default="postgresql://rossmann:changeme123@localhost:5432/rossmann", alias="DATABASE_URL")
    postgres_host: str = Field(default="localhost", alias="POSTGRES_HOST")
    postgres_port: int = Field(default=5432, alias="POSTGRES_PORT")
    postgres_user: str = Field(default="rossmann", alias="POSTGRES_USER")
    postgres_password: str = Field(default="changeme123", alias="POSTGRES_PASSWORD")
    postgres_db: str = Field(default="rossmann", alias="POSTGRES_DB")

    # Dash Dashboard
    dash_port: int = Field(default=8050, alias="DASH_PORT")
    dash_debug: bool = Field(default=False, alias="DASH_DEBUG")
    dash_host: str = Field(default="0.0.0.0", alias="DASH_HOST")

    # Monitoring
    enable_metrics: bool = Field(default=True, alias="ENABLE_METRICS")
    metrics_port: int = Field(default=9090, alias="METRICS_PORT")

    # Security
    secret_key: str = Field(default="change-me-in-production", alias="SECRET_KEY")
    cors_origins: List[str] = Field(default=["*"], alias="CORS_ORIGINS")

    # Health Check
    health_check_interval: int = Field(default=30, alias="HEALTH_CHECK_INTERVAL")

    # Sentry
    sentry_dsn: Optional[str] = Field(default=None, alias="SENTRY_DSN")
    sentry_environment: str = Field(default="development", alias="SENTRY_ENVIRONMENT")
    sentry_release: str = Field(default="1.0.0", alias="SENTRY_RELEASE")
    sentry_traces_sample_rate: float = Field(default=0.1, alias="SENTRY_TRACES_SAMPLE_RATE")

    # Features
    prediction_horizon_days: int = Field(default=6, alias="PREDICTION_HORIZON_DAYS")
    feature_store_path: str = Field(default="data/features", alias="FEATURE_STORE_PATH")


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance"""
    return Settings()


# Export for easy access
settings = get_settings()