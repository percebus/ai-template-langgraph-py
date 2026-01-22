from dataclasses import Field

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    """Base class for settings."""

    model_config = SettingsConfigDict(
        extra="ignore",
        case_sensitive=False,
        env_prefix="",
        env_nested_delimiter="__",
    )

    debug: bool = Field(default=False)

    dry_run: bool = Field(default=True)

    environment: str = Field(min_length=2)

    azure_openai_endpoint: str = Field(min_length=10)
    azure_openai_deployment: str = Field(min_length=2)
    azure_openai_api_version: str = Field(min_length=2)
