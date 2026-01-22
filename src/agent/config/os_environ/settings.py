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

    debug: bool = Field(default=False)  # type: ignore

    dry_run: bool = Field(default=True)  # type: ignore

    environment: str = Field(min_length=2)  # type: ignore

    azure_openai_endpoint: str = Field(min_length=10)  # type: ignore
    azure_openai_deployment: str = Field(min_length=2)  # type: ignore
    azure_openai_api_version: str = Field(min_length=2)  # type: ignore
