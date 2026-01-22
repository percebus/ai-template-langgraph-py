from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from agent.config.os_environ.azure_openai import AzureOpenAISettings


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

    azure_openai: AzureOpenAISettings = Field(default_factory=AzureOpenAISettings)  # type: ignore
