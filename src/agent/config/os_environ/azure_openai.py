from pydantic import BaseModel, Field


class AzureOpenAISettings(BaseModel):
    endpoint: str = Field(min_length=10)

    deployment: str = Field(min_length=2)

    api_version: str = Field(min_length=2)
