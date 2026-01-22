from __future__ import annotations

from typing import TYPE_CHECKING

from azure.core.credentials import TokenCredential
from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from dotenv import load_dotenv
from lagom import Container
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_openai import AzureChatOpenAI
from langchain_openai.chat_models.base import BaseChatOpenAI
from langgraph.graph import StateGraph

from agent.agent import ChatAgent
from agent.config.os_environ.settings import Settings
from agent.dependency_injection.aliases import CognitiveServicesAccessToken
from agent.utils.context import Context
from agent.utils.state import State

if TYPE_CHECKING:
    from lagom.interfaces import ReadableContainer


def create_settings(ctr: ReadableContainer) -> Settings:
    """Factory function to create Settings instance."""
    load_dotenv()
    return Settings()


container = Container()

container[Settings] = create_settings

container[DefaultAzureCredential] = DefaultAzureCredential
container[TokenCredential] = lambda c: c[DefaultAzureCredential]

container[CognitiveServicesAccessToken] = lambda c: c[TokenCredential].get_token("https://cognitiveservices.azure.com/.default")

container[AzureChatOpenAI] = lambda c: AzureChatOpenAI(
    azure_endpoint=c[Settings].azure_openai_endpoint,
    azure_deployment=c[Settings].azure_openai_deployment,
    api_version=c[Settings].azure_openai_api_version,
    azure_ad_token_provider=get_bearer_token_provider(c[TokenCredential], "https://cognitiveservices.azure.com/.default"),
)

container[BaseChatOpenAI] = lambda c: c[AzureChatOpenAI]
container[BaseChatModel] = lambda c: c[BaseChatOpenAI]

container[ChatAgent] = lambda c: ChatAgent(chat_client=c[BaseChatModel])

# fmt: off
container[StateGraph] = lambda c: StateGraph(State, context_schema=Context) \
    .add_node(c[ChatAgent].call_model) \
    .add_edge("__start__", "call_model") \
    .compile(name="New Graph") # type: ignore
# fmt: on
