from __future__ import annotations

from azure.core.credentials import TokenCredential
from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from dotenv import load_dotenv
from lagom import Container
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.runnables.graph import Graph
from langchain_openai import AzureChatOpenAI
from langchain_openai.chat_models.base import BaseChatOpenAI
from langgraph.graph import StateGraph, START
from langchain_core.runnables.base import Runnable
from langgraph.graph.state import CompiledStateGraph

from agent.agent import ChatAgent
from agent.config.os_environ.azure_openai import AzureOpenAISettings
from agent.config.os_environ.settings import Settings
from agent.dependency_injection.aliases import CognitiveServicesAccessToken
from agent.utils.context import Context
from agent.utils.state import State
from langchain_core.tools.base import BaseTool


def create_settings() -> Settings:
    """Factory function to create Settings instance."""
    load_dotenv()
    return Settings()  # pyright: ignore[reportCallIssue]


container = Container()

container[Settings] = create_settings()  # Singleton
container[AzureOpenAISettings] = lambda c: c[Settings].azure_openai

container[DefaultAzureCredential] = DefaultAzureCredential
container[TokenCredential] = lambda c: c[DefaultAzureCredential]  # type: ignore

container[CognitiveServicesAccessToken] = lambda c: c[TokenCredential].get_token("https://cognitiveservices.azure.com/.default")

container[AzureChatOpenAI] = lambda c: AzureChatOpenAI(
    azure_endpoint=c[AzureOpenAISettings].endpoint,
    azure_deployment=c[AzureOpenAISettings].deployment,
    api_version=c[AzureOpenAISettings].api_version,
    azure_ad_token_provider=get_bearer_token_provider(c[TokenCredential], "https://cognitiveservices.azure.com/.default"),
)

container[BaseChatOpenAI] = lambda c: c[AzureChatOpenAI]
container[BaseChatModel] = lambda c: c[BaseChatOpenAI]  # type: ignore

container[list[BaseTool]] = lambda c: []  # type: ignore
container[Runnable] = lambda c: c[BaseChatModel]\
    .bind_tools(c[list[BaseTool]])  # type: ignore

container[ChatAgent] = lambda c: ChatAgent(runnable=c[Runnable])

# TODO move to an langgraph folder of sorts. i.e. "utils"
# fmt: off
container[StateGraph] = lambda c: (  # type: ignore
    StateGraph(State, context_schema=Context)
        .add_node("call_model", c[ChatAgent].call_model)   # pyright: ignore[reportUnknownMemberType]
        .add_edge(START, "call_model")
)
# fmt: on


container[CompiledStateGraph] = lambda c: c[StateGraph].compile(name="Compiled Graph")  # type: ignore
container[Graph] = lambda c: c[CompiledStateGraph].get_graph(xray=True)  # type: ignore
