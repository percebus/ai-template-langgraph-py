from __future__ import annotations

from collections.abc import Awaitable
from typing import TYPE_CHECKING

from azure.core.credentials import TokenCredential
from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from dotenv import load_dotenv
from lagom import Container
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.runnables.base import Runnable
from langchain_core.runnables.graph import Graph
from langchain_core.tools.base import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_openai import AzureChatOpenAI
from langchain_openai.chat_models.base import BaseChatOpenAI
from langgraph.graph import StateGraph
from langgraph.graph.state import CompiledStateGraph

from agent.config.os_environ.azure_openai import AzureOpenAISettings
from agent.config.os_environ.settings import Settings
from agent.dependency_injection.aliases import CognitiveServicesAccessToken
from agent.lang_graph.context import Context
from agent.lang_graph.model.invoker.protocol import ModelInvokerProtocol
from agent.lang_graph.model.invoker.runnable import ModelInvoker
from agent.lang_graph.state_graph.my import MyStateGraph
from agent.lang_graph.states.a2a import A2AMessagesState
from agent.lang_graph.tools.invoker.by_name import ToolInvoker
from agent.lang_graph.tools.invoker.protocol import ToolInvokerProtocol
from agent.lang_graph.tools.math import add, divide, multiply

LOCAL_TOOLS = [add, multiply, divide]

if TYPE_CHECKING:
    from lagom.interfaces import ReadableContainer


def create_settings() -> Settings:
    """Factory function to create Settings instance."""
    load_dotenv()
    return Settings()  # type: ignore[call-arg, unused-ignore]


def create_multi_server_mcp_client(container: ReadableContainer) -> MultiServerMCPClient:
    """Factory function to create MCP client instance."""
    settings = container[Settings]

    entries = {}
    for k, v in settings.mcp_urls.items():
        entry = {
            "transport": "http",
            "url": str(v),
        }

        entries[k] = entry

    return MultiServerMCPClient(entries)  # type: ignore[arg-type]


async def get_all_tools_async(container: ReadableContainer) -> list[BaseTool]:
    """Factory function to get all tools (local + remote) asynchronously."""
    mcp_client = container[MultiServerMCPClient]
    remote_tools = await mcp_client.get_tools()
    return LOCAL_TOOLS + remote_tools


async def init_async(container: Container) -> None:
    """Factory function to create and compile the state graph."""
    container[list[BaseTool]] = await container[Awaitable[list[BaseTool]]]  # type: ignore[type-abstract]


container = Container()

container[Settings] = create_settings()  # Singleton
container[AzureOpenAISettings] = lambda c: c[Settings].azure_openai

container[DefaultAzureCredential] = DefaultAzureCredential
container[TokenCredential] = lambda c: c[DefaultAzureCredential]  # type: ignore

container[CognitiveServicesAccessToken] = lambda c: c[TokenCredential].get_token(
    "https://cognitiveservices.azure.com/.default"
)  # XXX? not really used

container[AzureChatOpenAI] = lambda c: AzureChatOpenAI(
    azure_endpoint=c[AzureOpenAISettings].endpoint,
    azure_deployment=c[AzureOpenAISettings].deployment,
    api_version=c[AzureOpenAISettings].api_version,
    azure_ad_token_provider=get_bearer_token_provider(c[TokenCredential], "https://cognitiveservices.azure.com/.default"),
)

container[BaseChatOpenAI] = lambda c: c[AzureChatOpenAI]
container[BaseChatModel] = lambda c: c[BaseChatOpenAI]  # type: ignore

# remote tools
container[MultiServerMCPClient] = create_multi_server_mcp_client

# All tools, including locals
container[Awaitable[list[BaseTool]]] = get_all_tools_async  # type: ignore[type-abstract]

container[Runnable] = lambda c: c[BaseChatModel].bind_tools(c[list[BaseTool]])  # type: ignore

container[ModelInvoker] = lambda c: ModelInvoker(runnable=c[Runnable])
container[ModelInvokerProtocol] = lambda c: c[ModelInvoker]  # type: ignore[type-abstract]

container[ToolInvoker] = lambda c: ToolInvoker(tools=c[list[BaseTool]])
container[ToolInvokerProtocol] = lambda c: c[ToolInvoker]  # type: ignore[type-abstract]

# fmt: off
container[MyStateGraph] = lambda c: MyStateGraph(
    model_invoker=c[ModelInvokerProtocol],# type: ignore[unused-ignore]
    tool_invoker=c[ToolInvokerProtocol],
)
# fmt: on

container[StateGraph[A2AMessagesState, Context]] = lambda c: c[MyStateGraph].state_graph
container[StateGraph] = lambda c: c[StateGraph[A2AMessagesState, Context]]

container[CompiledStateGraph] = lambda c: c[StateGraph].compile(name="Compiled Graph")  # type: ignore[unused-ignore]
container[Graph] = lambda c: c[CompiledStateGraph].get_graph(xray=True)
