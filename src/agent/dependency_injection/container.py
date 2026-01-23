from __future__ import annotations

from collections.abc import Awaitable
from typing import TYPE_CHECKING

from azure.core.credentials import TokenCredential
from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from dotenv import load_dotenv
from lagom import Container
from langchain_core.language_models.chat_models import BaseChatModel
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
from agent.lang_graph.model.invoker.runnable import RunnnableModelInvoker
from agent.lang_graph.state_graph.my import MyStateGraph
from agent.lang_graph.states.a2a import A2AMessagesState
from agent.lang_graph.tools.invoker.async_tools import AsyncToolInvoker
from agent.lang_graph.tools.invoker.protocol import ToolInvokerProtocol
from agent.lang_graph.tools.math import add, divide, multiply

LOCAL_TOOLS = [add, multiply, divide]

if TYPE_CHECKING:
    from lagom.interfaces import ReadableContainer


def create_settings() -> Settings:
    """Factory function to create Settings instance."""
    load_dotenv()
    return Settings()  # type: ignore[call-arg, unused-ignore] # FIXME


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

    return MultiServerMCPClient(entries)  # type: ignore[arg-type] # FIXME


async def get_all_tools_async(container: ReadableContainer) -> list[BaseTool]:
    """Factory function to get all tools (local + remote) asynchronously."""
    mcp_client = container[MultiServerMCPClient]
    remote_tools = await mcp_client.get_tools()
    return LOCAL_TOOLS + remote_tools


async def create_tool_invoker_async(container: ReadableContainer) -> AsyncToolInvoker:
    tools_list = await container[Awaitable[list[BaseTool]]]  # type: ignore[type-abstract] # FIXME
    tools = {tool.name: tool for tool in tools_list}
    return AsyncToolInvoker(tools=tools)


async def create_my_state_graph_async(container: ReadableContainer) -> MyStateGraph:
    chat_model = container[BaseChatModel]  # type: ignore[type-abstract] # FIXME
    tool_invoker = await container[Awaitable[ToolInvokerProtocol]]  # type: ignore[type-abstract] # FIXME
    tools: list[BaseTool] = list(tool_invoker.tools.values())
    runnable = chat_model.bind_tools(tools)  # pyright: ignore[reportUnknownMemberType] # FIXME
    model_invoker = RunnnableModelInvoker(runnable=runnable)
    return MyStateGraph(
        model_invoker=model_invoker,
        tool_invoker=tool_invoker,
    )


async def create_state_graph_async(container: ReadableContainer) -> StateGraph[A2AMessagesState, Context]:
    my_state_graph = await container[Awaitable[MyStateGraph]]  # type: ignore[type-abstract] # FIXME
    return my_state_graph.state_graph


async def create_compiled_state_graph_async(container: ReadableContainer) -> CompiledStateGraph[A2AMessagesState, Context]:
    state_graph = await container[Awaitable[StateGraph[A2AMessagesState, Context]]]  # type: ignore[type-abstract] # FIXME
    return state_graph.compile(name="Compiled Graph")  # type: ignore[return-value] # FIXME


async def get_graph_async(container: ReadableContainer) -> Graph:
    compiled_state_graph = await container[Awaitable[CompiledStateGraph[A2AMessagesState, Context]]]  # type: ignore[type-abstract] # FIXME
    return compiled_state_graph.get_graph(xray=True)


container = Container()

container[Settings] = create_settings()  # Singleton
container[AzureOpenAISettings] = lambda c: c[Settings].azure_openai

container[DefaultAzureCredential] = DefaultAzureCredential
container[TokenCredential] = lambda c: c[DefaultAzureCredential]  # type: ignore # FIXME

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
container[BaseChatModel] = lambda c: c[BaseChatOpenAI]  # type: ignore # FIXME

# local tools
container[list[BaseTool]] = LOCAL_TOOLS

# remote tools
container[MultiServerMCPClient] = create_multi_server_mcp_client

# All tools, including locals
container[Awaitable[list[BaseTool]]] = get_all_tools_async  # type: ignore[type-abstract] # FIXME

container[Awaitable[AsyncToolInvoker]] = create_tool_invoker_async  # type: ignore[type-abstract] # FIXME
container[Awaitable[ToolInvokerProtocol]] = lambda c: c[Awaitable[AsyncToolInvoker]]  # type: ignore[type-abstract] # FIXME

container[Awaitable[MyStateGraph]] = create_my_state_graph_async  # type: ignore[type-abstract]

container[Awaitable[StateGraph[A2AMessagesState, Context]]] = create_state_graph_async  # type: ignore[type-abstract] # FIXME
container[Awaitable[CompiledStateGraph[A2AMessagesState, Context]]] = create_compiled_state_graph_async  # type: ignore[type-abstract] # FIXME
container[Awaitable[CompiledStateGraph]] = lambda c: c[Awaitable[CompiledStateGraph[A2AMessagesState, Context]]]  # type: ignore[type-abstract, type-arg] # FIXME

container[Awaitable[Graph]] = get_graph_async  # type: ignore[type-abstract] # FIXME
