from __future__ import annotations

from azure.core.credentials import TokenCredential
from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from dotenv import load_dotenv
from lagom import Container
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.runnables.base import Runnable
from langchain_core.runnables.graph import Graph
from langchain_core.tools.base import BaseTool
from langchain_openai import AzureChatOpenAI
from langchain_openai.chat_models.base import BaseChatOpenAI
from langgraph.graph import StateGraph
from langgraph.graph.state import CompiledStateGraph

from agent.config.os_environ.azure_openai import AzureOpenAISettings
from agent.config.os_environ.settings import Settings
from agent.dependency_injection.aliases import CognitiveServicesAccessToken
from agent.lang_graph.context import Context
from agent.lang_graph.model.invoker import ModelInvoker
from agent.lang_graph.model.protocol import ModelInvokerProtocol
from agent.lang_graph.state_graph.my import MyStateGraph
from agent.lang_graph.states.a2a import A2AMessagesState
from agent.lang_graph.tools.invoker import ToolInvoker
from agent.lang_graph.tools.math import add, divide, multiply
from agent.lang_graph.tools.protocol import ToolInvokerProtocol


def create_settings() -> Settings:
    """Factory function to create Settings instance."""
    load_dotenv()
    return Settings()  # type: ignore[call-arg, unused-ignore]


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

container[list[BaseTool]] = lambda c: [add, multiply, divide]

container[dict[str, BaseTool]] = lambda c: {tool.name: tool for tool in c[list[BaseTool]]}

container[Runnable] = lambda c: c[BaseChatModel].bind_tools(c[list[BaseTool]])  # type: ignore

container[ModelInvoker] = lambda c: ModelInvoker(runnable=c[Runnable])
container[ModelInvokerProtocol] = lambda c: c[ModelInvoker]

container[ToolInvoker] = lambda c: ToolInvoker(tools_by_name=c[dict[str, BaseTool]])
container[ToolInvokerProtocol] = lambda c: c[ToolInvoker]

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
