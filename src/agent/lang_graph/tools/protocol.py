from typing import Any, Protocol

from langchain_core.tools.base import BaseTool

from agent.lang_graph.states.a2a import A2AMessagesState


class ToolInvokerProtocol(Protocol):
    tools_by_name: dict[str, BaseTool]

    def invoke(self, state: A2AMessagesState) -> dict[str, list[Any]]: ...
