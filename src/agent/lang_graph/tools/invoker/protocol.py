from typing import Any, Protocol

from langchain_core.tools.base import BaseTool

from agent.lang_graph.states.a2a import A2AMessagesState


class ToolInvokerProtocol(Protocol):
    tools: dict[str, BaseTool]

    async def invoke_async(self, state: A2AMessagesState) -> dict[str, list[Any]]:  # TODO? async?
        ...
