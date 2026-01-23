from typing import Any, Protocol

from langchain_core.tools.base import BaseTool

from agent.lang_graph.states.a2a import A2AMessagesState


class ToolInvokerProtocol(Protocol):
    tools: list[BaseTool]

    @property
    def tools_by_name(self) -> dict[str, BaseTool]:
        """Retrieves tools and maps them by name."""
        return {tool.name: tool for tool in self.tools}

    async def invoke_async(self, state: A2AMessagesState) -> dict[str, list[Any]]:  # TODO? async?
        ...
