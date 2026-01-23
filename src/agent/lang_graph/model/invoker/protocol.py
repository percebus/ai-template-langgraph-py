from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from langchain.messages import SystemMessage
    from langgraph.runtime import Runtime

    from agent.lang_graph.context import Context
    from agent.lang_graph.states.a2a import A2AMessagesState


class ModelInvokerProtocol(Protocol):

    # TODO read from jinja
    system_message: SystemMessage

    # SRC: https://docs.langchain.com/langsmith/server-a2a#creating-an-a2a-compatible-agent
    async def invoke_async(self, state: A2AMessagesState, runtime: Runtime[Context]) -> dict[str, Any]:
        """Process conversationa l messages and returns output using OpenAI."""
        ...
