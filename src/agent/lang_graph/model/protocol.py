from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from langchain.messages import SystemMessage
    from langchain_core.runnables.base import Runnable
    from langgraph.runtime import Runtime

    from agent.lang_graph.context import Context
    from agent.lang_graph.state import State


class ModelInvokerProtocol(Protocol):
    runnable: Runnable  # type: ignore[unused-ignore, type-arg]

    # TODO read from jinja
    system_message: SystemMessage

    # SRC: https://docs.langchain.com/langsmith/server-a2a#creating-an-a2a-compatible-agent
    async def call_model(self, state: State, runtime: Runtime[Context]) -> dict[str, Any]:
        """Process conversational messages and returns output using OpenAI."""
        ...
