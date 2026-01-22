from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol

from langchain.messages import AIMessage, SystemMessage

if TYPE_CHECKING:
    from langchain_core.messages.base import BaseMessage
    from langchain_core.runnables.base import Runnable
    from langchain_core.runnables.utils import Output
    from langgraph.runtime import Runtime

    from agent.lang_graph.context import Context
    from agent.lang_graph.state import State



class ModelInvokerProtocol(Protocol):
    runnable: Runnable

    # TODO read from jinja
    system_message: SystemMessage

    # SRC: https://docs.langchain.com/langsmith/server-a2a#creating-an-a2a-compatible-agent
    async def call_model(self, state: State, runtime: Runtime[Context]) -> dict[str, Any]:
        """Process conversational messages and returns output using OpenAI."""
        ...
