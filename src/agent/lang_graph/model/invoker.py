from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from langchain.messages import AIMessage, SystemMessage

if TYPE_CHECKING:
    from langchain_core.messages.base import BaseMessage
    from langchain_core.runnables.base import Runnable
    from langchain_core.runnables.utils import Output
    from langgraph.runtime import Runtime

    from agent.lang_graph.context import Context
    from agent.lang_graph.states.a2a import A2AMessagesState


@dataclass
class ModelInvoker:
    runnable: Runnable = field()  # type:ignore

    # TODO read from jinja
    system_message: SystemMessage = field(
        default_factory=lambda: SystemMessage(content="You are a helpful conversational agent. Keep responses brief and engaging.")
    )

    # SRC: https://docs.langchain.com/langsmith/server-a2a#creating-an-a2a-compatible-agent
    async def invoke_async(self, state: A2AMessagesState, runtime: Runtime[Context]) -> dict[str, Any]:
        """Process conversational messages and returns output using OpenAI."""

        # Create messages for OpenAI API
        messages: list[BaseMessage] = [self.system_message] + state.messages

        ai_response: str = ""
        tool_calls: list[dict[str, Any]] = []
        try:
            response: Output = await self.runnable.ainvoke(messages)  # type: ignore
            ai_response = response.content  # type: ignore
            if response.tool_calls:  # type: ignore
                tool_calls = response.tool_calls  # type: ignore

        except Exception as e:
            print(f"Error calling OpenAI API: {e}")  # FIXME logging
            ai_response = f"I received your message but had trouble processing it. Error: {str(e)[:50]}..."

        # Create a response message
        response_message = AIMessage(content=ai_response, tool_calls=tool_calls)  # , **response) # TODO

        return {"messages": state.messages + [response_message]}
