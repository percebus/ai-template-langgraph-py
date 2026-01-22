from __future__ import annotations

from dataclasses import dataclass, field
from re import A
from typing import TYPE_CHECKING, Any
from langchain_core.runnables.base import Runnable
from langchain.messages import SystemMessage, AIMessage
from langchain_core.runnables.utils import Output

if TYPE_CHECKING:
    from langgraph.runtime import Runtime

    from agent.utils.context import Context
    from agent.utils.state import State


@dataclass
class ChatAgent:

    runnable: Runnable = field()  # pyright: ignore[reportMissingTypeArgument]

    system_message: SystemMessage = field(default_factory=lambda: SystemMessage(content="You are a helpful conversational agent. Keep responses brief and engaging."))


    # SRC: https://docs.langchain.com/langsmith/server-a2a#creating-an-a2a-compatible-agent
    async def call_model(self, state: State, runtime: Runtime[Context]) -> dict[str, Any]:
        """Process conversational messages and returns output using OpenAI."""

        # Create messages for OpenAI API
        # TODO read from jinja
        openai_messages = [self.system_message] + state.messages

        ai_response: str = ""
        try:
            # Make OpenAI API call
            response: Output = await self.runnable.ainvoke(openai_messages)
            ai_response = response.content

        except Exception as e:
            print(f"Error calling OpenAI API: {e}")  # FIXME logging
            ai_response = f"I received your message but had trouble processing it. Error: {str(e)[:50]}..."

        # Create a response message
        response_message = AIMessage(content=ai_response)

        return {"messages": state.messages + [response_message]}
