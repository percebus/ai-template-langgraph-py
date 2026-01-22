from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from langchain_openai.chat_models.base import BaseChatOpenAI
    from langgraph.runtime import Runtime

    from agent.utils.context import Context
    from agent.utils.state import State


@dataclass
class ChatAgent:
    openai: BaseChatOpenAI = field()

    # SRC: https://docs.langchain.com/langsmith/server-a2a#creating-an-a2a-compatible-agent
    async def call_model(self, state: State, runtime: Runtime[Context]) -> dict[str, Any]:
        """Process conversational messages and returns output using OpenAI."""

        # Process the incoming messages
        latest_message = state.messages[-1] if state.messages else {}
        user_content = latest_message.get("content", "No message content")

        # Create messages for OpenAI API
        openai_messages = [
            {"role": "system", "content": "You are a helpful conversational agent. Keep responses brief and engaging."},
            {"role": "user", "content": user_content},
        ]

        try:
            # Make OpenAI API call
            response = await self.openai.client.chat.completions.create(
                model="gpt-3.5-turbo", messages=openai_messages, max_tokens=100, temperature=0.7
            )

            ai_response = response.choices[0].message.content

        except Exception as e:
            ai_response = f"I received your message but had trouble processing it. Error: {str(e)[:50]}..."

        # Create a response message
        response_message = {"role": "assistant", "content": ai_response}

        return {"messages": state.messages + [response_message]}
