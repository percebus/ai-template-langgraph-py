from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from langchain.messages import ToolMessage
from langchain_core.tools.base import BaseTool

from agent.lang_graph.states.a2a import A2AMessagesState
from agent.lang_graph.tools.protocol import ToolInvokerProtocol

if TYPE_CHECKING:
    from langchain_core.messages.base import BaseMessage


@dataclass
class ToolInvoker(ToolInvokerProtocol):
    tools_by_name: dict[str, BaseTool] = field()

    def invoke(self, state: A2AMessagesState) -> dict[str, list[Any]]:
        """Performs the tool call"""

        result = state.messages.copy()
        last_message: BaseMessage = state.messages[-1]
        tool_calls: list[dict[str, Any]] | Any = last_message.tool_calls  # type: ignore
        if not tool_calls:
            raise ValueError("No tool calls found in the last message.")

        for tool_call in tool_calls:  # pyright: ignore[reportUnknownVariableType]
            tool = self.tools_by_name[tool_call["name"]]
            observation: Any = tool.invoke(tool_call["args"])  # pyright: ignore[reportUnknownMemberType]
            tool_message = ToolMessage(content=observation, tool_call_id=tool_call["id"])
            result.append(tool_message)

        return {"messages": result}
