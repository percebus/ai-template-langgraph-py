from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

from langgraph.graph import END, START, StateGraph

from agent.lang_graph.context import Context
from agent.lang_graph.model.protocol import ModelInvokerProtocol
from agent.lang_graph.states.a2a import A2AMessagesState
from agent.lang_graph.tools.protocol import ToolInvokerProtocol

if TYPE_CHECKING:
    from langchain_core.messages.base import BaseMessage


@dataclass
class MyStateGraph:
    model_invoker: ModelInvokerProtocol = field()

    tool_invoker: ToolInvokerProtocol = field()

    state_graph: StateGraph[A2AMessagesState, Context] = field(init=False)

    def should_continue(self, state: A2AMessagesState) -> Literal["invoke_tool", END]:  # type: ignore
        """Decide if we should continue the loop or stop based upon whether the LLM made a tool call"""

        last_message: BaseMessage = state.messages[-1]
        tool_calls: list[dict[str, Any]] | Any = last_message.tool_calls  # type: ignore

        # If the LLM makes a tool call, then perform an action
        if tool_calls:
            return "invoke_tool"

        # Otherwise, we stop (reply to the user)
        return END

    def __post_init__(self) -> None:
        # SRC: https://docs.langchain.com/oss/python/langgraph/quickstart#6-build-and-compile-the-agent
        # fmt:  off
        self.state_graph = (
            StateGraph(A2AMessagesState, context_schema=Context)  # type: ignore[unused-ignore]
                # Nodes
                .add_node("invoke_model", self.model_invoker.invoke_async)  # pyright: ignore[reportUnknownMemberType]
                .add_node("invoke_tool", self.tool_invoker.invoke)  # type: ignore[unused-ignore]

                # Edges
                .add_edge(START, "invoke_model")
                .add_conditional_edges("invoke_model", self.should_continue, ["invoke_tool", END])  # type: ignore[unused-ignore]
                .add_edge("invoke_tool", "invoke_model")
        )
        # fmt: on
