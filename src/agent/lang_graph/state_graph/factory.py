from dataclasses import dataclass, field

from langgraph.graph import START, StateGraph

from agent.lang_graph.context import Context
from agent.lang_graph.model.protocol import ModelInvokerProtocol
from agent.lang_graph.state import State


@dataclass
class StateGraphFactory:
    model_invoker: ModelInvokerProtocol = field()

    def create(self) -> StateGraph[State, Context]:  # type: ignore[unused-ignore, type-arg]
        # fmt:  off
        return (
            StateGraph(State, context_schema=Context)  # type: ignore[unused-ignore]
                .add_node("call_model", self.model_invoker.call_model)  # pyright: ignore[reportUnknownMemberType]
                .add_edge(START, "call_model")
        )
        # fmt: on
