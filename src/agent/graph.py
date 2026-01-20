"""LangGraph single-node graph template.

Returns a predefined response. Replace logic and configuration as needed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, TypedDict

from langgraph.graph import StateGraph

if TYPE_CHECKING:
    from langgraph.runtime import Runtime


class Context(TypedDict):
    """Context parameters for the agent.

    Set these when creating assistants OR when invoking the graph.
    See: https://langchain-ai.github.io/langgraph/cloud/how-tos/configuration_cloud/
    """

    my_configurable_param: str


@dataclass
class State:
    """Input state for the agent.

    Defines the initial structure of incoming data.
    See: https://langchain-ai.github.io/langgraph/concepts/low_level/#state
    """

    changeme: str = "example"


async def call_model(state: State, runtime: Runtime[Context]) -> dict[str, Any]:
    """Process input and returns output.

    Can use runtime context to alter behavior.
    """
    result = (runtime.context or {}).get("my_configurable_param")
    output = f"output from call_model.  Configured with {result}"
    return {"changeme": output}


# Define the graph
graph = StateGraph(State, context_schema=Context).add_node(call_model).add_edge("__start__", "call_model").compile(name="New Graph")
