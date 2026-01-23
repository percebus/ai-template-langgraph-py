from typing import Protocol

from langgraph.graph import StateGraph

from agent.lang_graph.context import Context
from agent.lang_graph.states.a2a import A2AMessagesState


class StateGraphProtocol(Protocol):
    state_graph: StateGraph[A2AMessagesState, Context]
