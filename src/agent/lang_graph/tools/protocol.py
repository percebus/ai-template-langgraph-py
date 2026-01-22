from typing import Any, Protocol

from agent.lang_graph.states.a2a import A2AMessagesState


class ToolInvokerProtocol(Protocol):
    def invoke(self, state: A2AMessagesState) -> dict[str, list[Any]]: ...
