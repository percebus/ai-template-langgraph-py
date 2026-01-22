from dataclasses import dataclass
from typing import Any


@dataclass
class State:
    """Input state for the agent.

    Defines the initial structure for A2A conversational messages.
    """

    # SRC: https://docs.langchain.com/langsmith/server-a2a#creating-an-a2a-compatible-agent
    # "To be compatible with the A2A “text” parts, the agent must have a messages key in state."
    messages: list[dict[str, Any]]
