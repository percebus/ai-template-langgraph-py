import asyncio
from pprint import pprint
from typing import Any

from langchain.messages import HumanMessage
from langgraph.graph.state import CompiledStateGraph
from langchain_core.messages.base import BaseMessage

from agent.dependency_injection.container import Container


async def run(container: Container) -> None:
    graph = container[CompiledStateGraph]  # pyright: ignore[reportUnknownVariableType]

    msg = input("Enter your message: ")
    while msg:
        message = HumanMessage(content=msg)
        messages = [message]
        response: dict[str, Any] = await graph.ainvoke({"messages": messages})
        messages: list[BaseMessage] = response.get("messages", [])
        for message in messages:
            pprint(message)

        msg = input()


def main() -> None:
    from agent.dependency_injection.container import container

    asyncio.run(run(container))


if __name__ == "__main__":
    main()
