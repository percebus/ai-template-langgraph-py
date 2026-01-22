import asyncio
from pprint import pprint
from typing import TYPE_CHECKING, Any

from lagom import Container
from langchain.messages import HumanMessage
from langgraph.graph.state import CompiledStateGraph

if TYPE_CHECKING:
    from langchain_core.messages.base import BaseMessage


async def run(container: Container) -> None:
    graph = container[CompiledStateGraph]  # pyright: ignore[reportUnknownVariableType]

    msg = input("Enter your message: ")
    while msg:
        message = HumanMessage(content=msg)
        messages: list[BaseMessage] = [message]
        response: dict[str, Any] = await graph.ainvoke({"messages": messages})  # pyright: ignore[reportUnknownMemberType]
        new_messages: list[Any] = response.get("messages", [])  # FIXME list[BaseMessage]
        for message in new_messages:
            pprint(message)

        msg = input()


def main() -> None:
    from agent.dependency_injection.container import container

    asyncio.run(run(container))


if __name__ == "__main__":
    main()
