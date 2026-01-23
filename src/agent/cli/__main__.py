import asyncio
from collections.abc import Awaitable
from pprint import pprint
from typing import TYPE_CHECKING, Any

from lagom import Container
from langchain.messages import HumanMessage
from langgraph.graph.state import CompiledStateGraph

if TYPE_CHECKING:
    from langchain_core.messages.base import BaseMessage


async def run_async(container: Container) -> None:
    graph = await container[Awaitable[CompiledStateGraph]]  # type: ignore[type-abstract, type-arg]

    msg = input("Enter your message: ")
    while msg:
        message = HumanMessage(content=msg)
        messages: list[BaseMessage] = [message]
        response: dict[str, Any] = await graph.ainvoke({"messages": messages})  # pyright: ignore[reportUnknownMemberType]
        new_messages: list[Any] = response.get("messages", [])  # FIXME list[BaseMessage]
        for message in new_messages:
            pprint(message)

        msg = input()


async def main_async() -> None:
    from agent.dependency_injection.container import container

    await run_async(container)


if __name__ == "__main__":
    asyncio.run(main_async())
