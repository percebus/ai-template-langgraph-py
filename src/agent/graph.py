import asyncio
from collections.abc import Awaitable

from langgraph.graph.state import CompiledStateGraph


async def create_graph_async() -> CompiledStateGraph:  # type: ignore # FIXME
    from agent.dependency_injection.container import container

    return await container[Awaitable[CompiledStateGraph]]  # type: ignore # FIXME


graph = asyncio.run(create_graph_async())  # type: ignore[unused-ignore] # FIXME
