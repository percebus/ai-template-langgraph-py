import operator
from typing import Annotated, TypedDict

from langchain.messages import AnyMessage


# SRC: https://docs.langchain.com/oss/python/langgraph/quickstart#2-define-state
class MessagesState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]

    llm_calls: int
