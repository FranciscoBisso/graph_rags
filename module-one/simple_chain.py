"""
Build a simple chain that combines 4 concepts:
  1) Using chat messages as our graph state
  2) Using chat models in graph nodes
  3) Binding tools to our chat model
  4) Executing tool calls in graph nodes
"""

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_groq import ChatGroq
from langgraph.graph import START, StateGraph, END
from langgraph.graph import MessagesState
from langgraph.graph.state import CompiledStateGraph
from rich import print
from typing import Callable


load_dotenv()

llm: ChatGroq = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.0,
)


def multiply(first_int: int, second_int: int) -> int:
    """
    Multiplies two integers.
        Args:
            first_int: first int
            second_int: second int
        Returns:
            The product of first_int and second_int.
    """
    return first_int * second_int


def bind_tools_to_llm(chat_model: ChatGroq, tools: list[Callable]):
    """
    Binds tools to a chat model.
        Args:
            chat_model: The chat model to bind tools to.
            tools: A list of tools to bind to the chat model.

        Returns:
            A chat model with the tools bound.
    """
    return chat_model.bind_tools(tools)


# Node
def tool_calling_llm_node(state: MessagesState) -> MessagesState:
    """
    Node that calls a tool.
        Args:
            state: The current `state` of the graph.

        Returns:
            The new `state` after processing the original `state`.
    """
    llm_with_tools = bind_tools_to_llm(chat_model=llm, tools=[multiply])

    return MessagesState(messages=[llm_with_tools.invoke(state["messages"])])


def create_graph() -> CompiledStateGraph:
    """
    Builds and compiles a graph with a node that calls a tool.
        Returns:
            An instance of `CompiledStateGraph`.
    """

    # Build graph
    builder = StateGraph(MessagesState)
    # Add nodes
    builder.add_node("tool_calling_llm_node", tool_calling_llm_node)
    # Add edges (graph's logic)
    builder.add_edge(START, "tool_calling_llm_node")
    builder.add_edge("tool_calling_llm_node", END)
    graph = builder.compile()

    return graph


def invoke_graph(graph: CompiledStateGraph, user_input: str):
    """
    Invokes the compiled graph with the given name.
        Args:
            graph: The compiled graph to invoke.
        Returns:
            The result of the graph execution.
    """
    return graph.invoke(MessagesState(messages=[HumanMessage(content=user_input)]))


if __name__ == "__main__":
    response = invoke_graph(graph=create_graph(), user_input=input("User: ").strip())
    print(response)
