"""
Build a simple chain that combines 4 concepts:
    1) Using chat messages as our graph state
    2) Using chat models in graph nodes
    3) Binding tools to our chat model
    4) Executing tool calls in graph nodes
"""

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, START, END
from langgraph.graph import MessagesState
from langgraph.graph.message import add_messages
from langgraph.graph.state import CompiledStateGraph
from rich import print

load_dotenv()

LLM: ChatGroq = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.0,
)

# Initial state
initial_messages = [
    AIMessage(content="Hello! How can I assist you?", name="Model"),
    HumanMessage(
        content="I'm looking for information on marine biology.", name="Lance"
    ),
]

# New message to add
new_message = AIMessage(
    content="Sure, I can help with that. What specifically are you interested in?",
    name="Model",
)

# Test
add_messages(initial_messages, new_message)


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


llm_with_tools = LLM.bind_tools([multiply])


# Node
def tool_calling_llm(state: MessagesState):
    """
    Node that calls a tool.
        Args:
            state: The current `state` of the graph.

        Returns:
            The new `state` after processing the original `state`.
    """
    return {"messages": [llm_with_tools.invoke(state["messages"])]}


# Build graph


def build_and_compile_graph() -> CompiledStateGraph:
    """
    Build a graph with a node that calls a tool.
        Returns:
            An instance of `CompiledStateGraph`.
    """
    builder = StateGraph(MessagesState)
    builder.add_node("tool_calling_llm", tool_calling_llm)
    builder.add_edge(START, "tool_calling_llm")
    builder.add_edge("tool_calling_llm", END)
    graph = builder.compile()

    return graph


if __name__ == "__main__":
    print(f"[bold #ce9178]RES:[/] {LLM}")
