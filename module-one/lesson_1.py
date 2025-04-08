"""
The Simplest Graph: Build a simple graph with 3 nodes and one conditional edge.
"""

import random
from langgraph.graph import StateGraph, START, END
from pydantic import BaseModel, Field
from rich import print
from typing_extensions import Literal


class State(BaseModel):
    """
    The `State` schema serves as the input schema for all `Nodes` and `Edges` in the graph
    """

    graph_state: str = Field(
        default_factory=str,
    )


def node_1(state):
    """
    Node 1:
    Takes the original `state` and returns a new state: original state + " I am" appended.
        Args:
            state: The current state of the graph.

        Returns:
            The new state after processing the original state.
    """

    print(f"[bold #9cdcfe]••• Node 1 •••[/]")
    return {"graph_state": state.graph_state + " I am"}


def node_2(state):
    """
    Node 2:
    Takes the original `state` and returns a new state: original state + " happy!" appended.

    Args:
        state: The current state of the graph.

    Returns:
        The new state after processing the original state.
    """

    print(f"[bold #ce9178]••• Node 2 •••[/]")
    return {"graph_state": state.graph_state + " happy!"}


def node_3(state):
    """
    Node 3:
    Takes the original `state` and returns a new state: original state + " sad!" appended.
        Args:
            state: The current state of the graph.

        Returns:
            The new state after processing the original state.
    """

    print(f"[bold #4ec9b0]••• Node 3 •••[/]")
    return {"graph_state": state.graph_state + " sad!"}


def decide_mood(state) -> Literal["node_2", "node_3"]:
    """
    Decide on the next `node` to visit based on the current `state`
        Args:
            state: The current state of the graph.

        Returns:
            The next node to visit.
    """

    # Often, we will use state to decide on the next node to visit
    user_input = state.graph_state

    # Here, let's just do a 50 / 50 split between nodes 2, 3
    if random.random() < 0.5:
        # 50% of the time, we return Node 2
        return "node_2"

    # 50% of the time, we return Node 3
    return "node_3"


# Build graph
builder = StateGraph(State)
builder.add_node("node_1", node_1)
builder.add_node("node_2", node_2)
builder.add_node("node_3", node_3)

# Logic
builder.add_edge(START, "node_1")
builder.add_conditional_edges("node_1", decide_mood)
builder.add_edge("node_2", END)
builder.add_edge("node_3", END)

# Add
graph = builder.compile()


def invoke_graph(compiled_graph, name: str):
    """
    Invokes the compiled graph with the given name.

    Args:
        compiled_graph: The compiled graph to invoke.
        name: The name to use in the graph.

    Returns:
        The result of the graph execution.
    """
    return compiled_graph.invoke({"graph_state": f"Hi, this is {name}."})


if __name__ == "__main__":
    print(invoke_graph(graph, "Pipo"))
