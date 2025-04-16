"""
The Simplest Graph: build a simple graph with 3 nodes and one conditional edge.
"""

import random
from typing import TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from rich import print
from typing_extensions import Literal


class GraphState(TypedDict):
	"""
	The first thing you do when you define a graph is define the `GraphState` of the graph.
	The `GraphState` consists of the schema of the graph as well as reducer functions:
	which specify how to apply updates to the state.
	Graph's `nodes` communicate by reading and writing to a shared state.
	"""

	graph_state: str


def node_1(state: GraphState) -> GraphState:
	"""
	Node 1: Takes the original `state` and returns a new state: original state + "I am " appended.
	    Args:
	        state: The current `state` of the graph.

	    Returns:
	        The new `state` after processing the original `state`.
	"""

	print("[bold #9cdcfe]••• Node 1 •••[/]")
	state["graph_state"] = f"{state['graph_state']} I am"
	return state


def node_2(state: GraphState) -> GraphState:
	"""
	Node 2: Takes the node_one `state` and returns a new state: node_one `state` + "happy!" appended.
	    Args:
	        state: The current `state` of the graph.

	    Returns:
	        The new `state` after processing the previous `state`.
	"""

	print("[bold #ffd700]••• Node 2 •••[/]")
	state["graph_state"] = f"{state['graph_state']} happy!"
	return state


def node_3(state: GraphState) -> GraphState:
	"""
	Node 3: Takes the node_one `state` and returns a new state: node_one `state` + "sad!" appended.
	    Args:
	        state: The current `state` of the graph (inicial `state` + node_one `state`).

	    Returns:
	        The new `state` after processing the previous `state`.
	"""

	print("[bold #4ec9b0]••• Node 3 •••[/]")
	state["graph_state"] = f"{state['graph_state']} sad!"
	return state


def decide_mood(state: GraphState) -> Literal["node_2", "node_3"]:
	"""
	Randomly decide on the next `node` to visit.
	  (Often, we will use the current `state` to decide on the next `node` to visit)
	    Args:
	        state: The current `state` of the graph (inicial `state` + node_one `state`).

	    Returns:
	        The next `node` to visit.
	"""

	# Often, we will use state to decide on the next node to visit
	user_input = state["graph_state"]  # noqa: F841, pylint: disable=unused-variable (disable linters' rules)

	return random.choice(["node_2", "node_3"])


def build_and_compile_graph() -> CompiledStateGraph:
	"""
	Build a graph with 3 nodes and one conditional edge.
	    Returns:
	        An instance of `CompiledStateGraph`.
	"""
	# Build a graph whose nodes communicate by reading and writing to a shared state.
	builder = StateGraph(GraphState)
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

	return graph


def invoke_graph(graph: CompiledStateGraph) -> str:
	"""
	Invokes the compiled graph with the given name.
	    Args:
	        graph: The compiled graph to invoke.
	    Returns:
	        The result of the graph execution.
	"""
	user_name: str = input("What is your name? ")
	name: str = user_name.strip().title() if user_name else "John Doe"
	response = graph.invoke({"graph_state": f"Hi, this is {name}. "})
	return response["graph_state"]


if __name__ == "__main__":
	compiled_graph = build_and_compile_graph()
	res = invoke_graph(compiled_graph)
	print(f"[bold light_slate_blue]RES: [/] [bold orange1]{res}[/]")
