"""
Build a graph that uses `messages` as state and a chat model with bound tools.
This is a simple example of an `agent`. The `LLM` is directing the control flow either by calling a `tool` or just responding directly.
"""

from typing import Callable

from dotenv import load_dotenv
from langchain_core.runnables.base import Runnable
from langchain_groq import ChatGroq
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from rich import print

# from langchain_core.messages import HumanMessage


load_dotenv()


LLM: ChatGroq = ChatGroq(
	model="llama-3.1-8b-instant",
	temperature=0.0,
)


def multiply(first_int: int, second_int: int) -> int:
	"""
	A tool to multiply two integers.
	    Args:
	        first_int: first int
	        second_int: second int
	    Returns:
	        The product of first_int and second_int.
	"""
	return first_int * second_int


def bind_tools_to_llm(chat_model: ChatGroq, tools: list[Callable]) -> Runnable:
	"""
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
	Node that binds tools to a chat model and calls a tool.
	    Args:
	        state: The current `state` of the graph.

	    Returns:
	        The new `state` after processing the original `state`.
	"""
	llm_with_tools = bind_tools_to_llm(chat_model=LLM, tools=[multiply])
	state["messages"] = [llm_with_tools.invoke(state["messages"])]

	return state


def compile_graph() -> CompiledStateGraph:
	"""
	Builds and compiles a graph with a node that calls a tool.
	    Returns:
	        An instance of `CompiledStateGraph`.
	"""
	# Build a graph whose nodes communicate by reading and writing to a shared state
	builder = StateGraph(MessagesState)
	# Add nodes
	builder.add_node("tool_calling_llm_node", tool_calling_llm_node)
	builder.add_node("tools", ToolNode([multiply]))
	# Add edges
	builder.add_edge(START, "tool_calling_llm_node")
	builder.add_conditional_edges(
		"tool_calling_llm_node",
		tools_condition,
		# IF the latest message (result) from assistant is a tool call -> tools_condition routes to tools
		# ELSE -> tools_condition routes to END
	)
	builder.add_edge("tools", END)

	return builder.compile()


def invoke_graph(compiled_graph: CompiledStateGraph):
	"""
	Invokes the compiled graph with the given name.
	    Args:
	        compiled_graph: The compiled graph to invoke.
	        user_msg: The user message to invoke the graph with.
	    Returns:
	        The result of the graph execution.
	"""
	user_input: str = input("User: ")
	if not user_input.strip():
		print("❗️❗️❗️ [bold red]MUST PROVIDE A MESSAGE TO INVOKE THE GRAPH[/] ❗️❗️❗️")
		invoke_graph(compiled_graph)

	# The input is a dict `{"messages": [("human", user_input)]}` or, another valid way: `{"messages": [HumanMessage(content=user_input)]}`,
	# sets the initial condition/starting value for the graph's state dict
	result = compiled_graph.invoke({"messages": [("human", user_input)]})
	return result


if __name__ == "__main__":
	graph: CompiledStateGraph = compile_graph()
	response = invoke_graph(compiled_graph=graph)
	for m in response["messages"]:
		print("\n\n", m)
		print("\n\n", m)
