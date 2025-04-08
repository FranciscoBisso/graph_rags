"""
The Simplest Graph: Build a simple graph with 3 nodes and one conditional edge.
"""

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field
from typing_extensions import Annotated


load_dotenv()


class State(BaseModel):
    """
    Represents the state of a conversation.
    State management follows an append strategy for messages rather than replacement.
    """

    messages: Annotated[list, add_messages] = Field(
        default_factory=list,
        description=(
            "List of conversation messages."
            "The `add_messages` function in the annotation defines how this state key should be updated"
            "(in this case, it appends messages to the list, rather than overwriting them)"
        ),
    )


graph_builder = StateGraph(State)

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.0,
)


def chatbot(state: State) -> State:
    """
    Chatbot node:
        Takes a state and returns a new state with the updated messages.
    """
    response = llm.invoke(state.messages)
    return State(messages=[response])


graph_builder.add_node("chatbot", chatbot)
graph_builder.add_edge(START, "chatbot")
graph_builder.add_edge("chatbot", END)
graph = graph_builder.compile()


def stream_graph_updates(user_input: str):
    """
    Stream graph updates:
        Takes a user input and streams the graph updates.
    """
    initial_state = State(messages=[HumanMessage(content=user_input)])
    for event in graph.stream(initial_state):
        for value in event.values():
            print("Assistant:", value["messages"][-1].content)


def main() -> None:
    """Main interaction loop with error handling."""
    fallback_msg = "What do you know about LangGraph?"

    while True:
        try:
            user_msg = input("User: ").strip()
            if not user_msg:  # Skip empty messages
                continue
            if user_msg.lower() in ["quit", "exit", "q"]:
                print("Goodbye!")
                break
            stream_graph_updates(user_msg)
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except EOFError:
            # Fallback if input() is not available
            print("User: " + fallback_msg)
            stream_graph_updates(fallback_msg)
            break
        except Exception as e:
            print(f"An error occurred: {str(e)}")
            break


if __name__ == "__main__":
    main()
