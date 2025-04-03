"""
MODULE CERO - INTRODUCTION
"""

from dotenv import load_dotenv
from langchain_community.tools import DuckDuckGoSearchResults
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import HumanMessage
from langchain_groq import ChatGroq
from rich import print


# .VENV VARIABLES
load_dotenv()

SEARCH_QUERY: str = "¿Cómo se interpreta el silencio de la administración en el derecho administrativo nacional argentino?"


def chat_with_llm():
    """Function to chat with the LLM."""

    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0.0,
    )
    messages = [HumanMessage(content="Hello world", name="Jon Doe")]
    response = llm.invoke(messages)
    print(f"ChatGroq:\n{response}\n\n[bold light_coral]{'===' * 25}[/]\n\n")


def tavily_search(query: str):
    """Function to search on Tavily."""

    search = TavilySearchResults(max_results=3)
    docs = search.invoke(query)
    print(
        f"\t[bold cornflower_blue]TavilySearchResults:[/]\n\n{docs}\n\n[bold light_coral]{'===' * 25}[/]\n\n"
    )


def duckduckgo_search(query: str):
    """Function to search on DuckDuckGo."""

    search = DuckDuckGoSearchResults(num_results=3, output_format="list")
    docs = search.invoke(query)
    print(
        f"\t[bold cornflower_blue]DuckDuckGoSearchResults:[/]\n\n{docs}\n\n[bold light_coral]{'===' * 25}[/]\n\n"
    )


if __name__ == "__main__":
    chat_with_llm()
    tavily_search(SEARCH_QUERY)
    duckduckgo_search(SEARCH_QUERY)
