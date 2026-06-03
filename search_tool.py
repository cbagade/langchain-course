from langchain_core.tools import tool
from langchain_tavily import TavilySearch


@tool
def search_web(search_query: str) -> str:
    """
    Search over the internet for the provided query and return a concise answer.

    Use this tool when the user asks for current information, web results,
    external facts, latest updates, or anything that should be verified online.

    Args:
        search_query: The exact web search query to look up on the internet.

    Returns:
        A string containing the search result or answer found from the internet.
    """
    tavily_search = TavilySearch(max_results=2, include_answer=True)
    result = tavily_search.invoke(search_query)
    return str(result)
