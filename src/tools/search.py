import os
from tavily import TavilyClient

def web_search(query: str) -> str:
    """
    Performs a web search for a given query using the Tavily API.

    Args:
        query: The search query.

    Returns:
        A formatted string of search results or an error message.
    """
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        return "Error: TAVILY_API_KEY not found. Please set it in your .env file."

    try:
        client = TavilyClient(api_key=api_key)
        response = client.search(query=query, search_depth="basic")

        if response and response['results']:
            results = response['results']
            formatted_results = []
            for result in results:
                title = result.get('title', 'No Title')
                url = result.get('url', '#')
                formatted_results.append(f"- [{title}]({url})")
            return "\n".join(formatted_results)
        else:
            return f"No results found for '{query}'."

    except Exception as e:
        return f"An unexpected error occurred during web search: {e}"
