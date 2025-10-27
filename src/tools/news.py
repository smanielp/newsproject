import os
from newsapi import NewsApiClient
from newsapi.newsapi_exception import NewsAPIException

def get_news(category: str) -> str:
    """
    Fetches top headlines for a given category using the News API.

    Args:
        category: The category of news to fetch.

    Returns:
        A formatted string of news articles or an error message.
    """
    api_key = os.getenv("NEWS_API_KEY")
    if not api_key:
        return "Error: NEWS_API_KEY not found. Please set it in your .env file."

    try:
        newsapi = NewsApiClient(api_key=api_key)
        top_headlines = newsapi.get_top_headlines(category=category, language='en', page_size=5)

        if top_headlines['status'] == 'ok' and top_headlines['articles']:
            articles = top_headlines['articles']
            formatted_articles = []
            for article in articles:
                title = article.get('title', 'No Title')
                url = article.get('url', '#')
                formatted_articles.append(f"- [{title}]({url})")
            return "\n".join(formatted_articles)
        else:
            return f"No news found for the category '{category}'."

    except NewsAPIException as e:
        return f"Error fetching news from News API: {e}"
    except Exception as e:
        return f"An unexpected error occurred: {e}"
