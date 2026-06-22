"""Web search via Tavily (built for AI agents: returns pre-cleaned content)."""

from tavily import TavilyClient

from backend.config import settings

_client: TavilyClient | None = None


def _get_client() -> TavilyClient:
    global _client
    if _client is None:
        if not settings.TAVILY_API_KEY:
            raise ValueError("TAVILY_API_KEY is not set")
        _client = TavilyClient(api_key=settings.TAVILY_API_KEY)
    return _client


def search_web(query: str, max_results: int | None = None) -> list[dict]:
    """Return a list of {url, title, content, score} for the query.

    include_raw_content=False keeps responses small; full text is scraped
    separately. Flip to True as a fallback if scraping proves unreliable.
    """
    max_results = max_results or settings.MAX_RESULTS_PER_SEARCH
    try:
        response = _get_client().search(
            query=query,
            search_depth="advanced",
            max_results=max_results,
            include_raw_content=False,
        )
        return response.get("results", [])
    except Exception as e:  # noqa: BLE001 - non-fatal, log and return empty
        print(f"[search] error for '{query}': {e}")
        return []
