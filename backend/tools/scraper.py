"""Async web scraper. trafilatura first (best content extraction), BS4 fallback.

Every path returns "" on failure -- 20-30% of URLs fail (403, timeout, bot
detection) and must never raise into the agent flow.
"""

import httpx
import trafilatura
from bs4 import BeautifulSoup

from backend.config import settings

_HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; ResearchBot/1.0)"}
_MAX_CHARS = 8000


def _extract_with_bs4(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header",
                     "aside", "form", "iframe"]):
        tag.decompose()
    main = (soup.find("article") or soup.find("main")
            or soup.find(id="content") or soup.find(class_="content")
            or soup.find("body"))
    return main.get_text(separator=" ", strip=True) if main else ""


async def scrape_page(url: str, timeout: int | None = None) -> str:
    """Fetch and extract clean main-content text, capped at 8000 chars."""
    timeout = timeout or settings.MAX_SCRAPE_TIMEOUT
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get(
                url, timeout=timeout, follow_redirects=True, headers=_HEADERS
            )
        if resp.status_code != 200:
            return ""

        text = trafilatura.extract(resp.text, include_comments=False,
                                   include_tables=False) or ""
        if not text:
            text = _extract_with_bs4(resp.text)
        return text[:_MAX_CHARS]
    except Exception:  # noqa: BLE001 - non-fatal
        return ""
