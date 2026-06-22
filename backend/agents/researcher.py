"""Researcher agent: for the current focus area, search -> select -> scrape ->
chunk/embed/store, accumulating RetrievedContent into state."""

import asyncio
from datetime import datetime, timezone

from backend.agents.utils import invoke_json
from backend.graph.state import ResearchState
from backend.llm import get_llm
from backend.tools.scraper import scrape_page
from backend.tools.search import search_web
from backend.tools.vector_store import store_content


def _gen_queries(llm, focus, already_used: list[str]) -> list[str]:
    prompt = (
        "Given this research focus area:\n"
        f"Title: {focus['title']}\n"
        f"Description: {focus['description']}\n\n"
        "Generate 2-3 specific web search queries that would find the most "
        "relevant information. Return ONLY a JSON array of strings."
    )
    try:
        queries = invoke_json(llm, prompt)
        if isinstance(queries, dict):  # tolerate {"queries": [...]}
            queries = next(iter(queries.values()))
    except Exception:  # noqa: BLE001 - fall back to the focus title
        queries = [focus["title"]]
    used = set(already_used)
    return [q for q in queries if isinstance(q, str) and q not in used][:3]


def _select_urls(llm, results: list[dict], focus) -> list[str]:
    if not results:
        return []
    listing = "\n".join(
        f"{i}. {r.get('title', '')} - {r.get('url', '')}"
        for i, r in enumerate(results)
    )
    prompt = (
        f"Focus area: {focus['title']} - {focus['description']}\n\n"
        f"Search results:\n{listing}\n\n"
        "Select the 2-3 most relevant URLs. Return ONLY a JSON array of URL strings."
    )
    try:
        urls = invoke_json(llm, prompt)
        if isinstance(urls, dict):
            urls = next(iter(urls.values()))
        urls = [u for u in urls if isinstance(u, str)]
    except Exception:  # noqa: BLE001 - fall back to top results by score
        urls = []
    if not urls:
        urls = [r["url"] for r in results[:3]]
    return urls[:3]


async def _scrape_all(urls: list[str]) -> dict[str, str]:
    texts = await asyncio.gather(*(scrape_page(u) for u in urls))
    return {u: t for u, t in zip(urls, texts) if t}


def researcher_node(state: ResearchState) -> dict:
    idx = state["current_focus_index"]
    focus = state["focus_areas"][idx]
    llm = get_llm(temperature=0.3)
    session_id = state["session_id"]
    errors = list(state.get("errors", []))

    # Step 1: generate queries
    queries = _gen_queries(llm, focus, state.get("search_queries_used", []))

    # Step 2: execute searches
    results: list[dict] = []
    for q in queries:
        results.extend(search_web(q))

    # Step 3: select best URLs (dedup, skip already-seen content URLs)
    urls = _select_urls(llm, results, focus)
    result_meta = {r["url"]: r for r in results}

    # Step 4: scrape
    scraped = asyncio.run(_scrape_all(urls))

    # Step 5+6: chunk/embed/store and build RetrievedContent
    new_content = []
    now = datetime.now(timezone.utc).isoformat()
    for url, text in scraped.items():
        meta = result_meta.get(url, {})
        metadata = {
            "url": url,
            "title": meta.get("title", url),
            "focus_area": focus["title"],
            "timestamp": now,
        }
        try:
            store_content(text, metadata, session_id)
        except Exception as e:  # noqa: BLE001
            errors.append(f"researcher store {url}: {e}")
            continue
        new_content.append({
            "url": url,
            "title": metadata["title"],
            "content": text,
            "focus_area": focus["title"],
            "timestamp": now,
            "relevance_score": float(meta.get("score", 0.0)),
        })

    if not new_content:
        errors.append(f"researcher: no content retrieved for '{focus['title']}'")

    # Mark this focus area done (rebuild list since dict mutation isn't tracked)
    focus_areas = [dict(f) for f in state["focus_areas"]]
    focus_areas[idx]["status"] = "completed"

    return {
        "retrieved_content": new_content,  # operator.add appends
        "search_queries_used": state.get("search_queries_used", []) + queries,
        "focus_areas": focus_areas,
        "current_focus_index": idx,  # advanced in should_continue routing
        "iteration_count": state.get("iteration_count", 0) + 1,
        "errors": errors,
        "status": "fact_checking",
    }
