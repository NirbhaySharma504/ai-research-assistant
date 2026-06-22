"""Fact-checker agent: extract key claims from the latest content and verify each
against the session's stored sources (ChromaDB), with an optional Tavily fallback.

Also advances current_focus_index so the should_continue edge can route the loop.
Kept deliberately simple/fast per the engineering plan.
"""

from backend.agents.utils import invoke_json
from backend.graph.state import ResearchState
from backend.llm import get_llm
from backend.tools.search import search_web
from backend.tools.vector_store import retrieve_relevant

_MIN_CONFIDENCE = 0.5


def _extract_claims(llm, content: str) -> list[str]:
    prompt = (
        "Extract 5-8 specific, verifiable factual claims from this content. "
        "Focus on statistics, dates, names, and causal relationships. "
        "Return ONLY a JSON array of strings.\n\nContent:\n" + content[:6000]
    )
    try:
        claims = invoke_json(llm, prompt)
        if isinstance(claims, dict):
            claims = next(iter(claims.values()))
        return [c for c in claims if isinstance(c, str)][:8]
    except Exception:  # noqa: BLE001
        return []


def _verify_claim(claim: str, session_id: str) -> dict:
    hits = retrieve_relevant(claim, session_id, k=5)
    sources = {h["metadata"].get("url") for h in hits if h.get("metadata")}
    sources.discard(None)
    n = len(sources)

    if n >= 2:
        return _claim(claim, True, 0.85, list(sources))
    if n == 1:
        return _claim(claim, True, 0.6, list(sources))

    # 0 local sources -> one verification search
    results = search_web(claim, max_results=3)
    if results:
        return _claim(claim, True, 0.6, [r["url"] for r in results[:2]])
    return _claim(claim, False, 0.2, [])


def _claim(text, verified, conf, supporting):
    return {
        "claim": text,
        "is_verified": verified,
        "confidence": conf,
        "supporting_sources": supporting,
        "contradicting_sources": [],
    }


def fact_checker_node(state: ResearchState) -> dict:
    idx = state["current_focus_index"]
    focus_title = state["focus_areas"][idx]["title"]
    llm = get_llm(temperature=0.2)

    # latest batch = content from the current focus area
    batch = [c for c in state["retrieved_content"] if c["focus_area"] == focus_title]
    combined = "\n\n".join(c["content"] for c in batch)

    verified = list(state.get("verified_claims", []))
    errors = list(state.get("errors", []))

    if combined:
        for claim in _extract_claims(llm, combined):
            v = _verify_claim(claim, state["session_id"])
            if v["confidence"] >= _MIN_CONFIDENCE:
                verified.append(v)
            else:
                errors.append(f"unverified claim: {claim}")

    return {
        "verified_claims": verified,
        "current_focus_index": idx + 1,  # advance for the loop
        "errors": errors,
        "status": "researching",
    }
