"""Synthesizer agent: RAG over the session store + verified claims -> cited answer."""

from backend.graph.state import ResearchState
from backend.llm import get_llm
from backend.tools.vector_store import retrieve_relevant

SYSTEM_PROMPT = """You are a research synthesis expert. Generate a comprehensive answer \
to the research query using ONLY the provided context.

Rules:
- For every specific fact or statistic, include a citation marker [N] referring to the \
numbered context items
- Structure the answer with clear sections using markdown headers
- Do NOT include any information not present in the context
- End with a 'Key Takeaways' section with 3-5 bullet points
- Be comprehensive but not repetitive"""


def synthesizer_node(state: ResearchState) -> dict:
    session_id = state["session_id"]
    query = state["query"]

    # Step 1: RAG retrieval
    chunks = retrieve_relevant(query, session_id, k=15)

    # Step 2: build numbered context; map each number -> source for citations
    context_lines = []
    citation_map = {}
    n = 0
    for claim in state.get("verified_claims", []):
        n += 1
        src = claim["supporting_sources"][0] if claim["supporting_sources"] else ""
        context_lines.append(f"[{n}] (verified claim) {claim['claim']}")
        citation_map[n] = {"url": src, "title": "Verified claim", "quote": claim["claim"]}
    for ch in chunks:
        n += 1
        meta = ch.get("metadata", {})
        context_lines.append(f"[{n}] {ch['content']}")
        citation_map[n] = {
            "url": meta.get("url", ""),
            "title": meta.get("title", ""),
            "quote": ch["content"][:200],
        }

    if not context_lines:
        return {
            "final_answer": "No content could be retrieved for this query.",
            "citations": [],
            "errors": state.get("errors", []) + ["synthesizer: empty context"],
            "status": "evaluating",
        }

    context_block = "\n\n".join(context_lines)
    llm = get_llm(temperature=0.4)
    prompt = (
        f"{SYSTEM_PROMPT}\n\nResearch query: {query}\n\n"
        f"Context:\n{context_block}"
    )
    answer = llm.invoke(prompt).content

    # Step 4: keep only citations actually referenced in the answer
    import re
    used = {int(m) for m in re.findall(r"\[(\d+)\]", answer)}
    citations = [
        {"number": num, **citation_map[num]}
        for num in sorted(used)
        if num in citation_map
    ]

    return {
        "final_answer": answer,
        "citations": citations,
        "status": "evaluating",
    }
