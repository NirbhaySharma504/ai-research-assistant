"""Shared execution layer over the compiled graph.

Both the CLI and the FastAPI/WebSocket layer call run_streaming() so progress
reporting lives in exactly one place. `emit` is a plain sync callback invoked once
per graph node with a progress event; the API adapts it to an asyncio.Queue.
"""

from collections.abc import Callable
from functools import lru_cache
from typing import Any

from backend.graph.research_graph import build_research_graph, initial_state
from backend.observability import setup_tracing

setup_tracing()  # enables LangSmith tracing if configured in .env (else a no-op)

# Human-readable label per node, surfaced to the UI as live progress.
_NODE_LABELS = {
    "planner": "Planning research focus areas",
    "researcher": "Searching & scraping sources",
    "fact_checker": "Verifying claims against sources",
    "synthesizer": "Synthesizing cited answer",
    "evaluator": "Scoring answer quality (RAGAS)",
}


@lru_cache(maxsize=2)
def _graph(with_fact_checker: bool = True):
    return build_research_graph(with_fact_checker)


def _progress_detail(node: str, update: dict) -> str:
    if node == "planner":
        return f"{len(update.get('focus_areas', []))} focus areas identified"
    if node == "researcher":
        return f"+{len(update.get('retrieved_content', []))} sources retrieved"
    if node == "fact_checker":
        return f"{len(update.get('verified_claims', []))} claims verified so far"
    if node == "synthesizer":
        return "Answer drafted with citations"
    if node == "evaluator":
        scores = update.get("ragas_scores")
        return "RAGAS scores computed" if scores else "Evaluation finished"
    return ""


def run_streaming(
    query: str,
    session_id: str,
    max_iterations: int = 3,
    emit: Callable[[dict], None] | None = None,
    with_fact_checker: bool = True,
) -> dict[str, Any]:
    """Run the graph, emitting a progress event per node, and return the final state.

    Each emitted event: {type, node, label, status, detail}. The returned dict is
    the fully accumulated ResearchState (graph.stream yields per-node deltas).
    Set with_fact_checker=False for the ablation variant.
    """
    emit = emit or (lambda _e: None)
    state = initial_state(query, session_id, max_iterations)
    final: dict[str, Any] = dict(state)

    for step in _graph(with_fact_checker).stream(state):
        node, update = next(iter(step.items()))
        # stream "updates" mode yields each node's delta; retrieved_content has an
        # operator.add reducer in-graph, so accumulate it here too (don't overwrite).
        for k, v in update.items():
            if k == "retrieved_content" and isinstance(v, list):
                final[k] = final.get(k, []) + v
            else:
                final[k] = v
        emit(
            {
                "type": "progress",
                "node": node,
                "label": _NODE_LABELS.get(node, node),
                "status": update.get("status", ""),
                "detail": _progress_detail(node, update),
            }
        )

    return final


def run_ablation_pair(
    query: str, session_id: str, max_iterations: int = 3
) -> tuple[dict, dict]:
    """Controlled fact-checker ablation over an IDENTICAL retrieved corpus.

    Runs the full pipeline once (planner → research → fact-check → synthesize →
    evaluate), then re-synthesizes & re-evaluates on the SAME ChromaDB session with the
    verified claims removed. Because both variants retrieve from the same corpus, the
    only changed variable is the fact-checker's verified_claims — removing the
    retrieval variance that confounds running two independent pipelines.

    Returns (full_state, no_factcheck_state).
    """
    from backend.agents.evaluator import evaluator_node
    from backend.agents.synthesizer import synthesizer_node

    full = run_streaming(query, session_id, max_iterations, with_fact_checker=True)

    # Same session/corpus, but drop the verified claims and re-synthesize + re-score.
    nofc = dict(full)
    nofc.update(
        verified_claims=[],
        final_answer="",
        citations=[],
        synthesis_contexts=[],
        ragas_scores=None,
    )
    nofc.update(synthesizer_node(nofc))
    nofc.update(evaluator_node(nofc))
    return full, nofc
