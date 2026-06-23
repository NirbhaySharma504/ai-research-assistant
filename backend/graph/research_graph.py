"""Assemble and compile the LangGraph research StateGraph."""

from langgraph.graph import END, StateGraph

from backend.agents.evaluator import evaluator_node
from backend.agents.fact_checker import fact_checker_node
from backend.agents.planner import planner_node
from backend.agents.researcher import researcher_node
from backend.agents.synthesizer import synthesizer_node
from backend.graph.edges import should_continue
from backend.graph.state import ResearchState


def _skip_fact_checker_node(state: ResearchState) -> dict:
    """Ablation stand-in: advance the loop WITHOUT verifying any claims.

    Keeps the node name/edges identical so the graph topology is unchanged; the only
    difference is that verified_claims stays empty, isolating the fact-checker's
    contribution to answer quality (see benchmark/run_eval.py)."""
    return {
        "current_focus_index": state["current_focus_index"] + 1,
        "status": "researching",
    }


def build_research_graph(with_fact_checker: bool = True):
    graph = StateGraph(ResearchState)

    graph.add_node("planner", planner_node)
    graph.add_node("researcher", researcher_node)
    graph.add_node(
        "fact_checker",
        fact_checker_node if with_fact_checker else _skip_fact_checker_node,
    )
    graph.add_node("synthesizer", synthesizer_node)
    graph.add_node("evaluator", evaluator_node)

    graph.set_entry_point("planner")
    graph.add_edge("planner", "researcher")
    graph.add_edge("researcher", "fact_checker")
    graph.add_conditional_edges(
        "fact_checker",
        should_continue,
        {"continue_research": "researcher", "synthesize": "synthesizer"},
    )
    graph.add_edge("synthesizer", "evaluator")
    graph.add_edge("evaluator", END)

    return graph.compile()


def initial_state(query: str, session_id: str, max_iterations: int = 3) -> dict:
    return {
        "query": query,
        "session_id": session_id,
        "focus_areas": [],
        "current_focus_index": 0,
        "retrieved_content": [],
        "search_queries_used": [],
        "verified_claims": [],
        "final_answer": "",
        "citations": [],
        "synthesis_contexts": [],
        "ragas_scores": None,
        "messages": [],
        "iteration_count": 0,
        "max_iterations": max_iterations,
        "errors": [],
        "status": "planning",
    }
