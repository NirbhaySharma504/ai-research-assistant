"""Conditional routing for the research loop."""

from backend.graph.state import ResearchState


def should_continue(state: ResearchState) -> str:
    """After fact-checking, loop back to research the next focus area, or synthesize.

    fact_checker_node has already advanced current_focus_index, so we compare it
    directly against the number of focus areas.
    """
    all_done = state["current_focus_index"] >= len(state["focus_areas"])
    over_limit = state.get("iteration_count", 0) >= state.get("max_iterations", 3)
    has_error = state.get("status") == "error"

    if has_error or all_done or over_limit:
        return "synthesize"
    return "continue_research"
