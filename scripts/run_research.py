"""CLI runner for the research graph (the MVP gate).

Usage:
    .venv/bin/python -m scripts.run_research "What causes climate change?"
"""

import sys
import uuid

from backend.graph.research_graph import build_research_graph, initial_state


def main():
    if len(sys.argv) < 2:
        print('Usage: python -m scripts.run_research "<query>" [max_iterations]')
        sys.exit(1)

    query = sys.argv[1]
    max_iterations = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    session_id = uuid.uuid4().hex[:12]

    print(f"\n=== Researching: {query!r} (session {session_id}) ===\n")
    graph = build_research_graph()
    state = initial_state(query, session_id, max_iterations)

    # stream yields per-node deltas (default "updates" mode); accumulate them so
    # `final` holds the complete state (final_answer comes from synthesizer, but
    # ragas_scores from the later evaluator step).
    final = dict(state)
    for step in graph.stream(state):
        node, update = next(iter(step.items()))
        final.update(update)
        status = update.get("status", "")
        extra = ""
        if node == "planner":
            extra = f"{len(update.get('focus_areas', []))} focus areas"
        elif node == "researcher":
            extra = f"+{len(update.get('retrieved_content', []))} sources"
        elif node == "fact_checker":
            extra = f"{len(update.get('verified_claims', []))} verified claims total"
        print(f"[{node:<12}] status={status} {extra}")

    print("\n" + "=" * 70)
    print("FINAL ANSWER\n")
    print(final.get("final_answer", "(none)"))

    print("\n" + "=" * 70)
    print("CITATIONS\n")
    for c in final.get("citations", []):
        print(f"[{c['number']}] {c.get('title', '')} - {c.get('url', '')}")

    print("\n" + "=" * 70)
    print("RAGAS SCORES:", final.get("ragas_scores"))
    if final.get("errors"):
        print("\nNON-FATAL ERRORS:")
        for e in final["errors"]:
            print(" -", e)


if __name__ == "__main__":
    main()
